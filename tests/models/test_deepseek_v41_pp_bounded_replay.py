# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which pipeline stages run the decoder-side SWA bounded replay
(VLLM_DSV41_PP_BOUNDED_REPLAY)."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

import vllm.models.deepseek_v41.nvidia.model as nvidia_model
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.models.deepseek_v41.common.pipeline import (
    decoder_replay_stage_reason,
    get_sharing_dependencies,
    validate_local_sharing,
)
from vllm.models.deepseek_v41.decoder_replay_layers import (
    DecoderReplayLayers,
    ReplayBatch,
)

Model = nvidia_model.DeepseekV4Model

NUM_LAYERS, CUT = 40, 20
# V4.1's layer topology: KV sources 2/8/14/20, index sources up to 36, the
# candidate source at the cut.
CONFIG = SimpleNamespace(
    num_hidden_layers=NUM_LAYERS,
    compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
    kv_source_layer_ids=[2, 8, 14, CUT],
    index_source_layer_ids=[2, 8, 14, CUT, 24, 28, 32, 36],
    candidate_source_layer_id=CUT,
    candidate_topk_blocks=2048,
    sliding_window=128,
    engram_layer_ids=[1, 14],
)


def _stages(*sizes: int) -> list[tuple[int, int]]:
    bounds = [0]
    for size in sizes:
        bounds.append(bounds[-1] + size)
    return list(zip(bounds, bounds[1:]))




def test_stage_without_cut_or_tail_is_rejected_with_flag() -> None:
    """A stage past the cut would replay against KV another stage wrote for
    every row, and one missing later layers cannot finish the replayed rows;
    the flag covers neither (nor does local sharing allow them for V4.1)."""
    assert decoder_replay_stage_reason(21, NUM_LAYERS, CUT, NUM_LAYERS, True)
    assert decoder_replay_stage_reason(0, 30, CUT, NUM_LAYERS, True)
    with pytest.raises(NotImplementedError):
        validate_local_sharing(get_sharing_dependencies(CONFIG, _stages(30, 10)))


def _model(stage: tuple[int, int], pipeline_sharing=None) -> SimpleNamespace:
    """The attributes DeepseekV4Model's replay gate reads, for one stage."""
    swa = SimpleNamespace(bounded_replay=True)
    model = SimpleNamespace(
        start_layer=stage[0],
        end_layer=stage[1],
        config=CONFIG,
        use_sequence_parallel=False,
        pipeline_sharing=pipeline_sharing,
        layers=[SimpleNamespace(attn=SimpleNamespace(swa_cache_layer=swa))]
        * NUM_LAYERS,
    )
    model._decoder_replay_supported = partial(Model._decoder_replay_supported, model)
    return model


VLLM_CONFIG = SimpleNamespace(
    parallel_config=SimpleNamespace(
        prefill_context_parallel_size=1, use_ubatching=False
    ),
    speculative_config=None,
)


@pytest.mark.parametrize(
    ("sizes", "off", "on"),
    [
        ((40,), [True], [True]),
        # The stage starting at the cut replays only under the flag.
        ((20, 20), [False, False], [False, True]),
        ((14, 26), [False, True], [False, True]),
        ((8, 6, 6, 20), [False] * 4, [False, False, False, True]),
    ],
)
def test_stage_replay_decisions(
    monkeypatch: pytest.MonkeyPatch, sizes, off: list[bool], on: list[bool]
) -> None:
    """DeepseekV4Model.__init__'s gate, per stage of each partition."""
    stages = _stages(*sizes)
    validate_local_sharing(get_sharing_dependencies(CONFIG, stages))
    for flag, expected in ((False, off), (True, on)):
        monkeypatch.setattr(nvidia_model.envs, "VLLM_DSV41_PP_BOUNDED_REPLAY", flag)
        enabled = [
            Model._decoder_replay_enabled(_model(stage), VLLM_CONFIG, CUT)
            for stage in stages
        ]
        assert enabled == expected


@pytest.mark.parametrize("flag", [False, True])
def test_model_gate_follows_flag(monkeypatch: pytest.MonkeyPatch, flag: bool) -> None:
    """PP 20,20's second stage replays only under the flag, the single stage
    always does, and a stage receiving shared state never starts at the cut,
    saying so when the flag asked for it."""
    monkeypatch.setattr(nvidia_model.envs, "VLLM_DSV41_PP_BOUNDED_REPLAY", flag)
    warnings: list[str] = []
    monkeypatch.setattr(
        nvidia_model.logger,
        "warning_once",
        lambda msg, *args: warnings.append(msg % args),
    )
    supported = Model._decoder_replay_supported
    assert supported(_model((0, NUM_LAYERS)), VLLM_CONFIG, CUT)
    assert supported(_model((CUT, NUM_LAYERS)), VLLM_CONFIG, CUT) == flag
    assert not supported(_model((0, CUT)), VLLM_CONFIG, CUT)
    shared = _model((CUT, NUM_LAYERS), pipeline_sharing=object())
    warnings.clear()
    assert not supported(shared, VLLM_CONFIG, CUT)
    assert ("is ignored with deepseek_v41_pp_sharing" in warnings[0]) == flag


def _fail(*args, **kwargs):
    raise AssertionError("must not run")


@pytest.mark.parametrize("start", [0, CUT])
def test_run_stage_hands_replay_its_inputs(start: int) -> None:
    """A stage starting at the cut skips the pre-replay layers and hands the
    replay the incoming stream and pre-mix with no residual or mixes; one
    starting earlier hands it what its layers return."""
    tokens = 5
    stream, pre_mix = torch.randn(tokens, 4, 8), torch.rand(tokens, 4)
    positions, input_ids = torch.arange(tokens), torch.arange(tokens) + 7
    pre_replay = tuple(torch.randn(tokens, 4) for _ in range(5))
    calls: dict[str, tuple] = {}

    def run_layers(layers, *args):
        calls["run_layers"] = (layers, *args)
        return pre_replay

    def replay(*args):
        calls["replay"] = args
        return "out", "pre_mix", "aux"

    model = SimpleNamespace(
        start_layer=start,
        decoder_replay_start=CUT,
        _run_layers=run_layers if start < CUT else _fail,
        _collapse=_fail,
        decoder_replay_layers=replay,
    )
    out = Model._run_stage(
        model, stream, positions, input_ids, pre_mix, {}, None, None, tokens
    )
    assert out == ("out", "pre_mix", ["aux"])
    if start == CUT:
        expected = (stream, positions, input_ids, pre_mix, None, None, None)
    else:
        assert calls["run_layers"][0] == range(start, CUT)
        hidden, residual, post_mix, res_mix, last_pre_mix = pre_replay
        expected = (hidden, positions, input_ids, last_pre_mix, post_mix, res_mix)
        expected += (residual,)
    assert len(calls["replay"]) == len(expected)
    for got, want in zip(calls["replay"], expected):
        assert got is want


def test_replay_batch_passes_missing_states() -> None:
    """The replay batch gathers its rows of the given states and passes the
    states a stage starting at the cut lacks through as None."""
    tokens, rows = 9, torch.tensor([0, 4, 5, 8])
    stream, pre_mix = torch.randn(tokens, 4, 8), torch.rand(tokens, 4)
    positions = torch.arange(tokens)
    replay_context = ForwardContext(
        no_compile_layers={}, attn_metadata={}, slot_mapping={}
    )
    seen: dict[str, tuple] = {}

    def run_layers(hidden, *states):
        seen["states"] = states
        seen["context"] = get_forward_context()
        return hidden.sum(1), states[2]

    layers = DecoderReplayLayers(128, run_layers, _fail, set(), "swa")
    layers.replay_batch = ReplayBatch(rows, replay_context)
    out, out_pre_mix = layers._run(stream, positions, None, pre_mix, None, None, None)
    assert seen["context"] is replay_context
    got_positions, input_ids, got_pre_mix, *rest = seen["states"]
    assert input_ids is None and rest == [None, None, None]
    torch.testing.assert_close(got_positions, positions[rows])
    torch.testing.assert_close(got_pre_mix, pre_mix[rows])
    torch.testing.assert_close(out[rows], stream[rows].sum(1))
    torch.testing.assert_close(out_pre_mix[rows], pre_mix[rows])
    trimmed = torch.ones(tokens, dtype=torch.bool)
    trimmed[rows] = False
    assert not out[trimmed].any() and not out_pre_mix[trimmed].any()


def test_stage_first_write_kv_matches_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no residual (a stage starting at the cut), write_kv hands the
    attention's KV write the input forward hands the attention."""

    def mhc_pre(residual, fn, *args, pre_mix=None, norm_weight=None, norm_eps=0.0):
        layer_input = (residual * pre_mix[..., None]).sum(1) * norm_weight + fn.sum()
        return None, None, layer_input, None

    class Attn:
        def __init__(self) -> None:
            self.inputs: list[torch.Tensor] = []

        def __call__(self, positions, x, _):
            self.inputs.append(x)
            raise StopIteration  # forward's attention input is all we need

        def forward_kv(self, positions, x):
            self.inputs.append(x)

    monkeypatch.setattr(nvidia_model, "mhc_pre_delayed_tilelang", mhc_pre)
    hc, hidden, tokens = 4, 8, 5
    layer = SimpleNamespace(
        mhc_stream=None,
        engram=None,
        use_sequence_parallel=False,
        hc_attn_fn=torch.randn(hc * (hc + 2), hc * hidden),
        hc_ffn_fn=torch.randn(hc * (hc + 2), hc * hidden),
        hc_attn_scale=torch.ones(3),
        hc_attn_base=torch.zeros(hc * (hc + 2)),
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_post_alpha=2.0,
        hc_sinkhorn_iters=20,
        attn_norm=SimpleNamespace(weight=torch.randn(hidden), variance_epsilon=1e-6),
        attn=Attn(),
    )
    stream = torch.randn(tokens, hc, hidden)
    pre_mix = torch.rand(tokens, hc)
    positions = torch.arange(tokens)
    with pytest.raises(StopIteration):
        nvidia_model.DeepseekV4DecoderLayer.forward(
            layer, stream, positions, None, pre_mix
        )
    nvidia_model.DeepseekV4DecoderLayer.write_kv(
        layer, stream, positions, None, pre_mix, None, None, None
    )
    forward_input, write_kv_input = layer.attn.inputs
    torch.testing.assert_close(write_kv_input, forward_input, rtol=0, atol=0)
