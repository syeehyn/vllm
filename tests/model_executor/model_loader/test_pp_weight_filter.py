# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline-parallel checkpoint weight filter.

Each PP rank must read from disk only the tensors of its own stage. The
checkpoint names below are the ones of moonshotai/Kimi-K3.1-Pro
(`model.safetensors.index.json`). The models are the real
`KimiK3ForConditionalGeneration` / `KimiLinearForCausalLM` classes around tiny
CPU submodules, so the rule under test is the one the loader finds on them.
"""

import logging
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm.config.load import LoadConfig
from vllm.distributed import parallel_state
from vllm.model_executor.model_loader import DefaultModelLoader, weight_utils
from vllm.model_executor.model_loader.pp_weight_filter import (
    PPCheckpointWeightFilter,
    find_pp_layer_range,
    get_pp_stage_prefetch_split,
)
from vllm.model_executor.model_loader.weight_utils import (
    _prefetch_all_checkpoints,
    safetensors_files_with_used_weights,
    safetensors_weights_iterator,
)
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    StageMissingLayer,
    is_pp_missing_parameter,
    make_layers,
)
from vllm.models.kimi_k3.nvidia import model as kimi

pytestmark = pytest.mark.cpu_test

LM = "language_model.model"

# Per-layer tensor names of the K3.1 checkpoint (MoE / KDA / MLA+indexer).
LAYER_SUFFIXES = (
    "block_sparse_moe.experts.0.w1.weight_packed",
    "block_sparse_moe.experts.3.w2.weight_scale",
    "block_sparse_moe.experts.12.w3.weight_packed",
    "block_sparse_moe.gate.e_score_correction_bias",
    "block_sparse_moe.shared_experts.gate_proj.weight",
    "input_layernorm.weight",
    "self_attention_res_proj.weight",
    "self_attn.A_log",
    "self_attn.q_conv1d.weight",
    "self_attn.indexer.k_norm.bias",
    "self_attn.indexer.wq_b.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.o_proj.weight",
)
FIRST_STAGE_NAMES = (f"{LM}.embed_tokens.weight",)
LAST_STAGE_NAMES = (
    "language_model.lm_head.weight",
    f"{LM}.norm.weight",
    f"{LM}.output_attn_res_norm.weight",
    f"{LM}.output_attn_res_proj.weight",
)
EVERY_STAGE_NAMES = (
    "mm_projector.post_norm.weight",
    "mm_projector.proj.0.weight",
    "mm_projector.proj.2.weight",
    "vision_tower.encoder.blocks.0.wqkv.weight",
    "vision_tower.encoder.blocks.3.mlp.fc1.weight",
    "vision_tower.encoder.blocks.12.norm1.weight",
    "vision_tower.encoder.final_layernorm.weight",
    "vision_tower.patch_embed.proj.weight",
)


def layer_names(layer_idx: int, prefix: str = LM) -> list[str]:
    return [f"{prefix}.layers.{layer_idx}.{suffix}" for suffix in LAYER_SUFFIXES]


def set_groups(monkeypatch, pp_rank=0, pp_size=1, tp_rank=0, tp_size=1):
    """Make this process rank (*pp_rank*, *tp_rank*) of a PP x TP layout."""
    pp_group = SimpleNamespace(
        rank_in_group=pp_rank,
        world_size=pp_size,
        is_first_rank=pp_rank == 0,
        is_last_rank=pp_rank == pp_size - 1,
    )
    tp_group = SimpleNamespace(rank_in_group=tp_rank, world_size=tp_size)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: tp_group)
    # The model module binds the accessor at import.
    monkeypatch.setattr(kimi, "get_pp_group", lambda: pp_group)


def _leaf() -> nn.Module:
    return nn.Linear(1, 1, bias=False)


class TinyLinearModel(nn.Module):
    """`KimiLinearModel`: PP-aware embedding, decoder stack and output head."""

    def __init__(self, num_layers: int):
        super().__init__()
        pp = parallel_state.get_pp_group()
        self.embed_tokens = _leaf() if pp.is_first_rank else PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            num_layers, lambda prefix: _leaf(), prefix="layers"
        )
        for name in ("norm", "output_attn_res_norm", "output_attn_res_proj"):
            setattr(self, name, _leaf() if pp.is_last_rank else PPMissingLayer())


def _bare(cls: type[nn.Module]) -> nn.Module:
    """An instance of the real class without its constructor, which builds
    CUDA modules; the tests attach tiny submodules under the real names."""
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    return module


def tiny_causal_lm(num_layers: int) -> nn.Module:
    model = _bare(kimi.KimiLinearForCausalLM)
    model.model = TinyLinearModel(num_layers)
    is_last_rank = parallel_state.get_pp_group().is_last_rank
    model.lm_head = _leaf() if is_last_rank else PPMissingLayer()
    return model


def tiny_kimi(num_layers: int, text_only: bool = False) -> nn.Module:
    """`KimiK3ForConditionalGeneration`: vision tower and projector on every
    stage, the language model split across stages. With *text_only* the tower
    is marked as not run, as `_mark_tower_model` does for an image limit of 0.
    """
    model = _bare(kimi.KimiK3ForConditionalGeneration)
    vision_tower = nn.ModuleDict(
        {"encoder": nn.ModuleDict({"blocks": nn.ModuleList([_leaf()] * 13)})}
    )
    mm_projector = nn.ModuleDict({"linear_1": _leaf(), "linear_2": _leaf()})
    if text_only:
        model.vision_tower = StageMissingLayer("image", vision_tower)
        model.mm_projector = StageMissingLayer("image", mm_projector)
    else:
        model.vision_tower = vision_tower
        model.mm_projector = mm_projector
    model.language_model = tiny_causal_lm(num_layers)
    return model


def build(
    monkeypatch, num_layers, pp_rank, pp_size, partition=None, text_only=False, **tp
) -> nn.Module:
    if partition is None:
        monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    else:
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    set_groups(monkeypatch, pp_rank, pp_size, **tp)
    return tiny_kimi(num_layers, text_only)


def test_filter_records_its_stage(monkeypatch):
    model = build(monkeypatch, 8, 2, 4, partition="3,2,2,1", tp_rank=5, tp_size=8)
    unused = model.is_unused_checkpoint_weight
    assert (unused.pp_rank, unused.pp_size) == (2, 4)
    assert unused.layer_range == (5, 7)
    assert unused.describe_stage() == "pp rank 2 of 4, layers [5, 7)"
    assert get_pp_stage_prefetch_split() == (5, 8)

    # A model that records no layer range still says which stage it is.
    set_groups(monkeypatch, pp_rank=1, pp_size=2)
    assert (
        PPCheckpointWeightFilter(nn.Linear(1, 1)).describe_stage()
        == "pp rank 1 of 2, layer range unknown"
    )


@pytest.mark.parametrize(
    ("pp_rank", "own_layers"),
    [(0, range(0, 3)), (1, range(3, 5)), (2, range(5, 7)), (3, range(7, 8))],
)
def test_predicate_follows_custom_partition(monkeypatch, pp_rank, own_layers):
    """VLLM_PP_LAYER_PARTITION=3,2,2,1 over 8 layers."""
    model = build(monkeypatch, 8, pp_rank, 4, partition="3,2,2,1")
    unused = model.is_unused_checkpoint_weight
    assert find_pp_layer_range(model) == (own_layers.start, own_layers.stop)

    for layer_idx in range(8):
        expected_unused = layer_idx not in own_layers
        for name in layer_names(layer_idx):
            assert unused(name) is expected_unused, name
    for name in FIRST_STAGE_NAMES:
        assert unused(name) is (pp_rank != 0), name
    for name in LAST_STAGE_NAMES:
        assert unused(name) is (pp_rank != 3), name
    for name in EVERY_STAGE_NAMES:
        assert not unused(name), name


def test_predicate_k31_partition(monkeypatch):
    """The K3.1 deployment: 93 layers over 8 stages as 12,12,12,12,12,12,12,9."""
    ranges = []
    for pp_rank in range(8):
        model = build(monkeypatch, 93, pp_rank, 8, partition="12,12,12,12,12,12,12,9")
        ranges.append(find_pp_layer_range(model))
        unused = model.is_unused_checkpoint_weight
        start, end = ranges[-1]
        probe = f"{LM}.layers.{{}}.block_sparse_moe.experts.5.w1.weight_packed"
        assert [i for i in range(93) if not unused(probe.format(i))] == list(
            range(start, end)
        )
    assert ranges == [(12 * i, 12 * i + 12) for i in range(7)] + [(84, 93)]


def test_predicate_does_not_confuse_numbered_containers(monkeypatch):
    # Stage 1 of 2 over 16 layers owns layers [8, 16).
    model = build(monkeypatch, 16, 1, 2)
    unused = model.is_unused_checkpoint_weight

    # "layers.1" is missing, "layers.1x" are not: whole path components only.
    assert unused(f"{LM}.layers.1.input_layernorm.weight")
    for layer_idx in (10, 11, 12, 15):
        assert not unused(f"{LM}.layers.{layer_idx}.input_layernorm.weight")
    # Expert ids are not layer ids, whichever stage owns the same number.
    assert unused(f"{LM}.layers.3.block_sparse_moe.experts.12.w1.weight_packed")
    assert not unused(f"{LM}.layers.12.block_sparse_moe.experts.3.w1.weight_packed")
    # Neither are vision tower blocks, projector layers or indexer sublayers.
    for name in EVERY_STAGE_NAMES:
        assert not unused(name), name
    assert not unused(f"{LM}.layers.8.self_attn.indexer.k_norm.weight")
    assert unused(f"{LM}.layers.7.self_attn.indexer.k_norm.weight")
    # Names outside the module tree are kept rather than guessed at.
    assert not unused("layers.1.input_layernorm.weight")
    assert not unused("some_new_module.layers.1.weight")
    assert not unused("language_model")

    # The other stage sees the mirror image.
    model = build(monkeypatch, 16, 0, 2)
    unused = model.is_unused_checkpoint_weight
    assert not unused(f"{LM}.layers.1.input_layernorm.weight")
    assert unused(f"{LM}.layers.12.input_layernorm.weight")
    assert not unused(f"{LM}.layers.3.block_sparse_moe.experts.12.w1.weight_packed")


def test_predicate_takes_raw_checkpoint_names(monkeypatch):
    """`is_unused_checkpoint_weight` sees names before `hf_to_vllm_mapper`."""
    model = build(monkeypatch, 8, 1, 4, partition="3,2,2,1")
    unused = model.is_unused_checkpoint_weight
    # K3 checkpoints name the decoder `language_model.layers.N`.
    assert unused("language_model.layers.0.input_layernorm.weight")
    assert not unused("language_model.layers.3.input_layernorm.weight")
    assert unused("language_model.layers.5.input_layernorm.weight")

    # Text-only `KimiLinearForCausalLM` checkpoint: no mapper, no prefix.
    text_unused = model.language_model.is_unused_checkpoint_weight
    assert isinstance(text_unused, PPCheckpointWeightFilter)
    assert text_unused("model.layers.0.input_layernorm.weight")
    assert not text_unused("model.layers.4.input_layernorm.weight")
    assert text_unused("model.embed_tokens.weight")
    assert text_unused("lm_head.weight")


def test_predicate_agrees_with_load_weights(monkeypatch):
    """The rule is the one `KimiLinearModel.load_weights` applies after the
    read: `is_pp_missing_parameter` on the name relative to the decoder."""
    for pp_rank in range(4):
        model = build(monkeypatch, 8, pp_rank, 4, partition="3,2,2,1")
        decoder = model.language_model.model
        names = [n for i in range(8) for n in layer_names(i)]
        names += [f"{LM}.embed_tokens.weight", f"{LM}.norm.weight"]
        for name in names:
            relative = name.removeprefix(f"{LM}.")
            assert model.is_unused_checkpoint_weight(name) is is_pp_missing_parameter(
                relative, decoder
            ), name


def test_single_stage_installs_no_rule(monkeypatch):
    model = build(monkeypatch, 8, 0, 1)
    assert getattr(model, "is_unused_checkpoint_weight", None) is None
    # A filter built anyway rejects nothing.
    unused = PPCheckpointWeightFilter(model, model.hf_to_vllm_mapper)
    assert unused.missing_modules == frozenset()
    names = [n for i in range(8) for n in layer_names(i)]
    names += [*FIRST_STAGE_NAMES, *LAST_STAGE_NAMES, *EVERY_STAGE_NAMES]
    assert not any(unused(name) for name in names)


@pytest.mark.parametrize(
    "cls", [kimi.KimiK3ForConditionalGeneration, kimi.KimiLinearForCausalLM]
)
def test_kimi_classes_expose_the_filter_only_under_pp(monkeypatch, cls):
    """What `DefaultModelLoader.get_all_weights` reads off the model."""
    set_groups(monkeypatch, pp_rank=0, pp_size=1)
    assert getattr(_bare(cls), "is_unused_checkpoint_weight", None) is None

    set_groups(monkeypatch, pp_rank=2, pp_size=4)
    model = _bare(cls)
    unused = getattr(model, "is_unused_checkpoint_weight", None)
    assert isinstance(unused, PPCheckpointWeightFilter)
    assert (unused.pp_rank, unused.pp_size) == (2, 4)
    # One rule per model, so the placeholder paths are collected once.
    assert model.is_unused_checkpoint_weight is unused
    assert "is_unused_checkpoint_weight" not in dict(model.named_modules())


def test_kimi_filter_maps_checkpoint_names(monkeypatch):
    """The multimodal class applies its `hf_to_vllm_mapper`, the text-only
    class takes the names as they are."""
    model = build(monkeypatch, 8, 2, 4, partition="3,2,2,1")
    unused = model.is_unused_checkpoint_weight
    assert [i for i in range(8) if not unused(layer_names(i)[0])] == [5, 6]
    legacy = [layer_names(i, "language_model")[0] for i in range(8)]
    assert [i for i in range(8) if not unused(legacy[i])] == [5, 6]
    assert unused("language_model.lm_head.weight")
    assert not unused("mm_projector.proj.0.weight")

    text_unused = model.language_model.is_unused_checkpoint_weight
    text_names = [layer_names(i, "model")[0] for i in range(8)]
    assert [i for i in range(8) if not text_unused(text_names[i])] == [5, 6]
    # Names of the multimodal checkpoint are outside the text model's tree.
    assert not text_unused(legacy[0])


def test_stage_missing_tower_is_unused(monkeypatch):
    """A tower the deployment does not run (text-only serving) is a
    `StageMissingLayer`; `load_weights` drops its weights like those of a
    `PPMissingLayer`, so they are not read either."""
    model = build(monkeypatch, 8, 1, 4, partition="3,2,2,1", text_only=True)
    unused = model.is_unused_checkpoint_weight
    names = [n for i in range(8) for n in layer_names(i)]
    names += [*FIRST_STAGE_NAMES, *LAST_STAGE_NAMES, *EVERY_STAGE_NAMES]
    mapper = model.hf_to_vllm_mapper
    for name in names:
        assert unused(name) is is_pp_missing_parameter(mapper.map_name(name), model)
    assert all(unused(name) for name in EVERY_STAGE_NAMES)
    assert [n for n in names if not unused(n)] == layer_names(3) + layer_names(4)


# --- checkpoint on disk ------------------------------------------------------

# Shard layout: 02 straddles the stage 0 / stage 1 boundary of the partition
# 3,2,2,1, 04 the stage 2 / stage 3 boundary; 05 holds what the first and last
# stage need; 06 and 07 what every stage needs.
SHARD_LAYERS = {
    "model-00001-of-00007.safetensors": [0, 1],
    "model-00002-of-00007.safetensors": [2, 3],
    "model-00003-of-00007.safetensors": [4, 5],
    "model-00004-of-00007.safetensors": [6, 7],
}
HEAD_SHARD = "model-00005-of-00007.safetensors"
PROJECTOR_SHARD = "model-00006-of-00007.safetensors"
VISION_SHARD = "model-00007-of-00007.safetensors"


def _tensors(names) -> dict[str, torch.Tensor]:
    return {name: torch.full((2,), float(i)) for i, name in enumerate(names)}


@pytest.fixture
def checkpoint(tmp_path):
    contents: dict[str, list[str]] = {
        shard: [n for i in layers for n in layer_names(i)]
        for shard, layers in SHARD_LAYERS.items()
    }
    contents[HEAD_SHARD] = [*FIRST_STAGE_NAMES, *LAST_STAGE_NAMES]
    contents[PROJECTOR_SHARD] = [n for n in EVERY_STAGE_NAMES if "mm_proj" in n]
    contents[VISION_SHARD] = [n for n in EVERY_STAGE_NAMES if "vision" in n]
    for shard, names in contents.items():
        save_file(_tensors(names), tmp_path / shard)
    return tmp_path, contents


@pytest.fixture
def disk_reads(monkeypatch):
    """Record every tensor read and every file prefetched; run the prefetch
    inline instead of in a daemon thread."""
    reads = SimpleNamespace(tensors=[], opened=[], prefetched=[])
    real_safe_open = weight_utils.safe_open

    class RecordingFile:
        def __init__(self, path, *args, **kwargs):
            self._path = os.path.basename(path)
            self._cm = real_safe_open(path, *args, **kwargs)

        def __enter__(self):
            self._f = self._cm.__enter__()
            return self

        def __exit__(self, *exc):
            return self._cm.__exit__(*exc)

        def keys(self):
            return self._f.keys()

        def metadata(self):
            return self._f.metadata()

        def get_tensor(self, name):
            reads.tensors.append(name)
            reads.opened.append(self._path)
            return self._f.get_tensor(name)

    class InlineThread:
        def __init__(self, target, **kwargs):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(weight_utils, "safe_open", RecordingFile)
    # Only the module's own background thread: the prefetch executor keeps
    # the real `threading.Thread`.
    monkeypatch.setattr(weight_utils, "threading", SimpleNamespace(Thread=InlineThread))
    monkeypatch.setattr(
        weight_utils,
        "_prefetch_checkpoint",
        lambda path, block_size: reads.prefetched.append(os.path.basename(path)),
    )
    return reads


def load_names(model, folder, **load_config) -> list[str]:
    loader = DefaultModelLoader(LoadConfig(**load_config))
    model_config = SimpleNamespace(model=str(folder), revision=None)
    return [name for name, _ in loader.get_all_weights(model_config, model)]


def test_shard_list_per_stage(monkeypatch, checkpoint):
    folder, _ = checkpoint
    files = sorted(str(p) for p in folder.glob("*.safetensors"))
    selected = {}
    for pp_rank in range(4):
        model = build(monkeypatch, 8, pp_rank, 4, partition="3,2,2,1")
        kept = safetensors_files_with_used_weights(
            files, model.is_unused_checkpoint_weight
        )
        selected[pp_rank] = [int(os.path.basename(f)[6:11]) for f in kept]
    assert selected == {
        0: [1, 2, 5, 6, 7],  # layers 0-2, embedding
        1: [2, 3, 6, 7],  # layers 3-4
        2: [3, 4, 6, 7],  # layers 5-6
        3: [4, 5, 6, 7],  # layer 7, norm + lm_head
    }


def test_boundary_shard_reads_one_stage_only(monkeypatch, checkpoint, disk_reads):
    folder, contents = checkpoint
    model = build(monkeypatch, 8, 1, 4, partition="3,2,2,1")

    names = load_names(model, folder)

    expected = [
        *layer_names(3),
        *layer_names(4),
        *(n for n in EVERY_STAGE_NAMES if "mm_proj" in n),
        *(n for n in EVERY_STAGE_NAMES if "vision" in n),
    ]
    assert sorted(names) == sorted(expected)
    # Nothing else left the disk: not layer 2 from the boundary shard 02, not
    # layer 5 from the boundary shard 03.
    assert sorted(disk_reads.tensors) == sorted(expected)
    assert set(disk_reads.opened) == {
        "model-00002-of-00007.safetensors",
        "model-00003-of-00007.safetensors",
        PROJECTOR_SHARD,
        VISION_SHARD,
    }


def test_stages_partition_the_checkpoint(monkeypatch, checkpoint, disk_reads):
    folder, contents = checkpoint
    loaded_by: dict[str, list[int]] = {}
    for pp_rank in range(4):
        model = build(monkeypatch, 8, pp_rank, 4, partition="3,2,2,1")
        for name in load_names(model, folder):
            loaded_by.setdefault(name, []).append(pp_rank)

    all_names = {n for names in contents.values() for n in names}
    assert set(loaded_by) == all_names
    for name, ranks in loaded_by.items():
        if name in EVERY_STAGE_NAMES:
            assert ranks == [0, 1, 2, 3], name
        else:
            assert len(ranks) == 1, name
    assert loaded_by[FIRST_STAGE_NAMES[0]] == [0]
    assert all(loaded_by[name] == [3] for name in LAST_STAGE_NAMES)
    # No tensor was read that its reader did not hand to the model.
    assert len(disk_reads.tensors) == sum(len(r) for r in loaded_by.values())


def test_single_stage_loads_everything_unfiltered(
    monkeypatch, checkpoint, disk_reads, iterator_calls
):
    folder, contents = checkpoint
    calls = iterator_calls
    model = build(monkeypatch, 8, 0, 1)

    names = load_names(model, folder, safetensors_load_strategy="prefetch")

    all_names = sorted(n for names in contents.values() for n in names)
    assert sorted(names) == all_names
    assert sorted(disk_reads.tensors) == all_names
    assert calls[0]["is_unused_weight"] is None
    assert calls[0]["prefetch_rank_and_world_size"] is None
    # torch.distributed is not initialized: one rank prefetches every shard.
    assert sorted(disk_reads.prefetched) == sorted(contents)


@pytest.fixture
def iterator_calls(monkeypatch):
    """Keyword arguments of each `safetensors_weights_iterator` call."""
    calls = []
    real_iterator = safetensors_weights_iterator

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_iterator(*args, **kwargs)

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.default_loader.safetensors_weights_iterator",
        spy,
    )
    return calls


@pytest.fixture
def loader_log():
    records: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("vllm.model_executor.model_loader.default_loader")
    handler = Capture(level=logging.INFO)
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield records
    logger.setLevel(level)
    logger.removeHandler(handler)


class DefaultRuleModel(nn.Module, VllmModelForTextGeneration):
    """Inherits `VllmModel.is_unused_checkpoint_weight`, which rejects
    nothing, as the Transformers-backend models do."""


class DrafterModel(nn.Module):
    """A drafter that takes one shard of the target checkpoint."""

    def is_unused_checkpoint_weight(self, name: str) -> bool:
        return not name.startswith("mm_projector.")


@pytest.mark.parametrize(
    ("model_cls", "shards"),
    [(DefaultRuleModel, range(1, 8)), (DrafterModel, [6])],
)
def test_rule_unrelated_to_pp_keeps_the_global_split(
    monkeypatch, checkpoint, disk_reads, iterator_calls, loader_log, model_cls, shards
):
    """Such a rule gives every stage the same shards, so they stay dealt over
    the whole world: with the per-stage split each stage would prefetch all of
    them. No tensor is skipped inside a shard and no stage line is logged."""
    folder, contents = checkpoint
    kept = [f"model-{i:05d}-of-00007.safetensors" for i in shards]
    for pp_rank in range(2):
        for tp_rank in range(2):
            disk_reads.prefetched.clear()
            disk_reads.tensors.clear()
            set_groups(monkeypatch, pp_rank, 2, tp_rank, 2)

            names = load_names(
                model_cls(), folder, safetensors_load_strategy="prefetch"
            )

            assert sorted(names) == sorted(n for f in kept for n in contents[f])
            assert sorted(disk_reads.tensors) == sorted(names)
            assert iterator_calls[-1]["is_unused_weight"] is None
            assert iterator_calls[-1]["prefetch_rank_and_world_size"] is None
            # torch.distributed is not initialized here: the global world is
            # one rank, which takes every kept shard whatever its TP rank.
            assert sorted(disk_reads.prefetched) == kept
    assert not [r for r in loader_log if r.startswith("PP weight filter")]


def test_rule_rejecting_everything_is_not_applied(
    monkeypatch, tmp_path, checkpoint, disk_reads, iterator_calls
):
    """Same fallback as without PP: e.g. a drafter whose weights live in
    another source keeps receiving the full stream."""
    folder, contents = checkpoint
    set_groups(monkeypatch, pp_rank=1, pp_size=4)

    class RejectsAll(nn.Module):
        def is_unused_checkpoint_weight(self, name: str) -> bool:
            return True

    names = load_names(RejectsAll(), folder)
    assert sorted(names) == sorted(n for names in contents.values() for n in names)

    # The stage rule likewise: a source holding only another stage's layers.
    other = tmp_path / "other"
    other.mkdir()
    save_file(_tensors(layer_names(0)), other / "model.safetensors")
    model = build(monkeypatch, 8, 1, 4, partition="3,2,2,1")
    names = load_names(model, other)
    assert sorted(names) == sorted(layer_names(0))
    assert iterator_calls[-1]["is_unused_weight"] is None
    assert iterator_calls[-1]["prefetch_rank_and_world_size"] is None


def test_iterator_skips_unused_tensors_before_reading(checkpoint, disk_reads):
    folder, contents = checkpoint
    boundary = "model-00002-of-00007.safetensors"

    def is_layer_2(name: str) -> bool:
        return ".layers.2." in name

    names = [
        name
        for name, _ in safetensors_weights_iterator(
            [str(folder / boundary)], False, is_unused_weight=is_layer_2
        )
    ]
    assert sorted(names) == sorted(layer_names(3))
    assert sorted(disk_reads.tensors) == sorted(layer_names(3))

    # Without a rule, and with the eager strategy, nothing changes.
    names = [
        name
        for name, _ in safetensors_weights_iterator([str(folder / boundary)], False)
    ]
    assert sorted(names) == sorted(contents[boundary])
    names = [
        name
        for name, _ in safetensors_weights_iterator(
            [str(folder / boundary)], False, "eager", is_unused_weight=is_layer_2
        )
    ]
    assert sorted(names) == sorted(layer_names(3))


# --- prefetch ------------------------------------------------------------------


def test_prefetch_split_is_per_stage(monkeypatch, checkpoint, disk_reads):
    """The TP ranks of one stage prefetch that stage's shards between them,
    each shard once, and no shard of another stage."""
    folder, _ = checkpoint
    tp_size = 4
    stage_files = {
        0: [1, 2, 5, 6, 7],
        1: [2, 3, 6, 7],
        2: [3, 4, 6, 7],
        3: [4, 5, 6, 7],
    }
    for pp_rank, expected in stage_files.items():
        per_tp_rank = []
        for tp_rank in range(tp_size):
            disk_reads.prefetched.clear()
            model = build(
                monkeypatch,
                8,
                pp_rank,
                4,
                partition="3,2,2,1",
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
            load_names(model, folder, safetensors_load_strategy="prefetch")
            per_tp_rank.append([int(f[6:11]) for f in disk_reads.prefetched])
        # Round-robin over the stage's own list.
        assert per_tp_rank == [expected[r::tp_size] for r in range(tp_size)]
        assert sorted(f for files in per_tp_rank for f in files) == expected


def test_prefetch_all_checkpoints_split(monkeypatch, disk_reads):
    files = [f"model-{i:05d}-of-00012.safetensors" for i in range(1, 13)]

    def prefetched(**kwargs) -> list[str]:
        disk_reads.prefetched.clear()
        _prefetch_all_checkpoints(files, **kwargs)
        return sorted(disk_reads.prefetched)

    # Default: the global torch.distributed world (not initialized -> 1 rank).
    assert prefetched() == files
    # Global rank 17 of 64 would have taken shard 18 of 96; within its stage
    # (pp rank 2, tp rank 1 of 8) it takes every 8th of the stage's 12 files.
    assert prefetched(rank_and_world_size=(1, 8)) == [files[1], files[9]]
    shares = [prefetched(rank_and_world_size=(r, 8)) for r in range(8)]
    assert sorted(f for share in shares for f in share) == files
    assert [len(share) for share in shares] == [2, 2, 2, 2, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="outside of world size"):
        _prefetch_all_checkpoints(files, rank_and_world_size=(8, 8))


# --- log line --------------------------------------------------------------------


def test_log_line_states_stage_and_selection(monkeypatch, checkpoint, loader_log):
    folder, _ = checkpoint
    model = build(monkeypatch, 8, 1, 4, partition="3,2,2,1")
    load_names(model, folder)

    lines = [r for r in loader_log if r.startswith("PP weight filter")]
    assert len(lines) == 1
    files = sorted(folder.glob("*.safetensors"))
    total = sum(f.stat().st_size for f in files) / 1024**3
    kept = sum(f.stat().st_size for f in files if f.name[10] in "2367") / 1024**3
    assert lines[0] == (
        "PP weight filter: pp rank 1 of 4, layers [3, 5): 4 of 7 safetensors "
        f"shards selected ({kept:.2f} GiB of {total:.2f} GiB)"
    )
