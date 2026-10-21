# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SWA bounded replay through the SM80 (Ampere) sparse-attention path.

The metadata builder clamps every prefill window and gather at the request's
``replay_start``; the Ampere decode kernel must then never touch a KV slot below
that position, and the ``-1``-padded SWA rows must stay finite.
"""

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.models.deepseek_v41.sparse_mla import DeepseekV41SparseSWAMetadataBuilder
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

CUDA = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 8

pytestmark = pytest.mark.skipif(not CUDA, reason="needs an SM80 (Ampere) GPU")

DEVICE = torch.device("cuda")
WINDOW = 8
CACHE_BLOCK_SIZE = 32
BYTES_PER_TOKEN = 584
HEAD_DIM = 512
H_Q = 16


def _make_builder(
    window: int = WINDOW,
    block_size: int = CACHE_BLOCK_SIZE,
    max_num_batched_tokens: int = 64,
) -> DeepseekV41SparseSWAMetadataBuilder:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3-0.6B",
        max_model_len=4096,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=8,
        hf_config_override={"sliding_window": window},
    )
    spec = SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=HEAD_DIM,
        dtype=torch.bfloat16,
        sliding_window=window,
        cache_dtype_str="auto",
        model_version="deepseek_v4",
        bounded_replay=True,
    )
    return DeepseekV41SparseSWAMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["layer0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _make_batch(seq_lens: list[int], query_lens: list[int]):
    """Decode-first batch: contiguous blocks per request, slots = block * BS."""
    num_reqs = len(seq_lens)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    query_start_loc[1:] = torch.tensor(
        query_lens, dtype=torch.int32, device=DEVICE
    ).cumsum(0)
    num_tokens = int(query_start_loc[-1])
    max_blocks = max((s + CACHE_BLOCK_SIZE - 1) // CACHE_BLOCK_SIZE for s in seq_lens)
    block_table = torch.arange(
        num_reqs * max_blocks, dtype=torch.int32, device=DEVICE
    ).view(num_reqs, max_blocks)
    return dict(
        query_start_loc=query_start_loc,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
        block_table=block_table,
        slot_mapping=torch.arange(num_tokens, dtype=torch.int64, device=DEVICE),
        num_tokens=num_tokens,
        max_blocks=max_blocks,
    )


def _build(builder, seq_lens, query_lens, replay_starts=None):
    batch = _make_batch(seq_lens, query_lens)
    replay_start = (
        None
        if replay_starts is None
        else torch.tensor(replay_starts, dtype=torch.int32, device=DEVICE)
    )
    metadata = builder.build(
        0,
        CommonAttentionMetadata(
            query_start_loc=batch["query_start_loc"],
            query_start_loc_cpu=batch["query_start_loc"].cpu(),
            seq_lens=batch["seq_lens"],
            seq_lens_cpu_upper_bound=batch["seq_lens"].cpu(),
            num_reqs=len(seq_lens),
            num_actual_tokens=batch["num_tokens"],
            max_query_len=max(query_lens),
            max_seq_len=max(seq_lens),
            block_table_tensor=batch["block_table"],
            slot_mapping=batch["slot_mapping"],
            causal=True,
        ),
        replay_start=replay_start,
    )
    return metadata, batch["block_table"]


def _ref_rows(seq_lens, query_lens, block_table, replay_starts, window=WINDOW):
    """Reference paged rows: the causal window clamped at replay_start."""
    rows, lens = [], []
    block_table_cpu = block_table.cpu()
    for req, (seq_len, query_len) in enumerate(zip(seq_lens, query_lens)):
        prefix_len = seq_len - query_len
        for i in range(query_len):
            pos = prefix_len + i
            start = max(pos - (window - 1), 0, replay_starts[req])
            row = []
            for p in range(start, pos + 1):
                block = int(block_table_cpu[req, p // CACHE_BLOCK_SIZE])
                row.append(block * CACHE_BLOCK_SIZE + p % CACHE_BLOCK_SIZE)
            lens.append(len(row))
            rows.append(row + [-1] * (window - len(row)))
    return rows, lens


def _slot_positions(block_table, req, slots):
    """Invert the (contiguous) block table: global slot -> token position."""
    blocks = block_table.cpu()[req]
    positions = []
    for slot in slots:
        block, offset = divmod(int(slot), CACHE_BLOCK_SIZE)
        index = (blocks == block).nonzero()
        assert index.numel() == 1, f"slot {slot} is not in request {req}'s blocks"
        positions.append(int(index[0]) * CACHE_BLOCK_SIZE + offset)
    return positions


def test_prefill_window_and_gather_clamp_at_replay_start():
    """A mixed decode+prefill batch: no prefill index points below replay_start."""
    seq_lens = [64, 40, 40, 12]
    query_lens = [1, 1, 24, 12]
    replay_starts = [0, 0, 16, 0]
    builder = _make_builder()
    md, block_table = _build(builder, seq_lens, query_lens, replay_starts)

    assert md.replay_start.cpu().tolist() == replay_starts
    assert (md.num_decodes, md.num_prefills) == (2, 2)
    assert (md.num_decode_tokens, md.num_prefill_tokens) == (2, 36)

    # gather_len = query_len + min(prefix_len - replay_start, window - 1).
    assert md.prefill_gather_lens.cpu().tolist() == [
        24 + min(16 - 16, WINDOW - 1),
        12 + min(0, WINDOW - 1),
    ]

    rows, lens = _ref_rows(seq_lens, query_lens, block_table, replay_starts)
    prefill_rows = rows[2:]
    prefill_lens = lens[2:]
    assert md.prefill_swa_lens.cpu().tolist() == prefill_lens
    assert md.prefill_swa_indices[:, 0].cpu().tolist() == prefill_rows

    # The replaying request's first row sees exactly [replay_start, pos], and a
    # full window appears only once the query has WINDOW-1 tokens above it.
    assert prefill_lens[0] == 1
    assert prefill_lens[WINDOW - 2] == WINDOW - 1
    assert prefill_lens[WINDOW - 1] == WINDOW

    # Every prefill row names a contiguous run of positions that starts at or
    # above the request's replay start, and ends at its query token.
    token = 0
    for req in range(2, len(seq_lens)):
        prefix_len = seq_lens[req] - query_lens[req]
        for i in range(query_lens[req]):
            row_len = prefill_lens[token]
            positions = _slot_positions(
                block_table, req, prefill_rows[token][:row_len]
            )
            assert positions == list(range(min(positions), positions[-1] + 1))
            assert min(positions) >= max(
                replay_starts[req], prefix_len + i - WINDOW + 1
            )
            assert positions[-1] == prefix_len + i
            token += 1


def test_ampere_decode_never_reads_below_replay_start():
    """Garbage in the KV slots below replay_start must not change the output."""
    from vllm.models.deepseek_v41.ampere.sparse_decode import ampere_sparse_decode
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )

    seq_len, query_len, replay_start = 40, 8, 16
    builder = _make_builder()
    md, _ = _build(builder, [seq_len], [query_len], [replay_start])
    swa_indices = md.prefill_swa_indices[:, 0].contiguous()
    assert swa_indices.shape == (query_len, WINDOW)
    assert int(swa_indices.min()) >= replay_start

    torch.manual_seed(0)
    k = torch.randn(seq_len, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    slots = torch.arange(seq_len, dtype=torch.int64, device=DEVICE)

    def encode(rows):
        cache = torch.zeros(
            (2, CACHE_BLOCK_SIZE, BYTES_PER_TOKEN), dtype=torch.uint8, device=DEVICE
        )
        quantize_and_insert_k_cache(
            rows,
            cache,
            slots,
            block_size=CACHE_BLOCK_SIZE,
            bytes_per_token=BYTES_PER_TOKEN,
        )
        return cache

    clean = encode(k)
    garbage = clean.clone()
    # Slots 0..replay_start-1 are the only cache bytes below the replay start.
    polluted = torch.randint(
        0,
        256,
        (replay_start * BYTES_PER_TOKEN,),
        dtype=torch.uint8,
        device=DEVICE,
    )
    garbage.view(-1)[: polluted.numel()] = polluted
    assert not torch.equal(clean.view(-1)[: polluted.numel()], polluted)

    q = torch.randn(query_len, H_Q, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    sink = torch.randn(H_Q, dtype=torch.float32, device=DEVICE)
    sm_scale = HEAD_DIM**-0.5

    def decode(cache):
        out = torch.zeros(
            query_len, H_Q, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
        )
        ampere_sparse_decode(q, cache, swa_indices, None, None, sm_scale, sink, out)
        return out

    with_clean = decode(clean)
    with_garbage = decode(garbage)
    assert torch.isfinite(with_clean).all()
    assert torch.equal(with_clean, with_garbage)


def test_padded_swa_rows_stay_finite_and_match_the_trimmed_row():
    """A short sequence's ``-1`` SWA padding is masked, not attended to."""
    from vllm.models.deepseek_v41.ampere.sparse_decode import ampere_sparse_decode
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )

    builder = _make_builder()
    md, _ = _build(builder, [3], [1])
    assert md.num_decode_tokens == 1
    padded = md.decode_swa_indices[:, 0].contiguous()
    row_len = int(md.decode_swa_lens[0])
    assert row_len == 3
    assert padded[0, row_len:].tolist() == [-1] * (WINDOW - row_len)

    torch.manual_seed(1)
    k = torch.randn(3, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    cache = torch.zeros(
        (1, CACHE_BLOCK_SIZE, BYTES_PER_TOKEN), dtype=torch.uint8, device=DEVICE
    )
    quantize_and_insert_k_cache(
        k,
        cache,
        torch.arange(3, dtype=torch.int64, device=DEVICE),
        block_size=CACHE_BLOCK_SIZE,
        bytes_per_token=BYTES_PER_TOKEN,
    )

    q = torch.randn(1, H_Q, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    sink = torch.randn(H_Q, dtype=torch.float32, device=DEVICE)
    sm_scale = HEAD_DIM**-0.5

    def decode(indices):
        out = torch.zeros(
            indices.shape[0], H_Q, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
        )
        ampere_sparse_decode(q, cache, indices, None, None, sm_scale, sink, out)
        return out

    with_pad = decode(padded)
    trimmed = decode(padded[:, :row_len].contiguous())
    assert torch.isfinite(with_pad).all()
    rel = (with_pad[0].float() - trimmed[0].float()).abs().max()
    rel /= trimmed[0].float().abs().max()
    assert rel < 2e-2, f"max rel err {rel:.3e}"


def test_bounded_replay_spec_reaches_the_builder_metadata():
    """The spec's bounded_replay flag is what the launch flag toggles."""
    builder = _make_builder()
    assert builder.kv_cache_spec.bounded_replay is True
    assert builder.window_size == WINDOW
