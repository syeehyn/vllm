# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sparse_attn_indexer on the Triton MQA logits kernels (CUDA below SM90),
driven as the Kimi K3.1 indexer drives it: the top-k of every row against
float64 math over the dequantized Q and K."""

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
from vllm.config import CUDAGraphMode
from vllm.models.kimi_k3.nvidia.dsa import KimiK3DSAIndexer
from vllm.models.kimi_k3.nvidia.ops.dsa_indexer import (
    indexer_k_norm_quant_insert,
    indexer_q_quant_fold_weights,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import mqa_logits_use_mx_scales
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerPrefillMetadata,
    build_prefill_chunk_metadata,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or mqa_logits_use_mx_scales(),
    reason="Triton MQA logits need CUDA and fp32 indexer key scales",
)

HEAD_DIM = 128
NUM_HEADS = 32
ENTRY = HEAD_DIM + 4  # fp8 values + one fp32 scale per token
BLOCK = 64  # K3.1 kernel block size
MLA_BYTES = 1152  # bf16 MLA bytes per token ahead of the indexer page
ROW_STRIDE = 82944  # packed block row: 64 x (1152 + 132), 1152-aligned
TOPK = 2048
PREFIX = "model.layers.4.self_attn.indexer.k_cache"


@pytest.fixture(autouse=True)
def triton_logits(monkeypatch: pytest.MonkeyPatch):
    """The path CUDA below SM90 takes; forced, so SM90 covers it as well."""
    monkeypatch.setattr(sparse_indexer, "use_triton_mqa_logits", lambda: True)


def _make_cache(num_blocks: int) -> torch.Tensor:
    """One DSA layer's indexer page view `[num_blocks, BLOCK, ENTRY]` inside
    packed block rows, as the op receives it."""
    raw = torch.zeros(num_blocks, ROW_STRIDE, dtype=torch.uint8, device="cuda")
    return torch.as_strided(
        raw,
        size=(num_blocks, BLOCK, ENTRY),
        stride=(ROW_STRIDE, ENTRY, 1),
        storage_offset=BLOCK * MLA_BYTES,
    )


def _dequant(cache: torch.Tensor) -> torch.Tensor:
    """Every slot of the indexer cache as float64 `[num_blocks * BLOCK, D]`."""
    flat = cache.reshape(cache.shape[0], -1)
    values = flat[:, : BLOCK * HEAD_DIM].contiguous().view(current_platform.fp8_dtype())
    scales = flat[:, BLOCK * HEAD_DIM :].contiguous().view(torch.float32)
    return values.view(-1, HEAD_DIM).float().double() * scales.reshape(-1, 1).double()


def _assert_topk(topk: torch.Tensor, ref: torch.Tensor, visible: torch.Tensor):
    """Row i of `topk` is a top-k of `ref[i, :visible[i]]`, -1 padded. Scores
    within fp32 rounding of the k-th one may swap."""
    hidden = torch.arange(ref.shape[1], device=ref.device) >= visible[:, None]
    valid = topk >= 0
    assert int(topk.max()) < ref.shape[1]
    picked = torch.zeros(ref.shape, dtype=torch.int32, device=ref.device)
    picked.scatter_add_(1, topk.clamp(min=0).long(), valid.int())
    assert int(picked.max()) <= 1, "duplicate index"
    assert not (picked.bool() & hidden).any(), "index past the row's context"
    assert torch.equal(picked.sum(1), visible.clamp(max=TOPK))

    scores = ref.masked_fill(hidden, float("-inf"))
    kth = scores.topk(min(TOPK, ref.shape[1]), dim=1).values[:, -1]
    worst = scores.masked_fill(picked == 0, float("inf")).amin(1)
    tol = 1e-5 * ref.masked_fill(hidden, 0).abs().amax(1)
    bad = torch.nonzero(worst < kth - tol).flatten().tolist()
    assert not bad, f"rows {bad[:8]} miss the reference top-k"


@pytest.mark.parametrize(
    "decodes,prefills,next_n,logits_mb",
    [
        # The last request's keys start past column 0 of the gathered K.
        pytest.param([], [(0, 700), (0, 2500)], 1, 512, id="prefill_chunk"),
        # A 1 MiB logits budget splits each chunk's rows over several calls.
        pytest.param([], [(0, 600), (3000, 300)], 1, 1, id="prefill_continuation"),
        pytest.param(
            [(999, 1), (2999, 1), (6099, 1), (4499, 1)], [], 1, 512, id="uniform_decode"
        ),
        # Past 32768 keys the decode top-k takes its multi-CTA radix path, on
        # fp32 logits whose row stride is the batch's longest context.
        pytest.param([(40000, 1), (33001, 1)], [], 1, 512, id="long_decode"),
        # The middle request holds one token of next_n = 2: the padded path.
        pytest.param([(2998, 2), (4100, 1), (2500, 2)], [], 2, 512, id="ragged_decode"),
        pytest.param(
            [(2999, 1), (99, 1)],
            [(300, 200), (2100, 400)],
            1,
            512,
            id="decode_and_prefill",
        ),
    ],
)
@torch.inference_mode()
def test_topk_matches_float64_reference(
    monkeypatch: pytest.MonkeyPatch,
    workspace_init,
    decodes: list[tuple[int, int]],
    prefills: list[tuple[int, int]],
    next_n: int,
    logits_mb: int,
):
    """`decodes` and `prefills` are (cached context, new tokens) per request.
    Each new token's top-k over its request's keys, written by the fused
    insert into a packed cache, is the top-k of the float64 scores."""
    torch.manual_seed(0)
    monkeypatch.setattr(envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", logits_mb)
    reqs = decodes + prefills
    num_reqs, num_decodes = len(reqs), len(decodes)
    query_lens = torch.tensor([n for _, n in reqs], dtype=torch.int32)
    seq_lens_cpu = torch.tensor([c + n for c, n in reqs], dtype=torch.int32)
    seq_lens = seq_lens_cpu.cuda()
    query_start_loc_cpu = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc_cpu[1:] = query_lens.cumsum(0)
    num_tokens = int(query_start_loc_cpu[-1])
    num_decode_tokens = int(query_start_loc_cpu[num_decodes])

    width = (int(seq_lens_cpu.max()) + BLOCK - 1) // BLOCK
    num_blocks = num_reqs * width + 1
    perm = torch.randperm(num_blocks, device="cuda").int()
    block_table = perm[: num_reqs * width].view(num_reqs, width).contiguous()
    cache = _make_cache(num_blocks)

    def slots(req: int, start: int, stop: int) -> torch.Tensor:
        pos = torch.arange(start, stop, device="cuda")
        return block_table[req, pos // BLOCK].long() * BLOCK + pos % BLOCK

    # The context was inserted by earlier steps, this step's keys right before
    # scoring; that insert also clears the step's top-k rows.
    context_slots = torch.cat([slots(r, 0, c) for r, (c, _) in enumerate(reqs)])
    step_slots = torch.cat([slots(r, c, c + n) for r, (c, n) in enumerate(reqs)])
    topk = torch.full(
        (max(num_tokens, num_decodes * next_n), TOPK),
        7,
        dtype=torch.int32,
        device="cuda",
    )
    k_w = torch.rand(HEAD_DIM, device="cuda") + 0.5
    k_b = torch.randn(HEAD_DIM, device="cuda") * 0.1
    for slot_mapping, topk_rows in (
        (context_slots, topk.new_empty(context_slots.shape[0], TOPK)),
        (step_slots, topk),
    ):
        rows = slot_mapping.shape[0]
        if rows == 0:
            continue
        k = torch.randn(rows, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        indexer_k_norm_quant_insert(
            k, k_w, k_b, 1e-6, slot_mapping, cache, topk_rows, rows
        )

    decode = prefill = None
    if decodes:
        decode_lens = query_lens[:num_decodes].cuda()
        # Native decode: token j of a request sees L - next_n + j + 1 keys.
        offsets = torch.arange(next_n, dtype=torch.int32, device="cuda")
        decode = DeepSeekV32IndexerDecodeMetadata(
            block_table=block_table[:num_decodes],
            seq_lens=(seq_lens[:num_decodes, None] - next_n + 1 + offsets).contiguous(),
            decode_lens=decode_lens,
            requires_padding=bool(decode_lens.min() != decode_lens.max()),
            schedule_metadata=torch.empty(0, dtype=torch.int32, device="cuda"),
        )
    if prefills:
        chunk = build_prefill_chunk_metadata(
            start_idx=num_decodes,
            end_idx=num_reqs,
            query_start_loc=query_start_loc_cpu.cuda(),
            query_start_loc_cpu=query_start_loc_cpu,
            uncompressed_seq_lens=seq_lens,
            compressed_seq_lens=seq_lens,
            compressed_seq_lens_cpu=seq_lens_cpu,
            block_table=block_table,
            compress_ratio=1,
        )
        assert chunk is not None
        prefill = DeepseekV32IndexerPrefillMetadata(
            [chunk], max_prefill_seq_len=int(seq_lens_cpu[num_decodes:].max())
        )
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=seq_lens,
        max_seq_len=int(seq_lens_cpu.max()),
        slot_mapping=step_slots,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=len(prefills),
        num_prefill_tokens=num_tokens - num_decode_tokens,
        decode=decode,
        prefill=prefill,
    )
    monkeypatch.setattr(
        sparse_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={PREFIX: metadata},
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        ),
    )
    indexer = SimpleNamespace(
        use_fp4_cache=False,
        index_softmax_scale=HEAD_DIM**-0.5,
        indexer_weight_scale=NUM_HEADS**-0.5,
        padded_index_num_heads=NUM_HEADS,
        k_cache=SimpleNamespace(prefix=PREFIX, kv_cache=cache),
        quant_block_size=HEAD_DIM,
        scale_fmt="ue8m0",
        topk_tokens=TOPK,
        index_head_dim=HEAD_DIM,
        max_model_len=width * BLOCK,
        max_total_seq_len=width * BLOCK,
        topk_indices_buffer=topk,
        use_fused_k_insert=True,
        use_pcp=False,
        _parallel_config=SimpleNamespace(pcp_shard_decode_requests=False),
        dense_mha_metadata_layer_name="",
        dcp_rank=0,
        dcp_world_size=1,
        cp_kv_cache_interleave_size=1,
        topk_backend="auto",
    )
    q = torch.randn(
        num_tokens, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    w = torch.randn(num_tokens, NUM_HEADS, dtype=torch.bfloat16, device="cuda")
    hidden_states = torch.empty(num_tokens, 16, dtype=torch.bfloat16, device="cuda")
    KimiK3DSAIndexer.score(indexer, hidden_states, q, w, None)

    weight_scale = indexer.index_softmax_scale * indexer.indexer_weight_scale
    q_fp8, weights = indexer_q_quant_fold_weights(q, w, weight_scale, NUM_HEADS)
    q_deq, weights = q_fp8.float().double(), weights.double()
    k_deq = _dequant(cache)
    for r, (context, new_tokens) in enumerate(reqs):
        seq_len = context + new_tokens
        keys = k_deq[slots(r, 0, seq_len)]
        start = int(query_start_loc_cpu[r])
        # A decode token is the j-th of next_n; a prefill token sees itself.
        first = seq_len - next_n + 1 if r < num_decodes else context + 1
        for lo in range(0, new_tokens, 128):
            rows = slice(start + lo, start + min(lo + 128, new_tokens))
            ref = torch.einsum("mhd,nd->mhn", q_deq[rows], keys).relu_()
            ref = (ref * weights[rows, :, None]).sum(1)
            visible = first + torch.arange(lo, lo + ref.shape[0], device="cuda")
            _assert_topk(topk[rows], ref, visible)
