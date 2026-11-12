# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the Triton MQA logits kernels."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_mqa_logits_triton,
    fp8_paged_mqa_logits_triton,
)

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")


def _quantize_k_per_row(
    k_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each key row to FP8 and return its dequantization scale."""
    amax = k_bf16.abs().float().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    sf = amax / 448.0
    k_fp8 = (k_bf16.float() / sf).to(torch.float8_e4m3fn)
    return k_fp8, sf.squeeze(-1)


def _pack_paged_kv(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack BF16 KV into the layout produced by `indexer_k_quant_and_cache`.

    Physical bytes per block are segregated: all `block_size * head_dim` fp8 K
    bytes first, then `block_size * 4` fp32 scale bytes. The outer
    `[NB, BS, 1, D+4]` shape matches the production cache allocation.
    """
    num_blocks, block_size, head_dim = kv_bf16.shape
    amax = kv_bf16.abs().float().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    sf = (amax / 448.0).to(torch.float32)
    k_fp8 = (kv_bf16.float() / sf).to(torch.float8_e4m3fn)
    packed = torch.empty(
        (num_blocks, block_size, 1, head_dim + 4),
        dtype=torch.uint8,
        device=kv_bf16.device,
    )
    flat = packed.view(num_blocks, -1)
    k_end = block_size * head_dim
    flat[:, :k_end] = k_fp8.reshape(num_blocks, -1).view(torch.uint8)
    flat[:, k_end:] = sf.reshape(num_blocks, -1).view(torch.uint8)
    return packed


# References adapted from DeepGEMM (test_attention.py) — used as the spec
# the Triton kernels must agree with. They score the dequantized bytes in
# float64, so the comparison measures the kernels' own rounding.
def _fp8_mqa_logits_ref(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute dense MQA indexing logits from dequantized keys."""
    k_fp8, scale = kv
    seq_len_kv = k_fp8.shape[0]
    k = k_fp8.float().double() * scale.double()[:, None]
    q = q.float().double()
    arange = torch.arange(0, seq_len_kv, device=q.device)[None, :]
    mask = (arange >= cu_seqlen_ks[:, None]) & (arange < cu_seqlen_ke[:, None])
    score = torch.einsum("mhd,nd->hmn", q, k)
    logits = (score.relu() * weights.double().unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    return logits.masked_fill(~mask, float("-inf"))


def _fp8_paged_mqa_logits_ref(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute reference indexing logits from a paged FP8 key cache."""
    fp8_dtype = torch.float8_e4m3fn
    batch_size, next_n, _, dim = q.size()
    num_blocks, block_size = kv_cache.shape[0], kv_cache.shape[1]
    flat = kv_cache.view(num_blocks, -1)
    k_end = block_size * dim
    kv_data = (
        flat[:, :k_end].reshape(num_blocks, block_size, 1, dim).view(fp8_dtype).float()
    )
    scale = flat[:, k_end:].view(torch.float32).reshape(num_blocks, block_size, 1, 1)
    q = q.float().double()
    weights = weights.double()
    kv_data = kv_data.double() * scale.double()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float64,
    )
    if context_lens.ndim == 1:
        token_offsets = torch.arange(next_n - 1, -1, -1, device=q.device)
        context_lens = context_lens[:, None] - token_offsets[None, :]
    context_lens_list = context_lens.tolist()
    for i in range(batch_size):
        token_context_lens = context_lens_list[i]
        context_len = max(token_context_lens)
        q_offsets = torch.tensor(token_context_lens, device=q.device) - 1
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_data[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size,
                (block_rk + 1) * block_size,
                device=q.device,
            )
            mask = (k_offsets[None, :] < context_lens[i, :, None]) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = (torch.relu(s) * weight_slice[..., None]).sum(dim=0)
            block_start = block_rk * block_size
            block_end = min((block_rk + 1) * block_size, max_model_len)
            block_width = block_end - block_start
            logits[
                i * next_n : (i + 1) * next_n,
                block_start:block_end,
            ] = torch.where(
                k_offsets[None, :block_width] <= q_offsets[:, None],
                s[:, :block_width],
                float("-inf"),
            )
    return logits


# The kernels multiply exact E4M3 values with an fp32 accumulator, so only the
# fp32 rounding of the per-head scale, weight and sum separates them from the
# float64 references.
_ATOL = 1e-4
_RTOL = 1e-5


@pytest.mark.parametrize("M,N", [(64, 64), (128, 256), (256, 512)])
@pytest.mark.parametrize("num_heads", [16, 32])
@pytest.mark.parametrize("partial_mask", [False, True])
@pytest.mark.parametrize("clean_logits", [True, False])
def test_fp8_mqa_logits_triton_matches_torch(
    M, N, num_heads, partial_mask, clean_logits
):
    """`clean_logits=True` requires full-tensor agreement with the reference;
    `clean_logits=False` is only contractually correct on the in-range
    `[ks, ke)` slots, which is what `finite` already masks here (the reference
    writes -inf outside that range)."""
    torch.manual_seed(0)
    head_dim = 128
    device = "cuda"

    q_bf16 = torch.randn(M, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k_bf16 = torch.randn(N, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(M, num_heads, dtype=torch.float32, device=device)
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)
    k_fp8, k_scales = _quantize_k_per_row(k_bf16)

    if partial_mask:
        # Exercise the non-trivial ks/ke masking branch (chunked prefill).
        ks = torch.arange(M, dtype=torch.int32, device=device) % (N // 4)
        ke = ks + torch.randint(1, N // 2, (M,), dtype=torch.int32, device=device)
        ke = torch.minimum(ke, torch.tensor(N, dtype=torch.int32, device=device))
    else:
        ks = torch.zeros(M, dtype=torch.int32, device=device)
        ke = torch.full((M,), N, dtype=torch.int32, device=device)

    # Strided inputs must retain the same per-row scales and masking ranges.
    k_scales, ks, ke = [
        torch.stack((tensor, torch.zeros_like(tensor)), dim=-1)[:, 0]
        for tensor in (k_scales, ks, ke)
    ]

    out_torch = _fp8_mqa_logits_ref(q_fp8, (k_fp8, k_scales), weights, ks, ke)
    out_triton = fp8_mqa_logits_triton(
        q_fp8, (k_fp8, k_scales), weights, ks, ke, clean_logits=clean_logits
    )

    if clean_logits:
        assert torch.equal(
            torch.isinf(out_torch) & (out_torch < 0),
            torch.isinf(out_triton) & (out_triton < 0),
        )
    finite = ~torch.isinf(out_torch)
    if finite.any():
        torch.testing.assert_close(
            out_triton[finite].double(), out_torch[finite], atol=_ATOL, rtol=_RTOL
        )


@pytest.mark.parametrize("M,N", [(64, 256), (128, 512)])
@pytest.mark.parametrize("num_heads", [16, 32])
def test_fp8_mqa_logits_triton_row_local_layout(M, N, num_heads):
    """`max_seqlen_k` reproduces DeepGEMM's row-local contract.

    The indexer's top-k and candidate selection expect column `j` of row `m`
    to be K position `ks[m] + j` with the row right-aligned to zero, which is
    what DeepGEMM emits and what chunked prefill needs when requests start at
    different offsets.
    """
    torch.manual_seed(0)
    head_dim = 128
    device = "cuda"

    q_bf16 = torch.randn(M, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k_bf16 = torch.randn(N, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(M, num_heads, dtype=torch.float32, device=device)
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)
    k_fp8, k_scales = _quantize_k_per_row(k_bf16)

    ks = torch.arange(M, dtype=torch.int32, device=device) % (N // 4)
    ke = ks + torch.randint(1, N // 2, (M,), dtype=torch.int32, device=device)
    ke = torch.minimum(ke, torch.tensor(N, dtype=torch.int32, device=device))
    width = int((ke - ks).max().item())

    abs_ref = _fp8_mqa_logits_ref(q_fp8, (k_fp8, k_scales), weights, ks, ke)
    expected = torch.full((M, width), float("-inf"), dtype=torch.float64, device=device)
    for m in range(M):
        start, end = int(ks[m].item()), int(ke[m].item())
        expected[m, : end - start] = abs_ref[m, start:end]

    out = fp8_mqa_logits_triton(
        q_fp8,
        (k_fp8, k_scales),
        weights,
        ks,
        ke,
        clean_logits=True,
        max_seqlen_k=width,
    )
    assert out.shape == (M, width)
    assert torch.equal(
        torch.isinf(out) & (out < 0), torch.isinf(expected) & (expected < 0)
    )
    finite = ~torch.isinf(expected)
    torch.testing.assert_close(
        out[finite].double(), expected[finite], atol=_ATOL, rtol=_RTOL
    )


def test_fp8_mqa_logits_triton_clean_logits_false_overwrites_masked():
    """Regression test for PR #38476 comment 4398225404: when `clean_logits=False`
    skips the `-inf` pre-fill, the kernel itself must still write `-inf` to
    masked positions. Otherwise the K-tile early-exit branch leaves
    uninitialized memory in the output, downstream top-k reads garbage, and the
    model collapses to repeating a single token.

    Trigger: a narrow per-row `[ks, ke)` so that BLOCK_N tiles past every row's
    `ke` hit the early-exit path. We assert that `clean_logits=False` produces
    bit-identical output to `clean_logits=True` on every position (the flag is
    a perf opt, not a behavior change)."""
    torch.manual_seed(0)
    M, N, num_heads, head_dim = 256, 1024, 32, 128
    device = "cuda"

    q_fp8 = torch.randn(M, num_heads, head_dim, dtype=torch.bfloat16, device=device).to(
        torch.float8_e4m3fn
    )
    k_bf16 = torch.randn(N, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(M, num_heads, dtype=torch.float32, device=device)
    k_fp8, k_scales = _quantize_k_per_row(k_bf16)

    # Narrow per-row range: most rows' `[ks, ke)` cover ~1/16 of N, so most
    # K-tiles past `ke` hit the kernel's early-exit branch.
    ks = torch.arange(M, dtype=torch.int32, device=device) % (N // 8)
    ke = ks + N // 16
    ke = torch.minimum(ke, torch.full_like(ke, N))

    out_clean = fp8_mqa_logits_triton(
        q_fp8, (k_fp8, k_scales), weights, ks, ke, clean_logits=True
    )
    out_dirty = fp8_mqa_logits_triton(
        q_fp8, (k_fp8, k_scales), weights, ks, ke, clean_logits=False
    )

    # Every position the clean run wrote `-inf` to (i.e. outside `[ks, ke)`)
    # must also be `-inf` in the dirty run — otherwise the early-exit left
    # uninitialized memory and downstream top-k will pick from garbage.
    inf_mask = torch.isinf(out_clean) & (out_clean < 0)
    bad = inf_mask & ~(torch.isinf(out_dirty) & (out_dirty < 0))
    if bad.any():
        # Surface a useful error: how many positions, and a sample of the
        # leaked values so it's obvious they're not `-inf`.
        leaked_count = int(bad.sum().item())
        sample = out_dirty[bad].flatten()[:8].tolist()
        raise AssertionError(
            f"clean_logits=False left {leaked_count} positions un-initialised "
            f"(should be -inf). Sample leaked values: {sample}. "
            "Likely cause: the kernel's K-tile early-exit branch returns "
            "without writing -inf, so downstream top-k reads garbage."
        )


def test_fp8_mqa_logits_triton_long_gathered_k():
    """A chunk whose gathered K spans more than 65535 tiles of the smallest
    BLOCK_N, the CUDA limit on grid axes 1 and 2. The head count is one no
    other test uses, so the autotune sweep launches every BLOCK_N."""
    torch.manual_seed(0)
    M, num_heads, head_dim = 2, 24, 128
    N = 65535 * 32 + 33
    device = "cuda"

    q_fp8 = torch.randn(M, num_heads, head_dim, dtype=torch.bfloat16, device=device).to(
        torch.float8_e4m3fn
    )
    weights = torch.randn(M, num_heads, dtype=torch.float32, device=device)
    k_bytes = torch.empty(N, head_dim, dtype=torch.uint8, device=device)
    k_scales = torch.empty(N, dtype=torch.float32, device=device)
    for start in range(0, N, 2**18):
        rows = slice(start, min(start + 2**18, N))
        k_rows, k_scales[rows] = _quantize_k_per_row(
            torch.randn(
                rows.stop - rows.start, head_dim, dtype=torch.bfloat16, device=device
            )
        )
        k_bytes[rows] = k_rows.view(torch.uint8)
    k_fp8 = k_bytes.view(torch.float8_e4m3fn)
    # Row 0 spans every tile; row 1 only the last ones.
    tail = 1000
    ks = torch.tensor([0, N - tail], dtype=torch.int32, device=device)
    ke = torch.full((M,), N, dtype=torch.int32, device=device)

    out = fp8_mqa_logits_triton(
        q_fp8, (k_fp8, k_scales), weights, ks, ke, clean_logits=True, max_seqlen_k=N
    )

    assert out.shape == (M, N)
    assert torch.isinf(out[1, tail:]).all()
    positions = torch.cat(
        [
            torch.arange(64, device=device),
            torch.randint(64, N - 64, (4096,), device=device),
            torch.arange(N - 64, N, device=device),
        ]
    )
    q64, w64 = q_fp8.float().double(), weights.double()
    for m in range(M):
        pos = positions[positions >= ks[m]]
        k64 = k_bytes[pos].view(torch.float8_e4m3fn).float().double()
        k64 *= k_scales[pos, None].double()
        expected = ((q64[m] @ k64.T).relu() * w64[m, :, None]).sum(dim=0)
        torch.testing.assert_close(
            out[m, pos - ks[m]].double(), expected, atol=_ATOL, rtol=_RTOL
        )


@pytest.mark.parametrize(
    "batch_size,next_n,context_len",
    [
        (1, 1, 128),
        (1, 1, 512),
        (2, 1, 256),
        (1, 4, 512),  # speculative decoding with next_n=4
        (2, 4, 130),  # active max length is not block-aligned
    ],
)
@pytest.mark.parametrize("num_heads", [16, 32])
@pytest.mark.parametrize("clean_logits", [True, False])
def test_fp8_paged_mqa_logits_triton_matches_torch(
    batch_size, next_n, context_len, num_heads, clean_logits
):
    """`clean_logits=True` requires whole-tensor agreement; `clean_logits=False`
    is only correct on `[:, :context_len]` (downstream topk reads only that
    range), so all comparisons below are sliced accordingly."""
    torch.manual_seed(0)
    head_dim = 128
    block_size = 64
    device = "cuda"

    total_blocks = 64
    max_blocks = (context_len + block_size - 1) // block_size + 4

    kv_bf16 = torch.randn(
        total_blocks, block_size, head_dim, dtype=torch.bfloat16, device=device
    )
    kv_packed = _pack_paged_kv(kv_bf16)

    q_fp8 = torch.randn(
        batch_size,
        next_n,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)

    weights = torch.randn(
        batch_size * next_n, num_heads, dtype=torch.float32, device=device
    )

    if next_n > 1 and context_len == 130:
        context_lens = torch.tensor(
            [128, 129, 129, 130], dtype=torch.int32, device=device
        ).expand(batch_size, -1)
    else:
        context_lens = torch.full(
            (batch_size,), context_len, dtype=torch.int32, device=device
        )
    block_tables = torch.randint(
        0,
        total_blocks,
        (batch_size, max_blocks),
        dtype=torch.int32,
        device=device,
    )

    max_model_len = context_len

    out_torch = _fp8_paged_mqa_logits_ref(
        q_fp8, kv_packed, weights, context_lens, block_tables, max_model_len
    )
    out_triton = fp8_paged_mqa_logits_triton(
        q_fp8,
        kv_packed,
        weights,
        context_lens,
        block_tables,
        max_model_len,
        clean_logits=clean_logits,
    )

    if context_lens.ndim == 1:
        token_offsets = torch.arange(next_n - 1, -1, -1, device=device)
        readable_lens = context_lens[:, None] - token_offsets[None, :]
    else:
        readable_lens = context_lens
    positions = torch.arange(context_len, device=device)
    readable = positions[None, :] < readable_lens.reshape(-1, 1)
    inf_torch = (torch.isinf(out_torch) & (out_torch < 0))[:, :context_len]
    inf_triton = (torch.isinf(out_triton) & (out_triton < 0))[:, :context_len]
    assert torch.equal(inf_torch[readable], inf_triton[readable])
    finite = readable & ~inf_torch
    if finite.any():
        torch.testing.assert_close(
            out_triton[:, :context_len][finite].double(),
            out_torch[:, :context_len][finite],
            atol=_ATOL,
            rtol=_RTOL,
        )


def test_fp8_paged_mqa_logits_triton_strided_pool_no_int32_overflow():
    """Unified-KV-pool layer views have a large block stride; with enough
    blocks, int32 `block_idx * stride` exceeds 2**31 and wraps to a negative
    offset (IMA, or silent corruption when the wrapped address is mapped).
    Uses a padded pool whose stride puts the referenced blocks past the
    overflow threshold and checks parity against the torch reference."""
    torch.manual_seed(0)
    head_dim = 128
    block_size = 64
    batch_size, next_n, num_heads = 2, 1, 16
    context_len = 256
    device = "cuda"

    # Row payload is block_size * (head_dim + 4) = 8448 B; pad the pool block
    # stride to 2 MiB so int32 wraps at block_idx >= 2**31 / 2**21 = 1024.
    pool_stride = 2**21
    total_blocks = 1040
    required_bytes = (
        total_blocks * pool_stride
        + total_blocks * block_size * head_dim * 2
        + total_blocks * block_size * (head_dim + 4)
        + total_blocks * block_size * head_dim * (4 + 8)
        + 64 * 2**20  # headroom for kernel workspace and reference temporaries
    )
    free_bytes, _ = torch.accelerator.get_memory_info()
    if free_bytes < required_bytes:
        pytest.skip(f"needs {required_bytes / 2**30:.3f} GiB free device memory")

    row_elems = block_size * (head_dim + 4)
    kv_bf16 = torch.randn(
        total_blocks, block_size, head_dim, dtype=torch.bfloat16, device=device
    )
    kv_contig = _pack_paged_kv(kv_bf16)

    backing = torch.zeros(total_blocks * pool_stride, dtype=torch.uint8, device=device)
    kv_strided = backing.as_strided(
        (total_blocks, block_size, 1, head_dim + 4),
        (pool_stride, head_dim + 4, head_dim + 4, 1),
    )
    kv_strided.copy_(kv_contig.view(total_blocks, block_size, 1, head_dim + 4))
    assert kv_strided.view(total_blocks, -1).stride(0) == pool_stride
    assert kv_strided.view(total_blocks, -1).shape[1] == row_elems

    q_fp8 = torch.randn(
        batch_size, next_n, num_heads, head_dim, dtype=torch.bfloat16, device=device
    ).to(torch.float8_e4m3fn)
    weights = torch.randn(
        batch_size * next_n, num_heads, dtype=torch.float32, device=device
    )
    context_lens = torch.full(
        (batch_size,), context_len, dtype=torch.int32, device=device
    )
    # Every referenced block sits past the int32 overflow threshold.
    max_blocks = (context_len + block_size - 1) // block_size
    block_tables = torch.randint(
        1024,
        total_blocks,
        (batch_size, max_blocks),
        dtype=torch.int32,
        device=device,
    )

    out_torch = _fp8_paged_mqa_logits_ref(
        q_fp8, kv_strided, weights, context_lens, block_tables, context_len
    )
    out_triton = fp8_paged_mqa_logits_triton(
        q_fp8,
        kv_strided,
        weights,
        context_lens,
        block_tables,
        context_len,
        clean_logits=True,
    )

    inf_torch = torch.isinf(out_torch) & (out_torch < 0)
    inf_triton = torch.isinf(out_triton) & (out_triton < 0)
    assert torch.equal(inf_torch, inf_triton)
    finite = ~inf_torch
    torch.testing.assert_close(
        out_triton[finite].double(), out_torch[finite], atol=_ATOL, rtol=_RTOL
    )


@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("layout", ["packed_row", "kernel_page"])
def test_fp8_paged_mqa_logits_triton_packed_pool_pages(layout, num_heads):
    """Indexer pages inside a packed KV pool read the same as a compact cache.

    The sparse indexer passes `kv_cache.unsqueeze(-2)` of a strided view.
    `packed_row`: one 64-token page per packed block row, so the block stride
    is the row and the page starts past the row's other bytes. `kernel_page`:
    larger manager blocks are split into contiguous 64-token pages numbered
    across the whole pool, so page ids are large; here every referenced page
    starts past 2**31 bytes.
    """
    torch.manual_seed(0)
    head_dim = 128
    block_size = 64
    entry = head_dim + 4
    page_bytes = block_size * entry
    num_pages = 16
    device = "cuda"

    if layout == "packed_row":
        # 64 x (656 B MLA + 132 B indexer) per block row, 656-aligned.
        page_stride, first_page, page_offset = 50512, 0, block_size * 656
    else:
        page_stride, first_page, page_offset = page_bytes, 2**31 // page_bytes + 1, 0
    backing_bytes = (first_page + num_pages) * page_stride
    free_bytes, _ = torch.accelerator.get_memory_info()
    if free_bytes < backing_bytes + 64 * 2**20:
        pytest.skip(f"needs {backing_bytes / 2**30:.3f} GiB free device memory")

    kv_contig = _pack_paged_kv(
        torch.randn(
            num_pages, block_size, head_dim, dtype=torch.bfloat16, device=device
        )
    )
    backing = torch.zeros(backing_bytes, dtype=torch.uint8, device=device)
    kv_pool = backing.as_strided(
        (first_page + num_pages, block_size, entry),
        (page_stride, entry, 1),
        page_offset,
    ).unsqueeze(-2)
    kv_pool[first_page:].copy_(kv_contig)
    assert kv_pool.is_contiguous() == (layout == "kernel_page")

    # Ragged decode batch with (B, 1) context lengths, as the indexer passes.
    context_lens = torch.tensor([[256], [130], [193]], dtype=torch.int32, device=device)
    batch_size, max_len = context_lens.shape[0], int(context_lens.max())
    q_fp8 = torch.randn(
        batch_size, 1, num_heads, head_dim, dtype=torch.bfloat16, device=device
    ).to(torch.float8_e4m3fn)
    weights = torch.randn(batch_size, num_heads, dtype=torch.float32, device=device)
    block_tables = torch.randint(
        0,
        num_pages,
        (batch_size, cdiv(max_len, block_size)),
        dtype=torch.int32,
        device=device,
    )

    out_torch = _fp8_paged_mqa_logits_ref(
        q_fp8, kv_contig, weights, context_lens, block_tables, max_len
    )
    out_contig = fp8_paged_mqa_logits_triton(
        q_fp8, kv_contig, weights, context_lens, block_tables, max_len
    )
    out_pool = fp8_paged_mqa_logits_triton(
        q_fp8, kv_pool, weights, context_lens, block_tables + first_page, max_len
    )

    # Same kernel over the same bytes: the layout must not change the result.
    torch.testing.assert_close(out_pool, out_contig, atol=0, rtol=0)
    inf_torch = torch.isinf(out_torch) & (out_torch < 0)
    assert torch.equal(inf_torch, torch.isinf(out_pool) & (out_pool < 0))
    finite = ~inf_torch
    torch.testing.assert_close(
        out_pool[finite].double(), out_torch[finite], atol=_ATOL, rtol=_RTOL
    )


def test_sparse_indexer_fp8_byte_padding():
    """Packing storage bytes preserves FP8 values without native FP8 conversion."""
    from vllm.v1.attention.ops.common import pack_seq_triton

    values = torch.arange(24, device="cuda", dtype=torch.uint8).reshape(6, 2, 2)
    lengths = torch.tensor([1, 3, 2], device="cuda")
    packed = pack_seq_triton(values, lengths, pad_value=0)
    expected = torch.zeros((3, 3, 2, 2), device="cuda", dtype=torch.uint8)
    expected[0, :1] = values[:1]
    expected[1, :3] = values[1:4]
    expected[2, :2] = values[4:]
    torch.testing.assert_close(packed, expected, atol=0, rtol=0)
