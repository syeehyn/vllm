# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark SM80 sparse-MLA verify-decode variants for DeepSeek-V4.1.

The DSpark verify pass feeds 5-6 query rows per request through
`ampere_sparse_attn_split`. This compares the shipped split kernel with a
variant that:

* loads the MLA latent once per key block instead of twice -- `k` and `v` are
  the same 512-wide buffer today, read once as `[D, N]` and once as `[N, D]`;
* sizes the head block to the actual head count instead of masking half of a
  fixed 16-wide tile (h_q is 8 at TP8).

    .venv/bin/python benchmarks/kernels/bench_dsv41_ampere_verify.py
    .venv/bin/python benchmarks/kernels/bench_dsv41_ampere_verify.py --capture
"""

from __future__ import annotations

import argparse

import torch
import triton

from vllm.models.deepseek_v41.ampere.sparse_attn_split import (
    _BLOCK_N,
    _choose_splits,
    _combine_kernel,
    ampere_sparse_attn_split,
)
from vllm.triton_utils import LOG2E, tl, triton

_DIM = 512


@triton.jit
def _split_kernel_v2(
    q_ptr,
    kv_ptr,
    indices_ptr,
    acc_ptr,
    emax_ptr,
    esum_ptr,
    seq_kv,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_kv_token,
    stride_idx_token,
    stride_acc_token,
    stride_acc_head,
    stride_acc_split,
    stride_st_token,
    stride_st_head,
    sm_scale,
    index_topk: tl.constexpr,
    per_split: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_t = tl.program_id(0)
    cur_hb = tl.program_id(1)
    cur_s = tl.program_id(2)

    offs_h = cur_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr
        + cur_t * stride_q_token
        + offs_h[:, None] * stride_q_head
        + offs_d[None, :],
        mask=mask_h[:, None],
        other=0.0,
    )

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    start = cur_s * per_split
    for n0 in range(start, tl.minimum(start + per_split, index_topk), BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        idx = tl.load(
            indices_ptr + cur_t * stride_idx_token + offs_n,
            mask=offs_n < index_topk,
            other=-1,
        )
        mask_kv = (idx >= 0) & (idx < seq_kv)
        kv_tile = tl.load(
            kv_ptr + idx[:, None] * stride_kv_token + offs_d[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(kv_tile)) * sm_scale
        qk = tl.where(mask_h[:, None] & mask_kv[None, :], qk, -1.0e30)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        rescale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc = acc * rescale[:, None] + tl.dot(p.to(kv_tile.dtype), kv_tile)
        e_sum = e_sum * rescale + tl.sum(p, 1)
        e_max = n_e_max

    base = (
        cur_t * stride_acc_token + offs_h * stride_acc_head + cur_s * stride_acc_split
    )
    tl.store(acc_ptr + base[:, None] + offs_d[None, :], acc, mask=mask_h[:, None])
    st = cur_t * stride_st_token + offs_h * stride_st_head + cur_s
    tl.store(emax_ptr + st, e_max, mask=mask_h)
    tl.store(esum_ptr + st, e_sum, mask=mask_h)


def sparse_verify_v2(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    block_h: int | None = None,
) -> torch.Tensor:
    num_tokens, h_q, dim = q.shape
    topk = indices.shape[-1]
    bh = block_h or min(16, h_q)
    h_blocks = triton.cdiv(h_q, bh)
    n_splits = _choose_splits(num_tokens, h_blocks, topk)
    per_split = triton.cdiv(triton.cdiv(topk, _BLOCK_N), n_splits) * _BLOCK_N

    acc = torch.empty(
        (num_tokens, h_q, n_splits, dim), dtype=torch.float32, device=q.device
    )
    emax = torch.empty(
        (num_tokens, h_q, n_splits), dtype=torch.float32, device=q.device
    )
    esum = torch.empty_like(emax)
    idx2d = indices.view(num_tokens, -1)

    _split_kernel_v2[(num_tokens, h_blocks, n_splits)](
        q,
        kv,
        idx2d,
        acc,
        emax,
        esum,
        kv.shape[0],
        h_q,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        idx2d.stride(0),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        emax.stride(0),
        emax.stride(1),
        sm_scale * LOG2E,
        index_topk=topk,
        per_split=per_split,
        BLOCK_H=bh,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=dim,
        num_warps=4,
        num_stages=2,
    )
    out = torch.empty((num_tokens, h_q, dim), dtype=q.dtype, device=q.device)
    _combine_kernel[(num_tokens, h_q)](
        acc,
        emax,
        esum,
        attn_sink,
        out,
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        emax.stride(0),
        emax.stride(1),
        out.stride(0),
        out.stride(1),
        LOG2E_C=LOG2E,
        HAS_SINK=attn_sink is not None,
        n_splits=n_splits,
        BLOCK_D=dim,
        num_warps=4,
    )
    return out


def make_inputs(
    num_tokens: int, h_q: int, topk: int, valid: int, seq_kv: int, device: str
):
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(num_tokens, h_q, _DIM, generator=g, dtype=torch.float32).to(
        device=device, dtype=torch.bfloat16
    )
    kv = torch.randn(
        seq_kv, 1, _DIM, generator=g, dtype=torch.float32, device="cpu"
    ).to(device=device, dtype=torch.bfloat16)
    idx = torch.randint(
        0, seq_kv, (num_tokens, 1, topk), generator=g, dtype=torch.int32
    )
    idx[:, :, valid:] = -1
    idx = idx.to(device)
    sink = torch.randn(h_q, generator=g, dtype=torch.float32).to(device)
    return q, kv, idx, sink.to(torch.bfloat16)


def bench_decode(args) -> None:
    from vllm.models.deepseek_v41.ampere.sparse_decode import ampere_sparse_decode

    device = "cuda"
    block_size = 128
    num_blocks = 640
    seq_slots = num_blocks * block_size
    g = torch.Generator(device="cpu").manual_seed(0)
    cache = torch.randint(
        0, 256, (num_blocks, block_size, 584), generator=g, dtype=torch.uint8
    ).to(device)
    sink = torch.randn(64, generator=g, dtype=torch.float32).to(device)

    print(
        f"{'T':>3} {'swa':>4} {'topk':>5} {'decode us':>10} {'attn us':>9} "
        f"{'gather %':>9}"
    )
    for num_tokens in (1, 5, 6, 16):
        h_q = 64
        q = torch.randn(
            num_tokens, h_q, 512, generator=g, dtype=torch.float32
        ).to(device=device, dtype=torch.bfloat16)
        swa_idx = torch.randint(
            0, seq_slots, (num_tokens, 1, 128), generator=g, dtype=torch.int32
        ).to(device)
        topk_idx = torch.randint(
            0, seq_slots, (num_tokens, 1, 512), generator=g, dtype=torch.int32
        ).to(device)
        out = torch.empty(
            (num_tokens, h_q, 512), dtype=torch.bfloat16, device=device
        )

        def run():
            ampere_sparse_decode(
                q, cache, swa_idx, cache, topk_idx, 1.0 / 512**0.5, sink, out
            )

        # Gather+attention vs attention alone on the already-gathered latent.
        kv = torch.empty(
            (num_tokens * 640, 1, 512), dtype=torch.bfloat16, device=device
        )
        idx = torch.arange(num_tokens * 640, device=device, dtype=torch.int32).view(
            num_tokens, 1, 640
        )

        def attn_only():
            ampere_sparse_attn_split(q, kv, idx, 1.0 / 512**0.5, sink, out=out)

        if args.capture:
            graphs = {}
            for name, fn in (("run", run), ("attn", attn_only)):
                fn()
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    fn()
                graphs[name] = gr
            t_full = timeit(graphs["run"].replay, args.iters)
            t_attn = timeit(graphs["attn"].replay, args.iters)
        else:
            t_full = timeit(run, args.iters)
            t_attn = timeit(attn_only, args.iters)

        print(
            f"{num_tokens:>3} {128:>4} {640:>5} {t_full:>10.2f} {t_attn:>9.2f} "
            f"{100.0 * (t_full - t_attn) / max(t_full, 1e-9):>8.0f}%"
        )


def timeit(fn, iters: int) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true", help="time under CUDA graphs")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument(
        "--decode",
        action="store_true",
        help="time the full gather+dequant+attention decode (584-byte FP8 cache)",
    )
    args = ap.parse_args()

    if args.decode:
        bench_decode(args)
        return

    device = "cuda"
    seq_kv = 70_000
    sm_scale = 1.0 / (_DIM**0.5)
    cases = [
        (1, 8, 640, 640),
        (6, 8, 640, 640),
        (6, 8, 192, 133),
        (5, 8, 192, 133),
        (16, 8, 640, 640),
        (6, 64, 640, 640),
    ]

    header = (
        f"{'T':>3} {'h_q':>4} {'topk':>5} {'valid':>6} "
        f"{'base us':>9} {'v2 us':>9} {'v2 bh8 us':>10} {'base/v2':>8} {'maxdiff':>9}"
    )
    print(header)
    for num_tokens, h_q, topk, valid in cases:
        q, kv, idx, sink = make_inputs(num_tokens, h_q, topk, valid, seq_kv, device)

        base = ampere_sparse_attn_split(q, kv, idx, sm_scale, sink)
        v2 = sparse_verify_v2(q, kv, idx, sm_scale, sink)
        v2_bh = sparse_verify_v2(q, kv, idx, sm_scale, sink, block_h=8)
        diff = (base.float() - v2.float()).abs().max().item()

        if args.capture:
            graphs = {}
            for name, fn in (("base", lambda: ampere_sparse_attn_split(
                q, kv, idx, sm_scale, sink)),
                ("v2", lambda: sparse_verify_v2(q, kv, idx, sm_scale, sink)),
                ("v2bh", lambda: sparse_verify_v2(
                    q, kv, idx, sm_scale, sink, block_h=8))):
                fn()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn()
                graphs[name] = g
            t_base = timeit(graphs["base"].replay, args.iters)
            t_v2 = timeit(graphs["v2"].replay, args.iters)
            t_v2b = timeit(graphs["v2bh"].replay, args.iters)
        else:
            t_base = timeit(
                lambda: ampere_sparse_attn_split(q, kv, idx, sm_scale, sink),
                args.iters,
            )
            t_v2 = timeit(
                lambda: sparse_verify_v2(q, kv, idx, sm_scale, sink), args.iters
            )
            t_v2b = timeit(
                lambda: sparse_verify_v2(q, kv, idx, sm_scale, sink, block_h=8),
                args.iters,
            )

        print(
            f"{num_tokens:>3} {h_q:>4} {topk:>5} {valid:>6} "
            f"{t_base:>9.2f} {t_v2:>9.2f} {t_v2b:>10.2f} "
            f"{t_base / t_v2:>8.2f} {diff:>9.2e}"
        )


if __name__ == "__main__":
    main()
