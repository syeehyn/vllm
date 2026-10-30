# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-K tensor-core mHC prenorm GEMM for SM8x, where DeepGEMM is unavailable.

Same contract as DeepGEMM's ``tf32_hc_prenorm_gemm``: ``out[s] = x[:, Ks] @ fn[:, Ks].T``
and ``sqrsum[s] = sum(x[:, Ks]^2)`` for each of ``n_splits`` K slices, summed by the
downstream mHC kernels. The TileLang fallback it replaces runs one CTA per token on
CUDA cores and re-reads all of ``fn`` per token (~95 us at 192 tokens on A100 for
~10 MB of data). Here each program multiplies a [BLOCK_M, K/n_splits] slab with
mma.sync in 3xTF32, which matches the fp32 result (x is bf16, exact in TF32; fn is
split into TF32 hi/lo parts), so numerics are unchanged.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _hc_prenorm_kernel(
    x_ptr,
    fn_ptr,
    out_ptr,
    sq_ptr,
    num_tokens,
    stride_x,
    stride_fn,
    stride_os,
    stride_ot,
    stride_ss,
    N: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m_ok = rm < num_tokens
    n_ok = rn < N
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    x_rows = x_ptr + rm.to(tl.int64)[:, None] * stride_x
    fn_rows = fn_ptr + rn[:, None] * stride_fn
    k0 = pid_s * K_PER_SPLIT
    for k in range(k0, k0 + K_PER_SPLIT, BLOCK_K):
        xt = tl.load(x_rows + (k + rk)[None, :], mask=m_ok[:, None], other=0.0)
        xt = xt.to(tl.float32)
        ft = tl.load(fn_rows + (k + rk)[None, :], mask=n_ok[:, None], other=0.0)
        acc = tl.dot(xt, tl.trans(ft), acc, input_precision="tf32x3")
        sq += tl.sum(xt * xt, 1)
    out = out_ptr + pid_s * stride_os + rm[:, None] * stride_ot + rn[None, :]
    tl.store(out, acc, mask=m_ok[:, None] & n_ok[None, :])
    tl.store(sq_ptr + pid_s * stride_ss + rm, sq, mask=m_ok)


_BLOCK_K = 64


def supports_triton_hc_prenorm(input_size: int, n_splits: int) -> bool:
    return input_size % n_splits == 0 and (input_size // n_splits) % _BLOCK_K == 0


def triton_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    n_splits: int,
) -> None:
    num_tokens, input_size = x.shape
    n_out = fn.shape[0]
    assert x.dtype == torch.bfloat16 and fn.dtype == torch.float32
    assert out.shape == (n_splits, num_tokens, n_out)
    assert sqrsum.shape == (n_splits, num_tokens)
    assert supports_triton_hc_prenorm(input_size, n_splits)
    if num_tokens == 0:
        return
    # 16-row tiles keep >= 100 CTAs busy at decode sizes; wider ones win from ~1K
    # tokens (A100-40GB sweep at the split buckets 1/4/16).
    block_m = 16 if num_tokens <= 512 else 64
    _hc_prenorm_kernel[(triton.cdiv(num_tokens, block_m), n_splits)](
        x,
        fn,
        out,
        sqrsum,
        num_tokens,
        x.stride(0),
        fn.stride(0),
        out.stride(0),
        out.stride(1),
        sqrsum.stride(0),
        N=n_out,
        K_PER_SPLIT=input_size // n_splits,
        BLOCK_M=block_m,
        BLOCK_N=triton.next_power_of_2(max(n_out, 16)),
        BLOCK_K=_BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
