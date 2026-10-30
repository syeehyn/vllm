# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM8x split-K prenorm matches fp32 and the TileLang kernel it replaces."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only kernels"
)


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 192, 513, 4096])
@pytest.mark.parametrize("n_splits", [1, 4, 16])
def test_triton_hc_prenorm_matches_fp32(num_tokens, n_splits):
    from vllm.model_executor.kernels.mhc.triton_prenorm import (
        triton_hc_prenorm_gemm,
    )

    torch.manual_seed(0)
    hidden, hc_mult, n_out = 5120, 4, 24
    x = torch.randn(num_tokens, hc_mult * hidden, device="cuda").to(torch.bfloat16)
    fn = torch.randn(n_out, hc_mult * hidden, device="cuda") * 0.02
    out = torch.empty(n_splits, num_tokens, n_out, device="cuda")
    sqrsum = torch.empty(n_splits, num_tokens, device="cuda")
    triton_hc_prenorm_gemm(x, fn, out, sqrsum, n_splits)

    want = x.double() @ fn.double().T
    torch.testing.assert_close(out.sum(0).double(), want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        sqrsum.sum(0).double(), (x.double() ** 2).sum(-1), rtol=1e-6, atol=1e-3
    )


def test_triton_hc_prenorm_matches_tilelang():
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _HC_PRENORM_GEMM_TILELANG_KERNEL,
    )
    from vllm.model_executor.kernels.mhc.triton_prenorm import (
        triton_hc_prenorm_gemm,
    )

    torch.manual_seed(1)
    hidden, hc_mult, n_out, num_tokens = 5120, 4, 24, 192
    x = torch.randn(num_tokens, hc_mult * hidden, device="cuda").to(torch.bfloat16)
    fn = torch.randn(n_out, hc_mult * hidden, device="cuda") * 0.02
    ref_out = torch.empty(1, num_tokens, n_out, device="cuda")
    ref_sq = torch.empty(1, num_tokens, device="cuda")
    _HC_PRENORM_GEMM_TILELANG_KERNEL(x, fn, ref_out, ref_sq, hidden, hc_mult)
    out = torch.empty(16, num_tokens, n_out, device="cuda")
    sq = torch.empty(16, num_tokens, device="cuda")
    triton_hc_prenorm_gemm(x, fn, out, sq, 16)
    torch.testing.assert_close(out.sum(0), ref_out[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sq.sum(0), ref_sq[0], rtol=1e-5, atol=1e-3)
