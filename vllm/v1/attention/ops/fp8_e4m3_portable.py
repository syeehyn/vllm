# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable OCP E4M3 <-> float conversions for Triton on pre-SM89 GPUs.

Triton's CUDA backend refuses the ``fp8e4nv`` *type* on sm_80/sm_86 -- not just
conversions to and from it, but any use at all, including ``bitcast=True``::

    ValueError: type fp8e4nv not supported in this architecture.
                The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')

That blocks every DeepSeek-V4/V4.1 FP8 KV-cache kernel on Ampere, since the
cache records are OCP E4M3 bytes.

``fp8e4b15`` *is* accepted.  Two implementations are possible:

* a pure-Triton bias shift (reinterpret as ``e4b15``, scale by ``2**-8``), which
  is bit-exact for decoding but rounds *ties away from zero* when encoding;
* the packaged ``fp8e4nv`` bitcode (``vllm.v1.attention.ops.fp8e4nv``), hand
  written PTX that re-targets to any arch and performs saturating RNE.

This module uses the bitcode, so encoding is bit-identical to native
``Tensor.to(torch.float8_e4m3fn)``.  Kernels that inline it must pass
``extern_libs=FP8E4NV_EXTERN_LIBS`` at launch on machines without native FP8
support (``has_native_fp8e4nv()`` is False).

On sm_89 and newer the native ``fp8e4nv`` conversions are used instead, so these
helpers are safe to call unconditionally.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.fp8e4nv import (
    FP8E4NV_EXTERN_LIBS,
    convert_from_fp8e4m3,
    convert_to_fp8e4m3,
)


def has_native_fp8e4nv() -> bool:
    """True when Triton can lower ``fp8e4nv`` on this device (sm_89+)."""
    if not current_platform.is_cuda():
        return True
    cap = current_platform.get_device_capability()
    if cap is None:
        return True
    return (cap.major, cap.minor) >= (8, 9)


@triton.jit
def e4m3_bytes_to_float(x_u8, NATIVE_FP8: tl.constexpr):
    """Decode OCP E4M3 bytes to fp32. ``x_u8`` is a uint8 tile.

    Bit-exact for every finite E4M3 encoding on both paths.
    """
    if NATIVE_FP8:
        return x_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    # Saturating RNE through the packaged E4M3 bitcode; exact for every byte.
    return convert_from_fp8e4m3(x_u8, tl.float32)


@triton.jit
def float_to_e4m3_bytes(x, NATIVE_FP8: tl.constexpr):
    """Encode a float tile to OCP E4M3 bytes (uint8).

    The non-native path rounds exact ties away from zero rather than to even;
    see the module docstring.
    """
    if NATIVE_FP8:
        return x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    # Saturating RNE through the packaged E4M3 bitcode; matches native exactly.
    return convert_to_fp8e4m3(x)


# ---------------------------------------------------------------------------
# Thin tensor-level wrappers, mainly for tests and one-off conversions.
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["n"])
def _decode_kernel(src, dst, n, NATIVE_FP8: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    b = tl.load(src + off, mask=mask, other=0)
    tl.store(dst + off, e4m3_bytes_to_float(b, NATIVE_FP8), mask=mask)


@triton.jit(do_not_specialize=["n"])
def _encode_kernel(src, dst, n, NATIVE_FP8: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    x = tl.load(src + off, mask=mask, other=0.0)
    tl.store(dst + off, float_to_e4m3_bytes(x, NATIVE_FP8), mask=mask)


def e4m3_to_float(x: torch.Tensor) -> torch.Tensor:
    """Decode a uint8 / float8_e4m3fn tensor to fp32."""
    if x.dtype == torch.float8_e4m3fn:
        x = x.view(torch.uint8)
    assert x.dtype == torch.uint8, f"expected uint8 bytes, got {x.dtype}"
    x = x.contiguous()
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    n = x.numel()
    BLOCK = 1024
    native = has_native_fp8e4nv()
    _decode_kernel[(triton.cdiv(n, BLOCK),)](
        x,
        out,
        n,
        NATIVE_FP8=native,
        BLOCK=BLOCK,
        **({} if native else {"extern_libs": FP8E4NV_EXTERN_LIBS}),
    )
    return out


def float_to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Encode a float tensor to OCP E4M3, returned as float8_e4m3fn."""
    x = x.contiguous().to(torch.float32)
    out = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    n = x.numel()
    BLOCK = 1024
    native = has_native_fp8e4nv()
    _encode_kernel[(triton.cdiv(n, BLOCK),)](
        x,
        out,
        n,
        NATIVE_FP8=native,
        BLOCK=BLOCK,
        **({} if native else {"extern_libs": FP8E4NV_EXTERN_LIBS}),
    )
    return out.view(torch.float8_e4m3fn)
