# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused FP8 sparse-MLA decode for DeepSeek-V4.1 on SM80.

Drop-in for `ampere_sparse_decode` (sparse_decode.py), enabled with
VLLM_DSV41_AMPERE_FUSED_DECODE=1. That path first gathers every slot the SWA and
compressed index lists name into a dense BF16 buffer [tokens x 640, 512] (`_gather_dequant_slots_kernel`)
and then runs split-KV attention over the buffer (`_split_kernel` + `_combine_kernel`).
On A100 at 192 sequences per microbatch the gather and the split together cost ~0.56 ms
per layer, ~25% of a decode step.

Here the split kernel reads the 584-byte records straight from both paged caches:
448 E4M3 bytes in 64-wide UE8M0 blocks, 64 BF16 RoPE dims, and the block's scales
packed after its token data. Each [BLOCK_N, 512] tile is decoded in registers
(Marlin-style E4M3 -> BF16, 4 bytes per PTX instance) and multiplied by its
power-of-two scale, which is exact in BF16, so K/V values equal the gather path's.
The RoPE tail is loaded into cols 448..511 of the same [BLOCK_N, 512] tile, so a
tile costs one QK and one PV dot (MLA: K and V are the same latent). When the grid
has a single split (large batches), the kernel normalizes with the attention sink
and stores bf16 itself; otherwise the fork's log-sum-exp combine runs unchanged.
`merged=False` / `direct=False` restore the separate-RoPE-dot and always-combine
variants for A/B.
"""

import torch

from vllm.triton_utils import LOG2E, tl, triton

BYTES_PER_TOKEN = 584
FP8_DIM = 448
ROPE_DIM = 64
TOKEN_DATA = FP8_DIM + 2 * ROPE_DIM  # 576
SCALES = 8
HEAD_DIM = 512
_BLOCK_H = 16
_BLOCK_N = 16
_TARGET_CTAS = 256
_MAX_SPLITS = 32


@triton.jit
def _e4m3x4_to_bf16(u):
    return tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32 a, b, s, m, c, z;
        mov.b32 m, 0x7B807B80;
        mov.b32 z, 0x80008000;
        prmt.b32 a, $2, 0, 0x1404;
        prmt.b32 b, $2, 0, 0x3424;
        and.b32 s, a, 0x80008000;
        and.b32 c, a, 0x7F007F00;
        shr.b32 c, c, 4;
        or.b32 a, c, s;
        and.b32 s, b, 0x80008000;
        and.b32 c, b, 0x7F007F00;
        shr.b32 c, c, 4;
        or.b32 b, c, s;
        fma.rn.bf16x2 $0, a, m, z;
        fma.rn.bf16x2 $1, b, m, z;
        }
        """,
        constraints="=r,=r,r",
        args=[u],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _record_ptrs(base_ptr, slot, CB: tl.constexpr, block_stride):
    """Byte pointers to the token data and the scale row of `slot` (masked slots -> 0)."""
    s = tl.where(slot >= 0, slot, 0).to(tl.int64)  # int64: blk * block_stride overflows int32
    blk = s // CB
    pos = s % CB
    rec = base_ptr + blk * block_stride + pos * 576
    sc = base_ptr + blk * block_stride + CB * 576 + pos * 8
    return rec, sc


@triton.jit
def _decode_nope(rec, sc, valid, BLOCK_N: tl.constexpr):
    """[BLOCK_N, 512] bf16: the 448 dequantized NoPE dims, cols >= 448 zero."""
    cols = tl.arange(0, 512)
    raw = tl.load(rec[:, None] + cols[None, :], mask=valid[:, None] & (cols[None, :] < 448), other=0)
    x = _e4m3x4_to_bf16(raw)
    groups = tl.arange(0, 8)
    # Only 7 of the 8 scale bytes are used (448 / 64); the 8th may hold anything.
    e = tl.load(sc[:, None] + groups[None, :], mask=valid[:, None] & (groups[None, :] < 7), other=127)
    scale = tl.exp2(e.to(tl.float32) - 127.0)  # UE8M0: the byte is the fp32 exponent field
    scale_full = tl.reshape(tl.broadcast_to(scale[:, :, None], (BLOCK_N, 8, 64)), (BLOCK_N, 512))
    return (x.to(tl.float32) * scale_full).to(tl.bfloat16)


@triton.jit
def _load_rope(rec, valid, WIDE: tl.constexpr):
    """BF16 RoPE tail: [BLOCK_N, 64], or placed at cols 448..511 of a zero [BLOCK_N, 512]."""
    rope_ptr = (rec + 448).to(tl.pointer_type(tl.bfloat16))
    if WIDE:
        cols = tl.arange(0, 512)
        return tl.load(rope_ptr[:, None] + (cols[None, :] - 448), mask=valid[:, None] & (cols[None, :] >= 448), other=0.0)
    else:
        rcols = tl.arange(0, 64)
        return tl.load(rope_ptr[:, None] + rcols[None, :], mask=valid[:, None], other=0.0)


@triton.jit
def _fused_split_kernel(
    q_ptr,
    swa_ptr, swa_idx_ptr, swa_stride, SWA_CB: tl.constexpr, W_SWA: tl.constexpr,
    cmp_ptr, cmp_idx_ptr, cmp_stride, CMP_CB: tl.constexpr, W_CMP: tl.constexpr,
    acc_ptr, emax_ptr, esum_ptr, sink_ptr, out_ptr,
    h_q,
    stride_q_token, stride_q_head,
    stride_swa_idx, stride_cmp_idx,
    stride_acc_token, stride_acc_head, stride_acc_split,
    stride_st_token, stride_st_head,
    stride_out_token, stride_out_head,
    sm_scale,
    per_split: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MERGED: tl.constexpr,  # RoPE in cols 448..511 of one [BLOCK_N, 512] tile: 2 dots per tile instead of 4
    DIRECT: tl.constexpr,  # single split: normalize (with the sink) and store bf16 here, no combine pass
    HAS_SINK: tl.constexpr,
):
    cur_t = tl.program_id(0)
    cur_hb = tl.program_id(1)
    cur_s = tl.program_id(2)
    TOTAL: tl.constexpr = W_SWA + W_CMP

    offs_h = cur_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q
    d512 = tl.arange(0, 512)
    d64 = tl.arange(0, 64)
    q_base = q_ptr + cur_t * stride_q_token + offs_h[:, None] * stride_q_head
    if MERGED:
        q_nope = tl.load(q_base + d512[None, :], mask=mask_h[:, None], other=0.0)
        q_rope = q_nope  # unused
    else:
        q_nope = tl.load(q_base + d512[None, :], mask=mask_h[:, None] & (d512[None, :] < 448), other=0.0)
        q_rope = tl.load(q_base + 448 + d64[None, :], mask=mask_h[:, None], other=0.0)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, 512], dtype=tl.float32)
    acc_r = tl.zeros([BLOCK_H, 64], dtype=tl.float32)

    start = cur_s * per_split
    for n0 in range(start, tl.minimum(start + per_split, TOTAL), BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        # W_SWA is a multiple of BLOCK_N, so a tile is entirely SWA or entirely compressed;
        # only the index load and pointer math branch, the record loads are shared.
        if n0 < W_SWA:
            slot = tl.load(swa_idx_ptr + cur_t * stride_swa_idx + offs_n, mask=offs_n < W_SWA, other=-1)
            rec, sc = _record_ptrs(swa_ptr, slot, SWA_CB, swa_stride)
        else:
            j = offs_n - W_SWA
            slot = tl.load(cmp_idx_ptr + cur_t * stride_cmp_idx + j, mask=j < W_CMP, other=-1)
            rec, sc = _record_ptrs(cmp_ptr, slot, CMP_CB, cmp_stride)
        valid = slot >= 0
        kn = _decode_nope(rec, sc, valid, BLOCK_N)
        if MERGED:
            kv = tl.where(d512[None, :] < 448, kn, _load_rope(rec, valid, True))
            qk = tl.dot(q_nope, tl.trans(kv))
        else:
            kr = _load_rope(rec, valid, False)
            qk = tl.dot(q_nope, tl.trans(kn)) + tl.dot(q_rope, tl.trans(kr))
        qk = qk * sm_scale
        qk = tl.where(mask_h[:, None] & valid[None, :], qk, -1.0e30)
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        rescale = tl.exp2(e_max - n_e_max)
        # Zero masked keys explicitly: in a tile with no valid key, qk - n_e_max is 0.
        p = tl.where(valid[None, :], tl.exp2(qk - n_e_max[:, None]), 0.0)
        pb = p.to(tl.bfloat16)
        if MERGED:
            acc = acc * rescale[:, None] + tl.dot(pb, kv)
        else:
            acc = acc * rescale[:, None] + tl.dot(pb, kn)
            acc_r = acc_r * rescale[:, None] + tl.dot(pb, kr)
        e_sum = e_sum * rescale + tl.sum(p, 1)
        e_max = n_e_max

    if DIRECT:
        denom = e_sum
        if HAS_SINK:
            denom += tl.exp2(tl.load(sink_ptr + offs_h, mask=mask_h, other=0.0) * 1.4426950408889634 - e_max)
        inv = tl.where(denom > 0, 1.0 / tl.where(denom > 0, denom, 1.0), 0.0)
        o = out_ptr + cur_t * stride_out_token + offs_h[:, None] * stride_out_head
        if MERGED:
            tl.store(o + d512[None, :], (acc * inv[:, None]).to(out_ptr.dtype.element_ty), mask=mask_h[:, None])
        else:
            tl.store(o + d512[None, :], (acc * inv[:, None]).to(out_ptr.dtype.element_ty), mask=mask_h[:, None] & (d512[None, :] < 448))
            tl.store(o + 448 + d64[None, :], (acc_r * inv[:, None]).to(out_ptr.dtype.element_ty), mask=mask_h[:, None])
    else:
        base = cur_t * stride_acc_token + offs_h * stride_acc_head + cur_s * stride_acc_split
        if MERGED:
            tl.store(acc_ptr + base[:, None] + d512[None, :], acc, mask=mask_h[:, None])
        else:
            tl.store(acc_ptr + base[:, None] + d512[None, :], acc, mask=mask_h[:, None] & (d512[None, :] < 448))
            tl.store(acc_ptr + base[:, None] + 448 + d64[None, :], acc_r, mask=mask_h[:, None])
        st = cur_t * stride_st_token + offs_h * stride_st_head + cur_s
        tl.store(emax_ptr + st, e_max, mask=mask_h)
        tl.store(esum_ptr + st, e_sum, mask=mask_h)


@triton.jit
def _combine_kernel(acc_ptr, emax_ptr, esum_ptr, sink_ptr, out_ptr,
                    stride_acc_token, stride_acc_head, stride_acc_split,
                    stride_st_token, stride_st_head, stride_out_token, stride_out_head,
                    LOG2E_C: tl.constexpr, HAS_SINK: tl.constexpr, n_splits: tl.constexpr, BLOCK_D: tl.constexpr):
    # Same as the fork's _combine_kernel (sparse_attn_split.py).
    cur_t = tl.program_id(0)
    cur_h = tl.program_id(1)
    splits = tl.arange(0, n_splits)
    st = cur_t * stride_st_token + cur_h * stride_st_head + splits
    emax = tl.load(emax_ptr + st)
    esum = tl.load(esum_ptr + st)
    m = tl.max(emax, 0)
    scale = tl.exp2(emax - m)
    denom = tl.sum(esum * scale, 0)
    if HAS_SINK:
        denom += tl.exp2(tl.load(sink_ptr + cur_h) * LOG2E_C - m)
    offs_d = tl.arange(0, BLOCK_D)
    part = tl.load(acc_ptr + cur_t * stride_acc_token + cur_h * stride_acc_head
                   + splits[:, None] * stride_acc_split + offs_d[None, :])
    acc = tl.sum(part * scale[:, None], 0)
    out = tl.where(denom > 0, acc / tl.where(denom > 0, denom, 1.0), 0.0)
    tl.store(out_ptr + cur_t * stride_out_token + cur_h * stride_out_head + offs_d, out.to(out_ptr.dtype.element_ty))


def _choose_splits(num_tokens: int, h_blocks: int, topk: int, block_n: int) -> int:
    n_blocks = max(1, triton.cdiv(topk, block_n))
    want = _TARGET_CTAS // max(1, num_tokens * h_blocks)
    want = min(want, _MAX_SPLITS, max(1, n_blocks // 2))
    return 1 << (want.bit_length() - 1) if want > 1 else 1


def fused_sparse_decode(
    q: torch.Tensor,  # [T, H, 512] bf16
    swa_cache: torch.Tensor,  # [NB, CB, 584] uint8
    swa_indices: torch.Tensor,  # [T, W_SWA] global slots, -1 padding
    compressed_cache: torch.Tensor | None,
    topk_indices: torch.Tensor | None,  # [T, W_CMP]
    sm_scale: float,
    attn_sink: torch.Tensor | None,  # [H] fp32
    out: torch.Tensor,  # [T, H, 512]
    num_warps: int = 4,
    num_stages: int = 2,
    block_n: int = _BLOCK_N,
    merged: bool = True,
    direct: bool = True,
) -> torch.Tensor:
    """Same contract as ampere_sparse_decode."""
    T, h_q, dim = q.shape
    assert dim == HEAD_DIM and swa_cache.shape[-1] == BYTES_PER_TOKEN
    swa_idx = swa_indices.reshape(T, -1)
    if swa_idx.stride(-1) != 1:
        swa_idx = swa_idx.contiguous()
    if swa_idx.shape[1] % block_n:
        # A key tile must not straddle the SWA/compressed boundary; pad with -1.
        swa_idx = torch.nn.functional.pad(swa_idx, (0, -swa_idx.shape[1] % block_n), value=-1)
    w_swa = swa_idx.shape[1]
    if compressed_cache is not None and topk_indices is not None:
        cmp_idx = topk_indices.reshape(T, -1)
        if cmp_idx.stride(-1) != 1:
            cmp_idx = cmp_idx.contiguous()
        cmp = compressed_cache
        assert cmp.shape[-1] == BYTES_PER_TOKEN
    else:
        cmp_idx = swa_idx[:, :0]
        cmp = swa_cache
    w_cmp = cmp_idx.shape[1]
    total = w_swa + w_cmp

    block_h = min(_BLOCK_H, h_q)
    h_blocks = triton.cdiv(h_q, block_h)
    n_splits = _choose_splits(T, h_blocks, total, block_n)
    per_split = triton.cdiv(triton.cdiv(total, block_n), n_splits) * block_n
    direct = direct and n_splits == 1
    if direct:
        acc = emax = esum = out  # unused
        acc_strides, st_strides = (0, 0, 0), (0, 0)
    else:
        acc = torch.empty((T, h_q, n_splits, dim), dtype=torch.float32, device=q.device)
        emax = torch.empty((T, h_q, n_splits), dtype=torch.float32, device=q.device)
        esum = torch.empty_like(emax)
        acc_strides, st_strides = acc.stride()[:3], emax.stride()[:2]
    _fused_split_kernel[(T, h_blocks, n_splits)](
        q,
        swa_cache, swa_idx, swa_cache.stride(0), swa_cache.shape[1], w_swa,
        cmp, cmp_idx, cmp.stride(0), cmp.shape[1], w_cmp,
        acc, emax, esum, attn_sink, out,
        h_q,
        q.stride(0), q.stride(1),
        swa_idx.stride(0), max(1, cmp_idx.stride(0)),
        *acc_strides,
        *st_strides,
        out.stride(0), out.stride(1),
        sm_scale * LOG2E,
        per_split=per_split,
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        MERGED=merged,
        DIRECT=direct,
        HAS_SINK=attn_sink is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if direct:
        return out
    _combine_kernel[(T, h_q)](
        acc, emax, esum, attn_sink, out,
        acc.stride(0), acc.stride(1), acc.stride(2),
        emax.stride(0), emax.stride(1),
        out.stride(0), out.stride(1),
        LOG2E_C=LOG2E,
        HAS_SINK=attn_sink is not None,
        n_splits=n_splits,
        BLOCK_D=dim,
        num_warps=4,
    )
    return out
