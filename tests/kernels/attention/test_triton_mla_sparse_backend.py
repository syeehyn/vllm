# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TRITON_MLA_SPARSE: sparse MLA for CUDA devices that no compiled sparse MLA
kernel supports (SM8x).

The CUDA tests use the Kimi-K3.1 shapes at TP8: 12 of 96 MLA heads, top-k
2048, and 32 index heads of 128 dims.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.mla import triton_mla_sparse
from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
from vllm.v1.attention.backends.mla.triton_mla_sparse import (
    TritonMLASparseBackend,
    TritonMLASparseImpl,
)
from vllm.v1.attention.backends.mla.xpu_mla_sparse import XPUMLASparseMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.ops.mqa_logits_triton import (
    _fp8_mqa_logits_kernel,
    _fp8_paged_mqa_logits_kernel,
)
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    _BLOCK_DV,
    _DIM_QK,
    KV_SPLITS_CANDIDATES,
    _sparse_mla_kernel_final,
    _sparse_mla_kernel_split,
)
from vllm.v1.attention.selector import AttentionSelectorConfig
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import allocate_kv_cache, customize_attention_spec

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="TRITON_MLA_SPARSE is a CUDA backend"
)

_NUM_HEADS = 12
_TOPK = 2048
_INDEX_HEADS = 32
_INDEX_HEAD_DIM = 128
_SCALE = 192**-0.5
_BLOCK_SIZE = 64


def _packed_kv_cache(device: str) -> tuple[MLAAttentionSpec, torch.Tensor]:
    """The sparse MLA layer's spec and `[blocks, tokens, 576]` cache view
    when each block also holds an indexer page, as in the K3.1 block row."""
    config = VllmConfig(cache_config=CacheConfig(block_size=_BLOCK_SIZE))
    config.cache_config.kv_cache_layout = KVCacheLayout.BLHNC.name
    spec = customize_attention_spec(
        TritonMLASparseBackend,
        MLAAttentionSpec(
            block_size=_BLOCK_SIZE,
            num_kv_heads=1,
            head_size=_DIM_QK,
            dtype=torch.bfloat16,
        ),
    )
    specs = {
        "mla": spec,
        "indexer": MLAAttentionSpec(
            block_size=_BLOCK_SIZE,
            num_kv_heads=1,
            head_size=_INDEX_HEAD_DIM + 4,
            dtype=torch.uint8,
            cache_role=SparseCacheRole.INDEXER,
        ),
    }
    group = KVCacheGroupSpec(
        list(specs),
        UniformTypeKVCacheSpecs(block_size=_BLOCK_SIZE, kv_cache_specs=specs),
    )
    cache = get_kv_cache_config_from_groups(config, [group], 2**23)
    views = allocate_kv_cache(cache, torch.device(device), KVCacheLayout.BLHNC)
    return spec, views["mla"].squeeze(1)


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("major,expected", [(7, False), (8, True), (12, True)])
def test_requires_sm80(major: int, expected: bool):
    assert (
        TritonMLASparseBackend.supports_compute_capability(DeviceCapability(major, 0))
        is expected
    )


@pytest.mark.skip_global_cleanup
def test_registered_backend_serves_every_token_through_topk():
    backend = AttentionBackendEnum.TRITON_MLA_SPARSE.get_class()
    assert backend is TritonMLASparseBackend
    assert backend.get_name() == AttentionBackendEnum.TRITON_MLA_SPARSE.name
    assert backend.is_mla() and backend.is_sparse()
    assert backend.get_impl_cls().supports_dense_mha_prefill is False


@pytest.mark.skip_global_cleanup
def test_packed_blocks_stay_whole_rows_apart():
    """The impl reads the cache as flat 576-wide rows, so the indexer page
    between two blocks must not leave them a fraction of a row apart."""
    spec, kv_cache = _packed_kv_cache("cpu")
    assert spec.block_stride_alignment == spec.state_content_size_bytes == 1152
    assert kv_cache.stride(0) > _BLOCK_SIZE * _DIM_QK

    rows, block_stride_rows = flat_kv_row_view(kv_cache, _BLOCK_SIZE)
    kv_cache[1, 0].fill_(3)
    torch.testing.assert_close(rows[block_stride_rows], kv_cache[1, 0])


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    "model_dtype,cache_dtype,expected_kv_dtype",
    [
        (torch.float16, "auto", torch.float16),
        (torch.bfloat16, "auto", torch.bfloat16),
        (torch.float16, "bfloat16", torch.bfloat16),
        (torch.bfloat16, "float16", torch.float16),
    ],
)
def test_warmup_uses_serving_dtypes_and_indexer_shape(
    monkeypatch, model_dtype, cache_dtype, expected_kv_dtype
):
    """Autotune results are keyed on dtypes, head counts and the indexer page
    size, so a warmup that differs from serving re-tunes on the first request."""
    attention_dtypes = set()
    logits_warmups = {}
    monkeypatch.setattr(
        triton_mla_sparse,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=model_dtype)),
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "triton_mla_sparse_attention",
        lambda q, kv, *args, **kwargs: attention_dtypes.add((q.dtype, kv.dtype)),
    )
    for name in ("warmup_fp8_mqa_logits_triton", "warmup_fp8_paged_mqa_logits_triton"):
        monkeypatch.setattr(
            triton_mla_sparse,
            name,
            lambda _name=name, **kwargs: logits_warmups.update({_name: kwargs}),
        )
    topk_indices_buffer = torch.empty(1, 16, dtype=torch.int32)
    impl = SimpleNamespace(
        topk_indices_buffer=topk_indices_buffer,
        num_heads=16,
        softmax_scale=1.0,
        _sm_count=1,
        kv_cache_dtype=cache_dtype,
    )
    # An indexer that pads its 24 query heads to 32 before scoring.
    indexer = SimpleNamespace(
        index_num_heads=24, padded_index_num_heads=32, index_head_dim=128
    )

    TritonMLASparseImpl._warmup_autotune(impl, indexer)

    assert attention_dtypes == {(model_dtype, expected_kv_dtype)}
    shape = dict(num_heads=32, head_dim=128, device=topk_indices_buffer.device)
    assert logits_warmups == {
        "warmup_fp8_mqa_logits_triton": shape,
        "warmup_fp8_paged_mqa_logits_triton": shape | dict(block_size=64),
    }


@requires_cuda
@pytest.mark.parametrize("minor", [0, 6, 9])
def test_sm8x_selects_triton_sparse_for_bf16_kv(minor: int):
    from vllm.platforms.cuda import CudaPlatform

    valid, _ = CudaPlatform.get_valid_backends(
        device_capability=DeviceCapability(8, minor),
        attn_selector_config=AttentionSelectorConfig(
            head_size=_DIM_QK,
            dtype=torch.bfloat16,
            kv_cache_dtype="bfloat16",
            block_size=None,
            use_mla=True,
            use_sparse=True,
        ),
        num_heads=_NUM_HEADS,
    )
    assert [candidate.backend for candidate in valid] == [
        AttentionBackendEnum.TRITON_MLA_SPARSE
    ]


@requires_cuda
@pytest.mark.parametrize("major", [9, 10, 12])
def test_sm90_and_newer_priorities_are_unchanged(major: int):
    from vllm.platforms.cuda import _get_backend_priorities

    assert AttentionBackendEnum.TRITON_MLA_SPARSE not in _get_backend_priorities(
        use_mla=True, device_capability=DeviceCapability(major, 0), head_size=_DIM_QK
    )


def _k31_impl() -> TritonMLASparseImpl:
    # Stands in for KimiK3DSAIndexer: the impl reads only these attributes.
    indexer = SimpleNamespace(
        topk_indices_buffer=torch.empty(64, _TOPK, dtype=torch.int32, device="cuda"),
        padded_index_num_heads=_INDEX_HEADS,
        index_head_dim=_INDEX_HEAD_DIM,
    )
    return TritonMLASparseImpl(
        num_heads=_NUM_HEADS,
        head_size=_DIM_QK,
        scale=_SCALE,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="bfloat16",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
        kv_lora_rank=_BLOCK_DV,
        indexer=indexer,
    )


@pytest.fixture(scope="module")
def k31_impl() -> TritonMLASparseImpl:
    return _k31_impl()


_AUTOTUNED_KERNELS = (
    _sparse_mla_kernel_final,
    _sparse_mla_kernel_split,
    _fp8_mqa_logits_kernel,
    _fp8_paged_mqa_logits_kernel,
)


def _tuned_keys() -> list[set[tuple]]:
    """Per kernel, the autotune keys tuned so far, argument dtypes included."""
    return [set(kernel.cache) for kernel in _AUTOTUNED_KERNELS]


@requires_cuda
def test_k31_init_tunes_the_serving_kernels_once(k31_impl):
    """Building a layer tunes the real kernels for the shapes serving runs
    them with; the model's other DSA layers then reuse the result."""
    final, split, prefill_logits, paged_logits = (
        {key[: len(kernel.keys)] for key in kernel.cache}
        for kernel in _AUTOTUNED_KERNELS
    )
    assert (_TOPK, _NUM_HEADS) in final
    assert {(_TOPK, n, _NUM_HEADS) for n in KV_SPLITS_CANDIDATES if n > 1} <= split
    assert (_INDEX_HEADS, _INDEX_HEAD_DIM) in prefill_logits
    assert (_INDEX_HEADS, _INDEX_HEAD_DIM, _BLOCK_SIZE) in paged_logits

    tuned = _tuned_keys()
    _k31_impl()
    assert _tuned_keys() == tuned


@requires_cuda
@pytest.mark.parametrize("query_lens", [(1,), (32, 32)])
def test_k31_topk_attention_on_packed_cache_matches_reference(k31_impl, query_lens):
    """A single decode token takes the split-KV path and a batch the
    single-pass one; both must address the rows of a cache whose blocks are
    more than `block_size` rows apart, without tuning anything new."""
    torch.manual_seed(0)
    _, kv_cache = _packed_kv_cache("cuda")
    kv_cache.copy_(torch.randn(kv_cache.shape, dtype=kv_cache.dtype, device="cuda"))

    # The second request's context leaves most of its top-k slots empty.
    context_lens = (3000, 700)[: len(query_lens)]
    num_tokens = sum(query_lens)
    req_ids = torch.repeat_interleave(
        torch.arange(len(query_lens), dtype=torch.int32), torch.tensor(query_lens)
    ).cuda()
    block_table = torch.zeros(len(query_lens), 48, dtype=torch.int32, device="cuda")
    free_blocks = iter((torch.randperm(kv_cache.shape[0] - 1) + 1).tolist())
    topk_indices = k31_impl.topk_indices_buffer[:num_tokens].fill_(-1)
    for req, context_len in enumerate(context_lens):
        for block in range(-(-context_len // _BLOCK_SIZE)):
            block_table[req, block] = next(free_blocks)
    for token, req in enumerate(req_ids.tolist()):
        positions = torch.randperm(context_lens[req], device="cuda")[:_TOPK]
        topk_indices[token, : positions.numel()] = positions

    # forward_mqa reads only the fields that are not placeholders here.
    placeholder = torch.empty(0, dtype=torch.int32, device="cuda")
    metadata = XPUMLASparseMetadata(
        num_reqs=len(query_lens),
        max_query_len=max(query_lens),
        max_seq_len=max(context_lens),
        num_actual_tokens=num_tokens,
        query_start_loc=placeholder,
        slot_mapping=placeholder,
        block_table=block_table,
        req_id_per_token=req_ids,
        block_size=_BLOCK_SIZE,
        topk_tokens=_TOPK,
        num_decodes=len(query_lens),
        num_decode_tokens=num_tokens,
    )
    q = torch.randn(
        num_tokens, _NUM_HEADS, _DIM_QK, dtype=torch.bfloat16, device="cuda"
    )

    tuned = _tuned_keys()
    out, _ = k31_impl.forward_mqa(q, kv_cache, metadata, None)
    assert _tuned_keys() == tuned

    assert out.shape == (num_tokens, _NUM_HEADS, _BLOCK_DV)
    for token, req in enumerate(req_ids.tolist()):
        positions = topk_indices[token]
        positions = positions[positions >= 0].long()
        blocks = block_table[req, positions // _BLOCK_SIZE].long()
        keys = kv_cache[blocks, positions % _BLOCK_SIZE].double()
        probs = (q[token].double() @ keys.T * _SCALE).softmax(dim=-1)
        # BF16 rounds the probabilities before the PV product, and the output.
        torch.testing.assert_close(
            out[token].double(), probs @ keys[:, :_BLOCK_DV], atol=2e-2, rtol=0
        )
