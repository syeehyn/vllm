# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-Triton sparse MLA backend for CUDA devices without a compiled sparse
MLA kernel (SM8x)."""

from typing import ClassVar

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import get_kv_cache_torch_dtype
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.sparse_utils import align_blocks_to_rows
from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
    XPUMLASparseImpl,
    XPUMLASparseMetadata,
    XPUMLASparseMetadataBuilder,
)
from vllm.v1.attention.ops.mqa_logits_triton import (
    warmup_fp8_mqa_logits_triton,
    warmup_fp8_paged_mqa_logits_triton,
)
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    _DIM_QK,
    KV_SPLITS_CANDIDATES,
    triton_mla_sparse_attention,
)
from vllm.v1.kv_cache_interface import AttentionSpec

# Fallback indexer shape for indexers that expose neither attribute name.
# The autotune key includes (num_heads, head_dim), so a wrong warmup shape
# forces a re-tune on the first real request.
_INDEXER_NUM_HEADS = 64
_INDEXER_HEAD_DIM = 128
# The CUDA indexer backends page their key cache by 64 tokens whatever the
# KV cache manager's block size is.
_INDEXER_KERNEL_BLOCK_SIZE = 64


class TritonMLASparseMetadataBuilder(XPUMLASparseMetadataBuilder):
    # XPU base keeps NEVER (not validated under cudagraph); this subclass
    # claims UNIFORM_BATCH for the CUDA/Triton path.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class TritonMLASparseImpl(XPUMLASparseImpl):
    """Triton sparse-MLA impl with split-KV decode, which spreads a
    single-query decode over the device instead of one program per head
    group."""

    # Every token goes through the top-k MQA path (see XPUMLASparseMetadata).
    supports_dense_mha_prefill = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sm_count: int | None = None
        if self.topk_indices_buffer is not None:
            self._sm_count = num_compute_units(self.topk_indices_buffer.device.index)
        self._warmup_autotune(self._indexer)

    def _warmup_autotune(self, indexer) -> None:
        """Prime `@triton.autotune` caches at init so the first request
        doesn't pay the inline config-sweep cost."""
        if self.topk_indices_buffer is None:
            return
        device = self.topk_indices_buffer.device
        topk = self.topk_indices_buffer.shape[-1]
        cfg = get_current_vllm_config_or_none()
        model_dtype = cfg.model_config.dtype if cfg is not None else torch.bfloat16
        kv_dtype = get_kv_cache_torch_dtype(self.kv_cache_dtype, model_dtype)
        q = torch.empty(1, self.num_heads, _DIM_QK, dtype=model_dtype, device=device)
        kv = torch.empty(64, 1, _DIM_QK, dtype=kv_dtype, device=device)
        indices = torch.zeros(1, 1, topk, dtype=torch.int32, device=device)
        for splits in KV_SPLITS_CANDIDATES:
            triton_mla_sparse_attention(
                q,
                kv,
                indices,
                sm_scale=self.softmax_scale,
                num_kv_splits=splits,
                sm_count=self._sm_count,
            )
        # Layers that reuse another layer's top-k have no indexer to warm.
        if indexer is None:
            return
        # Indexers that pad the query heads before scoring expose the padded
        # count, which is the one the logits kernels are specialized on.
        indexer_num_heads = getattr(
            indexer,
            "padded_index_num_heads",
            getattr(indexer, "n_head", _INDEXER_NUM_HEADS),
        )
        indexer_head_dim = getattr(
            indexer,
            "index_head_dim",
            getattr(indexer, "head_dim", _INDEXER_HEAD_DIM),
        )
        warmup_fp8_mqa_logits_triton(
            num_heads=indexer_num_heads, head_dim=indexer_head_dim, device=device
        )
        warmup_fp8_paged_mqa_logits_triton(
            num_heads=indexer_num_heads,
            head_dim=indexer_head_dim,
            block_size=_INDEXER_KERNEL_BLOCK_SIZE,
            device=device,
        )

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_c_and_k_pe_cache: torch.Tensor,  # [rows, d_qk]
        topk_indices: torch.Tensor,  # [sq, topk]
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output = triton_mla_sparse_attention(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            sm_scale=self.softmax_scale,
            sm_count=self._sm_count,
        )
        return output[:, : self.num_heads, :]


class TritonMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # The CUDA indexer backends need 64-token pages and share the block
        # size with this backend. The base-class MultipleOf(1) would let
        # auto-selection settle on 16, which the indexer then rejects.
        return [MultipleOf(64)]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        return align_blocks_to_rows(spec)

    @staticmethod
    def get_metadata_cls() -> type[XPUMLASparseMetadata]:
        return XPUMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["TritonMLASparseMetadataBuilder"]:
        return TritonMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["TritonMLASparseImpl"]:
        return TritonMLASparseImpl

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)
