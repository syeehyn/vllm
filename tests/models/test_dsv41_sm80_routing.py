# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 routing for DeepSeek-V4.1: capability gates select the Ampere path.

These are CPU-only unit tests: they check the selection logic that decides
whether an A100/A800 run reaches the BF16 Triton sparse-MLA implementation
instead of FlashMLA's SM90a/SM100f kernels.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41 import attention as dsv41_attention
from vllm.models.deepseek_v41.nvidia import model as dsv41_model
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla import sparse_swa


def _config(backend=None, pcp=1):
    return SimpleNamespace(
        attention_config=SimpleNamespace(backend=backend),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            use_ubatching=False,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        kernel_config=SimpleNamespace(moe_backend=None),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
    )


@pytest.mark.parametrize(
    "capability,expected_name",
    [
        (DeviceCapability(8, 0), "DeepseekV4AmpereAttention"),
        (DeviceCapability(8, 6), "DeepseekV4AmpereAttention"),
        (DeviceCapability(9, 0), "DeepseekV4FlashMLAAttention"),
        (DeviceCapability(10, 0), "DeepseekV4FlashMLAAttention"),
        (DeviceCapability(12, 0), "DeepseekV4FlashInferSM120Attention"),
    ],
)
def test_default_backend_by_capability(monkeypatch, capability, expected_name):
    monkeypatch.setattr(
        dsv41_model.current_platform,
        "get_device_capability",
        lambda device=None: capability,
    )
    if capability.major == 10:
        monkeypatch.setattr(
            dsv41_model.DeepseekV4MegaAttnAttention,
            "is_available_for",
            classmethod(lambda cls, cfg: False),
        )
    cls = dsv41_model._select_dsv4_attn_cls(_config())
    assert cls.__name__ == expected_name


def test_explicit_flashmla_backend_on_sm80_routes_to_ampere(monkeypatch):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    monkeypatch.setattr(
        dsv41_model.current_platform,
        "get_device_capability",
        lambda device=None: DeviceCapability(8, 0),
    )
    cls = dsv41_model._select_dsv4_attn_cls(
        _config(backend=AttentionBackendEnum.FLASHMLA_SPARSE_DSV41)
    )
    assert cls.__name__ == "DeepseekV4AmpereAttention"


def test_ampere_keeps_natural_head_count():
    from vllm.models.deepseek_v41.ampere.attention import (
        DeepseekV4AmpereAttention,
    )

    assert DeepseekV4AmpereAttention.get_padded_num_q_heads(8) == 8
    assert DeepseekV4AmpereAttention.get_padded_num_q_heads(64) == 64


@pytest.mark.parametrize(
    "family, cuda, opt_in, want",
    [
        (80, True, False, False),  # SM80 default: the V4 584 B record
        (80, True, True, True),  # VLLM_DSV41_AMPERE_KV_MXFP8=1
        (90, True, True, False),  # the opt-in is SM8x only
        (100, True, False, True),  # SM100 always reads the V4.1 record
        (80, False, True, False),  # ROCm keeps 584 (its mono runner asserts it)
    ],
)
def test_v41_kv_record_gate(monkeypatch, family, cuda, opt_in, want):
    platform = dsv41_attention.current_platform
    monkeypatch.setattr(
        platform, "is_device_capability_family", staticmethod(lambda m: m == family)
    )
    monkeypatch.setattr(platform, "is_cuda", staticmethod(lambda: cuda))
    monkeypatch.setattr(dsv41_attention.envs, "VLLM_DSV41_AMPERE_KV_MXFP8", opt_in)
    assert dsv41_attention._use_v41_mxfp8_kv_record() is want


def test_tile_scheduler_skipped_without_flashmla(monkeypatch):
    """Ampere has no FlashMLA C extension: no tile-scheduler plan is built."""
    builder = sparse_swa.DeepseekSparseSWAMetadataBuilder.__new__(
        sparse_swa.DeepseekSparseSWAMetadataBuilder
    )
    monkeypatch.setattr(
        sparse_swa, "is_flashmla_sparse_supported", lambda: (False, "sm80")
    )
    monkeypatch.setattr(
        sparse_swa.current_platform, "is_rocm", staticmethod(lambda: False)
    )
    monkeypatch.setattr(
        sparse_swa.current_platform, "is_xpu", staticmethod(lambda: False)
    )
    monkeypatch.setattr(
        sparse_swa.current_platform,
        "is_device_capability_family",
        staticmethod(lambda major: False),
    )
    builder._layer_types = {"swaonly"}
    built = builder.build_tile_scheduler(4)
    assert set(built.values()) == {None}


def test_portable_fp8_roundtrip_matches_native():
    """fp8_e4m3_portable decodes every finite E4M3 byte on CPU-only runs."""
    from vllm.v1.attention.ops import fp8_e4m3_portable

    if not torch.cuda.is_available():
        pytest.skip("kernel is Triton/CUDA")
    device = torch.device("cuda")
    raw = torch.arange(256, dtype=torch.int32, device=device).to(torch.uint8)
    got = fp8_e4m3_portable.e4m3_to_float(raw.view(torch.uint8))
    ref = raw.view(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    finite = torch.isfinite(ref) & (raw != 0x7F) & (raw != 0xFF)
    torch.testing.assert_close(got[finite], ref[finite], rtol=0, atol=0)
