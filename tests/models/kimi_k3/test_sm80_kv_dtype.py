# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3.1's default KV cache dtype on SM8x, where the only sparse MLA
backend, TRITON_MLA_SPARSE, reads BF16 rows instead of fp8_ds_mla."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.config import KimiK3ForConditionalGenerationConfig
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig

# Kimi-K3.1 carries its DSA config on the text backbone.
K31_CONFIG = SimpleNamespace(
    text_config=SimpleNamespace(
        dsa_config=SimpleNamespace(
            index_head_dim=128, index_num_heads=32, index_topk=2048
        )
    )
)
# kv_lora_rank + qk_rope_head_dim, and the heads per rank at TP8.
K31_MLA_HEAD_SIZE = 512 + 64
K31_NUM_HEADS = 96 // 8

TRITON_MLA_SPARSE = AttentionBackendEnum.TRITON_MLA_SPARSE
FLASHMLA_SPARSE = AttentionBackendEnum.FLASHMLA_SPARSE
FLASHINFER_MLA_SPARSE = AttentionBackendEnum.FLASHINFER_MLA_SPARSE


def _k31_cache_dtype(cache_dtype, backend=None):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=K31_CONFIG),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        attention_config=SimpleNamespace(backend=backend),
        speculative_config=None,
    )
    KimiK3ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
    return vllm_config.cache_config.cache_dtype


def _fake_cuda_device(monkeypatch, major: int | None) -> None:
    """Make the platform a CUDA device of capability `major`.0; `None` is a
    CUDA platform that cannot read its capability."""
    capability = None if major is None else DeviceCapability(major, 0)

    def has_device_capability(wanted, device_id=0):
        if capability is None:
            return False
        if isinstance(wanted, tuple):
            return capability >= wanted
        return capability.to_int() >= wanted

    def is_device_capability_family(wanted, device_id=0):
        return capability is not None and capability.major == wanted // 10

    fakes = {
        "is_cuda": lambda: True,
        "get_device_capability": lambda device_id=0: capability,
        "has_device_capability": has_device_capability,
        "is_device_capability_family": is_device_capability_family,
    }
    for name, fake in fakes.items():
        monkeypatch.setattr(current_platform, name, fake)


@pytest.mark.parametrize("backend", [None, TRITON_MLA_SPARSE])
def test_sm8x_defaults_to_bf16_kv(monkeypatch, backend):
    _fake_cuda_device(monkeypatch, 8)
    assert _k31_cache_dtype("auto", backend) == "bfloat16"


@pytest.mark.parametrize("major", [8, 9])
@pytest.mark.parametrize(
    ("backend", "cache_dtype", "expected"),
    [
        (None, "bfloat16", "bfloat16"),
        (None, "float16", "float16"),
        (None, "fp8", "fp8_ds_mla"),
        (None, "fp8_ds_mla", "fp8_ds_mla"),
        (None, "nvfp4_ds_mla", "nvfp4_ds_mla"),
        (TRITON_MLA_SPARSE, "fp8", "fp8_ds_mla"),
        # A backend the user picked keeps the dtype that backend reads.
        (FLASHMLA_SPARSE, "auto", "fp8_ds_mla"),
        (FLASHINFER_MLA_SPARSE, "auto", "fp8_e4m3"),
    ],
)
def test_explicit_choice_resolves_on_sm8x_as_on_sm90(
    monkeypatch, major, backend, cache_dtype, expected
):
    _fake_cuda_device(monkeypatch, major)
    assert _k31_cache_dtype(cache_dtype, backend) == expected


@pytest.mark.parametrize(
    ("major", "backend", "expected"),
    [
        (9, None, "fp8_ds_mla"),
        (9, TRITON_MLA_SPARSE, "fp8_ds_mla"),
        (10, None, "fp8_e4m3"),
        (12, None, "fp8_ds_mla"),
        # Unknown capability, e.g. no GPU is visible where the config is built.
        (None, None, "fp8_ds_mla"),
    ],
)
def test_default_is_unchanged_off_sm8x(monkeypatch, major, backend, expected):
    _fake_cuda_device(monkeypatch, major)
    assert _k31_cache_dtype("auto", backend) == expected


def test_default_is_unchanged_off_cuda(monkeypatch):
    _fake_cuda_device(monkeypatch, 8)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    assert _k31_cache_dtype("auto") == "fp8_ds_mla"


@pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(80)
    ),
    reason="needs an SM8x CUDA device",
)
def test_sm8x_default_kv_dtype_selects_triton_sparse(default_vllm_config):
    """On the device itself, the default is a dtype the selector can serve."""
    cache_dtype = _k31_cache_dtype("auto")
    assert cache_dtype == "bfloat16"

    valid, invalid_reasons = current_platform.get_valid_backends(
        device_capability=current_platform.get_device_capability(),
        attn_selector_config=AttentionSelectorConfig(
            head_size=K31_MLA_HEAD_SIZE,
            dtype=torch.bfloat16,
            kv_cache_dtype=cache_dtype,
            block_size=None,
            use_mla=True,
            use_sparse=True,
        ),
        num_heads=K31_NUM_HEADS,
    )
    assert [candidate.backend for candidate in valid] == [TRITON_MLA_SPARSE], (
        invalid_reasons
    )
