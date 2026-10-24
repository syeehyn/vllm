# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Triton sparse-MLA backend's NoPE-512 (GLM-5.3) gates.

GLM-5.3-Flash is rope-free MLA: qk_nope_head_dim=256, qk_rope_head_dim=0,
v_head_dim=256, kv_lora_rank=512 -> head_size 512, which is also the kernel's
absorbed NoPE width. These tests pin the selection contract only:

* ``get_supported_head_sizes`` advertises 512 next to the 576 V3.2/GLM-5.1
  shape, so the backend becomes a candidate on SM80;
* ``supports_combination`` admits 512 only for rope-free models, since the
  kernel treats the whole QK width as NoPE.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.triton_mla_sparse import (
    TritonMLASparseBackend,
)

SM80 = DeviceCapability(major=8, minor=0)


def _supports_combination(head_size: int) -> str | None:
    return TritonMLASparseBackend.supports_combination(
        head_size,
        torch.bfloat16,
        "auto",
        128,
        use_mla=True,
        has_sink=False,
        use_sparse=True,
        use_mm_prefix=False,
        device_capability=SM80,
    )


@pytest.fixture
def rope_free_model(monkeypatch):
    """Patch get_current_vllm_config so the model looks like GLM-5.3 NoPE."""
    import vllm.config as cfg

    monkeypatch.setattr(
        cfg,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(qk_rope_head_dim=0)
            )
        ),
    )


@pytest.fixture
def rope_carrying_model(monkeypatch):
    """Patch get_current_vllm_config with a rope-carrying (576-style) model."""
    import vllm.config as cfg

    monkeypatch.setattr(
        cfg,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(qk_rope_head_dim=64)
            )
        ),
    )


def test_supported_head_sizes_include_nope_512():
    assert TritonMLASparseBackend.get_supported_head_sizes() == [512, 576]


def test_nope_512_is_accepted(rope_free_model):
    assert _supports_combination(512) is None


def test_rope_carrying_512_is_rejected(rope_carrying_model):
    reason = _supports_combination(512)
    assert reason is not None and "rope-free" in reason


def test_576_is_accepted(rope_carrying_model):
    assert _supports_combination(576) is None
