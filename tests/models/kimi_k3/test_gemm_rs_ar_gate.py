# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys

import pytest

from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.platforms import current_platform

pytestmark = pytest.mark.cpu_test

_GEMM_RS_AR = "vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar"


@pytest.mark.parametrize("use_sequence_parallel", [False, True])
def test_gemm_rs_ar_is_not_imported_off_sm100(
    monkeypatch, use_sequence_parallel: bool
) -> None:
    """With the flags on, a device that cannot run GEMM-RS/AR declines before
    importing the kernel module and the CuTe DSL it pulls in."""
    monkeypatch.setenv("VLLM_KIMI_K3_GEMM_AR", "1")
    monkeypatch.setenv("VLLM_ENABLE_GEMM_RS", "1")
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: False,
    )
    # A `None` entry makes an import of the module raise ImportError.
    monkeypatch.setitem(sys.modules, _GEMM_RS_AR, None)

    assert not kimi_model.maybe_init_gemm_rs_ar(None, use_sequence_parallel)
