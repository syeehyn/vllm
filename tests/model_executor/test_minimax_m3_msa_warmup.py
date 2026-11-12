# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from types import SimpleNamespace

import pytest

from vllm.model_executor.warmup import kernel_warmup

pytestmark = pytest.mark.cpu_test

_MODEL = "vllm.models.minimax_m3.nvidia.model"
_WARMUP = "vllm.model_executor.warmup.minimax_m3_msa_warmup"


def test_msa_warmup_is_not_imported_for_other_models(monkeypatch) -> None:
    """Without the MiniMax-M3 model module loaded, the MSA warmup, and the CuTe
    DSL kernels its modules import, stay unimported."""
    monkeypatch.delitem(sys.modules, _MODEL, raising=False)
    # A `None` entry makes an import of the module raise ImportError.
    monkeypatch.setitem(sys.modules, _WARMUP, None)

    kernel_warmup._warmup_minimax_m3_msa(SimpleNamespace())


def test_msa_warmup_runs_for_a_minimax_m3_model(monkeypatch) -> None:
    calls = []
    monkeypatch.setitem(sys.modules, _MODEL, SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, _WARMUP, SimpleNamespace(minimax_m3_msa_warmup=calls.append)
    )
    worker = SimpleNamespace()

    kernel_warmup._warmup_minimax_m3_msa(worker)

    assert calls == [worker]
