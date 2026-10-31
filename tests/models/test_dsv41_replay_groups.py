# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SWA bounded replay pads the prefix-cacheable KV groups' slot-mapping rows on
a prefix hit. The group indices must come from the KV config the slot mappings
were built for: CUDA-graph memory profiling binds a throwaway per-stage config
first (PP0 on DeepSeek V4.1: 9 groups, cacheable at 7; real: 6 groups, at 4),
and indices cached from it padded a row past the end of the [6, N] mapping,
which overwrote a layer-19 norm weight with 0xFF on A100."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.nvidia import model_state as ms


def _specs(cacheable: list[int], n: int) -> list:
    return [SimpleNamespace(prefix_cacheable=i in cacheable) for i in range(n)]


def test_cacheable_groups_from_matching_config():
    assert ms.replay_cacheable_groups(_specs([4], 6), 6) == [4]


def test_profiling_config_indices_are_rejected():
    # The per-stage profiling grouping seen on PP0: 9 groups, cacheable at 7.
    with pytest.raises(RuntimeError, match="slot mappings have 6 rows"):
        ms.replay_cacheable_groups(_specs([7], 9), 6)


def test_initialize_kv_cache_drops_cached_replay_groups():
    state = object.__new__(ms.DeepseekV41ModelState)
    state._replay = (128, torch.tensor([7], dtype=torch.int32))
    state.decoder_replay_layers = None
    state.initialize_kv_cache(SimpleNamespace(kv_cache_groups=[]), None)
    assert state._replay is None


def test_initialize_kv_cache_drops_kept_slot_mappings_with_decoder_replay():
    state = object.__new__(ms.DeepseekV41ModelState)
    state._replay = (128, torch.tensor([7], dtype=torch.int32))
    state.decoder_replay_layers = object()
    state._kept_slot_mappings = torch.zeros(9, 8, dtype=torch.int64)
    state.initialize_kv_cache(SimpleNamespace(kv_cache_groups=[]), None)
    assert state._replay is None and state._kept_slot_mappings is None
