# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skip checkpoint weights of other pipeline-parallel stages before disk I/O.

Under pipeline parallelism each rank builds only its own decoder layers; the
others are `PPMissingLayer` placeholders, as are the embedding / final norm /
LM head on the stages that do not own them. `load_weights` already drops the
tensors of those placeholders, but only after the loader has read them. For a
checkpoint on network storage that is most of the load time: every rank reads
the whole model to keep 1/pp_size of it.

`PPCheckpointWeightFilter` answers the same question from the checkpoint name
alone, so a model can expose it as `is_unused_checkpoint_weight` and the loader
drops whole shards and single tensors without reading them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper


def get_pp_stage() -> tuple[int, int] | None:
    """Return `(pp_rank, pp_size)` of this process, or `None` when it is not
    one of several pipeline-parallel stages.

    Read from the PP group, the source `make_layers` partitions by.
    """
    from vllm.distributed.parallel_state import get_pp_group

    try:
        pp_group = get_pp_group()
    except AssertionError:
        # The PP group is not initialized (offline tools, unit tests).
        return None
    if pp_group.world_size <= 1:
        return None
    return pp_group.rank_in_group, pp_group.world_size


def get_pp_stage_prefetch_split() -> tuple[int, int]:
    """Return `(rank, world_size)` for splitting the checkpoint prefetch among
    the ranks that load the same pipeline-parallel stage.

    These are the ranks of this process' tensor-parallel group: the rank layout
    is `... x PP x PCP x TP` (see `ensure_model_parallel_initialized`), so a TP
    group never spans two stages, and it is read from the group itself instead
    of being derived from the global `torch.distributed` rank. With prefill
    context parallelism a stage has several TP groups; each then prefetches the
    whole stage, which is redundant only where they share a page cache.
    """
    from vllm.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    return tp_group.rank_in_group, tp_group.world_size


def find_pp_layer_range(model: nn.Module) -> tuple[int, int] | None:
    """Return `(start_layer, end_layer)` of the decoder stack built by
    `make_layers`, or `None` when no module of *model* records one."""
    for module in model.modules():
        start = getattr(module, "start_layer", None)
        end = getattr(module, "end_layer", None)
        if (
            isinstance(start, int)
            and isinstance(end, int)
            and not isinstance(start, bool)
            and not isinstance(end, bool)
        ):
            return start, end
    return None


class PPCheckpointWeightFilter:
    """Checkpoint-name predicate: True for weights of `PPMissingLayer` modules.

    The rule is read off the module tree instead of parsing layer indices out
    of names: a weight is unused iff, after *mapper* has turned its checkpoint
    name into a module path, one of its dotted prefixes names a
    `PPMissingLayer`. That is the test `is_pp_missing_parameter` applies in
    `load_weights`, so numbered containers that are not pipeline stages
    (`experts.N`, vision tower blocks, projector layers) can never match, and
    custom partitions (`VLLM_PP_LAYER_PARTITION`) are honoured because the
    placeholders come from `make_layers`.

    Only renames that keep a weight under the same module prefix may happen
    after *mapper* (e.g. `.q_proj` -> `.in_proj_qkvgfab` inside one decoder
    layer). Names the mapper drops or that match no placeholder are kept.

    Args:
        model: The model whose `load_weights` receives the checkpoint. Its
            module tree is read on the first call, so the filter can be built
            in `__init__` before all submodules exist.
        mapper: The `WeightsMapper` that `load_weights` applies to checkpoint
            names, if any.

    """

    def __init__(self, model: nn.Module, mapper: WeightsMapper | None = None):
        # Not a registered submodule: bypass nn.Module bookkeeping when this
        # object is itself stored on *model*.
        self._model = model
        self._mapper = mapper
        self._missing_modules: frozenset[str] | None = None

    @property
    def missing_modules(self) -> frozenset[str]:
        """Module paths of the `PPMissingLayer` placeholders in the model."""
        if self._missing_modules is None:
            from vllm.model_executor.models.utils import PPMissingLayer

            self._missing_modules = frozenset(
                name
                for name, module in self._model.named_modules()
                if name and isinstance(module, PPMissingLayer)
            )
        return self._missing_modules

    @property
    def layer_range(self) -> tuple[int, int] | None:
        return find_pp_layer_range(self._model)

    def __call__(self, name: str) -> bool:
        missing = self.missing_modules
        if not missing:
            return False
        if self._mapper is not None:
            mapped = self._mapper.map_name(name)
            if mapped is None:
                return False
            name = mapped
        # Test every proper dotted prefix ("a", "a.b", ...) of the name. A set
        # lookup per prefix keeps this cheap for checkpoints with hundreds of
        # thousands of tensors, and matching whole path components means
        # "layers.1" never claims "layers.12.*".
        end = name.find(".")
        while end != -1:
            if name[:end] in missing:
                return True
            end = name.find(".", end + 1)
        return False
