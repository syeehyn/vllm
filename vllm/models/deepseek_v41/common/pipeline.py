# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-sharing dependencies of a DeepSeek V4.1 pipeline partition."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SharingDependency:
    kind: str
    source_layer: int
    consumer_layer: int
    source_stage: int
    consumer_stage: int


def get_sharing_dependencies(
    config: Any, stage_ranges: list[tuple[int, int]]
) -> tuple[SharingDependency, ...]:
    """Resolve KV, index and candidate sources before constructing any layers."""
    num_layers = config.num_hidden_layers
    if (
        not stage_ranges
        or stage_ranges[0][0] != 0
        or stage_ranges[-1][1] != num_layers
        or any(start >= end for start, end in stage_ranges)
        or any(a[1] != b[0] for a, b in zip(stage_ranges, stage_ranges[1:]))
    ):
        raise ValueError("DeepSeek V4.1 pipeline stages must partition all layers")
    owners = {
        layer: stage
        for stage, (start, end) in enumerate(stage_ranges)
        for layer in range(start, end)
    }
    ratios = config.compress_ratios
    if len(ratios) < num_layers:
        raise ValueError(
            "DeepSeek V4.1 compress_ratios must cover every backbone layer"
        )

    sources = {}
    for kind, field in (
        ("kv", "kv_source_layer_ids"),
        ("index", "index_source_layer_ids"),
    ):
        values = tuple(getattr(config, field, None) or ())
        if (
            any(
                type(layer) is not int or not 0 <= layer < num_layers
                for layer in values
            )
            or values != tuple(sorted(set(values)))
            or any(ratios[layer] == 0 for layer in values)
        ):
            raise ValueError(
                f"DeepSeek V4.1 {field} must contain sorted, unique compressed layers"
            )
        sources[kind] = values

    if not set(sources["kv"]).issubset(sources["index"]):
        raise ValueError("DeepSeek V4.1 KV sources must also publish index keys")

    dependencies = []

    def append(kind: str, source: int, consumer: int) -> None:
        if source != consumer:
            dependencies.append(
                SharingDependency(
                    kind, source, consumer, owners[source], owners[consumer]
                )
            )

    for layer in range(num_layers):
        if ratios[layer] == 0:
            continue
        for kind, values in sources.items():
            preceding = [source for source in values if source <= layer]
            if not preceding:
                raise ValueError(
                    f"DeepSeek V4.1 layer {layer} has no preceding {kind} source"
                )
            append(kind, preceding[-1], layer)
            if kind == "kv" and layer in sources["index"]:
                append("index_k", preceding[-1], layer)

    candidate = getattr(config, "candidate_source_layer_id", -1)
    if candidate >= 0 and getattr(config, "candidate_topk_blocks", 0) > 0:
        if candidate not in sources["index"]:
            raise ValueError("DeepSeek V4.1 candidate source must be an index source")
        for layer in sources["index"]:
            if layer > candidate:
                append("candidate", candidate, layer)
    return tuple(dependencies)


def validate_local_sharing(dependencies: tuple[SharingDependency, ...]) -> None:
    """Reject stage cuts that require sharing tensors between pipeline ranks."""
    cross_stage = [d for d in dependencies if d.source_stage != d.consumer_stage]
    if cross_stage:
        detail = "; ".join(
            f"{d.kind} source layer {d.source_layer} (stage {d.source_stage}) -> "
            f"layer {d.consumer_layer} (stage {d.consumer_stage})"
            for d in cross_stage[:4]
        )
        raise NotImplementedError(
            "DeepSeek V4.1 pipeline partition crosses layer-sharing dependencies: "
            f"{detail}. Keep each sharing group on one stage or explicitly enable "
            "the experimental deepseek_v41_pp_sharing eager path."
        )


def decoder_replay_stage_reason(
    start_layer: int,
    end_layer: int,
    cut: int,
    num_layers: int,
    allow_start_at_cut: bool,
) -> str | None:
    """Why the stage holding layers ``[start_layer, end_layer)`` cannot run the
    decoder-side SWA bounded replay from the last KV source ``cut``, or None.

    The replay writes layer ``cut``'s KV for every row and then runs layers
    ``cut``.. on each request's trailing window, so the stage must hold all of
    them. With every sharing group local to its stage, the rows' hidden states
    are all such a stage reads from earlier ones, so one starting at ``cut``
    (PP 20,20) qualifies too with ``allow_start_at_cut``. Earlier stages hold
    KV sources that need every row, so they never replay.
    """
    holds_cut = start_layer <= cut if allow_start_at_cut else start_layer < cut
    if not holds_cut or end_layer < num_layers:
        return (
            "the pipeline stage holding the last KV source layer must also "
            "hold every layer after it"
            + ("" if allow_start_at_cut else ", and a layer before it")
        )
    return None
