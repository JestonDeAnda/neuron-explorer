"""Executed virtual-depth schedule for the xLLM Huginn checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neuron_explorer.attribution.moe.types import MoEExecutionSlot


def _model_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    model = config.get("model", config)
    if not isinstance(model, Mapping):
        raise ValueError("xLLM config must contain a model mapping")
    return model


def execution_schedule(config: Mapping[str, Any]) -> tuple[MoEExecutionSlot, ...]:
    """Unroll the non-hierarchical xLLM Huginn execution schedule."""
    model = _model_config(config)
    hierarchical_state = model.get("huginn_hierarchical_state")
    if hierarchical_state != "none":
        raise ValueError(
            "only non-hierarchical Huginn execution is supported; "
            f"got {hierarchical_state!r}"
        )

    prelude = int(model["huginn_prelude_layers"])
    recurrent = int(model["huginn_recurrent_layers"])
    coda = int(model["huginn_coda_layers"])
    recurrences = int(model["loop_times"])
    num_layers = int(model["num_layers"])
    if min(prelude, recurrent, coda, recurrences) < 0:
        raise ValueError("Huginn schedule sizes must be non-negative")
    if prelude + recurrent + coda != num_layers:
        raise ValueError(
            "Huginn physical layer partition does not match num_layers: "
            f"{prelude}+{recurrent}+{coda}!={num_layers}"
        )

    slots: list[MoEExecutionSlot] = []

    def append(
        stage: str,
        recurrence_iteration: int | None,
        physical_layer: int,
        physical_block: int,
    ) -> None:
        slots.append(
            MoEExecutionSlot(
                len(slots),
                stage,
                recurrence_iteration,
                physical_layer,
                physical_block,
            )
        )

    for index in range(prelude):
        append("prelude", None, index, index)
    for iteration in range(recurrences):
        for index in range(recurrent):
            append("recurrent", iteration, prelude + index, index)
    for index in range(coda):
        append("coda", None, prelude + recurrent + index, index)
    return tuple(slots)
