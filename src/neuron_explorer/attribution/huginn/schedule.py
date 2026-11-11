"""Virtual-depth schedule for Huginn's prelude, recurrent core, and coda."""

from __future__ import annotations

from dataclasses import dataclass

HUGINN_PRELUDE_BLOCKS = 2
HUGINN_RECURRENT_BLOCKS = 4
HUGINN_RECURRENCE = 32
HUGINN_CODA_BLOCKS = 2


@dataclass(frozen=True)
class ExecutionSlot:
    """One executed SandwichBlock, distinct from the shared parameter block."""

    virtual_depth: int
    stage: str
    recurrence_iteration: int | None
    physical_layer: int
    physical_block: int


def execution_schedule(
    n_prelude: int,
    n_recurrent: int,
    mean_recurrence: int,
    n_coda: int,
) -> list[ExecutionSlot]:
    """Unroll actual block executions.

    Physical layer ids follow Raven's constructor: prelude ``0..P-1``, each
    shared recurrent block ``P+i``, and coda blocks after ``P + R * recurrence``.
    """
    if min(n_prelude, n_recurrent, mean_recurrence, n_coda) < 0:
        raise ValueError("schedule sizes must be non-negative")
    slots: list[ExecutionSlot] = []
    for index in range(n_prelude):
        slots.append(
            ExecutionSlot(
                virtual_depth=index,
                stage="prelude",
                recurrence_iteration=None,
                physical_layer=index,
                physical_block=index,
            )
        )
    for iteration in range(mean_recurrence):
        for index in range(n_recurrent):
            slots.append(
                ExecutionSlot(
                    virtual_depth=n_prelude + iteration * n_recurrent + index,
                    stage="recurrent",
                    recurrence_iteration=iteration,
                    physical_layer=n_prelude + index,
                    physical_block=index,
                )
            )
    coda_origin = n_prelude + n_recurrent * mean_recurrence
    for index in range(n_coda):
        slots.append(
            ExecutionSlot(
                virtual_depth=coda_origin + index,
                stage="coda",
                recurrence_iteration=None,
                physical_layer=coda_origin + index,
                physical_block=index,
            )
        )
    return slots


def shared_weight_id(slot: ExecutionSlot, neuron: int) -> str:
    """Identify the shared MLP neuron, independent of recurrence iteration."""
    return f"{slot.stage}:{slot.physical_block}:n{neuron}"
