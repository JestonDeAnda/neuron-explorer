"""Protocol implemented by runtime-specific MoE linearization backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch

from neuron_explorer.attribution.moe.types import (
    MoEContributions,
    MoEExecutionSlot,
    MoEModelIdentity,
    RoutedNeuron,
)


@runtime_checkable
class MoELinearizationBackend(Protocol):
    identity: MoEModelIdentity
    schedule: Sequence[MoEExecutionSlot]

    def forward(
        self,
        token_ids: Sequence[int],
        *,
        copies: int = 1,
        replay: bool = False,
        mode: str = "direct",
    ) -> torch.Tensor:
        raise NotImplementedError

    def routed_neurons(self) -> Sequence[RoutedNeuron]:
        """Executed routed neurons of the last forward, possibly lazy.

        The axis can hold tens of millions of entries, so callers index it and
        never iterate or copy it. Equality between two axes of one backend must
        not iterate them either.
        """
        raise NotImplementedError

    def candidate_effects(
        self, completion_start: int, token_ids: Sequence[int]
    ) -> torch.Tensor:
        raise NotImplementedError

    def direct_targets(
        self,
        targets: Sequence[int],
        completion_start: int,
        token_ids: Sequence[int],
        *,
        sources: Sequence[int] | None = None,
    ) -> MoEContributions:
        """Attribute targets whose nonnegative indices address ``routed_neurons``.

        With ``sources``, return routed columns for those axis indices only, in
        that order, and set ``routed_columns`` to them.
        """
        raise NotImplementedError

    def activations(self, neurons: Sequence[RoutedNeuron]) -> torch.Tensor:
        raise NotImplementedError

    def completion_logprob(
        self, row: int, completion_start: int, token_ids: Sequence[int]
    ) -> torch.Tensor:
        raise NotImplementedError
