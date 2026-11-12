"""Runtime-neutral immutable records for MoE attribution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MoEExecutionSlot:
    virtual_depth: int
    stage: str
    recurrence_iteration: int | None
    physical_layer: int
    physical_block: int

    def as_dict(self) -> dict[str, int | str | None]:
        return {
            "virtual_depth": self.virtual_depth,
            "stage": self.stage,
            "recurrence_iteration": self.recurrence_iteration,
            "physical_layer": self.physical_layer,
            "physical_block": self.physical_block,
        }


@dataclass(frozen=True)
class RoutedNeuron:
    """One expert-local activation.

    ``expert_id`` is globally stable within the model, while
    ``runtime_expert_id`` is the runtime-local shard or dispatch ID.
    ``physical_block`` must uniquely identify a shared parameter block within
    its stage, which keeps ``shared_weight_id`` stable across recurrence.
    """

    slot: MoEExecutionSlot
    position: int
    expert_id: int
    expert_neuron: int
    expert_width: int
    router_rank: int | None
    routing_weight: float | None
    runtime_expert_id: int | None = None
    expert_kind: str = "routed"

    def __post_init__(self) -> None:
        if self.expert_kind not in {"routed", "shared"}:
            raise ValueError(f"unsupported expert kind: {self.expert_kind!r}")
        has_route = self.router_rank is not None and self.routing_weight is not None
        if (self.expert_kind == "routed") != has_route:
            raise ValueError("only routed experts have router rank and weight")
        if self.runtime_expert_id is None:
            object.__setattr__(self, "runtime_expert_id", self.expert_id)

    @property
    def serialized_neuron(self) -> int:
        return self.expert_id * self.expert_width + self.expert_neuron

    @property
    def shared_weight_id(self) -> str:
        return (
            f"{self.slot.stage}:{self.slot.physical_block}:"
            f"e{self.expert_id}:n{self.expert_neuron}"
        )


@dataclass(frozen=True)
class RoutingSnapshot:
    """Padded routes with an explicit selected count per slot and position."""

    selected_experts: torch.Tensor
    routing_weights: torch.Tensor
    selected_expert_counts: torch.Tensor
    router_activation: str
    router_normalization: str

    def __post_init__(self) -> None:
        if self.selected_experts.ndim != 3:
            raise ValueError("selected experts must have slot, position, and rank axes")
        if self.routing_weights.shape != self.selected_experts.shape:
            raise ValueError("routing weights must align with selected experts")
        if self.selected_expert_counts.shape != self.selected_experts.shape[:2]:
            raise ValueError("selected counts must have slot and position axes")

    def routes_at(self, slot: int, position: int) -> tuple[tuple[int, float], ...]:
        count = int(self.selected_expert_counts[slot, position])
        return tuple(
            (
                int(self.selected_experts[slot, position, rank]),
                round(float(self.routing_weights[slot, position, rank]), 6),
            )
            for rank in range(count)
        )


@dataclass(frozen=True)
class MoESourceLeaf:
    """Identity for one runtime-specific non-token source leaf."""

    name: str
    kind: str = "runtime"
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class MoEContributions:
    """Direct contributions; target indices address ``routed_neurons``.

    ``routed_columns`` lists the routed-axis index of each ``routed`` column
    when only selected sources were requested; ``None`` means every column.
    ``routed_neurons`` may be a lazy sequence and must not be copied.
    """

    routed: torch.Tensor
    token_embeddings: torch.Tensor
    source_contributions: torch.Tensor
    targets: torch.Tensor
    routed_neurons: Sequence[RoutedNeuron]
    source_leaves: tuple[MoESourceLeaf, ...] = ()
    routed_columns: tuple[int, ...] | None = None

    @property
    def state(self) -> torch.Tensor:
        """Compatibility aggregate; named source columns remain authoritative."""
        return self.source_contributions.sum(-1)


@dataclass(frozen=True)
class MoEModelIdentity:
    model_family: str
    model_name: str
    num_experts: int
    expert_width: int
    top_k: int
    router_activation: str
    router_normalization: str
    has_shared_experts: bool = False
    device_placement: str = "unspecified"
    expert_parallel: bool = False
    expert_parallel_capable: bool = False
    direct_edge_semantics: str = "backend_declared_direct"
    auxiliary: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "model_family": self.model_family,
            "model_name": self.model_name,
            "num_experts": self.num_experts,
            "expert_width": self.expert_width,
            "top_k": self.top_k,
            "router_activation": self.router_activation,
            "router_normalization": self.router_normalization,
            "has_shared_experts": self.has_shared_experts,
            "device_placement": self.device_placement,
            "expert_parallel": self.expert_parallel,
            "expert_parallel_capable": self.expert_parallel_capable,
            "direct_edge_semantics": self.direct_edge_semantics,
            "auxiliary": dict(self.auxiliary),
        }
