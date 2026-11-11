"""Runtime-neutral immutable records for MoE attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    slot: MoEExecutionSlot
    position: int
    expert_id: int
    expert_neuron: int
    expert_width: int
    router_rank: int
    routing_weight: float

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
    selected_experts: torch.Tensor
    routing_weights: torch.Tensor
    router_activation: str
    router_normalization: str


@dataclass(frozen=True)
class MoEContributions:
    routed: torch.Tensor
    token_embeddings: torch.Tensor
    state: torch.Tensor
    targets: torch.Tensor
    routed_neurons: tuple[RoutedNeuron, ...]


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
    expert_parallel: bool = False
    auxiliary: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "model_name": self.model_name,
            "num_experts": self.num_experts,
            "expert_width": self.expert_width,
            "top_k": self.top_k,
            "router_activation": self.router_activation,
            "router_normalization": self.router_normalization,
            "has_shared_experts": self.has_shared_experts,
            "expert_parallel": self.expert_parallel,
            "auxiliary": dict(self.auxiliary),
        }
