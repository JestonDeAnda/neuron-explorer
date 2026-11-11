"""Small deterministic eager-PyTorch MoE backends used by contract tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from neuron_explorer.attribution.moe.types import (
    MoEContributions,
    MoEExecutionSlot,
    MoEModelIdentity,
    RoutedNeuron,
    RoutingSnapshot,
)

RouterStrategy = Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]


def _sigmoid_selected(
    logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = logits.sigmoid()
    selected_scores, selected = scores.topk(top_k, dim=-1)
    return selected, selected_scores / selected_scores.sum(-1, keepdim=True)


def _softmax_selected(
    logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = logits.softmax(dim=-1)
    selected_scores, selected = scores.topk(top_k, dim=-1)
    return selected, selected_scores


class TinyEagerMoEBackend:
    """One-layer MoE whose routing policy is supplied as a callable."""

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        router_strategy: RouterStrategy,
        router_activation: str,
        router_normalization: str,
    ) -> None:
        self.identity = MoEModelIdentity(
            model_family="tiny-eager-moe",
            model_name=f"tiny-{num_experts}e-top{top_k}-{router_activation}",
            num_experts=num_experts,
            expert_width=3,
            top_k=top_k,
            router_activation=router_activation,
            router_normalization=router_normalization,
            auxiliary={"implementation": "eager-pytorch"},
        )
        self.schedule = (
            MoEExecutionSlot(
                virtual_depth=0,
                stage="body",
                recurrence_iteration=None,
                physical_layer=0,
                physical_block=0,
            ),
        )
        self._router_strategy = router_strategy
        self._embedding = torch.linspace(-0.8, 0.9, 17 * 4).reshape(17, 4)
        self._router = torch.linspace(-0.2, 0.25, 4 * num_experts).reshape(
            4, num_experts
        )
        self._router_bias = torch.arange(num_experts, 0, -1, dtype=torch.float) * 2.0
        base = torch.linspace(-0.7, 0.8, num_experts * 4 * 3)
        self._gate = base.reshape(num_experts, 4, 3)
        self._up = base.flip(0).reshape(num_experts, 4, 3)
        self._down = torch.linspace(-0.5, 0.6, num_experts * 3 * 4).reshape(
            num_experts, 3, 4
        )
        self._output = torch.linspace(-0.4, 0.45, 4 * 17).reshape(4, 17)
        self._neurons: tuple[RoutedNeuron, ...] = ()
        self._routing_snapshot: RoutingSnapshot | None = None
        self._leaf: torch.Tensor | None = None
        self._embedding_leaf: torch.Tensor | None = None
        self._logits: torch.Tensor | None = None
        self._index: dict[RoutedNeuron, int] = {}
        self._token_ids: tuple[int, ...] | None = None

    @property
    def routing_snapshot(self) -> RoutingSnapshot:
        if self._routing_snapshot is None:
            raise RuntimeError("forward must be called before reading routes")
        return self._routing_snapshot

    def forward(
        self,
        token_ids: Sequence[int],
        *,
        copies: int = 1,
        replay: bool = False,
        mode: str = "direct",
    ) -> torch.Tensor:
        del replay, mode
        ids = torch.tensor([int(token) for token in token_ids], dtype=torch.long)
        embedding = self._embedding[ids].detach().requires_grad_(True)
        router_logits = embedding.detach() @ self._router + self._router_bias
        selected, weights = self._router_strategy(router_logits, self.identity.top_k)

        neurons: list[RoutedNeuron] = []
        values: list[torch.Tensor] = []
        for position in range(len(ids)):
            hidden = embedding[position].detach()
            for rank in range(self.identity.top_k):
                expert = int(selected[position, rank])
                activation = torch.nn.functional.silu(hidden @ self._gate[expert]) * (
                    hidden @ self._up[expert]
                )
                for expert_neuron in range(self.identity.expert_width):
                    neurons.append(
                        RoutedNeuron(
                            slot=self.schedule[0],
                            position=position,
                            expert_id=expert,
                            expert_neuron=expert_neuron,
                            expert_width=self.identity.expert_width,
                            router_rank=rank,
                            routing_weight=float(weights[position, rank]),
                        )
                    )
                    values.append(activation[expert_neuron])

        leaf = torch.stack(values).detach().requires_grad_(True)
        hidden_rows: list[torch.Tensor] = []
        offset = 0
        for position in range(len(ids)):
            update = torch.zeros(4)
            for rank in range(self.identity.top_k):
                expert = int(selected[position, rank])
                expert_values = leaf[offset : offset + self.identity.expert_width]
                update = update + weights[position, rank] * (
                    expert_values @ self._down[expert]
                )
                offset += self.identity.expert_width
            hidden_rows.append(embedding[position] + update)
        logits = torch.stack(hidden_rows) @ self._output

        self._neurons = tuple(neurons)
        self._index = {neuron: index for index, neuron in enumerate(self._neurons)}
        self._routing_snapshot = RoutingSnapshot(
            selected_experts=selected.detach(),
            routing_weights=weights.detach(),
            router_activation=self.identity.router_activation,
            router_normalization=self.identity.router_normalization,
        )
        self._leaf = leaf
        self._embedding_leaf = embedding
        self._logits = logits
        self._token_ids = tuple(int(token) for token in token_ids)
        return logits.unsqueeze(0).expand(copies, -1, -1)

    def routed_neurons(self) -> Sequence[RoutedNeuron]:
        return self._neurons

    def candidate_effects(
        self, completion_start: int, token_ids: Sequence[int]
    ) -> torch.Tensor:
        self._ensure_forward(token_ids)
        target = self.completion_logprob(0, completion_start, token_ids)
        assert self._leaf is not None
        gradient = torch.autograd.grad(target, self._leaf, retain_graph=True)[0]
        return (gradient * self._leaf).detach()

    def direct_targets(
        self,
        targets: Sequence[int],
        completion_start: int,
        token_ids: Sequence[int],
    ) -> MoEContributions:
        self._ensure_forward(token_ids)
        assert self._leaf is not None
        assert self._embedding_leaf is not None
        scalar_targets = [
            (
                self.completion_logprob(0, completion_start, token_ids)
                if target == -1
                else self._leaf[int(target)]
            )
            for target in targets
        ]
        routed_rows: list[torch.Tensor] = []
        embedding_rows: list[torch.Tensor] = []
        for scalar in scalar_targets:
            routed_gradient, embedding_gradient = torch.autograd.grad(
                scalar,
                (self._leaf, self._embedding_leaf),
                retain_graph=True,
                allow_unused=True,
            )
            if routed_gradient is None:
                routed_gradient = torch.zeros_like(self._leaf)
            if embedding_gradient is None:
                embedding_gradient = torch.zeros_like(self._embedding_leaf)
            routed_rows.append((routed_gradient * self._leaf).detach())
            embedding_rows.append(
                (embedding_gradient * self._embedding_leaf).sum(-1).detach()
            )
        return MoEContributions(
            routed=torch.stack(routed_rows),
            token_embeddings=torch.stack(embedding_rows),
            state=torch.zeros(len(targets)),
            targets=torch.stack([target.detach() for target in scalar_targets]),
            routed_neurons=self._neurons,
        )

    def activations(self, neurons: Sequence[RoutedNeuron]) -> torch.Tensor:
        if self._leaf is None:
            raise RuntimeError("forward must be called before reading activations")
        return torch.stack([self._leaf[self._index[neuron]].detach() for neuron in neurons])

    def completion_logprob(
        self, row: int, completion_start: int, token_ids: Sequence[int]
    ) -> torch.Tensor:
        del row
        self._ensure_forward(token_ids)
        assert self._logits is not None
        values = [
            self._logits[position - 1].log_softmax(-1)[int(token_ids[position])]
            for position in range(completion_start, len(token_ids))
        ]
        return torch.stack(values).sum()

    def _ensure_forward(self, token_ids: Sequence[int]) -> None:
        requested = tuple(int(token) for token in token_ids)
        if self._logits is None or self._token_ids != requested:
            self.forward(token_ids)


def sigmoid_top2_backend() -> TinyEagerMoEBackend:
    return TinyEagerMoEBackend(
        num_experts=4,
        top_k=2,
        router_strategy=_sigmoid_selected,
        router_activation="sigmoid",
        router_normalization="selected_renormalized",
    )


def softmax_top1_backend() -> TinyEagerMoEBackend:
    return TinyEagerMoEBackend(
        num_experts=3,
        top_k=1,
        router_strategy=_softmax_selected,
        router_activation="softmax",
        router_normalization="all_experts",
    )
