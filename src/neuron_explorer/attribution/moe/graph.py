"""Backend-independent MoE attribution graph construction."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from neuron_explorer.attribution.moe.backend import MoELinearizationBackend


def _influence(matrix: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    current, total = weight @ matrix, torch.zeros_like(weight)
    for _ in range(matrix.shape[0]):
        if current.abs().max().item() == 0:
            break
        total = total + current
        current = current @ matrix
    return total


def _serialize_edges(
    adjacency: torch.Tensor,
    edge_mask: torch.Tensor,
    remap: Mapping[int, int],
) -> list[list[int | float]]:
    targets, sources = edge_mask.nonzero(as_tuple=True)
    return [
        [remap[int(source)], remap[int(target)], round(float(adjacency[target, source]), 5)]
        for target, source in zip(targets.tolist(), sources.tolist())
        if int(source) in remap and int(target) in remap
    ]


def build_moe_graph(
    backend: MoELinearizationBackend,
    token_ids: Sequence[int],
    completion_start: int,
    *,
    k: int = 512,
    batch: int = 4,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    max_edges: int | None = 3000,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a graph using only the runtime-neutral backend protocol."""
    ids = [int(token) for token in token_ids]
    if completion_start < 1 or completion_start >= len(ids):
        raise ValueError("completion must be non-empty and follow a non-empty prompt")
    if k < 1 or batch < 1:
        raise ValueError("k and batch must be positive")
    if max_edges is not None and max_edges < 1:
        raise ValueError("max_edges must be positive or None")

    backend.forward(ids, mode="total")
    routed_axis = tuple(backend.routed_neurons())
    effects = backend.candidate_effects(completion_start, ids).detach().float()
    if effects.ndim != 1 or effects.numel() != len(routed_axis):
        raise ValueError("candidate effects must align with the routed-neuron axis")
    effective_k = min(int(k), len(routed_axis))
    chosen = effects.abs().topk(effective_k).indices
    selected = tuple(routed_axis[int(index)] for index in chosen)
    captured = float(
        effects.abs()[chosen].sum() / effects.abs().sum().clamp(min=1e-12)
    )

    backend.forward(ids, copies=batch, replay=True, mode="direct")
    activations = backend.activations(selected).detach().float()
    length = len(ids)
    logical_target_rows = list(range(effective_k)) + [-1]
    global_targets = [int(index) for index in chosen] + [-1]
    chunks = []
    source_axis = None
    for start in range(0, len(logical_target_rows), batch):
        rows = logical_target_rows[start : start + batch]
        requested = global_targets[start : start + batch]
        contribution = backend.direct_targets(requested, completion_start, ids)
        if tuple(contribution.routed_neurons) != tuple(backend.routed_neurons()):
            raise ValueError("direct contributions changed the routed-neuron axis")
        if contribution.routed.shape != (len(rows), len(routed_axis)):
            raise ValueError("routed contributions have an invalid shape")
        if contribution.token_embeddings.shape != (len(rows), length):
            raise ValueError("token-embedding contributions have an invalid shape")
        if contribution.source_contributions.shape != (
            len(rows),
            len(contribution.source_leaves),
        ):
            raise ValueError("named source contributions have an invalid shape")
        if source_axis is None:
            source_axis = contribution.source_leaves
        elif source_axis != contribution.source_leaves:
            raise ValueError("direct contributions changed the named source axis")
        chunks.append((rows, requested, contribution))

    source_axis = tuple(source_axis or ())
    source_start = effective_k + length
    target_node = source_start + len(source_axis)
    n_nodes = target_node + 1
    adjacency = torch.zeros(n_nodes, n_nodes)
    other = torch.zeros(n_nodes)
    completion_value: float | None = None

    for logical_rows, requested, contribution in chunks:
        rows = [target_node if row == -1 else row for row in logical_rows]
        row_tensor = torch.tensor(rows, dtype=torch.long)
        adjacency[row_tensor, :effective_k] = contribution.routed[:, chosen].float()
        adjacency[row_tensor, effective_k : effective_k + length] = (
            contribution.token_embeddings.float()
        )
        adjacency[row_tensor, source_start:target_node] = (
            contribution.source_contributions.float()
        )
        values = contribution.targets.detach().float().reshape(len(rows))
        other[row_tensor] = values - adjacency[row_tensor].sum(1)
        if requested[-1] == -1:
            completion_value = float(values[-1])

    diagonal = adjacency.diagonal().clone()
    other = other + diagonal
    adjacency.fill_diagonal_(0)
    absolute = adjacency.abs()
    normal = absolute / (
        absolute.sum(1, keepdim=True) + other.abs()[:, None]
    ).clamp(min=1e-10)
    downstream = torch.zeros(n_nodes)
    downstream[target_node] = 1.0
    influence = _influence(normal, downstream)
    order = influence[:effective_k].sort(descending=True)
    cumulative = order.values.cumsum(0) / order.values.sum().clamp(min=1e-10)
    keep = torch.zeros(n_nodes, dtype=torch.bool)
    keep[order.indices[: int((cumulative < node_threshold).sum()) + 1]] = True
    keep[effective_k:] = True
    pruned = normal * keep[:, None] * keep[None, :]
    pruned_influence = _influence(pruned, downstream) + downstream
    edge_score = pruned * pruned_influence[:, None]
    edge_mask = _edge_mask(edge_score, edge_threshold, max_edges)
    used = edge_mask.any(0) | edge_mask.any(1)
    used[target_node] = True
    kept_ids = used.nonzero().flatten().tolist()
    remap = {old: new for new, old in enumerate(kept_ids)}

    nodes: list[dict[str, Any]] = []
    for index in kept_ids:
        if index < effective_k:
            routed = selected[index]
            slot = routed.slot
            nodes.append(
                {
                    "kind": "neuron",
                    "layer": slot.virtual_depth,
                    "virtual_depth": slot.virtual_depth,
                    "stage": slot.stage,
                    "recurrence_iteration": slot.recurrence_iteration,
                    "physical_layer": slot.physical_layer,
                    "physical_block": slot.physical_block,
                    "pos": routed.position,
                    "n": routed.serialized_neuron,
                    "physical_neuron": routed.serialized_neuron,
                    "expert_id": routed.expert_id,
                    "runtime_expert_id": routed.runtime_expert_id,
                    "expert_neuron": routed.expert_neuron,
                    "expert_kind": routed.expert_kind,
                    "router_rank": routed.router_rank,
                    "routing_weight": routed.routing_weight,
                    "shared_weight_id": routed.shared_weight_id,
                    "act": round(float(activations[index]), 4),
                    "gate": (
                        round(float(routed.routing_weight), 4)
                        if routed.routing_weight is not None
                        else None
                    ),
                    "influence": round(float(influence[index]), 5),
                    "other": round(float(other[index]), 4),
                }
            )
        elif index < source_start:
            nodes.append(
                {
                    "kind": "embed",
                    "layer": -1,
                    "pos": index - effective_k,
                    "n": 0,
                    "act": None,
                    "influence": round(float(influence[index]), 5),
                }
            )
        elif index < target_node:
            source = source_axis[index - source_start]
            nodes.append(
                {
                    "kind": "embed",
                    "layer": -1,
                    "pos": -1,
                    "n": index - source_start,
                    "act": None,
                    "influence": round(float(influence[index]), 5),
                    "source_kind": source.kind,
                    "source_name": source.name,
                    "source_metadata": dict(source.metadata),
                }
            )
        else:
            nodes.append(
                {
                    "kind": "logit",
                    "layer": len(backend.schedule),
                    "pos": completion_start,
                    "n": -1,
                    "act": round(float(completion_value or 0.0), 6),
                    "influence": 1.0,
                    "other": round(float(other[index]), 4),
                    "target_kind": "completion_logprob",
                    "target_token_ids": ids[completion_start:],
                    "prediction_positions": list(
                        range(completion_start - 1, length - 1)
                    ),
                }
            )

    info = dict(metadata or {})
    graph_config = dict(info.get("graph_config") or {})
    graph_config.update(
        {
            "k": int(k),
            "effective_k": effective_k,
            "batch": int(batch),
            "node_threshold": node_threshold,
            "edge_threshold": edge_threshold,
            "max_edges": max_edges,
            "target": "teacher_forced_completion_logprob",
            "edge_semantics": backend.identity.direct_edge_semantics,
        }
    )
    info["graph_config"] = graph_config
    info["model_identity"] = backend.identity.as_dict()
    info["schedule"] = [slot.as_dict() for slot in backend.schedule]
    info["source_leaves"] = [
        {"name": source.name, "kind": source.kind, "metadata": dict(source.metadata)}
        for source in source_axis
    ]
    value = float(completion_value or 0.0)
    probability = math.exp(value)
    return {
        "token": ids[completion_start],
        "prob": round(probability, 10),
        "target": {
            "kind": "completion_logprob",
            "token_ids": ids[completion_start:],
            "prediction_positions": list(range(completion_start - 1, length - 1)),
        },
        "completion_token_ids": ids[completion_start:],
        "prompt_token_ids": ids[:completion_start],
        "completion_logprob": round(value, 6),
        "nodes": nodes,
        "edges": _serialize_edges(adjacency, edge_mask, remap),
        "stats": {
            "candidates": effective_k,
            "candidate_share": round(captured, 4),
            "kept_neurons": sum(node["kind"] == "neuron" for node in nodes),
            "kept_edges": int(edge_mask.sum()),
            "completion_tokens": length - completion_start,
            "completion_logprob": round(value, 6),
            "sequence_probability": round(probability, 10),
        },
        "metadata": info,
    }


def _edge_mask(
    edge_score: torch.Tensor,
    edge_threshold: float,
    max_edges: int | None,
) -> torch.Tensor:
    flat = edge_score.flatten()
    positive = flat[flat > 0]
    if positive.numel() == 0:
        return torch.zeros_like(edge_score, dtype=torch.bool)
    ranked = positive.sort(descending=True).values
    cumulative = ranked.cumsum(0) / ranked.sum().clamp(min=1e-10)
    cut_index = min(int((cumulative < edge_threshold).sum()), len(ranked) - 1)
    mask = (edge_score >= ranked[cut_index]) & (edge_score > 0)
    if max_edges and int(mask.sum()) > max_edges:
        eligible = mask.flatten().nonzero().flatten()
        scores = flat[eligible]
        chosen = eligible[scores.topk(max_edges).indices]
        capped = torch.zeros_like(flat, dtype=torch.bool)
        capped[chosen] = True
        mask = capped.reshape_as(edge_score)
    return mask
