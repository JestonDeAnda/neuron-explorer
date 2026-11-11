"""Build a Huginn attribution graph for a full teacher-forced completion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from neuron_explorer.attribution.huginn.linearized import HuginnLinearized
from neuron_explorer.attribution.huginn.schedule import shared_weight_id


def _seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _influence(matrix: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Path-sum of a causal edge matrix. ``matrix`` rows are targets."""
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
    """Serialize a target-row/source-column matrix as source→target edges."""
    targets, sources = edge_mask.nonzero(as_tuple=True)
    return [
        [remap[int(source)], remap[int(target)], round(float(adjacency[target, source]), 5)]
        for target, source in zip(targets.tolist(), sources.tolist())
        if int(source) in remap and int(target) in remap
    ]


def build_huginn_graph(
    model,
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
    """Attribute the sum of completion-token log-probabilities.

    Candidate neurons are ranked by total effect. Direct edges use the half
    rule and keep raw signed weights. ``metadata`` is stored unchanged except
    for the effective candidate count recorded inside ``graph_config``.
    """
    ids = [int(token) for token in token_ids]
    if completion_start < 1 or completion_start >= len(ids):
        raise ValueError("completion must be non-empty and follow a non-empty prompt")
    info = dict(metadata or {})
    seed = int(info.get("seed", 0))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        return _build(
            model,
            ids,
            completion_start,
            k,
            batch,
            node_threshold,
            edge_threshold,
            max_edges,
            info,
            seed,
        )
    finally:
        for parameter in parameters:
            parameter.requires_grad_(True)


def _build(
    model,
    ids: list[int],
    completion_start: int,
    k: int,
    batch: int,
    node_threshold: float,
    edge_threshold: float,
    max_edges: int | None,
    metadata: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    length = len(ids)
    with HuginnLinearized(model, mode="total") as linearized:
        _seed(seed)
        logits = linearized.forward(ids)
        token_logprobs = _token_logprobs(logits[0].detach(), ids, completion_start)
        logprob = torch.stack(token_logprobs).sum()
        total = linearized.contributions(
            linearized.completion_logprob_target([(completion_start, ids)]), keep=False
        )
        score = total["mlp"][:, 0]
        neurons = score.shape[-1]
        flat = score.abs().flatten()
        k = min(int(k), int(flat.numel()))
        top = flat.topk(k).indices
        depth, rest = top // (length * neurons), top % (length * neurons)
        position, neuron = rest // neurons, rest % neurons
        captured = float(flat[top].sum() / flat.sum().clamp(min=1e-12))
        del total, score, flat

        linearized.mode = "direct"
        _seed(seed)
        linearized.forward(ids, copies=batch)
        activations = torch.stack(
            [linearized.mlp[layer]["a"].detach()[0] for layer in range(len(linearized.slots))]
        ).float()[depth, position, neuron]
        gates = torch.stack(
            [linearized.mlp[layer]["g"][0] for layer in range(len(linearized.slots))]
        ).float()[depth, position, neuron]
        n_nodes = k + length + 1
        device = activations.device
        adjacency = torch.zeros(n_nodes, n_nodes, device=device)
        other = torch.zeros(n_nodes, device=device)
        targets = list(range(k)) + [k + length]
        logit_value = None
        for start in range(0, len(targets), batch):
            chunk = targets[start : start + batch]
            chosen = [index for index in chunk if index < k]
            pieces = []
            if chosen:
                pieces.append(
                    linearized.neuron_target(
                        [
                            (int(depth[index]), int(position[index]), int(neuron[index]))
                            for index in chosen
                        ]
                    )
                )
            if chunk[-1] == k + length:
                pieces.append(
                    linearized.completion_logprob(len(chosen), completion_start, ids)[None]
                )
            target = torch.cat(pieces)
            contribution = linearized.contributions(
                target, keep=start + batch < len(targets)
            )
            rows = torch.tensor(chunk, device=device)
            selected = contribution["mlp"][
                depth[:, None],
                torch.arange(len(chunk), device=device)[None],
                position[:, None],
                neuron[:, None],
            ].T
            adjacency[rows, :k] = selected
            adjacency[rows, k : k + length] = contribution["embed"][: len(chunk)]
            other[rows] = target.detach().float() - adjacency[rows].sum(1)
            logit_value = float(target[-1].detach()) if chunk[-1] == k + length else logit_value
            del contribution

    absolute = adjacency.abs()
    normal = absolute / (
        absolute.sum(1, keepdim=True) + other.abs()[:, None]
    ).clamp(min=1e-10)
    downstream = torch.zeros(n_nodes, device=device)
    downstream[k + length] = 1.0
    influence = _influence(normal, downstream)
    order = influence[:k].sort(descending=True)
    cumulative = order.values.cumsum(0) / order.values.sum().clamp(min=1e-10)
    keep = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    keep[order.indices[: int((cumulative < node_threshold).sum()) + 1]] = True
    keep[k:] = True
    pruned = normal * keep[:, None] * keep[None, :]
    pruned_influence = _influence(pruned, downstream) + downstream
    edge_score = pruned * pruned_influence[:, None]
    flat_edges = edge_score.flatten()
    ranked = flat_edges.sort(descending=True).values
    cut_index = min(
        int(((ranked.cumsum(0) / ranked.sum().clamp(min=1e-10)) < edge_threshold).sum()),
        len(ranked) - 1,
    )
    cutoff = ranked[cut_index]
    edge_mask = (edge_score >= cutoff) & (edge_score > 0)
    if max_edges and int(edge_mask.sum()) > max_edges:
        edge_mask = edge_mask & (edge_score >= flat_edges.topk(max_edges).values[-1])
    used = edge_mask.any(0) | edge_mask.any(1)
    used[k + length] = True
    kept_ids = used.nonzero().flatten().tolist()
    remap = {old: new for new, old in enumerate(kept_ids)}
    nodes = []
    for index in kept_ids:
        if index < k:
            slot = linearized.slots[int(depth[index])]
            neuron_index = int(neuron[index])
            nodes.append(
                {
                    "kind": "neuron",
                    "layer": slot.virtual_depth,
                    "pos": int(position[index]),
                    "n": neuron_index,
                    "act": round(float(activations[index]), 4),
                    "gate": round(float(gates[index]), 3),
                    "influence": round(float(influence[index]), 5),
                    "other": round(float(other[index]), 4),
                    "virtual_depth": slot.virtual_depth,
                    "stage": slot.stage,
                    "recurrence_iteration": slot.recurrence_iteration,
                    "physical_layer": slot.physical_layer,
                    "physical_neuron": neuron_index,
                    "shared_weight_id": shared_weight_id(slot, neuron_index),
                }
            )
        elif index < k + length:
            nodes.append(
                {
                    "kind": "embed",
                    "layer": -1,
                    "pos": index - k,
                    "n": 0,
                    "act": None,
                    "influence": round(float(influence[index]), 5),
                }
            )
        else:
            nodes.append(
                {
                    "kind": "logit",
                    "layer": len(linearized.slots),
                    "pos": completion_start,
                    "n": -1,
                    "act": round(float(logprob), 6),
                    "influence": 1.0,
                    "other": round(float(other[index]), 4),
                    "target_kind": "completion_logprob",
                    "target_token_ids": ids[completion_start:],
                    "prediction_positions": list(range(completion_start - 1, length - 1)),
                }
            )
    edges = _serialize_edges(adjacency, edge_mask, remap)
    graph_config = dict(metadata.get("graph_config") or {})
    graph_config.setdefault("k", k)
    graph_config["effective_k"] = k
    graph_config.setdefault("max_edges", max_edges)
    graph_config.setdefault("batch", batch)
    graph_config.setdefault("node_threshold", node_threshold)
    graph_config.setdefault("edge_threshold", edge_threshold)
    graph_config.setdefault("target", "teacher_forced_completion_logprob")
    graph_config.setdefault("edge_semantics", "local_linear_half_rule_direct")
    graph_config.setdefault("target_transform", "differentiated_log_softmax")
    graph_config.setdefault("output_aggregation", "sum_over_completion_tokens")
    metadata = dict(metadata)
    metadata["graph_config"] = graph_config
    metadata["seed"] = seed
    first = float(token_logprobs[0])
    sequence_probability = float(torch.exp(logprob))
    return {
        "token": ids[completion_start],
        "prob": round(sequence_probability, 10),
        "target": {
            "kind": "completion_logprob",
            "token_ids": ids[completion_start:],
            "prediction_positions": list(range(completion_start - 1, length - 1)),
        },
        "completion_token_ids": ids[completion_start:],
        "prompt_token_ids": ids[:completion_start],
        "completion_logprob": round(float(logprob), 6),
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "candidates": k,
            "candidate_share": round(captured, 4),
            "kept_neurons": int(used[:k].sum()),
            "kept_edges": int(edge_mask.sum()),
            "completion_tokens": length - completion_start,
            "completion_logprob": round(float(logprob), 6),
            "first_token_logprob": round(first, 6),
            "first_token_probability": round(float(torch.exp(token_logprobs[0])), 10),
            "sequence_probability": round(sequence_probability, 10),
            "token_logprobs": [round(float(item), 6) for item in token_logprobs],
            "logit_from_candidates": (
                round(float(adjacency[k + length].sum()) / logit_value, 4) if logit_value else None
            ),
        },
        "metadata": metadata,
    }


def _token_logprobs(row: torch.Tensor, ids: list[int], completion_start: int) -> list[torch.Tensor]:
    values = []
    for position in range(completion_start - 1, len(ids) - 1):
        values.append(torch.log_softmax(row[position].float(), dim=-1)[ids[position + 1]])
    return values
