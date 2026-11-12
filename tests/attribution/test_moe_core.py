import math
from dataclasses import FrozenInstanceError

import pytest
import torch

from neuron_explorer.attribution.moe import (
    MoEExecutionSlot,
    MoELinearizationBackend,
    MoEModelIdentity,
    MoESourceLeaf,
    RoutedNeuron,
    RoutingSnapshot,
    build_moe_graph,
)
from neuron_explorer.attribution.moe.graph import _edge_mask
from neuron_explorer.attribution.moe.reference import (
    softmax_top1_backend,
    sigmoid_top2_backend,
)


def test_routed_neuron_has_global_and_shared_parameter_identity():
    slot = MoEExecutionSlot(
        virtual_depth=3,
        stage="recurrent",
        recurrence_iteration=1,
        physical_layer=7,
        physical_block=0,
    )
    neuron = RoutedNeuron(
        slot=slot,
        position=4,
        expert_id=2,
        expert_neuron=5,
        expert_width=16,
        router_rank=1,
        routing_weight=0.25,
    )

    assert neuron.serialized_neuron == 37
    assert neuron.shared_weight_id == "recurrent:0:e2:n5"
    with pytest.raises(FrozenInstanceError):
        neuron.expert_id = 3


def test_shared_neuron_has_no_route_and_distinguishes_runtime_expert_id():
    slot = MoEExecutionSlot(0, "body", None, 4, 2)
    neuron = RoutedNeuron(
        slot=slot,
        position=1,
        expert_id=11,
        runtime_expert_id=2,
        expert_neuron=3,
        expert_width=8,
        expert_kind="shared",
        router_rank=None,
        routing_weight=None,
    )

    assert neuron.serialized_neuron == 91
    assert neuron.runtime_expert_id == 2
    assert neuron.router_rank is None
    assert neuron.routing_weight is None


def test_routing_snapshot_supports_variable_selected_counts():
    snapshot = RoutingSnapshot(
        selected_experts=torch.tensor([[[1, 3], [2, -1]]]),
        routing_weights=torch.tensor([[[0.6, 0.4], [0.8, 0.0]]]),
        selected_expert_counts=torch.tensor([[2, 1]]),
        router_activation="sigmoid",
        router_normalization="selected_renormalized",
    )

    assert snapshot.routes_at(0, 0) == ((1, 0.6), (3, 0.4))
    assert snapshot.routes_at(0, 1) == ((2, 0.8),)


def test_model_identity_declares_placement_and_is_hashable():
    identity = MoEModelIdentity(
        model_family="test",
        model_name="test",
        num_experts=2,
        expert_width=4,
        top_k=2,
        router_activation="sigmoid",
        router_normalization="selected_renormalized",
        device_placement="single_device",
        expert_parallel=True,
        expert_parallel_capable=True,
        auxiliary=(("runtime", "test"),),
    )

    assert hash(identity)
    assert identity.as_dict()["device_placement"] == "single_device"
    assert identity.as_dict()["expert_parallel_capable"] is True


@pytest.mark.parametrize(
    ("factory", "num_experts", "top_k", "activation", "normalization"),
    [
        (sigmoid_top2_backend, 4, 2, "sigmoid", "selected_renormalized"),
        (softmax_top1_backend, 3, 1, "softmax", "all_experts"),
    ],
)
def test_reference_backends_obey_runtime_neutral_contract(
    factory, num_experts, top_k, activation, normalization
):
    backend = factory()
    backend.forward([1, 3, 5, 7], mode="total")

    assert isinstance(backend, MoELinearizationBackend)
    assert backend.identity.num_experts == num_experts
    assert backend.identity.router_activation == activation
    assert backend.identity.router_normalization == normalization
    assert backend.identity.device_placement == "cpu"
    assert backend.identity.expert_parallel_capable is False
    routes = backend.routing_snapshot
    assert routes.selected_experts.shape[-1] == top_k
    if normalization == "selected_renormalized":
        assert routes.routing_weights.sum(-1) == pytest.approx(
            torch.ones_like(routes.routing_weights.sum(-1))
        )
    else:
        assert torch.all((routes.routing_weights > 0) & (routes.routing_weights < 1))
    selected = routes.selected_experts[routes.selected_experts >= 0]
    assert len({tuple(row.tolist()) for row in routes.selected_experts[0]}) > 1
    assert set(selected.flatten().tolist()) <= set(range(num_experts))
    assert {neuron.expert_id for neuron in backend.routed_neurons()} == set(
        selected.flatten().tolist()
    )
    assert routes.selected_expert_counts.tolist() == [[top_k] * 4]
    if activation == "softmax":
        expected = backend.full_router_probabilities[
            0, torch.arange(4), routes.selected_experts[0, :, 0]
        ]
        assert routes.routing_weights[0, :, 0] == pytest.approx(expected)


def test_reference_backend_rebuilds_for_different_same_length_tokens():
    reused = sigmoid_top2_backend()
    reused.forward([1, 3, 5, 7])

    actual = reused.candidate_effects(2, [2, 4, 6, 8])
    fresh = sigmoid_top2_backend()
    fresh.forward([2, 4, 6, 8])
    expected = fresh.candidate_effects(2, [2, 4, 6, 8])

    assert actual == pytest.approx(expected)


def test_neuron_direct_target_uses_upstream_sources_not_itself():
    backend = sigmoid_top2_backend()
    token_ids = [1, 3, 5, 7]
    backend.forward(token_ids)

    contribution = backend.direct_targets([0], 2, token_ids)

    assert contribution.routed[0, 0] == 0
    assert contribution.targets[0] == pytest.approx(
        backend.activations([backend.routed_neurons()[0]])[0]
    )
    assert contribution.token_embeddings[0].abs().sum() > 0
    assert contribution.source_leaves == (MoESourceLeaf("initial_state"),)
    assert contribution.source_contributions.shape == (1, 1)


class _RuntimeNeutralView:
    """Expose exactly the protocol surface and no router implementation details."""

    def __init__(self, backend):
        self.identity = backend.identity
        self.schedule = backend.schedule
        self._backend = backend

    def forward(self, token_ids, *, copies=1, replay=False, mode="direct"):
        return self._backend.forward(token_ids, copies=copies, replay=replay, mode=mode)

    def routed_neurons(self):
        return self._backend.routed_neurons()

    def candidate_effects(self, completion_start, token_ids):
        return self._backend.candidate_effects(completion_start, token_ids)

    def direct_targets(self, targets, completion_start, token_ids):
        return self._backend.direct_targets(targets, completion_start, token_ids)

    def activations(self, neurons):
        return self._backend.activations(neurons)

    def completion_logprob(self, row, completion_start, token_ids):
        return self._backend.completion_logprob(row, completion_start, token_ids)


@pytest.mark.parametrize("factory", [sigmoid_top2_backend, softmax_top1_backend])
def test_generic_graph_serializes_only_selected_routes(factory):
    concrete = factory()
    backend = _RuntimeNeutralView(concrete)
    token_ids = [1, 3, 5, 7]

    graph = build_moe_graph(
        backend,
        token_ids,
        completion_start=2,
        k=4,
        batch=2,
        max_edges=20,
    )

    neurons = [node for node in graph["nodes"] if node["kind"] == "neuron"]
    assert neurons
    assert len(graph["edges"]) <= 20
    for node in neurons:
        slot_index = next(
            index
            for index, slot in enumerate(concrete.schedule)
            if slot.virtual_depth == node["virtual_depth"]
        )
        expected_routes = concrete.routing_snapshot.routes_at(slot_index, node["pos"])
        expected_expert, expected_weight = expected_routes[node["router_rank"]]
        assert node["n"] == (
            node["expert_id"] * concrete.identity.expert_width + node["expert_neuron"]
        )
        assert node["physical_neuron"] == node["n"]
        assert node["expert_id"] == expected_expert
        assert node["runtime_expert_id"] == expected_expert
        assert node["routing_weight"] == pytest.approx(expected_weight)
        assert 0 <= node["router_rank"] < concrete.identity.top_k
        assert 0.0 < node["routing_weight"] <= 1.0
    for source, destination, weight in graph["edges"]:
        assert 0 <= source < len(graph["nodes"])
        assert 0 <= destination < len(graph["nodes"])
        assert math.isfinite(weight)
        assert source != destination
        source_node = graph["nodes"][source]
        destination_node = graph["nodes"][destination]
        assert source_node["kind"] != "logit"
        assert destination_node["kind"] != "embed"
        if source_node["kind"] == "neuron":
            assert destination_node["kind"] == "logit" or (
                source_node["virtual_depth"] < destination_node["virtual_depth"]
            )

    assert graph["metadata"]["model_identity"] == concrete.identity.as_dict()
    assert graph["metadata"]["schedule"] == [
        slot.as_dict() for slot in concrete.schedule
    ]
    assert graph["metadata"]["graph_config"]["k"] == 4
    assert graph["metadata"]["graph_config"]["batch"] == 2
    assert graph["metadata"]["graph_config"]["max_edges"] == 20
    assert (
        graph["metadata"]["graph_config"]["edge_semantics"]
        == concrete.identity.direct_edge_semantics
    )
    assert any(
        graph["nodes"][source]["kind"] == "embed"
        and graph["nodes"][destination]["kind"] == "neuron"
        for source, destination, _ in graph["edges"]
    )
    assert graph["metadata"]["source_leaves"] == [
        {"name": "initial_state", "kind": "runtime", "metadata": {}}
    ]


def test_edge_mask_enforces_strict_cap_under_ties():
    mask = _edge_mask(torch.ones(3, 3), edge_threshold=1.0, max_edges=2)

    assert int(mask.sum()) == 2
