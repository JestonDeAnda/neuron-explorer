import math
from dataclasses import FrozenInstanceError

import pytest
import torch

from neuron_explorer.attribution.moe import (
    MoEExecutionSlot,
    MoELinearizationBackend,
    RoutedNeuron,
    build_moe_graph,
)
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
    routes = backend.routing_snapshot
    assert routes.selected_experts.shape[-1] == top_k
    if normalization == "selected_renormalized":
        assert routes.routing_weights.sum(-1) == pytest.approx(torch.ones(4))
    else:
        assert torch.all((routes.routing_weights > 0) & (routes.routing_weights < 1))
    assert set(routes.selected_experts.flatten().tolist()) < set(range(num_experts))
    assert {neuron.expert_id for neuron in backend.routed_neurons()} == set(
        routes.selected_experts.flatten().tolist()
    )


def test_reference_backend_rebuilds_for_different_same_length_tokens():
    reused = sigmoid_top2_backend()
    reused.forward([1, 3, 5, 7])

    actual = reused.candidate_effects(2, [2, 4, 6, 8])
    fresh = sigmoid_top2_backend()
    fresh.forward([2, 4, 6, 8])
    expected = fresh.candidate_effects(2, [2, 4, 6, 8])

    assert actual == pytest.approx(expected)


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
    selected = set(concrete.routing_snapshot.selected_experts.flatten().tolist())
    for node in neurons:
        assert node["n"] == (
            node["expert_id"] * concrete.identity.expert_width + node["expert_neuron"]
        )
        assert node["physical_neuron"] == node["n"]
        assert node["expert_id"] in selected
        assert 0 <= node["router_rank"] < concrete.identity.top_k
        assert 0.0 < node["routing_weight"] <= 1.0
    for source, destination, weight in graph["edges"]:
        assert 0 <= source < len(graph["nodes"])
        assert 0 <= destination < len(graph["nodes"])
        assert math.isfinite(weight)

    assert graph["metadata"]["model_identity"] == concrete.identity.as_dict()
    assert graph["metadata"]["schedule"] == [
        slot.as_dict() for slot in concrete.schedule
    ]
    assert graph["metadata"]["graph_config"]["k"] == 4
    assert graph["metadata"]["graph_config"]["batch"] == 2
    assert graph["metadata"]["graph_config"]["max_edges"] == 20
