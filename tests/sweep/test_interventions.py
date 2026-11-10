import numpy as np
import torch
from types import SimpleNamespace

from neuron_explorer.sweep.interventions import (
    DALLAS,
    build_state_swap_experiments,
    build_state_swap_edits,
    city_token_position,
)
from neuron_explorer.sweep import swap_state


class FakeTokenizer:
    def __init__(self):
        self.encoded = {
            "Fact: Dallas is in": [10, 20, 30, 20, 40],
            " Dallas": [20],
        }

    def __call__(self, text, add_special_tokens=True):
        del add_special_tokens
        return type("Tokens", (), {"input_ids": self.encoded[text]})()


def test_city_token_position_uses_last_matching_city_token():
    assert city_token_position(
        FakeTokenizer(), "Fact: Dallas is in", "Dallas"
    ) == 3


def test_build_state_swap_edits_reads_source_activations():
    activations = {
        2: np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        4: np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
    }

    assert build_state_swap_edits(
        target_position=6,
        group=[(2, 1), (4, 2)],
        source_activations=activations,
        source_position=1,
        scale=2.0,
    ) == [(2, 6, 1, 10.0), (4, 6, 2, 24.0)]


def test_explore_has_torch_available_for_activation_ranking(monkeypatch, tmp_path):
    class FakeModel:
        tok = type(
            "Tokenizer",
            (),
            {"decode": staticmethod(lambda ids: str(ids[0]))},
        )()

        def ids(self, text):
            del text
            return [0, 1]

        def run(self, text, capture=False):
            if not capture:
                return object(), None
            value = 1.0 if text == DALLAS else 2.0
            activations = {
                layer: torch.full((2, 8110), value) for layer in range(13)
            }
            return object(), activations

        def top(self, logprobs, k):
            del logprobs, k
            return []

    steer = {
        "up": np.zeros((13, 8110, 5), dtype=np.int8),
        "down": np.zeros((13, 8110, 5), dtype=np.int8),
    }
    monkeypatch.setattr(swap_state, "load_vocabulary", lambda paths: ["other"])
    monkeypatch.setattr(swap_state.np, "load", lambda path: steer)

    swap_state.explore(FakeModel(), SimpleNamespace(cache=tmp_path))


def test_state_swap_experiments_accept_attribution_rows():
    class ExperimentTokenizer:
        def __call__(self, text, add_special_tokens=True):
            del add_special_tokens
            encoded = [3, 7, 9] if text == DALLAS else [7]
            return type("Tokens", (), {"input_ids": encoded})()

    model = type("Model", (), {"tok": ExperimentTokenizer()})()
    activations = {
        2: np.array([[0.0, 1.5, 0.0]]),
        4: np.array([[0.0, 0.0, 2.5]]),
    }
    texas = [(2, 1, -0.7, 0.91)]
    california = [(4, 2, 2.5, 0.83)]

    experiments = build_state_swap_experiments(
        model, activations, 0, texas, california
    )
    pair_experiments = build_state_swap_experiments(
        model, activations, 0, [(2, 1)], [(4, 2)]
    )

    assert pair_experiments == experiments
    assert (
        "Texas top 5 -> 0",
        DALLAS,
        [(2, 1, 1, 0.0)],
    ) in experiments
    assert (
        "California top 5 -> 2 x source",
        DALLAS,
        [(4, 1, 2, 5.0)],
    ) in experiments
    assert (
        "Texas top 5 -> 0, California top 5 -> 2 x",
        DALLAS,
        [(2, 1, 1, 0.0), (4, 1, 2, 5.0)],
    ) in experiments
