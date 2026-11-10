import numpy as np

from neuron_explorer.sweep.interventions import (
    build_state_swap_edits,
    city_token_position,
)


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
