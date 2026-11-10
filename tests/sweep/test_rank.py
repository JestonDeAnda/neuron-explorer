from neuron_explorer.sweep.rank import compute_units
from neuron_explorer.sweep.shift import shift_scores


def _rows():
    common = {"layer": 2, "prompt": "p1", "dose": 0.5, "rel_push": 0.5}
    return [
        {
            **common,
            "kind": "baseline",
            "neuron": -1,
            "ref_neuron": None,
            "dose": 0.0,
            "rel_push": 0.0,
            "text": "baseline common words",
            "token_ids": [1, 2, 3, 4, 5],
            "finish": "stop",
            "logprob": [-0.1],
        },
        {
            **common,
            "kind": "neuron",
            "neuron": 7,
            "ref_neuron": 7,
            "text": "amber amber novel common",
            "token_ids": [1, 2, 3, 4, 1],
            "finish": "stop",
            "logprob": [-0.2, -0.4],
        },
        {
            **common,
            "kind": "neuron",
            "neuron": 7,
            "ref_neuron": 7,
            "prompt": "p2",
            "rel_push": 0.7,
            "text": "amber novel bright common",
            "token_ids": [1, 2, 3, 4, 5],
            "finish": "length",
            "logprob": [],
        },
        {
            **common,
            "kind": "random",
            "neuron": 0,
            "ref_neuron": 3,
            "text": "cobalt novel common",
            "token_ids": [1, 1, 1, 1, 1],
            "finish": "stop",
            "logprob": [-0.5],
        },
        {
            **common,
            "kind": "random",
            "neuron": 0,
            "ref_neuron": 3,
            "prompt": "p2",
            "rel_push": 0.7,
            "text": "cobalt plain common",
            "token_ids": [1, 2, 1, 2, 1],
            "finish": "stop",
            "logprob": [-0.3],
        },
        {
            **common,
            "kind": "random",
            "neuron": 1,
            "ref_neuron": 4,
            "text": "violet novel common",
            "token_ids": [1, 2, 3, 4],
            "finish": "stop",
            "logprob": [-0.1],
        },
        {
            **common,
            "kind": "random",
            "neuron": 1,
            "ref_neuron": 4,
            "prompt": "p2",
            "rel_push": 0.7,
            "text": "violet plain common",
            "token_ids": [1, 2, 3, 4, 5, 6],
            "finish": "length",
            "logprob": [-0.2],
        },
    ]


def test_compute_units_matches_recorded_legacy_output():
    records = compute_units(_rows())

    assert records == [
        {
            "kind": "neuron",
            "neuron": 7,
            "layer": 2,
            "ref_neuron": 7,
            "z": 1.234,
            "mass": 1.727,
            "words": [["amber", 2, 1.23], ["novel", 2, 0.49]],
            "per_prompt": {"amber": ["p1", "p2"], "novel": ["p1", "p2"]},
            "dose": {
                "+0.5": {
                    "rel_push": 0.6,
                    "distinct4": 1.0,
                    "capped": 0.5,
                    "logprob": -0.15,
                }
            },
            "null_pct": 1.0,
        },
        {
            "kind": "random",
            "neuron": 0,
            "layer": 2,
            "ref_neuron": 3,
            "z": 1.234,
            "mass": 1.234,
            "words": [["cobalt", 2, 1.23]],
            "per_prompt": {"cobalt": ["p1", "p2"]},
            "dose": {
                "+0.5": {
                    "rel_push": 0.6,
                    "distinct4": 0.75,
                    "capped": 0.0,
                    "logprob": -0.4,
                }
            },
            "null_pct": 0.0,
        },
        {
            "kind": "random",
            "neuron": 1,
            "layer": 2,
            "ref_neuron": 4,
            "z": 1.234,
            "mass": 1.234,
            "words": [["violet", 2, 1.23]],
            "per_prompt": {"violet": ["p1", "p2"]},
            "dose": {
                "+0.5": {
                    "rel_push": 0.6,
                    "distinct4": 1.0,
                    "capped": 0.5,
                    "logprob": -0.15,
                }
            },
            "null_pct": 0.0,
        },
    ]


def test_shift_scores_matches_recorded_legacy_output():
    scores = shift_scores(_rows())

    assert scores == {
        ("neuron", 7): {
            "shift": 6.11,
            "up": [("amber", 2.83), ("novel", 0.82)],
            "down": [("cobalt", -0.82), ("plain", -0.82), ("violet", -0.82)],
        },
        ("random", 0): {
            "shift": 4.09,
            "up": [("cobalt", 2.83)],
            "down": [("violet", -1.26)],
        },
        ("random", 1): {
            "shift": 4.09,
            "up": [("violet", 2.83)],
            "down": [("cobalt", -1.26)],
        },
    }
