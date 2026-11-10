"""Shared model and state-swap intervention helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from neuron_explorer.attribution import build_graph
from neuron_explorer.paths import ProjectPaths

DALLAS = "Fact: the capital of the state containing Dallas is"
OTHERS = {
    "San Francisco": "Fact: the capital of the state containing San Francisco is",
    "Los Angeles": "Fact: the capital of the state containing Los Angeles is",
    "Miami": "Fact: the capital of the state containing Miami is",
    "Chicago": "Fact: the capital of the state containing Chicago is",
}
CAPITALS = [
    " Austin",
    " Sacramento",
    " Tallahassee",
    " Springfield",
    " Houston",
    " Dallas",
    " Los",
    " San",
]


class Model:
    """Qwen model wrapper supporting position-specific MLP activation edits."""

    def __init__(self, model_name="Qwen/Qwen3.5-4B", dtype=torch.bfloat16):
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        self.dev = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.tok = AutoTokenizer.from_pretrained(model_name)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name, dtype=dtype
        )
        self.model = model.to(self.dev).eval()
        self.lm = self.model.model.language_model
        self.edits = {}
        self.capture = None
        for layer_index, layer in enumerate(self.lm.layers):
            layer.mlp.down_proj.register_forward_pre_hook(
                self._hook(layer_index)
            )

    def _hook(self, layer_index):
        def hook(module, inputs):
            del module
            hidden = inputs[0]
            if self.capture is not None:
                self.capture[layer_index] = hidden[0].float().cpu()
            layer_edits = self.edits.get(layer_index)
            if not layer_edits:
                return None
            hidden = hidden.clone()
            for position, (neurons, values) in layer_edits.items():
                hidden[
                    0,
                    position,
                    torch.tensor(neurons, device=hidden.device),
                ] = torch.tensor(
                    values, device=hidden.device, dtype=hidden.dtype
                )
            return (hidden,)

        return hook

    def ids(self, text):
        return self.tok(text).input_ids

    @torch.inference_mode()
    def run(self, text, edits=(), capture=False):
        """Return next-token log-probabilities and optional MLP activations."""
        self.edits, self.capture = {}, ({} if capture else None)
        for layer, position, neuron, value in edits:
            neurons, values = self.edits.setdefault(layer, {}).setdefault(
                position, ([], [])
            )
            neurons.append(neuron)
            values.append(value)
        try:
            output = self.model(
                input_ids=torch.tensor([self.ids(text)], device=self.dev),
                use_cache=False,
            )
            logprobs = output.logits[0, -1].float().log_softmax(-1).cpu()
            return logprobs, self.capture
        finally:
            self.edits, self.capture = {}, None

    def top(self, logprobs, k=8):
        values, indices = logprobs.topk(k)
        return [
            (
                self.tok.decode([int(token)]),
                round(float(value.exp()), 3),
            )
            for value, token in zip(values, indices)
        ]

    def prob(self, logprobs, word):
        token = self.tok(word, add_special_tokens=False).input_ids[0]
        return float(logprobs[token].exp())


def load_vocabulary(paths: ProjectPaths | None = None):
    """Load the explorer vocabulary cache."""
    cache = (paths or ProjectPaths.discover()).cache
    with (cache / "vocab.json").open() as handle:
        return json.load(handle)


vocab = load_vocabulary


def city_token_position(tokenizer, text: str, city: str) -> int:
    """Return the last occurrence of the city's final token in ``text``."""
    ids = tokenizer(text).input_ids
    city_ids = tokenizer(" " + city, add_special_tokens=False).input_ids
    if not city_ids:
        raise ValueError(f"city has no tokens: {city!r}")
    try:
        return max(index for index, token in enumerate(ids) if token == city_ids[-1])
    except ValueError as error:
        raise ValueError(f"city token not found in text: {city!r}") from error


def build_state_swap_edits(
    target_position: int,
    group: Iterable[tuple[int, int]],
    source_activations: Mapping[int, np.ndarray],
    source_position: int,
    scale: float = 1.0,
):
    """Build edits that copy a source state's neuron values to one position."""
    return [
        (
            layer,
            target_position,
            neuron,
            scale * float(source_activations[layer][source_position, neuron]),
        )
        for layer, neuron in group
    ]


def attribution_group(model, text, position, max_layer, k):
    """Rank graph neurons at one position by attribution influence."""
    graph = build_graph(
        model.lm, model.model.lm_head, model.ids(text), k=k
    )
    rows = [
        (node["layer"], node["n"], node["act"], node["influence"])
        for node in graph["nodes"]
        if node["kind"] == "neuron"
        and node["pos"] == position
        and node["layer"] <= max_layer
    ]
    return sorted(rows, key=lambda row: -row[3])


def annotate_intervention_graph(model, graph, ids, vocabulary, steer_index):
    """Add token text and steer-lens labels to an intervention graph."""
    graph["tokens"] = [model.tok.decode([token]) for token in ids]
    graph["target"] = model.tok.decode([graph["token"]])
    for node in graph["nodes"]:
        node["tok"] = graph["tokens"][node["pos"]]
        if node["kind"] == "neuron":
            pole = "up" if node["act"] > 0 else "down"
            node["lens"] = [
                vocabulary[token]
                for token in steer_index[pole][
                    node["layer"], node["n"], :5
                ]
            ]
    return graph


def build_state_swap_experiments(
    model,
    source_activations,
    source_position,
    texas: Sequence[tuple[int, int]],
    california: Sequence[tuple[int, int]],
):
    """Build the reusable core Dallas-to-California experiment matrix."""
    target_position = city_token_position(model.tok, DALLAS, "Dallas")

    def copied(group, scale):
        return build_state_swap_edits(
            target_position,
            group,
            source_activations,
            source_position,
            scale,
        )

    experiments = [("baseline", DALLAS, [])]
    for count in (5, 10, 20, 40):
        experiments.append(
            (
                f"Texas top {count} -> 0",
                DALLAS,
                [
                    (layer, target_position, neuron, 0.0)
                    for layer, neuron in texas[:count]
                ],
            )
        )
        for scale in (1, 2, 4):
            experiments.append(
                (
                    f"California top {count} -> {scale} x source",
                    DALLAS,
                    copied(california[:count], scale),
                )
            )
    return experiments
