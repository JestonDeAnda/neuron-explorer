"""Follow-up controls for the Dallas-to-California state swap."""

import json

import torch

from neuron_explorer.paths import ProjectPaths
from neuron_explorer.sweep.interventions import (
    DALLAS,
    OTHERS,
    Model,
    build_state_swap_edits,
    city_token_position,
)


def show(model, text, edits, words, label):
    logprobs, _ = model.run(text, edits)
    probabilities = " ".join(
        f"{word.strip()} {model.prob(logprobs, word):.3f}"
        for word in words
    )
    print(f"  {label:38s}", probabilities, "|", model.top(logprobs, 4))


@torch.inference_mode()
def greedy(model, text, edits, n=12):
    ids = model.ids(text)
    for _ in range(n):
        model.edits = {}
        for layer, position, neuron, value in edits:
            neurons, values = model.edits.setdefault(layer, {}).setdefault(
                position, ([], [])
            )
            neurons.append(neuron)
            values.append(value)
        logits = model.model(
            input_ids=torch.tensor([ids], device=model.dev), use_cache=False
        ).logits[0, -1]
        ids.append(int(logits.argmax()))
    model.edits = {}
    return model.tok.decode(ids[len(model.ids(text)) :])


def run_followups(model, california, source_activations, source_position):
    def edits(text, city, group, scale):
        return build_state_swap_edits(
            city_token_position(model.tok, text, city),
            group,
            source_activations,
            source_position,
            scale,
        )

    dallas_edits = edits(DALLAS, "Dallas", california[:5], 2)
    print("== California-group neurons at ' Dallas'")
    show(
        model,
        DALLAS,
        dallas_edits,
        [" Austin", " Sacramento"],
        "top 5 together",
    )
    print("\n== does Dallas now sit in California?")
    probes = [
        ("Fact: Dallas is a city in the state of", [" Texas", " California"]),
        ("Q: Which state is Dallas in?\nA: Dallas is in", [" Texas", " California"]),
        (
            "Fact: the governor of the state containing Dallas is",
            [" Greg", " Gavin"],
        ),
        (
            "Fact: the largest city in the state containing Dallas is",
            [" Houston", " Los"],
        ),
    ]
    for text, words in probes:
        swapped = edits(text, "Dallas", california[:5], 2)
        show(model, text, [], words, "baseline: " + text[:30])
        show(model, text, swapped, words, "swapped")
        print("    greedy, swapped:", repr(greedy(model, text, swapped)))

    print("\n== other cities, the same edit at the city token")
    cities = [
        ("Chicago", OTHERS["Chicago"]),
        ("Miami", OTHERS["Miami"]),
        ("Seattle", "Fact: the capital of the state containing Seattle is"),
        ("Boston", "Fact: the capital of the state containing Boston is"),
    ]
    for city, text in cities:
        swapped = edits(text, city, california[:5], 2)
        show(model, text, [], [" Sacramento"], f"{city} baseline")
        show(model, text, swapped, [" Sacramento"], f"{city} swapped")


def main(argv=None):
    del argv
    paths = ProjectPaths.discover()
    with (paths.runs / "attribution" / "swap_state.json").open() as handle:
        saved = json.load(handle)
    california = [tuple(row[:2]) for row in saved["california"]]
    model = Model()
    source = OTHERS["San Francisco"]
    source_position = len(model.ids(source)) - 2
    _, source_activations = model.run(source, capture=True)
    run_followups(model, california, source_activations, source_position)


if __name__ == "__main__":
    main()
