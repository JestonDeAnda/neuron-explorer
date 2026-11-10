from importlib.resources import files

import pytest


@pytest.mark.parametrize(
    "relative_path",
    [
        "static/index.html",
        "static/app.js",
        "prompts/agent_prompt.md",
        "prompts/agent_methodology.md",
    ],
)
def test_explorer_resources_are_packaged(relative_path):
    resource = files("neuron_explorer.explorer").joinpath(relative_path)

    assert resource.read_text(encoding="utf-8")
