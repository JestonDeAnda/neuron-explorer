"""Interactive neuron explorer domain."""

from typing import Any


def __getattr__(name: str) -> Any:
    """Load model-backed runtime code only when requested."""
    if name == "Runtime":
        from neuron_explorer.explorer.runtime import Runtime

        return Runtime
    raise AttributeError(name)

__all__ = ["Runtime"]
