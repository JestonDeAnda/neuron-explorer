"""Memory-based batch settings for one Huginn GPU."""

from __future__ import annotations


def choose_edge_batch(peak_bytes: int, total_bytes: int) -> dict[str, int]:
    """Choose direct-edge copies and equal-length graph batching.

    The measured peak is one graph at edge batch 1, including weights. Up to
    80% of device memory and at most four such slots are used. Four slots are
    split into edge batch 2 and graph concurrency 2 so both stay inside the
    budget. The product is the memory multiplier. A second model copy is never
    requested.
    """
    if peak_bytes <= 0 or total_bytes <= 0:
        slots = 1
    else:
        budget = int(total_bytes * 0.80)
        slots = max(1, min(4, budget // peak_bytes))
    if slots >= 4:
        edge_batch, graph_concurrency = 2, 2
    else:
        edge_batch, graph_concurrency = slots, 1
    return {
        "edge_batch": edge_batch,
        "graph_concurrency": graph_concurrency,
        "model_copies": 1,
    }
