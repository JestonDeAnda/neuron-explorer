"""Raven/Huginn attribution graphs.

Qwen3.5 linearization does not apply to Raven. This package unrolls Huginn's
prelude, shared recurrent block, and coda, then attributes the teacher-forced
completion log-probability.
"""

from neuron_explorer.attribution.huginn.batch import graph_paths, plan_row, run_batch
from neuron_explorer.attribution.huginn.graph import build_huginn_graph
from neuron_explorer.attribution.huginn.protocol import encode_condition
from neuron_explorer.attribution.huginn.schedule import execution_schedule, shared_weight_id
from neuron_explorer.attribution.huginn.seed import row_seed

__all__ = [
    "build_huginn_graph",
    "encode_condition",
    "execution_schedule",
    "graph_paths",
    "plan_row",
    "row_seed",
    "run_batch",
    "shared_weight_id",
]
