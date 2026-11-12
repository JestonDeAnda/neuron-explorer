# Development

## Setup and checks

```bash
uv sync --dev
uv run pytest
```

The package uses a `src/` layout. Reusable behavior belongs under
`src/neuron_explorer/`; `scripts/` contains executable entry points. Installed
commands are declared in `pyproject.toml`. Run command help without loading
optional GPU dependencies:

```bash
uv run python scripts/steer_sweep.py --help
uv run neuron-explorer --help
uv run neuron-plot-attribution --help
```

Generated data should stay under `runs/`, `stats/`, `captures/`, `cache/`, or
an explicitly configured root. See [Explorer](explorer.md) for path
environment variables, [sweeps](sweeps.md) for executable data formats, and
[attribution](attribution.md) for the stable graph APIs.

## MoE attribution tests

The MoE code has two layers with separate tests:

- `tests/attribution/test_moe_core.py` covers the runtime-neutral contract
  in `neuron_explorer.attribution.moe` with the eager reference backends and a
  synthetic two-million-entry lazy axis. That test fails if the graph builder
  iterates or slices the axis, requests direct contributions for every routed
  column, or keeps a previous direct chunk alive.
- `tests/attribution/test_moe_huginn.py` covers the xLLM schedule, runtime
  identity, mounts, linearization, graph wrapper, resume, manifest, benchmark,
  and smoke. Its tiny modules carry the public class names and attributes of
  the pinned xLLM classes, so `XLLMHuginnLinearized` patches them exactly as
  it patches the real runtime. Runner tests replace only `load_xllm_huginn`
  with a 25-expert, width-2432 tiny model and run on CPU.

```bash
uv run pytest tests/attribution/test_moe_core.py tests/attribution/test_moe_huginn.py -q
uv run python scripts/run_moe_huginn_attribution.py --help
uv run python scripts/moe_huginn_mount_check.py --help
```

Importing `neuron_explorer.attribution.moe` or its `xllm_huginn` package must
never import `xllm`; only `load_xllm_huginn` imports it, after verifying the
pinned source checkout. Several mount and source tests create temporary git
repositories, so they fail inside sandboxes that forbid `git init` outside the
workspace. A new MoE runtime implements `MoELinearizationBackend`, returns a
lazy routed axis, honors `direct_targets(..., sources=...)`, and gets its own
subpackage beside `xllm_huginn`; do not change `attribution/huginn/` defaults,
which belong to the dense version-2 Huginn-0125 graphs.

## `mirror.py`

Purpose: recursively mirror the public “Neural Networks Are Already Clean”
site and same-origin assets for research preservation.

```bash
uv run python scripts/mirror.py source/already-clean
```

The sole positional input is the output directory. The command starts at
`https://already-clean.goodfire.com/`, discovers same-origin links in HTML,
CSS, JavaScript, JSON, Markdown, and common asset references, and recreates URL
paths beneath the output directory. Binary files retain their downloaded
bytes; text files are not rewritten. `_fails.txt` contains one tab-separated
URL/error pair per failed fetch.

This command needs unrestricted public internet access and local disk space;
it uses only the Python standard library and 24 download threads. Existing
local files are not fetched again, so interrupted mirrors resume at file
granularity, although existing text is rescanned for links. `_fails.txt` is
rewritten after each complete crawl.

## `owl_delta_vs_base.py`

Purpose: compare selected tawny-owl neuron weight deltas with the corresponding
Qwen3-8B base-model rows and columns, downloading only required safetensor
byte ranges where possible.

```bash
uv run python scripts/owl_delta_vs_base.py \
  tawny-owl/neurons.safetensors \
  runs/owl_delta_vs_base.json
```

The input safetensors file must contain, for each represented layer,
`layers.<L>.neuron_index`, `gate_proj_rows_add`, `up_proj_rows_add`, and
`down_proj_columns_add`. The output is a JSON array with `layer`, `neuron`,
and `gate`, `up`, `down` objects. Each projection object reports base-weight
norm `w_norm`, delta norm `d_norm`, relative norm `rel`, cosine `cos`, and
parallel component `along`.

It requires NumPy, safetensors, internet access to
`Qwen/Qwen3-8B`, and enough RAM to hold each selected layer's full
down-projection matrix. GPU is not used. Requests and output are recomputed
from scratch; this script has no persistent download cache or resume behavior,
and the output file is overwritten after successful analysis.

## Documentation checks

`tests/test_documentation.py` requires every Python/shell executable under
`scripts/` to appear in the domain usage pages, resolves local Markdown links,
and rejects legacy executable paths in current docs, source docstrings, and
script comments:

```bash
uv run pytest tests/test_documentation.py -v
```
