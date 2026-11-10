# Domain Package Refactor Design

## Goal

Convert the repository to a standard `src`-layout Python package organized by
domain, make attribution graphs an independently usable module, keep executable
entry points under `scripts/`, and move user and research documentation under
`docs/`.

This is a breaking migration. The old Python module paths and commands under
`explorer/` and `sweep/` will not be retained as compatibility wrappers.

## Package layout

```text
src/neuron_explorer/
  __init__.py
  paths.py
  attribution/
    __init__.py
    graph.py
    schema.py
    annotations.py
    catalogs.py
    render_svg.py
  explorer/
    __init__.py
    runtime.py
    server.py
    agent_mcp.py
    build_layout.py
    build_connections.py
    build_embeddings.py
    static/
  sweep/
    __init__.py
    neuron_stats.py
    steer_sweep.py
    steer_lens.py
    rank.py
    shift.py
    depth_report.py
    merge_runs.py
    capture_prefill.py
    swap_state.py
    swap_followup.py

scripts/
  serve_explorer.py
  plot_attribution.py
  plot_interventions.py
  arith_forms.py
  check_attribution.py
  check_capture_prefill.py
  check_equivalence.py
  bench.py
  peek.py
  mirror.py
  owl_delta_vs_base.py
  setup_pod.sh
  run_layer.sh
  run_depth.sh

docs/
  README.md
  explorer.md
  sweeps.md
  attribution.md
  interventions.md
  development.md
  research/
    attribution_memo.md
    research_log.md
    notes.md
  slides/
    attribution_slides.html
```

Code belongs in `src/neuron_explorer` when it exposes reusable functions,
classes, data structures, or domain logic. Files in `scripts` parse command-line
arguments, call package APIs, and report results; they do not contain reusable
model or graph logic.

The browser application is part of the explorer package and is shipped as
package data. Repository images used by the top-level README may remain at the
repository root.

## Attribution domain

The attribution package has no dependency on the explorer HTTP server, its
runtime object, or repository-local annotation files.

### Graph computation

`neuron_explorer.attribution.graph` owns `Linearized`, `build_graph`, and the
Qwen3.5 patched forward implementations. `build_graph` produces the numerical,
structured graph only. It does not add token strings, interpretations, names,
or steer-lens values.

The computation API continues to accept the language model, LM head, token IDs,
target token ID, pruning controls, and optional clamps. It returns a mapping
compatible with the existing graph JSON:

- graph metadata: target token ID, probability, and pruning statistics;
- nodes: neuron, embedding, and logit nodes;
- edges: source-node index, destination-node index, and signed attribution;
- intervention markers such as `clamped`.

### Schema and persistence

`neuron_explorer.attribution.schema` defines typed graph records and
`load_graph` / `save_graph`. JSON remains the canonical interchange format and
is written before SVG rendering. The schema retains existing field names so
the browser and existing saved results can consume migrated graphs.

Schema validation checks graph version, required fields, node kinds, node-index
bounds, and numeric edge weights. Newly written graphs include a schema version;
the loader accepts current unversioned graphs as version 1.

### Annotation catalogs

`neuron_explorer.attribution.annotations` defines the catalog protocol and the
operation that enriches an already computed graph. A catalog lookup is keyed
only by `(layer, neuron)` and may return:

```python
{
    "explanation": str | None,
    "steer_up": list[str],
    "steer_down": list[str],
    "kl_up": float | None,
    "kl_down": float | None,
    "provenance": dict | None,
}
```

`annotate_graph` receives token strings, target text, and an optional catalog.
It stores the explanation as the existing node `name`, chooses `steer_up` or
`steer_down` from the sign of the node's activation, and stores the selected
tokens as the existing node `lens`. It may also retain the complete annotation
record in an optional `annotation` field for programmatic analysis.

`neuron_explorer.attribution.catalogs` provides:

- `DatasetCatalog`, loaded from JSON or JSONL interpretation records;
- `CacheCatalog`, backed by the current names JSON, steer-index NPZ, optional
  logit-lens NPZ fallback, and vocabulary JSON;
- `CompositeCatalog`, resolving each missing field from ordered catalogs;
- `EmptyCatalog`, for graphs that should contain no interpretability metadata.

Dataset records accept explicit `layer` and `neuron` fields. Input validation
rejects duplicate keys unless the caller selects an explicit overwrite policy.
The selected catalog is passed when generating a graph or page; graph
computation itself is unchanged by catalog contents.

### Rendering

`neuron_explorer.attribution.render_svg` owns graph selection, layout, node and
edge tooltips, and SVG generation. It consumes only a graph mapping. A graph
without annotation metadata is valid and uses the layer/neuron identifier as
its label. When present, explanation text takes precedence over steer-lens
tokens for visible labels.

## Other domains

### Explorer

The explorer package owns the local FastAPI service, live model runtime, MCP
bridge, cache builders, and static web application. The server imports the
attribution package through its public API and uses a configured catalog to
annotate API responses.

The server remains local-only. Its behavior, endpoints, live steering,
activation reading, lens search, connection graph, and agent integration remain
functionally unchanged apart from import and resource paths.

### Sweep

Reusable sweep and analysis logic moves under `neuron_explorer.sweep`.
Hardware-facing commands remain thin scripts. Internal imports use absolute
package imports instead of modifying `sys.path`.

Experiment-specific report generators that combine model execution, graph
annotation, and presentation remain executable scripts, but reusable helpers
move into the appropriate attribution or sweep module.

## Paths and configuration

`neuron_explorer.paths` centralizes project data paths. It must not assume the
process current working directory. Defaults resolve in this order:

1. explicit function or CLI argument;
2. relevant environment variable;
3. repository root when running from a source checkout;
4. current working directory for externally installed package data outputs.

The primary configurable roots are runs, stats, captures, explorer cache, and
output. CLI commands expose the paths they consume or write. Shell scripts
derive the repository root from their own location and invoke package entry
points through the active environment.

Static files and prompt templates are accessed with `importlib.resources`.
Model and generated-data paths use `pathlib.Path`. No migrated module inserts
directories into `sys.path`.

## Commands

`pyproject.toml` uses setuptools with `package-dir = {"" = "src"}` and includes
explorer static assets and prompt templates as package data. Common commands
are exposed as project scripts where a stable user-facing command is useful;
files under `scripts/` remain directly runnable for research workflows.

All documented commands run through `uv run`, ensuring the package is imported
from the project environment. Examples include serving the explorer, creating
an attribution JSON/SVG pair, checking attribution exactness, generating
intervention reports, building sweep statistics, and launching a layer sweep.

## Documentation

The top-level `README.md` becomes a concise project overview and links to:

- `docs/explorer.md`: installation, server startup, UI features, model/cache
  requirements, and agent integration;
- `docs/sweeps.md`: every sweep, analysis, merge, inspection, and environment
  command with inputs and outputs;
- `docs/attribution.md`: Python API, CLI, catalog formats, graph JSON schema,
  SVG generation, and batch-analysis examples;
- `docs/interventions.md`: state-swap, follow-up controls, arithmetic forms,
  and intervention graph workflows;
- `docs/development.md`: package layout, adding modules and commands, tests,
  path configuration, and generated artifacts;
- `docs/research/`: existing research notes and attribution memo;
- `docs/slides/`: the existing attribution presentation.

Every executable script is listed in one of these documents with its purpose,
required inputs, command example, and outputs. All internal links and source
path references are updated to the new layout.

## Migration rules

The migration is performed with moves where possible so Git retains history.
Existing user modifications are preserved. Generated caches, model data, and
run outputs are not rewritten.

All Python imports, shell commands, doc references, frontend comments, and
prompt references are searched for old `explorer/` and `sweep/` paths and
updated. Old Python source files are removed after all consumers use the new
package. There are no compatibility forwarding modules.

## Testing and acceptance

Tests are added under `tests/` and cover:

- graph JSON validation and versioned/unversioned round trips;
- dataset, cache, empty, and composite catalog behavior;
- per-field fallback and duplicate-record handling;
- activation-sign selection of the steer-lens pole;
- annotation of graphs with and without catalog results;
- SVG rendering of raw and annotated graphs;
- path resolution independent of the current working directory;
- CLI generation from an existing graph JSON without loading a model;
- package resources for explorer static files and prompts.

The migrated mathematical attribution checks remain available as an executable
validation script. CPU-safe unit tests do not download Qwen3.5 or require CUDA.

Acceptance requires:

1. the package installs successfully with `uv sync`;
2. the unit test suite passes;
3. package imports work from outside the repository directory;
4. attribution JSON can be loaded, annotated from either catalog type, and
   rendered to SVG;
5. documented CLI help commands execute;
6. no Python module depends on the old package paths or `sys.path` mutations;
7. documentation describes every retained command and its inputs and outputs.
