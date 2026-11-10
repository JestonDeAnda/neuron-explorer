# Domain Package Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the repository to a documented, domain-oriented `src` package and expose attribution graph computation, annotation, persistence, and rendering as independent APIs.

**Architecture:** Reusable code lives in `src/neuron_explorer/{attribution,explorer,sweep}` and all executable research workflows live in `scripts`. Attribution computation returns annotation-free structured data; catalogs enrich the data, and renderers consume it. A shared path configuration object replaces current-working-directory assumptions and `sys.path` mutations.

**Tech Stack:** Python 3.11+, setuptools `src` layout, PyTorch, Transformers, NumPy, FastAPI, pytest.

## Global Constraints

- This is a breaking migration; do not add forwarding modules under the old `explorer/` or `sweep/` paths.
- Preserve all existing user modifications and generated data.
- Do not download Qwen3.5 or require CUDA in the unit test suite.
- Use absolute package imports; no migrated Python module may mutate `sys.path`.
- JSON is the canonical attribution graph interchange format and must be written before SVG.
- Every retained executable must be documented with purpose, inputs, example command, and outputs.

---

### Task 1: Establish the package and path foundation

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `src/neuron_explorer/__init__.py`
- Create: `src/neuron_explorer/paths.py`
- Create: `tests/test_paths.py`

**Interfaces:**
- Produces: `ProjectPaths.discover(root: Path | None = None) -> ProjectPaths`
- Produces: `ProjectPaths.runs`, `.stats`, `.captures`, `.cache`, and `.output`
- Produces: environment overrides `NEURON_EXPLORER_ROOT`, `NEURON_EXPLORER_RUNS`, `NEURON_EXPLORER_STATS`, `NEURON_EXPLORER_CAPTURES`, and `NEURON_EXPLORER_CACHE`

- [ ] **Step 1: Add failing path-resolution tests**

```python
def test_explicit_root_does_not_depend_on_cwd(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    elsewhere = tmp_path / "elsewhere"
    root.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    paths = ProjectPaths.discover(root)
    assert paths.runs == root / "runs"
    assert paths.cache == root / "cache"

def test_environment_overrides_one_data_directory(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    custom = tmp_path / "custom-runs"
    root.mkdir()
    monkeypatch.setenv("NEURON_EXPLORER_RUNS", str(custom))
    assert ProjectPaths.discover(root).runs == custom
```

- [ ] **Step 2: Run the tests and verify the import fails**

Run: `uv run pytest tests/test_paths.py -v`

Expected: FAIL because `neuron_explorer.paths` does not exist.

- [ ] **Step 3: Configure setuptools and implement the immutable path object**

Add setuptools build metadata, `package-dir = {"" = "src"}`, package discovery,
package data for explorer static assets and prompt templates, and a pytest dev
dependency. Implement `ProjectPaths` with `pathlib.Path`, explicit/environment
override precedence, source-checkout detection, and installed-package fallback.
Do not create directories during discovery.

- [ ] **Step 4: Run the focused tests**

Run: `uv sync && uv run pytest tests/test_paths.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the foundation**

```bash
git add pyproject.toml uv.lock .gitignore src/neuron_explorer tests/test_paths.py
git commit -m "build: establish src package and shared paths"
```

---

### Task 2: Add structured attribution schema and annotation catalogs

**Files:**
- Create: `src/neuron_explorer/attribution/__init__.py`
- Create: `src/neuron_explorer/attribution/schema.py`
- Create: `src/neuron_explorer/attribution/annotations.py`
- Create: `src/neuron_explorer/attribution/catalogs.py`
- Create: `tests/attribution/test_schema.py`
- Create: `tests/attribution/test_catalogs.py`
- Create: `tests/attribution/test_annotations.py`

**Interfaces:**
- Produces: `load_graph(path: Path) -> dict`
- Produces: `save_graph(graph: Mapping, path: Path) -> Path`
- Produces: `validate_graph(graph: Mapping) -> None`
- Produces: `NeuronAnnotation`, `NeuronCatalog.lookup(layer: int, neuron: int)`
- Produces: `DatasetCatalog.from_path(path, duplicate="error")`
- Produces: `CacheCatalog(names_path, steer_path, vocab_path, logit_lens_path=None)`
- Produces: `CompositeCatalog(*catalogs)` and `EmptyCatalog`
- Produces: `annotate_graph(graph, *, tokens, target, catalog, lens_tokens=5) -> dict`

- [ ] **Step 1: Write schema tests for versioned and existing unversioned graphs**

Tests must prove that `save_graph` adds `schema_version: 1`, that `load_graph`
accepts an unversioned current-format fixture, and that validation rejects an
edge whose node index is out of bounds.

- [ ] **Step 2: Verify the schema tests fail**

Run: `uv run pytest tests/attribution/test_schema.py -v`

Expected: FAIL because the schema module is absent.

- [ ] **Step 3: Implement JSON persistence and validation**

Preserve existing node and edge field names. Use UTF-8 and
`ensure_ascii=False`. Write through a temporary sibling and replace the target
so interrupted writes do not leave partial JSON.

- [ ] **Step 4: Write catalog tests**

Use tiny JSON and JSONL fixtures to test:

```python
record = {
    "layer": 5,
    "neuron": 8109,
    "explanation": "Texas state",
    "steer_up": [" Texas"],
    "steer_down": [" California"],
}
```

Cover JSON arrays, JSONL, duplicate rejection, explicit `duplicate="last"`,
NPZ/vocabulary-backed cache lookup, logit-lens fallback, empty lookup, and
per-field composite fallback.

- [ ] **Step 5: Verify catalog tests fail, then implement the catalog classes**

Run before implementation: `uv run pytest tests/attribution/test_catalogs.py -v`

Expected: FAIL because catalog classes are absent.

Implement immutable annotation values and normalize aliases `name` to
`explanation`, `n` to `neuron`, and token sequences to lists of strings.
`CompositeCatalog` fills each missing field in priority order rather than
selecting one whole record.

- [ ] **Step 6: Write annotation tests**

Prove that positive activation selects `steer_up`, negative activation selects
`steer_down`, missing records leave a valid raw node, input graphs are not
mutated, and complete source metadata is retained under `annotation`.

- [ ] **Step 7: Verify failure, implement `annotate_graph`, and rerun**

Run: `uv run pytest tests/attribution -v`

Expected after implementation: PASS.

- [ ] **Step 8: Commit the structured graph metadata layer**

```bash
git add src/neuron_explorer/attribution tests/attribution
git commit -m "feat: add attribution schemas and annotation catalogs"
```

---

### Task 3: Move attribution computation and SVG rendering into the package

**Files:**
- Move: `explorer/attribution.py` → `src/neuron_explorer/attribution/graph.py`
- Create: `src/neuron_explorer/attribution/render_svg.py`
- Create: `tests/attribution/test_render_svg.py`
- Create: `tests/attribution/test_public_api.py`
- Modify: `src/neuron_explorer/attribution/__init__.py`

**Interfaces:**
- Produces: `build_graph`, `Linearized`
- Produces: `select_graph`, `layout_graph`, `render_svg`
- Public imports: `from neuron_explorer.attribution import build_graph, annotate_graph, load_graph, save_graph, render_svg`

- [ ] **Step 1: Add public API and renderer characterization tests**

Build a minimal graph fixture containing embedding, neuron, and logit nodes.
Assert that raw and annotated graphs both render, signed edges use distinct
colors, explanation text has priority over lens text, and tooltips contain the
layer/neuron/position identifier.

- [ ] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/attribution/test_render_svg.py tests/attribution/test_public_api.py -v`

Expected: FAIL because renderer and public graph imports are absent.

- [ ] **Step 3: Move computation without changing its numerical behavior**

Use a filesystem move, update only package imports, and retain the existing
`Linearized`, pruning, clamping, and patched-forward implementation. Export the
documented graph API from `attribution/__init__.py`.

- [ ] **Step 4: Extract rendering from `sweep/plot_attribution.py`**

Move `esc`, `select`, `layout`, `svg`, node naming, tooltips, and collision
helpers into `render_svg.py`. Rename public functions to the interfaces above;
keep private helpers private. Rendering must not import the explorer runtime.

- [ ] **Step 5: Run attribution unit tests**

Run: `uv run pytest tests/attribution -v`

Expected: PASS.

- [ ] **Step 6: Commit attribution package extraction**

```bash
git add src/neuron_explorer/attribution tests/attribution explorer/attribution.py
git commit -m "refactor: extract attribution graph package"
```

---

### Task 4: Migrate the explorer domain and package resources

**Files:**
- Move: `explorer/runtime.py` → `src/neuron_explorer/explorer/runtime.py`
- Move: `explorer/server.py` → `src/neuron_explorer/explorer/server.py`
- Move: `explorer/agent_mcp.py` → `src/neuron_explorer/explorer/agent_mcp.py`
- Move: `explorer/build_layout.py` → `src/neuron_explorer/explorer/build_layout.py`
- Move: `explorer/build_connections.py` → `src/neuron_explorer/explorer/build_connections.py`
- Move: `explorer/build_embeddings.py` → `src/neuron_explorer/explorer/build_embeddings.py`
- Move: `explorer/static/` → `src/neuron_explorer/explorer/static/`
- Move: `explorer/agent_prompt.md` → `src/neuron_explorer/explorer/prompts/agent_prompt.md`
- Move: `explorer/agent_methodology.md` → `src/neuron_explorer/explorer/prompts/agent_methodology.md`
- Create: `src/neuron_explorer/explorer/__init__.py`
- Create: `tests/explorer/test_resources.py`
- Create: `tests/explorer/test_runtime_annotation.py`

**Interfaces:**
- Produces: `Runtime(paths: ProjectPaths | None = None)`
- Produces: `create_app(paths: ProjectPaths | None = None) -> FastAPI`
- Consumes: attribution public APIs and catalog abstractions

- [ ] **Step 1: Add failing package-resource tests**

Assert `importlib.resources.files("neuron_explorer.explorer")` can read
`static/index.html`, `static/app.js`, `prompts/agent_prompt.md`, and
`prompts/agent_methodology.md`.

- [ ] **Step 2: Add a model-free runtime annotation test**

Extract a small helper that annotates a returned graph using an injected
catalog and tokenizer decoder. Test that it works without constructing
`Runtime` or loading model weights.

- [ ] **Step 3: Verify both tests fail**

Run: `uv run pytest tests/explorer -v`

Expected: FAIL because explorer package resources and helper are absent.

- [ ] **Step 4: Move explorer files and replace path globals**

Pass `ProjectPaths` into runtime/server construction. Replace cache, runs,
stats, names, upload, and agent-work paths with the shared path object.
Resolve static and bundled prompt files through `importlib.resources`.
Represent per-machine local prompt additions as a cache/config path, not a
package resource.

- [ ] **Step 5: Replace attribution enrichment in `Runtime.attribution`**

The runtime calls `build_graph`, decodes token strings, and calls
`annotate_graph` with a configured catalog. The server no longer loops over
nodes to add names after `/api/attribution`; its catalog includes the names
source. Keep the existing API response field names for the frontend.

- [ ] **Step 6: Update MCP startup and static mounting**

Start the MCP bridge with `python -m neuron_explorer.explorer.agent_mcp`.
Materialize traversable package resources only where FastAPI requires a real
directory. Preserve local-only host/origin checks.

- [ ] **Step 7: Run explorer tests and import smoke tests**

Run:

```bash
EXPLORER_NO_MODEL=1 uv run pytest tests/explorer -v
EXPLORER_NO_MODEL=1 uv run python -c "from neuron_explorer.explorer.server import app; print(app.title)"
```

Expected: tests pass and the import exits without model download.

- [ ] **Step 8: Commit explorer migration**

```bash
git add src/neuron_explorer/explorer tests/explorer explorer
git commit -m "refactor: migrate explorer into domain package"
```

---

### Task 5: Migrate reusable sweep and intervention logic

**Files:**
- Move reusable modules from `sweep/` to `src/neuron_explorer/sweep/`
- Create: `src/neuron_explorer/sweep/__init__.py`
- Create: `src/neuron_explorer/sweep/interventions.py`
- Create: `tests/sweep/test_rank.py`
- Create: `tests/sweep/test_interventions.py`
- Modify imports throughout `src/neuron_explorer/explorer/server.py`

**Interfaces:**
- Preserves: `compute_units`, `shift_scores`, rollout/statistics helpers
- Produces: shared intervention `Model`, prompts, vocabulary loading, graph
  annotation helpers, and state-swap experiment builders

- [ ] **Step 1: Add characterization tests for CPU-safe sweep logic**

Use tiny in-memory rollout rows to pin `compute_units` and `shift_scores`.
Test city-token lookup and intervention edit construction with a fake tokenizer
and arrays. Import the desired APIs from `neuron_explorer.sweep`; do not
instantiate Qwen or vLLM.

- [ ] **Step 2: Run the tests and verify the desired imports fail**

Run: `uv run pytest tests/sweep -v`

Expected: FAIL because `neuron_explorer.sweep` has not been created. Expected
assertion values must be copied from one recorded run of the current
`sweep/rank.py` and `sweep/shift.py` functions before moving those files.

- [ ] **Step 3: Move reusable modules**

Move `rank.py`, `shift.py`, `neuron_stats.py`, `steer_sweep.py`,
`steer_lens.py`, `depth_report.py`, `merge_runs.py`, `capture_prefill.py`,
`swap_state.py`, and `swap_followup.py` domain logic. Convert CLI `main`
functions into argument-free package functions plus `main(argv=None)` only
where project-script entry points require them.

- [ ] **Step 4: Extract shared intervention helpers**

Move the reusable `Model`, prompt constants, graph annotation, experiment
definitions, and vocabulary/cache access into `sweep/interventions.py`.
Replace every `sys.path.insert` and sibling-file import with absolute package
imports.

- [ ] **Step 5: Update explorer imports and run tests**

Run:

```bash
uv run pytest tests/sweep tests/explorer -v
uv run python -c "from neuron_explorer.sweep.rank import compute_units"
```

Expected: PASS.

- [ ] **Step 6: Commit sweep package migration**

```bash
git add src/neuron_explorer/sweep tests/sweep sweep src/neuron_explorer/explorer/server.py
git commit -m "refactor: migrate sweep logic into domain package"
```

---

### Task 6: Rebuild all executable entry points under `scripts`

**Files:**
- Create or rewrite: `scripts/serve_explorer.py`
- Create: `scripts/plot_attribution.py`
- Move/rewrite: `scripts/plot_interventions.py`
- Move/rewrite: `scripts/arith_forms.py`
- Move/rewrite: all retained sweep/check/build CLI scripts
- Move: `sweep/setup_pod.sh` → `scripts/setup_pod.sh`
- Move: `sweep/run_layer.sh` → `scripts/run_layer.sh`
- Move: `sweep/run_depth.sh` → `scripts/run_depth.sh`
- Modify: `pyproject.toml`
- Create: `tests/scripts/test_plot_attribution.py`
- Create: `tests/scripts/test_cli_help.py`

**Interfaces:**
- Produces console commands `neuron-explorer` and `neuron-plot-attribution`
- All Python scripts support `--help` without model loading
- `plot_attribution.py --json INPUT --out OUTPUT` loads, validates, optionally
  annotates, writes canonical JSON if enrichment changes it, and renders SVG

- [ ] **Step 1: Add a failing JSON-to-SVG CLI test**

Run the script in a subprocess from a temporary working directory against a
minimal graph fixture. Assert exit code 0, SVG creation, and no model/network
access.

- [ ] **Step 2: Add parameterized `--help` smoke tests**

Cover every retained Python file in `scripts/`. Each command must parse help
before importing optional vLLM/CUDA-only dependencies, or defer those imports
until execution.

- [ ] **Step 3: Verify tests fail**

Run: `uv run pytest tests/scripts -v`

Expected: FAIL because migrated scripts are absent.

- [ ] **Step 4: Create thin scripts for package APIs**

Each script contains argument parsing and orchestration only. Add explicit path
arguments for every consumed/generated data root. `plot_attribution.py`
supports current cache catalogs, dataset catalogs, ordered composition, and no
annotations.

- [ ] **Step 5: Move and repair shell scripts**

Resolve repository root relative to `${BASH_SOURCE[0]}`. Invoke Python with
`uv run` package commands, replace all `sweep/...` references, retain pod setup
semantics, and use `scripts/...` in usage examples and log comments.

- [ ] **Step 6: Add project scripts and run smoke tests**

Run:

```bash
uv sync
uv run pytest tests/scripts -v
uv run neuron-plot-attribution --help
EXPLORER_NO_MODEL=1 uv run neuron-explorer --help
```

Expected: PASS.

- [ ] **Step 7: Commit executable migration**

```bash
git add scripts pyproject.toml uv.lock tests/scripts sweep
git commit -m "refactor: move executable workflows into scripts"
```

---

### Task 7: Move and rewrite documentation

**Files:**
- Modify: `README.md`
- Create: `docs/README.md`
- Create: `docs/explorer.md`
- Create: `docs/sweeps.md`
- Create: `docs/attribution.md`
- Create: `docs/interventions.md`
- Create: `docs/development.md`
- Move: `attribution_memo.md` → `docs/research/attribution_memo.md`
- Move: `research_log.md` → `docs/research/research_log.md`
- Move: `NOTES.md` → `docs/research/notes.md`
- Move: `attribution_slides.html` → `docs/slides/attribution_slides.html`
- Modify: source docstrings, shell comments, frontend comments, and prompt references
- Create: `tests/test_documentation.py`

**Interfaces:**
- Produces a documentation index and one authoritative usage page per domain
- Documents the dataset catalog schema and attribution graph JSON schema

- [ ] **Step 1: Add documentation integrity tests**

Test that every Python/shell executable under `scripts/` is mentioned in one
of the usage documents, local Markdown links resolve, and no documentation or
source docstring refers to executable paths beginning `sweep/` or
`explorer/*.py`.

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/test_documentation.py -v`

Expected: FAIL on old paths and missing pages.

- [ ] **Step 3: Move existing research documents without losing edits**

Use filesystem moves for the memo, log, notes, and slides. Update relative
links and code references in them to `src/neuron_explorer/...` and
`scripts/...`.

- [ ] **Step 4: Write user documentation**

For every retained command record:

- purpose and hardware requirements;
- input files and their schemas;
- complete `uv run` command;
- output files and their schemas;
- resumability/cache behavior;
- relevant environment variables.

In `docs/attribution.md`, include Python examples for raw graph computation,
dataset and cache catalogs, composite fallback, JSON save/load, SVG rendering,
and programmatic analysis across many graph JSON files.

- [ ] **Step 5: Rewrite top-level README and internal references**

Keep README concise: project purpose, installation, first explorer command,
first attribution command, directory map, and links to detailed docs. Update
source docstrings, browser comments, prompts, and shell comments.

- [ ] **Step 6: Run documentation tests**

Run: `uv run pytest tests/test_documentation.py -v`

Expected: PASS.

- [ ] **Step 7: Commit documentation migration**

```bash
git add README.md docs src scripts tests/test_documentation.py attribution_memo.md research_log.md NOTES.md attribution_slides.html
git commit -m "docs: document domain package workflows"
```

---

### Task 8: Remove obsolete layout and verify the complete repository

**Files:**
- Remove empty old directories and obsolete files under `explorer/` and `sweep/`
- Modify any remaining files found by path/import audits

**Interfaces:**
- Produces the final breaking `src` layout with no compatibility shims

- [ ] **Step 1: Audit old imports, paths, and `sys.path` mutations**

Run:

```bash
rg 'sys\.path|from (attribution|runtime|rank|swap_state|plot_attribution) import' src scripts
rg '(python )?(sweep|explorer)/[A-Za-z0-9_.-]+\.(py|sh)' README.md docs src scripts
```

Expected: no matches except historical prose that explicitly describes the
pre-migration layout.

- [ ] **Step 2: Run the full CPU-safe suite**

Run: `EXPLORER_NO_MODEL=1 uv run pytest -v`

Expected: PASS with no model download.

- [ ] **Step 3: Run package and command smoke checks from outside the repo**

Run:

```bash
cd /tmp
uv run --project /Users/wangjunqi/FILES/10_Repos/neuron-explorer python -c \
  "import neuron_explorer; from neuron_explorer.attribution import load_graph, render_svg"
uv run --project /Users/wangjunqi/FILES/10_Repos/neuron-explorer neuron-plot-attribution --help
```

Expected: both commands exit 0.

- [ ] **Step 4: Run the existing tiny mathematical attribution check**

Run: `uv run python scripts/check_attribution.py --tiny`

Expected: every forward, edge-sum, and attribution exactness check reports
within its existing tolerance.

- [ ] **Step 5: Check packaging contents**

Build a wheel and inspect it to confirm attribution modules, explorer static
files, and prompt templates are included:

```bash
uv build
python -m zipfile -l dist/neuron_explorer-*.whl
```

- [ ] **Step 6: Review the final diff without overwriting user changes**

Run: `git status --short && git diff --stat HEAD~7 && git diff --check`

Confirm generated data is absent, original user-modified content survived its
moves, and whitespace checks pass.

- [ ] **Step 7: Commit final cleanup**

```bash
git add -A
git commit -m "chore: complete domain package migration"
```
