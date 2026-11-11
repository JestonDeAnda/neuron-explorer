# Attribution graphs

Attribution freezes Qwen3.5's non-MLP nonlinear quantities at their forward
values, treats MLP activations and embeddings as sources, and traces direct
activation-times-gradient contributions to a centered next-token logit. The
graph builder ranks candidates in total-effect (RelP) mode, computes direct
incoming edges in batches, then prunes nodes and edges by path influence.

For interpretation details and limitations, read the
[attribution graph Q&A](attribution-graph-QA.md) and the preserved
[research memo](research/attribution_memo.md).

## Command-line graph creation and rendering

### `plot_attribution.py`

Purpose: fetch a raw graph from a running explorer or load a saved graph,
optionally annotate it, and render standalone SVG.

```bash
uv run python scripts/plot_attribution.py \
  --text "Fact: the capital of the state containing Dallas is" \
  --token " Austin" --k 512 --edges 300 --labels 40 \
  --out runs/attribution/dallas.svg
```

The installed equivalent is `uv run neuron-plot-attribution`. Exactly one of
`--text` and `--json` is required. `--text` posts `text`, `k`, `chat`, and an
optional target token to `<url>/api/attribution`; `--url` defaults to
`http://127.0.0.1:8791`. The server needs the model loaded and may use CUDA,
MPS, or CPU. Rendering a `--json` input is CPU-only.

Text mode writes the SVG and a same-stem `.json`; the default SVG is
`runs/attribution/<prompt-slug>.svg`. JSON mode validates and renders the input,
and writes annotations back to that JSON if catalogs enrich it. SVG
`--edges`/`--labels` only control presentation, not graph computation. Existing
outputs are overwritten. Remote JSON is canonicalized and saved before SVG
rendering, so a renderer failure still leaves a reusable graph. The command has
no partial resume; a saved JSON can be rendered repeatedly without model
inference.

Catalog flags may be repeated and are applied in command-line order:

```bash
uv run python scripts/plot_attribution.py \
  --json runs/attribution/dallas.json \
  --dataset-catalog data/annotations.jsonl \
  --cache-catalog cache \
  --out runs/attribution/dallas-annotated.svg
```

`--catalog dataset:PATH` and `--catalog cache:DIR` are equivalent generic
forms. `--no-annotations` explicitly disables lookup and cannot be combined
with catalog flags. With no catalog flags the graph is rendered unchanged.

### `check_attribution.py`

Purpose: verify forward equivalence, completeness, edge linearity, RelP
completeness, and source-removal behavior.

```bash
uv run python scripts/check_attribution.py \
  --tiny --device cpu --targets 6
```

`--tiny` builds a random four-layer Qwen3.5 model and needs no weights. Without
it, the command loads `Qwen/Qwen3.5-4B` in float32 and tokenizes `--prompt`;
use `--device cuda`, `mps`, or `cpu` as available. Results are PASS/FAIL lines
on stdout and the process exits nonzero on a failed exactness check. It writes
no cache. Hugging Face weights are reused, but each validation run recomputes
all checks.

## Huginn-0125 attribution

Raven is not a Qwen3.5 model. `scripts/run_huginn_attribution.py` uses a separate
adapter that unrolls Huginn's 2 prelude blocks, 4 shared recurrent blocks
repeated 32 times, and 2 coda blocks as virtual depth 0..131. Each neuron keeps
that virtual depth, its stage, recurrence iteration, physical layer, physical
neuron, and a `shared_weight_id` that is stable across recurrence. The attributed
scalar is the teacher-forced sum of completion-token log-probabilities. Prompt
text follows `prompt_variant`: `answer_cue` appends a newline and `Answer:`,
and `native_bos` prepends the tokenizer BOS after tokenization. Completion
strings, including leading spaces, are encoded unchanged. Prompt and completion
are deliberately tokenized separately: this computes generation probability
conditioned on the fixed prompt tokens. Joint BPE tokenization can merge the
last prompt token with a completion that has no leading space (for example
`:` + `false`), which would change the conditioning context. The saved
`prompt_protocol.tokenization` records this choice. A row id fixes one
recurrent-state seed for all four conditions.

```bash
uv run python scripts/run_huginn_attribution.py \
  --model /path/to/huginn-0125 \
  --dataset /path/to/按任务 \
  --output-root /path/to/attribution/huginn-0125 \
  --commit "$(git rev-parse HEAD)" \
  --benchmark-first
```

Default fidelity is `--k 512` and `--max-edges 3000`. Graphs land in
`output/{json,svg}/<task-slug>/<sample-id>__<condition>.{json,svg}` with
`output/manifest.jsonl`, `output/progress.json`, and `output/summary.json`.
Validated JSON is skipped on resume; a missing SVG is redrawn from that JSON.
Resume matching is strict: implementation version, prompt/completion token IDs,
row seed, model identity, code commit, recurrence count, prompt protocol, and
graph configuration must all match. A stale or partial JSON is recomputed.
Model identity is a content SHA-256 over `config.json`, checkpoint-side Python
code, and every safetensors/PyTorch weight shard; computing it reads the full
checkpoint once at process startup.
`scripts/huginn_mount_check.py` checks the dataset, checkpoint, staged commit,
and, with `--mounts`, the four DLC storage mounts before any GPU work.

Huginn completion graphs have one aggregate output pseudo-node rather than one
next-token logit. Top-level `target.kind` is `completion_logprob`; it includes
all completion token IDs and prediction positions. `completion_logprob` is the
summed scalar and `prob`/`stats.sequence_probability` are its exponentiated
sequence probability. The output node repeats this target metadata. Edges still
use `[source, destination, signed_weight]`, while the internal matrix is stored
as target rows and source columns.

These weights are a local activation-times-gradient linearization. RMS
denominators and attention patterns are frozen, MLP products use the half rule,
and log-softmax remains differentiated. The `other` residual therefore contains
the local intercept plus omitted state and non-candidate effects; the graph is
not an exact additive decomposition of the nonlinear log-probability value.
Run metadata records `edge_semantics=local_linear_half_rule_direct` and
`target_transform=differentiated_log_softmax`.

## Raw graph computation API

`build_graph` is Torch- and Transformers-backed and is lazily imported:

```python
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from neuron_explorer.attribution import build_graph

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3.5-4B", dtype=torch.float32
).eval().to("mps")
ids = tok("Fact: the capital of France is").input_ids
graph = build_graph(
    model.model.language_model,
    model.lm_head,
    ids,
    token=None,              # top prediction; or an integer token ID
    k=512,                   # total-effect candidate neurons
    batch=16,                # direct-edge targets per backward pass
    node_threshold=0.8,      # cumulative node influence retained
    edge_threshold=0.98,     # cumulative edge influence retained
    max_edges=3000,
)
```

The model must be Qwen3.5 because `Linearized` patches its RMSNorm, full
attention, Gated DeltaNet, and MLP implementations. Float32 gives the strongest
forward-equivalence guarantee; bf16 is faster but introduces expected rounding
differences. Memory grows with prompt length, candidate count, and batch size.
Model parameters that originally require gradients are temporarily disabled
and restored even on failure.

Intervened graphs use absolute activation clamps:

```python
graph = build_graph(
    model.model.language_model,
    model.lm_head,
    ids,
    k=512,
    clamp={(5, 7, 8109): 0.0},  # (layer, token position, neuron): value
)
```

Clamped neurons become source-only nodes and carry `"clamped": true` when they
survive pruning. The raw builder returns integer token IDs but not decoded
`tokens` or `target`; add those through `annotate_graph` before catalog-backed
rendering.

`Linearized` is also public for research requiring unpruned contributions. Use
it as a context manager, call `forward(ids, copies=B)`, select one target per
batch row with `neuron_target` or `logit_target`, and call `contributions`.
The result has `"mlp"` shaped `[layers,B,tokens,neurons]` and `"embed"` shaped
`[B,tokens]`. The context manager always restores patched forwards.

## Annotation catalogs

All catalogs implement `lookup(layer, neuron) -> NeuronAnnotation | None`.
Annotations can contain an explanation, positive/negative steer tokens,
positive/negative KL values, and arbitrary provenance.

### Dataset catalog

`DatasetCatalog.from_path(path, duplicate="error")` accepts a JSON array or
JSONL. Each record requires `layer` and either `neuron` or alias `n`:

```json
{"layer":22,"neuron":2742,"explanation":"alien causal agents","steer_up":[" alien"," planet"],"steer_down":[" ordinary"],"kl_up":0.91,"kl_down":0.22,"provenance":{"source":"manual-v1"}}
```

`name` aliases `explanation`. Token fields may be a string or sequence.
Unknown fields are retained in provenance. Duplicate `(layer,neuron)` keys
raise by default; `duplicate="last"` lets the last record win.

```python
from pathlib import Path
from neuron_explorer.attribution import DatasetCatalog

catalog = DatasetCatalog.from_path(
    Path("data/annotations.jsonl"), duplicate="last"
)
```

### Cache catalog

`CacheCatalog` adapts explorer cache files:

- `names.json`: object keyed by `L<layer>/N<neuron>`.
- `vocab.json`: token-string array indexed by cached token IDs.
- `steer_index.npz`: `up`, `down`, and optional `up_kl`, `down_kl` arrays.
- optional `lens_index.npz`: `up` and `down` fallback token arrays.

```python
from pathlib import Path
from neuron_explorer.attribution import CacheCatalog

root = Path("cache")
catalog = CacheCatalog(
    root / "names.json",
    root / "steer_index.npz",
    root / "vocab.json",
    root / "lens_index.npz",
)
```

Missing files and out-of-range entries are treated as absent. Steer tokens
take precedence over logit-lens tokens independently for each pole; KL is read
only from the steer cache.

### Composite and empty catalogs

`CompositeCatalog(first, second, ...)` merges at field granularity. For every
field, the first non-empty value wins; a later catalog fills only missing
fields. This is also the CLI precedence rule because repeated catalog flags
preserve order. `EmptyCatalog` always returns no metadata.

```python
from neuron_explorer.attribution import CompositeCatalog, annotate_graph

catalog = CompositeCatalog(dataset_catalog, cache_catalog)
annotated = annotate_graph(
    graph,
    tokens=tok.convert_ids_to_tokens(ids),
    target=tok.decode([graph["token"]]),
    catalog=catalog,
    lens_tokens=5,
)
```

`annotate_graph` deep-copies the graph, adds top-level `tokens` and `target`,
adds each node's `tok`, and adds neuron `name`, sign-selected `lens`, and full
`annotation`. It does not mutate the source graph.

## Attribution graph JSON schema

`schema_version` is currently `1`. Unversioned files are accepted as version 1
on load. Required top-level fields are:

- `token`: integer target token ID.
- `prob`: target probability from the linearized forward.
- `nodes`: node object array.
- `edges`: `[source_index, destination_index, weight]` arrays.
- `stats`: computation/pruning summary object.

Saved annotated single-token graphs commonly add `tokens` (decoded
prompt-token strings) and `target` (decoded target). Huginn completion graphs
instead use a structured `target` object with `kind`, `token_ids`, and
`prediction_positions`, plus `prompt_token_ids`, `completion_token_ids`, and
`completion_logprob`. Studies may add `title`, `group`, `edits`, `watch`, or
`top`.

Every node requires `kind`, `layer`, `pos`, `n`, `act`, and `influence`.
`kind` is one of:

- `embed`: `layer=-1`, `n=0`, activation normally null.
- `neuron`: model layer and neuron indices, activation and path influence.
- `logit`: final pseudo-layer, target token ID in `n`, influence `1.0`. For an
  aggregate Huginn completion target, `n=-1` and the node has `target_kind`,
  `target_token_ids`, and `prediction_positions`.

Builder-produced neuron nodes may also include `gate`, `other`, and
`clamped`. Annotation adds `tok`, `name`, `lens`, and `annotation`. Edge
indices must be in bounds and weights must be finite numbers. Edges point from
source to destination; weights are raw signed attributions.

Builder `stats` contains `candidates`, `candidate_share`,
`logit_from_candidates`, `logit_other`, `kept_neurons`, and `kept_edges`.
The validator intentionally permits extra fields for study-specific metadata,
but rejects missing required fields, invalid node kinds, malformed edges,
unsupported versions, and non-finite weights.

## Save, load, validate, and render

```python
from pathlib import Path
from neuron_explorer.attribution import (
    load_graph, render_svg, save_graph, validate_graph
)

validate_graph(annotated)
path = save_graph(annotated, Path("runs/attribution/france.json"))
loaded = load_graph(path)
Path("runs/attribution/france.svg").write_text(
    render_svg(loaded, max_edges=300, labels=40),
    encoding="utf-8",
)
```

`save_graph` adds the current schema version, validates, creates parent
directories, and atomically replaces the destination with UTF-8 JSON.
`load_graph` adds version 1 to legacy unversioned objects and validates before
returning. `render_svg` returns a complete standalone SVG string. Rendering is
pure and CPU-only; `max_edges` picks the highest influence-weighted edges and
`labels` limits named neuron labels.

## Programmatic analysis across many graphs

Treat files as validated records rather than parsing JSON directly:

```python
from collections import Counter, defaultdict
from pathlib import Path
from neuron_explorer.attribution import load_graph

files = sorted(Path("runs/attribution").glob("*.json"))
graphs = [(path, load_graph(path)) for path in files]

frequency = Counter()
influence = defaultdict(float)
for path, graph in graphs:
    seen = set()
    for node in graph["nodes"]:
        if node["kind"] != "neuron":
            continue
        key = (node["layer"], node["n"])
        influence[key] += float(node["influence"])
        seen.add(key)  # count at most once per graph
    frequency.update(seen)

for key, count in frequency.most_common(20):
    print(key, count, influence[key] / count)
```

For role- or token-position comparisons, include `node["pos"]` in the key.
For intervention cohorts, filter top-level `group` or `edits`. Do not compare
raw influence sums without deciding whether repeated positions of the same
neuron should collapse; graphs represent `(layer, position, neuron)` nodes,
while many scientific questions concern `(layer, neuron)` identities.
