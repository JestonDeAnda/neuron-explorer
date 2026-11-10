# Intervention studies

These retained commands reproduce the project's Dallas-to-California state
swap and cross-surface-form arithmetic experiments. They load the full
Qwen3.5 model, edit MLP activations at specific token positions, and recompute
the rest of the real forward pass. They are research workflows with fixed
prompts, not general-purpose intervention CLIs.

All commands use `ProjectPaths`: `NEURON_EXPLORER_ROOT` changes the data root,
and `NEURON_EXPLORER_RUNS` / `NEURON_EXPLORER_CACHE` override the important
directories. CUDA is strongly recommended; MPS may work for smaller individual
runs, while the graph suites are memory- and compute-intensive.

## `swap_state.py`

Purpose: inspect Texas/California candidates or build the state-swap
experiment result consumed by later commands.

```bash
uv run python scripts/swap_state.py swap \
  --max-layer 12 --k 512 \
  --out runs/attribution/swap_state.json
```

The `explore` subcommand prints baselines, cache-lens matches, and Dallas versus
San Francisco activations:

```bash
uv run python scripts/swap_state.py explore --max-layer 12 --k 512
```

Inputs are Qwen3.5 weights plus `cache/vocab.json` and
`cache/steer_index.npz`. `swap` computes attribution groups, runs fixed
interventions, prints probabilities, and writes JSON with `texas`,
`california`, and `results`; each group row contains layer, neuron, activation,
and influence. It requires Torch, Transformers, NumPy, and enough accelerator
memory for the model and attribution graph. The output is overwritten and no
experiment-level cache or resume is used.

## `swap_followup.py`

Purpose: run fixed follow-up controls using the saved California group:
alternate Dallas probes, greedy completions, and edits applied to other cities.

```bash
uv run python scripts/swap_followup.py \
  --input runs/attribution/swap_state.json
```

It reads `runs/attribution/swap_state.json`, loads Qwen3.5, captures the San
Francisco source activations, and prints watched-token probabilities and
greedy text to stdout. It creates no files. Accelerator and dependency
requirements match `swap_state.py`; model weights use the Hugging Face cache,
but the forward passes are recomputed on every run.

## `plot_interventions.py`

Purpose: compute attribution graphs for every fixed state-swap experiment,
render each graph, build an index, and create a self-contained HTML bundle.

```bash
uv run python scripts/plot_interventions.py \
  --input runs/attribution/swap_state.json \
  --cache-dir cache \
  --output-dir runs/attribution/interventions \
  --bundle-out runs/attribution/state_swap.html \
  --model Qwen/Qwen3.5-4B --k 512
```

Inputs are the swap JSON, `vocab.json`, `steer_index.npz`, and model weights.
For each experiment it writes `NN_<slug>.json` and `.svg`; the directory also
gets `index.html`, and `--bundle-out` gets all SVGs inlined. JSON graph records
include normal graph fields plus title/group, edits, watched probabilities,
and real-model top tokens. Existing per-experiment JSON files skip graph
recomputation unless `--redo` is set, but the corresponding SVG must also be
present because index/bundle construction reads it. The index is refreshed
after each experiment, providing coarse resume. This is a long GPU workflow;
float32/bf16 differences are reported as warnings rather than hidden.

## `arith_forms.py`

Purpose: compare one arithmetic calculation written as digits, English words,
and Chinese numerals; find role-matched shared neurons; and run ablation and
activation-transplant tests against a control calculation.

```bash
uv run python scripts/arith_forms.py \
  --model Qwen/Qwen3.5-4B --k 1024 \
  --cache-dir cache --output-dir runs/attribution/arith
```

Inputs are model weights, `cache/vocab.json`, and
`cache/steer_index.npz`. The output directory receives six graph JSON files,
six SVGs, and `report.json`; its parent receives `arith_forms.html` with
findings, tables, and inlined SVG. Graph JSON follows the schema in the
[attribution guide](attribution.md), while `report.json` contains prompts,
pairwise overlap arrays, layer/role profiles, shared neuron records, ablation,
transplant, and write-test results.

Existing graph JSON files are loaded unless `--redo` is supplied; overlap,
causal tests, report JSON, SVG, and HTML are regenerated every run. Deleting a
stale graph or using `--redo` refreshes graph computation. The workflow
requires NumPy, Torch, Transformers, attribution dependencies, and a capable
GPU; Hugging Face weights are cached.
