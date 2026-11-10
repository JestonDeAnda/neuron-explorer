# Steering sweeps

Sweeps first measure each neuron's activation scale, then add its
down-projection direction during generation. The retained production workflow
uses Goodfire's steering fork of vLLM on CUDA. Analysis-only commands generally
run on CPU.

## Pod and orchestration scripts

### `setup_pod.sh`

Purpose: prepare a fresh `/workspace` CUDA pod with uv, a Python 3.12 virtual
environment, the steering vLLM fork, FLA kernels, and model weights.

```bash
MODEL=Qwen/Qwen3.5-4B uv run -- bash scripts/setup_pod.sh
```

It requires Linux, NVIDIA CUDA, internet access, `apt`, and writable
`/workspace`. It writes `/workspace/env.sh`, `/workspace/download.log`,
`.venv/`, and `vllm-steer/`. Existing clone and virtual-environment directories
are reused, Hugging Face and uv caches live under `/workspace`, and the model
download itself resumes through Hugging Face tooling.

### `run_layer.sh`

Purpose: run a full stats pass and full-neuron `main_rel` sweep for one layer.
Inputs are a layer number and run name; later arguments pass to
`steer_sweep.py`.

```bash
uv run -- bash scripts/run_layer.sh 22 q35-4b-L22 --gpu-mem 0.85
```

It requires the pod environment above and CUDA. It creates
`stats/q35-4b_L22.npz`, `runs/q35-4b-L22.{jsonl,meta.json}`, captures, and
`stats_L22.log`/`sweep_q35-4b-L22.log`. Existing stats are skipped; the sweep
resumes from JSONL keys already present. A successful run appends
`SWEEP_DONE` to its log.

### `run_depth.sh`

Purpose: run one short multi-layer stats pass, then a random-neuron sweep for
each requested layer.

```bash
uv run -- bash scripts/run_depth.sh "4 9 13 17" 384 64
```

Arguments are the space-separated layers, neurons per layer (default 384), and
random controls (default 64). It needs the same CUDA pod. Outputs are per-layer
stats and `runs/q35-4b-L<layer>-r384.{jsonl,meta.json}`, plus `depth.log`,
`depth_stats.log`, and sweep logs. Existing stats are reused, each sweep
resumes at rollout granularity, failures are recorded, and later layers still
run.

## Measuring and generating

### `neuron_stats.py`

Purpose: estimate four SwiGLU sign-regime extrema, absolute activation scale,
residual-norm quantiles, and down-projection columns.

```bash
uv run python scripts/neuron_stats.py \
  --model Qwen/Qwen3.5-4B --layers 22 --tokens 2000000 \
  --seq-len 1024 --batch 16 --seed 0 --out stats/q35-4b
```

The input corpus streams FineWeb-Edu and UltraChat from Hugging Face. For each
layer it writes `<out>_L<layer>.npz` with `pp_max`, `nn_max`, `pn_min`,
`np_min`, `abs_max` (`[neurons]`), `w_down` (`[neurons, hidden]`, float16),
`resid_norm_q10_50_90`, and model/layer/token metadata. It requires CUDA and
enough memory for Qwen3.5 plus the requested batch; `datasets`,
`transformers`, Torch, and network/cache access are required. The command
recomputes and overwrites outputs; Hugging Face datasets and weights are
cached, but there is no partial resume.

### `steer_sweep.py`

Purpose: generate one rollout per `(unit, prompt, dose)`, plus baselines and
random-direction controls.

```bash
uv run python scripts/steer_sweep.py \
  --stats stats/q35-4b_L22.npz --design main_rel --dose-unit rel \
  --neurons all --random-controls 256 --run q35-4b-L22 \
  --out-dir runs --capture-root captures --capture-layers=+0,+1,+4,+8
```

`--neurons` accepts `all`, `random:128`, a single ID, ranges, or comma-separated
mixtures. `main`, `calib`, and `main_rel` are the implemented designs.
Additional controls are `--model`, `--max-tokens`, `--top-k`,
`--capture-layout` (`packed` or `sharded`), `--seed`, `--chunk`,
`--max-configs`, `--max-num-seqs`, and `--gpu-mem`; an empty
`--capture-layers` disables captures.

The stats NPZ schema is the one above. Output JSONL contains `layer`, `kind`,
`neuron`, `ref_neuron`, `prompt`, `dose`, `add`, `rel_push`, generated `text`,
finish reason, token IDs, chosen logprobs, top token IDs/logprobs, and
`capture_id`. `<run>.meta.json` records model, prompts, design, paths, tokenized
prompts, and capture settings. Filesystem captures are written under the
capture root by the vLLM fork. CUDA and that fork are mandatory. On rerun,
existing `(kind, neuron, prompt, dose)` JSONL records are skipped; each chunk
is flushed, making interrupted runs resumable.

### `steer_lens.py`

Purpose: measure average next-token delta-logprob and KL for positive and
negative relative pushes of every neuron.

```bash
uv run python scripts/steer_lens.py \
  --model Qwen/Qwen3.5-4B --layers 0-31 --dose 0.8 \
  --neurons-per-batch 96 --k 12 --out stats/steer_lens
```

It uses four built-in contexts and writes `<out>_L<layer>.npz` containing
`up_tok`, `down_tok` (`[neurons,k]` token IDs), `up_kl`, `down_kl`, dose, and
contexts. CUDA and model weights are required. Hugging Face weights are
cached, but completed layers are not detected: reruns recompute and overwrite
each requested layer.

## Analysis and reporting

### `rank.py`

Purpose: rank units by distinctive-word cluster mass against random controls.

```bash
uv run python scripts/rank.py runs/q35-4b-L22.jsonl \
  --out runs/q35-4b-L22.units.jsonl --max-rel 1.6 \
  --top-words 12 --show 40 --find 2742
```

Input is sweep JSONL. Output is one JSONL record per unit with `kind`,
`neuron`, `layer`, `ref_neuron`, strongest-word `z`, top-five `mass`,
`null_pct`, `words`, `per_prompt`, and per-dose coherence metrics. It is
CPU-only and needs NumPy. Output is replaced atomically only at normal file
write granularity; there is no cache or resume.

### `depth_report.py`

Purpose: compare neuron and random-control steering across layers.

```bash
uv run python scripts/depth_report.py \
  --layers 0-31 --neurons 384 --controls 64 \
  --out runs/depth_report.json
```

It searches `runs/q35-4b-L<layer>-r<neurons>.jsonl`, falling back to
`-full.jsonl` and deterministic subsampling. It writes the requested JSON
array of layer metrics and writes `.units.jsonl` beside non-subsampled subset
runs. The report includes null/neuron quantiles, shares above controls,
coherence summaries, and top units. CPU/NumPy only; missing layers are skipped.
Outputs are recomputed and overwritten, not resumed.

### `merge_runs.py`

Purpose: deduplicate and join shards in command-line order.

```bash
uv run python scripts/merge_runs.py \
  runs/q35-4b-L22-s0.jsonl runs/q35-4b-L22-s1.jsonl \
  --out runs/q35-4b-L22-full.jsonl
```

Inputs must use the sweep JSONL schema and the first input must have a sibling
`.meta.json`. Output contains the first record for each
`(kind,neuron,prompt,dose)` and a copied metadata object with a `shards` list.
CPU-only with no optional dependencies. The output and metadata are overwritten
and the operation does not resume.

### `peek.py`

Purpose: inspect raw rollouts, dose-response curves, or distinctive words in a
terminal.

```bash
uv run python scripts/peek.py runs/q35-4b-L22.jsonl \
  --neuron 2742 --kind neuron --width 110
```

Use `--curves` for aggregated dose metrics or `--words --max-rel 1.6 --top 6`
for per-unit words. Input is sweep JSONL; output is stdout only. It is
CPU/NumPy analysis, creates no cache, and is safe to rerun.

### `bench.py`

Purpose: benchmark unsteered, steered, capture, and logprob vLLM conditions.

```bash
uv run python scripts/bench.py \
  --stats stats/q35-4b_L22.npz --n 1280 --tokens 256 \
  --max-num-seqs 320 --max-configs 512 --gpu-mem 0.85
```

`--only` accepts comma-separated condition names such as
`steered,sweep-varlen`. Input is a stats NPZ. Results are throughput lines on
stdout; temporary captures go to `/tmp/bench-captures`. CUDA and the steering
vLLM fork are mandatory. Model files are cached, benchmark captures may be
reused/overwritten by request ID, and there is no benchmark resume.

## Capture reconstruction and validation

### `capture_prefill.py`

Purpose: reconstruct last-prompt MLP inputs for existing sweep rows using
teacher-forced Hugging Face prefills.

```bash
uv run python scripts/capture_prefill.py \
  runs/q35-4b-L22-full.jsonl --model Qwen/Qwen3.5-4B \
  --batch 256 --seed 0 --out-dir captures
```

Each run needs its sibling metadata JSON. Output
`<run>.mlp_in.npy` is float16 `[rows,capture_layers,hidden]`, aligned with
JSONL lines; `<run>.mlp_in.json` records layers, hook, position, and row count.
CUDA and Transformers are required. If the final NPY exists the entire run is
skipped. Otherwise a `.tmp` memmap is replaced only after completion; partial
temporary files are not resumed.

### `check_capture_prefill.py`

Purpose: compare reconstructed NPY rows with surviving vLLM filesystem
captures.

```bash
uv run python scripts/check_capture_prefill.py \
  runs/mainrel-e2e.jsonl captures/mainrel-e2e
```

The first input is sweep JSONL; the second is a fork capture directory. The
command also reads `captures/<run>.mlp_in.{npy,json}` and prints rollout counts
plus median/p95/max relative error by layer. It requires NumPy and the steering
fork's capture reader, but performs no model inference and can run on CPU. It
writes nothing and has no cache/resume behavior.

### `check_equivalence.py`

Purpose: validate HF versus vLLM steering and captured-activation
reconstruction on a new GPU environment.

```bash
uv run python scripts/check_equivalence.py \
  --stats stats/q35-4b_L22.npz --neurons 2742,17,5000 \
  --doses 10,-2 --prompt "Who are you?" --new 48 \
  --capture-gen 8 --capture-root captures-check
```

It reads the stats NPZ, runs both engines, prints token-prefix agreement and
capture errors, and stores temporary vLLM captures under `--capture-root`.
CUDA, Transformers, and the steering vLLM fork are required. The capture root
is deleted before each run, so this is deliberately non-resumable.
