# Explorer

The explorer combines saved steering runs with live Qwen3.5 inference. The map
shows model layers and neuron sheets; the list ranks units or filters live
activations; neuron panels expose rollouts, steer/logit lenses, direct
connections, chat, and attribution.

## Data locations and environment

By default the server uses `runs/`, `stats/`, `captures/`, `cache/`, and
`output/` under the project root. `--root` changes the root, while `--runs`,
`--stats`, `--captures`, `--cache`, and `--output` override individual paths.
The corresponding reusable environment variables are
`NEURON_EXPLORER_ROOT`, `NEURON_EXPLORER_RUNS`, `NEURON_EXPLORER_STATS`,
`NEURON_EXPLORER_CAPTURES`, `NEURON_EXPLORER_CACHE`, and
`NEURON_EXPLORER_OUTPUT`. Retained scripts that use these stores also accept
`--root` and a focused override such as `--runs-dir`, `--cache-dir`, or
`--out-dir`; explicit flags take precedence over environment defaults.

Live inference defaults to `Qwen/Qwen3.5-4B`. `EXPLORER_MODEL` selects another
model and `EXPLORER_NO_MODEL=1` prevents weight download/loading. CUDA is
fastest, Apple MPS is supported, and CPU is mainly useful for cache-only work.
`PORT` is retained for server internals, but the command's `--port` is the
authoritative launch option.

## `serve_explorer.py`

Purpose: serve the FastAPI application and bundled browser UI. Inputs are the
generated directories above; no schema conversion occurs. Example:

```bash
uv run python scripts/serve_explorer.py \
  --host 127.0.0.1 --port 8791 --root .
```

The equivalent installed command is `uv run neuron-explorer`. Output is an
HTTP server, not a file. It reuses existing run and cache files and has no
resume checkpoint; restarting is safe. Live model endpoints lazily load model
weights. Use `EXPLORER_CLAUDE_MODEL` to select the external Claude CLI model
used by the in-browser agent.

## `agent_mcp.py`

Purpose: expose explorer operations as an MCP stdio server for the Claude tab
or another MCP client. The input is JSON-RPC on stdin and an already-running
explorer HTTP API. Example:

```bash
EXPLORER_URL=http://127.0.0.1:8791 \
  uv run python scripts/agent_mcp.py
```

Responses are written to stdout; it creates no data files. `EXPLORER_URL`
changes the API base URL and `EXPLORER_CHAT` supplies chat context. It needs
the normal Python dependencies and network access only to the selected local
or remote explorer. There is no cache/resume layer of its own.

## `build_embeddings.py`

Purpose: embed every rollout and build the two-dimensional UMAP used by the
embeddings view. Input is sweep JSONL: one rollout object per line, including
`text`, `kind`, `neuron`, `prompt`, and `dose`.

```bash
uv run python scripts/build_embeddings.py runs/q35-4b-L22.jsonl \
  --model BAAI/bge-small-en-v1.5 --chars 600 --neighbors 15
```

Outputs beside the input are `<run>.emb.npy` (normalized float16 sentence
embeddings, one row per JSONL line) and `<run>.umap.json` with parallel `x`,
`y`, `row`, `kind`, `neuron`, `prompt`, and `dose` arrays. Sentence
Transformers uses CUDA, MPS, or CPU automatically; UMAP is CPU-heavy. Existing
outputs are overwritten, so there is no incremental resume or cache check.
Downloaded model weights use the Hugging Face cache.

## `build_layout.py`

Purpose: arrange each layer's neurons on a square sheet by similar write
directions (`writes`) or read weights (`reads`). It downloads model config and
safetensors, cosine-normalizes weights, applies PCA and UMAP, then snaps points
to a grid.

```bash
uv run python scripts/build_layout.py \
  --model Qwen/Qwen3.5-4B --kind writes --pca 50 --neighbors 30 \
  --out cache/layout_writes.npz
```

Input is a Hugging Face model whose intermediate size is a perfect square.
Output is `cache/layout_<kind>.npz`, containing `cell[L,N]` int16 grid indices.
This is CPU/RAM intensive and requires `huggingface_hub`, `safetensors`,
NumPy, and UMAP; it does not require a GPU. Hugging Face downloads are cached,
but the layout file is recomputed and overwritten on every run.

## `build_connections.py`

Purpose: precompute each neuron's strongest direct-path readers and writers
across all layer pairs.

```bash
uv run python scripts/build_connections.py \
  --model Qwen/Qwen3.5-4B --k 24 --out cache/connections.npz
```

Input is the model's MLP and post-attention-normalization weights. Output is a
compressed NPZ containing `{down,up}_{layer,n,z,sign}` arrays shaped
`[layers, neurons, k]` plus scalar `k`. CUDA is recommended (about five
minutes on a 4090); MPS is selected when CUDA is absent, while CPU is not
implemented by this command. It requires Torch, Hugging Face Hub, and
safetensors. Model shards use the Hugging Face cache; the requested output is
fully recomputed and overwritten, with no resume.

## Related data

Create sweep inputs with [steering sweep commands](sweeps.md). Build or render
next-token graphs with the [attribution guide](attribution.md).
