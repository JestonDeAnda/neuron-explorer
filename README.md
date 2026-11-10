# neuron explorer

Steering-first tools for studying Qwen3.5 MLP neurons. The project records
controlled steering sweeps, serves an interactive 3D explorer, computes
next-token attribution graphs, and runs causal intervention studies.

![neuron explorer](explorer.png)

## Install and start

```bash
uv sync --dev
uv run neuron-explorer
```

Open <http://127.0.0.1:8791>. The explorer reads generated data from `runs/`,
`stats/`, `captures/`, and `cache/`. It can use CUDA, MPS, or CPU for live
model operations; set `EXPLORER_NO_MODEL=1` for cache-only browsing.

Create an attribution graph through the running explorer:

```bash
uv run neuron-plot-attribution \
  --text "Fact: the capital of the state containing Dallas is" \
  --out runs/attribution/dallas.svg
```

## Repository map

- `src/neuron_explorer/`: reusable explorer, attribution, sweep, and tool code.
- `scripts/`: retained executable workflows and compatibility entry scripts.
- `tests/`: package, command, and documentation checks.
- `docs/`: user, API, research, and developer documentation.
- `runs/`, `stats/`, `captures/`, `cache/`: generated or downloaded data.

## Documentation

- [Documentation index](docs/README.md)
- [Explorer](docs/explorer.md)
- [Steering sweeps](docs/sweeps.md)
- [Attribution APIs and formats](docs/attribution.md)
- [Intervention studies](docs/interventions.md)
- [Development and utility commands](docs/development.md)
- [Attribution graph questions and answers](docs/attribution-graph-QA.md)
- [Research archive](docs/research/README.md)
