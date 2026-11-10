# Final Review Fixes

## Status

All final-review findings were addressed in implementation commit
`54ffb6763ff918633696f78767db7ba9264b6689`.

## RED evidence

1. Custom-port security tests initially failed six cases with
   `TypeError: create_app() got an unexpected keyword argument 'port'`.
2. The remote graph persistence regression initially failed because
   `neuron_explorer.attribution.cli` had no focused `fetch_graph` /
   `run_plot_attribution` API and saved remote JSON only after SVG rendering.
3. The script-boundary audit found package-owned `parse_args`, `sys.argv`,
   import-time workflows, and generic `importlib` wrappers. The all-script help
   check showed placeholder help instead of real flags.
4. The path audit found fixed `captures/` in `check_capture_prefill`, fixed
   `runs/` in `depth_report`, and equivalent fixed defaults in retained
   builders/reports.
5. The first required tiny attribution run exposed a refactor closure defect:
   `NameError: name 'ok' is not defined`.

## GREEN implementation and architecture

- `create_app(paths, port=N)` now installs local-only middleware per app,
  accepting exactly `localhost:N`, `127.0.0.1:N`, and `[::1]:N`. The console
  passes its selected port into the factory, while module-level
  `app = create_app()` retains the default import/uvicorn behavior.
- Remote attribution generation canonicalizes and saves the enriched JSON
  before calling the SVG renderer.
- Every retained script under `scripts/` now constructs and parses its real
  CLI. Package modules expose import-safe focused `run_*` functions and no
  package module calls `parse_args` or reads `sys.argv`.
- Installed console commands target the packaged thin scripts. The wheel
  includes scripts and explorer resources but excludes root cache/data/run
  directories.
- `ProjectPaths` now supports `NEURON_EXPLORER_OUTPUT` in addition to root,
  runs, stats, captures, and cache overrides. Commands resolve omitted path
  flags through `ProjectPaths`; `depth_report` has `--runs-dir`, and
  `check_capture_prefill` has `--captures`.
- Documentation now records the complete path/environment model and changed
  flags/examples.
- The moved slide deck was whitespace-normalized; `git diff --check` is clean.
- All six shipped cache datasets remain tracked:
  `act_scale.npz`, `connections.npz`, `layout_writes.npz`, `lens_index.npz`,
  `steer_index.npz`, and `vocab.json`.

## Verification commands and results

```text
EXPLORER_NO_MODEL=1 uv run pytest -v
=> 93 passed, 1 Starlette deprecation warning

EXPLORER_NO_MODEL=1 uv run python scripts/check_attribution.py --tiny
=> exit 0; all forward, completeness, linearity, removal, and RelP checks PASS

EXPLORER_NO_MODEL=1 uv run pytest tests/scripts/test_cli_help.py -v
=> all 24 retained script help checks PASS with representative real flags

outside temporary CWD:
  import neuron_explorer and command modules with NEURON_EXPLORER_ROOT
  run depth_report with NEURON_EXPLORER_RUNS
  run neuron-explorer --help
  run neuron-plot-attribution --help
=> PASS

uv build --wheel --out-dir <temporary-directory>
wheel ZIP inspection
=> neuron_explorer-0.1.0-py3-none-any.whl built
=> 76 files; scripts/resources present; root cache/data/runs/stats/captures/output absent

git diff --check
package parser/sys.argv/importlib audit
script sys.path/importlib audit
=> PASS

IDE lint diagnostics for src/, scripts/, and tests/
=> no errors
```

## Limitations and warnings

- Unit tests intentionally did not load production model weights, use CUDA, or
  access the network.
- The tiny attribution check used Transformers' correct but slower reference
  implementations because optional `causal_conv1d` and
  `flash-linear-attention` accelerators are not installed.
- The test suite reports one upstream Starlette/httpx deprecation warning.
- Wheel construction reports setuptools package-data discovery warnings for
  resource directories, but the wheel inspection confirms all required
  resources are present and root datasets are absent.
