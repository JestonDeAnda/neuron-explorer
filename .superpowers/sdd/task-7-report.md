# Task 7 report

## Status

Complete. Research documents were moved without discarding their content, the
current command surface was audited from implementation and parser definitions,
and authoritative user/developer documentation now covers every retained
Python and shell executable.

## Delivered

- Replaced the top-level README with a concise project/install/first-command
  guide and links to domain documentation.
- Added the documentation index and domain guides for explorer, sweeps,
  attribution, interventions, and development.
- Documented each retained command's purpose, actual inputs/options, complete
  `uv run` invocation, outputs and schemas, hardware/dependency requirements,
  environment variables, and cache/resume behavior.
- Documented raw attribution computation; dataset, cache, composite, and empty
  catalogs; field-level precedence; graph JSON version/schema; annotation;
  atomic save/load; SVG rendering; and multi-graph programmatic analysis.
- Moved the memo, log, notes, and slides into `docs/research/` and
  `docs/slides/`, updating only migrated references. Kept
  `docs/attribution-graph-QA.md` in place and linked it from current docs.
- Updated stale source docstrings, frontend comments, and research/slide code
  references to package or `scripts/` paths.
- Added documentation integrity tests for executable coverage, local Markdown
  links, and legacy executable paths.

## TDD evidence

The new test was run before the migration and failed for five undocumented
scripts plus legacy README/source paths. After implementation:

```text
uv run pytest tests/test_documentation.py -v
3 passed in 0.05s
```

Full regression suite:

```text
uv run pytest
78 passed, 5 warnings in 4.99s
```

`git diff --check` passed and IDE diagnostics reported no linter errors in the
edited source/test files.

## Concerns

The five pre-existing full-suite warnings are unhandled Hugging Face background
thread `ProxyError: 403 Forbidden` warnings in explorer server-factory tests.
They did not fail tests and are unrelated to the documentation changes.
