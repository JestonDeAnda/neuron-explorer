from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
PYTHON_SCRIPTS = [
    "agent_mcp.py",
    "arith_forms.py",
    "bench.py",
    "build_connections.py",
    "build_embeddings.py",
    "build_layout.py",
    "capture_prefill.py",
    "check_attribution.py",
    "check_capture_prefill.py",
    "check_equivalence.py",
    "depth_report.py",
    "merge_runs.py",
    "mirror.py",
    "neuron_stats.py",
    "owl_delta_vs_base.py",
    "peek.py",
    "plot_attribution.py",
    "plot_interventions.py",
    "rank.py",
    "serve_explorer.py",
    "steer_lens.py",
    "steer_sweep.py",
    "swap_followup.py",
    "swap_state.py",
]


@pytest.mark.parametrize("name", PYTHON_SCRIPTS)
def test_script_help_precedes_optional_runtime_imports(
    name: str, tmp_path: Path
) -> None:
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        "import builtins\n"
        "_import = builtins.__import__\n"
        "def guarded(name, *args, **kwargs):\n"
        "    if name.split('.')[0] in {'torch', 'transformers', 'vllm', 'datasets'}:\n"
        "        raise AssertionError(f'optional dependency imported during --help: {name}')\n"
        "    return _import(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)

    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT),
            "python",
            str(ROOT / "scripts" / name),
            "--help",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
