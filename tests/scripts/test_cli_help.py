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
    "huginn_mount_check.py",
    "run_huginn_attribution.py",
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

REAL_OPTIONS = {
    "agent_mcp.py": ["stdio MCP server"],
    "arith_forms.py": ["--cache-dir", "--output-dir"],
    "bench.py": ["--stats", "--max-num-seqs"],
    "build_connections.py": ["--model", "--out"],
    "build_embeddings.py": ["jsonl", "--neighbors"],
    "build_layout.py": ["--kind", "--pca"],
    "capture_prefill.py": ["runs", "--out-dir"],
    "check_attribution.py": ["--tiny", "--targets"],
    "check_capture_prefill.py": ["run", "--captures"],
    "check_equivalence.py": ["--stats", "--capture-root"],
    "huginn_mount_check.py": ["--dataset", "--model", "--output-root"],
    "run_huginn_attribution.py": ["--model", "--dataset", "--output-root", "--benchmark-first"],
    "depth_report.py": ["--runs-dir", "--out"],
    "merge_runs.py": ["inputs", "--out"],
    "mirror.py": ["output", "--base-url"],
    "neuron_stats.py": ["--layers", "--out"],
    "owl_delta_vs_base.py": ["delta", "output"],
    "peek.py": ["jsonl", "--curves"],
    "plot_attribution.py": ["--text", "--dataset-catalog"],
    "plot_interventions.py": ["--input", "--bundle-out"],
    "rank.py": ["jsonl", "--top-words"],
    "serve_explorer.py": ["--port", "--root"],
    "steer_lens.py": ["--neurons-per-batch", "--out"],
    "steer_sweep.py": ["--stats", "--out-dir"],
    "swap_followup.py": ["--input"],
    "swap_state.py": ["explore", "--out"],
}


@pytest.mark.parametrize("name", PYTHON_SCRIPTS)
def test_script_help_shows_real_options_before_optional_runtime_imports(
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
    for option in REAL_OPTIONS[name]:
        assert option in result.stdout
