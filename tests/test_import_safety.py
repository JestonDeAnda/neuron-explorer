from __future__ import annotations

import ast
import importlib
import pkgutil
import sys
from pathlib import Path

import neuron_explorer


def _package_modules() -> list[str]:
    return sorted(
        module.name
        for module in pkgutil.walk_packages(
            neuron_explorer.__path__, neuron_explorer.__name__ + "."
        )
    )


def test_every_package_module_imports_without_parsing_or_work(monkeypatch):
    def forbidden_parse(*args, **kwargs):
        raise AssertionError("package module called parse_args during import")

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", forbidden_parse)
    monkeypatch.setattr(sys, "argv", ["import-safety-sentinel"])

    for name in _package_modules():
        importlib.import_module(name)


def test_package_modules_do_not_own_cli_parsing_or_sys_argv():
    root = Path(next(iter(neuron_explorer.__path__)))
    violations = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "parse_args"
            ):
                violations.append(f"{path}: parse_args")
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "sys"
                and node.attr == "argv"
            ):
                violations.append(f"{path}: sys.argv")
    assert not violations, "\n".join(violations)
