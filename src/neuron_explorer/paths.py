"""Shared paths for generated project data."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Immutable locations used by neuron-explorer commands."""

    root: Path
    runs: Path
    stats: Path
    captures: Path
    cache: Path
    output: Path

    @classmethod
    def discover(cls, root: Path | None = None) -> ProjectPaths:
        """Resolve paths without creating directories.

        Source checkouts are discovered from the package location, independent
        of the current working directory. Installed packages intentionally use
        the current working directory as their output-root fallback.
        """
        project_root = cls._project_root(root)
        return cls(
            root=project_root,
            runs=cls._override("NEURON_EXPLORER_RUNS", project_root / "runs"),
            stats=cls._override("NEURON_EXPLORER_STATS", project_root / "stats"),
            captures=cls._override(
                "NEURON_EXPLORER_CAPTURES", project_root / "captures"
            ),
            cache=cls._override("NEURON_EXPLORER_CACHE", project_root / "cache"),
            output=project_root / "output",
        )

    @staticmethod
    def _project_root(explicit_root: Path | None) -> Path:
        if explicit_root is not None:
            return Path(explicit_root)

        environment_root = os.environ.get("NEURON_EXPLORER_ROOT")
        if environment_root:
            return Path(environment_root)

        package_path = Path(__file__).resolve()
        for parent in package_path.parents:
            if (
                (parent / "pyproject.toml").is_file()
                and (parent / "src" / "neuron_explorer").is_dir()
            ):
                return parent

        return Path.cwd()

    @staticmethod
    def _override(variable: str, default: Path) -> Path:
        value = os.environ.get(variable)
        return Path(value) if value else default
