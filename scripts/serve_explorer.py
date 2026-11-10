#!/usr/bin/env python3
"""Thin command wrapper for ``neuron_explorer.explorer.cli``."""

import argparse
import importlib
import sys


def main():
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        argparse.ArgumentParser(description=__doc__).parse_args()
    command = importlib.import_module("neuron_explorer.explorer.cli")
    entry = getattr(command, "main", None)
    return entry() if entry is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
