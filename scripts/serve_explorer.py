#!/usr/bin/env python3
"""Serve the neuron explorer web application."""
import argparse
from pathlib import Path

def build_parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8791); p.add_argument("--root",type=Path)
    for name in ("runs","stats","captures","cache","output"): p.add_argument(f"--{name}",type=Path)
    return p

def main():
    a=build_parser().parse_args(); from neuron_explorer.explorer.cli import run_server; return run_server(a)
if __name__ == "__main__": raise SystemExit(main())
