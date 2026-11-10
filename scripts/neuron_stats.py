#!/usr/bin/env python3
"""Compute per-neuron activation statistics."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model",default="Qwen/Qwen3.5-4B"); p.add_argument("--layers",default="22"); p.add_argument("--tokens",type=int,default=2_000_000); p.add_argument("--seq-len",type=int,default=1024); p.add_argument("--batch",type=int,default=16); p.add_argument("--seed",type=int,default=0); p.add_argument("--root",type=Path); p.add_argument("--out",type=Path)
    a=p.parse_args(); from neuron_explorer.paths import ProjectPaths; a.out=str(a.out or ProjectPaths.discover(a.root).stats/"q35-4b")
    from neuron_explorer.sweep.neuron_stats import run_neuron_stats; run_neuron_stats(a)
if __name__ == "__main__": main()
