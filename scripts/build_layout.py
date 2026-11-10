#!/usr/bin/env python3
"""Build seriated neuron-sheet layouts."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model",default="Qwen/Qwen3.5-4B"); p.add_argument("--kind",default="writes",choices=["writes","reads"]); p.add_argument("--pca",type=int,default=50); p.add_argument("--neighbors",type=int,default=30); p.add_argument("--root",type=Path); p.add_argument("--out",type=Path)
    a=p.parse_args(); from neuron_explorer.paths import ProjectPaths; paths=ProjectPaths.discover(a.root); a.out=a.out or paths.cache/f"layout_{a.kind}.npz"
    from neuron_explorer.explorer.build_layout import run_build_layout; run_build_layout(a)
if __name__ == "__main__": main()
