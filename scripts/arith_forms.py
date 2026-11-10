#!/usr/bin/env python3
"""Build cross-form arithmetic attribution graphs and reports."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k",type=int,default=1024); p.add_argument("--redo",action="store_true"); p.add_argument("--model",default="Qwen/Qwen3.5-4B")
    p.add_argument("--root",type=Path); p.add_argument("--cache-dir",type=Path); p.add_argument("--output-dir",type=Path)
    a=p.parse_args()
    from neuron_explorer.paths import ProjectPaths
    paths=ProjectPaths.discover(a.root); a.cache_dir=str(a.cache_dir or paths.cache); a.output_dir=str(a.output_dir or paths.runs/"attribution"/"arith")
    from neuron_explorer.sweep.arith_forms import run_arith_forms
    run_arith_forms(a)
if __name__ == "__main__": main()
