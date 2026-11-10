#!/usr/bin/env python3
"""Capture downstream prefill states for sweep runs."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("runs",nargs="+"); p.add_argument("--model",default="Qwen/Qwen3.5-4B"); p.add_argument("--batch",type=int,default=256); p.add_argument("--seed",type=int,default=0); p.add_argument("--root",type=Path); p.add_argument("--out-dir",type=Path)
    a=p.parse_args(); from neuron_explorer.paths import ProjectPaths; a.out_dir=str(a.out_dir or ProjectPaths.discover(a.root).captures)
    from neuron_explorer.sweep.capture_prefill import run_capture_prefill; run_capture_prefill(a)
if __name__ == "__main__": main()
