#!/usr/bin/env python3
"""Build next-token steer lenses for neurons."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model",default="Qwen/Qwen3.5-4B"); p.add_argument("--layers",default="0-31"); p.add_argument("--dose",type=float,default=.8); p.add_argument("--neurons-per-batch",type=int,default=96); p.add_argument("--k",type=int,default=12); p.add_argument("--root",type=Path); p.add_argument("--out",type=Path)
    a=p.parse_args(); from neuron_explorer.paths import ProjectPaths; a.out=str(a.out or ProjectPaths.discover(a.root).stats/"steer_lens")
    from neuron_explorer.sweep.steer_lens import run_steer_lens; run_steer_lens(a)
if __name__ == "__main__": main()
