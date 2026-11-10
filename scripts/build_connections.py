#!/usr/bin/env python3
"""Build direct-path connection cache."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model",default="Qwen/Qwen3.5-4B"); p.add_argument("--k",type=int,default=24); p.add_argument("--root",type=Path); p.add_argument("--out",type=Path)
    a=p.parse_args(); from neuron_explorer.paths import ProjectPaths; paths=ProjectPaths.discover(a.root); a.out=a.out or paths.cache/"connections.npz"
    from neuron_explorer.explorer.build_connections import run_build_connections; run_build_connections(a)
if __name__ == "__main__": main()
