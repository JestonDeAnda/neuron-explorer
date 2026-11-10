#!/usr/bin/env python3
"""Build the per-layer sweep depth report."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--layers",default="0-31"); p.add_argument("--neurons",type=int,default=384); p.add_argument("--controls",type=int,default=64); p.add_argument("--root",type=Path); p.add_argument("--runs-dir",type=Path); p.add_argument("--out",type=Path)
    a=p.parse_args(); from neuron_explorer.paths import ProjectPaths; paths=ProjectPaths.discover(a.root); a.runs_dir=str(a.runs_dir or paths.runs); a.out=str(a.out or paths.runs/"depth_report.json")
    from neuron_explorer.sweep.depth_report import run_depth_report; run_depth_report(a)
if __name__ == "__main__": main()
