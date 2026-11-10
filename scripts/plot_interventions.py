#!/usr/bin/env python3
"""Render attribution graphs for state-swap interventions."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--k",type=int,default=512); p.add_argument("--redo",action="store_true"); p.add_argument("--model",default="Qwen/Qwen3.5-4B"); p.add_argument("--root",type=Path); p.add_argument("--input",type=Path); p.add_argument("--cache-dir",type=Path); p.add_argument("--output-dir",type=Path); p.add_argument("--bundle-out",type=Path)
    a=p.parse_args(); from neuron_explorer.paths import ProjectPaths; paths=ProjectPaths.discover(a.root); a.input=str(a.input or paths.runs/"attribution"/"swap_state.json"); a.cache_dir=str(a.cache_dir or paths.cache); a.output_dir=str(a.output_dir or paths.runs/"attribution"/"interventions"); a.bundle_out=str(a.bundle_out or paths.runs/"attribution"/"state_swap.html")
    from neuron_explorer.sweep.plot_interventions import run_plot_interventions; run_plot_interventions(a)
if __name__ == "__main__": main()
