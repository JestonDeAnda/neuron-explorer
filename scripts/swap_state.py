#!/usr/bin/env python3
"""Explore or execute the city-state neuron swap."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("cmd",choices=["explore","swap"]); p.add_argument("--max-layer",type=int,default=12); p.add_argument("--k",type=int,default=512); p.add_argument("--root",type=Path); p.add_argument("--out",type=Path); a=p.parse_args()
    from neuron_explorer.paths import ProjectPaths; a.paths=ProjectPaths.discover(a.root); a.out=str(a.out or a.paths.runs/"attribution"/"swap_state.json")
    from neuron_explorer.sweep.swap_state import run_swap_state; run_swap_state(a)
if __name__ == "__main__": main()
