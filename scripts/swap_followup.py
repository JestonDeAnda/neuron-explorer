#!/usr/bin/env python3
"""Run follow-up controls for the Dallas-to-California state swap."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--root",type=Path); p.add_argument("--input",type=Path); a=p.parse_args()
    from neuron_explorer.paths import ProjectPaths; path=a.input or ProjectPaths.discover(a.root).runs/"attribution"/"swap_state.json"
    from neuron_explorer.sweep.swap_followup import run_swap_followup; run_swap_followup(path)
if __name__ == "__main__": main()
