#!/usr/bin/env python3
"""Check steering and capture equivalence."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model",default="Qwen/Qwen3.5-4B"); p.add_argument("--stats",required=True); p.add_argument("--neurons",default="2742,17,5000"); p.add_argument("--doses",default="10,-2"); p.add_argument("--prompt",default="Who are you?"); p.add_argument("--new",type=int,default=48); p.add_argument("--capture-gen",type=int,default=8); p.add_argument("--root",type=Path); p.add_argument("--capture-root",type=Path)
    a=p.parse_args(); from neuron_explorer.paths import ProjectPaths; a.capture_root=str(a.capture_root or ProjectPaths.discover(a.root).captures/"check")
    from neuron_explorer.sweep.check_equivalence import run_equivalence_check; run_equivalence_check(a)
if __name__ == "__main__": main()
