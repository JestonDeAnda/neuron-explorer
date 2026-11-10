#!/usr/bin/env python3
"""Check attribution completeness and linearity."""
import argparse

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--tiny",action="store_true"); p.add_argument("--device",default="cpu"); p.add_argument("--prompt",default="The capital of the state containing Dallas is"); p.add_argument("--targets",type=int,default=6)
    a=p.parse_args(); from neuron_explorer.sweep.check_attribution import run_attribution_check; return run_attribution_check(a)
if __name__ == "__main__": raise SystemExit(main())
