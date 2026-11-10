#!/usr/bin/env python3
"""Merge sharded sweep runs."""
import argparse

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("inputs",nargs="+"); p.add_argument("--out",required=True); a=p.parse_args()
    from neuron_explorer.sweep.merge_runs import run_merge; run_merge(a)
if __name__ == "__main__": main()
