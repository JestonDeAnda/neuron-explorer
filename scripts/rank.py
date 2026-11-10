#!/usr/bin/env python3
"""Rank sweep units by distinctive word clusters."""
import argparse

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("jsonl"); p.add_argument("--out",required=True); p.add_argument("--max-rel",type=float,default=1.6); p.add_argument("--top-words",type=int,default=12); p.add_argument("--show",type=int,default=30); p.add_argument("--find",default=""); a=p.parse_args()
    from neuron_explorer.sweep.rank import run_rank; run_rank(a)
if __name__ == "__main__": main()
