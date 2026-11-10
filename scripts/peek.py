#!/usr/bin/env python3
"""Inspect sweep rollouts and unit summaries."""
import argparse

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("jsonl"); p.add_argument("--neuron",type=int); p.add_argument("--kind"); p.add_argument("--curves",action="store_true"); p.add_argument("--words",action="store_true"); p.add_argument("--max-rel",type=float,default=1.6); p.add_argument("--top",type=int,default=6); p.add_argument("--width",type=int,default=110); a=p.parse_args()
    from neuron_explorer.sweep.peek import run_peek; run_peek(a)
if __name__ == "__main__": main()
