#!/usr/bin/env python3
"""Compare tawny-owl neuron deltas with original weights."""
import argparse

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("delta"); p.add_argument("output"); a=p.parse_args()
    from neuron_explorer.tools.owl_delta_vs_base import run_owl_delta; run_owl_delta(a.delta,a.output)
if __name__ == "__main__": main()
