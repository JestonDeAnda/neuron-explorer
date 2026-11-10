#!/usr/bin/env python3
"""Mirror an explorer site and its linked assets."""
import argparse

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("output"); p.add_argument("--base-url",default="https://already-clean.goodfire.com/"); p.add_argument("--workers",type=int,default=24); a=p.parse_args()
    from neuron_explorer.tools.mirror import run_mirror; run_mirror(a.output,a.base_url,a.workers)
if __name__ == "__main__": main()
