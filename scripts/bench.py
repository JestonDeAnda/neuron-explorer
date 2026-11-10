#!/usr/bin/env python3
"""Benchmark steering-fork throughput."""
import argparse

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model",default="Qwen/Qwen3.5-4B"); p.add_argument("--stats",required=True)
    p.add_argument("--n",type=int,default=1280); p.add_argument("--tokens",type=int,default=256); p.add_argument("--max-num-seqs",type=int,default=320)
    p.add_argument("--max-configs",type=int,default=512); p.add_argument("--gpu-mem",type=float,default=.85); p.add_argument("--only",default=""); p.add_argument("--capture-root",default="/tmp/bench-captures")
    a=p.parse_args(); from neuron_explorer.sweep.bench import run_benchmark; run_benchmark(a)
if __name__ == "__main__": main()
