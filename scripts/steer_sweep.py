#!/usr/bin/env python3
"""Run a resumable neuron steering sweep."""
import argparse
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model",default="Qwen/Qwen3.5-4B"); p.add_argument("--stats",required=True); p.add_argument("--neurons",default="all"); p.add_argument("--design",default="main",choices=["calib","main","main_rel"]); p.add_argument("--dose-unit",default="absmax",choices=["absmax","rel"]); p.add_argument("--run",required=True); p.add_argument("--root",type=Path); p.add_argument("--out-dir",type=Path); p.add_argument("--max-tokens",type=int,default=256); p.add_argument("--top-k",type=int,default=5); p.add_argument("--capture-layers",default="+0,+1,+4,+8"); p.add_argument("--capture-root",type=Path); p.add_argument("--capture-layout",default="packed",choices=["packed","sharded"]); p.add_argument("--random-controls",type=int,default=256); p.add_argument("--seed",type=int,default=0); p.add_argument("--chunk",type=int,default=4000); p.add_argument("--max-configs",type=int,default=512); p.add_argument("--max-num-seqs",type=int,default=256); p.add_argument("--gpu-mem",type=float,default=.85)
 a=p.parse_args(); from neuron_explorer.paths import ProjectPaths; paths=ProjectPaths.discover(a.root); a.out_dir=str(a.out_dir or paths.runs); a.capture_root=str(a.capture_root or paths.captures)
 from neuron_explorer.sweep.steer_sweep import run_steer_sweep; run_steer_sweep(a)
if __name__ == "__main__": main()
