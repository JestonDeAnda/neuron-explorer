#!/usr/bin/env python3
"""Compare local prefill captures with vLLM fork captures."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("run"); p.add_argument("fork_dir"); p.add_argument("--root",type=Path); p.add_argument("--captures",type=Path)
    a=p.parse_args(); from neuron_explorer.paths import ProjectPaths; captures=a.captures or ProjectPaths.discover(a.root).captures
    from neuron_explorer.sweep.check_capture_prefill import run_capture_check; run_capture_check(a.run,a.fork_dir,captures)
if __name__ == "__main__": main()
