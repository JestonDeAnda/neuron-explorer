#!/usr/bin/env python3
"""Embed sweep rollouts and build their UMAP layout."""
import argparse

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("jsonl"); p.add_argument("--model",default="BAAI/bge-small-en-v1.5"); p.add_argument("--chars",type=int,default=600); p.add_argument("--neighbors",type=int,default=15)
    a=p.parse_args(); from neuron_explorer.explorer.build_embeddings import run_build_embeddings; run_build_embeddings(a)
if __name__ == "__main__": main()
