#!/usr/bin/env python3
"""Run the explorer stdio MCP server."""
import argparse

def main():
    argparse.ArgumentParser(description="Explorer stdio MCP server").parse_args()
    from neuron_explorer.explorer.agent_mcp import run_server
    run_server()

if __name__ == "__main__":
    main()
