#!/usr/bin/env python3
"""Local LLM Benchmark Tool — entry point. The implementation lives in `llmbench/`."""

import sys

from llmbench.cli import main

if __name__ == "__main__":
    sys.exit(main())
