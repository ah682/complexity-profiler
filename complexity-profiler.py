#!/usr/bin/env python3
"""
Big-O Complexity Analyzer - Standalone Script Entry Point

This script can be run directly without installation:
    python complexity_profiler.py analyze merge_sort --sizes 100,1000

Or after making it executable:
    chmod +x complexity_profiler.py (Unix/Mac)
    ./complexity_profiler.py analyze merge_sort
"""

import sys
from pathlib import Path

# Add package to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent))

from complexity_profiler.cli.main import cli

if __name__ == "__main__":
    cli()
