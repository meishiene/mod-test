#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from k3perf.runner import cli

if __name__ == "__main__":
    cli("L2")
