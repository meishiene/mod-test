"""中转验真入口：权威实现在 forensics 包，本脚本只做转发。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forensics.runner import cli

if __name__ == "__main__":
    cli()
