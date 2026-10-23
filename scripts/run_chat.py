#!/usr/bin/env python3
"""对话窗口入口：自由输入单轮对话，完整响应原样归档。

权威实现在 ``chatpane`` 包（cases / core / runner），本脚本只做路径转发。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatpane.runner import cli


if __name__ == "__main__":
    cli()
