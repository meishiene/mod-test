#!/usr/bin/env python3
"""K3 供应商临时案例集（Kimi 测试模块）的 Web 入口。

权威实现是 ``kvv/temp_runner.py``；本脚本只做 sys.path 引导与参数转发，
与官方套件入口 ``scripts/run_kvv.py`` 并列。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvv.temp_runner import main

if __name__ == "__main__":
    main()
