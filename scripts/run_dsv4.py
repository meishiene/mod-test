#!/usr/bin/env python3
"""DeepSeek V4 专项测试的 Web 入口。

用例定义与用例实现都以 dsv4 包为唯一权威：
dsv4/cases.py 提供展示元数据，dsv4/core.py 提供 MODELS/CASE_SPECS 与 handler，
dsv4/runner.py 负责装配与结果落盘。本脚本只做参数转发，不含任何用例逻辑。

（init 时代的旧副本 k3perf/deepseek_v4.py 与 scripts/run_deepseek_v4.py 已删除；
它缺 DS-V4-18 与整个 P2 段、只接受 --priority/--case，位置传参会直接中止。）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dsv4.runner import main

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        raise SystemExit(2)
