#!/usr/bin/env python3
"""Gemini 3.x 专项测试的 Web 入口。

用例定义与实现都以 gemini3 包为唯一权威：gemini3/cases.py 提供展示元数据，
gemini3/core.py 提供 GeminiTier 矩阵、CASE_SPECS 与 handler，gemini3/runner.py
负责装配与结果落盘。本脚本只做参数转发，不含任何用例逻辑。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemini3.runner import main

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        raise SystemExit(2)
