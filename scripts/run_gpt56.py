#!/usr/bin/env python3
"""GPT 专项测试的 Web 入口（GPT-5.6 / GPT-6.1）。

用例定义与用例实现都以 gpt56 包为唯一权威：
gpt56/cases.py 提供展示元数据（含 model_variant 档位），gpt56/core.py 提供 MODELS/CASE_SPECS 与 handler，
gpt56/runner.py 按用例所属档位装配对应 CASE_SPECS。本脚本只做参数转发，不含任何用例逻辑。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpt56.runner import main

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        raise SystemExit(2)
