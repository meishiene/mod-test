"""Gemini 3.x（``gemini-3.8-flash`` / ``gemini-3.1-pro-preview``）专项兼容性测试套件。

权威实现是 ``gemini3`` 包：``cases.py`` 提供展示元数据，``core.py`` 提供档位矩阵、
CASE_SPECS 与 handler，``runner.py`` 负责装配与结果落盘。
"""

from .cases import CASES

__all__ = ["CASES"]
