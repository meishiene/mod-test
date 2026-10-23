"""Seedance 视频生成套件（协议面 ``seedance``）。

对外只导出用例元数据；执行实现见 ``seedance.core`` / ``seedance.runner``。
"""
from __future__ import annotations

from .cases import CASES, SEEDANCE_CASE_ID

__all__ = ["CASES", "SEEDANCE_CASE_ID"]
