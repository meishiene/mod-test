"""Gemini 3.x 用例的测试面登记（model / verify / channel）。

分层要点
--------
- ``model``：基础能力 + 正向协议能力（思考档位、函数闭环、流式、视觉、长上下文、
  隐式缓存、截断、stop、systemInstruction、多模态函数响应、并行调用、长输出）。
- ``verify``：AUTH-G3-* —— 随机挑战、思考真实性、档位梯度与下限反向指纹、
  身份未降级、网关档位白名单探测、注入识别、思考签名真实性。
- ``channel``：内置工具注册（googleSearch/urlContext/codeExecution）、mode=ANY 强制调用、
  mediaResolution、candidateCount、safetySettings、显式缓存、音视频输入、思考 token 计数、
  P2 全套端点与校验契约。
"""
from __future__ import annotations

from common.faces import FACE_CHANNEL, FACE_MODEL, FACE_VERIFY

CHANNEL_CASE_IDS: frozenset[str] = frozenset({
    # P0 里的网关/严格语义/内置工具
    "G3-E03", "G3-E04", "G3-E05", "G3-E08", "G3-E09", "G3-E10", "G3-E11",
    "G3-E12", "G3-E15",
    # P1 里的档位矩阵与非主面输入
    "G3-S04", "G3-S05", "G3-S07", "G3-S08", "G3-S09", "G3-S10",
    # P2 契约与端点（C11 usageMetadata 留在 model：它是用量证据）
    "G3-C01", "G3-C02", "G3-C03", "G3-C04", "G3-C05", "G3-C06", "G3-C07",
    "G3-C08", "G3-C09", "G3-C10", "G3-C12", "G3-C13", "G3-C14", "G3-C15",
    "G3-C16",
})

VERIFY_CASE_IDS: frozenset[str] = frozenset({
    f"AUTH-G3-{i:02d}" for i in range(1, 13)
})


def _build() -> dict[str, str]:
    from .core import CASE_SPECS

    table: dict[str, str] = {}
    for spec in CASE_SPECS:
        if spec.case_id in CHANNEL_CASE_IDS:
            table[spec.case_id] = FACE_CHANNEL
        elif spec.case_id in VERIFY_CASE_IDS:
            table[spec.case_id] = FACE_VERIFY
        else:
            table[spec.case_id] = FACE_MODEL
    return table


CASE_FACES: dict[str, str] = _build()
UNREGISTERED: tuple[str, ...] = ()
