"""Fable 5 用例的测试面登记（model / verify / channel）。

分层口径
--------
- ``model``：基础能力 + 正向协议能力（思考生命周期、工具闭环、结构化输出、流式、
  视觉、长上下文、缓存命中、截断、system/stop）。这些换一家中转也该成立。
- ``verify``：AUTH-* 全套 —— 随机挑战、思考真实性、身份未降级、注入识别、零注入侧信道。
- ``channel``：P1 的 beta 特性（task budget / compaction / memory / 代码执行 / PTC /
  fallback / 1h TTL / inference_geo）+ P2 的端点与校验契约（错误对象、/v1/models、
  count_tokens、未移除参数应拒、畸形 JSON、未知字段、anthropic-version 头、响应对象结构）
  + P0 里「应被拒绝」的三条（thinking=disabled、budget_tokens、无 [DONE] 收尾语义）
  与 tool_choice=any。

判据：**能不能产出正确内容** → model；**有没有被换掉** → verify；
**端点/字段/严格语义在不在** → channel。
"""
from __future__ import annotations

from common.faces import FACE_CHANNEL, FACE_MODEL, FACE_VERIFY

CHANNEL_CASE_IDS: frozenset[str] = frozenset({
    # P0 里的负向/协议细节
    "FABLE5-E03", "FABLE5-E04", "FABLE5-E08", "FABLE5-E13",
    # S 卡：beta 特性（S11 长输出流式属模型能力，留在 model）
    "FABLE5-S01", "FABLE5-S02", "FABLE5-S03", "FABLE5-S04", "FABLE5-S05",
    "FABLE5-S06", "FABLE5-S07", "FABLE5-S08", "FABLE5-S09", "FABLE5-S10",
    # C 卡：鉴权/错误对象/已移除参数/请求体校验/端点（C13-C15 是 usage 与词表，留 model）
    "FABLE5-C01", "FABLE5-C02", "FABLE5-C03", "FABLE5-C04", "FABLE5-C05",
    "FABLE5-C06", "FABLE5-C07", "FABLE5-C08", "FABLE5-C09", "FABLE5-C10",
    "FABLE5-C11", "FABLE5-C12", "FABLE5-C16", "FABLE5-C17", "FABLE5-C18",
    "FABLE5-C19", "FABLE5-C20", "FABLE5-C21",
})

VERIFY_CASE_IDS: frozenset[str] = frozenset({
    f"AUTH-F{i:02d}" for i in range(1, 15)
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
