"""GLM 5.3 用例的测试面登记（model / verify / channel）。

GLM 套件没有独立的「契约卡」，通道面散落在指纹卡里（tool_choice 严格语义、
参数是否被静默忽略、错误信封），条数不多但都会在中转链路上直接 failed，
把整批结论拖成「不准入」。这里把它们单列出来，默认不跑。
"""
from __future__ import annotations

from common.faces import FACE_CHANNEL, FACE_MODEL, FACE_VERIFY

CHANNEL_CASE_IDS: frozenset[str] = frozenset({
    "GLM-53-10",   # tool_choice=required 强制调用
    "GLM-53-F03",  # tool_choice 强制调用指纹
    "GLM-53-F06",  # 参数静默忽略检测
})

VERIFY_CASE_IDS: frozenset[str] = frozenset({
    "GLM-53-R01",
    "GLM-53-R02",
    "GLM-53-R03",
    "GLM-53-F01",  # 模型身份一致性
    "GLM-53-F08",  # 满血证据汇总
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
