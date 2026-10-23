"""千问（Qwen）用例的测试面登记（model / verify / channel）。

与 ``qwen/relay_profile.py`` 的关系
-----------------------------------
``relay_profile.CASE_FACES`` 是**判定面**（core/trust 阻断，thinking/tools/... 只记 gap），
本文件是**执行面**（默认跑不跑）。``tests/test_faces.py`` 钉死同向：
``relay_profile`` 里非 core/trust 的用例，在这里不能是 model/verify —— 否则
「默认跑 + 失败阻断」或「默认不跑 + 勾选后不阻断」的口径会互相矛盾。
"""
from __future__ import annotations

from common.faces import FACE_CHANNEL, FACE_MODEL, FACE_VERIFY

#: 通道契约面：思考预算/上限参数、tool_choice 严格语义、采样取值域、错误信封。
#: 视觉（QWEN-V01）留在 model —— 它测的是模型会不会读图，不是网关透不透传。
CHANNEL_CASE_IDS: frozenset[str] = frozenset({
    "QWEN-T03",  # thinking_budget 限制推理长度（参数是否被网关采纳）
    "QWEN-T05",  # 思考模式 max_tokens 上限契约
    "QWEN-T06",  # 非流式推理内容载体（字段口径）
    "QWEN-A02",  # tool_choice=required 强制调用（OpenAI 严格语义）
    "QWEN-F04",  # 采样参数取值域指纹
    "QWEN-F05",  # 错误信封泄露
})

#: 验真面：身份回显、档位/上下文自述、指纹汇总。
VERIFY_CASE_IDS: frozenset[str] = frozenset({
    "QWEN-R01",
    "QWEN-R02",
    "QWEN-R03",
    "QWEN-F01",  # 响应型号回显一致性（异族替换＝信任级）
    "QWEN-F02",  # 档位与代际自述
    "QWEN-F03",  # 上下文窗口自述与真实长输入
    "QWEN-F06",  # 档位指纹汇总
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
