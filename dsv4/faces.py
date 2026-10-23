"""DeepSeek V4 用例的测试面登记（model / verify / channel）。

与 ``dsv4/relay_profile.py`` 的分工
-----------------------------------
``relay_profile`` 回答「这条失败算阻断还是算网关缺口」，是**判定**；
本文件回答「这条用例要不要跑」，是**执行范围**。两者必须同向：

- 本文件标为 ``channel`` 的用例，在 ``relay_profile`` 里也必须是 ``relay_optional`` 面
  （``tests/test_faces.py`` 钉死这条双向一致性），否则会出现「默认不跑、勾选后也不阻断」
  或反过来「默认跑、失败就阻断」的口径撕裂。

``DS_V4_FINGERPRINT``（满血指纹）不经过本套件的 ``CASE_SPECS``（走 ``authenticity``），
在 Web 层按卡片标为 ``verify``，故不在此表内。
"""
from __future__ import annotations

from common.faces import FACE_CHANNEL, FACE_MODEL, FACE_VERIFY

#: 通道契约面：第二条协议（Anthropic / Responses）、beta（FIM / Prefix / Strict Tool）、
#: 附加端点（Files / 余额 / 模型列表）、以及「越界应被拒」这类纯校验语义。
CHANNEL_CASE_IDS: frozenset[str] = frozenset({
    # Strict Tool / Chat Prefix / Beta FIM
    "DS-V4-19", "DS-V4-20", "DS-V4-21", "DS-V4-22", "DS-V4-23",
    # Anthropic Messages 兼容面
    "DS-V4-24", "DS-V4-35", "DS-V4-48", "DS-V4-49", "DS-V4-60",
    # Responses API 面
    "DS-V4-25", "DS-V4-34", "DS-V4-50", "DS-V4-51", "DS-V4-61",
    # Files / 余额 / 模型列表
    "DS-V4-41", "DS-V4-54", "DS-V4-55", "DS-V4-43",
    # FIM 互斥族
    "DS-V4-47", "DS-V4-58",
    # 纯参数/请求体校验（合法值能否用仍留在 model 面）
    "DS-V4-30", "DS-V4-31", "DS-V4-33", "DS-V4-37", "DS-V4-38", "DS-V4-39",
    "DS-V4-40", "DS-V4-42", "DS-V4-44", "DS-V4-45", "DS-V4-53", "DS-V4-56",
    "DS-V4-57", "DS-V4-59",
})

#: 验真面：本套件的用例以能力为主，身份取证在 FINGERPRINT / 签名两张卡里。
VERIFY_CASE_IDS: frozenset[str] = frozenset()


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

#: 登记校验用：CASE_SPECS 里存在但没有面归属的用例（正常应为空）。
UNREGISTERED: tuple[str, ...] = ()
