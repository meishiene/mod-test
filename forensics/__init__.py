"""中转验真套件（跨模型、跨协议的中转取证探针）。

与各厂商专项套件的分工：专项套件回答「这个模型是否满血」，本套件回答
「这条中转链路有没有作弊痕迹」。三个用例都不依赖具体厂商，任何文字面协议
（OpenAI Chat Completions / Anthropic Messages / Gemini GenerateContent）都能跑：

- ``IMPLICIT_IDENTITY``：隐式身份指纹（对标 ztest D11，并修正其判定口径——
  族投票与档位自述分离，档位自述不可靠只留痕）；
- ``IMAGE_UNDERSTANDING``：图像内容理解（对标 ztest D13，五级看图答题，
  夹具全部内嵌生成，不依赖外网）；
- ``ERROR_ENVELOPE_LEAKAGE``：错误信封泄露（对标 ztest S4，按「高/中/低」
  三级泄露特征分级裁定，而不只是计数）。

判定遵守平台口径：信任级矛盾才 ``failed``；启发式信号 ``warning`` 不阻断；
平台前提不满足（账号/瞬时故障/不支持视觉输入）记 ``blocked`` 只记录不阻断。
"""
from forensics.cases import (
    CASES,
    ERROR_ENVELOPE_LEAKAGE,
    FORENSICS_CASE_IDS,
    FORENSICS_PROTOCOL_FAMILIES,
    FORENSICS_PROTOCOLS,
    IMPLICIT_IDENTITY,
    IMAGE_UNDERSTANDING,
    ForensicsCase,
)
from forensics.core import ForensicsConfig, ForensicsOutcome, run

__all__ = [
    "CASES",
    "ERROR_ENVELOPE_LEAKAGE",
    "FORENSICS_CASE_IDS",
    "FORENSICS_PROTOCOL_FAMILIES",
    "FORENSICS_PROTOCOLS",
    "IMPLICIT_IDENTITY",
    "IMAGE_UNDERSTANDING",
    "ForensicsCase",
    "ForensicsConfig",
    "ForensicsOutcome",
    "run",
]
