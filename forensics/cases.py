"""中转验真的展示元数据（Web 用例卡片只消费本文件，不含请求逻辑）。

三个用例的请求次数都是固定小探针（3 / 5 / 3 次），并发恒为 1：取证类探针
彼此有先后依赖的语义（错误信封的三个子探针共用同一 nonce 口径），串行才能
保证「每个证据行都能对上号」。
"""
from __future__ import annotations

from dataclasses import dataclass

IMPLICIT_IDENTITY = "IMPLICIT_IDENTITY"
IMAGE_UNDERSTANDING = "IMAGE_UNDERSTANDING"
ERROR_ENVELOPE_LEAKAGE = "ERROR_ENVELOPE_LEAKAGE"

FORENSICS_CASE_IDS: tuple[str, ...] = (IMPLICIT_IDENTITY, IMAGE_UNDERSTANDING, ERROR_ENVELOPE_LEAKAGE)

# ``protocols`` 是**家族**（与 platform 的 supports_test_kind 口径一致）：
# openai 家族覆盖 openai / qianfan；volcengine 是独立家族；图像面协议
# （openai_images / gemini_images）不在此列，靠协议自身的 test_kinds 白名单拦回。
FORENSICS_PROTOCOL_FAMILIES: tuple[str, ...] = ("openai", "volcengine", "anthropic", "gemini")

# 具体**协议 id** 白名单（执行器的最后一道闸）。openai_responses 是独立线格式
# （input/instructions），取证探针未实现，不放行。
FORENSICS_PROTOCOLS: tuple[str, ...] = ("openai", "volcengine", "qianfan", "qwen", "anthropic", "gemini")


@dataclass(frozen=True)
class ForensicsCase:
    case_id: str
    title: str
    description: str
    cost: str
    requests: int
    protocols: tuple[str, ...] = FORENSICS_PROTOCOL_FAMILIES
    admission: str = (
        "信任级矛盾（两条以上独立指纹与声明族冲突 / 错误信封泄露密钥等内部信息）才 failed；"
        "启发式信号（单条指纹矛盾、档位自述不符、渠道拓扑外泄）记 warning 并下调准入但不判模型失败；"
        "账号/瞬时故障/不支持视觉输入记 blocked 只记录不阻断。"
    )


CASES: dict[str, ForensicsCase] = {
    IMPLICIT_IDENTITY: ForensicsCase(
        case_id=IMPLICIT_IDENTITY,
        title="隐式身份指纹",
        description=(
            "不依赖响应签名的身份取证：① 让模型写脚本，看注释块里 Model/Author 自述"
            "（代码签名行为指纹）；② 自述最大上下文窗口，与声明模型的族区间对照；"
            "③ 自述知识截止日期，与可疑下限对照。族投票与声明模型矛盾才是硬信号；"
            "档位自述（如自称 Sonnet 而声明 Opus）只留痕——模型自述版本长期不可靠。"
        ),
        cost="低（3 次短探针）",
        requests=3,
    ),
    IMAGE_UNDERSTANDING: ForensicsCase(
        case_id=IMAGE_UNDERSTANDING,
        title="图像内容理解",
        description=(
            "五级看图答题（夹具全部内嵌生成，结果确定）：① 纯色识别；② 8×8 棋盘数深色格"
            "（计数）；③ 点阵数字 OCR（每次随机，防记忆）；④ 左右双色空间关系；"
            "⑤ 底色与图上文字冲突（GREEN 蓝字黄底，考视觉压制文本先验）。"
            "全过 passed，3～4 过 warning，≤2 过 failed；不支持视觉输入记 blocked。"
        ),
        cost="中（5 次带图请求）",
        requests=5,
    ),
    ERROR_ENVELOPE_LEAKAGE: ForensicsCase(
        case_id=ERROR_ENVELOPE_LEAKAGE,
        title="错误信封泄露",
        description=(
            "畸形 JSON / 不存在的模型 / 不存在的路径三类 4xx，扫描响应体是否外泄内部信息："
            "密钥、堆栈、内网 IP、内部文件路径、SQL 报错为高危（failed）；"
            "渠道分组拓扑、内部域名为中危（warning）；中转软件标识只留痕不扣分。"
            "另外两条独立判定：畸形 JSON 得到 2xx（解析过宽）、无通道模型得到 2xx"
            "（疑似静默回退/替换）。"
        ),
        cost="低（3 次错误探针）",
        requests=3,
    ),
}
