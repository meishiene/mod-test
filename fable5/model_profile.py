#!/usr/bin/env python3
"""按被测模型名解析线上参数形态（thinking / 模型标识），唯一权威。

**为什么需要它**：同一个 Messages 协议在不同上游对 ``thinking`` 的支持并不一致，
而且分裂是**按模型族**发生的，不是按端点发生的。实测（2026-10-20，
``results/rerun-a3b2*/``，New API 中转 → AWS Bedrock）：

- 4.5 系（haiku-4.5 / opus-4.5 / sonnet-4.5）：只认 ``{"type":"enabled","budget_tokens":N}``；
  发 ``adaptive`` 直接 400 ``adaptive thinking is not supported on this model``。
- 4.6 系（opus-4.6 / sonnet-4.6）：两档都通，但 ``adaptive`` 不透传 thinking 块
  （响应只剩 text ⇒ 取不到签名），``enabled`` 才稳定产出带签名的 thinking。
- 5 系（sonnet-5 / opus-5-5）：只认 ``{"type":"adaptive"}`` + ``output_config.effort``；
  发 ``enabled`` 直接 400 ``"***.***.enabled" is not supported for this model``。

三档映射在 Anthropic 第一方面同样成立（第一方 4.x 也接受 enabled+budget，
5 系同样接受 adaptive+effort），所以按模型名解析对两类端点是安全的。

同一文件还提供 Bedrock modelId 的归一：Bedrock 会把响应里的 ``model`` 写成
``anthropic.claude-haiku-4-5-20251001-v1:0``，与请求名 ``claude-haiku-4-5-20251001``
是**同一模型的不同写法**。归一是为了不把通道命名差异读成"换模型"。
"""
from __future__ import annotations

import re

ENABLED = "enabled"
ADAPTIVE = "adaptive"
ADAPTIVE_EFFORT = "adaptive_effort"

# thinking 档位 → 线上请求片段
_BEDROCK_PREFIX = "anthropic."
_BEDROCK_SUFFIX_RE = re.compile(r"-v\d+(?::\d+)?$")
_FAMILY_RE = re.compile(r"^claude-(opus|sonnet|haiku|fable)-(.+)$")

# 思考预算：只在 enabled 档使用；adaptive 由模型自己决定。
DEFAULT_THINKING_BUDGET = 1024
# adaptive_effort 档的 effort 等级（通道要求显式给，否则 5 系不出思考）。
DEFAULT_EFFORT = "high"


def normalize_model_id(name: str | None) -> str:
    """把 Bedrock 形态的 modelId 归一成 Anthropic 的模型名（幂等）。

    ``anthropic.claude-haiku-4-5-20251001-v1:0`` → ``claude-haiku-4-5-20251001``。
    非 Bedrock 写法原样返回（只去空白、转小写）。日期后缀 ``-20251001`` 不是版本号，
    不会被剥掉（版本后缀必须形如 ``-v1`` / ``-v1:0``）。
    """
    value = str(name or "").strip().lower()
    if value.startswith(_BEDROCK_PREFIX):
        value = value[len(_BEDROCK_PREFIX):]
    return _BEDROCK_SUFFIX_RE.sub("", value)


def same_model_id(actual: str | None, expected: str | None) -> bool:
    """响应 model 字段是否指向请求的那个模型（只做 Bedrock 归一，再做严格相等）。

    ``anthropic.claude-haiku-4-5-20251001-v1:0`` 与 ``claude-haiku-4-5-20251001`` 视为同一模型；
    但**不做**前缀/别名模糊匹配：请求 haiku 回 opus 这类真换模型必须不等，交给调用方判 failed。
    名字与 ``signature_structure.model_id_compatible``（签名内标识的前缀兼容）区分开：
    那个容忍 ``claude-fable-5`` ↔ ``claude-fable-5-1``，本函数不容忍。
    """
    left, right = normalize_model_id(actual), normalize_model_id(expected)
    return bool(left) and left == right


def resolve_thinking_mode(model: str | None) -> str:
    """按模型族解析 thinking 档位：``enabled`` / ``adaptive`` / ``adaptive_effort``。

    未识别的一律回退 ``adaptive``（与引入本函数前的默认行为一致），
    不按"哪家模型"硬编码白名单。
    """
    value = normalize_model_id(model)
    match = _FAMILY_RE.match(value)
    if not match:
        return ADAPTIVE
    family, version = match.group(1), match.group(2)
    if family == "fable":
        # fable 系只支持 adaptive（见 fable5/core.py 的思考常开约定）。
        return ADAPTIVE
    parts = version.split("-")
    major = parts[0]
    minor = parts[1] if len(parts) > 1 else ""
    if major == "4" and minor in {"5", "6"}:
        # 4.5：通道不支持 adaptive；4.6：adaptive 不透传 thinking。
        return ENABLED
    if major == "5":
        # 5 系只认 adaptive + output_config.effort。
        return ADAPTIVE_EFFORT
    return ADAPTIVE


def thinking_payload(mode: str, budget: int = DEFAULT_THINKING_BUDGET,
                     effort: str = DEFAULT_EFFORT) -> tuple[dict, dict | None]:
    """按档位返回 ``(thinking, output_config)``；非 adaptive_effort 档 ``output_config`` 为 None。"""
    if mode == ENABLED:
        return {"type": "enabled", "budget_tokens": int(budget)}, None
    if mode == ADAPTIVE_EFFORT:
        return {"type": "adaptive", "display": "summarized"}, {"effort": effort}
    return {"type": "adaptive", "display": "summarized"}, None
