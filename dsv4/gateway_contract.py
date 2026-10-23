"""中转网关口径（gateway/relay verdict）：把「平台/网关侧限制」与「模型自身能力缺陷」分开。

背景（2026-10-14）
------------------
`dsv4/` 的 61 条用例是按 **DeepSeek 官方文档** 设计的（Chat Completions / Thinking Mode /
Vision / Files API / Anthropic API / Responses API / Beta / Context Caching），但实际常常打在
**中转或托管网关**上（例如百度千帆 `qianfan.baidubce.com/v2` 托管的 `deepseek-v4-pro-0813`）。
网关会：

- 不暴露官方部分端点（`/beta/*`、`/anthropic/*`、根域部分路径）→ 404；
- 收窄能力面（只让特定模型收 `image_url`、只认自己的工具类型白名单、参数取值域不同）；
- 限流（429 `OverRateLimit`）；
- 账号/平台前提未满足（未开通对象存储 → 5xx `OptInRequired`）。

这些都不是「模型不行」。判定原则（用户口径：**非模型自身能力缺陷、不影响模型使用即可**）：

===========  ==================================================================
 归类          判法
===========  ==================================================================
 前提不满足    端点缺失 / 限流 / 账号未开通 / 该模型本就不提供这个能力面 → ``blocked``
 不影响使用    功能可用但与文档有别（取值域收窄、字段强校验、能力不透传）→ ``warning``
 影响使用      请求确实进入模型、且基础可用能力不达标 → ``failed``
===========  ==================================================================

``blocked`` 只记录、不阻断准入（见 `docs/DeepSeek专项测试用例说明` 的统一判定口径）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# —— 端点不存在：官方路径在该网关上根本没有 ——
ENDPOINT_MISSING_MARKERS = (
    "resourcenotfound", "resource not found", "no such", "unknown url",
    "not found", "invalid url", "unsupported url",
)

# —— 限流/配额：瞬时或配额类，先重试，重试耗尽记 blocked ——
RATE_LIMIT_MARKERS = (
    "overratelimit", "over rate limit", "rate limit", "ratelimited",
    "too many requests", "request limit", "tpm limit", "rpm limit", "qps",
)

# —— 账号/平台前提未满足：开通服务、余额、权限、配额 ——
ACCOUNT_PRECONDITION_MARKERS = (
    "optinrequired", "you must active", "bos service", "not activated", "not subscribed",
    "insufficient balance", "credit balance is too low", "arrearage", "quota exceeded",
    "permission denied", "access denied", "not authorized", "unpurchased", "not opened",
    # 网关侧模型可用性/访问权（2026-10-14 千帆实测：Responses 面对未注册模型返回
    # 401 "The model does not exist or you do not have access to it."——同一模型在
    # chat 面可正常调用，故这是网关该能力面未开放，不是模型能力缺陷）。
    "does not exist or you do not have access", "model does not exist",
)

# —— 网关不接受某种内容/能力：能力面在网关侧被收窄 ——
CAPABILITY_UNSUPPORTED_MARKERS = (
    "is only supported by certain models", "only supported by certain models",
    "not supported by this model", "unsupported content", "unsupported type",
    "content type is not supported", "does not support", "not supported",
)

# —— 网关自有的参数取值域（比官方更窄或不同）——
PARAM_DOMAIN_MARKERS = (
    "supported values are", "must be one of", "valid values are", "allowed values",
    "invalid value for", "supported parameters are", "must be none or", "must be one or",
    "possible values are", "expected one of", "should be one of", "only accepts",
)

# 官方文档口径：图片输入（Vision）是 `deepseek-flash` 的能力面——
# "The `deepseek-flash` model accepts images alongside text"（Vision 指南）；
# Files API 上传的图片同样"used together with the `deepseek-flash` model"。
# `deepseek-v4-pro` 官方并未声明支持图片输入，故在 Pro 上测视觉属于「该模型不提供该能力面」。
#
# 2026-10-21 修正：**不能按名字里有没有 `flash` 判定**。官方定价页「图像理解」一栏只给
# `deepseek-flash`（模型版本 DeepSeek-V4.1-Flash-1017）打勾，`deepseek-v4-pro` 打叉；
# 而 `deepseek-v4-flash-0731`（上游 id `deepseek-v4-flash-ga-260731`）是 2026-07-31 的
# 纯文本版本，早于视觉首次对外开放（2026-08-21 的 vision-exp），官方从未声明其支持图片。
# 实测（`api.oinone.top`，domestic-ark 分组）：0731 与 pro 收到图片块都返回 200 且
# prompt_tokens 都恰好 +75（基线 24 → 99），但两者都答不出图（0731 答"无法回答"/OCR 错成
# 9008，pro 的颜色也答错）⇒ **token 增量是网关侧记账产物，不能作为"图片被理解"的证据**；
# 只有 V4.1-Flash 系（同一请求 24 → 208）才真的看图。
VISION_CAPABLE_PATTERNS = (
    r"deepseek-flash$",       # 官方现行名 = DeepSeek-V4.1-Flash-1017
    r"deepseek-v4-flash$",    # 旧名，官方明确由 V4.1-Flash-1017 承接
    r"vision-exp",            # 已下线的视觉实验版，同样由最新 Flash 承接
    r"v4[._-]?1[._-]?flash",  # V4.1-Flash 家族（v4.1-flash-1017 / v4.1-flash-ark ...）
    r"1017$",                 # 上游把版本号写进模型 id 时（...261017）
)
# 视觉开放之前的 Flash 版本（纯文本），即使名字含 flash 也不得判为视觉可用。
VISION_TEXT_ONLY_PATTERNS = (
    r"0731",                  # V4-Flash-0731
    r"260731",
)

# 能力名常量
CAPABILITY_VISION = "vision"


@dataclass(frozen=True)
class GatewayIssue:
    """一条「网关/平台侧」结论：带原因与建议状态（blocked 或 warning）。"""

    kind: str
    status: str
    reason: str


def _normalize(message: str | None) -> str:
    return (message or "").lower()


def _matches(text: str, markers: tuple[str, ...]) -> str | None:
    for marker in markers:
        if marker in text:
            return marker
    return None


def model_supports_capability(model: str | None, capability: str) -> bool:
    """被测模型是否声明支持该能力面（按官方文档）。

    只有视觉需要分流：官方 Vision / Files API / 定价页把图片输入归属 `deepseek-flash`
    （现行版本 DeepSeek-V4.1-Flash-1017；退休名 `deepseek-v4-flash-vision-exp` 亦由最新
    Flash 承接）；`deepseek-v4-pro` 未声明支持图片；**`deepseek-v4-flash-0731` 这类视觉
    开放前的日期版本同样不支持**（见模块内 VISION_* 注释的实测证据）。

    判定顺序很重要：先排除「纯文本历史版本」，再认「视觉可用版本」——只按 `flash`
    子串匹配会把 0731 误判成视觉模型，进而把"模型不会看图"读成"网关/模型能力不达标"。
    """
    if capability != CAPABILITY_VISION:
        return True
    name = (model or "").lower()
    if not name:
        return False
    if any(re.search(pattern, name) for pattern in VISION_TEXT_ONLY_PATTERNS):
        return False
    return any(re.search(pattern, name) for pattern in VISION_CAPABLE_PATTERNS)


def classify_gateway_issue(
    status: int,
    message: str | None = None,
    *,
    family: str = "openai",
    capability: str | None = None,
    model: str | None = None,
) -> GatewayIssue | None:
    """把一次失败的响应归类为「网关/平台侧问题」；返回 None 表示应由模型/协议自身负责。

    参数
    ----
    status      HTTP 状态码
    message     错误文案（取 `error.message` / `message` / 原始响应文本）
    family      openai / beta / anthropic / root
    capability  该用例探测的能力面（如 ``vision``），用于判断"模型本就不提供"的情形
    model       被测模型名
    """
    text = _normalize(message)
    probe = f"{family} 端点/路径"

    if status == 404:
        return GatewayIssue(
            kind="endpoint_missing", status="blocked",
            reason=f"{probe}在该网关上不存在（HTTP 404）：该特性未被触达，属链路/网关能力面问题，"
                   "不是被测对象的行为结论；需换官方 base_url 或网关已暴露的路径复跑",
        )

    if status == 429 or _matches(text, RATE_LIMIT_MARKERS):
        return GatewayIssue(
            kind="rate_limited", status="blocked",
            reason=f"网关限流（HTTP {status}）：属配额/瞬时类，不是模型行为；重试耗尽后记 blocked",
        )

    if status >= 500 and _matches(text, ACCOUNT_PRECONDITION_MARKERS):
        return GatewayIssue(
            kind="account_precondition", status="blocked",
            reason=f"平台前提未满足（HTTP {status}）：{message or ''}".strip()[:200],
        )

    if 400 <= status < 500:
        if _matches(text, ACCOUNT_PRECONDITION_MARKERS):
            return GatewayIssue(
                kind="account_precondition", status="blocked",
                reason=f"平台前提未满足（HTTP {status}）：{message or ''}".strip()[:200],
            )
        if _matches(text, CAPABILITY_UNSUPPORTED_MARKERS):
            if capability and not model_supports_capability(model, capability):
                return GatewayIssue(
                    kind="capability_not_in_model", status="blocked",
                    reason=f"被测模型 {model} 未声明提供该能力面（HTTP {status}）：{message or ''}".strip()[:200],
                )
            return GatewayIssue(
                kind="capability_unsupported_by_gateway", status="warning",
                reason=f"该网关不支持此内容/能力（HTTP {status}）：{message or ''}".strip()[:200],
            )
        if _matches(text, PARAM_DOMAIN_MARKERS):
            return GatewayIssue(
                kind="param_domain_narrowed", status="warning",
                reason=f"网关自有取值域与官方文档不同（HTTP {status}）：{message or ''}".strip()[:200],
            )

    if status >= 500:
        return GatewayIssue(
            kind="upstream_error", status="blocked",
            reason=f"网关/上游 5xx（HTTP {status}）：属基础设施故障，不是模型行为；需重试确认",
        )

    return None
