"""GPT 专项兼容性测试（GPT-5.6 / GPT-6.1 / GPT-6 家族）。

GPT-5.6 部分：测试依据 OpenAI GPT-5.6 官方 API 文档（《Model guidance》《Reasoning models》
《Text generation》），使用原始 HTTP 请求，避免 SDK 对扩展字段的过滤。

GPT-6.1 部分：测试依据 GPT-6.1 Sol（2026-09-29 发布的单快照 gpt-6.1-sol）的模型页与迁移指南，
**独立判定、不复用 5.6 判据**——6.1 移除了 none/minimal 档位、把工具面收敛到 Responses API、
并把缓存口径改为 ttl 仅 30m + 最小前缀 1024 token，详见下方常量区与
``docs/GPT6.1专项测试用例设计.md``。

GPT-6 家族部分（2026-09-22 发布 gpt-6-sol / gpt-6-astra，luna 后续补齐）：**同一家族的三个档位
契约互不相同，不能互套判据**（详见 ``ModelTier`` 能力矩阵与 ``docs/GPT6家族专项测试用例设计.md``）：
- ``gpt-6-astra``：effort 无 none/minimal（low 最低），Chat 通道不承载 function tools（须走 Responses）
  —— 与 6.1 同侧；
- ``gpt-6-sol`` / ``gpt-6-luna``：effort 六档含 none，Chat 函数调用**仅在 effort=none** 下可用
  —— 与 6.1 **方向相反**，照搬 6.1 的"接受 none 即非 6.1"会把正确的 Sol 判成失败；
- 家族共用（可直接复用 6.1 判据）：1.05M 上下文 / 128K 最大输出 / ``prompt_cache_options.ttl`` 仅 "30m" /
  最小可缓存前缀 1024 token / Responses 内置工具面。

三档通过 ``CaseSpec.variant`` 严格分区（见 ``run()``），档位差异由 ``ModelTier`` 矩阵驱动。

反向指纹（"接受 X ⇒ 后端不是该模型"）的**归因纪律**（2026-10-10 实测教训）：中转层可能用与模型无关的
统一枚举白名单放行档位，或把 Chat 请求翻译成 Responses 后调用工具，这两类都会让"该被拒的没被拒"。
因此这类指纹只在能区分**模型口径**与**网关口径**时才允许判 failed：
- 档位下限：先探测网关声明的合法档位集合（``gateway_effort_universe``），若其中含该模型官方已移除的档位
  ⇒ 判定"请求侧校验与模型无关"，指纹失效，记 blocked；
- Chat 工具面：若响应带协议改写证据（``usage.billing_usage.source=oai_responses``）⇒ 属信任级协议改写，
  记 failed 但归因于网关翻译，**不得**表述为"后端不是该模型"。

能力面依据（2026-09 文档检索）：
- 模型：gpt-5.6-sol / gpt-5.6-terra / gpt-5.6-luna；`gpt-5.6` 别名默认路由到 sol。
- 主推 Responses API（/v1/responses），Chat Completions（/v1/chat/completions）兼容保留。
- reasoning.effort 六档 none/low/medium/high/xhigh/max，缺省 medium；reasoning.mode=pro；
  reasoning.context（auto/all_turns/current_turn）；reasoning.summary。
- previous_response_id 状态续接；store=false 时以 encrypted_content 回放加密推理。
- 隐式缓存（cached_tokens / cache_write_tokens）；prompt_cache_options.mode/ttl。
- Programmatic Tool Calling（PTC）；Structured Outputs（text.format json_schema strict）。

与 dsv4 不同：GPT-5.6 无 beta / anthropic 特殊端点，所有请求都挂在 openai 家族根
（{base}/v1）下，不存在"剥版本后缀"的地址派生问题。

判定口径补充（2026-09-24）：上游 5xx / 连接错误属瞬时故障，先按 GPT56Config.max_retries
退避重试，重试耗尽仍失败则该用例记 `blocked`（平台级前提不满足），不记 `failed`。
4xx 与结构化 `model_not_found` 是确定性结论，既不重试也不降级。
"""
from __future__ import annotations

import asyncio
import base64
import html
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
from common.png_fixture import png_data_url
from common.test_framework import HttpExecutor, ResultLog

MODELS = ("gpt-5.6",)

# 模型身份令牌白名单：别名 gpt-5.6 官方承诺路由到 gpt-5.6-sol，
# 因此响应 model 落在任一 GPT-5.6 令牌上都视为身份一致。
MODEL_TOKENS = ("gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")

# ---------------------------------------------------------------------------
# GPT-6.1（2026-09-29 DevDay，单快照 gpt-6.1-sol）——能力面依据（2026-10 检索）
# ---------------------------------------------------------------------------
# 与 GPT-5.6 的关键差异（决定了 6.1 用例为何不能直接套用 5.6 判据）：
#   1. 模型 ID 为单快照 ``gpt-6.1-sol``（点号非连字符），无 ``gpt-6.1`` 组合别名；
#      ``gpt-61-sol`` / ``gpt-6-1-sol`` 不解析。GPT-6.1 Astra 已取消、Terra/Luna 未发布。
#   2. ``reasoning.effort`` 只接受 low/medium(默认)/high/xhigh/max——**none 与 minimal 被移除**，
#      400 且 param=reasoning.effort（"Unsupported value: 'none' is not supported with the
#      'gpt-6.1-sol' model. Supported values are: ..."）。这是相对 5.6（六档含 none）的破坏性变更。
#   3. 工具调用**只在 Responses API**：Chat Completions 仅承载无 tools 的请求，
#      带 tools 返回 400（"Function tools with reasoning_effort are not supported for
#      gpt-6.1-sol in /v1/chat/completions. To use function tools, use /v1/responses..."）。
#   4. 采样参数（temperature/top_p/top_logprobs/logprobs）只在 effort=none 下允许；
#      6.1 无 none ⇒ 这些参数在 Chat 上不可用（带上即失败）。
#   5. 缓存：``prompt_cache_retention`` 改名 ``prompt_cache_options.ttl``，**仅接受 "30m"**；
#      最小可缓存前缀 1024 token；缓存写入按 1.25× 计价；写操作在 usage 里以
#      ``cache_write_tokens`` / 命中以 ``cached_tokens`` 体现。
#   6. 切档不失效缓存：把 ``configuration_update`` 作为 input 项追加（顶层 reasoning.effort 保持不变），
#      相邻 configuration_update 项被拒，且不能与自动压缩/截断同用。
#   7. 上下文 1,050,000 / 最大输入 922,000 / 最大输出 128,000；>272K 输入整单重定价。
#   8. 模态 text+image 入、text 出；不支持 audio/video、微调、Realtime、Assistants。
GPT61_MODEL_TOKENS = ("gpt-6.1-sol", "gpt-6.1")
# 权威单快照 ID：AUTH 指纹要求逐字命中（别名 gpt-6.1 仅留痕）。
GPT61_FLAGSHIP = "gpt-6.1-sol"
# 6.1 合法档位（none/minimal 已被移除）。
GPT61_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# 6.1 必须拒绝的档位——这是区分"真 6.1"与"5.6/6.0 套壳"的硬信号。
GPT61_REJECTED_EFFORTS = ("none", "minimal")
# Responses 内置工具面（P1 逐项注册）。
GPT61_BUILTIN_TOOLS = ("web_search", "file_search", "code_interpreter", "computer_use",
                       "apply_patch", "hosted_shell", "image_generation", "skills",
                       "tool_search", "mcp")

# ---------------------------------------------------------------------------
# 输出上限下限（2026-10-10）
# ---------------------------------------------------------------------------
# 思考无法真正关闭：GPT-6.x 只有 effort=none 才关思考，而多数内容类用例按档位默认
# （low/medium）或省缺执行；推理 token 计入 max_output_tokens，小上限会被思考吃满，
# 响应以 status=incomplete 收尾、**零可见正文**——需要正文/工具调用/暗号复述的判据
# 必然假失败（同类教训见 k3perf/cases.py 与 volcperf/cases.py:T0）。
# 因此凡"需要可见正文或完整收尾"的用例，输出上限一律不低于本值；刻意验证截断的
# 用例（GPT56-20 / GPT61-30，max_output_tokens=16）与"超上限应被拒"（200000）不受此限。
CONTENT_MIN_OUTPUT_TOKENS = 2048

# ---------------------------------------------------------------------------
# 鹈鹕骑车 SVG（三档共用，2026-10-12）
# ---------------------------------------------------------------------------
# 提示词**逐字冻结**：需求方明确要求"提示词一点都不要动"，故保留原文（中文全角逗号、
# "SVG" 大小写、"不要read" 与 "不要调用imagegen" 之间无空格、结尾无句号）。
# tests/test_gpt56.py 对该字符串做了逐字断言——它是跨批次可比性的前提，
# 顺手"优化"文案（补空格/换标点/加句号）会让历史结论与新批次不可比。
PELICAN_PROMPT = "你给我画一个鹈鹕骑自行车的SVG，直接输出完整SVG，不要read，不要调用imagegen"
# SVG 是长产物：推理 token 与正文共享同一预算（见 CONTENT_MIN_OUTPUT_TOKENS 的教训），
# 预算给小了必然截断成半截 SVG ⇒ 产物不可查看。这里给足，宁可留空也不误判。
PELICAN_MAX_OUTPUT_TOKENS = 8192
# 一次跑多轮（2026-10-12）：轮数 × 并发由界面/CLI 传入，同一提示词重复 N 次。
# 上限刻意远低于通用压测（100000/1000）：每轮都是一次 8192 token 的长产物直出，
# 200 轮已是极重的批次；并发上限按同一量级收紧，避免把中转网关打挂。
PELICAN_MAX_REQUESTS = 200
PELICAN_MAX_CONCURRENCY = 32
# 单轮（轮数=1）时的产物文件名——老批次与单轮体验完全不变（web/server.py:VIEWABLE_ARTIFACTS
# 的固定白名单即这三个名字）。
PELICAN_ARTIFACT_FILES: tuple[str, ...] = ("pelican.html", "pelican.svg", "pelican_response.md")
PELICAN_ARTIFACT_TITLES = {
    "pelican.html": "鹈鹕骑车 SVG（渲染视图，浏览器直接打开）",
    "pelican.svg": "鹈鹕骑车 SVG（原始 SVG 文件）",
    "pelican_response.md": "模型完整响应原文",
}
# 多轮时：`pelican.html` 变成**总览页**（各轮渲染视图排在一起），各成功轮的原始 SVG 落成
# `pelican-<轮次>.svg`（总览页里的相对链接直达，也在整批下载包内），响应原文按轮分段写进
# `pelican_response.md`。`pelican.svg` 不再写——它只代表"某一轮"，留着会误导。
PELICAN_ROUND_SVG_PATTERN = re.compile(r"pelican-\d+\.svg")
# 多轮批次里的产物清单（报告/记录页据此展示）；单轮仍用 PELICAN_ARTIFACT_FILES。
PELICAN_MULTI_ARTIFACT_FILES: tuple[str, ...] = ("pelican.html", "pelican_response.md")

# ---------------------------------------------------------------------------
# GPT-6 家族档位能力矩阵（2026-10 官方口径 + 2026-10-10 实测）
# ---------------------------------------------------------------------------
# 为什么需要矩阵：同一家族的三个档位在"档位相关"契约上互不相同，判据期望值必须按**被测模型**
# 解析，不能按套件写死。把型号画像集中在这里，handler 只问矩阵。
#
#   档位            effort 档位集              Chat 函数调用        采样参数
#   gpt-6.1-sol     low..max（无 none/minimal） 拒绝（须 Responses） 永不可用
#   gpt-6-astra     low..max（无 none/minimal） 拒绝（须 Responses） 永不可用
#   gpt-6-sol       none..max（六档含 none）    仅 effort=none 可用  仅 effort=none 可用
#   gpt-6-luna      none..max（六档含 none）    仅 effort=none 可用  仅 effort=none 可用
#
# 实测留痕（2026-10-10，mux.la / group muxla_company）：
#   * /v1/models 只有 8 个模型，**无 gpt-6-luna**（503 model_not_found "No available channel"）
#     ⇒ luna 按 available=False 留位不可选，误派发整批 blocked（不记模型失败）。
#   * reasoning.effort 的请求侧枚举白名单是**统一七档**（对 gpt-6.1-sol 与 gpt-6-sol 返回同一份
#     "Supported values"），且 none/minimal 对三个档位全部 200 ⇒ 档位指纹在中转层失效。
#   * Chat + tools + tool_choice=required 对 gpt-6.1-sol 也 200 且真发 tool_calls，响应同时带
#     usage.billing_usage.source="oai_responses" ⇒ Chat 被翻译成 Responses，属协议改写。
CHAT_TOOLS_REJECTED = "rejected"        # Chat 通道不承载 function tools（须走 Responses）
CHAT_TOOLS_NONE_ONLY = "none_only"      # 仅 effort=none 下允许 Chat 函数调用
SAMPLING_REJECTED = "rejected"          # 采样参数永不可用（无 none 档位）
SAMPLING_NONE_ONLY = "none_only"        # 采样参数仅在 effort=none 下允许


@dataclass(frozen=True)
class ModelTier:
    """单个模型档位的契约画像：决定"档位相关"判据的期望值。"""

    model_id: str
    label: str
    variant: str
    # 官方合法档位集（按官方口径，不是网关口径）。
    effort_levels: tuple[str, ...]
    # 官方明确移除、必须被拒的档位（空元组表示该档位集包含 none/minimal）。
    rejected_efforts: tuple[str, ...]
    chat_tools: str
    sampling: str
    # 官方默认档位；None 表示文档未明示（此时只否决"落到被移除档位"，不否决具体值）。
    default_effort: str | None = "medium"
    # 官方承诺可路由到该档位的别名（响应 model 回显别名时留痕，不否决）。
    aliases: tuple[str, ...] = ()
    # 目标网关是否已开放该档位通道。
    available: bool = True
    unavailable_reason: str = ""
    note: str = ""

    @property
    def accepted_tokens(self) -> tuple[str, ...]:
        """身份判定可接受的响应 model 令牌：权威 ID + 官方别名。"""
        return (self.model_id, *self.aliases)


GPT6_EFFORT_FLOOR = ("none", "minimal")

MODEL_TIERS: tuple[ModelTier, ...] = (
    ModelTier(
        model_id="gpt-6.1-sol", label="GPT-6.1 Sol", variant="gpt6.1",
        effort_levels=("low", "medium", "high", "xhigh", "max"),
        rejected_efforts=GPT6_EFFORT_FLOOR,
        chat_tools=CHAT_TOOLS_REJECTED,
        sampling=SAMPLING_REJECTED,
        aliases=("gpt-6.1",),
        note="单快照；无 gpt-6.1 组合别名（别名回显仅留痕）",
    ),
    ModelTier(
        model_id="gpt-6-sol", label="GPT-6 Sol", variant="gpt6",
        effort_levels=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        rejected_efforts=(),
        chat_tools=CHAT_TOOLS_NONE_ONLY,
        sampling=SAMPLING_NONE_ONLY,
        note="六档含 none；Chat 函数调用与采样参数仅在 effort=none 下可用",
    ),
    ModelTier(
        model_id="gpt-6-astra", label="GPT-6 Astra", variant="gpt6",
        effort_levels=("low", "medium", "high", "xhigh", "max"),
        rejected_efforts=GPT6_EFFORT_FLOOR,
        chat_tools=CHAT_TOOLS_REJECTED,
        sampling=SAMPLING_REJECTED,
        default_effort=None,
        note="最高能力档；无 none/minimal，工具面须走 Responses；默认档位文档未明示",
    ),
    ModelTier(
        model_id="gpt-6-luna", label="GPT-6 Luna", variant="gpt6",
        effort_levels=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        rejected_efforts=(),
        chat_tools=CHAT_TOOLS_NONE_ONLY,
        sampling=SAMPLING_NONE_ONLY,
        available=False,
        unavailable_reason="2026-10-10 实测目标网关无 gpt-6-luna 通道（503 model_not_found）",
        note="与 sol 同契约；网关未开放通道，留位不可选",
    ),
)

TIER_BY_MODEL: dict[str, ModelTier] = {tier.model_id: tier for tier in MODEL_TIERS}

# 套件档位 → 权威档位画像：被测模型名无法识别时（网关自定义 ID）回落，避免判据悬空。
VARIANT_FALLBACK_TIER: dict[str, ModelTier] = {
    "gpt6.1": TIER_BY_MODEL["gpt-6.1-sol"],
    "gpt6": TIER_BY_MODEL["gpt-6-sol"],
}

# Web 下拉用：GPT-6 家族可派发 / 全部型号（不可用的置灰展示，说明"留位不可选"）。
GPT6_MODELS = tuple(t.model_id for t in MODEL_TIERS if t.variant == "gpt6" and t.available)
GPT6_ALL_MODELS = tuple(t.model_id for t in MODEL_TIERS if t.variant == "gpt6")


def tier_for_model(model: str) -> ModelTier | None:
    """按被测模型名解析档位画像；无法识别返回 None（由调用方决定回落策略）。

    只做**精确匹配 + 前缀匹配**，不做模糊包含：``gpt-6-1-sol`` 这类连字符变体必须落到 None，
    否则会把"模型 ID 严格解析"这一条判据架空。
    """
    name = (model or "").strip().lower()
    if not name:
        return None
    if name in TIER_BY_MODEL:
        return TIER_BY_MODEL[name]
    for tier in MODEL_TIERS:
        if name.startswith(tier.model_id):
            return tier
    return None


REASONING_CONTEXT_MODES = {"auto", "all_turns", "current_turn"}


@dataclass(frozen=True)
class GPT56Config:
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 300.0
    verify_ssl: bool = True
    trust_env: bool = False
    # 瞬时故障（上游 5xx / 连接错误）的重试次数与退避基数：网关抖动不该被记成模型失败。
    # 重试耗尽后仍失败 → 用例按 blocked 收口（见 _transient_reason）。
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0


@dataclass
class ApiResponse:
    status: int
    body: dict[str, Any] | None
    text: str
    elapsed: float
    headers: dict[str, str] = field(default_factory=dict)
    upstream_request_id: str | None = None
    stream: dict[str, Any] | None = None


@dataclass
class CaseResult:
    case_id: str
    name: str
    model: str
    passed: bool
    status: str = "passed"
    details: str = ""
    duration_seconds: float = 0.0
    observations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    name: str
    priority: str
    handler: str
    # 用例归属的模型档位：``gpt5.6`` / ``gpt6.1``；run() 按该字段装配，避免串档。
    variant: str = "gpt5.6"


CASE_SPECS = (
    # ---- P0 核心专项（16 条）----
    CaseSpec("GPT56-01", "模型别名可调用", "P0", "basic"),
    CaseSpec("GPT56-02", "默认思考档位", "P0", "default_effort"),
    CaseSpec("GPT56-03", "effort=none 关闭思考", "P0", "effort_none"),
    CaseSpec("GPT56-04", "思考强度档位接受", "P0", "effort_levels"),
    CaseSpec("GPT56-05", "reasoning summary 摘要", "P0", "reasoning_summary"),
    CaseSpec("GPT56-06", "reasoning.context 回显", "P0", "reasoning_context"),
    CaseSpec("GPT56-07", "工具调用闭环（Responses）", "P0", "tool_loop"),
    CaseSpec("GPT56-08", "Structured Outputs", "P0", "json_schema_output"),
    CaseSpec("GPT56-09", "Chat Completions 兼容", "P0", "chat_compat"),
    CaseSpec("GPT56-10", "previous_response_id 多轮状态", "P0", "previous_response_state"),
    CaseSpec("GPT56-11", "store=false 加密推理回放", "P0", "encrypted_replay"),
    CaseSpec("GPT56-12", "隐式缓存命中", "P0", "cache_hit"),
    CaseSpec("GPT56-13", "小输出截断可检测", "P0", "truncation"),
    CaseSpec("GPT56-14", "Responses 流式", "P0", "responses_stream"),
    CaseSpec("GPT56-15", "Chat 流式", "P0", "chat_stream"),
    CaseSpec("GPT56-16", "instructions 系统指令生效", "P0", "instructions"),
    CaseSpec("GPT56-34", "三档模型均可调用", "P0", "tier_models"),
    CaseSpec("GPT56-35", "别名路由到旗舰档", "P0", "alias_flagship"),
    CaseSpec("GPT56-36", "图像识别（内联 base64）", "P0", "image_recognition"),
    CaseSpec("GPT56-37", "图像 detail 档位", "P0", "image_detail_tiers"),
    CaseSpec("GPT56-40", "后台模式（background）", "P0", "background_mode"),
    CaseSpec("GPT56-41", "响应对象结构契约", "P0", "response_object_contract"),
    # ---- P1 增强专项（7 条）----
    CaseSpec("GPT56-17", "pro 模式可用", "P1", "pro_mode"),
    CaseSpec("GPT56-18", "reasoning.context=all_turns 持久化", "P1", "context_all_turns"),
    CaseSpec("GPT56-19", "Programmatic Tool Calling 注册", "P1", "ptc"),
    CaseSpec("GPT56-20", "显式缓存选项", "P1", "explicit_cache"),
    CaseSpec("GPT56-21", "缓存 TTL 选项", "P1", "cache_ttl"),
    CaseSpec("GPT56-22", "非法 json_schema 拒绝", "P1", "strict_invalid_schema"),
    CaseSpec("GPT56-23", "未知模型拒绝", "P1", "unknown_model"),
    CaseSpec("GPT56-49", "显式缓存断点（增量判据）", "P1", "explicit_cache_breakpoint"),
    CaseSpec("GPT56-53", "multi-agent beta", "P1", "multi_agent"),
    # ---- P2 契约与健壮性（10 条）----
    CaseSpec("GPT56-24", "Responses Usage 完整性", "P2", "responses_usage"),
    CaseSpec("GPT56-25", "Chat Usage 完整性", "P2", "chat_usage"),
    CaseSpec("GPT56-26", "Chat 流式 Usage chunk", "P2", "chat_stream_usage"),
    CaseSpec("GPT56-27", "非法 temperature 拒绝", "P2", "temperature_invalid"),
    CaseSpec("GPT56-28", "非法 Messages 拒绝", "P2", "messages_invalid"),
    CaseSpec("GPT56-29", "Tool Choice none", "P2", "tool_choice_none"),
    CaseSpec("GPT56-30", "Tool Choice required", "P2", "tool_choice_required"),
    CaseSpec("GPT56-31", "Stop 停止序列", "P2", "stop_sequence"),
    CaseSpec("GPT56-32", "max_completion_tokens 边界接受", "P2", "max_completion_tokens"),
    CaseSpec("GPT56-33", "非法 input 拒绝", "P2", "invalid_input"),
    CaseSpec("GPT56-54", "无效 API Key 鉴权契约", "P2", "invalid_key_auth"),
    CaseSpec("GPT56-55", "错误对象结构契约", "P2", "error_object_contract"),
    CaseSpec("GPT56-56", "模型列表端点与三档一致性", "P2", "models_list"),
    CaseSpec("GPT56-57", "输入 token 计数端点", "P2", "input_tokens_endpoint"),
    CaseSpec("GPT56-61", "流式语义事件与无 DONE", "P2", "stream_semantic_events"),
    # ---- 产物类（1 条，三档共用同一 handler）----
    CaseSpec("GPT56-62", "鹈鹕骑车 SVG 生成（产物可直接查看）", "PELICAN", "pelican_svg"),
    # ---- FINGERPRINT 满血真实性指纹（6 条）----
    CaseSpec("AUTH-G01", "随机动态工具诱导调用", "AUTH", "auth_random_tool"),
    CaseSpec("AUTH-G02", "随机 Structured Output 绑定", "AUTH", "auth_random_json"),
    CaseSpec("AUTH-G03", "随机多轮记忆", "AUTH", "auth_random_memory"),
    CaseSpec("AUTH-G04", "思考真实性（summary）", "AUTH", "auth_reasoning_summary"),
    CaseSpec("AUTH-G05", "流式事件完整性", "AUTH", "auth_stream_events"),
    CaseSpec("AUTH-G06", "effort 静默吞参对照", "AUTH", "auth_effort_contrast"),
    CaseSpec("AUTH-G07", "中转注入识别（instructions 回显）", "AUTH", "auth_injection"),
    CaseSpec("AUTH-G08", "非标准字段检测", "AUTH", "auth_nonstandard_fields"),
    CaseSpec("AUTH-G10", "档位能力梯度", "AUTH", "auth_tier_gradient"),
    CaseSpec("AUTH-G11", "previous_response_id 真实性", "AUTH", "auth_response_id_authenticity"),
    CaseSpec("AUTH-G12", "Chat 协议未被改写", "AUTH", "auth_chat_not_translated"),
    # ==================================================================
    # GPT-6.1（gpt-6.1-sol 单快照）：独立编号 GPT61-* / AUTH-G61-*，
    # 与 5.6 用例并列但独立判定（不复用 5.6 判据）。
    # ---- P0 核心专项（19 条）----
    CaseSpec("GPT61-01", "模型身份（单快照 gpt-6.1-sol）", "P0", "g61_identity", "gpt6.1"),
    CaseSpec("GPT61-02", "默认思考档位 medium", "P0", "g61_default_effort", "gpt6.1"),
    CaseSpec("GPT61-03", "effort=none 必须被拒", "P0", "g61_effort_none_rejected", "gpt6.1"),
    CaseSpec("GPT61-04", "effort=minimal 必须被拒", "P0", "g61_effort_minimal_rejected", "gpt6.1"),
    CaseSpec("GPT61-05", "五档合法（无 none/minimal）", "P0", "g61_effort_levels", "gpt6.1"),
    CaseSpec("GPT61-06", "Chat 纯文本可用", "P0", "g61_chat_text", "gpt6.1"),
    CaseSpec("GPT61-07", "Chat 工具面被拒（工具只在 Responses）", "P0", "g61_chat_tools_rejected", "gpt6.1"),
    CaseSpec("GPT61-08", "采样参数与 effort 互斥", "P0", "g61_sampling_params_rejected", "gpt6.1"),
    CaseSpec("GPT61-09", "工具调用闭环（Responses）", "P0", "g61_tool_loop", "gpt6.1"),
    CaseSpec("GPT61-10", "Structured Outputs", "P0", "g61_json_schema_output", "gpt6.1"),
    CaseSpec("GPT61-11", "previous_response_id 多轮状态", "P0", "g61_previous_response_state", "gpt6.1"),
    CaseSpec("GPT61-12", "store=false 加密推理回放", "P0", "g61_encrypted_replay", "gpt6.1"),
    CaseSpec("GPT61-13", "显式缓存断点（≥1024 前缀，增量判据）", "P0", "g61_explicit_cache_breakpoint", "gpt6.1"),
    CaseSpec("GPT61-14", "configuration_update 切档不失效缓存", "P0", "g61_config_update_cache", "gpt6.1"),
    CaseSpec("GPT61-15", "输出截断可检测", "P0", "g61_truncation", "gpt6.1"),
    CaseSpec("GPT61-16", "Responses 流式（typed SSE）", "P0", "g61_responses_stream", "gpt6.1"),
    CaseSpec("GPT61-17", "instructions 系统指令生效", "P0", "g61_instructions", "gpt6.1"),
    CaseSpec("GPT61-18", "图像输入（内联 base64）", "P0", "g61_image_recognition", "gpt6.1"),
    CaseSpec("GPT61-19", "响应对象结构契约", "P0", "g61_response_object_contract", "gpt6.1"),
    # ---- P1 增强专项（6 条）----
    CaseSpec("GPT61-20", "Responses 内置工具逐项注册", "P1", "g61_builtin_tools", "gpt6.1"),
    CaseSpec("GPT61-21", "allowed_tools 工具开关", "P1", "g61_allowed_tools", "gpt6.1"),
    CaseSpec("GPT61-22", "显式缓存 TTL 仅 30m", "P1", "g61_cache_ttl", "gpt6.1"),
    CaseSpec("GPT61-23", "configuration_update 相邻项被拒", "P1", "g61_config_update_adjacent", "gpt6.1"),
    CaseSpec("GPT61-24", "/responses/compact 压缩端点", "P1", "g61_compact_endpoint", "gpt6.1"),
    CaseSpec("GPT61-25", "模型 ID 严格解析", "P1", "g61_model_id_strictness", "gpt6.1"),
    # ---- P2 契约与健壮性（11 条）----
    CaseSpec("GPT61-26", "Responses Usage 契约", "P2", "g61_responses_usage", "gpt6.1"),
    CaseSpec("GPT61-27", "Chat Usage 契约", "P2", "g61_chat_usage", "gpt6.1"),
    CaseSpec("GPT61-28", "Chat 流式 Usage chunk", "P2", "g61_chat_stream_usage", "gpt6.1"),
    CaseSpec("GPT61-29", "非法 effort 错误对象四字段", "P2", "g61_effort_error_contract", "gpt6.1"),
    CaseSpec("GPT61-30", "非法 input 拒绝", "P2", "g61_invalid_input", "gpt6.1"),
    CaseSpec("GPT61-31", "max_output_tokens 超上限拒绝", "P2", "g61_max_output_tokens_rejected", "gpt6.1"),
    CaseSpec("GPT61-32", "无效 API Key 鉴权契约", "P2", "g61_invalid_key_auth", "gpt6.1"),
    CaseSpec("GPT61-33", "未知模型拒绝", "P2", "g61_unknown_model", "gpt6.1"),
    CaseSpec("GPT61-34", "/models 端点与 6.1 身份", "P2", "g61_models_list", "gpt6.1"),
    CaseSpec("GPT61-35", "输入 token 计数端点", "P2", "g61_input_tokens_endpoint", "gpt6.1"),
    CaseSpec("GPT61-36", "流式语义事件与无 DONE", "P2", "g61_stream_semantic_events", "gpt6.1"),
    # ---- 产物类（1 条，复用 5.6 的 pelican_svg handler，提示词与判据跨档一致）----
    CaseSpec("GPT61-37", "鹈鹕骑车 SVG 生成（产物可直接查看）", "PELICAN", "pelican_svg", "gpt6.1"),
    # ---- FINGERPRINT 满血真实性指纹（12 条）----
    CaseSpec("AUTH-G61-01", "模型身份令牌（gpt-6.1-sol）", "AUTH", "g61_auth_identity", "gpt6.1"),
    CaseSpec("AUTH-G61-02", "effort 下限指纹（接受 none 即非 6.1）", "AUTH", "g61_auth_effort_floor", "gpt6.1"),
    CaseSpec("AUTH-G61-03", "Chat 工具面指纹（Chat+tools 成功即非 6.1）", "AUTH", "g61_auth_chat_tool_plane", "gpt6.1"),
    CaseSpec("AUTH-G61-04", "随机动态工具诱导调用", "AUTH", "g61_auth_random_tool", "gpt6.1"),
    CaseSpec("AUTH-G61-05", "随机 Structured Output 绑定", "AUTH", "g61_auth_random_json", "gpt6.1"),
    CaseSpec("AUTH-G61-06", "随机多轮记忆", "AUTH", "g61_auth_random_memory", "gpt6.1"),
    CaseSpec("AUTH-G61-07", "思考真实性（low 仍产生 reasoning）", "AUTH", "g61_auth_reasoning", "gpt6.1"),
    CaseSpec("AUTH-G61-08", "缓存计费证据", "AUTH", "g61_auth_cache_evidence", "gpt6.1"),
    CaseSpec("AUTH-G61-09", "中转注入识别（instructions 回显）", "AUTH", "g61_auth_injection", "gpt6.1"),
    CaseSpec("AUTH-G61-10", "非标准字段检测", "AUTH", "g61_auth_nonstandard_fields", "gpt6.1"),
    CaseSpec("AUTH-G61-11", "previous_response_id 真实性", "AUTH", "g61_auth_response_id_authenticity", "gpt6.1"),
    CaseSpec("AUTH-G61-12", "Chat 协议未被改写", "AUTH", "g61_auth_chat_not_translated", "gpt6.1"),
    # ==================================================================
    # GPT-6 家族（gpt-6-sol / gpt-6-astra / gpt-6-luna）：编号 GPT6-* / AUTH-G6-*。
    #
    # 与 6.1 的关系：**共用一个 handler 池，但期望值不共享**。家族共用项（缓存机制、
    # Responses 内置工具面、结构化输出、多轮状态、流式、图像、响应对象契约、通用 AUTH）
    # 直接复用 6.1 的 handler；档位相关项（档位下限、Chat 工具面、采样参数、身份令牌）
    # 由 handler 从 ``ModelTier`` 矩阵按被测模型解析期望值——所以同一份 handler 对
    # sol/luna 判"必须接受 none"、对 astra 判"必须拒绝 none"。
    #
    # 为什么不能只改 model 复用 6.1 套件：6.1 的判据把"接受 none"当成 failed 的身份指纹，
    # 而 sol/luna 官方支持 none，直接复用会把正确后端判成失败（2026-10-10 复盘结论）。
    # ---- P0 核心专项（19 条）----
    CaseSpec("GPT6-01", "模型身份（权威 ID 逐字命中）", "P0", "g61_identity", "gpt6"),
    CaseSpec("GPT6-02", "默认思考档位", "P0", "g61_default_effort", "gpt6"),
    CaseSpec("GPT6-03", "effort=none 档位下限判定（双向）", "P0", "g61_effort_none_rejected", "gpt6"),
    CaseSpec("GPT6-04", "effort=minimal 档位下限判定（双向）", "P0", "g61_effort_minimal_rejected", "gpt6"),
    CaseSpec("GPT6-05", "其余档位全部合法", "P0", "g61_effort_levels", "gpt6"),
    CaseSpec("GPT6-06", "Chat 纯文本可用", "P0", "g61_chat_text", "gpt6"),
    CaseSpec("GPT6-07", "Chat 工具面契约（按档位双向）", "P0", "g61_chat_tools_rejected", "gpt6"),
    CaseSpec("GPT6-08", "采样参数与 effort 互斥", "P0", "g61_sampling_params_rejected", "gpt6"),
    CaseSpec("GPT6-09", "工具调用闭环（Responses）", "P0", "g61_tool_loop", "gpt6"),
    CaseSpec("GPT6-10", "Structured Outputs", "P0", "g61_json_schema_output", "gpt6"),
    CaseSpec("GPT6-11", "previous_response_id 多轮状态", "P0", "g61_previous_response_state", "gpt6"),
    CaseSpec("GPT6-12", "store=false 加密推理回放", "P0", "g61_encrypted_replay", "gpt6"),
    CaseSpec("GPT6-13", "显式缓存断点（≥1024 前缀，增量判据）", "P0", "g61_explicit_cache_breakpoint", "gpt6"),
    CaseSpec("GPT6-14", "configuration_update 切档不失效缓存", "P0", "g61_config_update_cache", "gpt6"),
    CaseSpec("GPT6-15", "输出截断可检测", "P0", "g61_truncation", "gpt6"),
    CaseSpec("GPT6-16", "Responses 流式（typed SSE）", "P0", "g61_responses_stream", "gpt6"),
    CaseSpec("GPT6-17", "instructions 系统指令生效", "P0", "g61_instructions", "gpt6"),
    CaseSpec("GPT6-18", "图像输入（内联 base64）", "P0", "g61_image_recognition", "gpt6"),
    CaseSpec("GPT6-19", "响应对象结构契约", "P0", "g61_response_object_contract", "gpt6"),
    # ---- P1 增强专项（6 条）----
    CaseSpec("GPT6-20", "Responses 内置工具逐项注册", "P1", "g61_builtin_tools", "gpt6"),
    CaseSpec("GPT6-21", "allowed_tools 工具开关", "P1", "g61_allowed_tools", "gpt6"),
    CaseSpec("GPT6-22", "显式缓存 TTL 仅 30m", "P1", "g61_cache_ttl", "gpt6"),
    CaseSpec("GPT6-23", "configuration_update 相邻项被拒", "P1", "g61_config_update_adjacent", "gpt6"),
    CaseSpec("GPT6-24", "/responses/compact 压缩端点", "P1", "g61_compact_endpoint", "gpt6"),
    CaseSpec("GPT6-25", "模型 ID 严格解析", "P1", "g61_model_id_strictness", "gpt6"),
    # ---- P2 契约与健壮性（11 条）----
    CaseSpec("GPT6-26", "Responses Usage 契约", "P2", "g61_responses_usage", "gpt6"),
    CaseSpec("GPT6-27", "Chat Usage 契约", "P2", "g61_chat_usage", "gpt6"),
    CaseSpec("GPT6-28", "Chat 流式 Usage chunk", "P2", "g61_chat_stream_usage", "gpt6"),
    CaseSpec("GPT6-29", "非法 effort 错误对象四字段", "P2", "g61_effort_error_contract", "gpt6"),
    CaseSpec("GPT6-30", "非法 input 拒绝", "P2", "g61_invalid_input", "gpt6"),
    CaseSpec("GPT6-31", "max_output_tokens 超上限拒绝", "P2", "g61_max_output_tokens_rejected", "gpt6"),
    CaseSpec("GPT6-32", "无效 API Key 鉴权契约", "P2", "g61_invalid_key_auth", "gpt6"),
    CaseSpec("GPT6-33", "未知模型拒绝", "P2", "g61_unknown_model", "gpt6"),
    CaseSpec("GPT6-34", "/models 端点与档位身份", "P2", "g61_models_list", "gpt6"),
    CaseSpec("GPT6-35", "输入 token 计数端点", "P2", "g61_input_tokens_endpoint", "gpt6"),
    CaseSpec("GPT6-36", "流式语义事件与无 DONE", "P2", "g61_stream_semantic_events", "gpt6"),
    # ---- 产物类（1 条，复用 5.6 的 pelican_svg handler，提示词与判据跨档一致）----
    CaseSpec("GPT6-37", "鹈鹕骑车 SVG 生成（产物可直接查看）", "PELICAN", "pelican_svg", "gpt6"),
    # ---- FINGERPRINT 满血真实性指纹（12 条）----
    CaseSpec("AUTH-G6-01", "模型身份令牌（权威 ID 逐字）", "AUTH", "g61_auth_identity", "gpt6"),
    CaseSpec("AUTH-G6-02", "档位下限指纹（按档位画像双向）", "AUTH", "g61_auth_effort_floor", "gpt6"),
    CaseSpec("AUTH-G6-03", "Chat 工具面指纹（双向 + 协议改写成因）", "AUTH", "g61_auth_chat_tool_plane", "gpt6"),
    CaseSpec("AUTH-G6-04", "随机动态工具诱导调用", "AUTH", "g61_auth_random_tool", "gpt6"),
    CaseSpec("AUTH-G6-05", "随机 Structured Output 绑定", "AUTH", "g61_auth_random_json", "gpt6"),
    CaseSpec("AUTH-G6-06", "随机多轮记忆", "AUTH", "g61_auth_random_memory", "gpt6"),
    CaseSpec("AUTH-G6-07", "思考真实性（low 仍产生 reasoning）", "AUTH", "g61_auth_reasoning", "gpt6"),
    CaseSpec("AUTH-G6-08", "缓存计费证据", "AUTH", "g61_auth_cache_evidence", "gpt6"),
    CaseSpec("AUTH-G6-09", "中转注入识别（instructions 回显）", "AUTH", "g61_auth_injection", "gpt6"),
    CaseSpec("AUTH-G6-10", "非标准字段检测", "AUTH", "g61_auth_nonstandard_fields", "gpt6"),
    CaseSpec("AUTH-G6-11", "previous_response_id 真实性", "AUTH", "g61_auth_response_id_authenticity", "gpt6"),
    CaseSpec("AUTH-G6-12", "Chat 协议未被改写", "AUTH", "g61_auth_chat_not_translated", "gpt6"),
)


def _rng_token(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8].upper()}"


def _response(body: dict[str, Any] | None) -> dict[str, Any]:
    """Responses API 的 response 对象（非流式 body 本身就是 response）。"""
    return body or {}


def _output_items(body: dict[str, Any] | None) -> list[dict[str, Any]]:
    return (body or {}).get("output") or []


def _output_text(body: dict[str, Any] | None) -> str:
    """拼接 Responses 输出里的 message 文本。"""
    parts: list[str] = []
    for item in _output_items(body):
        if item.get("type") == "message":
            for block in item.get("content") or []:
                if isinstance(block, dict) and block.get("type") in {"output_text", "text"}:
                    parts.append(str(block.get("text") or ""))
    return "".join(parts)


def _reasoning_items(body: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [i for i in _output_items(body) if i.get("type") == "reasoning"]


def _reasoning_object(body: dict[str, Any] | None) -> dict[str, Any]:
    obj = (body or {}).get("reasoning")
    return obj if isinstance(obj, dict) else {}


def _usage(body: dict[str, Any] | None) -> dict[str, Any]:
    return (body or {}).get("usage") or {}


def _reasoning_tokens(body: dict[str, Any] | None) -> int | None:
    details = _usage(body).get("output_tokens_details") or {}
    value = details.get("reasoning_tokens")
    return int(value) if isinstance(value, (int, float)) else None


def _cached_input_tokens(body: dict[str, Any] | None) -> int:
    usage = _usage(body)
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    value = details.get("cached_tokens")
    return int(value) if isinstance(value, (int, float)) else 0


def _cache_write_tokens(body: dict[str, Any] | None) -> int:
    """缓存写入 token（显式缓存计价项，作为缓存能力的旁证）。"""
    usage = _usage(body)
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    value = details.get("cache_write_tokens")
    return int(value) if isinstance(value, (int, float)) else 0


def _chat_message(body: dict[str, Any] | None) -> dict[str, Any]:
    choices = (body or {}).get("choices") or []
    return (choices[0].get("message") or {}) if choices else {}


def _chat_finish_reason(body: dict[str, Any] | None) -> str | None:
    """Chat 响应的 finish_reason（``length`` = 被输出上限截断）。"""
    choices = (body or {}).get("choices") or []
    if choices and isinstance(choices[0], dict):
        return choices[0].get("finish_reason")
    return None


# 完整 SVG：需要成对的开标签与闭标签，故用非贪婪区间匹配并**要求闭合**。
_SVG_BLOCK_PATTERN = re.compile(r"<svg\b.*?</svg>", re.S | re.I)
_SVG_OPEN_PATTERN = re.compile(r"<svg\b", re.I)


def _extract_svg(text: str) -> str:
    """从响应原文里取出第一段**完整** SVG（模型常把它包在 ``` 代码块里）。"""
    match = _SVG_BLOCK_PATTERN.search(text or "")
    return match.group(0).strip() if match else ""


def _pelican_viewer_html(svg: str) -> str:
    """把 SVG 包成可独立打开的页面（避免只给 .svg 时浏览器直接进 XML 视图）。"""
    return (
        "<!doctype html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        "<title>鹈鹕骑车 SVG · 模型生成结果</title>\n"
        "<style>html,body{margin:0;min-height:100%;background:#fff;color:#111}"
        "body{display:flex;align-items:center;justify-content:center;padding:24px;box-sizing:border-box}"
        "main{width:min(920px,100%);border:1px solid #e5e7eb;border-radius:12px;padding:20px;background:#fff}"
        "main svg{display:block;width:100%;height:auto;max-height:78vh}"
        "footer{margin-top:14px;font:12px/1.6 -apple-system,'PingFang SC',sans-serif;color:#6b7280;text-align:center}"
        "</style>\n</head>\n<body>\n<main>\n" + svg + "\n</main>\n"
        "<footer>由测试平台生成：模型响应中 SVG 原文的渲染视图（原始文件见同目录 pelican.svg）</footer>\n"
        "</body>\n</html>\n"
    )


def _trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def _pelican_round_reason(status: int | None, content: str, svg: str, finish_reason: str | None,
                          error: str | None) -> str:
    """单轮未产出完整 SVG 的原因（已产出完整 SVG 时返回空串）。"""
    if error:
        return f"请求异常（{error}）"
    if status != 200:
        return f"HTTP {status}，未取得可查看的响应正文"
    if not content.strip():
        return f"响应无可见正文（finish_reason={finish_reason}）"
    if svg:
        return ""
    if _SVG_OPEN_PATTERN.search(content) and finish_reason == "length":
        return (f"响应含未闭合的 <svg>：输出被上限截断（finish_reason=length，"
                f"max_completion_tokens={PELICAN_MAX_OUTPUT_TOKENS}）")
    if _SVG_OPEN_PATTERN.search(content):
        return "响应含 <svg> 却没有闭合标签，未产出完整（可成对开闭）的 SVG"
    return f"响应正文未包含 SVG（{len(content)} 字符文本/说明，未按要求直出）"


def _pelican_round(index: int, status: int | None, content: str, finish_reason: str | None,
                   error: str | None, duration_seconds: float) -> dict[str, Any]:
    """一轮请求的执行结果与证据（供裁定、落盘与 observations 共同使用）。

    ``transient`` 与 ``gpt56.core._is_transient_record`` 同口径：连接错误/传输异常或 5xx 属
    上游瞬时故障，不得归因于被测模型；4xx 与"拿到 200 却没产出"是确定性结论。
    """
    svg = _extract_svg(content)
    return {
        "round": index,
        "http_status": status,
        "content": content,
        "svg": svg,
        "finish_reason": finish_reason,
        "error": error,
        "transient": error is not None or status is None or _is_transient_status(status),
        "response_chars": len(content),
        "svg_chars": len(svg),
        "svg_found": bool(svg),
        "duration_seconds": round(duration_seconds, 3),
        "reason": _pelican_round_reason(status, content, svg, finish_reason, error),
    }


def _pelican_transcript(records: list[dict[str, Any]]) -> str:
    """多轮时的响应原文：按轮分段，正文**原样**贴出（不套代码围栏，避免与模型自带围栏打架）。"""
    blocks = ["# 鹈鹕骑车 SVG · 各轮响应原文", ""]
    for item in records:
        status = (f"HTTP {item['http_status']}" if item["http_status"] is not None else "请求异常")
        verdict = "已产出完整 SVG" if item["svg_found"] else f"未产出完整 SVG（{item['reason']}）"
        blocks += [f"## 第 {item['round']} 轮", "",
                   f"> {status} · {item['duration_seconds']:.2f}s · finish_reason={item['finish_reason']} · {verdict}",
                   "", item["content"] or f"（无响应正文：{item['reason']}）", "", "---", ""]
    return "\n".join(blocks).rstrip("\n") + "\n"


def _pelican_overview_html(records: list[dict[str, Any]]) -> str:
    """多轮时的总览页：各轮渲染视图排在一起，失败轮给出原因，并链接各轮原始 SVG 文件。

    相对链接（``pelican-<轮次>.svg``）在 Web 的 artifact 路由下解析为同目录的产物请求；
    下载包里按同目录打开同样有效。SVG 原文直接内联——内容来自被测响应，页面仅用于本地查看。
    """
    ok = sum(1 for item in records if item["svg_found"])
    sections: list[str] = []
    for item in records:
        status = (f"HTTP {item['http_status']}" if item["http_status"] is not None else "请求异常")
        link = (f'<a href="pelican-{item["round"]}.svg">原始 SVG 文件</a> · '
                if item["svg_found"] else "")
        head = (f"<h2>第 {item['round']} 轮"
                f"<span class=\"{'ok' if item['svg_found'] else 'bad'}\">"
                f"{'已产出完整 SVG' if item['svg_found'] else '未产出完整 SVG'}</span></h2>")
        meta = (f"<p class=\"meta\">{html.escape(status)} · {item['duration_seconds']:.2f}s · "
                f"finish_reason={html.escape(str(item['finish_reason']))} · 响应 {item['response_chars']} 字符 · "
                f"{link}<span>{html.escape(item['reason'])}</span></p>")
        if item["svg_found"]:
            body = f"<div class=\"canvas\">{item['svg']}</div>"
        else:
            preview = (item["content"] or "（无响应正文）")[:1200]
            body = f"<pre class=\"fail\">{html.escape(preview)}</pre>"
        sections.append(f"<section>{head}{meta}{body}</section>")
    return (
        "<!doctype html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        "<title>鹈鹕骑车 SVG · 多轮结果总览</title>\n"
        "<style>html,body{margin:0;background:#fff;color:#111;font:14px/1.6 -apple-system,'PingFang SC',sans-serif}"
        "body{padding:24px}main{width:min(980px,100%);margin:0 auto}"
        "h1{font-size:18px;margin:0 0 6px}.summary{color:#6b7280;margin:0 0 20px}"
        "section{border:1px solid #e5e7eb;border-radius:12px;padding:16px 18px;margin-bottom:16px}"
        "h2{font-size:14px;margin:0 0 6px;display:flex;gap:10px;align-items:center}"
        "h2 span{font-weight:500;font-size:12px;padding:1px 8px;border-radius:999px}"
        "h2 span.ok{background:#dcfce7;color:#166534}h2 span.bad{background:#fee2e2;color:#991b1b}"
        ".meta{color:#6b7280;font-size:12px;margin:0 0 12px;display:flex;gap:8px;flex-wrap:wrap}"
        ".meta a{color:#2563eb}.meta span{color:#991b1b}"
        ".canvas{display:flex;justify-content:center}.canvas svg{display:block;width:100%;height:auto;max-height:70vh}"
        "pre.fail{background:#f9fafb;border:1px solid #eef0f3;border-radius:8px;padding:12px;white-space:pre-wrap;color:#6b7280;margin:0}"
        "</style>\n</head>\n<body>\n<main>\n"
        f"<h1>鹈鹕骑车 SVG · 多轮结果总览</h1>\n"
        f"<p class=\"summary\">共 {len(records)} 轮，{ok} 轮产出完整 SVG；提示词逐字固定——"
        f"「{html.escape(PELICAN_PROMPT)}」</p>\n"
        + "\n".join(sections) +
        "\n<footer style=\"margin-top:8px;font-size:12px;color:#6b7280\">"
        "由测试平台生成：各轮响应中 SVG 原文的渲染视图（各轮原始文件见同目录 pelican-&lt;轮次&gt;.svg）"
        "</footer>\n</main>\n</body>\n</html>\n"
    )


def _is_rejection(status: int) -> bool:
    """4xx 且非 404 —— 404 表示端点不存在（链路/配置问题），不构成"参数被正确拒绝"。"""
    return 400 <= status < 500 and status != 404


def _endpoint_missing_reason(status: int, endpoint: str) -> str | None:
    if status != 404:
        return None
    return f"{endpoint} 端点不存在（HTTP 404），用例目标特性未被触达；属链路/配置问题（检查 base_url），不是被测对象的行为结论"


def _error_body(body: dict[str, Any] | None) -> dict[str, Any] | None:
    """取结构化 error 对象；响应体可能是非 dict（JSON 字符串、列表等），须防御。"""
    err = (body if isinstance(body, dict) else {}).get("error")
    return err if isinstance(err, dict) else None


def _error_text(body: dict[str, Any] | None) -> str:
    """错误文案（结构化 error 优先，兼容纯文本错误体），统一小写便于匹配。"""
    err = _error_body(body)
    raw = json.dumps(err, ensure_ascii=False) if err else json.dumps(body or "", ensure_ascii=False)
    return raw.lower()


# 账号层限制文案：中转/账号通道不支持某能力时的显式拒绝（真实批次证据：
# mux.la 中转对 previous_response_id 返回 400 "previous_response_id requires
# an OpenAI API-key account for HTTP requests"）。命中即记为"平台级前提不满足"。
_ACCOUNT_LIMITATION_MARKERS = (
    "requires an openai api-key account",
    "requires an api key account",
    "requires an api-key account",
    "api-key account for http requests",
)


def _account_limitation_reason(status: int, body: dict[str, Any] | None, feature: str) -> str | None:
    """上游账号层明确拒绝该能力时给出 blocked 理由（前提不满足，无法观测）。"""
    if status == 200 or status == 404:
        return None
    text = _error_text(body)
    if not any(marker in text for marker in _ACCOUNT_LIMITATION_MARKERS):
        return None
    return (f"{feature}被上游账号层拒绝（HTTP {status}）：当前账号/通道类型不支持该能力，"
            f"属平台级前提不满足，用例目标特性未被触达（原文：{(_error_body(body) or {}).get('message') or text[:160]}）")


def _model_not_found_reason(requested_model: str, status: int, body: dict[str, Any] | None) -> str | None:
    """结构化"模型不存在"错误——含网关用非 4xx 状态码承载该语义的情况。

    必须同时满足：① 响应体是结构化 error；② 错误码/文案明确指向"模型不存在"，
    且指向的是被请求的那个模型（避免把"渠道整体不可用"误读成"模型校验存在"）。
    """
    err = _error_body(body)
    if not err:
        return None
    code = str(err.get("code") or "").strip().lower()
    message = str(err.get("message") or "").strip().lower()
    if "model_not_found" not in code and "model not found" not in message and "no available channel for model" not in message:
        return None
    if requested_model.lower() not in message and "model_not_found" not in code:
        return None
    if 400 <= status < 500:
        return f"未知模型被拒绝（HTTP {status}，code={code or 'n/a'}）且返回结构化 error，模型校验契约成立"
    return f"未知模型被拒绝（HTTP {status}，code={code or 'n/a'}）：网关用非 4xx 状态码承载了「模型不存在」的语义，模型校验契约成立"


# ---------------------------------------------------------------------------
# 瞬时故障（上游 5xx / 连接错误）
#
# 真实批次证据（2026-09-24，results/GPT56_P0-20260924-153058/requests.jsonl）：
# mux.la 中转对 GPT56-12 / GPT56-16 返回
#   HTTP 503 {"code":"system_memory_overloaded",
#             "message":"system memory overloaded (current: 91.5%, threshold: 90%)"}
# 且 GPT56-12 第二跳叠加 ConnectError。这类故障不由被测模型决定，请求也未产生有效响应，
# 因此：① 先做有限重试（5xx / 连接错误）；② 重试耗尽后用例按 blocked 收口，不记 failed。
# 注意与"上游明确拒绝某参数"（4xx）区分：后者是确定性判定，既不重试也不降级。
# ---------------------------------------------------------------------------
_TRANSIENT_STATUS_MIN = 500


def _is_transient_status(status: int | None) -> bool:
    return isinstance(status, int) and _TRANSIENT_STATUS_MIN <= status < 600


def _is_definitive_error(body: dict[str, Any] | None) -> bool:
    """结构化且明确指向"模型不存在"的错误是确定性结论，不是网关抖动，不重试。

    真实批次证据：New API 用 HTTP 503 + code=model_not_found 承载"未知模型"（GPT56-23），
    该状态码虽在 5xx 区间，语义上却是确定性回答，重试只会浪费配额。
    """
    err = _error_body(body)
    if not err:
        return False
    return str(err.get("code") or "").strip().lower() == "model_not_found"


def _is_transient_record(record: Any) -> bool:
    """传输层失败（连接错误/超时）或 5xx 视为瞬时故障；确定性错误与 4xx 不算。"""
    if record.transport_status != "completed":
        return True
    if _is_definitive_error(record.response_body):
        return False
    return _is_transient_status(record.http_status)


def _transient_event(record: Any, path: str, attempts: int) -> dict[str, Any]:
    error = record.transport_error or {}
    message = (_error_body(record.response_body) or {}).get("message") or str(record.response_text or "")[:160]
    return {"path": path, "attempts": attempts, "http_status": record.http_status,
            "transport_error": error.get("type"), "message": str(message)[:200]}


def _transient_reason(event: dict[str, Any], total: int) -> str:
    """瞬时故障的 blocked 理由，需自解释（谁报的错、重试了几次、为何不算模型失败）。"""
    retried = f"已重试 {event['attempts'] - 1} 次仍失败" if event["attempts"] > 1 else "未启用重试"
    more = f"；本用例累计 {total} 次瞬时故障" if total > 1 else ""
    if event.get("transport_error"):
        cause = f"链路错误 {event['transport_error']}（请求未完成）"
    else:
        cause = f"HTTP {event['http_status']}（原文：{event.get('message')}）"
    return (f"上游瞬时故障：{cause}，{retried}{more}。该故障不由被测模型决定、请求未产生有效响应，"
            f"用例目标特性未被触达，属平台级前提不满足（基础设施抖动），按 blocked 收口，建议重跑本批次")


# ---------------------------------------------------------------------------
# AUTH-G07/G08（中转注入 / 非标准字段）与 GPT56-34/35（档位）的判据
# ---------------------------------------------------------------------------
# Responses API 响应对象的字段白名单（官方 schema 超集；phase / safety_identifier
# 属合法字段）。凡出现在白名单外即视为中转/多供应商网关特征（AUTH-G08）。
_RESPONSE_SCHEMA_FIELDS = frozenset({
    "id", "object", "created_at", "status", "error", "incomplete_details",
    "instructions", "metadata", "model", "output", "parallel_tool_calls",
    "temperature", "tool_choice", "tools", "top_p", "background",
    "reasoning", "service_tier", "store", "truncation", "usage", "text",
    "safety_identifier", "previous_response_id", "conversation", "prompt",
    "prompt_cache_key", "prompt_cache_options", "include", "max_output_tokens",
    "max_tool_calls", "metadata", "stream", "sequence_number", "phase",
})

# 真实批次（2026-09-24 mux.la 中转）实测出现的非标准字段，作 AUTH-G08 的校准依据。
_RELAY_FIELD_MARKERS = ("billing_usage", "access_programs", "claude_cache_creation", "new_api_error")

# 别名 gpt-5.6 官方承诺路由到旗舰档（GPT56-35 的判据）。
FLAGSHIP_TIER = "gpt-5.6-sol"
ALL_TIERS = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")

# AUTH-G07：短问句（输入体量 < 100 token）的 input_tokens 膨胀阈值。
# 真实批次证据：注入 Codex 系统提示词后简单问句 input_tokens 达 4400+。
_INPUT_TOKEN_INFLATION_THRESHOLD = 2000

# AUTH-G11：官方 response id 形如 resp_ + 长 hex（真实批次 chatcmpl- 前缀即协议混用）。
_RESPONSE_ID_PATTERN = re.compile(r"^resp_[0-9a-f]{20,}$")

# Responses 流式语义事件（GPT56-61）：规范内无 [DONE]，靠 response.completed 收尾。
_RESPONSES_STREAM_CORE_EVENTS = ("response.created", "response.output_text.delta", "response.completed")

# AUTH-G12：Chat 响应出现任一 Responses 特征即判"协议被改写"（GPT56-31 的 stop 丢参即此因）。
_CHAT_TRANSLATION_MARKERS = ("oai_responses",)


def _injected_instructions_reason(body: dict[str, Any] | None) -> str | None:
    """AUTH-G07：未传 instructions 却被上游替换/填充即注入（信任级失败）。"""
    if "instructions" not in (body or {}):
        return None
    value = (body or {}).get("instructions")
    if value is None:
        return None
    text = str(value)
    return (f"响应 instructions 被上游填充（长度 {len(text)}，开头：{text[:80]!r}）——"
            "未传 instructions 的请求收到注入内容，本次响应内容不由请求单独决定，属信任级失败")


def _relay_fields(body: dict[str, Any] | None) -> list[str]:
    """AUTH-G08：扫描响应顶层非标准字段。"""
    extra = [k for k in (body or {}) if k not in _RESPONSE_SCHEMA_FIELDS]
    marked = [k for k in extra if any(m in k for m in _RELAY_FIELD_MARKERS)]
    return sorted(set(marked) | {k for k in extra if k.startswith("claude_cache_creation")}) or sorted(extra)


def _chat_translation_evidence(body: dict[str, Any] | None) -> str | None:
    """AUTH-G12：Chat 响应出现 Responses 特征时给出证据描述。"""
    b = body or {}
    usage = b.get("usage") or {}
    source = ((usage.get("billing_usage") or {}).get("source")) if isinstance(usage, dict) else None
    if isinstance(source, str) and any(m in source for m in _CHAT_TRANSLATION_MARKERS):
        return f"usage.billing_usage.source={source!r}（Chat 被翻译成 Responses）"
    if b.get("object") == "response":
        return "object=='response'（Responses 特征出现在 Chat 端点）"
    if str(b.get("id") or "").startswith("resp_"):
        return f"id 以 resp_ 开头（{b.get('id')!r}）"
    return None


# ---------------------------------------------------------------------------
# GPT-6.1 专用判定辅助
# ---------------------------------------------------------------------------
# 6.1 拒绝档位的错误文案特征（官方原句："Unsupported value: 'none' is not supported with the
# 'gpt-6.1-sol' model. Supported values are: 'low', 'medium', 'high', 'xhigh', and 'max'."）。
_EFFORT_REJECTION_MARKERS = ("unsupported value", "not supported", "unsupported_value",
                             "reasoning.effort", "supported values")

# 探测网关档位白名单用的必然非法值（不会命中任何真实档位）。
_EFFORT_UNIVERSE_PROBE_VALUE = "__probe_effort__"


def _effort_rejection_evidence(status: int, body: dict[str, Any] | None, effort: str) -> str | None:
    """判断 4xx 是否是"因非法档位被拒"（含 param/文案证据）。

    返回证据串；None 表示不是（或不是因档位）。用于 GPT61-03/04/29 与 AUTH-G61-02：
    官方对 none/minimal 承诺**明确 400 且 param=reasoning.effort**，故这里要求文案级证据，
    避免把"网关整体 400（如鉴权失败）"误判成"档位被正确校验"。
    """
    if not _is_rejection(status):
        return None
    err = _error_body(body) or {}
    text = _error_text(body)
    param = str(err.get("param") or "")
    code = str(err.get("code") or "")
    if effort.lower() in text or "reasoning" in param.lower() or "effort" in param.lower() or "reasoning" in text:
        return (f"HTTP {status} 且错误指向 reasoning.effort（code={code or 'n/a'}, param={param or 'n/a'}）")
    if any(marker in text for marker in _EFFORT_REJECTION_MARKERS):
        return f"HTTP {status} 且错误文案命中档位校验特征（code={code or 'n/a'}）"
    return None


def _mentions_responses_api(body: dict[str, Any] | None) -> bool:
    """6.1 的 Chat 工具面拒绝文案会指引改用 Responses API——命中即证据增强。"""
    text = _error_text(body)
    return "responses" in text and ("/v1/responses" in text or "use the responses" in text
                                    or "use /v1/responses" in text or "v1/responses" in text)


def _variant_of(case_id: str) -> str:
    """用例编号 → 模型档位。

    顺序敏感：``GPT61-*``/``AUTH-G61-*`` 必须先判，否则会被 ``GPT6-*`` 前缀抢走。
    """
    if case_id.startswith(("GPT61-", "AUTH-G61-")):
        return "gpt6.1"
    if case_id.startswith(("GPT6-", "AUTH-G6-")):
        return "gpt6"
    return "gpt5.6"


def _declared_effort_values(body: dict[str, Any] | None) -> tuple[str, ...] | None:
    """从 4xx 错误文案里抽取网关**自己声明**的合法 effort 档位集合。

    官方错误文案形如 "Supported values are: 'low', 'medium', 'high', 'xhigh', and 'max'."。
    若该集合含被测模型官方已移除的档位（如 gpt-6.1-sol 返回的集合里带 none/minimal），
    说明请求侧校验与模型无关（中转层统一白名单），此时"none 被接受"不能作为身份指纹依据。
    """
    text = _error_text(body)
    lowered = text.lower()
    marker = "supported values"
    if marker not in lowered:
        return None
    tail = lowered.split(marker, 1)[1]
    found = re.findall(r"'([a-z_]+)'", tail)
    if not found:
        return None
    # 去重保序：同一档位在文案里可能被单引号与双引号各引一次。
    return tuple(dict.fromkeys(found))



class GPT56Runner:
    def __init__(self, config: GPT56Config, client: httpx.AsyncClient | None = None, audit_dir: Path | None = None):
        self.config = config
        self._external_client = client
        self.client = client
        self.executor: HttpExecutor | None = None
        # 执行证据：每个请求的完整 record 实时落到 <audit_dir>/requests.jsonl，
        # 供统一准入校验的 evidence_present 使用。
        self.audit_dir = audit_dir
        self.log: ResultLog | None = None
        self.current_case_id: str | None = None
        # 本用例内"重试耗尽仍失败"的瞬时故障记录（5xx / 连接错误），供 run() 改判 blocked。
        self.transient_events: list[dict[str, Any]] = []
        # 当前装配的模型档位（gpt5.6 / gpt6.1 / gpt6），由 run() 设定；handler 据此回落档位画像。
        self.variant: str = "gpt5.6"
        # 鹈鹕骑车用例的「请求轮数 × 并发数」（默认 1×1）：由 runner.main 按 CLI/界面参数设定，
        # 其余用例的请求数是写死的，不读这两个值。多轮只影响轮次数量，提示词与判据不变。
        self.pelican_rounds: int = 1
        self.pelican_concurrency: int = 1
        # 网关声明的 effort 合法档位集合（按型号缓存一次）：用于区分"模型口径"与"网关口径"。
        self._effort_universe: dict[str, tuple[str, ...] | None] = {}

    async def __aenter__(self) -> "GPT56Runner":
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
                trust_env=self.config.trust_env,
                headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            )
        self.executor = HttpExecutor(timeout=self.config.timeout, verify_ssl=self.config.verify_ssl, trust_env=self.config.trust_env, client=self.client)
        await self.executor.__aenter__()
        if self.audit_dir is not None:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            self.log = ResultLog(self.audit_dir / "requests.jsonl")
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self.log:
            self.log.close()
            self.log = None
        if self.executor:
            await self.executor.__aexit__(*args)
        if self._external_client is None and self.client:
            await self.client.aclose()

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"

    async def request(self, path: str, payload: dict[str, Any], *, stream: bool = False,
                      extra_headers: dict[str, str] | None = None) -> ApiResponse:
        if self.executor is None:
            self.executor = HttpExecutor(timeout=self.config.timeout, verify_ssl=self.config.verify_ssl, trust_env=self.config.trust_env, client=self.client)
            await self.executor.__aenter__()
        headers = {"X-Request-ID": f"GPT56-{uuid.uuid4().hex[:16]}"}
        if extra_headers:
            # 逐请求覆盖客户端默认头（如 GPT56-54 的伪造 Authorization）。httpx 对同名键
            # 以请求级头为准，客户端默认的鉴权头不会泄漏。
            headers.update(extra_headers)
        body = {k: v for k, v in payload.items() if k != "_case_id"}
        case_id = payload.get("_case_id") or self.current_case_id or "GPT56-request"
        # 瞬时故障（5xx / 连接错误）做有限重试；4xx 与确定性错误（model_not_found）不重试。
        # 每一次尝试都独立写入 requests.jsonl，重试过程本身也是证据。
        attempts = 0
        while True:
            attempts += 1
            record = await self.executor.execute(case_id=case_id, url=self._url(path), payload=body, headers=headers, stream=stream)
            if self.log:
                await self.log.write(record)
            if not _is_transient_record(record) or attempts > self.config.max_retries:
                break
            await asyncio.sleep(self.config.retry_backoff_seconds * (2 ** (attempts - 1)))
        if _is_transient_record(record):
            self.transient_events.append(_transient_event(record, path, attempts))
        if not stream:
            return ApiResponse(record.http_status or 0, record.response_body if isinstance(record.response_body, dict) else None, record.response_text, record.timing.get("e2e_ms", 0) / 1000, record.response_headers, record.provider_request_id)
        # 流式：Responses 是 typed SSE（response.* 事件），Chat 是 choices delta SSE。
        events = [e for e in record.parsed_sse_events if isinstance(e, dict)]
        types = [str(e.get("type")) for e in events if e.get("type")]
        text = "".join(str(e.get("delta") or "") for e in events if e.get("type") == "response.output_text.delta")
        completed = next((e for e in reversed(events) if e.get("type") == "response.completed"), None)
        completed_response = (completed or {}).get("response") or {}
        info = {
            "text": text or record.content,
            "event_types": types,
            "completed": completed_response,
            "usage": completed_response.get("usage") or record.usage or {},
            "done": record.done_received,
            "invalid_events": record.invalid_sse_events,
            "finish_reasons": record.finish_reasons,
            "reasoning": record.reasoning,
            "raw_sse_events": record.raw_sse_events,
        }
        return ApiResponse(record.http_status or 0, None, record.raw_response_body or record.response_text, record.timing.get("e2e_ms", 0) / 1000, record.response_headers, record.provider_request_id, stream=info)

    async def responses(self, model: str, **kwargs: Any) -> ApiResponse:
        extra_headers = kwargs.pop("extra_headers", None)
        payload = {"model": model, "max_output_tokens": CONTENT_MIN_OUTPUT_TOKENS, **kwargs}
        return await self.request("responses", payload, stream=bool(payload.get("stream")), extra_headers=extra_headers)

    async def raw_request(self, method: str, path: str, *, payload: dict[str, Any] | None = None,
                          extra_headers: dict[str, str] | None = None) -> ApiResponse:
        """非 JSON-POST 通道（GET /models、GET /responses/{id} 等）。

        与 request() 共享同一套重试与证据写入逻辑（execute_raw 落同一份 requests.jsonl），
        保证新增端点用例的执行证据同样可审计。
        """
        if self.executor is None:
            self.executor = HttpExecutor(timeout=self.config.timeout, verify_ssl=self.config.verify_ssl, trust_env=self.config.trust_env, client=self.client)
            await self.executor.__aenter__()
        case_id = self.current_case_id or "GPT56-request"
        headers = {"X-Request-ID": f"GPT56-{uuid.uuid4().hex[:16]}"}
        if extra_headers:
            headers.update(extra_headers)
        attempts = 0
        while True:
            attempts += 1
            record = await self.executor.execute_raw(case_id=case_id, method=method, url=self._url(path), headers=headers, json_body=payload)
            if self.log:
                await self.log.write(record)
            if not _is_transient_record(record) or attempts > self.config.max_retries:
                break
            await asyncio.sleep(self.config.retry_backoff_seconds * (2 ** (attempts - 1)))
        if _is_transient_record(record):
            self.transient_events.append(_transient_event(record, f"{method} {path}", attempts))
        return ApiResponse(record.http_status or 0, record.response_body if isinstance(record.response_body, dict) else None, record.response_text, record.timing.get("e2e_ms", 0) / 1000, record.response_headers, record.provider_request_id)

    async def chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> ApiResponse:
        payload = {"model": model, "messages": messages, "max_completion_tokens": 2048, **kwargs}
        return await self.request("chat/completions", payload, stream=bool(payload.get("stream")))

    async def run(self, models: tuple[str, ...] = MODELS, priorities: tuple[str, ...] = ("P0", "P1"),
                  case_ids: set[str] | None = None, variant: str = "gpt5.6",
                  faces: tuple[str, ...] | None = None) -> list[CaseResult]:
        """按模型档位（variant）装配用例：GPT-5.6 / GPT-6.1 / GPT-6 家族的判定口径不同，不得串档。"""
        # 面过滤（默认 model+verify）：内置工具注册、PTC/multi-agent、端点与参数校验契约
        # 属通道面，中转链路上大面积不通过，默认不跑。
        from common.faces import ALL_FACES, filter_specs, selected_faces
        from .faces import CASE_FACES

        # 定向复测（case_ids）是用户显式点名，面过滤让位——否则点通道面用例会被静默跳过。
        active = ALL_FACES if case_ids else selected_faces(faces)
        self.variant = variant
        results: list[CaseResult] = []
        for model in models:
            tier = self._tier(model)
            if tier is not None and not tier.available:
                # 网关未开放该档位通道（如 gpt-6-luna）：整批 blocked 并附自解释理由。
                # 绝不逐条发起请求——通道不存在时打 40+ 次必然全 503，只会制造噪声。
                results.extend(await self._unavailable_tier_results(model, tier, priorities, case_ids, active))
                continue
            selected = [spec for spec in CASE_SPECS
                        if spec.variant == variant and spec.priority in priorities
                        and (not case_ids or spec.case_id in case_ids)]
            for spec in filter_specs(selected, CASE_FACES, active):
                started = time.perf_counter()
                self.current_case_id = spec.case_id
                self.transient_events = []
                try:
                    outcome = await getattr(self, f"case_{spec.handler}")(model)
                    if len(outcome) == 4:
                        passed, details, observations, status = outcome
                    else:
                        passed, details, observations = outcome
                        status = "passed" if passed else "failed"
                    if status in ("failed", "error") and self.transient_events and not (observations or {}).get("keeps_failure_verdict"):
                        # 上游瞬时故障（5xx/连接错误）使请求未产生有效响应：不可归因于被测模型，
                        # 按平台级前提不满足记 blocked，避免基础设施抖动污染"模型能力"结论。
                        # 例外：handler 已确认存在**真实缺陷**（observations["keeps_failure_verdict"]）时
                        # 保留 failed —— 多轮用例里"某轮 5xx"与"某轮拿到 200 却没产出"可同时发生，
                        # 后者是模型侧证据，不能因前者被整批抹成 blocked。
                        details = _transient_reason(self.transient_events[0], len(self.transient_events))
                        status = "blocked"
                        observations = {**(observations or {}), "transient_failures": list(self.transient_events)}
                    results.append(CaseResult(spec.case_id, spec.name, model, passed, status, details, time.perf_counter() - started, observations))
                except Exception as exc:
                    if self.transient_events:
                        results.append(CaseResult(spec.case_id, spec.name, model, False, "blocked",
                                                  _transient_reason(self.transient_events[0], len(self.transient_events)),
                                                  time.perf_counter() - started,
                                                  {"transient_failures": list(self.transient_events)}))
                    else:
                        results.append(CaseResult(spec.case_id, spec.name, model, False, "error", f"{type(exc).__name__}: {exc}", time.perf_counter() - started))
        self.current_case_id = None
        return results

    # ------------------------------------------------------------------
    # 档位画像与"模型口径 vs 网关口径"判别
    # ------------------------------------------------------------------
    def _tier(self, model: str) -> ModelTier | None:
        """被测模型的档位画像；模型名不可识别时回落到当前套件的权威档位。

        回落只对 6.x 套件生效（VARIANT_FALLBACK_TIER 只登记了 gpt6/gpt6.1）：
        网关常把用户传的 ID 改成自家别名，此时判据仍应按该套件的权威档位执行。
        """
        tier = tier_for_model(model)
        if tier is not None:
            return tier
        return VARIANT_FALLBACK_TIER.get(self.variant)

    async def gateway_effort_universe(self, model: str) -> tuple[str, ...] | None:
        """探测网关**自己声明**的 effort 合法档位集合（按型号缓存一次）。

        用一个必然非法的值触发 4xx，读错误文案里的 "Supported values"。
        探测不到（非 4xx / 文案无该段）返回 None——此时不得放宽任何判据。
        """
        if model not in self._effort_universe:
            r = await self.responses(model, input="只回复 OK",
                                     reasoning={"effort": _EFFORT_UNIVERSE_PROBE_VALUE},
                                     max_output_tokens=16)
            self._effort_universe[model] = (
                _declared_effort_values(r.body) if _is_rejection(r.status) else None
            )
        return self._effort_universe[model]

    async def gateway_ignores_model_effort_floor(self, model: str, tier: ModelTier) -> str | None:
        """返回"网关按统一档位集校验、未按模型裁剪"的证据串；无证据返回 None。

        这是把"官方已移除的档位被 200 接受"从 failed 降为 blocked 的**唯一**依据：
        没有该证据就无法区分"后端不是该模型"与"网关不按模型校验"，后者属平台口径问题。
        """
        universe = await self.gateway_effort_universe(model)
        if not universe:
            return None
        removed = [effort for effort in tier.rejected_efforts if effort in universe]
        if not removed:
            return None
        return (f"网关声明的 effort 合法集为 {list(universe)}，含 {tier.label} 官方已移除的 {removed}"
                f" ⇒ 请求侧档位校验与模型无关（中转层统一白名单），无法据此判定模型身份")

    async def _unavailable_tier_results(self, model: str, tier: ModelTier, priorities: tuple[str, ...],
                                        case_ids: set[str] | None,
                                        faces: tuple[str, ...] | None = None) -> list[CaseResult]:
        """网关未开放该档位通道：整批按 blocked 收口，理由自解释。"""
        from common.faces import ALL_FACES, face_of, selected_faces
        from .faces import CASE_FACES

        # 定向复测（case_ids）是用户显式点名，面过滤让位——否则点通道面用例会被静默跳过。
        active = ALL_FACES if case_ids else selected_faces(faces)
        # 先打一次端点，保证证据目录里有请求记录（"整批零请求"会被统一证据校验当成流程异常）。
        await self.raw_request("GET", "models")
        reason = (f"{tier.unavailable_reason or f'{tier.model_id} 在目标网关无可用通道'}"
                  f"——未能验证：模型通道不存在，非模型能力失败")
        return [
            CaseResult(spec.case_id, spec.name, model, False, "blocked", reason, 0.0,
                       {"tier": tier.model_id, "tier_available": False})
            for spec in CASE_SPECS
            if spec.variant == self.variant and spec.priority in priorities
            and (not case_ids or spec.case_id in case_ids)
            and face_of(CASE_FACES, spec.case_id) in active
        ]

    # ------------------------------------------------------------------
    # 工具定义：Responses 用扁平 function 格式，Chat 用嵌套 function 格式。
    # ------------------------------------------------------------------
    @staticmethod
    def _tool_name() -> str:
        return "get_order_status"

    @classmethod
    def _tool_responses(cls) -> dict[str, Any]:
        return {"type": "function", "name": cls._tool_name(), "description": "查询订单状态", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"], "additionalProperties": False}, "strict": False}

    @classmethod
    def _tool_chat(cls) -> dict[str, Any]:
        return {"type": "function", "function": {"name": cls._tool_name(), "description": "查询订单状态", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}}

    @staticmethod
    def _function_calls(body: dict[str, Any] | None) -> list[dict[str, Any]]:
        return [i for i in _output_items(body) if i.get("type") == "function_call"]

    @staticmethod
    def _chat_tool_calls(body: dict[str, Any] | None) -> list[dict[str, Any]]:
        return _chat_message(body).get("tool_calls") or []

    # ------------------------------------------------------------------
    # P0 核心专项
    # ------------------------------------------------------------------
    async def case_basic(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}])
        response_model = str((r.body or {}).get("model") or "")
        token_ok = any(token in response_model for token in MODEL_TOKENS)
        return r.status == 200 and bool(_chat_message(r.body).get("content")) and token_ok, "基础问答应成功且响应 model 落在 GPT-5.6 身份令牌白名单内", {"response_model": response_model}

    async def case_default_effort(self, model: str):
        r = await self.responses(model, input="计算 17*23，只给最终答案。")
        obj = _reasoning_object(r.body)
        observations = {"reasoning": obj, "http_status": r.status}
        if r.status != 200:
            return False, f"默认请求应成功（HTTP {r.status}）", observations
        if "effort" not in obj:
            # 官方文档承诺"省略时默认 medium"且可通过响应 reasoning 字段确认；字段未回显记差异留痕。
            return True, "响应未回显 reasoning.effort，无法确认默认档位（文档承诺默认 medium；属网关/线上差异，不判模型失败）", observations, "warning"
        if obj.get("effort") == "medium":
            return True, "省略 reasoning 时默认档位为 medium，符合文档承诺", observations
        return False, f"默认档位应为 medium，实际为 {obj.get('effort')!r}", observations

    async def case_effort_none(self, model: str):
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "none"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        items = _reasoning_items(r.body)
        tokens = _reasoning_tokens(r.body)
        observations = {"reasoning_items": len(items), "reasoning_tokens": tokens, "http_status": r.status}
        if r.status != 200:
            return False, f"effort=none 应被接受（HTTP {r.status}）", observations
        if items:
            return False, "effort=none 时不应产生 reasoning 输出项", observations
        if tokens is None:
            return True, "effort=none 未产生 reasoning 输出项；usage 未回显 reasoning_tokens（细节字段缺失，留痕）", observations, "warning"
        if tokens != 0:
            return False, f"effort=none 时 reasoning_tokens 应为 0，实际 {tokens}", observations
        return True, "effort=none 无 reasoning 输出项且 reasoning_tokens=0，关思考开关生效", observations

    async def case_effort_levels(self, model: str):
        statuses: dict[str, int] = {}
        for effort in ("low", "medium", "high", "xhigh", "max"):
            r = await self.responses(model, input="只回复 OK", reasoning={"effort": effort}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
            statuses[effort] = r.status
        return all(v == 200 for v in statuses.values()), "low/medium/high/xhigh/max 五档均应被接受（none 由 GPT56-03 覆盖）", statuses

    async def case_reasoning_summary(self, model: str):
        r = await self.responses(model, input="计算 19*31，只给答案。", reasoning={"effort": "low", "summary": "auto"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        items = _reasoning_items(r.body)
        has_summary = any(i.get("summary") for i in items)
        observations = {"reasoning_items": len(items), "has_summary": has_summary, "http_status": r.status, "response": r.text[:300]}
        if r.status == 200 and has_summary:
            return True, "summary=auto 时输出包含非空 reasoning 摘要项", observations
        if r.status == 200:
            return True, "summary=auto 被接受但未返回摘要内容（网关可能不透传摘要；文档/线上差异留痕）", observations, "warning"
        if _is_rejection(r.status):
            return True, f"文档承诺的 reasoning.summary 参数被上游拒绝（HTTP {r.status}；文档/线上差异留痕）", observations, "warning"
        return False, f"summary 请求返回非预期状态 HTTP {r.status}", observations

    async def case_reasoning_context(self, model: str):
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "low"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        obj = _reasoning_object(r.body)
        observations = {"reasoning": obj, "http_status": r.status}
        if r.status != 200:
            return False, f"请求应成功（HTTP {r.status}）", observations
        if "context" not in obj:
            return True, "响应未回显 reasoning.context（文档称可通过该字段确认生效模式；字段缺失属差异，留痕）", observations, "warning"
        if obj.get("context") in REASONING_CONTEXT_MODES:
            return True, f"reasoning.context 回显为 {obj.get('context')!r}，取值合法", observations
        return False, f"reasoning.context 取值非法：{obj.get('context')!r}", observations

    async def case_tool_loop(self, model: str):
        tool_marker = _rng_token("SHIP-")
        first = await self.responses(model, input="查询订单 A-731 的状态，必须调用 get_order_status 工具。", tools=[self._tool_responses()], tool_choice={"type": "function", "name": self._tool_name()})
        calls = self._function_calls(first.body)
        if first.status != 200 or not calls:
            return False, "第一跳未发起 function_call", {"http_status": first.status, "response": first.text[:500]}, "blocked" if first.status != 200 else "failed"
        call = calls[0]
        second = await self.responses(
            model,
            input=[
                {"role": "user", "content": "查询订单 A-731 的状态，必须调用 get_order_status 工具。"},
                *_output_items(first.body),
                {"type": "function_call_output", "call_id": call.get("call_id"), "output": tool_marker},
            ],
            tools=[self._tool_responses()],
        )
        text = _output_text(second.body)
        return second.status == 200 and tool_marker in text, "回传 function_call_output 后模型应基于工具结果作答", {"first_status": first.status, "second_status": second.status, "call_id": call.get("call_id"), "text": text[:200]}

    async def case_json_schema_output(self, model: str):
        marker = _rng_token()
        schema = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False}
        r = await self.responses(model, input=f"以 JSON 输出 value 字段，值必须是 {marker}，不要输出其他内容。", text={"format": {"type": "json_schema", "name": "answer", "strict": True, "schema": schema}})
        text = _output_text(r.body)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        return r.status == 200 and parsed == {"value": marker}, "Structured Outputs 应输出符合 schema 的可解析 JSON 且字段值精确绑定", {"http_status": r.status, "text": text[:300]}

    async def case_chat_compat(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        return r.status == 200 and bool(_chat_message(r.body).get("content")), "Chat Completions 基础问答与 max_completion_tokens 参数应被接受", {"http_status": r.status, "response_model": (r.body or {}).get("model")}

    async def case_previous_response_state(self, model: str):
        marker = _rng_token("ORBIT-")
        first = await self.responses(model, input=f"记住暗号 {marker}，只回复已记住。", max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        response_id = (first.body or {}).get("id")
        observations = {"first_status": first.status, "first_id": response_id}
        if first.status != 200 or not response_id:
            return False, "第一跳未返回 response id，previous_response_id 链路无法验证", observations, "blocked"
        second = await self.responses(model, input="暗号是什么？只回复暗号。", previous_response_id=response_id, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        text = _output_text(second.body)
        observations.update({"second_status": second.status, "second_error": second.text[:200], "text": text[:200]})
        # 账号层限制（如"requires an OpenAI API-key account for HTTP requests"）：平台级前提不满足，
        # 目标特性从未被触达，按 blocked 收口，不判模型失败。
        limit = _account_limitation_reason(second.status, second.body, "previous_response_id 会话续接")
        if limit:
            return False, limit, observations, "blocked"
        return second.status == 200 and marker in text, "previous_response_id 续接后应记得首轮随机暗号", observations

    async def case_encrypted_replay(self, model: str):
        marker = _rng_token("ZETA-")
        first = await self.responses(model, input=f"记住暗号 {marker}，只回复已记住。", store=False, reasoning={"effort": "low"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        items = _output_items(first.body)
        if first.status != 200:
            return False, f"store=false 请求应成功（HTTP {first.status}）", {"http_status": first.status}, "blocked"
        encrypted = [i for i in items if i.get("type") == "reasoning" and i.get("encrypted_content")]
        if not encrypted:
            # 官方文档承诺 store:false 时默认返回加密推理项供回放；没有则回放前提不满足。
            return False, ("store=false 响应未携带 encrypted_content（响应未产出任何 reasoning 项），"
                           "加密推理回放前提不满足；与 GPT56-05 / AUTH-G04 同源：上游未返回推理产物，"
                           "属前提不满足而非模型失败"), {"output_types": [i.get("type") for i in items]}, "blocked"
        second = await self.responses(
            model,
            input=[
                {"role": "user", "content": f"记住暗号 {marker}，只回复已记住。"},
                *items,
                {"role": "user", "content": "暗号是什么？只回复暗号。"},
            ],
            store=False,
            reasoning={"effort": "low"},
            max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS,
        )
        text = _output_text(second.body)
        return second.status == 200 and marker in text, "回放加密推理项后模型应记得首轮暗号", {"first_status": first.status, "second_status": second.status, "text": text[:200]}

    async def case_cache_hit(self, model: str):
        prefix = "缓存专项固定前缀-" + ("A1B2C3D4。" * 800)
        first = await self.chat(model, [{"role": "user", "content": prefix + "只回复一。"}], max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        await asyncio.sleep(2)
        second = await self.chat(model, [{"role": "user", "content": prefix + "只回复一。"}, _chat_message(first.body), {"role": "user", "content": "只回复二。"}], max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        baseline = _cached_input_tokens(first.body)
        observed = _cached_input_tokens(second.body)
        delta = observed - baseline
        observations = {
            "baseline_cached_tokens": baseline,
            "second_cached_tokens": observed,
            "delta_cached_tokens": delta,
            "cache_write_tokens": _cache_write_tokens(second.body),
            "prefix_chars": len(prefix),
            "first_status": first.status,
            "second_status": second.status,
        }
        if first.status != 200 or second.status != 200:
            return False, "缓存用例请求失败", observations
        # 判据必须是"本用例构造前缀带来的增量"：直接看绝对值会把上游固定前缀（系统提示词等）
        # 自带的缓存命中误判成前缀缓存能力（真实踩点：中转注入的 Codex 前缀在首跳就已命中 3840）。
        if delta > 0:
            if delta < 512:
                return False, "缓存命中增量过小（疑似噪声或部分命中），不足以证明构造前缀被缓存", observations, "warning"
            return True, "第二跳相对首跳新增缓存命中，构造前缀确实被缓存", observations
        if baseline > 0:
            return False, ("未观测到构造前缀带来的缓存增量；首跳已自带缓存命中（命中的是上游固定前缀，"
                           "不能归因于本用例前缀），隐式缓存能力无法判定"), observations, "blocked"
        # 隐式缓存官方明确为 best-effort（且缓存写入另有计价），未触发不算模型失败。
        return False, "未观测到缓存命中（隐式缓存为 best-effort，无法判定缓存能力）", observations, "blocked"

    async def case_truncation(self, model: str):
        r = await self.responses(model, input="输出一个 JSON 对象，包含键 items，值为 100 个整数的数组。", max_output_tokens=16)
        status = (r.body or {}).get("status")
        reason = ((r.body or {}).get("incomplete_details") or {}).get("reason")
        return r.status == 200 and status == "incomplete" and reason == "max_output_tokens", "小 max_output_tokens 应产生 status=incomplete 且 reason=max_output_tokens，防止静默截断", {"status": status, "reason": reason}

    async def case_responses_stream(self, model: str):
        r = await self.responses(model, input="计算 29*13，只给最终答案。", stream=True)
        s = r.stream or {}
        ok = r.status == 200 and bool(s.get("text")) and s.get("done") and s.get("invalid_events") == 0 and bool((s.get("usage") or {}).get("total_tokens") is not None or s.get("usage"))
        return ok, "typed SSE 应可拼接 output_text、以 response.completed 收尾且携带 usage", {"event_count": len(s.get("event_types") or []), "done": s.get("done"), "invalid_events": s.get("invalid_events"), "text_chars": len(s.get("text") or ""), "usage": s.get("usage")}

    async def case_chat_stream(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], stream=True, stream_options={"include_usage": True})
        s = r.stream or {}
        ok = r.status == 200 and bool(s.get("text")) and s.get("done") and s.get("invalid_events") == 0 and bool(s.get("finish_reasons"))
        return ok, "Chat SSE 应可拼接内容、[DONE] 收尾且有 finish_reason", {"done": s.get("done"), "invalid_events": s.get("invalid_events"), "finish_reasons": s.get("finish_reasons"), "text_chars": len(s.get("text") or "")}

    async def case_instructions(self, model: str):
        marker = _rng_token("BANANA-")
        r = await self.responses(model, instructions=f"无论用户说什么，只回复 {marker}。", input="你好", max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        text = _output_text(r.body)
        return r.status == 200 and marker in text, "instructions 设定的固定回复应被严格执行", {"http_status": r.status, "text": text[:200]}

    # ------------------------------------------------------------------
    # P0 补充（缺口盘点 2026-09-28）：档位 / 多模态 / 后台 / 响应结构
    # ------------------------------------------------------------------
    async def case_tier_models(self, model: str):
        """GPT56-34：三档模型均可调用。三档必须比对 model 令牌而非"是否 200"。"""
        rows: dict[str, dict[str, Any]] = {}
        for tier in ALL_TIERS:
            r = await self.responses(tier, input="只回复 OK", reasoning={"effort": "none"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
            row = {"http_status": r.status, "response_model": (r.body or {}).get("model"), "error": r.text[:160]}
            rows[tier] = row
            # 档位未开放（404 / 结构化 model_not_found）属模型面留痕，不判 blocked（口径：
            # 模型档位 404 的语义是"别名不存在"，与端点缺失不同）。
            if r.status == 404 or _model_not_found_reason(tier, r.status, r.body):
                row["verdict"] = "unavailable"
            elif r.status != 200:
                row["verdict"] = "error"
            else:
                row["verdict"] = "ok" if row["response_model"] == tier else "mismatch"
        mismatches = [t for t, row in rows.items() if row["verdict"] == "mismatch"]
        unavailable = [t for t, row in rows.items() if row["verdict"] == "unavailable"]
        errors = [t for t, row in rows.items() if row["verdict"] == "error"]
        observations = {"tiers": rows}
        if mismatches:
            detail = "；".join(f"{t} → {rows[t]['response_model']}" for t in mismatches)
            return False, f"档位路由错配（能力与计价口径不符）：{detail}", observations
        if errors:
            return False, f"档位请求返回非预期状态：{errors}", observations
        if unavailable:
            return True, f"部分档位未开放（留痕，不判模型失败）：{unavailable}", observations, "warning"
        return True, "三档 sol/terra/luna 均 200 且响应 model 与请求档位一致", observations

    async def case_alias_flagship(self, model: str):
        """GPT56-35：别名 gpt-5.6 应路由到旗舰档 sol。"""
        r = await self.responses("gpt-5.6", input="只回复 OK", reasoning={"effort": "none"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        m = str((r.body or {}).get("model") or "")
        observations = {"requested_model": "gpt-5.6", "response_model": m, "http_status": r.status}
        if r.status != 200:
            return False, f"别名请求应成功（HTTP {r.status}）", observations
        if m == FLAGSHIP_TIER:
            return True, f"别名 gpt-5.6 路由到旗舰档 {FLAGSHIP_TIER}，符合官方承诺", observations
        if m == "gpt-5.6":
            return True, "响应原样回显别名 gpt-5.6，网关未展开别名，路由无法验证（留痕）", observations, "warning"
        if m in ("gpt-5.6-terra", "gpt-5.6-luna"):
            return False, f"别名被路由到非旗舰档 {m}（能力与计价口径不符）", observations
        return True, f"响应 model={m!r} 不在已知档位内，无法确证路由目标（留痕）", observations, "warning"

    @staticmethod
    def _image_input(prompt: str, detail: str | None = None) -> list[dict[str, Any]]:
        part: dict[str, Any] = {"type": "input_image", "image_url": png_data_url()}
        if detail:
            part["detail"] = detail
        return [{"role": "user", "content": [{"type": "input_text", "text": prompt}, part]}]

    def _judge_image_recognition(self, img: "ApiResponse", baseline: "ApiResponse") -> tuple[bool, str, dict[str, Any], str]:
        """图像识别共用的判定（GPT56-36 与 GPT61-18 / GPT6-18 同一口径）。

        主判据是"同时答出红+蓝"（无图时基线会拒答或瞎猜），旁证是"带图 input_tokens 严格
        大于同提示纯文本基线"——**不用比例阈值**：中转会注入固定 instructions 前缀（实测占
        input 的 ≈99%），64×64 图片只贡献个位数 token，任何倍数阈值都会把"模型确实看图"
        误判成"图片被静默丢弃"。旁证不足只记 warning 留痕。
        """
        text = _output_text(img.body)
        img_tokens = (_usage(img.body) or {}).get("input_tokens")
        base_tokens = (_usage(baseline.body) or {}).get("input_tokens")
        observations = {"http_status": img.status, "text": text[:200], "image_input_tokens": img_tokens,
                        "text_baseline_input_tokens": base_tokens,
                        "baseline_text": _output_text(baseline.body)[:120]}
        if _is_rejection(img.status):
            return False, f"图像输入被上游拒绝（HTTP {img.status}）：{img.text[:160]}", observations, "blocked"
        if img.status != 200:
            return False, f"图像请求返回非预期状态 HTTP {img.status}", observations, "failed"
        if not ("红" in text and "蓝" in text):
            return False, f"200 但识别错误（应同时答出红、蓝）：{text[:120]}", observations, "failed"
        if not (isinstance(img_tokens, int) and isinstance(base_tokens, int) and img_tokens > base_tokens):
            return True, ("识别正确（红+蓝），但带图请求 input_tokens 未高于纯文本基线"
                          "（中转 usage 未体现图片增量，图片是否被计入无法取证，留痕）"), observations, "warning"
        return True, (f"识别出红+蓝，且带图 input_tokens 高于纯文本基线（{img_tokens} > {base_tokens}），"
                      "图像能力成立"), observations, "passed"

    async def case_image_recognition(self, model: str):
        """GPT56-36：图像识别（内联 base64）。必须同时证明"识别对"与"图真被看到"。"""
        prompt = "这张图上下两块纯色分别是什么颜色？必须同时答出两种颜色。"
        img = await self.responses(model, input=self._image_input(prompt), max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        baseline = await self.responses(model, input=prompt, reasoning={"effort": "none"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        return self._judge_image_recognition(img, baseline)

    async def case_image_detail_tiers(self, model: str):
        """GPT56-37：detail 四档（low/high/original/auto）+ 非法值观测。"""
        prompt = "这张图上下两块纯色分别是什么颜色？必须同时答出两种颜色。"
        rows: dict[str, dict[str, Any]] = {}
        for detail in ("low", "high", "original", "auto"):
            r = await self.responses(model, input=self._image_input(prompt, detail=detail), max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
            text = _output_text(r.body)
            rows[detail] = {"http_status": r.status, "recognized": r.status == 200 and "红" in text and "蓝" in text}
        invalid = await self.responses(model, input=self._image_input(prompt, detail="ultra"), max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        observations = {"tiers": rows, "invalid_detail_http_status": invalid.status}
        blocked = [d for d, row in rows.items() if 400 <= (row["http_status"] or 0) < 500]
        if blocked:
            return False, f"detail 档位被上游拒绝（HTTP 4xx）：{blocked}，档位能力未开放", observations, "blocked"
        failed = [d for d, row in rows.items() if row["http_status"] == 200 and not row["recognized"]]
        if failed:
            return False, f"detail 档位 200 但识别失败：{failed}", observations
        if not all(row["http_status"] == 200 for row in rows.values()):
            return False, f"detail 档位返回非预期状态：{rows}", observations
        if invalid.status == 200:
            return True, "四档 detail 均 200 且识别成功；非法 detail=ultra 被静默接受（取值口径文档仅给出范围，不硬判拒绝，留痕）", observations, "warning"
        return True, "四档 detail 均 200 且识别成功；非法 detail 被拒绝，档位校验完备", observations

    async def case_background_mode(self, model: str):
        """GPT56-40：background=true + 轮询。长任务时长不确定，超时记 blocked 不判 failed。"""
        r = await self.responses(model, input="只回复 OK", background=True, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        observations = {"submit_status": r.status, "initial_status": (r.body or {}).get("status"), "id": (r.body or {}).get("id")}
        if _is_rejection(r.status):
            return False, f"background 被上游拒绝（HTTP {r.status}）：{r.text[:160]}", observations, "blocked"
        if r.status != 200:
            return False, f"background 请求返回非预期状态 HTTP {r.status}", observations
        rid = (r.body or {}).get("id")
        if not rid:
            return False, "background 响应未携带 id，无法轮询", observations, "blocked"
        initial = (r.body or {}).get("status")
        if initial == "completed":
            return True, "同步直接返回 completed（未走后台语义但结果可用；留痕）", observations, "warning"
        if initial not in ("queued", "in_progress"):
            return False, f"background 初始 status 应为 queued/in_progress，实际 {initial!r}", observations
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            await asyncio.sleep(2)
            poll = await self.raw_request("GET", f"responses/{rid}")
            if poll.status == 200:
                status = (poll.body or {}).get("status")
                observations["poll_status"] = status
                if status == "completed":
                    return True, "background 任务轮询至 completed 且可取回结果", observations
                continue
            observations["poll_status"] = poll.status
            break
        return False, "background 轮询超时（长任务时长不确定，不判模型失败）", observations, "blocked"

    async def case_response_object_contract(self, model: str):
        """GPT56-41：响应对象结构契约（object/output/content/id/created_at/model/status/usage）。"""
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "none"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        body = r.body or {}
        messages = [i for i in _output_items(body) if i.get("type") == "message"]
        content_types = [c.get("type") for m in messages for c in (m.get("content") or []) if isinstance(c, dict)]
        detail_fields = ["id", "created_at", "model", "status", "usage"]
        missing_detail = [f for f in detail_fields if f not in body]
        observations = {"object": body.get("object"), "output_types": [i.get("type") for i in body.get("output") or []],
                        "content_types": content_types, "missing_detail_fields": missing_detail, "status": body.get("status")}
        if r.status != 200:
            return False, f"请求应成功（HTTP {r.status}）", observations
        if body.get("object") != "response" or not messages or "output_text" not in content_types:
            return False, "响应结构不符契约：object 应为 response、output 应含 message 项且 content 含 output_text", observations
        if missing_detail or body.get("status") != "completed":
            return True, f"核心结构合规；细节字段缺失或 status 异常：{missing_detail or body.get('status')!r}（留痕）", observations, "warning"
        return True, "响应对象结构完全符合契约（object/output/content 与全部细节字段）", observations

    # ------------------------------------------------------------------
    # P1 增强专项
    # ------------------------------------------------------------------
    async def case_pro_mode(self, model: str):
        r = await self.responses(model, input="只回复 OK", reasoning={"mode": "pro", "effort": "low"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        observations = {"http_status": r.status, "response": r.text[:300]}
        if r.status == 200 and _output_text(r.body):
            return True, "reasoning.mode=pro 请求成功且有输出", observations
        if _is_rejection(r.status):
            # pro 模式按聚合工作计费、属最新能力，网关未开通属前提不满足，不算模型失败。
            return False, f"pro 模式被上游拒绝（HTTP {r.status}）：{r.text[:200]}", observations, "blocked"
        return False, f"pro 模式返回非预期状态 HTTP {r.status}", observations

    async def case_context_all_turns(self, model: str):
        marker = _rng_token("CTX-")
        reasoning = {"effort": "low", "context": "all_turns"}
        first = await self.responses(model, input=f"记住暗号 {marker}，只回复已记住。", reasoning=reasoning, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        response_id = (first.body or {}).get("id")
        observations = {"first_status": first.status, "first_id": response_id}
        if first.status != 200 or not response_id:
            return False, "第一跳失败，all_turns 持久化链路无法验证", observations, "blocked"
        second = await self.responses(model, input="暗号是什么？只回复暗号。", previous_response_id=response_id, reasoning=reasoning, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        obj = _reasoning_object(second.body)
        text = _output_text(second.body)
        observations.update({"second_status": second.status, "second_error": second.text[:200], "context": obj.get("context"), "text": text[:200]})
        limit = _account_limitation_reason(second.status, second.body, "previous_response_id 会话续接")
        if limit:
            return False, limit, observations, "blocked"
        ok = second.status == 200 and obj.get("context") == "all_turns" and marker in text
        return ok, "all_turns + previous_response_id 应记住暗号且响应回显 context=all_turns", observations

    async def case_ptc(self, model: str):
        tools = [
            {"type": "programmatic_tool_calling"},
            {**self._tool_responses(), "allowed_callers": ["programmatic_tool_calling"]},
        ]
        r = await self.responses(model, input="查询订单 A-731 的状态。", tools=tools, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        observations = {"http_status": r.status, "output_types": [i.get("type") for i in _output_items(r.body)], "response": r.text[:300]}
        if r.status == 200:
            return True, "PTC 工具注册被接受（输出项类型见 observations；program/program_output/function_call 皆属可用）", observations
        if _is_rejection(r.status):
            # 最新能力，网关普遍滞后：拒绝记为文档/线上差异留痕，不判模型失败。
            return True, f"PTC 工具注册被上游拒绝（HTTP {r.status}；能力未开放/网关不支持，属文档/线上差异留痕）", observations, "warning"
        return False, f"PTC 注册返回非预期状态 HTTP {r.status}", observations

    async def case_explicit_cache(self, model: str):
        prefix = "显式缓存探测前缀-" + ("EXPL731。" * 400)
        r = await self.responses(model, input=f"{prefix}\n只回复 OK。", prompt_cache_options={"mode": "explicit"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        observations = {"http_status": r.status, "usage": _usage(r.body), "response": r.text[:200]}
        if r.status == 200 and _usage(r.body):
            return True, "prompt_cache_options.mode=explicit 被接受且返回 usage", observations
        if _is_rejection(r.status):
            return True, f"显式缓存选项被上游拒绝（HTTP {r.status}；文档/线上差异留痕）", observations, "warning"
        return False, f"显式缓存请求返回非预期状态 HTTP {r.status}", observations

    async def case_cache_ttl(self, model: str):
        prefix = "缓存 TTL 探测前缀-" + ("TTL731。" * 400)
        r = await self.responses(model, input=f"{prefix}\n只回复 OK。", prompt_cache_options={"mode": "explicit", "ttl": "1h"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        observations = {"http_status": r.status, "response": r.text[:200]}
        if r.status == 200:
            return True, "prompt_cache_options.ttl 被接受", observations
        if _is_rejection(r.status):
            # 文档称 ttl 取代 prompt_cache_retention，但取值口径未在检索到的文档中明确；
            # "接受"与"给出明确 4xx 参数校验"均可解释，其它状态才算失败。
            return True, f"ttl 选项返回明确的参数校验错误（HTTP {r.status}），取值口径以线上为准", observations, "warning"
        return False, f"ttl 请求返回非预期状态 HTTP {r.status}", observations

    async def case_strict_invalid_schema(self, model: str):
        schema = {"type": "object", "properties": {"x": {"type": "string", "minLength": 2}}, "required": ["x"], "additionalProperties": False}
        r = await self.responses(model, input="输出 x=ab", text={"format": {"type": "json_schema", "name": "bad", "strict": True, "schema": schema}}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        observations = {"http_status": r.status, "error": r.text[:300]}
        missing = _endpoint_missing_reason(r.status, "responses")
        if missing:
            return False, missing, observations, "blocked"
        if _is_rejection(r.status):
            return True, "strict 模式下不合规 schema 应返回 4xx（minLength 等关键词不受支持）", observations
        if r.status == 200:
            return True, "不合规 schema 未被上游拒绝（严格校验未执行；文档/线上差异留痕）", observations, "warning"
        return False, f"非法 schema 返回非预期状态 HTTP {r.status}", observations

    async def case_unknown_model(self, model: str):
        ghost = f"gpt-5.6-nonexistent-{uuid.uuid4().hex[:8]}"
        r = await self.responses(ghost, input="只回复 OK", max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        observations = {"requested_model": ghost, "http_status": r.status, "error": r.text[:300]}
        if r.status == 200:
            return False, "未知模型竟被路由成功，身份/路由契约不成立", observations
        # 网关可能用非 4xx 状态码承载"模型不存在"（真实批次：New API 对未知模型返回
        # 503 + code=model_not_found）。用例目标是"确认模型校验存在且非静默路由"，
        # 结构化且指向被请求模型的 model_not_found 已达成该目标，故先判通过并在 details 标注状态码差异。
        not_found = _model_not_found_reason(ghost, r.status, r.body)
        if not_found:
            return True, not_found, observations
        # 本用例的探测目标就是"模型不存在"：404 是文档预期行为，但必须返回结构化 error，
        # 用于区分"上游模型校验"与"网关/路径 404 页"——后者仍按 blocked 处理。
        if r.status == 404:
            if _error_body(r.body):
                return True, "未知模型被拒绝（HTTP 404）且返回结构化 error，模型校验契约成立", observations
            return False, "未知模型返回 404 但响应体不是结构化 error（可能是网关 404 页）", observations, "blocked"
        if _is_rejection(r.status) and _error_body(r.body):
            return True, f"未知模型被拒绝（HTTP {r.status}）且返回结构化 error，模型校验契约成立", observations
        if _is_rejection(r.status):
            return False, f"未知模型返回 HTTP {r.status} 但响应体不是结构化 error（可能是网关 404 页）", observations
        return False, f"未知模型返回非预期状态 HTTP {r.status}（响应体未给出指向模型不存在的结构化错误）", observations

    async def case_explicit_cache_breakpoint(self, model: str):
        """GPT56-49：显式缓存断点两跳。沿用 GPT56-12 的增量判据，不看绝对值。"""
        prefix = "显式缓存断点前缀-" + ("BRKP731。" * 400)

        def payload_input() -> list[dict[str, Any]]:
            return [{"role": "user", "content": [
                {"type": "input_text", "text": prefix},
                {"type": "input_text", "text": "只回复 OK。", "prompt_cache_breakpoint": {"mode": "explicit"}},
            ]}]

        options = {"mode": "explicit", "ttl": "30m"}
        first = await self.responses(model, input=payload_input(), prompt_cache_options=options, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        await asyncio.sleep(2)
        second = await self.responses(model, input=payload_input(), prompt_cache_options=options, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        base_cached = _cached_input_tokens(first.body)
        hit = _cached_input_tokens(second.body)
        delta = hit - base_cached
        write_tokens = _cache_write_tokens(second.body)
        observations = {"first_status": first.status, "second_status": second.status,
                        "baseline_cached_tokens": base_cached, "second_cached_tokens": hit,
                        "delta_cached_tokens": delta, "cache_write_tokens_second": write_tokens}
        if _is_rejection(first.status) or _is_rejection(second.status):
            return False, f"显式缓存断点被上游拒绝（HTTP {first.status}/{second.status}）：{first.text[:120]}", observations, "blocked"
        if first.status != 200 or second.status != 200:
            return False, f"显式缓存请求返回非预期状态（HTTP {first.status}/{second.status}）", observations
        if write_tokens > 0 or delta >= 512:
            return True, "显式缓存断点生效（cache_write_tokens>0 或第二跳缓存增量≥512）", observations
        if delta > 0:
            return True, f"缓存增量 {delta} 过小（疑似噪声/部分命中），不足以证明断点生效（留痕）", observations, "warning"
        return True, "两跳均无缓存字段或增量（显式缓存未触发；能力未观测，留痕）", observations, "warning"

    async def case_multi_agent(self, model: str):
        """GPT56-53：multi-agent beta。beta 未开放 → blocked；成本高（多 subagent）。"""
        r = await self.responses(
            model,
            input="用两个子智能体分别总结'测试'与'准入'两个主题，各一句话。",
            multi_agent={"enabled": True},
            max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS,
            extra_headers={"OpenAI-Beta": "responses_multi_agent=v1"},
        )
        observations = {"http_status": r.status, "output_types": [i.get("type") for i in _output_items(r.body)], "response": r.text[:200]}
        if r.status == 200 and _output_items(r.body):
            return True, "multi_agent.enabled + beta 头被接受且有输出", observations
        if _is_rejection(r.status):
            return False, f"multi-agent beta 被上游拒绝（HTTP {r.status}）：{r.text[:160]}", observations, "blocked"
        if r.status == 200:
            return True, "multi-agent 请求 200 但无输出项（留痕）", observations, "warning"
        return False, f"multi-agent 返回非预期状态 HTTP {r.status}", observations

    # ------------------------------------------------------------------
    # P2 契约与健壮性
    # ------------------------------------------------------------------
    async def case_responses_usage(self, model: str):
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "low"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        usage = _usage(r.body)
        i, o, t = usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens")
        core_ok = r.status == 200 and all(isinstance(v, int) and v >= 0 for v in (i, o, t)) and t == i + o
        details_ok = isinstance(usage.get("output_tokens_details"), dict) and usage["output_tokens_details"].get("reasoning_tokens") is not None
        observations = {"usage": usage}
        if not core_ok:
            return False, "Responses usage 应包含非负 input/output/total 且 total 守恒", observations
        if not details_ok:
            return True, "usage 核心三项完整且守恒；output_tokens_details.reasoning_tokens 缺失（细节字段，网关可能剥除；留痕）", observations, "warning"
        return True, "usage 完整且守恒，reasoning_tokens 明细存在", observations

    async def case_chat_usage(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        usage = _usage(r.body)
        p, c, t = usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")
        core_ok = r.status == 200 and all(isinstance(v, int) and v >= 0 for v in (p, c, t)) and t == p + c
        details_ok = isinstance(usage.get("prompt_tokens_details"), dict)
        observations = {"usage": usage}
        if not core_ok:
            return False, "Chat usage 应包含非负 prompt/completion/total 且 total 守恒", observations
        if not details_ok:
            return True, "usage 核心三项完整且守恒；prompt_tokens_details 缺失（细节字段缺失，留痕）", observations, "warning"
        return True, "usage 完整且守恒，prompt_tokens_details 存在", observations

    async def case_chat_stream_usage(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], stream=True, stream_options={"include_usage": True})
        s = r.stream or {}
        usage = s.get("usage") or {}
        ok = r.status == 200 and s.get("done") and isinstance(usage.get("total_tokens"), int)
        return ok, "include_usage 时流式最后一个 chunk 应携带 usage 且 [DONE] 正常收尾", {"usage": usage, "done": s.get("done")}

    async def case_temperature_invalid(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], temperature=2.5, max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        observations = {"http_status": r.status, "error": r.text[:300]}
        missing = _endpoint_missing_reason(r.status, "chat/completions")
        if missing:
            return False, missing, observations, "blocked"
        if _is_rejection(r.status):
            # 推理模型即使整体不支持自定义 temperature，4xx 同样构成"参数被校验拒绝"。
            return True, f"temperature=2.5 被拒绝（HTTP {r.status}），参数校验契约成立", observations
        if r.status == 200:
            return True, "temperature=2.5 被宽松接受（文档/线上行为差异，不判模型失败）", observations, "warning"
        return False, f"非法 temperature 返回非预期状态 HTTP {r.status}", observations

    async def case_messages_invalid(self, model: str):
        r = await self.request("chat/completions", {"model": model, "messages": "not-an-array", "max_completion_tokens": 64})
        observations = {"http_status": r.status, "error": r.text[:300]}
        missing = _endpoint_missing_reason(r.status, "chat/completions")
        if missing:
            return False, missing, observations, "blocked"
        return _is_rejection(r.status), "messages 非数组应返回 4xx", observations

    async def case_tool_choice_none(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "不要调用工具，只回复 OK"}], tools=[self._tool_chat()], tool_choice="none", max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        calls = self._chat_tool_calls(r.body)
        return r.status == 200 and not calls, "tool_choice=none 不得返回工具调用", {"http_status": r.status, "tool_calls": calls[:1]}

    async def case_tool_choice_required(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "查询订单 A-731 的状态"}], tools=[self._tool_chat()], tool_choice="required", max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        calls = self._chat_tool_calls(r.body)
        return r.status == 200 and bool(calls) and calls[0].get("function", {}).get("name") == self._tool_name(), "tool_choice=required 必须调用声明的工具", {"http_status": r.status, "tool_calls": calls[:1]}

    async def case_stop_sequence(self, model: str):
        marker = "<END>"
        r = await self.chat(model, [{"role": "user", "content": f"严格输出 ABC{marker}XYZ"}], stop=[marker], max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        content = str(_chat_message(r.body).get("content") or "")
        return r.status == 200 and marker not in content and "XYZ" not in content, "Stop 命中后不得输出停止串及其后内容", {"content": content}

    async def case_max_completion_tokens(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], max_completion_tokens=16)
        return r.status == 200, "max_completion_tokens=16 边界值应被接受（内容可被截断，仅校验状态）", {"http_status": r.status}

    async def case_invalid_input(self, model: str):
        r = await self.request("responses", {"model": model, "input": 42, "max_output_tokens": 64})
        observations = {"http_status": r.status, "error": r.text[:300]}
        missing = _endpoint_missing_reason(r.status, "responses")
        if missing:
            return False, missing, observations, "blocked"
        return _is_rejection(r.status), "input 传非法标量应返回 4xx", observations

    async def case_invalid_key_auth(self, model: str):
        """GPT56-54：无效 API Key 鉴权契约。200 = 鉴权完全失效（严重项，判 failed）。"""
        ghost_key = f"sk-invalid-{uuid.uuid4().hex[:12]}"
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "none"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS,
                                 extra_headers={"Authorization": f"Bearer {ghost_key}"})
        observations = {"http_status": r.status, "error": r.text[:300], "key_prefix": ghost_key[:14]}
        if r.status == 200:
            return False, "无效 API Key 竟通过鉴权并返回 200——鉴权完全失效（严重项）", observations
        if r.status == 401:
            if _error_body(r.body) or r.text:
                return True, "无效 API Key 被 401 拒绝且返回错误信息，鉴权契约成立", observations
            return True, "无效 API Key 被 401 拒绝（错误体为空，留痕）", observations, "warning"
        if r.status == 403:
            return True, f"无效 API Key 返回 403（非 401，结构差异留痕）：{r.text[:120]}", observations, "warning"
        if _is_rejection(r.status):
            return True, f"无效 API Key 返回 {r.status}（非 401/403，结构差异留痕）", observations, "warning"
        return False, f"无效 API Key 返回非预期状态 HTTP {r.status}", observations

    async def case_error_object_contract(self, model: str):
        """GPT56-55：错误对象结构契约（message/type/param/code 四字段）。"""
        r = await self.responses(model, input="只回复 OK", text={"verbosity": "nonsense"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        err = _error_body(r.body)
        fields = sorted(k for k in (err or {}) if k in ("message", "type", "param", "code")) if isinstance(err, dict) else []
        observations = {"http_status": r.status, "error_fields": fields, "error": r.text[:300]}
        if r.status == 200:
            return True, "非法 verbosity 未被拒绝（上游未校验该参数；留痕，不判模型失败）", observations, "warning"
        missing = _endpoint_missing_reason(r.status, "responses")
        if missing:
            return False, missing, observations, "blocked"
        if _is_rejection(r.status):
            if not isinstance(err, dict):
                return False, f"HTTP {r.status} 但响应体不是结构化 error 对象", observations
            if set(fields) == {"code", "message", "param", "type"}:
                return True, "参数错误返回 4xx 且 error 含 message/type/param/code 四字段，契约成立", observations
            if {"message", "type"} <= set(fields):
                return True, f"error 含 message/type，缺 {sorted(set(('param', 'code')) - set(fields))}（细节字段，留痕）", observations, "warning"
            return False, "error 对象缺少 message/type 核心字段", observations
        return False, f"非法 verbosity 返回非预期状态 HTTP {r.status}", observations

    async def case_models_list(self, model: str):
        """GPT56-56：GET /models 与三档一致性（别名口径交叉证据）。"""
        r = await self.raw_request("GET", "models")
        observations = {"http_status": r.status, "body_head": r.text[:200]}
        if r.status == 404:
            return False, "/models 端点不存在（HTTP 404），别名口径无法交叉验证", observations, "blocked"
        if r.status != 200:
            return True, f"/models 返回非预期状态 HTTP {r.status}（端点可用性留痕）", observations, "warning"
        body = r.body or {}
        entries = body.get("data") or []
        ids = {str(e.get("id") or "") for e in entries if isinstance(e, dict)}
        if body.get("object") != "list" or not entries or not any(ids):
            return False, f"/models 响应结构不符契约（object={body.get('object')!r}，条目数={len(entries)}）", observations
        present = [t for t in ALL_TIERS if t in ids]
        observations["tiers_present"] = present
        if len(present) == len(ALL_TIERS):
            return True, "/models 结构合规且三档齐备，与档位口径交叉一致", observations
        return True, f"/models 结构合规但三档仅出现 {present}（列表口径与实际可调用档位常见滞后，留痕）", observations, "warning"

    async def case_input_tokens_endpoint(self, model: str):
        """GPT56-57：POST /responses/input_tokens 与真实请求 usage 交叉验证。"""
        text = "输入 token 计数端点交叉验证用例。" + ("A1B2C3。" * 200)
        count = await self.request("responses/input_tokens", {"model": model, "input": text})
        observations = {"count_status": count.status, "count_body": str(count.text)[:200]}
        if count.status == 404:
            return False, "/responses/input_tokens 端点不存在（HTTP 404）", observations, "blocked"
        n = (count.body or {}).get("input_tokens")
        if count.status != 200 or not isinstance(n, int) or n <= 0:
            return True, f"计数端点未返回有效 input_tokens（HTTP {count.status}），留痕", observations, "warning"
        real = await self.responses(model, input=text, reasoning={"effort": "none"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        real_n = (_usage(real.body) or {}).get("input_tokens")
        observations["real_status"] = real.status
        observations["real_input_tokens"] = real_n
        if real.status != 200 or not isinstance(real_n, int) or real_n <= 0:
            return True, f"计数端点返回 {n}，但真实请求 usage 缺失无法交叉验证（留痕）", observations, "warning"
        deviation = abs(n - real_n) / max(n, real_n, 1)
        observations["deviation"] = round(deviation, 4)
        if deviation <= 0.10:
            return True, f"计数端点与真实 usage 同量级（{n} vs {real_n}，偏差 {deviation:.1%}）", observations
        return True, f"计数端点与真实 usage 偏离过大（{n} vs {real_n}，偏差 {deviation:.1%}；留痕）", observations, "warning"

    async def case_stream_semantic_events(self, model: str):
        """GPT56-61：Responses 流式语义事件与无 [DONE]（规范内以 response.completed 收尾）。"""
        r = await self.responses(model, input="计算 29*13，只给最终答案。", stream=True, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        s = r.stream or {}
        raw_events = s.get("raw_sse_events") or []
        has_done = any(str(e).strip() == "[DONE]" for e in raw_events)
        types = s.get("event_types") or []
        seqs: list[int] = []
        for raw in raw_events:
            try:
                obj = json.loads(str(raw))
            except json.JSONDecodeError:
                continue
            if isinstance(obj.get("sequence_number"), int):
                seqs.append(obj["sequence_number"])
        monotonic = all(b > a for a, b in zip(seqs, seqs[1:]))
        has_core = all(t in types for t in _RESPONSES_STREAM_CORE_EVENTS)
        observations = {"event_count": len(types), "event_head": types[:3], "event_tail": types[-3:],
                        "done_marker_present": has_done, "sequence_numbers": seqs[:12], "monotonic": monotonic}
        if r.status != 200:
            return False, f"流式请求应成功（HTTP {r.status}）", observations
        if has_done:
            return False, "Responses 事件流出现 [DONE]——Chat 收尾语义混入 Responses 协议（典型网关改写特征）", observations
        if seqs and not monotonic:
            return False, f"sequence_number 非严格单调递增：{seqs[:12]}", observations
        if has_core and monotonic:
            return True, "语义事件齐备（created/delta/completed）、序号严格单调且无 [DONE]，协议纯净", observations
        return True, f"流式事件不完整（核心事件缺失：{[t for t in _RESPONSES_STREAM_CORE_EVENTS if t not in types]}；留痕）", observations, "warning"

    # ------------------------------------------------------------------
    # FINGERPRINT 满血真实性指纹（随机挑战）
    # ------------------------------------------------------------------
    TOOL_PRIMING_PROMPTS = (
        "必须调用 {tool} 工具（参数 code 填 {marker}），不要自行回答。",
        "禁止直接回答。请先调用 {tool} 工具（参数 code 填 {marker}），再基于工具返回结果作答。",
    )

    async def case_auth_random_tool(self, model: str):
        tool_name = f"lookup_{uuid.uuid4().hex[:8]}"
        marker = _rng_token()
        tool = {"type": "function", "name": tool_name, "description": "返回测试数据", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"], "additionalProperties": False}}
        last: ApiResponse | None = None
        calls: list[dict[str, Any]] = []
        for prompt in self.TOOL_PRIMING_PROMPTS:
            r = await self.responses(model, input=prompt.format(tool=tool_name, marker=marker), tools=[tool], tool_choice="auto", max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
            last = r
            calls = self._function_calls(r.body)
            if r.status != 200:
                break
            if calls:
                break
        if last is None or last.status != 200 or not calls:
            return False, "随机动态工具两轮诱导均未触发 function_call（探测动态工具理解能力失败）", {"http_status": getattr(last, "status", None), "tool_name": tool_name}, "blocked" if (last is None or last.status != 200) else "failed"
        args_raw = calls[0].get("arguments") or ""
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except json.JSONDecodeError:
            args = {}
        bound = args.get("code") == marker
        return bound, f"模型应发起对随机工具 {tool_name} 的调用且参数绑定随机串", {"tool_name": tool_name, "arguments": args, "marker": marker}

    async def case_auth_random_json(self, model: str):
        marker = _rng_token()
        schema = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False}
        r = await self.responses(model, input=f"以 JSON 输出 value 字段，值必须是 {marker}。", text={"format": {"type": "json_schema", "name": "probe", "strict": True, "schema": schema}}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        text = _output_text(r.body)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        return r.status == 200 and parsed == {"value": marker}, f"结构化输出应严格绑定随机串 {marker}（缓存/中转固定应答无法预知）", {"http_status": r.status, "text": text[:200], "marker": marker}

    async def case_auth_random_memory(self, model: str):
        marker = _rng_token()
        first = await self.responses(model, input=f"记住暗号 {marker}，只回复已记住。", max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        response_id = (first.body or {}).get("id")
        observations = {"first_status": first.status, "first_id": response_id, "marker": marker}
        if first.status != 200 or not response_id:
            return False, "随机多轮记忆第一跳失败", observations, "blocked"
        second = await self.responses(model, input="暗号是什么？只回复暗号。", previous_response_id=response_id, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        text = _output_text(second.body)
        observations.update({"second_status": second.status, "second_error": second.text[:200], "text": text[:200]})
        limit = _account_limitation_reason(second.status, second.body, "previous_response_id 会话续接")
        if limit:
            return False, limit, observations, "blocked"
        return second.status == 200 and marker in text, f"跨轮记忆应复述随机暗号 {marker}", observations

    async def case_auth_reasoning_summary(self, model: str):
        r = await self.responses(model, input="计算 17*23，只给最终答案。", reasoning={"effort": "high", "summary": "auto"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        items = _reasoning_items(r.body)
        has_summary = any(i.get("summary") for i in items)
        observations = {"reasoning_items": len(items), "has_summary": has_summary, "http_status": r.status}
        if r.status == 200 and has_summary:
            return True, "high 档位返回非空 reasoning 摘要，思考真实性成立", observations
        if r.status == 200 and items:
            return True, "high 档位产生 reasoning 输出项但摘要为空（网关不透传摘要；留痕）", observations, "warning"
        if r.status == 200:
            # 与 P0 的 GPT56-05 统一口径：同一观测（200 但拿不到 reasoning 产物）不判模型失败——
            # 能力面（GPT56-05）记 warning 留痕，指纹面因"思考真实性"取证前提落空而记 blocked（无法观测）。
            # "疑似剥参"的硬信号由 AUTH-G06 的 none/high 对照承担，此处不重复计入 failed。
            return False, ("high 档位未产生任何 reasoning 输出项，思考真实性的取证前提不满足"
                           "（同观测见 GPT56-05：200 但无 reasoning 产物，属文档/线上差异；"
                           "剥参硬信号由 AUTH-G06 的 none/high 对照承担）"), observations, "blocked"
        return False, f"请求失败 HTTP {r.status}", observations, "blocked"

    async def case_auth_stream_events(self, model: str):
        marker = _rng_token("SSE-")
        r = await self.responses(model, input=f"只回复一次 {marker}。", stream=True, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        s = r.stream or {}
        types = s.get("event_types") or []
        ordered = types and types[0] == "response.created" and types[-1] == "response.completed"
        ok = r.status == 200 and ordered and s.get("invalid_events") == 0 and marker in (s.get("text") or "")
        return ok, "typed SSE 事件序列应 response.created 起始、response.completed 收尾且正文包含随机串", {"event_head": types[:3], "event_tail": types[-3:], "invalid_events": s.get("invalid_events"), "text_chars": len(s.get("text") or "")}

    async def case_auth_effort_contrast(self, model: str):
        question = "计算 19*31，只给最终答案。"
        none_r = await self.responses(model, input=question, reasoning={"effort": "none"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        high_r = await self.responses(model, input=question, reasoning={"effort": "high"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        none_tokens = _reasoning_tokens(none_r.body)
        high_tokens = _reasoning_tokens(high_r.body)
        high_items = _reasoning_items(high_r.body)
        observations = {"none_status": none_r.status, "high_status": high_r.status, "none_reasoning_tokens": none_tokens, "high_reasoning_tokens": high_tokens, "high_reasoning_items": len(high_items)}
        if none_r.status != 200 or high_r.status != 200:
            return False, "effort 对照用例请求失败", observations, "blocked"
        if high_tokens is None and not high_items:
            return False, "high 档位响应无 usage 明细且无 reasoning 输出项，无法对照", observations, "blocked"
        high_active = (high_tokens or 0) > 0 or bool(high_items)
        none_active = (none_tokens or 0) > 0 or bool(_reasoning_items(none_r.body))
        if high_active and not none_active:
            return True, "none 与 high 两档行为可区分（high 产生思考、none 未产生），effort 参数真实生效", observations
        if high_active and none_active:
            return True, "两档均产生思考内容（none 档未被严格执行；文档/线上差异留痕）", observations, "warning"
        return False, "high 与 none 两档 reasoning 均为 0 且无思考输出项：effort 参数疑似被网关静默剥参（典型中转特征）", observations

    async def case_auth_injection(self, model: str):
        """AUTH-G07：中转注入识别。未传 instructions 却被填充 = 信任级失败。"""
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "none"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        body = r.body or {}
        usage = _usage(body)
        input_tokens = usage.get("input_tokens")
        observations = {"http_status": r.status, "instructions_present": "instructions" in body,
                        "instructions_preview": str(body.get("instructions") or "")[:120],
                        "input_tokens": input_tokens}
        injected = _injected_instructions_reason(body)
        if injected:
            return False, injected, observations
        if "instructions" not in body:
            return True, "响应未回显 instructions 字段，无法确证无注入（留痕）", observations, "warning"
        if not isinstance(input_tokens, int):
            return True, "instructions 为空但 usage 缺 input_tokens，注入检测证据不全（留痕）", observations, "warning"
        if input_tokens > _INPUT_TOKEN_INFLATION_THRESHOLD:
            return False, (f"未传 instructions 的短问句 input_tokens={input_tokens}"
                           f"（阈值 {_INPUT_TOKEN_INFLATION_THRESHOLD}）——存在上游注入的系统提示词，信任级失败"), observations
        return True, f"instructions 为空且 input_tokens={input_tokens} 与输入体量相称，无注入迹象", observations

    async def case_auth_nonstandard_fields(self, model: str):
        """AUTH-G08：非标准字段检测（中转/多供应商网关特征）。"""
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "none"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        extras = _relay_fields(r.body)
        observations = {"http_status": r.status, "top_level_fields": sorted((r.body or {}).keys()), "nonstandard_fields": extras}
        if r.status != 200:
            return False, f"请求应成功（HTTP {r.status}）", observations, "blocked"
        if extras:
            return False, f"响应顶层出现非标准字段 {extras}——中转/多供应商网关特征，信任级失败（对齐 2026-09-24 真实批次归因）", observations
        return True, "响应顶层字段全部落在 OpenAI Responses schema 内，无网关改写迹象", observations

    async def case_auth_tier_gradient(self, model: str):
        """AUTH-G10：档位能力梯度。两档输出逐字相同且 reasoning_tokens 相等 = 疑似同一后端。"""
        question = "计算 1234*5678，只给最终数字。"
        luna = await self.responses("gpt-5.6-luna", input=question, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        sol = await self.responses("gpt-5.6-sol", input=question, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        luna_tokens = _reasoning_tokens(luna.body)
        sol_tokens = _reasoning_tokens(sol.body)
        luna_text = _output_text(luna.body)
        sol_text = _output_text(sol.body)
        observations = {"luna_status": luna.status, "sol_status": sol.status,
                        "luna_reasoning_tokens": luna_tokens, "sol_reasoning_tokens": sol_tokens,
                        "luna_answer": luna_text[:80], "sol_answer": sol_text[:80]}
        if luna.status != 200 or sol.status != 200:
            return False, f"档位对照请求失败（luna HTTP {luna.status} / sol HTTP {sol.status}）", observations, "blocked"
        same_output = luna_text == sol_text
        same_tokens = luna_tokens is not None and luna_tokens == sol_tokens
        if same_output and same_tokens:
            return False, (f"两档 reasoning_tokens 逐一相等（{luna_tokens}）且输出逐字相同——"
                           "档位被吞、疑似同一后端，梯度不成立"), observations
        return True, f"两档存在可观测梯度（luna={luna_tokens} / sol={sol_tokens}，输出不同：{not same_output}）", observations

    async def case_auth_response_id_authenticity(self, model: str):
        """AUTH-G11：previous_response_id 真实性（id 格式 + GET 可检索）。"""
        r = await self.responses(model, input="只回复 OK", max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        rid = str((r.body or {}).get("id") or "")
        observations = {"first_status": r.status, "response_id": rid}
        if r.status != 200 or not rid:
            return False, "首跳未取得 response id", observations, "blocked"
        if rid.startswith("chatcmpl-"):
            return False, f"response id 形如 chatcmpl-（{rid[:24]}…）——协议混用，Responses 持久化真实性不成立", observations
        if not _RESPONSE_ID_PATTERN.match(rid):
            return True, f"response id 格式不合规（{rid[:24]}…），无法确证官方持久化形态（留痕）", observations, "warning"
        fetched = await self.raw_request("GET", f"responses/{rid}")
        observations["fetch_status"] = fetched.status
        if fetched.status == 404:
            return False, "id 格式合规但 GET /responses/{id} 检索 404——持久化真实性存疑（可检索失败）", observations, "blocked"
        if fetched.status != 200:
            return False, f"检索端点返回 HTTP {fetched.status}，真实性无法验证", observations, "blocked"
        same = (fetched.body or {}).get("id") == rid
        observations["fetch_id"] = (fetched.body or {}).get("id")
        if not same:
            return False, "检索回的 id 与首跳不一致，持久化真实性不成立", observations
        return True, f"id 格式合规且可检索回同一对象（{rid[:24]}…），Responses 持久化真实性成立", observations

    async def case_auth_chat_not_translated(self, model: str):
        """AUTH-G12：Chat 协议未被改写（Responses 特征 = 网关翻译 = 信任级失败）。"""
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        body = r.body or {}
        evidence = _chat_translation_evidence(body)
        observations = {"http_status": r.status, "object": body.get("object"), "id": str(body.get("id") or "")[:32],
                        "translation_evidence": evidence}
        if r.status != 200:
            return False, f"Chat 请求应成功（HTTP {r.status}）", observations, "blocked"
        if evidence:
            return False, f"Chat 响应被网关改写：{evidence}——参数语义将丢失（GPT56-31 stop 丢参即此因），信任级失败", observations
        if body.get("object") == "chat.completion" and str(body.get("id") or "").startswith("chatcmpl-"):
            return True, "Chat 响应严格符合 Chat 契约（object/id 形态正确），协议未被改写", observations
        return True, f"Chat 响应缺少契约标识字段（object={body.get('object')!r}），无法确证协议未被改写（留痕）", observations, "warning"


    # ==================================================================
    # GPT-6.1（gpt-6.1-sol）——独立判定，不复用 5.6 判据
    #
    # 与 5.6 的差异决定了判据必须重写：none/minimal 从「合法档位」变成「必须被拒」；
    # Chat 从「兼容承载工具调用」变成「只承载无工具请求」；缓存出现 1024 token 下限、
    # ttl 仅 30m、configuration_update 切档等 6 家族新机制。
    # ==================================================================
    @staticmethod
    def _g61_tool(name: str = "get_order_status") -> dict[str, Any]:
        """Responses 扁平 function 工具定义。"""
        return {"type": "function", "name": name, "description": "查询订单状态",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                               "required": ["order_id"], "additionalProperties": False}, "strict": False}

    def _g61_cache_input(self) -> list[dict[str, Any]]:
        """构造 ≥1024 token 的前缀并挂显式缓存断点（6.1 最小可缓存前缀为 1024 token）。"""
        prefix = "6.1 缓存断点前缀-" + ("C61A7B。" * 900)
        return [{"role": "user", "content": [
            {"type": "input_text", "text": prefix},
            {"type": "input_text", "text": "只回复 OK。", "prompt_cache_breakpoint": {"mode": "explicit"}},
        ]}]

    # ---- P0 核心专项 ----
    async def case_g61_identity(self, model: str):
        """P0：Responses 基础问答 + 模型身份令牌（按被测档位的权威 ID 判定）。"""
        tier = self._tier(model)
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "low"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        response_model = str((r.body or {}).get("model") or "")
        observations = {"http_status": r.status, "requested_model": model, "response_model": response_model,
                        "tier": tier.model_id if tier else None}
        if r.status != 200:
            missing = _endpoint_missing_reason(r.status, "responses")
            if missing:
                return False, missing, observations, "blocked"
            return False, f"Responses 基础问答应成功（HTTP {r.status}）：{r.text[:160]}", observations
        if not _output_text(r.body):
            return False, "200 但无输出文本", observations
        flagship = tier.model_id if tier else GPT61_FLAGSHIP
        tokens = tier.accepted_tokens if tier else GPT61_MODEL_TOKENS
        if not any(token in response_model for token in tokens):
            return False, (f"响应 model={response_model!r} 不在 {flagship} 的身份令牌内"
                           "（疑似降级/错配后端）"), observations
        if response_model == flagship:
            return True, f"基础问答成功且身份为 {flagship}", observations
        return True, (f"基础问答成功；响应 model={response_model!r}"
                      f"（权威 ID 为 {flagship}，别名回显留痕）"), observations, "warning"

    async def case_g61_default_effort(self, model: str):
        """P0：省略 reasoning 时的默认档位（按档位画像判定）。"""
        tier = self._tier(model)
        r = await self.responses(model, input="计算 17*23，只给最终答案。")
        obj = _reasoning_object(r.body)
        expected = tier.default_effort if tier else "medium"
        observations = {"reasoning": obj, "http_status": r.status, "expected_effort": expected}
        if r.status != 200:
            return False, f"默认请求应成功（HTTP {r.status}）", observations
        if "effort" not in obj:
            return True, (f"响应未回显 reasoning.effort（文档承诺默认 {expected or '未明示'}；"
                          "字段缺失属网关差异，留痕）"), observations, "warning"
        actual = obj.get("effort")
        rejected = tier.rejected_efforts if tier else GPT6_EFFORT_FLOOR
        if actual in rejected:
            return False, (f"默认档位落到 {actual!r}——该档位已被 {tier.label if tier else '该模型'} 移除，"
                           "默认值配置错位（后端档位集与文档不符）"), observations
        if expected and actual == expected:
            return True, f"省略 reasoning 时默认档位为 {expected}，符合文档承诺", observations
        if not expected:
            return True, (f"默认档位为 {actual!r}；该档位文档未明示默认值（无法否决具体取值，留痕）"), observations, "warning"
        return False, f"默认档位应为 {expected}，实际为 {actual!r}", observations

    async def _effort_floor_case(self, model: str, effort: str):
        """档位下限判据：档位画像要求拒绝则必须拒绝，要求接受则必须接受。

        两侧都要有文案级证据，且"该拒未拒"只有在能排除网关口径问题时才判 failed。
        """
        tier = self._tier(model)
        must_reject = bool(tier and effort in tier.rejected_efforts)
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": effort}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        observations = {"http_status": r.status, "error": r.text[:300], "expected": "reject" if must_reject else "accept"}
        if r.status == 404:
            return False, _endpoint_missing_reason(r.status, "responses"), observations, "blocked"
        if must_reject:
            evidence = _effort_rejection_evidence(r.status, r.body, effort)
            if evidence:
                return True, (f"effort={effort} 被拒绝（{evidence}），"
                              f"符合 {tier.label} 档位契约（{effort} 已移除）"), observations
            if r.status == 200:
                relaxed = await self.gateway_ignores_model_effort_floor(model, tier)
                if relaxed:
                    return False, (f"effort={effort} 被 200 接受，但{relaxed}——"
                                   "该发现不能用于判定模型身份"), observations, "blocked"
                return False, (f"effort={effort} 被 200 接受——{tier.label} 官方已移除该档位，"
                               "接受该值即说明后端档位集与该模型不符"), observations
            if _is_rejection(r.status):
                return True, f"effort={effort} 被拒绝（HTTP {r.status}，响应体未含档位文案，留痕）", observations, "warning"
            return False, f"effort={effort} 返回非预期状态 HTTP {r.status}", observations
        if r.status == 200:
            return True, (f"effort={effort} 被接受（{tier.label if tier else '该档位'} 官方档位集含该档位）"), observations
        if _is_rejection(r.status):
            return False, (f"effort={effort} 被拒绝（HTTP {r.status}）——该档位官方文档明确支持，"
                           "拒绝即后端档位集与文档不符"), observations
        return False, f"effort={effort} 返回非预期状态 HTTP {r.status}", observations

    async def case_g61_effort_none_rejected(self, model: str):
        """P0：effort=none 的档位下限判定——6.1/Astra 必须拒绝，Sol/Luna 必须接受。"""
        return await self._effort_floor_case(model, "none")

    async def case_g61_effort_minimal_rejected(self, model: str):
        """P0：effort=minimal 的档位下限判定（同上，按档位画像取期望值）。"""
        return await self._effort_floor_case(model, "minimal")

    async def case_g61_effort_levels(self, model: str):
        """P0：除下限档位之外的全部档位都应被接受（下限由 03/04 覆盖）。"""
        tier = self._tier(model)
        levels = tuple(effort for effort in (tier.effort_levels if tier else GPT61_EFFORT_LEVELS)
                       if effort not in GPT6_EFFORT_FLOOR)
        statuses: dict[str, int] = {}
        for effort in levels:
            r = await self.responses(model, input="只回复 OK", reasoning={"effort": effort}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
            statuses[effort] = r.status
        return (all(v == 200 for v in statuses.values()),
                f"{'/'.join(levels)} 各档均应被接受（none/minimal 由 GPT61-03/04 覆盖）", statuses)

    async def case_g61_chat_text(self, model: str):
        """GPT61-06：Chat Completions 在 6.1 上仍承载无工具纯文本请求。"""
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        observations = {"http_status": r.status, "response_model": (r.body or {}).get("model"), "error": r.text[:200]}
        if r.status == 404:
            return False, _endpoint_missing_reason(r.status, "chat/completions"), observations, "blocked"
        ok = r.status == 200 and bool(_chat_message(r.body).get("content"))
        return ok, "6.1 的 Chat Completions 应承载无工具纯文本请求并正常作答", observations

    async def case_g61_chat_tools_rejected(self, model: str):
        """P0：Chat 工具面契约（按档位画像取期望值）。

        * ``rejected``（6.1 / Astra）：Chat 带 tools 应失败；
        * ``none_only``（Sol / Luna）：仅 effort=none 允许——none 必须成功发起工具调用，
          非 none 必须失败（双向判据，单侧不足以定论）。
        """
        tier = self._tier(model)
        policy = tier.chat_tools if tier else CHAT_TOOLS_REJECTED
        label = tier.label if tier else "该档位"
        tool = self._tool_chat()
        message = [{"role": "user", "content": "查询订单 A-731 的状态，必须调用 get_order_status 工具。"}]
        if policy == CHAT_TOOLS_NONE_ONLY:
            none_r = await self.chat(model, message, tools=[tool], reasoning_effort="none", max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
            none_calls = self._chat_tool_calls(none_r.body)
            mid_r = await self.chat(model, message, tools=[tool], reasoning_effort="medium", max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
            mid_calls = self._chat_tool_calls(mid_r.body)
            translation = _chat_translation_evidence(mid_r.body)
            observations = {"policy": policy, "tier": label,
                            "none": {"http_status": none_r.status, "tool_calls": none_calls[:1], "error": none_r.text[:200]},
                            "non_none": {"http_status": mid_r.status, "tool_calls": mid_calls[:1],
                                         "error": mid_r.text[:200], "translation_evidence": translation}}
            if none_r.status == 404:
                return False, _endpoint_missing_reason(none_r.status, "chat/completions"), observations, "blocked"
            if none_r.status != 200 or not none_calls:
                return False, (f"effort=none 下 Chat + tools 未发起工具调用（HTTP {none_r.status}）——"
                               f"{label} 官方允许 none 下的 Chat 函数调用"), observations
            if _is_rejection(mid_r.status):
                return True, (f"{label} Chat 工具面双向成立：effort=none 成功发起工具调用，"
                              f"非 none 被拒绝（HTTP {mid_r.status}）"), observations
            if mid_r.status == 200 and mid_calls:
                if translation:
                    return False, (f"非 none 下 Chat + tools 仍成功发起工具调用，且响应带协议改写证据"
                                   f"（{translation}）——网关把 Chat 翻译成 Responses，绕过了「仅 none 可用」"
                                   "的档位约束（信任级：协议改写）"), observations
                return False, (f"非 none 下 Chat + tools 成功发起工具调用，与 {label}"
                               "「Chat 函数调用仅在 effort=none 下可用」的文档约束不符"), observations
            if mid_r.status == 200:
                return True, "非 none 下 Chat + tools 返回 200 但未发起工具调用（网关静默忽略 tools，留痕）", observations, "warning"
            return False, f"非 none 下 Chat + tools 返回非预期状态 HTTP {mid_r.status}", observations
        r = await self.chat(model, message, tools=[tool], max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        calls = self._chat_tool_calls(r.body)
        translation = _chat_translation_evidence(r.body)
        observations = {"policy": policy, "tier": label, "http_status": r.status, "tool_calls": calls[:1],
                        "translation_evidence": translation, "error": r.text[:300],
                        "mentions_responses_api": _mentions_responses_api(r.body)}
        if r.status == 404:
            return False, _endpoint_missing_reason(r.status, "chat/completions"), observations, "blocked"
        if r.status == 200 and calls:
            if translation:
                return False, (f"Chat + tools 成功发起工具调用，且响应带协议改写证据（{translation}）——"
                               "网关把 Chat 翻译成 Responses，绕过了 Chat 通道的工具面拒绝契约"
                               "（信任级：协议改写；归因于网关链路，不等于后端不是该档位）"), observations
            return False, (f"Chat Completions 带 tools 竟成功发起工具调用——{label} 工具面只在 Responses API，"
                           "此通道原生支持工具即说明后端不是该档位"), observations
        if _is_rejection(r.status):
            extra = "，且错误文案指引改用 Responses API" if observations["mentions_responses_api"] else ""
            return True, f"Chat + tools 被拒绝（HTTP {r.status}{extra}），符合 {label}「工具只在 Responses」契约", observations
        if r.status == 200:
            return True, "Chat + tools 返回 200 但未发起工具调用（网关静默忽略 tools，留痕）", observations, "warning"
        return False, f"Chat + tools 返回非预期状态 HTTP {r.status}", observations

    async def case_g61_sampling_params_rejected(self, model: str):
        """P0：采样参数与 effort 互斥（按档位画像判定）。

        档位集无 none（6.1 / Astra）⇒ 采样参数永不可用；档位集含 none（Sol / Luna）⇒
        省略 effort（默认非 none）时仍应失败，只有 effort=none 下才允许。
        """
        tier = self._tier(model)
        policy = tier.sampling if tier else SAMPLING_REJECTED
        label = tier.label if tier else "该档位"
        default_r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}],
                                    temperature=0.7, top_p=0.9, max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        observations: dict[str, Any] = {"policy": policy, "tier": label,
                                        "default_effort": {"http_status": default_r.status, "error": default_r.text[:300]}}
        if default_r.status == 404:
            return False, _endpoint_missing_reason(default_r.status, "chat/completions"), observations, "blocked"
        if _is_rejection(default_r.status):
            return True, (f"temperature/top_p 在非 none 档位下被拒绝（HTTP {default_r.status}），"
                          "符合「采样参数仅在 effort=none 下允许」的互斥契约"), observations
        if default_r.status == 200 and policy == SAMPLING_NONE_ONLY:
            none_r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}],
                                     temperature=0.7, top_p=0.9, reasoning_effort="none", max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
            observations["none_effort"] = {"http_status": none_r.status, "error": none_r.text[:200]}
            return False, (f"非 none 档位下 temperature/top_p 仍被 200 接受（none 档位下 HTTP {none_r.status}）——"
                           f"{label} 文档称采样参数仅在 effort=none 下允许，档位互斥约束未生效"), observations
        if default_r.status == 200:
            return True, (f"采样参数被宽松接受（{label} 无 none 档位，文档称该参数永不可用；"
                          "线上差异留痕）"), observations, "warning"
        return False, f"采样参数请求返回非预期状态 HTTP {default_r.status}", observations

    async def case_g61_tool_loop(self, model: str):
        """GPT61-09：6.1 唯一的工具调用面（Responses）闭环。"""
        tool_marker = _rng_token("SHIP61-")
        tools = [self._g61_tool()]
        first = await self.responses(model, input="查询订单 A-731 的状态，必须调用 get_order_status 工具。",
                                     tools=tools, tool_choice={"type": "function", "name": "get_order_status"},
                                     max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        calls = self._function_calls(first.body)
        if first.status != 200 or not calls:
            return False, "第一跳未发起 function_call", {"http_status": first.status, "response": first.text[:500]}, "blocked" if first.status != 200 else "failed"
        call = calls[0]
        second = await self.responses(
            model,
            input=[
                {"role": "user", "content": "查询订单 A-731 的状态，必须调用 get_order_status 工具。"},
                *_output_items(first.body),
                {"type": "function_call_output", "call_id": call.get("call_id"), "output": tool_marker},
            ],
            tools=tools,
            max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS,
        )
        text = _output_text(second.body)
        return second.status == 200 and tool_marker in text, "回传 function_call_output 后模型应基于工具结果作答（6.1 工具面唯一入口）", {"first_status": first.status, "second_status": second.status, "call_id": call.get("call_id"), "text": text[:200]}

    async def case_g61_json_schema_output(self, model: str):
        marker = _rng_token()
        schema = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False}
        r = await self.responses(model, input=f"以 JSON 输出 value 字段，值必须是 {marker}，不要输出其他内容。",
                                 text={"format": {"type": "json_schema", "name": "answer", "strict": True, "schema": schema}},
                                 max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        text = _output_text(r.body)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        return r.status == 200 and parsed == {"value": marker}, "Structured Outputs 应输出符合 schema 的可解析 JSON 且字段值精确绑定", {"http_status": r.status, "text": text[:300]}

    async def case_g61_previous_response_state(self, model: str):
        marker = _rng_token("ORBIT61-")
        first = await self.responses(model, input=f"记住暗号 {marker}，只回复已记住。", max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        response_id = (first.body or {}).get("id")
        observations = {"first_status": first.status, "first_id": response_id}
        if first.status != 200 or not response_id:
            return False, "第一跳未返回 response id，previous_response_id 链路无法验证", observations, "blocked"
        second = await self.responses(model, input="暗号是什么？只回复暗号。", previous_response_id=response_id, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        text = _output_text(second.body)
        observations.update({"second_status": second.status, "second_error": second.text[:200], "text": text[:200]})
        limit = _account_limitation_reason(second.status, second.body, "previous_response_id 会话续接")
        if limit:
            return False, limit, observations, "blocked"
        return second.status == 200 and marker in text, "previous_response_id 续接后应记得首轮随机暗号", observations

    async def case_g61_encrypted_replay(self, model: str):
        marker = _rng_token("ZETA61-")
        first = await self.responses(model, input=f"记住暗号 {marker}，只回复已记住。", store=False,
                                     reasoning={"effort": "low"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        items = _output_items(first.body)
        if first.status != 200:
            return False, f"store=false 请求应成功（HTTP {first.status}）", {"http_status": first.status}, "blocked"
        encrypted = [i for i in items if i.get("type") == "reasoning" and i.get("encrypted_content")]
        if not encrypted:
            return False, ("store=false 响应未携带 encrypted_content（未产出 reasoning 项），"
                           "加密推理回放前提不满足；属取证前提缺失而非模型失败"), {"output_types": [i.get("type") for i in items]}, "blocked"
        second = await self.responses(
            model,
            input=[{"role": "user", "content": f"记住暗号 {marker}，只回复已记住。"}, *items,
                   {"role": "user", "content": "暗号是什么？只回复暗号。"}],
            store=False, reasoning={"effort": "low"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS,
        )
        text = _output_text(second.body)
        return second.status == 200 and marker in text, "回放加密推理项后模型应记得首轮暗号", {"first_status": first.status, "second_status": second.status, "text": text[:200]}

    async def case_g61_explicit_cache_breakpoint(self, model: str):
        """GPT61-13：显式缓存断点两跳。6.1 最小可缓存前缀 1024 token，判据取增量。"""
        options = {"mode": "explicit", "ttl": "30m"}
        first = await self.responses(model, input=self._g61_cache_input(), prompt_cache_options=options, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        await asyncio.sleep(2)
        second = await self.responses(model, input=self._g61_cache_input(), prompt_cache_options=options, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        base_cached = _cached_input_tokens(first.body)
        hit = _cached_input_tokens(second.body)
        delta = hit - base_cached
        write_tokens = _cache_write_tokens(second.body)
        observations = {"first_status": first.status, "second_status": second.status,
                        "baseline_cached_tokens": base_cached, "second_cached_tokens": hit,
                        "delta_cached_tokens": delta, "cache_write_tokens_second": write_tokens}
        if _is_rejection(first.status) or _is_rejection(second.status):
            return False, f"显式缓存断点被上游拒绝（HTTP {first.status}/{second.status}）：{first.text[:120]}", observations, "blocked"
        if first.status != 200 or second.status != 200:
            return False, f"显式缓存请求返回非预期状态（HTTP {first.status}/{second.status}）", observations
        if write_tokens > 0 or delta >= 512:
            return True, "显式缓存断点生效（cache_write_tokens>0 或第二跳缓存增量≥512）", observations
        if delta > 0:
            return True, f"缓存增量 {delta} 过小（疑似噪声/部分命中），不足以证明断点生效（留痕）", observations, "warning"
        return True, "两跳均无缓存字段或增量（显式缓存未触发；能力未观测，留痕）", observations, "warning"

    async def case_g61_config_update_cache(self, model: str):
        """GPT61-14：configuration_update 切档不应失效缓存（6 家族特性）。

        顶层 reasoning.effort 保持不变，把 configuration_update 作为 input 项追加；
        判据是切档后缓存仍命中（增量 ≥512 或出现 cache_write_tokens）。机制未开放 → blocked。
        """
        options = {"mode": "explicit", "ttl": "30m"}
        first = await self.responses(model, input=self._g61_cache_input(), prompt_cache_options=options,
                                     reasoning={"effort": "low"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        rid = (first.body or {}).get("id")
        observations = {"first_status": first.status, "first_id": rid,
                        "first_cached_tokens": _cached_input_tokens(first.body),
                        "first_cache_write_tokens": _cache_write_tokens(first.body)}
        if first.status != 200 or not rid:
            return False, "首跳未成功或未返回 response id，切档缓存链路无法验证", observations, "blocked"
        await asyncio.sleep(2)
        config_update = {"type": "configuration_update", "reasoning": {"effort": "high"}}
        second = await self.responses(model, input=[config_update, {"role": "user", "content": "只回复 OK。"}],
                                      previous_response_id=rid, prompt_cache_options=options,
                                      reasoning={"effort": "low"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        base = _cached_input_tokens(first.body)
        hit = _cached_input_tokens(second.body)
        delta = hit - base
        observations.update({"second_status": second.status, "second_cached_tokens": hit, "delta_cached_tokens": delta,
                             "second_cache_write_tokens": _cache_write_tokens(second.body), "second_error": second.text[:200]})
        limit = _account_limitation_reason(second.status, second.body, "previous_response_id 会话续接")
        if limit:
            return False, limit, observations, "blocked"
        if _is_rejection(second.status):
            return False, f"configuration_update / 会话续接被上游拒绝（HTTP {second.status}）：{second.text[:160]}", observations, "blocked"
        if second.status != 200:
            return False, f"第二跳返回非预期状态 HTTP {second.status}", observations
        if delta >= 512 or _cache_write_tokens(second.body) > 0:
            return True, "configuration_update 切档后缓存仍命中（切档未失效缓存），符合 6 家族契约", observations
        return True, f"切档链路可用但未观测到缓存命中（增量 {delta}；显式缓存为尽力行为，留痕）", observations, "warning"

    async def case_g61_truncation(self, model: str):
        """GPT61-15：小 max_output_tokens 应产生 status=incomplete 且 reason=max_output_tokens。

        **前提是请求侧的输出上限被透传**。中转（mux.la，实测 2026-10-10）不透传该参数：
        响应回显 ``max_output_tokens=null``，且请求 16 时实际产出 225 token（与 GPT61-31
        「超上限被静默接受」同一根因）。上限根本没被应用时，截断信号无从产生——这是通道
        量级问题，不能记成模型失败，故按 warning 留痕；只有上限确认生效却仍无截断信号才判 failed。
        """
        limit = 16
        r = await self.responses(model, input="输出一个 JSON 对象，包含键 items，值为 100 个整数的数组。",
                                 max_output_tokens=limit)
        body = r.body or {}
        status = body.get("status")
        reason = (body.get("incomplete_details") or {}).get("reason")
        echoed = body.get("max_output_tokens")
        out_tokens = (_usage(body) or {}).get("output_tokens")
        observations = {"status": status, "reason": reason, "requested_max_output_tokens": limit,
                        "echoed_max_output_tokens": echoed, "output_tokens": out_tokens}
        missing = _endpoint_missing_reason(r.status, "responses")
        if missing:
            return False, missing, observations, "blocked"
        if r.status != 200:
            return False, f"截断用例请求应成功（HTTP {r.status}）", observations
        if status == "incomplete" and reason == "max_output_tokens":
            return True, "小 max_output_tokens 产生 status=incomplete 且 reason=max_output_tokens，防止静默截断", observations
        over = isinstance(out_tokens, int) and out_tokens > limit
        if echoed is None or over:
            return True, (f"输出上限未被后端应用（回显 max_output_tokens={echoed!r}、实际 output_tokens={out_tokens}，"
                          f"请求值 {limit}）——中转未透传 max_output_tokens，截断信号无法验证（留痕）"), observations, "warning"
        return False, f"max_output_tokens 已生效但未产生截断信号（status={status!r}，reason={reason!r}）", observations

    async def case_g61_responses_stream(self, model: str):
        r = await self.responses(model, input="计算 29*13，只给最终答案。", stream=True, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        s = r.stream or {}
        ok = r.status == 200 and bool(s.get("text")) and s.get("done") and s.get("invalid_events") == 0 and bool(s.get("usage"))
        return ok, "typed SSE 应可拼接 output_text、以 response.completed 收尾且携带 usage", {"event_count": len(s.get("event_types") or []), "done": s.get("done"), "invalid_events": s.get("invalid_events"), "text_chars": len(s.get("text") or ""), "usage": s.get("usage")}

    async def case_g61_instructions(self, model: str):
        marker = _rng_token("BANANA61-")
        r = await self.responses(model, instructions=f"无论用户说什么，只回复 {marker}。", input="你好", max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        text = _output_text(r.body)
        return r.status == 200 and marker in text, "instructions 设定的固定回复应被严格执行", {"http_status": r.status, "text": text[:200]}

    async def case_g61_image_recognition(self, model: str):
        """GPT61-18：图像识别（判定已抽到 ``_judge_image_recognition``，与 5.6/6 家族同一口径）。

        为什么不用「带图 input_tokens > 纯文本基线 1.5×」：本套件跑在中转上，响应由中转
        注入固定 instructions 前缀（mux.la 实测 4216 token，占 input_tokens 的 ≈99%），
        64×64 图片自身只贡献个位数 token——任何比例阈值都**恒不可满足**，会把"模型确实看图"
        误判成"图片被静默丢弃"。2026-10-10 实测：带图答"红+蓝"、纯文本基线瞎猜"浅蓝+浅黄"。
        """
        prompt = "这张图上下两块纯色分别是什么颜色？必须同时答出两种颜色。"
        img = await self.responses(model, input=self._image_input(prompt), max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        baseline = await self.responses(model, input=prompt, reasoning={"effort": "low"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        return self._judge_image_recognition(img, baseline)

    async def case_g61_response_object_contract(self, model: str):
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "low"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        body = r.body or {}
        messages = [i for i in _output_items(body) if i.get("type") == "message"]
        content_types = [c.get("type") for m in messages for c in (m.get("content") or []) if isinstance(c, dict)]
        detail_fields = ["id", "created_at", "model", "status", "usage"]
        missing_detail = [f for f in detail_fields if f not in body]
        observations = {"object": body.get("object"), "output_types": [i.get("type") for i in body.get("output") or []],
                        "content_types": content_types, "missing_detail_fields": missing_detail, "status": body.get("status")}
        if r.status != 200:
            return False, f"请求应成功（HTTP {r.status}）", observations
        if body.get("object") != "response" or not messages or "output_text" not in content_types:
            return False, "响应结构不符契约：object 应为 response、output 应含 message 项且 content 含 output_text", observations
        if missing_detail or body.get("status") != "completed":
            return True, f"核心结构合规；细节字段缺失或 status 异常：{missing_detail or body.get('status')!r}（留痕）", observations, "warning"
        return True, "响应对象结构完全符合契约（object/output/content 与全部细节字段）", observations

    # ---- P1 增强专项 ----
    async def case_g61_builtin_tools(self, model: str):
        """GPT61-20：Responses 内置工具逐项注册（6.1 工具面清单）。"""
        rows: dict[str, dict[str, Any]] = {}
        for tool in GPT61_BUILTIN_TOOLS:
            r = await self.responses(model, input="只回复 OK", tools=[{"type": tool}], max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
            rows[tool] = {"http_status": r.status, "error": r.text[:120]}
        observations = {"tools": rows}
        accepted = [t for t, row in rows.items() if row["http_status"] == 200]
        rejected = [t for t, row in rows.items() if _is_rejection(row["http_status"])]
        other = [t for t in rows if t not in accepted and t not in rejected]
        if other:
            return False, f"内置工具注册返回非预期状态：{other}", observations
        if not accepted:
            return False, "全部内置工具注册被拒绝（工具面不可用，属平台前提不满足）", observations, "blocked"
        if rejected:
            return True, f"内置工具部分被拒（200 {len(accepted)}/{len(rows)}；被拒 {rejected}，网关滞后留痕）", observations, "warning"
        return True, f"{len(accepted)} 个内置工具注册均被接受（6.1 工具面完整）", observations

    async def case_g61_allowed_tools(self, model: str):
        """GPT61-21：工具开关用 tool_choice.allowed_tools（保持 tools 列表不变）。

        官方形态是把受限子集**嵌套在 `tool_choice` 下**（``tool_choice={"type":"allowed_tools",
        "mode":"auto"|"required","tools":[{"type":"function","name":..}]}``）。顶层 `allowed_tools`
        不是 Responses 参数，网关会直接丢弃——用它探测会把"参数没传对"读成"工具开关未生效"。
        2026-10-10 实测（mux.la / gpt-6.1-sol）：顶层形态下模型照调被排除的 delete_account；
        改嵌套形态后只调用允许的 get_order_status（auto 与 required 两种 mode 均生效）。
        """
        allowed = self._g61_tool()["name"]
        tools = [self._g61_tool(), self._g61_tool("delete_account")]
        tool_choice = {"type": "allowed_tools", "mode": "auto",
                       "tools": [{"type": "function", "name": allowed}]}
        r = await self.responses(
            model,
            input="必须调用 delete_account 工具删除用户 u1；若该工具不可用则只回复 FORBIDDEN。",
            tools=tools, tool_choice=tool_choice, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS,
        )
        calls = self._function_calls(r.body)
        called = [c.get("name") for c in calls]
        text = _output_text(r.body)
        observations = {"http_status": r.status, "called": called, "text": text[:160],
                        "allowed_tools": [allowed], "tool_choice": tool_choice, "error": r.text[:200]}
        if _is_rejection(r.status):
            return False, f"tool_choice.allowed_tools 被上游拒绝（HTTP {r.status}）：工具开关能力未开放", observations, "blocked"
        if r.status != 200:
            return False, f"allowed_tools 请求返回非预期状态 HTTP {r.status}", observations
        if "delete_account" in called:
            return False, "被 allowed_tools 排除的工具仍被调用——工具开关未生效", observations
        return True, f"allowed_tools 生效：仅允许的工具可被调用（called={called or '无'}）", observations

    async def case_g61_cache_ttl(self, model: str):
        """GPT61-22：prompt_cache_options.ttl 仅支持 "30m"。"""
        prefix = "6.1 TTL 前缀-" + ("TTL61。" * 900)
        ok_req = await self.responses(model, input=f"{prefix}\n只回复 OK。",
                                      prompt_cache_options={"mode": "explicit", "ttl": "30m"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        bad_req = await self.responses(model, input=f"{prefix}\n只回复 OK。",
                                       prompt_cache_options={"mode": "explicit", "ttl": "1h"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        observations = {"ttl_30m_status": ok_req.status, "ttl_1h_status": bad_req.status, "ttl_1h_error": bad_req.text[:200]}
        if ok_req.status == 404:
            return False, _endpoint_missing_reason(ok_req.status, "responses"), observations, "blocked"
        if ok_req.status != 200:
            return True, f"ttl=30m 未被接受（HTTP {ok_req.status}）；缓存选项口径与文档不同，留痕", observations, "warning"
        if bad_req.status == 200:
            return True, "ttl=30m 被接受；非法 ttl=1h 也被接受（文档称仅支持 30m，校验宽松，留痕）", observations, "warning"
        if _is_rejection(bad_req.status):
            return True, "ttl=30m 被接受且非法 ttl=1h 被拒绝，符合「仅支持 30m」的契约", observations
        return False, f"非法 ttl 返回非预期状态 HTTP {bad_req.status}", observations

    async def case_g61_config_update_adjacent(self, model: str):
        """GPT61-23：相邻 configuration_update 项应被拒绝。"""
        items = [
            {"type": "configuration_update", "reasoning": {"effort": "high"}},
            {"type": "configuration_update", "reasoning": {"effort": "low"}},
            {"role": "user", "content": "只回复 OK。"},
        ]
        r = await self.responses(model, input=items, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        observations = {"http_status": r.status, "error": r.text[:300]}
        if r.status == 404:
            return False, _endpoint_missing_reason(r.status, "responses"), observations, "blocked"
        if _is_rejection(r.status):
            return True, f"相邻 configuration_update 项被拒绝（HTTP {r.status}），符合文档约束", observations
        if r.status == 200:
            return True, "相邻 configuration_update 未被拒绝（文档称应被拒；该校验未执行，留痕）", observations, "warning"
        return False, f"返回非预期状态 HTTP {r.status}", observations

    async def case_g61_compact_endpoint(self, model: str):
        """GPT61-24：POST /responses/compact（6.1 模型页列出的压缩操作）。"""
        r = await self.request("responses/compact", {"model": model, "input": "只回复 OK"})
        observations = {"http_status": r.status, "body_head": r.text[:200]}
        if r.status == 404:
            return False, "POST /responses/compact 不存在（HTTP 404）——6.1 列出的压缩端点未开放", observations, "blocked"
        if r.status == 200:
            return True, "POST /responses/compact 可用（200）", observations
        if _is_rejection(r.status):
            return True, f"compaction 端点存在但拒绝当前请求（HTTP {r.status}，参数面待对齐，留痕）", observations, "warning"
        return True, f"compaction 端点返回非预期状态 HTTP {r.status}（可用性留痕）", observations, "warning"

    async def case_g61_model_id_strictness(self, model: str):
        """P1：模型 ID 严格解析——点号写法是权威 ID，连字符/去点变体不应被解析。"""
        tier = self._tier(model)
        flagship = tier.model_id if tier else GPT61_FLAGSHIP
        candidates = {flagship.replace(".", "-"), flagship.replace(".", ""),
                      flagship.replace("-sol", ".sol"), f"{flagship}-v2"}
        bad_ids = tuple(sorted(candidates - {flagship}))
        rows: dict[str, dict[str, Any]] = {}
        for bad in bad_ids:
            r = await self.responses(bad, input="只回复 OK", max_output_tokens=32)
            rows[bad] = {"http_status": r.status,
                         "resolved_model": (r.body or {}).get("model") if r.status == 200 else None,
                         "has_structured_error": bool(_error_body(r.body)), "error": r.text[:160]}
        observations = {"tier": flagship, "variants": rows}
        accepted = [m for m, row in rows.items() if row["http_status"] == 200]
        if accepted:
            return True, (f"变体 {accepted} 竟被解析成功（官方称权威写法只有 {flagship}；"
                          "网关宽松解析，留痕）"), observations, "warning"
        unresolved = [m for m, row in rows.items() if _is_rejection(row["http_status"])]
        if len(unresolved) == len(rows):
            return True, f"{list(bad_ids)} 均被 4xx 拒绝，模型 ID 严格解析（权威写法 {flagship}）", observations
        if any(not row["has_structured_error"] for row in rows.values()):
            return False, "模型 ID 变体返回 404 且无结构化错误（可能是网关 404 页），无法确证模型校验", observations, "blocked"
        return True, "模型 ID 变体均未解析成功（留痕）", observations, "warning"

    # ---- P2 契约与健壮性 ----
    async def case_g61_responses_usage(self, model: str):
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "low"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        usage = _usage(r.body)
        i, o, t = usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens")
        core_ok = r.status == 200 and all(isinstance(v, int) and v >= 0 for v in (i, o, t)) and t == i + o
        details = usage.get("input_tokens_details") or {}
        details_ok = isinstance(details, dict) and details.get("cached_tokens") is not None
        reasoning_ok = isinstance(usage.get("output_tokens_details"), dict)
        observations = {"usage": usage}
        if not core_ok:
            return False, "Responses usage 应包含非负 input/output/total 且 total 守恒", observations
        if not details_ok and not reasoning_ok:
            return True, "usage 核心三项完整且守恒；input_tokens_details.cached_tokens 与 output_tokens_details 均缺失（6.1 文档称 usage 暴露缓存与思考明细；细节字段被剥除，留痕）", observations, "warning"
        return True, "usage 完整且守恒，缓存/思考明细字段存在（6.1 可观测计费口径）", observations

    async def case_g61_chat_usage(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        usage = _usage(r.body)
        p, c, t = usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")
        core_ok = r.status == 200 and all(isinstance(v, int) and v >= 0 for v in (p, c, t)) and t == p + c
        details_ok = isinstance(usage.get("prompt_tokens_details"), dict)
        observations = {"usage": usage}
        if not core_ok:
            return False, "Chat usage 应包含非负 prompt/completion/total 且 total 守恒", observations
        if not details_ok:
            return True, "usage 核心三项完整且守恒；prompt_tokens_details 缺失（细节字段缺失，留痕）", observations, "warning"
        return True, "usage 完整且守恒，prompt_tokens_details 存在", observations

    async def case_g61_chat_stream_usage(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], stream=True, stream_options={"include_usage": True})
        s = r.stream or {}
        usage = s.get("usage") or {}
        ok = r.status == 200 and s.get("done") and isinstance(usage.get("total_tokens"), int)
        return ok, "include_usage 时流式最后一个 chunk 应携带 usage 且 [DONE] 正常收尾", {"usage": usage, "done": s.get("done")}

    async def case_g61_effort_error_contract(self, model: str):
        """P2：非法 effort 触发 4xx，错误对象含 message/type/param/code 四字段。

        用**必然非法**的值探测（不能用 none——它对 Sol/Luna 合法，对 6.1 也不必然非法）。
        """
        r = await self.responses(model, input="只回复 OK",
                                 reasoning={"effort": _EFFORT_UNIVERSE_PROBE_VALUE}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        err = _error_body(r.body)
        fields = sorted(k for k in (err or {}) if k in ("message", "type", "param", "code")) if isinstance(err, dict) else []
        observations = {"http_status": r.status, "error_fields": fields, "error": r.text[:300],
                        "probe_value": _EFFORT_UNIVERSE_PROBE_VALUE}
        if r.status == 200:
            return False, (f"非法 effort={_EFFORT_UNIVERSE_PROBE_VALUE!r} 未被拒绝——"
                           "网关对档位取值缺乏校验（任何值都放行）"), observations
        missing = _endpoint_missing_reason(r.status, "responses")
        if missing:
            return False, missing, observations, "blocked"
        if _is_rejection(r.status):
            if not isinstance(err, dict):
                return False, f"HTTP {r.status} 但响应体不是结构化 error 对象", observations
            if set(fields) == {"code", "message", "param", "type"}:
                return True, "非法 effort 返回 4xx 且 error 含 message/type/param/code 四字段，契约成立", observations
            if {"message", "type"} <= set(fields):
                return True, f"error 含 message/type，缺 {sorted(set(('param', 'code')) - set(fields))}（细节字段，留痕）", observations, "warning"
            return False, "error 对象缺少 message/type 核心字段", observations
        return False, f"非法 effort 返回非预期状态 HTTP {r.status}", observations

    async def case_g61_invalid_input(self, model: str):
        r = await self.request("responses", {"model": model, "input": 42, "max_output_tokens": 64})
        observations = {"http_status": r.status, "error": r.text[:300]}
        missing = _endpoint_missing_reason(r.status, "responses")
        if missing:
            return False, missing, observations, "blocked"
        return _is_rejection(r.status), "input 传非法标量应返回 4xx", observations

    async def case_g61_max_output_tokens_rejected(self, model: str):
        """GPT61-31：6.1 最大输出 128000，超出应被拒。"""
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "low"}, max_output_tokens=200000)
        observations = {"http_status": r.status, "error": r.text[:300]}
        missing = _endpoint_missing_reason(r.status, "responses")
        if missing:
            return False, missing, observations, "blocked"
        if _is_rejection(r.status):
            return True, f"max_output_tokens=200000 超出 6.1 上限（128000）被拒绝（HTTP {r.status}）", observations
        if r.status == 200:
            return True, "超上限的 max_output_tokens 被静默接受（未执行上限校验，留痕）", observations, "warning"
        return False, f"返回非预期状态 HTTP {r.status}", observations

    async def case_g61_invalid_key_auth(self, model: str):
        ghost_key = f"sk-invalid-{uuid.uuid4().hex[:12]}"
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "low"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS,
                                 extra_headers={"Authorization": f"Bearer {ghost_key}"})
        observations = {"http_status": r.status, "error": r.text[:300], "key_prefix": ghost_key[:14]}
        if r.status == 200:
            return False, "无效 API Key 竟通过鉴权并返回 200——鉴权完全失效（严重项）", observations
        if r.status == 401:
            if _error_body(r.body) or r.text:
                return True, "无效 API Key 被 401 拒绝且返回错误信息，鉴权契约成立", observations
            return True, "无效 API Key 被 401 拒绝（错误体为空，留痕）", observations, "warning"
        if r.status == 403:
            return True, f"无效 API Key 返回 403（非 401，结构差异留痕）：{r.text[:120]}", observations, "warning"
        if _is_rejection(r.status):
            return True, f"无效 API Key 返回 {r.status}（非 401/403，结构差异留痕）", observations, "warning"
        return False, f"无效 API Key 返回非预期状态 HTTP {r.status}", observations

    async def case_g61_unknown_model(self, model: str):
        ghost = f"gpt-6.1-nonexistent-{uuid.uuid4().hex[:8]}"
        r = await self.responses(ghost, input="只回复 OK", max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        observations = {"requested_model": ghost, "http_status": r.status, "error": r.text[:300]}
        if r.status == 200:
            return False, "未知模型竟被路由成功，身份/路由契约不成立", observations
        not_found = _model_not_found_reason(ghost, r.status, r.body)
        if not_found:
            return True, not_found, observations
        if r.status == 404:
            if _error_body(r.body):
                return True, "未知模型被拒绝（HTTP 404）且返回结构化 error，模型校验契约成立", observations
            return False, "未知模型返回 404 但响应体不是结构化 error（可能是网关 404 页）", observations, "blocked"
        if _is_rejection(r.status) and _error_body(r.body):
            return True, f"未知模型被拒绝（HTTP {r.status}）且返回结构化 error，模型校验契约成立", observations
        if _is_rejection(r.status):
            return False, f"未知模型返回 HTTP {r.status} 但响应体不是结构化 error", observations
        return False, f"未知模型返回非预期状态 HTTP {r.status}（未给出指向模型不存在的结构化错误）", observations

    async def case_g61_models_list(self, model: str):
        """P2：GET /models 应出现被测档位的身份令牌（按档位画像交叉验证）。"""
        tier = self._tier(model)
        flagship = tier.model_id if tier else GPT61_FLAGSHIP
        tokens = tier.accepted_tokens if tier else GPT61_MODEL_TOKENS
        r = await self.raw_request("GET", "models")
        observations = {"http_status": r.status, "body_head": r.text[:200], "tier": flagship}
        if r.status == 404:
            return False, "/models 端点不存在（HTTP 404），身份口径无法交叉验证", observations, "blocked"
        if r.status != 200:
            return True, f"/models 返回非预期状态 HTTP {r.status}（端点可用性留痕）", observations, "warning"
        body = r.body or {}
        entries = body.get("data") or []
        ids = {str(e.get("id") or "") for e in entries if isinstance(e, dict)}
        if body.get("object") != "list" or not entries or not any(ids):
            return False, f"/models 响应结构不符契约（object={body.get('object')!r}，条目数={len(entries)}）", observations
        present = [token for token in tokens if token in ids]
        observations["tier_tokens_present"] = present
        if flagship in ids:
            return True, f"/models 结构合规且包含 {flagship}，与档位身份口径交叉一致", observations
        if present:
            return True, f"/models 结构合规，仅出现 {present}（未含权威 ID {flagship}，留痕）", observations, "warning"
        return True, f"/models 结构合规但未出现 {flagship} 的任何标识（列表口径滞后，留痕）", observations, "warning"

    async def case_g61_input_tokens_endpoint(self, model: str):
        text = "输入 token 计数端点交叉验证用例（GPT-6.1）。" + ("A1B2C3。" * 200)
        count = await self.request("responses/input_tokens", {"model": model, "input": text})
        observations = {"count_status": count.status, "count_body": str(count.text)[:200]}
        if count.status == 404:
            return False, "/responses/input_tokens 端点不存在（HTTP 404）", observations, "blocked"
        n = (count.body or {}).get("input_tokens")
        if count.status != 200 or not isinstance(n, int) or n <= 0:
            return True, f"计数端点未返回有效 input_tokens（HTTP {count.status}），留痕", observations, "warning"
        real = await self.responses(model, input=text, reasoning={"effort": "low"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        real_n = (_usage(real.body) or {}).get("input_tokens")
        observations["real_status"] = real.status
        observations["real_input_tokens"] = real_n
        if real.status != 200 or not isinstance(real_n, int) or real_n <= 0:
            return True, f"计数端点返回 {n}，但真实请求 usage 缺失无法交叉验证（留痕）", observations, "warning"
        deviation = abs(n - real_n) / max(n, real_n, 1)
        observations["deviation"] = round(deviation, 4)
        if deviation <= 0.10:
            return True, f"计数端点与真实 usage 同量级（{n} vs {real_n}，偏差 {deviation:.1%}）", observations
        return True, f"计数端点与真实 usage 偏离过大（{n} vs {real_n}，偏差 {deviation:.1%}；留痕）", observations, "warning"

    async def case_g61_stream_semantic_events(self, model: str):
        r = await self.responses(model, input="计算 29*13，只给最终答案。", stream=True, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        s = r.stream or {}
        raw_events = s.get("raw_sse_events") or []
        has_done = any(str(e).strip() == "[DONE]" for e in raw_events)
        types = s.get("event_types") or []
        seqs: list[int] = []
        for raw in raw_events:
            try:
                obj = json.loads(str(raw))
            except json.JSONDecodeError:
                continue
            if isinstance(obj.get("sequence_number"), int):
                seqs.append(obj["sequence_number"])
        monotonic = all(b > a for a, b in zip(seqs, seqs[1:]))
        has_core = all(t in types for t in _RESPONSES_STREAM_CORE_EVENTS)
        observations = {"event_count": len(types), "event_head": types[:3], "event_tail": types[-3:],
                        "done_marker_present": has_done, "sequence_numbers": seqs[:12], "monotonic": monotonic}
        if r.status != 200:
            return False, f"流式请求应成功（HTTP {r.status}）", observations
        if has_done:
            return False, "Responses 事件流出现 [DONE]——Chat 收尾语义混入 Responses 协议（典型网关改写特征）", observations
        if seqs and not monotonic:
            return False, f"sequence_number 非严格单调递增：{seqs[:12]}", observations
        if has_core and monotonic:
            return True, "语义事件齐备（created/delta/completed）、序号严格单调且无 [DONE]，协议纯净", observations
        return True, f"流式事件不完整（核心事件缺失：{[t for t in _RESPONSES_STREAM_CORE_EVENTS if t not in types]}；留痕）", observations, "warning"

    # ---- FINGERPRINT 满血真实性指纹（6.1 专属反向契约 + 随机挑战）----
    async def case_g61_auth_identity(self, model: str):
        """AUTH：响应 model 必须逐字命中被测档位的权威 ID。"""
        tier = self._tier(model)
        flagship = tier.model_id if tier else GPT61_FLAGSHIP
        tokens = tier.accepted_tokens if tier else GPT61_MODEL_TOKENS
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "low"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        m = str((r.body or {}).get("model") or "")
        observations = {"requested_model": model, "response_model": m, "http_status": r.status, "tier": flagship}
        if r.status != 200:
            return False, f"身份取证请求失败（HTTP {r.status}）", observations, "blocked"
        if m == flagship:
            return True, f"响应 model 逐字命中权威 ID {flagship}，身份一致", observations
        if m in tokens:
            return True, f"响应 model={m!r} 属该档位令牌但非权威 ID {flagship}（别名回显，留痕）", observations, "warning"
        return False, f"响应 model={m!r} 不是 {flagship}（疑似降级/错配后端），身份指纹不成立", observations

    async def case_g61_auth_effort_floor(self, model: str):
        """AUTH：档位下限指纹（按档位画像）。

        档位集无 none/minimal（6.1 / Astra）：该档位被接受即需解释——先排除网关口径问题；
        档位集含 none/minimal（Sol / Luna）：下限本就该被接受，"被拒"才是异常。
        """
        tier = self._tier(model)
        label = tier.label if tier else "该模型"
        expectations = tier.rejected_efforts if tier else GPT61_REJECTED_EFFORTS
        if not expectations:
            floor_rows: dict[str, dict[str, Any]] = {}
            for effort in GPT6_EFFORT_FLOOR:
                r = await self.responses(model, input="只回复 OK", reasoning={"effort": effort}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
                floor_rows[effort] = {"http_status": r.status, "error": r.text[:200]}
            observations = {"tier": label, "expected_rejected": [], "floor_efforts": floor_rows}
            rejected = [e for e, row in floor_rows.items() if row["http_status"] != 200]
            if rejected:
                return False, (f"{label} 官方档位集含 {rejected}，但实测被拒绝"
                               f"（{ {e: row['http_status'] for e, row in floor_rows.items()} }）——"
                               "档位下限与文档不符"), observations
            return True, (f"{label} 档位下限为 {list(GPT6_EFFORT_FLOOR)} 且均被接受，"
                          "档位下限指纹成立"), observations
        outcomes: dict[str, dict[str, Any]] = {}
        for effort in expectations:
            r = await self.responses(model, input="只回复 OK", reasoning={"effort": effort}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
            outcomes[effort] = {"http_status": r.status, "error": r.text[:200],
                                "evidence": _effort_rejection_evidence(r.status, r.body, effort)}
        observations = {"tier": label, "expected_rejected": list(expectations), "rejected_efforts": outcomes}
        accepted = [e for e, row in outcomes.items() if row["http_status"] == 200]
        if accepted:
            relaxed = await self.gateway_ignores_model_effort_floor(model, tier) if tier else None
            if relaxed:
                return False, (f"{label} 官方已移除的档位 {accepted} 被 200 接受，但{relaxed}——"
                               "该发现属网关口径问题，不能用于判定模型身份"), observations, "blocked"
            return False, (f"{label} 已移除的档位 {accepted} 竟被 200 接受——"
                           "接受这些档位的后端与该档位契约不符，身份指纹不成立"), observations
        evidenced = [e for e, row in outcomes.items() if row["evidence"]]
        if len(evidenced) == len(outcomes):
            return True, (f"{list(expectations)} 均被拒且错误指向 reasoning.effort，"
                          f"{label} 档位下限指纹成立"), observations
        if all(_is_rejection(row["http_status"]) for row in outcomes.values()):
            return True, f"{list(expectations)} 均被 4xx 拒绝（未含档位文案，留痕）", observations, "warning"
        return False, (f"{list(expectations)} 返回非预期状态："
                       f"{ {e: row['http_status'] for e, row in outcomes.items()} }"), observations

    async def case_g61_auth_chat_tool_plane(self, model: str):
        """AUTH：Chat 工具面指纹（按档位画像判断；协议改写证据出现时归因于网关链路）。"""
        tier = self._tier(model)
        policy = tier.chat_tools if tier else CHAT_TOOLS_REJECTED
        label = tier.label if tier else "该档位"
        tool = self._tool_chat()
        message = [{"role": "user", "content": "查询订单 A-731 的状态，必须调用 get_order_status 工具。"}]
        if policy == CHAT_TOOLS_NONE_ONLY:
            none_r = await self.chat(model, message, tools=[tool], reasoning_effort="none", max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
            none_calls = self._chat_tool_calls(none_r.body)
            mid_r = await self.chat(model, message, tools=[tool], reasoning_effort="medium", max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
            mid_calls = self._chat_tool_calls(mid_r.body)
            observations = {"policy": policy, "tier": label,
                            "none": {"http_status": none_r.status, "tool_calls": none_calls[:1], "error": none_r.text[:200]},
                            "non_none": {"http_status": mid_r.status, "tool_calls": mid_calls[:1],
                                         "translation_evidence": _chat_translation_evidence(mid_r.body)}}
            if none_r.status != 200 or not none_calls:
                return False, (f"{label} 官方允许 effort=none 下的 Chat 函数调用，但实测未发起"
                               f"（HTTP {none_r.status}）"), observations
            if _is_rejection(mid_r.status):
                return True, (f"{label} 工具面指纹成立：none 下真发起工具调用、非 none 被拒"
                              f"（HTTP {mid_r.status}）"), observations
            if mid_r.status == 200 and mid_calls:
                translation = observations["non_none"]["translation_evidence"]
                if translation:
                    return False, (f"非 none 下 Chat + tools 仍成功发起工具调用，响应带协议改写证据"
                                   f"（{translation}）——网关翻译绕过了档位约束（信任级：协议改写）"), observations
                return False, (f"非 none 下 Chat + tools 成功发起工具调用，"
                               f"与 {label}「仅 effort=none 可用」的档位约束不符"), observations
            return True, "非 none 下 Chat + tools 未发起工具调用（网关静默忽略，留痕）", observations, "warning"
        r = await self.chat(model, message, tools=[tool], max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        calls = self._chat_tool_calls(r.body)
        translation = _chat_translation_evidence(r.body)
        observations = {"policy": policy, "tier": label, "http_status": r.status, "tool_calls": calls[:1],
                        "translation_evidence": translation, "error": r.text[:200]}
        if r.status == 200 and calls:
            if translation:
                return False, (f"Chat + tools 成功发起工具调用，且响应带协议改写证据（{translation}）——"
                               "网关把 Chat 翻译成 Responses，绕过了工具面拒绝契约"
                               "（信任级：协议改写；归因于网关链路，不等于后端不是该档位）"), observations
            return False, (f"Chat Completions + tools 成功发起工具调用——{label} 工具面只在 Responses API，"
                           "此通道原生支持工具即说明后端不是该档位，工具面指纹不成立"), observations
        if _is_rejection(r.status):
            return True, f"Chat + tools 被拒绝（HTTP {r.status}），符合 {label} 工具面契约", observations
        if r.status == 200:
            return True, "Chat + tools 返回 200 但未发起工具调用（网关静默忽略，无法取证，留痕）", observations, "warning"
        return False, f"Chat + tools 返回非预期状态 HTTP {r.status}", observations, "blocked"

    async def case_g61_auth_random_tool(self, model: str):
        tool_name = f"lookup_{uuid.uuid4().hex[:8]}"
        marker = _rng_token()
        tool = {"type": "function", "name": tool_name, "description": "返回测试数据",
                "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                               "required": ["code"], "additionalProperties": False}}
        last: ApiResponse | None = None
        calls: list[dict[str, Any]] = []
        for prompt in self.TOOL_PRIMING_PROMPTS:
            r = await self.responses(model, input=prompt.format(tool=tool_name, marker=marker), tools=[tool],
                                     tool_choice="auto", max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
            last = r
            calls = self._function_calls(r.body)
            if r.status != 200 or calls:
                break
        if last is None or last.status != 200 or not calls:
            return False, "随机动态工具两轮诱导均未触发 function_call（探测动态工具理解能力失败）", {"http_status": getattr(last, "status", None), "tool_name": tool_name}, "blocked" if (last is None or last.status != 200) else "failed"
        args_raw = calls[0].get("arguments") or ""
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except json.JSONDecodeError:
            args = {}
        bound = args.get("code") == marker
        return bound, f"模型应发起对随机工具 {tool_name} 的调用且参数绑定随机串", {"tool_name": tool_name, "arguments": args, "marker": marker}

    async def case_g61_auth_random_json(self, model: str):
        marker = _rng_token()
        schema = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False}
        r = await self.responses(model, input=f"以 JSON 输出 value 字段，值必须是 {marker}。",
                                 text={"format": {"type": "json_schema", "name": "probe", "strict": True, "schema": schema}},
                                 max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        text = _output_text(r.body)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        return r.status == 200 and parsed == {"value": marker}, f"结构化输出应严格绑定随机串 {marker}（缓存/中转固定应答无法预知）", {"http_status": r.status, "text": text[:200], "marker": marker}

    async def case_g61_auth_random_memory(self, model: str):
        marker = _rng_token()
        first = await self.responses(model, input=f"记住暗号 {marker}，只回复已记住。", max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        response_id = (first.body or {}).get("id")
        observations = {"first_status": first.status, "first_id": response_id, "marker": marker}
        if first.status != 200 or not response_id:
            return False, "随机多轮记忆第一跳失败", observations, "blocked"
        second = await self.responses(model, input="暗号是什么？只回复暗号。", previous_response_id=response_id, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        text = _output_text(second.body)
        observations.update({"second_status": second.status, "second_error": second.text[:200], "text": text[:200]})
        limit = _account_limitation_reason(second.status, second.body, "previous_response_id 会话续接")
        if limit:
            return False, limit, observations, "blocked"
        return second.status == 200 and marker in text, f"跨轮记忆应复述随机暗号 {marker}", observations

    async def case_g61_auth_reasoning(self, model: str):
        """AUTH-G61-07：6.1 的 low 档仍会思考（reasoning tokens 应 >0）。"""
        r = await self.responses(model, input="计算 137*29 并解释每一步，最后给出结果。",
                                 reasoning={"effort": "low"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        items = _reasoning_items(r.body)
        tokens = _reasoning_tokens(r.body)
        observations = {"http_status": r.status, "reasoning_items": len(items), "reasoning_tokens": tokens}
        if r.status != 200:
            return False, f"取证请求失败 HTTP {r.status}", observations, "blocked"
        if isinstance(tokens, int) and tokens > 0:
            return True, f"low 档产生 reasoning_tokens={tokens}，符合 6.1「low 仍会思考」", observations
        if items:
            return True, "low 档产生 reasoning 输出项但 usage 未回显 reasoning_tokens（留痕）", observations, "warning"
        return False, ("low 档未产生任何 reasoning 产物，思考真实性的取证前提不满足"
                       "（同观测在能力面记 warning，指纹面因无法取证记 blocked）"), observations, "blocked"

    async def case_g61_auth_cache_evidence(self, model: str):
        """AUTH-G61-08：缓存计费证据（usage 明细字段 + 两跳命中增量）。"""
        options = {"mode": "explicit", "ttl": "30m"}
        first = await self.responses(model, input=self._g61_cache_input(), prompt_cache_options=options, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        await asyncio.sleep(2)
        second = await self.responses(model, input=self._g61_cache_input(), prompt_cache_options=options, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        first_details = _usage(first.body).get("input_tokens_details")
        second_details = _usage(second.body).get("input_tokens_details")
        has_fields = isinstance(first_details, dict) and isinstance(second_details, dict)
        base = _cached_input_tokens(first.body)
        hit = _cached_input_tokens(second.body)
        delta = hit - base
        observations = {"first_status": first.status, "second_status": second.status, "has_cache_detail_fields": has_fields,
                        "baseline_cached_tokens": base, "second_cached_tokens": hit, "delta_cached_tokens": delta,
                        "cache_write_tokens_second": _cache_write_tokens(second.body)}
        if first.status != 200 or second.status != 200:
            return False, f"缓存取证请求失败（HTTP {first.status}/{second.status}）", observations, "blocked"
        if not has_fields:
            return False, "usage 未暴露 input_tokens_details（6.1 文档称缓存与思考明细可观测），缓存计费无法取证", observations, "blocked"
        if delta >= 512 or _cache_write_tokens(second.body) > 0:
            return True, "usage 暴露缓存明细且第二跳观测到命中/写入增量，缓存计费口径成立", observations
        return True, f"usage 暴露缓存明细但未观测到命中增量（{delta}；显式缓存为尽力行为，留痕）", observations, "warning"

    async def case_g61_auth_injection(self, model: str):
        """AUTH-G61-09：中转注入识别（未传 instructions 却被填充 = 信任级失败）。"""
        r = await self.responses(model, input="1+1 等于几？只回复数字。", reasoning={"effort": "low"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        observations = {"http_status": r.status, "instructions": str((r.body or {}).get("instructions"))[:80],
                        "input_tokens": (_usage(r.body) or {}).get("input_tokens")}
        if r.status != 200:
            return False, f"注入取证请求失败（HTTP {r.status}）", observations, "blocked"
        injected = _injected_instructions_reason(r.body)
        if injected:
            return False, injected, observations
        tokens = (_usage(r.body) or {}).get("input_tokens")
        if isinstance(tokens, int) and tokens > _INPUT_TOKEN_INFLATION_THRESHOLD:
            return False, (f"短问句 input_tokens={tokens} 远超输入体量（阈值 {_INPUT_TOKEN_INFLATION_THRESHOLD}）——"
                           "疑似上游注入大段系统提示词，响应内容不由本次请求单独决定（信任级失败）"), observations
        if "instructions" not in (r.body or {}):
            return True, "响应未回显 instructions 字段，无法确认是否被注入（证据不全，留痕）", observations, "warning"
        return True, "未传 instructions 时响应 instructions 为空且 input_tokens 正常，无注入迹象", observations

    async def case_g61_auth_nonstandard_fields(self, model: str):
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "low"}, max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        if r.status != 200:
            return False, f"字段取证请求失败（HTTP {r.status}）", {"http_status": r.status}, "blocked"
        extra = _relay_fields(r.body)
        observations = {"extra_fields": extra[:8], "response_keys": sorted((r.body or {}).keys())[:20]}
        if extra and any(any(m in k for m in _RELAY_FIELD_MARKERS) for k in extra):
            return False, f"响应出现中转/多供应商网关标记字段：{extra}（非标准字段即网关改写证据）", observations
        if extra:
            return True, f"响应含 schema 外字段 {extra[:6]}（未见已知中转标记，留痕）", observations, "warning"
        return True, "响应顶层字段全部落在 OpenAI Responses schema 白名单内，无非标准字段", observations

    async def case_g61_auth_response_id_authenticity(self, model: str):
        r = await self.responses(model, input="只回复 OK", max_output_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        rid = str((r.body or {}).get("id") or "")
        observations = {"first_status": r.status, "response_id": rid}
        if r.status != 200 or not rid:
            return False, "首跳未取得 response id", observations, "blocked"
        if rid.startswith("chatcmpl-"):
            return False, f"response id 形如 chatcmpl-（{rid[:24]}…）——协议混用，Responses 持久化真实性不成立", observations
        if not _RESPONSE_ID_PATTERN.match(rid):
            return True, f"response id 格式不合规（{rid[:24]}…），无法确证官方持久化形态（留痕）", observations, "warning"
        fetched = await self.raw_request("GET", f"responses/{rid}")
        observations["fetch_status"] = fetched.status
        if fetched.status == 404:
            return False, "id 格式合规但 GET /responses/{id} 检索 404——持久化真实性存疑（可检索失败）", observations, "blocked"
        if fetched.status != 200:
            return False, f"检索端点返回 HTTP {fetched.status}，真实性无法验证", observations, "blocked"
        same = (fetched.body or {}).get("id") == rid
        observations["fetch_id"] = (fetched.body or {}).get("id")
        if not same:
            return False, "检索回的 id 与首跳不一致，持久化真实性不成立", observations
        return True, f"id 格式合规且可检索回同一对象（{rid[:24]}…），Responses 持久化真实性成立", observations

    async def case_g61_auth_chat_not_translated(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], max_completion_tokens=CONTENT_MIN_OUTPUT_TOKENS)
        body = r.body or {}
        evidence = _chat_translation_evidence(body)
        observations = {"http_status": r.status, "object": body.get("object"), "id": str(body.get("id") or "")[:32],
                        "translation_evidence": evidence}
        if r.status != 200:
            return False, f"Chat 请求应成功（HTTP {r.status}）", observations, "blocked"
        if evidence:
            return False, f"Chat 响应被网关改写：{evidence}——参数语义将丢失，信任级失败", observations
        if body.get("object") == "chat.completion" and str(body.get("id") or "").startswith("chatcmpl-"):
            return True, "Chat 响应严格符合 Chat 契约（object/id 形态正确），协议未被改写", observations
        return True, f"Chat 响应缺少契约标识字段（object={body.get('object')!r}），无法确证协议未被改写（留痕）", observations, "warning"

    # ------------------------------------------------------------------
    # 产物类用例（三档共用同一 handler；不涉档位契约，只判"是否拿到可查看的完整产物"）
    # ------------------------------------------------------------------
    async def case_pelican_svg(self, model: str):
        """鹈鹕骑车 SVG：固定提示词直出，可按「请求轮数 × 并发数」一次跑多轮。

        轮数与并发由 ``self.pelican_rounds`` / ``self.pelican_concurrency`` 给出（默认 1×1，
        即原来的单轮行为）：同一句提示词重复 N 次，各轮结果一并落盘到同一结果目录。
        **提示词与判据跨轮次完全一致**——多轮只增加样本量，不改变任何期望值。

        产物写在结果目录（``self.audit_dir``，即 Web 记录页打开的那个目录）：

        * 单轮：``pelican.html``（内嵌 SVG 的独立页面）、``pelican.svg``（SVG 原文）、
          ``pelican_response.md``（模型完整响应原文）——与引入多轮之前完全一致；
        * 多轮：``pelican.html`` 变成**总览页**（各轮渲染视图排在一起，含逐轮状态与耗时，
          并链接到各轮原始文件）、``pelican-<轮次>.svg``（各成功轮的 SVG 原文）、
          ``pelican_response.md``（按轮分段的全部响应原文）。

        ``observations["artifacts"]`` 声明产物清单，``write_report`` 的「产物」小节据此
        生成可直接点击的链接；Web 侧由 ``/api/runs/<id>/artifact/<name>`` 提供同一目标。

        裁定只针对"这些轮次有没有都拿到完整产物"：
        * 全部轮次 200 + 成对开闭的 SVG ⇒ passed；
        * 存在**真实缺陷轮**（拿到响应却无完整 SVG / 非 200 且非瞬时）⇒ failed——即使并发中
          另有无响应的轮次，也保留该结论（``keeps_failure_verdict``，见 ``run()``）；
        * 只有上游瞬时故障轮（5xx / 连接错误，已按 ``max_retries`` 重试）⇒ 本处记 failed，
          由 ``run()`` 统一改判 blocked：没有有效响应就不谈模型能力，建议重跑。
        """
        rounds = max(1, int(getattr(self, "pelican_rounds", 1) or 1))
        # 并发不超过轮数：多出来的 worker 没有活干，只会让"并发数"这个参数骗人。
        concurrency = max(1, min(int(getattr(self, "pelican_concurrency", 1) or 1), rounds))
        semaphore = asyncio.Semaphore(concurrency)

        async def one_round(index: int) -> dict[str, Any]:
            async with semaphore:
                started = time.perf_counter()
                try:
                    r = await self.chat(model, [{"role": "user", "content": PELICAN_PROMPT}],
                                        max_completion_tokens=PELICAN_MAX_OUTPUT_TOKENS)
                except Exception as exc:  # 单轮传输层异常：记进本轮，不让整批变成 error
                    return _pelican_round(index, None, "", None, f"{type(exc).__name__}: {exc}",
                                          time.perf_counter() - started)
                body = r.body if isinstance(r.body, dict) else None
                content = str(_chat_message(body).get("content") or "")
                return _pelican_round(index, r.status, content, _chat_finish_reason(body), None,
                                      time.perf_counter() - started)

        # 逐轮请求同时发出（受限并发）；结果按轮次排序，保证证据与产物的顺序稳定可比。
        records = sorted(await asyncio.gather(*(one_round(i) for i in range(1, rounds + 1))),
                         key=lambda item: item["round"])
        ok = [item for item in records if item["svg_found"]]
        defects = [item for item in records if not item["svg_found"] and not item["transient"]]
        transients = [item for item in records if not item["svg_found"] and item["transient"]]

        artifacts, artifact_error = self._write_pelican_artifacts(records)
        first = records[0]
        observations: dict[str, Any] = {
            "http_status": first["http_status"],
            "prompt_chars": len(PELICAN_PROMPT),
            "rounds_total": rounds,
            "rounds_passed": len(ok),
            "rounds_failed": rounds - len(ok),
            "rounds_transient": len(transients),
            "concurrency": concurrency,
            "rounds": [{key: item[key] for key in ("round", "http_status", "response_chars", "finish_reason",
                                                   "svg_chars", "svg_found", "transient", "duration_seconds", "reason")}
                       for item in records],
            "response_chars": sum(item["response_chars"] for item in records),
            "finish_reason": first["finish_reason"],
            "svg_found": bool(ok),
            # 兼容既有单轮语义：首轮（多轮时为第一个成功轮）的 SVG 长度。
            "svg_chars": len(ok[0]["svg"]) if ok else 0,
            "response_preview": (first["content"] or "")[:400],
            "artifacts": artifacts,
        }
        if artifact_error:
            observations["artifact_error"] = artifact_error

        scope = "单轮" if rounds == 1 else f"{rounds} 轮（并发 {concurrency}）"
        if defects:
            # 真实缺陷轮存在：即便并发中另有瞬时故障轮，也要保留 failed（见 run() 的例外分支）。
            observations["keeps_failure_verdict"] = True
            listed = "；".join(f"第 {item['round']} 轮：{item['reason']}" for item in defects[:5])
            rest = f"，另 {len(defects) - 5} 轮同类" if len(defects) > 5 else ""
            extra = f"；另有 {len(transients)} 轮未取得响应（上游瞬时故障/连接错误）" if transients else ""
            return False, f"{scope}中 {len(defects)} 轮未产出完整 SVG（{listed}{rest}）{extra}", observations
        if ok:
            note = "；finish_reason=length，首轮正文虽已闭合但可能被截断，留痕" if first["finish_reason"] == "length" else ""
            head = (f"单轮直出完整 SVG（{len(ok[0]['svg'])} 字符，开闭标签成对）" if rounds == 1 else
                    f"{rounds} 轮全部直出完整 SVG（首轮 {len(ok[0]['svg'])} 字符，开闭标签成对）")
            tail = (f"，另 {len(transients)} 轮未取得响应（上游瞬时故障/连接错误）——本批不构成有效结论，建议重跑"
                    if transients else "")
            return False if transients else True, (f"{head}，产物已落盘到结果目录，"
                                                   f"可在记录页「产物」链接直接查看{note}{tail}"), observations
        return False, f"{scope}全部未取得有效响应（上游瞬时故障，已按 max_retries 重试）", observations

    def _write_pelican_artifacts(self, records: list[dict[str, Any]]) -> tuple[list[dict[str, str]], str | None]:
        """把各轮响应落盘成可直接查看的产物；返回 (产物清单, 错误说明)。

        未指定结果目录（裸 CLI 直跑）时不落盘，只留痕——响应原文仍在 requests.jsonl 里，
        报告里的预览也照给，不当成失败。
        """
        directory = self.audit_dir
        if directory is None:
            return [], "未指定结果目录，产物未落盘（响应原文仍见 requests.jsonl）"
        single = len(records) == 1
        try:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "pelican_response.md").write_text(
                records[0]["content"] if single else _pelican_transcript(records), encoding="utf-8")
            if single:
                svg = records[0]["svg"]
                if svg:
                    (directory / "pelican.svg").write_text(_trailing_newline(svg), encoding="utf-8")
                    (directory / "pelican.html").write_text(_pelican_viewer_html(svg), encoding="utf-8")
            else:
                for item in records:
                    if item["svg"]:
                        (directory / f"pelican-{item['round']}.svg").write_text(_trailing_newline(item["svg"]),
                                                                                encoding="utf-8")
                # 多轮一律写总览页（哪怕全部失败）：打开就能看到"哪几轮没出东西、为什么"。
                (directory / "pelican.html").write_text(_pelican_overview_html(records), encoding="utf-8")
        except OSError as exc:
            return [], f"产物落盘失败：{type(exc).__name__}: {exc}"
        # 声明顺序 = 报告与记录页的展示顺序：渲染视图在前，原始文件与响应原文在后。
        wanted = PELICAN_ARTIFACT_FILES if single else PELICAN_MULTI_ARTIFACT_FILES
        return [{"name": name, "title": PELICAN_ARTIFACT_TITLES.get(name, name)}
                for name in wanted if (directory / name).is_file()], None


def _report_artifacts(summary: dict[str, Any]) -> list[dict[str, str]]:
    """汇总本批所有用例声明的产物（去重保序）——write_report 的「产物」小节据此渲染。"""
    collected: list[dict[str, str]] = []
    for row in summary.get("results") or []:
        if not isinstance(row, dict):
            continue
        for item in (row.get("observations") or {}).get("artifacts") or []:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            name = str(item["name"])
            if any(existing["name"] == name for existing in collected):
                continue
            collected.append({"name": name, "title": str(item.get("title") or name)})
    return collected


def summarize_results(results: list[CaseResult]) -> dict[str, Any]:
    counts = {name: sum(r.status == name for r in results) for name in ("passed", "failed", "warning", "blocked", "error")}
    hard_failed = counts["failed"] + counts["error"]
    evaluated = counts["passed"] + hard_failed
    return {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "total": len(results),
            "passed": counts["passed"], "failed": hard_failed, "warning": counts["warning"], "blocked": counts["blocked"],
            "error": counts["error"], "pass_rate": counts["passed"] / evaluated if evaluated else 0.0,
            "all_passed": hard_failed == 0, "results": [asdict(r) for r in results]}


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 标题按模型档位给出，避免 GPT-6.x 批次被写成 "GPT-5.6 专项测试报告"。
    variant = str(summary.get("model_variant") or _variant_of(str(summary.get("case") or "")))
    title = {"gpt6.1": "GPT-6.1 专项测试报告", "gpt6": "GPT-6 家族专项测试报告"}.get(variant, "GPT-5.6 专项测试报告")
    model = summary.get("model")
    lines = ["# " + title, "", f"- 模型：{model}" if model else "- 模型：—"]
    if summary.get("pelican_rounds"):
        # 产物类用例可能一次跑多轮：报告必须自证执行规模，否则"7/10 轮成功"无从解释。
        lines.append(f"- 请求轮数：{summary['pelican_rounds']}"
                     f"（并发 {summary.get('pelican_concurrency') or 1}）")
    lines += [f"- 总计：{summary['total']}", f"- 通过：{summary['passed']}", f"- 失败：{summary['failed']}",
              f"- 警告：{summary['warning']}", f"- 阻塞/未观测：{summary['blocked']}",
              f"- 已评估用例通过率：{summary['pass_rate']:.1%}", "", "| 用例 | 模型 | 结果 | 耗时(s) | 说明 |", "|---|---|---:|---:|---|"]
    icons = {"passed": "✅ 通过", "failed": "❌ 失败", "error": "❌ 错误", "warning": "⚠️ 警告", "blocked": "⏸ 未验证"}
    for r in summary["results"]:
        detail = str(r["details"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {r['case_id']} {r['name']} | {r['model']} | {icons.get(r['status'], r['status'])} | {r['duration_seconds']:.2f} | {detail} |")
    lines.extend(_artifact_lines(summary))
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _artifact_lines(summary: dict[str, Any]) -> list[str]:
    """报告的「产物」小节：把用例声明的产物渲染成可直接点击的链接。

    链接走 Web 的 artifact 路由（`/api/runs/<id>/artifact/<name>`），前端 markdown 渲染器
    对站内相对链接同样出 `<a>`。无 record_id（裸 CLI 直跑）时只给结果目录内的文件名——
    报告本身与产物同目录，照着找即可。
    """
    artifacts = _report_artifacts(summary)
    if not artifacts:
        return []
    record_id = str(summary.get("record_id") or "").strip()
    lines = ["", "## 产物", "", "本批用例在结果目录内落盘的可直接查看产物：", ""]
    for artifact in artifacts:
        name, title = artifact["name"], artifact["title"]
        if record_id:
            lines.append(f"- [{title}](/api/runs/{record_id}/artifact/{name})（结果目录内 `{name}`）")
        else:
            lines.append(f"- {title}：结果目录内 `{name}`")
    lines += ["", "> 记录页上「产物」链接与上述目标一致；整批下载包里含同名文件。"]
    return lines
