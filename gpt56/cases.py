from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class GPT56Case:
    id: str
    title: str
    description: str
    cost: str
    priorities: tuple[str, ...]
    request_count: int
    concurrency: int = 1
    protocols: tuple[str, ...] = ("openai",)
    # 模型档位：``gpt5.6``（GPT-5.6 Sol/Terra/Luna 三档）、``gpt6.1``（GPT-6.1 Sol 单快照）
    # 或 ``gpt6``（GPT-6 家族 gpt-6-sol / gpt-6-astra / gpt-6-luna）。
    # Web「GPT 测试」模块按该字段在三个档位间切换用例列表与默认 MODEL，
    # runner 侧据此只装配对应档位的 CASE_SPECS（避免把 5.6/6.1 判据套到 GPT-6 家族上）。
    model_variant: str = "gpt5.6"


# 产物类用例的展示文案（三档共用同一 handler 与同一句提示词，故文案也共用一份）。
# 提示词逐字冻结在 gpt56/core.py:PELICAN_PROMPT，这里是给人看的说明，改动不影响判据。
PELICAN_DESCRIPTION = (
    "固定提示词「你给我画一个鹈鹕骑自行车的SVG，直接输出完整SVG，不要read，不要调用imagegen」"
    "直出：判「是否拿到完整（开闭标签成对）SVG」，并把响应原文落盘成可直接查看的产物——"
    "pelican.html（浏览器打开即见渲染视图）、pelican.svg（原始文件）、pelican_response.md（完整响应原文）；"
    "报告的「产物」小节给可点击链接，记录页同样可点开查看。"
    "**支持一次跑多轮**：界面/CLI 可指定请求轮数与并发数（同一提示词重复 N 次，判据跨轮一致），"
    "多轮时 pelican.html 为总览页（各轮渲染视图排在一起），逐轮原始 SVG 落成 pelican-<轮次>.svg。"
)
PELICAN_COST = "低（单轮一次直出；多轮按轮数线性增长）"


CASES = {
    "GPT56_P0": GPT56Case("GPT56_P0", "GPT-5.6 核心专项", "Responses API 推理全生命周期（默认档位/none/档位/摘要/context）、工具闭环、Structured Outputs、previous_response_id 与加密推理回放、隐式缓存、双协议流式与截断可检测；三档模型可调用、别名路由、图像识别与 detail 档位、后台模式、响应对象结构契约", "中～高（图像与后台轮询用例偏高）", ("P0",), 22),
    "GPT56_P1": GPT56Case("GPT56_P1", "GPT-5.6 增强专项", "pro 模式、reasoning.context=all_turns 持久化、Programmatic Tool Calling、显式缓存选项与 TTL、非法 json_schema 拒绝、未知模型拒绝；显式缓存断点两跳（增量判据）、multi-agent beta", "中～高（pro 模式按聚合工作计费、multi-agent 多 subagent 并发）", ("P1",), 9),
    "GPT56_P2": GPT56Case("GPT56_P2", "GPT-5.6 契约与健壮性", "双协议 Usage 契约、流式 usage chunk、非法 temperature/messages/input 拒绝、Tool Choice none/required、Stop 序列、max_completion_tokens 边界；无效 Key 鉴权、错误对象四字段、/models 与 token 计数端点、流式语义事件与无 [DONE]", "中", ("P2",), 15),
    "GPT56_FINGERPRINT": GPT56Case("GPT56_FINGERPRINT", "GPT-5.6 满血真实性指纹", "随机挑战动态工具、结构化输出绑定、多轮记忆、思考摘要、流式完整性与 effort 吞参对照，识别疑似降级或中转特征；中转注入识别（instructions/非标准字段）、档位能力梯度、response id 真实性、Chat 协议未被改写", "中～高（档位梯度双档对照）", ("AUTH",), 11),
    # ------------------------------------------------------------------
    # 产物类（三档各一条，共用同一个 pelican_svg handler）——提示词逐字冻结、
    # 跨档同一套判据，所以不涉及档位期望值差异。详见 docs/GPT5.6专项测试用例设计.md。
    # ------------------------------------------------------------------
    "GPT56_PELICAN": GPT56Case(
        "GPT56_PELICAN", "鹈鹕骑车 SVG 生成", PELICAN_DESCRIPTION,
        PELICAN_COST, ("PELICAN",), 1,
    ),
    # ------------------------------------------------------------------
    # GPT-6.1（2026-09-29 DevDay 发布，单快照 gpt-6.1-sol）——不复用 5.6 判据，
    # 以 6.1 自身的 breaking change 与缓存/工具面契约为骨架。详见
    # docs/GPT6.1专项测试用例设计.md。
    # ------------------------------------------------------------------
    "GPT61_P0": GPT56Case(
        "GPT61_P0", "GPT-6.1 核心专项",
        "6.1 自身契约主线：模型身份（单快照 gpt-6.1-sol）、默认档位 medium、"
        "effort=none/minimal 必须被拒（6.1 头号 breaking change）、五档合法、"
        "Chat 纯文本可用但工具面只在 Responses、采样参数（temperature 族）与 effort 互斥、"
        "Responses 工具闭环与 Structured Outputs、previous_response_id 与加密推理回放、"
        "显式缓存断点（≥1024 前缀，增量判据）、configuration_update 切档不失效缓存、"
        "截断可检测、流式、instructions、图像输入与响应对象结构契约",
        "中～高（config_update 与缓存用例为多跳）", ("P0",), 19,
        model_variant="gpt6.1",
    ),
    "GPT61_P1": GPT56Case(
        "GPT61_P1", "GPT-6.1 增强专项",
        "6.1 工具面与缓存机制：Responses 内置工具逐项注册（web_search/file_search/code_interpreter/"
        "computer_use/apply_patch/hosted_shell/image_generation/skills/tool_search/mcp）、"
        "allowed_tools 工具开关、显式缓存 TTL 仅接受 30m（非法值应被拒）、"
        "configuration_update 相邻项被拒、/responses/compact 压缩端点可用性、"
        "模型 ID 严格解析（gpt-61-sol / gpt-6-1-sol 不应解析）",
        "中～高（内置工具注册逐项探测）", ("P1",), 6,
        model_variant="gpt6.1",
    ),
    "GPT61_P2": GPT56Case(
        "GPT61_P2", "GPT-6.1 契约与健壮性",
        "双协议 Usage 契约（含 cached_tokens/cache_write_tokens/reasoning_tokens 明细）、"
        "Chat 流式 usage chunk、非法 effort 的错误对象四字段、非法 input 拒绝、"
        "max_output_tokens 超上限拒绝、无效 Key 鉴权、未知模型拒绝、/models 端点与 6.1 身份、"
        "token 计数端点交叉验证、Responses 流式语义事件与无 [DONE]",
        "中", ("P2",), 11,
        model_variant="gpt6.1",
    ),
    "GPT61_FINGERPRINT": GPT56Case(
        "GPT61_FINGERPRINT", "GPT-6.1 满血真实性指纹",
        "6.1 专属反向契约指纹：模型身份令牌必须落在 gpt-6.1-sol、"
        "none/minimal 必须被拒（接受即后端不是 6.1）、Chat+tools 必须失败（成功即后端支持 5.6/6.0 工具面）；"
        "叠加随机挑战动态工具、随机结构化输出绑定、随机多轮记忆、思考真实性、缓存计费证据、"
        "中转注入识别、非标准字段、response id 真实性、Chat 协议未被改写",
        "中～高（反向契约 + 随机挑战）", ("AUTH",), 12,
        model_variant="gpt6.1",
    ),
    "GPT61_PELICAN": GPT56Case(
        "GPT61_PELICAN", "鹈鹕骑车 SVG 生成", PELICAN_DESCRIPTION,
        PELICAN_COST, ("PELICAN",), 1,
        model_variant="gpt6.1",
    ),
    # ------------------------------------------------------------------
    # GPT-6 家族（2026-09-22 发布 gpt-6-sol / gpt-6-astra；luna 网关未开放通道）——
    # 与 6.1 共用 handler 池，但档位相关期望值由 ModelTier 矩阵按被测模型解析。
    # 详见 docs/GPT6家族专项测试用例设计.md。
    # ------------------------------------------------------------------
    "GPT6_P0": GPT56Case(
        "GPT6_P0", "GPT-6 核心专项",
        "家族共用主线 + 档位相关判据：模型身份（权威 ID 逐字命中被测档位）、默认档位、"
        "档位下限双向判定（astra 必须拒 none/minimal；sol/luna 必须接受）、其余档位合法、"
        "Chat 纯文本可用、Chat 工具面按档位双向（astra 须拒；sol/luna 仅 none 下可用）、"
        "采样参数互斥、Responses 工具闭环与 Structured Outputs、previous_response_id 与加密推理回放、"
        "显式缓存断点（≥1024 前缀）、configuration_update 切档不失效缓存、截断、流式、"
        "instructions、图像输入与响应对象结构契约",
        "中～高（config_update 与缓存用例为多跳）", ("P0",), 19,
        model_variant="gpt6",
    ),
    "GPT6_P1": GPT56Case(
        "GPT6_P1", "GPT-6 增强专项",
        "家族共用工具面与缓存机制：Responses 内置工具逐项注册（web_search/file_search/"
        "code_interpreter/computer_use/apply_patch/hosted_shell/image_generation/skills/"
        "tool_search/mcp）、allowed_tools 工具开关、显式缓存 TTL 仅接受 30m、"
        "configuration_update 相邻项被拒、/responses/compact 压缩端点可用性、"
        "模型 ID 严格解析（按被测档位生成邻近错误写法）",
        "中～高（内置工具注册逐项探测）", ("P1",), 6,
        model_variant="gpt6",
    ),
    "GPT6_P2": GPT56Case(
        "GPT6_P2", "GPT-6 契约与健壮性",
        "双协议 Usage 契约（含 cached_tokens/cache_write_tokens/reasoning_tokens 明细）、"
        "Chat 流式 usage chunk、非法 effort（必非法值探测）的错误对象四字段、非法 input 拒绝、"
        "max_output_tokens 超上限拒绝、无效 Key 鉴权、未知模型拒绝、/models 端点与档位身份交叉、"
        "token 计数端点交叉验证、Responses 流式语义事件与无 [DONE]",
        "中", ("P2",), 11,
        model_variant="gpt6",
    ),
    "GPT6_FINGERPRINT": GPT56Case(
        "GPT6_FINGERPRINT", "GPT-6 满血真实性指纹",
        "档位感知的反向契约指纹：身份令牌须落在被测档位权威 ID、档位下限指纹按档位画像双向判定"
        "（astra 接受 none 即为异常；sol/luna 拒 none 才是异常），且「该拒未拒」须先探测网关声明的"
        "effort 合法性集合——若与该模型官方档位集不符（中转层统一白名单），记 blocked 而非 failed；"
        "Chat 工具面指纹同时取证协议改写（usage.billing_usage.source）以区分网关翻译与后端能力；"
        "叠加随机挑战动态工具、随机结构化输出绑定、随机多轮记忆、思考真实性、缓存计费证据、"
        "中转注入识别、非标准字段、response id 真实性、Chat 协议未被改写",
        "中～高（反向契约 + 随机挑战 + 网关口径探测）", ("AUTH",), 12,
        model_variant="gpt6",
    ),
    "GPT6_PELICAN": GPT56Case(
        "GPT6_PELICAN", "鹈鹕骑车 SVG 生成", PELICAN_DESCRIPTION,
        PELICAN_COST, ("PELICAN",), 1,
        model_variant="gpt6",
    ),
}

# 只有产物类用例接受「请求轮数 / 并发数」（其余用例的请求数是写死的）。
# Web 层据此决定：① 是否给前端暴露轮数/并发输入框；② 是否给 CLI 追加 --requests/--concurrency。
PELICAN_CASE_IDS: tuple[str, ...] = tuple(
    name for name, case in CASES.items() if case.priorities == ("PELICAN",)
)
