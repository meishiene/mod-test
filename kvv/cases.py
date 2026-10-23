"""Kimi 测试模块（原「KVV 测试」）的用例元数据。

模块内含两条执行链，同属 `test_kind="kvv"`：

- **官方 KVV pytest 套件**：`CASES` 里的六条，由 `kvv/runner.py` 直接驱动
  `vendor_verifier/` 官方用例，入口脚本 `scripts/run_kvv.py`。
- **K3 供应商临时案例集**：`TEMP_CASE_ID` 一条入口（内部 24 个合并案例 / 68 条
  原子请求），由 `kvv/temp_runner.py` 的清单流水线执行，入口脚本
  `scripts/run_kvv_temp.py`，清单见 `kvv/kimi_k3_temp_cases.json`。

两条链的判定口径与协议面不同（官方套件按官方 pytest 退出码，K3 临时案例集走
Chat Completions），Web 端按 case id 分流，不做混跑。
"""
from __future__ import annotations
from dataclasses import dataclass

# K3 临时案例集在模块里的入口 id（结果目录名同为 K3_TEMP_SUITE-<时间戳>）。
TEMP_CASE_ID = "K3_TEMP_SUITE"

# 可手动选择的独立专项；不参与模型评分，也不自动加入经济组合。
OPTIONAL_UNSCORED_CASES = frozenset({"KVV_TOOL_SCHEMA"})

# K3 临时案例集走 Chat Completions 面（官方 KVV 套件同为 openai / volcengine）。
TEMP_PROTOCOLS: tuple[str, ...] = ("openai", "volcengine")


@dataclass(frozen=True)
class KvvCase:
    id: str
    title: str
    description: str
    cost: str
    request_count: int
    concurrency: int
    suites: tuple[str, ...]
    protocols: tuple[str, ...] = ("openai", "volcengine")


CASES = {
    "KVV_K3_FEATURES": KvvCase(
        "KVV_K3_FEATURES", "官方 K3 Features（核心）", "46 个流式/非流式检查项：response_format、tokenization、tool_choice；动态工具、Thinking Effort 和 Prompt Token 基线由独立入口执行", "高", 46, 1,
        ("k3_features_core",),
    ),
    "KVV_K3_DYNAMIC_TOOLS": KvvCase(
        "KVV_K3_DYNAMIC_TOOLS", "官方 K3 动态工具", "54 个流式/非流式检查项：messages[].tools 动态声明、多轮工具、与全局工具共存、工具选择及非法请求拒绝", "高", 54, 1,
        ("k3_dynamic_tools",),
    ),
    "KVV_K3_THINKING_EFFORT": KvvCase(
        "KVV_K3_THINKING_EFFORT", "官方 K3 Thinking Effort", "26 个检查项（含官方跳过项）：思考默认行为、thinking.effort 与 reasoning_effort 优先级、推理历史保留和流式推理；单项可能多次请求并按官方标记重试", "高", 26, 1,
        ("k3_thinking_effort",),
    ),
    "KVV_PROMPT_TOKENS": KvvCase(
        "KVV_PROMPT_TOKENS", "官方 K3 Prompt Token 基线", "55 条流式请求（44 个文本、11 个视觉样本），校验 usage.prompt_tokens 在官方基线至基线+3 范围内；另含 4 个本地边界测试", "高", 55, 1,
        ("prompt_tokens",),
    ),
    "KVV_PARAMS": KvvCase(
        "KVV_PARAMS", "官方参数约束", "执行官方 tests/params 全部用例；默认按 K3 契约判定（16 条：temperature 正向 1.0、负向 0.6/0.0），KVV_PARAM_CONTRACT=k2 时回到 K2.5 档（18 条）。请求数按上限 18 计", "中", 18, 1,
        ("params",),
    ),
    "KVV_TOOL_SCHEMA": KvvCase(
        "KVV_TOOL_SCHEMA", "官方 Tool JSON Schema", "直接执行官方 tests/tool_call_json_schema 全部 408 个流式/非流式用例", "极高", 408, 4,
        ("tool_call_json_schema",),
    ),
}
