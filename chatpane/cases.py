"""对话窗口（自由输入单轮对话）的展示元数据。

「对话窗口」不是能力判定套件，而是一个**取证式**的自由探针：把用户在界面上输入的
原文发出去，把上游返回的完整响应原样落盘（requests.jsonl / summary.json /
response.md / report.md），并复用平台的收尾与聚合口径，让每次对话都成为一条
可审计、可下载、可聚合的测试记录。

因此这里只有一个用例 ``CHAT_PROMPT``；请求数恒为 1，判定只看"请求是否真的拿到了
响应"，不看模型能力（能力判定由各专项套件负责）。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatCase:
    case_id: str
    title: str
    description: str
    cost: str
    protocols: tuple[str, ...]


CHAT_PROMPT = "CHAT_PROMPT"
CHAT_CASE_ID = CHAT_PROMPT

# ``protocols`` 是**家族**（不是协议 id）：与 platform 的
# ``supports_test_kind(protocol, kind, accepted_families)`` 口径一致。
# - ``openai`` 家族覆盖 openai / openai_responses / qianfan（三者在
#   chatpane 里共用同一份 Chat Completions/Responses 解析）；
# - ``volcengine`` 是独立家族（火山方舟），必须显式列出；
# - 图像面（``openai_images`` / ``gemini_images``）不在其中——图像协议虽然
#   家族同为 openai/gemini，但其 ``test_kinds`` 白名单不含 ``chat``，因此会被拦回。
CHAT_PROTOCOL_FAMILIES: tuple[str, ...] = ("openai", "volcengine", "anthropic", "gemini")

# 具体**协议 id** 白名单。Web 层用 `CHAT_PROTOCOL_FAMILIES` 做展示/匹配，执行器用这份
# id 白名单做最后一道闸：图像面协议（openai_images）的家族同样是 openai，只按家族判断
# 会把它放进文字面请求构造里，最终把 Chat Completions 请求体发到 /images/generations。
CHAT_PROTOCOLS: tuple[str, ...] = (
    "openai", "openai_responses", "volcengine", "qianfan", "anthropic", "gemini",
)

# 默认输出上限。思考型模型（Gemini 3.x / GPT-6 家族）的 thoughts 计入输出上限，
# 给小了会"零正文"——因此默认给一个足够宽的预算，界面可覆盖。
DEFAULT_MAX_TOKENS = 8192
MAX_MAX_TOKENS = 128000
# 提示词长度上限（Web 请求体上限 64 KiB，这里留出余量给系统提示词与 JSON 包装）。
MAX_PROMPT_CHARS = 20000
MAX_SYSTEM_CHARS = 4000

CASES: dict[str, ChatCase] = {
    CHAT_PROMPT: ChatCase(
        case_id=CHAT_PROMPT,
        title="对话窗口",
        description="把界面输入的原文发给被测模型，完整响应原样归档并回显",
        cost="低",
        protocols=CHAT_PROTOCOL_FAMILIES,
    ),
}
