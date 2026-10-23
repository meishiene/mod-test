"""对话窗口（自由输入单轮对话）。

``cases.py`` 提供展示元数据，``core.py`` 负责请求/SSE 累积/落盘，``runner.py``
负责从环境装配配置并调用统一收尾。
"""

from .cases import CASES, CHAT_CASE_ID, CHAT_PROMPT, DEFAULT_MAX_TOKENS

__all__ = ["CASES", "CHAT_CASE_ID", "CHAT_PROMPT", "DEFAULT_MAX_TOKENS"]
