"""Kimi（KVV）用例的测试面登记。

KVV 走官方 pytest（``vendor_verifier``），用例是 pytest 节点而不是本仓库的
``CASE_SPEC``，无法逐条过滤 —— 因此这里按**卡片**登记：

- ``KVV_TOOL_SCHEMA``（官方 408 个 tool_call_json_schema 用例）：严格 JSON Schema
  契约面 ⇒ ``channel``；可直接手动选择执行，仅保留结果、不计评分。
- ``KVV_PARAMS``（官方参数约束 16/18 条）：纯参数取值域 ⇒ ``channel``。
- ``KVV_K3_FEATURES``（46 条 response_format / tokenization / tool_choice）：
  主体是模型能不能按格式输出 ⇒ ``model``。
- ``KVV_K3_DYNAMIC_TOOLS`` / ``KVV_K3_THINKING_EFFORT``：K3 工具与思考能力 ⇒ ``model``。
- ``KVV_PROMPT_TOKENS``：官方文本/视觉 token 基线一致性 ⇒ ``verify``。
- ``K3_TEMP_SUITE``（68 条原子请求：连通/思考/采样/流式/JSON/工具/TTFT/缓存）：
  混合能力与性能观测，是 K3 型号验证的主力 ⇒ ``model``。
"""
from __future__ import annotations

from common.faces import FACE_CHANNEL, FACE_MODEL, FACE_VERIFY

CASE_FACES: dict[str, str] = {
    "KVV_K3_FEATURES": FACE_MODEL,
    "KVV_K3_DYNAMIC_TOOLS": FACE_MODEL,
    "KVV_K3_THINKING_EFFORT": FACE_MODEL,
    "KVV_PROMPT_TOKENS": FACE_VERIFY,
    "KVV_PARAMS": FACE_CHANNEL,
    "KVV_TOOL_SCHEMA": FACE_CHANNEL,
    "K3_TEMP_SUITE": FACE_MODEL,
}

UNREGISTERED: tuple[str, ...] = ()
