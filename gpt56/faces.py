"""GPT 系列（5.6 / 6.1 / 6）用例的测试面登记。

三档共用同一个 handler 池（``g61_*`` 被 6.1 与 6 家族同时引用），所以按 **handler**
登记一次即可覆盖全部 156 条规格；新增用例只要 handler 已登记就自动继承面归属，
未登记的 handler 会被 ``tests/test_faces.py`` 抓出来。

分层要点
--------
- ``model``：能不能产出正确内容 —— 思考档位、工具闭环、结构化输出、多轮状态、
  缓存命中、截断、流式、图像、产物（鹈鹕 SVG）。
- ``verify``：身份与指纹 —— 身份令牌、effort 下限反向契约（接受 none 即非 6.1）、
  Chat 工具面反向契约、随机挑战、思考真实性、注入识别、response id 真实性。
- ``channel``：端点/严格语义/beta —— 内置工具注册、allowed_tools、PTC、multi-agent、
  缓存 TTL、compact 端点、模型 ID 严格解析、usage 字段契约、鉴权/错误对象/models/
  token 计数、参数越界拒绝。
"""
from __future__ import annotations

from common.faces import FACE_CHANNEL, FACE_MODEL, FACE_VERIFY

#: handler → 面。未列出的 handler 默认 model（宁可多跑，不可漏跑）。
HANDLER_FACES: dict[str, str] = {
    # ---------------- 模型能力：正向可观测 ----------------
    "basic": FACE_MODEL,
    "default_effort": FACE_MODEL,
    "effort_none": FACE_MODEL,
    "effort_levels": FACE_MODEL,
    "reasoning_summary": FACE_MODEL,
    "reasoning_context": FACE_MODEL,
    "tool_loop": FACE_MODEL,
    "json_schema_output": FACE_MODEL,
    "chat_compat": FACE_MODEL,
    "previous_response_state": FACE_MODEL,
    "encrypted_replay": FACE_MODEL,
    "cache_hit": FACE_MODEL,
    "truncation": FACE_MODEL,
    "responses_stream": FACE_MODEL,
    "chat_stream": FACE_MODEL,
    "instructions": FACE_MODEL,
    "tier_models": FACE_MODEL,
    "alias_flagship": FACE_MODEL,
    "image_recognition": FACE_MODEL,
    "pro_mode": FACE_MODEL,
    "context_all_turns": FACE_MODEL,
    "explicit_cache": FACE_MODEL,
    "explicit_cache_breakpoint": FACE_MODEL,
    "responses_usage": FACE_MODEL,
    "chat_usage": FACE_MODEL,
    "chat_stream_usage": FACE_MODEL,
    "tool_choice_none": FACE_MODEL,
    "stop_sequence": FACE_MODEL,
    "max_completion_tokens": FACE_MODEL,
    "pelican_svg": FACE_MODEL,
    # g61 家族
    "g61_default_effort": FACE_MODEL,
    "g61_effort_levels": FACE_MODEL,
    "g61_chat_text": FACE_MODEL,
    "g61_tool_loop": FACE_MODEL,
    "g61_json_schema_output": FACE_MODEL,
    "g61_previous_response_state": FACE_MODEL,
    "g61_encrypted_replay": FACE_MODEL,
    "g61_explicit_cache_breakpoint": FACE_MODEL,
    "g61_config_update_cache": FACE_MODEL,
    "g61_truncation": FACE_MODEL,
    "g61_responses_stream": FACE_MODEL,
    "g61_instructions": FACE_MODEL,
    "g61_image_recognition": FACE_MODEL,
    "g61_responses_usage": FACE_MODEL,
    "g61_chat_usage": FACE_MODEL,
    "g61_chat_stream_usage": FACE_MODEL,

    # ---------------- 验真：有没有被换掉 ----------------
    "auth_random_tool": FACE_VERIFY,
    "auth_random_json": FACE_VERIFY,
    "auth_random_memory": FACE_VERIFY,
    "auth_reasoning_summary": FACE_VERIFY,
    "auth_stream_events": FACE_VERIFY,
    "auth_effort_contrast": FACE_VERIFY,
    "auth_injection": FACE_VERIFY,
    "auth_nonstandard_fields": FACE_VERIFY,
    "auth_tier_gradient": FACE_VERIFY,
    "auth_response_id_authenticity": FACE_VERIFY,
    "auth_chat_not_translated": FACE_VERIFY,
    "g61_identity": FACE_VERIFY,
    "g61_effort_none_rejected": FACE_VERIFY,
    "g61_effort_minimal_rejected": FACE_VERIFY,
    "g61_chat_tools_rejected": FACE_VERIFY,
    "g61_auth_identity": FACE_VERIFY,
    "g61_auth_effort_floor": FACE_VERIFY,
    "g61_auth_chat_tool_plane": FACE_VERIFY,
    "g61_auth_random_tool": FACE_VERIFY,
    "g61_auth_random_json": FACE_VERIFY,
    "g61_auth_random_memory": FACE_VERIFY,
    "g61_auth_reasoning": FACE_VERIFY,
    "g61_auth_cache_evidence": FACE_VERIFY,
    "g61_auth_injection": FACE_VERIFY,
    "g61_auth_nonstandard_fields": FACE_VERIFY,
    "g61_auth_response_id_authenticity": FACE_VERIFY,
    "g61_auth_chat_not_translated": FACE_VERIFY,

    # ---------------- 通道契约：端点 / beta / 严格语义 ----------------
    "image_detail_tiers": FACE_CHANNEL,
    "background_mode": FACE_CHANNEL,
    "response_object_contract": FACE_CHANNEL,
    "ptc": FACE_CHANNEL,
    "cache_ttl": FACE_CHANNEL,
    "strict_invalid_schema": FACE_CHANNEL,
    "unknown_model": FACE_CHANNEL,
    "multi_agent": FACE_CHANNEL,
    "temperature_invalid": FACE_CHANNEL,
    "messages_invalid": FACE_CHANNEL,
    "tool_choice_required": FACE_CHANNEL,
    "invalid_input": FACE_CHANNEL,
    "invalid_key_auth": FACE_CHANNEL,
    "error_object_contract": FACE_CHANNEL,
    "models_list": FACE_CHANNEL,
    "input_tokens_endpoint": FACE_CHANNEL,
    "stream_semantic_events": FACE_CHANNEL,
    "g61_sampling_params_rejected": FACE_CHANNEL,
    "g61_response_object_contract": FACE_CHANNEL,
    "g61_builtin_tools": FACE_CHANNEL,
    "g61_allowed_tools": FACE_CHANNEL,
    "g61_cache_ttl": FACE_CHANNEL,
    "g61_config_update_adjacent": FACE_CHANNEL,
    "g61_compact_endpoint": FACE_CHANNEL,
    "g61_model_id_strictness": FACE_CHANNEL,
    "g61_effort_error_contract": FACE_CHANNEL,
    "g61_invalid_input": FACE_CHANNEL,
    "g61_max_output_tokens_rejected": FACE_CHANNEL,
    "g61_invalid_key_auth": FACE_CHANNEL,
    "g61_unknown_model": FACE_CHANNEL,
    "g61_models_list": FACE_CHANNEL,
    "g61_input_tokens_endpoint": FACE_CHANNEL,
    "g61_stream_semantic_events": FACE_CHANNEL,
}


def _build() -> tuple[dict[str, str], tuple[str, ...]]:
    from .core import CASE_SPECS

    table: dict[str, str] = {}
    unregistered: list[str] = []
    for spec in CASE_SPECS:
        face = HANDLER_FACES.get(spec.handler)
        if face is None:
            unregistered.append(f"{spec.case_id}:{spec.handler}")
            face = FACE_MODEL
        table[spec.case_id] = face
    return table, tuple(unregistered)


CASE_FACES, UNREGISTERED = _build()
