"""对话窗口（chatpane）回归：四类线格式的回放、裁定口径与证据落盘。

mock 一律用 ``httpx.MockTransport`` 回放真实形状的响应体（含 SSE 逐事件与
usage），断言的是"拿到完整正文"这条主线上真正会被下游消费的字段。
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx

from chatpane.cases import CASES, CHAT_PROMPT
from chatpane.core import ChatConfig, build_payload, chat_once, run_chat
from chatpane.runner import config_from_env
from common.finalizer import finalize_result_dir


def _sse(events, *, done_marker=True):
    body = "".join(f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events)
    if done_marker:
        body += "data: [DONE]\n\n"
    return body.encode()


def _client(handler, **kwargs):
    return httpx.Client(transport=httpx.MockTransport(handler), **kwargs)


def _stream_response(body: bytes, status: int = 200, **headers) -> httpx.Response:
    return httpx.Response(
        status,
        headers={"content-type": "text/event-stream", **headers},
        content=body,
    )


def _config(**overrides) -> ChatConfig:
    base = {
        "api_key": "sk-test",
        "base_url": "https://mux.la/v1",
        "model": "some-model",
        "protocol": "openai",
        "prompt": "画一个鹈鹕骑自行车",
        "max_retries": 0,
        "retry_backoff_seconds": 0.0,
    }
    base.update(overrides)
    return ChatConfig(**base)


def test_case_declares_text_families_only():
    case = CASES[CHAT_PROMPT]
    # 家族白名单：openai 家族覆盖 openai/openai_responses/qianfan；volcengine 是独立家族。
    # 图像面（openai_images / gemini_images）靠协议自身的 test_kinds 白名单拦回。
    assert case.protocols == ("openai", "volcengine", "anthropic", "gemini")
    assert case.case_id == CHAT_PROMPT


def test_payloads_never_inject_the_benchmark_system_prompt():
    """对话窗口必须下发用户输入的系统提示词，而不是压测用的固定提示词。"""
    openai_body = build_payload(_config(system="你是主厨", max_tokens=64))
    assert openai_body["messages"][0] == {"role": "system", "content": "你是主厨"}
    assert openai_body["messages"][1]["content"] == "画一个鹈鹕骑自行车"
    assert openai_body["stream"] is True and openai_body["max_tokens"] == 64
    assert openai_body["stream_options"] == {"include_usage": True}
    assert all("performance-test assistant" not in str(message) for message in openai_body["messages"])

    responses = build_payload(_config(protocol="openai_responses", system="你是主厨"))
    assert responses["instructions"] == "你是主厨"
    assert responses["input"] == [{"role": "user", "content": "画一个鹈鹕骑自行车"}]
    assert "max_output_tokens" in responses and "max_tokens" not in responses

    anthropic = build_payload(_config(protocol="anthropic", system="你是主厨", temperature=0.3))
    assert anthropic["system"] == "你是主厨"
    assert anthropic["messages"] == [{"role": "user", "content": "画一个鹈鹕骑自行车"}]
    assert anthropic["temperature"] == 0.3

    gemini = build_payload(_config(protocol="gemini", system="你是主厨"))
    assert gemini["systemInstruction"] == {"parts": [{"text": "你是主厨"}]}
    assert gemini["contents"] == [{"role": "user", "parts": [{"text": "画一个鹈鹕骑自行车"}]}]
    # 不传温度时不得凭空补一个默认值（"是否下发"本身是被观测行为）。
    assert "temperature" not in gemini["generationConfig"]


def test_openai_chat_stream_is_accumulated_verbatim():
    events = [
        {"choices": [{"delta": {"content": "<svg xmlns="}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "\"http://www.w3.org/2000/svg\">"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 12, "completion_tokens": 7}},
    ]
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return _stream_response(_sse(events), **{"x-request-id": "req-1"})

    with _client(handler) as client:
        row = chat_once(_config(), client=client)
    assert row["verdict"] == "passed" and row["success"] is True
    assert row["response_text"] == '<svg xmlns="http://www.w3.org/2000/svg">'
    assert row["finish_reason"] == "stop"
    assert row["usage"] == {"prompt_tokens": 12, "completion_tokens": 7,
                            "reasoning_tokens": None, "total_tokens": 19}
    assert row["timing"]["ttfc_ms"] is not None and row["timing"]["e2e_ms"] is not None
    assert seen["url"] == "https://mux.la/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"


def test_responses_api_typed_stream_is_accumulated():
    events = [
        {"type": "response.created", "response": {"id": "resp_1"}},
        {"type": "response.output_text.delta", "delta": "hello "},
        {"type": "response.output_text.delta", "delta": "world"},
        {"type": "response.completed", "response": {"usage": {"input_tokens": 3, "output_tokens": 2}}},
    ]

    def handler(request):
        assert str(request.url) == "https://mux.la/v1/responses"
        return _stream_response(_sse(events, done_marker=False))

    with _client(handler) as client:
        row = chat_once(_config(protocol="openai_responses"), client=client)
    assert row["response_text"] == "hello world"
    assert row["finish_reason"] == "stop"
    assert row["usage"]["total_tokens"] == 5
    assert row["saw_done"] is True


def test_anthropic_stream_separates_text_and_thinking():
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 9}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "先想"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "答案"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}},
        {"type": "message_stop"},
    ]

    def handler(request):
        assert str(request.url) == "https://i-token.ai/v1/messages"
        assert request.headers.get("x-api-key") == "sk-test"
        return _stream_response(_sse(events))

    with _client(handler) as client:
        row = chat_once(_config(protocol="anthropic", base_url="https://i-token.ai/v1"), client=client)
    assert row["response_text"] == "答案"
    assert row["reasoning_text"] == "先想"
    assert row["usage"]["prompt_tokens"] == 9
    assert row["usage"]["completion_tokens"] == 4
    assert row["verdict"] == "passed"


def test_gemini_stream_ends_by_eof_without_done_marker():
    events = [
        {"candidates": [{"content": {"parts": [{"text": "鹈"}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "鹕"}]}, "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 6, "candidatesTokenCount": 2, "totalTokenCount": 8}},
    ]

    def handler(request):
        assert ":streamGenerateContent" in str(request.url)
        assert request.headers.get("x-goog-api-key") == "sk-test"
        return _stream_response(_sse(events, done_marker=False))

    with _client(handler) as client:
        row = chat_once(_config(protocol="gemini", base_url="https://mux.la/v1beta",
                                model="gemini-3.8-flash"), client=client)
    assert row["response_text"] == "鹈鹕"
    assert row["verdict"] == "passed"
    # Gemini 的 SSE 以 EOF 收尾：拿到 finishReason 即视为收尾，不应记成 missing done。
    assert row["saw_done"] is True
    assert row["usage"]["total_tokens"] == 8


def test_empty_response_is_warning_and_does_not_admit():
    def handler(request):
        return _stream_response(_sse([{"choices": [{"delta": {}, "finish_reason": "stop"}]}]))

    with _client(handler) as client:
        with tempfile.TemporaryDirectory() as directory:
            outcome = run_chat(_config(), Path(directory), client=client)
            summary = finalize_result_dir(outcome.result_dir)
    assert outcome.row["verdict"] == "warning"
    assert outcome.row["error_type"] == "empty_response"
    assert summary["admission_passed"] is False
    assert summary["results"][0]["status"] == "warning"


def test_upstream_rejections_are_classified_not_flattened():
    cases = [(400, "failed"), (401, "blocked"), (403, "blocked"), (404, "blocked"), (429, "blocked")]

    def make(status):
        def handler(request):
            return httpx.Response(status, json={"error": {"message": "nope"}})
        return handler

    for status, expected in cases:
        with _client(make(status)) as client:
            row = chat_once(_config(), client=client)
        assert row["verdict"] == expected, (status, row["verdict"])
        assert row["attempts"] == 1, f"4xx 不应重试：{status}"
        assert row["error_type"] == f"http_{status}"
        assert row["success"] is False


def test_transient_5xx_is_retried_before_blocked():
    calls = {"count": 0}

    def handler(request):
        calls["count"] += 1
        if calls["count"] < 3:
            return httpx.Response(503, json={"error": "overloaded"})
        return _stream_response(_sse([{"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}]))

    with _client(handler) as client:
        row = chat_once(_config(max_retries=2), client=client)
    assert calls["count"] == 3
    assert row["verdict"] == "passed" and row["response_text"] == "ok"

    def always_503(request):
        return httpx.Response(503, json={"error": "overloaded"})

    with _client(always_503) as client:
        failed = chat_once(_config(max_retries=2), client=client)
    assert failed["attempts"] == 3
    # 瞬时故障按 blocked 留痕，不写成模型问题。
    assert failed["verdict"] == "blocked"


def test_result_directory_keeps_evidence_and_raw_response():
    payload = 'data: ' + json.dumps({"choices": [{"delta": {"content": "<svg/>"}, "finish_reason": "stop"}]}) + "\n\n"

    def handler(request):
        return _stream_response(payload.encode())

    with _client(handler) as client:
        with tempfile.TemporaryDirectory() as directory:
            outcome = run_chat(_config(), Path(directory), client=client)
            result_dir = outcome.result_dir
            requests_path = result_dir / "requests.jsonl"
            assert requests_path.is_file() and requests_path.stat().st_size > 0
            assert (result_dir / "response.md").read_text(encoding="utf-8") == "<svg/>"
            # 证据里不得出现 API Key 原文。
            assert "sk-test" not in requests_path.read_text(encoding="utf-8")
            row = json.loads(requests_path.read_text(encoding="utf-8").strip())
            assert row["case"] == CHAT_PROMPT and row["prompt"] == "画一个鹈鹕骑自行车"
            report = (result_dir / "report.md").read_text(encoding="utf-8")
            assert "## 响应内容" in report and "<svg/>" in report
            summary = finalize_result_dir(result_dir)
    assert summary["admission_passed"] is True
    assert summary["test_kind"] == "chat"


def test_env_config_normalizes_api_root_and_guards_input():
    env = {
        "CHAT_API_KEY": "sk-x",
        "CHAT_BASE_URL": "https://mux.la",
        "CHAT_MODEL": "gemini-3.8-flash",
        "CHAT_PROTOCOL": "gemini",
        "CHAT_PROMPT": "你好",
        "CHAT_MAX_TOKENS": "2048",
    }
    with patch.dict(os.environ, env, clear=False):
        config = config_from_env()
    assert config.base_url == "https://mux.la/v1beta"
    assert config.endpoint == "https://mux.la/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse"
    assert config.max_tokens == 2048 and config.temperature is None

    # 空消息、越界输出上限与不支持的面都必须拦在发请求之前。
    for broken in ({"CHAT_PROMPT": "   "}, {"CHAT_MAX_TOKENS": "0"}, {"CHAT_PROTOCOL": "openai_images"}):
        with patch.dict(os.environ, {**env, **broken}, clear=False):
            try:
                config_from_env()
            except ValueError:
                continue
            raise AssertionError(f"未拦回非法配置: {broken}")
