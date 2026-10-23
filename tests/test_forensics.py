"""中转验真（forensics）回归：三条探针的回放、裁定口径、证据落盘与 Web 登记。

mock 一律用 ``httpx.MockTransport`` 回放真实形状的响应体（非流式 JSON），
断言的是报告与收尾真正会消费的字段：status / admission_passed / results / 证据行。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

from forensics.cases import (
    CASES,
    ERROR_ENVELOPE_LEAKAGE,
    FORENSICS_PROTOCOLS,
    IMPLICIT_IDENTITY,
    IMAGE_UNDERSTANDING,
)
from forensics.core import (
    ForensicsConfig,
    image_payload,
    probe_endpoint,
    probe_headers,
    response_text,
    run,
    scan_leakage,
    text_payload,
)
from volcperf.protocols import PROTOCOLS


# ---------------------------------------------------------------------------
# 夹具与登记
# ---------------------------------------------------------------------------


def test_case_metadata_and_protocol_whitelist():
    assert set(CASES) == {IMPLICIT_IDENTITY, IMAGE_UNDERSTANDING, ERROR_ENVELOPE_LEAKAGE}
    for case in CASES.values():
        assert case.protocols == ("openai", "volcengine", "anthropic", "gemini")
    assert CASES[IMPLICIT_IDENTITY].requests == 3
    assert CASES[IMAGE_UNDERSTANDING].requests == 5
    assert CASES[ERROR_ENVELOPE_LEAKAGE].requests == 3
    # 文字面协议必须放行 forensics；图像/视频面靠自己的白名单拦回。
    assert "forensics" in PROTOCOLS["openai"].test_kinds
    assert "forensics" in PROTOCOLS["anthropic"].test_kinds
    assert "forensics" in PROTOCOLS["gemini"].test_kinds
    assert "forensics" not in PROTOCOLS["openai_images"].test_kinds
    assert "forensics" not in PROTOCOLS["seedance"].test_kinds
    # 执行器协议闸：Responses 线格式未实现，不放行。
    assert "openai_responses" not in FORENSICS_PROTOCOLS


def test_web_registration_payload_tab_and_label():
    from web.server import forensics_case_payload

    for name in CASES:
        payload = forensics_case_payload(name)
        assert payload["test_kind"] == "forensics"
        assert payload["id"] == name
        assert payload["requests"] == CASES[name].requests
    static = Path(__file__).resolve().parent.parent / "web" / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    assert 'data-kind="forensics"' in html
    js = (static / "app.js").read_text(encoding="utf-8")
    assert "kind==='forensics'?" in js
    assert "'chat','forensics']" in js  # 刷新恢复白名单


def test_endpoints_headers_and_payloads_for_three_families():
    assert probe_endpoint("openai", "https://mux.la/v1", "m") == "https://mux.la/v1/chat/completions"
    assert probe_endpoint("anthropic", "https://mux.la/v1", "m") == "https://mux.la/v1/messages"
    assert probe_endpoint("gemini", "https://x.ai/v1beta", "g m") == "https://x.ai/v1beta/models/g%20m:generateContent"

    openai_headers = probe_headers("openai", "sk-x", "req-1")
    anthropic_headers = probe_headers("anthropic", "sk-x", "req-1")
    gemini_headers = probe_headers("gemini", "sk-x", "req-1")
    assert openai_headers["Authorization"] == "Bearer sk-x"
    assert anthropic_headers["x-api-key"] == "sk-x" and "anthropic-version" in anthropic_headers
    assert gemini_headers["x-goog-api-key"] == "sk-x"
    assert all(headers["Accept"] == "application/json" for headers in (openai_headers, anthropic_headers, gemini_headers))

    openai_body = text_payload("openai", "m", "hi", 512)
    assert openai_body == {"model": "m", "stream": False, "max_tokens": 512,
                           "messages": [{"role": "user", "content": "hi"}]}
    anthropic_body = text_payload("anthropic", "m", "hi", 512)
    assert "stream" not in anthropic_body and anthropic_body["max_tokens"] == 512
    gemini_body = text_payload("gemini", "m", "hi", 512)
    assert gemini_body["contents"][0]["parts"] == [{"text": "hi"}]
    assert gemini_body["generationConfig"] == {"maxOutputTokens": 512}

    png = b"\x89PNG-fake-bytes"
    image = image_payload("anthropic", "m", "看图", png, 1024)
    source = image["messages"][0]["content"][0]["source"]
    assert source["type"] == "base64" and source["media_type"] == "image/png" and source["data"]
    gemini_image = image_payload("gemini", "m", "看图", png, 1024)
    assert gemini_image["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "image/png"
    openai_image = image_payload("openai", "m", "看图", png, 1024)
    assert openai_image["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_scan_leakage_severity_levels():
    medium = scan_leakage('{"error":{"type":"new_api_error","message":"No available channel for model x '
                          'under group awsb-超稳 (distributor)"}}')
    assert {finding["pattern"] for finding in medium} == {"channel_topology", "relay_software"}
    assert all(finding["severity"] in {"medium", "low"} for finding in medium)
    high = scan_leakage('Traceback (most recent call last): File "/home/app/main.py", line 42, '
                        "psycopg duplicate key value, connect 10.0.0.7:5432")
    assert {finding["pattern"] for finding in high} <= {"stack_trace", "internal_path", "source_ref",
                                                        "sql_error", "private_network"}
    assert all(finding["severity"] == "high" for finding in high)
    assert scan_leakage('{"error":{"type":"not_found_error","message":"model not found"}}') == []


# ---------------------------------------------------------------------------
# mock 传输
# ---------------------------------------------------------------------------


def _config(**overrides) -> ForensicsConfig:
    base = {
        "api_key": "sk-test",
        "base_url": "https://mux.la/v1",
        "model": "claude-opus-4-8",
        "protocol": "openai",
        "max_retries": 0,
        "retry_backoff_seconds": 0.0,
        "record_id": "REC-1",
    }
    base.update(overrides)
    return ForensicsConfig(**base)


def _chat_content(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}], "model": "claude-opus-4-8"}


def _identity_client(votes: dict[str, str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        prompt = body["messages"][0]["content"]
        if "comment block" in prompt:
            return httpx.Response(200, json=_chat_content(votes["code"]))
        if "context window" in prompt:
            return httpx.Response(200, json=_chat_content(votes["context"]))
        return httpx.Response(200, json=_chat_content(votes["cutoff"]))

    return httpx.Client(transport=httpx.MockTransport(handler))


def _leakage_client(malformed=(400, '{"error":{"type":"invalid_request_error","message":"bad json"}}'),
                    missing=(400, '{"error":{"type":"not_found_error","message":"model not found"}}'),
                    wrong=(404, '{"error":{"type":"not_found_error","message":"not found"}}')) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/forensics-probe-" in url:
            status, text = wrong
        else:
            try:
                body = json.loads(request.read())
            except ValueError:
                status, text = malformed
            else:
                status, text = missing if str(body.get("model", "")).startswith("nonexistent-model-") else malformed
        return httpx.Response(status, text=text, headers={"content-type": "application/json"})

    return httpx.Client(transport=httpx.MockTransport(handler))


def _vision_client(answers: dict[str, str], status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, json={"error": {"type": "invalid_request_error",
                                                          "message": "image input not supported"}})
        body = json.loads(request.read())
        prompt = body["messages"][0]["content"][0]["text"]
        if "棋盘" in prompt:
            content = answers["chessboard"]
        elif "读出图片中的数字" in prompt:
            content = answers["ocr_digits"]
        elif "右半部分" in prompt:
            content = answers["spatial"]
        elif "背景底色" in prompt:
            content = answers["text_conflict"]
        else:
            content = answers["solid_color"]
        return httpx.Response(200, json=_chat_content(content))

    return httpx.Client(transport=httpx.MockTransport(handler))


def _status_by_step(outcome) -> dict[str, str]:
    return {row["step"]: row["status"] for row in outcome.summary["results"]}


# ---------------------------------------------------------------------------
# 用例一：隐式身份指纹
# ---------------------------------------------------------------------------


def test_identity_consistent_votes_pass(tmp_path: Path):
    client = _identity_client({
        "code": "```python\n# Tool Name : Date Diff\n# Generated : 2026-10-20\n"
                "# Model : Claude (Opus 4.8)\n# Author : Dev Team\n```\n",
        "context": "200000",
        "cutoff": "March 2025",
    })
    outcome = run(_config(), IMPLICIT_IDENTITY, tmp_path, client=client)
    statuses = _status_by_step(outcome)
    assert statuses == {"code_signature": "passed", "context_window": "passed", "knowledge_cutoff": "passed"}
    extras = outcome.summary["forensics"]
    assert extras["declared_family"] == "claude"
    assert extras["case_status"] == "passed" and outcome.summary["admission_passed"] is True
    assert outcome.summary["results"][0]["details"].startswith("代码头注释自述族 = claude")
    client.close()


def test_identity_single_contradiction_is_warning_and_not_admitted(tmp_path: Path):
    client = _identity_client({
        "code": "```python\n# Model : GPT-5 (OpenAI)\n# Author : OpenAI Team\n```\n",
        "context": "200000",
        "cutoff": "March 2025",
    })
    outcome = run(_config(), IMPLICIT_IDENTITY, tmp_path, client=client)
    statuses = _status_by_step(outcome)
    assert statuses["code_signature"] == "passed"  # 自述成功但不为声明族背书
    extras = outcome.summary["forensics"]
    assert extras["case_status"] == "warning" and extras["contradicting_votes"]
    assert outcome.summary["admission_passed"] is False
    assert outcome.summary["results"][0]["status"] == "passed"  # 子探针不压平成 failed
    client.close()


def test_identity_tier_mismatch_only_recorded(tmp_path: Path):
    """声明 opus、自述 sonnet：只留痕，不构成族矛盾（模型自述版本长期不可靠）。"""
    client = _identity_client({
        "code": "```python\n# Model : Claude (Sonnet 4.5)\n# Author : Anthropic\n```\n",
        "context": "200000",
        "cutoff": "January 2025",
    })
    outcome = run(_config(), IMPLICIT_IDENTITY, tmp_path, client=client)
    extras = outcome.summary["forensics"]
    assert extras["tier_self_report"]["mismatch"] is True
    assert extras["case_status"] == "passed" and outcome.summary["admission_passed"] is True
    assert "档位自述不符" in outcome.summary["assessment"]
    client.close()


def test_identity_stale_cutoff_is_warning(tmp_path: Path):
    client = _identity_client({
        "code": "```python\n# Model : Claude (Opus 4.8)\n# Author : Anthropic\n```\n",
        "context": "200000",
        "cutoff": "June 2023",
    })
    outcome = run(_config(), IMPLICIT_IDENTITY, tmp_path, client=client)
    statuses = _status_by_step(outcome)
    assert statuses["knowledge_cutoff"] == "warning"
    assert outcome.summary["admission_passed"] is False  # 任一子探针 warning 即下调准入
    client.close()


def test_identity_all_blocked_when_upstream_rejects(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"type": "authentication_error", "message": "bad key"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    outcome = run(_config(), IMPLICIT_IDENTITY, tmp_path, client=client)
    assert set(_status_by_step(outcome).values()) == {"blocked"}
    assert outcome.summary["forensics"]["case_status"] == "blocked"
    client.close()


# ---------------------------------------------------------------------------
# 用例二：图像内容理解
# ---------------------------------------------------------------------------


def test_vision_all_levels_correct(tmp_path: Path):
    client = _vision_client({"solid_color": "Yellow", "chessboard": "32",
                             "ocr_digits": "48271", "spatial": "红色", "text_conflict": "yellow"})
    outcome = run(_config(ocr_digits="48271"), IMAGE_UNDERSTANDING, tmp_path, client=client)
    statuses = _status_by_step(outcome)
    assert set(statuses.values()) == {"passed"}
    assert statuses["chessboard"] == "passed" and statuses["text_conflict"] == "passed"
    assert outcome.summary["forensics"]["accuracy"] == 1.0
    assert outcome.summary["admission_passed"] is True
    client.close()


def test_vision_partial_and_failure_thresholds(tmp_path: Path):
    partial = _vision_client({"solid_color": "Blue", "chessboard": "wrong",
                              "ocr_digits": "00000", "spatial": "red", "text_conflict": "yellow"})
    outcome = run(_config(ocr_digits="48271"), IMAGE_UNDERSTANDING, tmp_path, client=partial)
    assert outcome.summary["forensics"]["levels_correct"] == 2
    assert outcome.summary["forensics"]["case_status"] == "failed"  # 2/5 = 0.4 < 0.6
    assert outcome.summary["admission_passed"] is False
    partial.close()

    borderline = _vision_client({"solid_color": "Blue", "chessboard": "wrong",
                                 "ocr_digits": "48271", "spatial": "red", "text_conflict": "yellow"})
    outcome = run(_config(ocr_digits="48271"), IMAGE_UNDERSTANDING, tmp_path, client=borderline)
    assert outcome.summary["forensics"]["case_status"] == "warning"  # 3/5 = 0.6
    assert outcome.summary["admission_passed"] is True
    borderline.close()


def test_vision_unsupported_is_blocked_not_failed(tmp_path: Path):
    client = _vision_client({}, status=400)
    outcome = run(_config(), IMAGE_UNDERSTANDING, tmp_path, client=client)
    assert set(_status_by_step(outcome).values()) == {"blocked"}
    assert outcome.summary["forensics"]["case_status"] == "blocked"
    assert outcome.summary["admission_passed"] is True  # 平台前提不满足，不否决
    client.close()


# ---------------------------------------------------------------------------
# 用例三：错误信封泄露
# ---------------------------------------------------------------------------


def test_leakage_clean_envelopes_pass(tmp_path: Path):
    client = _leakage_client()
    outcome = run(_config(), ERROR_ENVELOPE_LEAKAGE, tmp_path, client=client)
    assert set(_status_by_step(outcome).values()) == {"passed"}
    assert outcome.summary["forensics"]["findings"] == []
    assert outcome.summary["admission_passed"] is True
    client.close()


def test_leakage_channel_topology_is_medium_warning(tmp_path: Path):
    client = _leakage_client(missing=(404, '{"error":{"type":"new_api_error","message":'
                                          '"No available channel for model x under group awsb-超稳 (distributor)"}}'))
    outcome = run(_config(), ERROR_ENVELOPE_LEAKAGE, tmp_path, client=client)
    statuses = _status_by_step(outcome)
    assert statuses["nonexistent_model"] == "warning"
    assert statuses["malformed_json"] == "passed" and statuses["wrong_path"] == "passed"
    assert outcome.summary["forensics"]["case_status"] == "warning"
    assert outcome.summary["admission_passed"] is True  # 中危只下调观感，不否决
    client.close()


def test_leakage_high_severity_fails(tmp_path: Path):
    client = _leakage_client(malformed=(500, "Traceback (most recent call last):\n"
                                            '  File "/home/app/main.py", line 7\n'))
    outcome = run(_config(), ERROR_ENVELOPE_LEAKAGE, tmp_path, client=client)
    statuses = _status_by_step(outcome)
    assert statuses["malformed_json"] == "blocked"  # 5xx 记 blocked，不判泄露
    client.close()

    client = _leakage_client(missing=(400, '{"error":{"message":"psycopg duplicate key value at '
                                           "/home/app/db.py:42 connect 10.0.0.7\"}}"))
    outcome = run(_config(), ERROR_ENVELOPE_LEAKAGE, tmp_path, client=client)
    assert _status_by_step(outcome)["nonexistent_model"] == "failed"
    assert outcome.summary["admission_passed"] is False
    client.close()


def test_leakage_silent_fallback_not_admitted(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/forensics-probe-" in url:
            return httpx.Response(404, text="not found")
        try:
            body = json.loads(request.read())
        except ValueError:
            return httpx.Response(400, json={"error": {"type": "invalid_request_error", "message": "bad"}})
        if str(body.get("model", "")).startswith("nonexistent-model-"):
            return httpx.Response(200, json=_chat_content("OK"))
        return httpx.Response(400, json={"error": {"type": "invalid_request_error", "message": "bad"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    outcome = run(_config(), ERROR_ENVELOPE_LEAKAGE, tmp_path, client=client)
    extras = outcome.summary["forensics"]
    assert extras["silent_fallback"] is True
    assert outcome.summary["admission_passed"] is False
    assert "静默放行" in outcome.summary["assessment"]
    client.close()


# ---------------------------------------------------------------------------
# 证据链与收尾
# ---------------------------------------------------------------------------


def test_evidence_chain_and_finalization(tmp_path: Path):
    from common.finalizer import finalize_result_dir

    client = _identity_client({
        "code": "```python\n# Model : Claude (Opus 4.8)\n# Author : Anthropic\n```\n",
        "context": "200000",
        "cutoff": "March 2025",
    })
    outcome = run(_config(), IMPLICIT_IDENTITY, tmp_path, client=client)
    evidence = (outcome.result_dir / "requests.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(evidence) == 3
    for line in evidence:
        row = json.loads(line)
        assert isinstance(row["http_status"], int) and row["http_status"] == 200
        assert row["timing"]["e2e_ms"] is not None
        assert "sk-test" not in line  # 鉴权头不落盘
    summary = finalize_result_dir(outcome.result_dir)
    assert summary["admission_passed"] is True
    assert (outcome.result_dir / "report.md").is_file()
    report = (outcome.result_dir / "report.md").read_text(encoding="utf-8")
    assert "隐式身份指纹" in report and "判定口径" in report
    client.close()


def test_gemini_response_text_extraction():
    body = {"candidates": [{"content": {"parts": [{"text": "你好"}, {"text": " world"}]}}]}
    assert response_text("gemini", body) == "你好 world"
    assert response_text("anthropic", {"content": [{"type": "text", "text": "a"}, {"type": "tool_use"}]}) == "a"
    assert response_text("openai", {"choices": [{"message": {"content": [
        {"type": "text", "text": "x"}, {"type": "text", "text": "y"}]}}]}) == "xy"
    assert response_text("openai", {"choices": []}) == ""
