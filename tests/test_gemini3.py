"""gemini3（Gemini 3.x 文本专项）用例矩阵、判定助手与 mock 上游端到端验证。

与 ``tests/test_fable5.py`` / ``tests/test_gpt56.py`` 同源：静态部分锁定用例集与档位矩阵，
动态部分用 ``httpx.MockTransport`` 造一个"忠实 Gemini 3.x 服务端"，跑完整套件验证
判定链（而不是只测单个 helper）。
"""
from __future__ import annotations

import asyncio
import json
import re
import tempfile
from collections import Counter
from pathlib import Path

import httpx
import pytest

from gemini3.cases import CASES as GEMINI3_SUITES
from gemini3.core import (
    CASE_SPECS,
    CONTENT_MIN_OUTPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MODEL_TIERS,
    MODELS,
    VARIANT_FALLBACK_TIER,
    Gemini3Config,
    Gemini3Runner,
    _declared_thinking_levels,
    _foreign_protocol_evidence,
    _injected_system_reason,
    _is_rejection,
    _LEVEL_UNIVERSE_PROBE_VALUE,
    _model_family_tokens,
    _relay_fields,
    _same_model_name,
    _text_refusal_marker,
    normalize_model_name,
    summarize_results,
    tier_for_model,
)
from gemini3.runner import _FOREIGN_PREFIXES, config_from_env, guard_model, read_finalized_summary
from volcperf.protocols import api_root, protocol_spec, supports_test_kind

KNOWN_MODELS = ("gemini-3.8-flash", "gemini-3.1-pro-preview")


# ---------------------------------------------------------------------------
# 用例矩阵
# ---------------------------------------------------------------------------
def test_suite_matrix_matches_design():
    """5 个套件、68 条用例；套件的「标称请求数」是展示用估算，必须不小于用例条数。"""
    assert sorted(GEMINI3_SUITES) == [
        "GEMINI3_BASIC", "GEMINI3_FINGERPRINT", "GEMINI3_P0", "GEMINI3_P1", "GEMINI3_P2",
    ]
    per_priority = Counter(spec.priority for spec in CASE_SPECS)
    assert per_priority == {"BASIC": 10, "P0": 20, "P1": 10, "P2": 16, "AUTH": 12}
    assert len(CASE_SPECS) == 68
    seen: set[str] = set()
    for name, case in GEMINI3_SUITES.items():
        assert case.priorities in (("BASIC",), ("P0",), ("P1",), ("P2",), ("AUTH",)), name
        # 一个套件只覆盖一个优先级档，且该档用例全部落在本套件里（不漏不重）。
        assert case.request_count >= per_priority[case.priorities[0]], name
        assert not (seen & set(case.priorities))
        seen |= set(case.priorities)
    assert seen == set(per_priority)
    # 两个模型共用同一批用例：只有 gemini3 一个档位。
    assert {case.model_variant for case in GEMINI3_SUITES.values()} == {"gemini3"}
    assert {case.protocols for case in GEMINI3_SUITES.values()} == {("gemini",)}


def test_every_spec_has_handler_and_unique_ids():
    """每条 CaseSpec 必须有同名 handler，且 id 不重复——防"改用例编号后忘改 handler"的静默遗漏。"""
    ids = [spec.case_id for spec in CASE_SPECS]
    assert len(ids) == len(set(ids))
    priority_counts: dict[str, int] = {}
    for spec in CASE_SPECS:
        priority_counts[spec.priority] = priority_counts.get(spec.priority, 0) + 1
        assert hasattr(Gemini3Runner, f"case_{spec.handler}"), f"{spec.case_id} 缺 handler {spec.handler}"
        assert spec.variant == "gemini3"
    assert priority_counts == {"BASIC": 10, "P0": 20, "P1": 10, "P2": 16, "AUTH": 12}


# ---------------------------------------------------------------------------
# 档位能力矩阵
# ---------------------------------------------------------------------------
def test_tier_matrix_contract():
    assert MODELS == KNOWN_MODELS
    assert len(MODEL_TIERS) == 2
    flash, pro = MODEL_TIERS
    assert (flash.model_id, pro.model_id) == KNOWN_MODELS
    for tier in MODEL_TIERS:
        # 3.8 Flash / 3.1 Pro 官方档位集一致：low/medium/high，minimal 已移除。
        assert tier.thinking_levels == ("low", "medium", "high")
        assert tier.rejected_levels == ("minimal",)
        assert tier.thinking_disableable is False
        assert tier.sampling_ignored_params == ("temperature", "topP", "topK")
    # 默认档位与"传即报错"参数按模型分区（这就是"不照搬 Claude/GPT"的地方）。
    assert flash.default_level == "medium"
    assert pro.default_level == "high"
    assert flash.unsupported_params == ("frequency_penalty", "presence_penalty", "candidate_count")
    assert flash.candidate_count_supported is False
    assert pro.unsupported_params == ()
    assert pro.candidate_count_supported is True
    assert flash.accepted_tokens == ("gemini-3.8-flash",)
    assert pro.accepted_tokens == ("gemini-3.1-pro-preview", "gemini-3.1-pro")


@pytest.mark.parametrize("raw,expected", [
    ("gemini-3.8-flash", "gemini-3.8-flash"),
    ("models/gemini-3.8-flash", "gemini-3.8-flash"),
    ("publishers/google/models/gemini-3.8-flash", "gemini-3.8-flash"),
    ("gemini-3.8-flash-001", "gemini-3.8-flash"),
    ("gemini-3.8-flash-preview", "gemini-3.8-flash"),
    ("gemini-3.8-flash-2026-09-02", "gemini-3.8-flash"),
    ("gemini-3.1-pro-preview", "gemini-3.1-pro-preview"),
    ("gemini-3.1-pro", "gemini-3.1-pro-preview"),          # 官方别名
    ("models/gemini-3.1-pro", "gemini-3.1-pro-preview"),
])
def test_tier_for_model_accepts_official_forms(raw, expected):
    tier = tier_for_model(raw)
    assert tier is not None and tier.model_id == expected


@pytest.mark.parametrize("raw", [
    "", "   ", "gemini-3.8-flash-lite", "gemini-3.8-flash-cyber", "gemini-3.7-flash",
    "gpt-5.6", "claude-fable-5", "kimi-k3", "deepseek-v4", "glm-5.3",
])
def test_tier_for_model_is_strict(raw):
    """不做宽松包含：同族不同型号 / 别家模型必须落到 None（否则"模型 ID 严格解析"被架空）。"""
    assert tier_for_model(raw) is None


def test_model_name_normalization_is_symmetric():
    assert normalize_model_name("models/Gemini-3.1-Pro-Preview") == "gemini-3.1-pro"
    assert _same_model_name("gemini-3.8-flash", "gemini-3.8-flash-001")
    assert _same_model_name("gemini-3.1-pro-preview", "gemini-3.1-pro")
    # 身份断言相对被测模型：请求 3.1 Pro 而响应 3.8 Flash 必须判为"被替换"。
    assert not _same_model_name("gemini-3.8-flash", "gemini-3.1-pro-preview")
    assert _model_family_tokens("gemini-3.1-pro-preview")[0] == "gemini-3.1-pro-preview"
    assert _model_family_tokens("") == MODELS


def test_variant_fallback_tier_is_flagged():
    """套件档位回落值存在，且回落时必须以 assumed=True 走 warning 分支。"""
    assert set(VARIANT_FALLBACK_TIER) == {"gemini3"}
    assert VARIANT_FALLBACK_TIER["gemini3"].model_id == "gemini-3.8-flash"
    runner = Gemini3Runner(Gemini3Config(api_key="k"), audit_dir=None)
    tier, assumed = runner._tier("gateway-custom-id")
    assert (tier.model_id, assumed) == ("gemini-3.8-flash", True)
    tier, assumed = runner._tier("gemini-3.1-pro-preview")
    assert (tier.model_id, assumed) == ("gemini-3.1-pro-preview", False)


def test_output_budget_floor_allows_visible_content():
    """Gemini 3.x 思考无法关闭：需要正文的用例输出上限不得低于 2048，否则必然零正文假失败。"""
    assert CONTENT_MIN_OUTPUT_TOKENS == 2048
    assert MAX_OUTPUT_TOKENS == 65536
    runner = Gemini3Runner(Gemini3Config(api_key="k"))
    captured: dict[str, object] = {}

    async def fake_request(path, payload, **kwargs):
        captured.update({"path": path, "payload": payload})
        return _Ready()

    class _Ready:
        status = 200

    runner.request = fake_request  # type: ignore[assignment]
    asyncio.run(runner.generate("gemini-3.8-flash", Gemini3Runner._user("hi")))
    assert captured["payload"]["generationConfig"]["maxOutputTokens"] == CONTENT_MIN_OUTPUT_TOKENS
    assert captured["path"] == "models/gemini-3.8-flash:generateContent"


def test_generate_stream_path_matches_native_sse_endpoint():
    runner = Gemini3Runner(Gemini3Config(api_key="k"))
    captured: dict[str, object] = {}

    async def fake_request(path, payload, **kwargs):
        captured.update({"path": path, "stream": kwargs.get("stream")})
        return object()

    runner.request = fake_request  # type: ignore[assignment]
    asyncio.run(runner.generate("gemini-3.1-pro-preview", Gemini3Runner._user("hi"), stream=True))
    assert captured["path"] == "models/gemini-3.1-pro-preview:streamGenerateContent?alt=sse"
    assert captured["stream"] is True


# ---------------------------------------------------------------------------
# 判定助手
# ---------------------------------------------------------------------------
def test_rejection_excludes_404():
    """404 = 端点/路由问题，不能当"参数被正确拒绝"（唯一例外由未知模型用例自行处理）。"""
    assert _is_rejection(400) and _is_rejection(422) and _is_rejection(403)
    assert not _is_rejection(404)
    assert not _is_rejection(200) and not _is_rejection(500)


def test_declared_thinking_levels_parsing():
    body = {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                      "message": "Invalid thinkingLevel. Supported values: 'minimal', 'low', 'medium', 'high'"}}
    assert _declared_thinking_levels(body) == ("minimal", "low", "medium", "high")
    # 无 "supported values" 段 ⇒ 不得凭空放宽任何判据。
    assert _declared_thinking_levels({"error": {"code": 400, "message": "invalid"}}) is None
    # 大写带引号形态走兜底正则。
    assert _declared_thinking_levels(
        {"error": {"message": "Supported values: LOW, MEDIUM, HIGH"}}) == ("low", "medium", "high")


def test_text_refusal_marker_ignores_late_apology():
    assert _text_refusal_marker("我不能提供入侵他人网络的方法。")
    assert _text_refusal_marker("I cannot help with that request.")
    # 致歉弱信号只认开头窗口：后文道歉不算拒答。
    assert _text_refusal_marker("先给步骤。" + "x" * 300 + "I'm sorry, but I") is None


def test_injection_and_foreign_protocol_detection():
    injected = {"systemInstruction": {"parts": [{"text": "You are a helpful assistant." * 2}]}}
    assert "注入了系统提示词" in (_injected_system_reason(injected) or "")
    assert _injected_system_reason({"systemInstruction": {"parts": [{"text": "ok"}]}}) is None
    assert _foreign_protocol_evidence({"choices": [{"message": {}}]}) is not None
    assert _foreign_protocol_evidence({"content": [{"type": "text"}], "usage": {}}) is not None
    assert _foreign_protocol_evidence({"candidates": [], "usageMetadata": {}}) is None


def test_relay_fields_flags_non_native_response_keys():
    native = {"candidates": [], "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2}}
    assert _relay_fields(native) == []
    relayed = {"candidates": [], "billing_usage": {"source": "oai_responses"},
               "usageMetadata": {"promptTokenCount": 1, "serviceTier": "default"}}
    assert _relay_fields(relayed) == ["billing_usage"]


# ---------------------------------------------------------------------------
# 配置与跨套件拦回
# ---------------------------------------------------------------------------
def test_config_from_env_builds_v1beta_root(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key-1")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://mux.la")
    monkeypatch.setenv("GEMINI_PROTOCOL", "gemini")
    monkeypatch.delenv("GEMINI_TRUST_ENV", raising=False)
    config = config_from_env()
    assert config.base_url == "https://mux.la/v1beta"
    assert config.api_key == "key-1"
    assert config.trust_env is False
    assert api_root("gemini", "https://mux.la/v1beta") == "https://mux.la/v1beta"
    assert api_root("gemini", "https://mux.la/v1beta/custom") == "https://mux.la/v1beta/custom"


def test_config_from_env_requires_credentials(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_BASE_URL", "https://mux.la")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        config_from_env()


def test_guard_model_blocks_foreign_families():
    assert guard_model("gemini-3.8-flash") is None
    assert guard_model("") is None
    for prefix in _FOREIGN_PREFIXES:
        assert guard_model(f"{prefix}-whatever") is not None


def test_protocol_registration_lets_gemini_kind_through():
    spec = protocol_spec("gemini")
    assert spec.family == "gemini"
    assert "gemini" in spec.test_kinds
    assert supports_test_kind("gemini", "gemini", ("gemini",))
    # 图像套件（gemini-image）与文本套件互不串面。
    assert not supports_test_kind("gemini", "gemini-image", ("gemini",))
    assert not supports_test_kind("gemini_images", "gemini", ("gemini",))


def test_read_finalized_summary_reads_disk_truth(tmp_path: Path):
    """退出码必须用收尾后的磁盘结论：finalize 可能把 admission_passed 从 True 下调。"""
    payload = {"admission_passed": False, "total": 3, "passed": 0}
    (tmp_path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    assert read_finalized_summary(tmp_path)["admission_passed"] is False


# ---------------------------------------------------------------------------
# 流式合并：Gemini SSE 每帧是 GenerateContentResponse，且没有 [DONE]
# ---------------------------------------------------------------------------
class _StreamRecord:
    def __init__(self, frames, status=200, body=""):
        self.http_status = status
        self.raw_sse_events = [json.dumps(frame, ensure_ascii=False) for frame in frames]
        self.parsed_sse_events = list(frames)
        self.invalid_sse_events = 0
        self.done_received = False
        self.raw_response_body = body
        self.response_text = body


def test_merge_stream_combines_frames_without_done():
    frames = [
        {"candidates": [{"content": {"parts": [{"text": "你"}]}}],
         "usageMetadata": {"promptTokenCount": 11}},
        {"candidates": [{"content": {"parts": [{"text": "好", "thought": True}]}, "finishReason": "STOP"}],
         "usageMetadata": {"candidatesTokenCount": 4, "thoughtsTokenCount": 9, "totalTokenCount": 24},
         "modelVersion": "gemini-3.8-flash"},
    ]
    merged, info = Gemini3Runner._merge_stream(_StreamRecord(frames))
    assert merged is not None
    assert info["frames"] == 2
    assert info["done"] is False              # Gemini 流没有 [DONE]
    assert info["text"] == "你"
    assert info["reasoning"] == "好"
    assert info["finish_reason"] == "STOP"
    assert merged["usageMetadata"]["promptTokenCount"] == 11
    assert merged["usageMetadata"]["candidatesTokenCount"] == 4
    assert merged["usageMetadata"]["thoughtsTokenCount"] == 9
    assert merged["modelVersion"] == "gemini-3.8-flash"


def test_merge_stream_parses_error_body_for_negative_helpers():
    """HTTP>=300 时 SSE 通道里是整段 JSON 错误体，负向助手要照常工作。"""
    body = json.dumps({"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                 "message": "Invalid thinkingLevel. Supported values: 'low', 'medium', 'high'"}})
    merged, info = Gemini3Runner._merge_stream(_StreamRecord([], status=400, body=body))
    assert merged is not None and merged["error"]["code"] == 400
    assert info["frames"] == 0
    assert _declared_thinking_levels(merged) == ("low", "medium", "high")


# ---------------------------------------------------------------------------
# mock 上游：一个"忠实 Gemini 3.x 服务端"
# ---------------------------------------------------------------------------
def _parts_text(body: dict) -> str:
    chunks = []
    for content in body.get("contents") or []:
        if isinstance(content, dict):
            for part in content.get("parts") or []:
                if isinstance(part, dict) and part.get("text"):
                    chunks.append(str(part["text"]))
    return "\n".join(chunks)


def _system_text(body: dict) -> str:
    system = body.get("systemInstruction") or {}
    return "".join(str(p.get("text") or "") for p in system.get("parts") or [] if isinstance(p, dict))


def _count_tokens(payload: dict) -> int:
    return max(1, len(json.dumps(payload, ensure_ascii=False)) // 4)


CODE_REPLY = (
    "```python\n"
    "def count_words(text):\n"
    "    result = {}\n"
    "    if not text:\n"
    "        return result\n"
    "    for word in text.lower().split():\n"
    "        result[word] = result.get(word, 0) + 1\n"
    "    return result\n"
    "```"
)
LONGFORM_REPLY = "\n".join([
    "- 先为每个关键路径写一个会失败的用例再补实现",
    "- 用例名称描述行为而不是实现细节便于重构",
    "- 每个缺陷修复都必须附一条能复现的回归用例",
    "- 测试数据用工厂函数构造避免跨用例耦合",
    "- 慢用例与快用例分层执行保持反馈速度",
])
REFUSAL_REPLY = "我不能提供入侵他人网络的方法，这属于违法行为，建议通过合法途径向运营商反馈。"


class FakeGemini:
    """只实现本套件用到的路由，按请求体内容决定应答。

    ``declared_levels`` 控制网关在 4xx 文案里声明的合法档位集，用来区分
    "网关统一白名单（→ blocked）"与"确属后端档位契约不符（→ failed）"。
    """

    def __init__(self, *, accept_minimal: bool = True,
                 declared_levels=("minimal", "low", "medium", "high")):
        self.accept_minimal = accept_minimal
        self.declared_levels = tuple(declared_levels)
        self.seen: list[tuple[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.seen.append((request.method, path))
        key = request.headers.get("x-goog-api-key") or ""
        if "invalid" in key.lower():
            return _error(401, "API key not valid. Please pass a valid API key.", "UNAUTHENTICATED")
        if request.method == "GET":
            return httpx.Response(200, json={"models": [{"name": f"models/{m}"} for m in KNOWN_MODELS]})
        body = json.loads(request.read() or b"{}")
        matched = re.search(r"/models/(.+?)(?::|$)", path)
        model = matched.group(1) if matched else ""
        if ":countTokens" in path:
            return httpx.Response(200, json={"totalTokens": _count_tokens(body)})
        for content in body.get("contents") or []:
            if isinstance(content, dict) and content.get("role") not in {"user", "model", "function"}:
                return _error(400, f"Invalid role: {content.get('role')}", "INVALID_ARGUMENT")
        if "nonexistent" in model:
            return _error(404, f"models/{model} is not found for API version v1beta", "NOT_FOUND")
        config = body.get("generationConfig") or {}
        thinking = config.get("thinkingConfig") or {}
        level = thinking.get("thinkingLevel")
        if level == _LEVEL_UNIVERSE_PROBE_VALUE:
            return _error(400, _level_message(self.declared_levels), "INVALID_ARGUMENT")
        if level == "minimal" and not self.accept_minimal:
            return _error(400, _level_message(("low", "medium", "high")), "INVALID_ARGUMENT")
        text, thoughts = self._answer(body, model, level)
        # promptTokenCount 与 :countTokens 走同一个换算函数，两个端点的量纲才可比——
        # 否则 G3-C10（交叉验证）会因为造数据不一致而永远落到 warning。
        prompt_tokens = _count_tokens({"contents": body.get("contents")})
        return httpx.Response(200, json=_generate_body(model, text, thoughts=thoughts,
                                                       prompt_tokens=prompt_tokens))

    @staticmethod
    def _answer(body: dict, model: str, level) -> tuple[str, int]:
        prompt, system = _parts_text(body), _system_text(body)
        thoughts = 64 if (level or "high") in {"low", "medium", "high"} and level else 0
        if "[G3]" in system:
            return "[G3] 已确认规则。", thoughts
        token = re.search(r"EXACT-[0-9A-F]{8}", prompt)
        if token:
            return token.group(0), thoughts
        if "4387 × 2963" in prompt:
            return "12998681", thoughts
        if "星港实验室" in prompt:
            return "检索模块可以上线。", thoughts
        if "count_words" in prompt:
            return CODE_REPLY, thoughts
        if "单元测试的建议" in prompt:
            return LONGFORM_REPLY, thoughts
        if (body.get("generationConfig") or {}).get("responseMimeType") == "application/json":
            return json.dumps({"city": "西安", "score": 42}, ensure_ascii=False), thoughts
        if "澳大利亚的首都" in prompt:
            return "堪培拉", thoughts
        if "质数" in prompt:
            return "2, 3, 5, 7, 11, 13, 17, 19", thoughts
        if "入侵邻居家的无线网络" in prompt:
            return REFUSAL_REPLY, thoughts
        return "OK", thoughts


def _level_message(levels) -> str:
    quoted = ", ".join(f"'{level}'" for level in levels)
    return f"Invalid thinkingLevel value. Supported values: {quoted}"


def _error(status: int, message: str, google_status: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": status, "message": message, "status": google_status}})


def _generate_body(model: str, text: str, *, thoughts: int = 0, prompt_tokens: int = 24) -> dict:
    usage = {"promptTokenCount": prompt_tokens, "candidatesTokenCount": max(1, len(text) // 3), "totalTokenCount": 0}
    if thoughts:
        usage["thoughtsTokenCount"] = thoughts
    usage["totalTokenCount"] = usage["promptTokenCount"] + usage["candidatesTokenCount"] + thoughts
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
            "usageMetadata": usage, "modelVersion": model, "responseId": "resp-fake-1"}


def run_suite(fake: FakeGemini, *, model: str = "gemini-3.8-flash",
              case_ids=None, priorities=("BASIC",), variant: str = "gemini3"):
    """在 mock 上游上跑指定用例，返回 (results, audit_dir)。"""
    def _go(tmp: Path):
        async def scenario():
            async with httpx.AsyncClient(transport=httpx.MockTransport(fake)) as client:
                runner = Gemini3Runner(Gemini3Config(api_key="test-key"), client=client, audit_dir=tmp)
                async with runner:
                    # 单测覆盖的是「判定」，不是「执行范围」：显式跑全部面，
                    # 否则通道面用例（内置工具/契约）会被默认面过滤掉。
                    return await runner.run((model,), priorities, case_ids=case_ids,
                                            variant=variant, faces=("model", "verify", "channel"))
        return asyncio.run(scenario())

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        results = _go(tmp)
        evidence = (tmp / "requests.jsonl")
        payload = evidence.read_text(encoding="utf-8") if evidence.is_file() else ""
    return results, payload


BASIC_IDS = {spec.case_id for spec in CASE_SPECS if spec.priority == "BASIC"}


def test_basic_suite_passes_end_to_end_on_faithful_upstream():
    """整套 BASIC 在忠实上游上必须全绿——覆盖识别、执行、结构、多轮、安全拒答各判据。"""
    fake = FakeGemini()
    results, evidence = run_suite(fake)
    statuses = {r.case_id: r.status for r in results}
    assert len(results) == 10
    assert statuses == {case_id: "passed" for case_id in sorted(BASIC_IDS)}, \
        {r.case_id: (r.status, r.details) for r in results if r.status != "passed"}
    # 证据链：每个请求都落进 requests.jsonl。
    assert evidence.strip()
    assert all(r.passed for r in results)
    summary = summarize_results(results)
    assert summary["all_passed"] is True and summary["pass_rate"] == 1.0
    # 走原生 generateContent 面，且请求路径与官方契约一致。
    assert any(path.endswith(":generateContent") for _, path in fake.seen)
    assert all(path.startswith("/v1beta/models/") for _, path in fake.seen)


def test_contract_and_auth_cases_pass_on_faithful_upstream():
    """P2 契约/负向用例：错误对象结构、未知模型、鉴权、/models、:countTokens。"""
    fake = FakeGemini()
    results, _ = run_suite(
        fake, priorities=("P2",),
        case_ids={"G3-C01", "G3-C02", "G3-C08", "G3-C09", "G3-C10"},
    )
    assert {r.case_id: r.status for r in results} == {
        "G3-C01": "passed", "G3-C02": "passed", "G3-C08": "passed",
        "G3-C09": "passed", "G3-C10": "passed",
    }, {r.case_id: (r.status, r.details) for r in results}


def test_thinking_level_floor_is_blocked_when_gateway_uses_unified_whitelist():
    """归因纪律：网关自己声明支持 minimal（统一白名单）⇒ 记 blocked，不得读成"后端不是该模型"。"""
    fake = FakeGemini(accept_minimal=True, declared_levels=("minimal", "low", "medium", "high"))
    results, _ = run_suite(fake, priorities=("P0",), case_ids={"G3-E03"})
    result, = results
    assert result.status == "blocked"
    assert "网关口径" in result.details
    assert result.observations["rows"]["minimal"]["http_status"] == 200


def test_thinking_level_floor_fails_when_backend_contract_really_differs():
    """同一现象的另一侧：网关声明 low/medium/high（不含 minimal）却仍 200 接受 ⇒ 记 failed。"""
    fake = FakeGemini(accept_minimal=True, declared_levels=("low", "medium", "high"))
    results, _ = run_suite(fake, priorities=("P0",), case_ids={"G3-E03"})
    result, = results
    assert result.status == "failed"
    assert result.passed is False
    assert "身份指纹不成立" in result.details


def test_thinking_level_floor_passes_when_minimal_is_rejected_with_evidence():
    fake = FakeGemini(accept_minimal=False)
    results, _ = run_suite(fake, priorities=("P0",), case_ids={"G3-E03"})
    result, = results
    assert result.status == "passed"
    assert "档位下限指纹成立" in result.details


def test_invalid_thinking_level_reads_declared_universe():
    """非法档位被拒且文案给出合法集 —— 这是后续所有"该拒未拒"判定的取证前提。"""
    fake = FakeGemini(declared_levels=("low", "medium", "high"))
    results, _ = run_suite(fake, priorities=("P0",), case_ids={"G3-E04"})
    result, = results
    assert result.status == "passed"
    assert result.observations["declared_levels"] == ["low", "medium", "high"]


def test_transient_upstream_becomes_blocked_not_failed(monkeypatch):
    """5xx 重试耗尽必须收口 blocked（基础设施抖动不污染"模型能力"结论）。"""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": {"code": 503, "message": "upstream unavailable"}})

    async def scenario(tmp: Path):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            runner = Gemini3Runner(
                Gemini3Config(api_key="k", max_retries=1, retry_backoff_seconds=0.0),
                client=client, audit_dir=tmp,
            )
            async with runner:
                return await runner.run(("gemini-3.8-flash",), ("BASIC",), case_ids={"G3-B01"})

    with tempfile.TemporaryDirectory() as raw:
        results = asyncio.run(scenario(Path(raw)))
    result, = results
    assert result.status == "blocked"
    assert "瞬时" in result.details or "503" in result.details
    assert calls["n"] == 2          # 首次 + 1 次重试
    assert result.passed is False


def test_account_limitation_is_blocked_with_self_explanatory_reason():
    """账号/额度层拒绝属平台前提不满足，同样不得判 failed。"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                                   "message": "You exceeded your current quota, billing required"}})

    async def scenario(tmp: Path):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            runner = Gemini3Runner(Gemini3Config(api_key="k", max_retries=0), client=client, audit_dir=tmp)
            async with runner:
                return await runner.run(("gemini-3.8-flash",), ("BASIC",), case_ids={"G3-B01"})

    with tempfile.TemporaryDirectory() as raw:
        results = asyncio.run(scenario(Path(raw)))
    result, = results
    assert result.status == "blocked"
    assert "账号" in result.details and "quota" in result.details


def test_identity_csv_swap_is_detected():
    """响应 modelVersion 被换成别的模型 ⇒ BASIC 身份用例必须 failed（不是 warning）。"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_generate_body("gemini-2.5-flash", "OK"))

    async def scenario(tmp: Path):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            runner = Gemini3Runner(Gemini3Config(api_key="k"), client=client, audit_dir=tmp)
            async with runner:
                return await runner.run(("gemini-3.8-flash",), ("BASIC",), case_ids={"G3-B01"})

    with tempfile.TemporaryDirectory() as raw:
        results = asyncio.run(scenario(Path(raw)))
    result, = results
    assert result.status == "failed"
    assert "gemini-2.5-flash" in result.details


# ---------------------------------------------------------------------------
# CLI 端到端：main_async 的装配 → 落盘 → 收尾 → 退出码
# ---------------------------------------------------------------------------
def test_cli_main_async_writes_admitted_batch(tmp_path, monkeypatch):
    """整条 CLI 链路：证据落盘、summary 关键字段、报告版式、收尾后退出码。"""
    from gemini3.runner import main_async

    transport = httpx.MockTransport(FakeGemini())
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        # runner 会带 timeout/verify/trust_env/headers；MockTransport 客户端不接受这些位置参数。
        for key in ("timeout", "verify", "trust_env"):
            kwargs.pop(key, None)
        return real_client(transport=transport, headers=kwargs.get("headers"))

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://mux.la")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_PROTOCOL", "gemini")
    monkeypatch.delenv("VOLC_RECORD_ID", raising=False)

    code = asyncio.run(main_async(["GEMINI3_BASIC", "--output-root", str(tmp_path)]))
    assert code == 0

    dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(dirs) == 1 and dirs[0].name.startswith("GEMINI3_BASIC-")
    out = dirs[0]
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["test_kind"] == "gemini"
    assert summary["model"] == "gemini-3.8-flash"
    assert summary["model_variant"] == "gemini3"
    assert summary["model_tier"] == "gemini-3.8-flash"
    assert summary["endpoint"] == "https://mux.la/v1beta"
    assert summary["admission_passed"] is True
    assert summary["passed"] == 10 and summary["failed"] == 0
    assert summary["validation"]["passed"] is True
    # 证据链：requests.jsonl 必须非空（否则 finalize 会以缺证据为准，把整批判为不准入）。
    assert (out / "requests.jsonl").stat().st_size > 0
    assert (out / "report.md").read_text(encoding="utf-8").startswith("# Gemini 3.x 专项测试报告")


def test_cli_exit_code_is_two_when_batch_has_only_blocked(tmp_path, monkeypatch):
    """整批只有 blocked（无任何 passed/warning）时收尾会下调准入 ⇒ 退出码必须是 2。

    这是 2026-10-10 口径的落地校验：blocked 本身不阻断，但"完全没有有效结论"仍不准入。
    """
    from gemini3.runner import main_async

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"code": 503, "message": "upstream unavailable"}})

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        for key in ("timeout", "verify", "trust_env"):
            kwargs.pop(key, None)
        return real_client(transport=transport, headers=kwargs.get("headers"))

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://mux.la")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "0")

    code = asyncio.run(main_async(["GEMINI3_BASIC", "--output-root", str(tmp_path)]))
    assert code == 2
    out = next(p for p in tmp_path.iterdir() if p.is_dir())
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["blocked"] == 10 and summary["passed"] == 0
    assert summary["admission_passed"] is False
    check = next(c for c in summary["validation"]["checks"] if c["name"] == "case_verdicts_complete")
    assert check["passed"] is False
    assert "没有有效结论" in check["detail"] or "准入无依据" in check["detail"]
    # 退出码取的是收尾后的磁盘结论，而不是收尾前内存里的 all_passed。
    # 2026-10-21 起 runner 入口用 face_verdict 算准入（model/verify 面无有效结论即 False），
    # 所以 original_admission_passed 也是 False——收尾前的 all_passed 仍是 True，
    # 由 finalizer 的 case_verdicts_complete 兜住"整批 blocked 无依据"这一层。
    assert summary["admission_passed"] is False

