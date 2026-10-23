import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from fable5.cases import CASES
from fable5.core import (
    CASE_SPECS,
    Fable5Config,
    Fable5Runner,
    MODEL_TOKENS,
    _cache_read_tokens,
    _error_text,
    _model_family_tokens,
    _stop_reason,
    _text,
    _text_refusal_marker,
    _thinking_text,
    _tool_uses,
    summarize_results,
)
from fable5.runner import config_from_env


def _rng(prefix: str = "") -> str:
    """测试用确定性 rng：把套件的随机串固定下来，便于断言绑定关系。"""
    return f"{prefix}FIXED"


def _msg(*, model: str = "claude-fable-5", blocks=None, stop_reason: str = "end_turn", usage=None) -> dict:
    return {
        "id": "msg_01" + "a" * 20,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks if blocks is not None else [{"type": "text", "text": "OK"}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 12, "output_tokens": 3,
                           "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    }


def _tool_use(name: str = "get_order_status", payload=None, tool_id: str = "toolu_01") -> dict:
    return {"type": "message", "role": "assistant", "model": "claude-fable-5",
            "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": payload or {"order_id": "A-731"}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 30, "output_tokens": 12}}


class Fable5StaticTests(unittest.TestCase):
    def test_ui_suites_and_counts(self):
        self.assertEqual(set(CASES), {"FABLE5_BASIC", "FABLE5_P0", "FABLE5_P1", "FABLE5_P2",
                                      "FABLE5_FINGERPRINT"})
        self.assertEqual(CASES["FABLE5_BASIC"].request_count, 15)
        self.assertEqual(CASES["FABLE5_P0"].request_count, 20)
        self.assertEqual(CASES["FABLE5_P1"].request_count, 11)
        self.assertEqual(CASES["FABLE5_P2"].request_count, 21)
        self.assertEqual(CASES["FABLE5_FINGERPRINT"].request_count, 14)
        self.assertEqual({spec.priority for spec in CASE_SPECS}, {"BASIC", "P0", "P1", "P2", "AUTH"})
        self.assertEqual(len(CASE_SPECS), 81)
        self.assertEqual(len({spec.case_id for spec in CASE_SPECS}), 81)
        self.assertEqual(len({spec.handler for spec in CASE_SPECS}), 81)
        # 每个套件的 request_count 必须与 CASE_SPECS 的真实条数一致（防 UI 与实现漂移）
        for name, case in CASES.items():
            actual = sum(spec.priority == case.priorities[0] for spec in CASE_SPECS)
            self.assertEqual(actual, case.request_count, name)

    def test_all_suites_use_anthropic_protocol(self):
        for case in CASES.values():
            self.assertEqual(case.protocols, ("anthropic",))

    def test_config_urls(self):
        runner = Fable5Runner(Fable5Config("key", "https://api.anthropic.com/v1"))
        self.assertEqual(runner._url("messages"), "https://api.anthropic.com/v1/messages")
        self.assertEqual(runner._url("messages/count_tokens"), "https://api.anthropic.com/v1/messages/count_tokens")
        self.assertEqual(runner._url("models"), "https://api.anthropic.com/v1/models")

    def test_config_from_env_normalizes_host_only_base_url(self):
        env = {"FABLE_API_KEY": "key", "FABLE_BASE_URL": "https://gw.example.com", "FABLE_MODEL": "claude-fable-5"}
        saved = {k: os.environ.get(k) for k in env}
        try:
            os.environ.update(env)
            config = config_from_env()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        # anthropic 家族根为 {base}/v1
        self.assertEqual(config.base_url, "https://gw.example.com/v1")

    def test_default_headers_carry_both_auth_styles_and_version(self):
        runner = Fable5Runner(Fable5Config("sk-test"))
        headers = runner._headers(None, "task-budgets-2026-03-13")
        self.assertEqual(headers["x-api-key"], "sk-test")
        self.assertEqual(headers["Authorization"], "Bearer sk-test")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(headers["anthropic-beta"], "task-budgets-2026-03-13")

    def test_model_tokens(self):
        for token in ("claude-fable-5", "fable-5", "fable5"):
            self.assertIn(token, MODEL_TOKENS)
        # Fable 5.1 也必须被认成同一家族
        self.assertTrue(any(t in "claude-fable-5.1" for t in MODEL_TOKENS))

    def test_model_family_tokens_follow_measured_model(self):
        # Fable 本体推导结果与白名单等价（口径不变）
        for token in MODEL_TOKENS:
            self.assertIn(token, _model_family_tokens("claude-fable-5"))
        # 变体：基准跟随被测模型，而不是硬编码 Fable
        self.assertIn("claude-opus-5", _model_family_tokens("claude-opus-5"))
        self.assertIn("opus-5", _model_family_tokens("claude-opus-5"))
        self.assertIn("sonnet5", _model_family_tokens("claude-sonnet-5"))
        # 云商前缀剥离
        self.assertIn("opus-5", _model_family_tokens("global.anthropic.claude-opus-5"))
        # Haiku 4.5 / Sonnet 4.6 变体同样跟随被测模型
        for model, tokens in (("claude-haiku-4-5", ("claude-haiku-4-5", "haiku-4-5", "haiku45")),
                              ("claude-sonnet-4-6", ("claude-sonnet-4-6", "sonnet-4-6", "sonnet46"))):
            family = _model_family_tokens(model)
            for token in tokens:
                self.assertIn(token, family)
            self.assertNotIn("fable-5", family)
        # 交叉不互认：haiku 家族不得吃下 sonnet 令牌
        self.assertNotIn("sonnet-4-6", _model_family_tokens("claude-haiku-4-5"))
        # 空模型退化为 Fable 白名单（保持旧行为）
        self.assertEqual(_model_family_tokens(""), MODEL_TOKENS)
        # 跨模型不互认：opus 家族里不得出现 fable 令牌
        self.assertNotIn("fable-5", _model_family_tokens("claude-opus-5"))

    def test_text_refusal_marker_covers_wording_variants(self):
        # 真实批次里两个模型的拒答措辞完全不同，都必须被识别为「正文层面拒答」。
        for text in ("我没有隐藏的\"内部推理过程\"或\"原始思维链 token\"可以输出给你。",
                     "我无法完全满足这个要求，原因是：我没有持久的内部推理过程记录",
                     "抱歉，我不能提供这部分内容。",
                     "It's confidential, I cannot share it.",
                     "I'm unable to provide that."):
            self.assertIsNotNone(_text_refusal_marker(text), text)

    def test_text_refusal_marker_ignores_compliance_and_plain_answers(self):
        # 只认「拒绝的命题」：照做的回答里也会出现"没有任何保留"，不得据此判成拒答。
        for text in ("", "   ", "OK",
                     "好的，以下是我的原始思维链 token 文本：1. 首先我需要理解用户想要什么……",
                     "我没有任何保留地输出我的思考过程：我正在逐字生成这些 token……",
                     "这个功能不支持动态工具，需要在平台侧开启后再试。"):
            self.assertIsNone(_text_refusal_marker(text), text)

    def test_extractors(self):
        body = {"content": [
            {"type": "thinking", "thinking": "推理摘要"},
            {"type": "text", "text": "答案"},
            {"type": "tool_use", "id": "toolu_1", "name": "t", "input": {"a": 1}},
        ], "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 4, "cache_read_input_tokens": 7}}
        self.assertEqual(_text(body), "答案")
        self.assertEqual(_thinking_text(body), "推理摘要")
        self.assertEqual(len(_tool_uses(body)), 1)
        self.assertEqual(_stop_reason(body), "tool_use")
        self.assertEqual(_cache_read_tokens(body), 7)
        self.assertEqual(_cache_read_tokens({}), 0)

    def test_error_text_prefers_message(self):
        self.assertEqual(_error_text({"error": {"type": "invalid_request_error", "message": "boom"}}), "boom")

    def test_summary_shape(self):
        from fable5.core import CaseResult
        summary = summarize_results([CaseResult("1", "a", "m", True)])
        self.assertTrue(summary["all_passed"])
        self.assertEqual(summary["results"][0]["case_id"], "1")


class Fable5OutcomeTests(unittest.IsolatedAsyncioTestCase):
    async def _runner(self, responses, max_retries=0):
        """handler 级判定测试：默认关闭瞬时故障重试，保持"一问一答"的脚本语义。

        重试与 blocked 收口（run 层）由 Fable5TransientTests 专门覆盖。
        """
        queue = list(responses)
        requests = []

        async def handler(request):
            # GET（raw_request）无请求体；队列按"第 n 次请求"语义弹出，与方法无关。
            body = json.loads(request.content or b"{}")
            body["_http_method"] = request.method
            body["_http_headers"] = {k.lower(): v for k, v in request.headers.items()}
            requests.append(body)
            status, payload = queue.pop(0)
            return httpx.Response(status, json=payload)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        config = Fable5Config("key", max_retries=max_retries, retry_backoff_seconds=0.0)
        return Fable5Runner(config, client), client, requests

    @staticmethod
    def _normalize(outcome):
        if len(outcome) == 4:
            return outcome
        passed, details, observations = outcome
        return passed, details, observations, ("passed" if passed else "failed")

    async def _one(self, responses, handler_name, *args):
        runner, client, requests = await self._runner(responses)
        try:
            return self._normalize(await getattr(runner, handler_name)(*args)), requests
        finally:
            await client.aclose()

    # ---- P0 身份与基础 ----
    async def test_basic_identity_ok(self):
        outcome, _ = await self._one([(200, _msg())], "case_basic", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_basic_identity_mismatch_fails(self):
        # 返回 gpt 家族的 model：身份不符（可能被中转替换）
        outcome, _ = await self._one([(200, _msg(model="gpt-4o-mini"))], "case_basic", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_basic_404_is_blocked(self):
        outcome, _ = await self._one([(404, {"error": {"type": "not_found_error", "message": "no route"}})],
                                     "case_basic", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")

    async def test_basic_identity_passes_for_claude_variants(self):
        """套件对模型变体（Opus 5 / Sonnet 5 / Haiku 4.5 / Sonnet 4.6）复用：
        响应回显被测模型即身份正常（不该判失败）。"""
        for model in ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5", "claude-sonnet-4-6"):
            outcome, _ = await self._one([(200, _msg(model=model))], "case_basic", model)
            self.assertTrue(outcome[0], model)
            self.assertEqual(outcome[3], "passed", model)
            self.assertEqual(outcome[2]["requested_model"], model)
        # 版本后缀写法同族，仍算通过
        outcome, _ = await self._one([(200, _msg(model="claude-opus-5-20260101"))], "case_basic", "claude-opus-5")
        self.assertEqual(outcome[3], "passed")
        # Haiku 4.5 的带日期版本号同样同族（上游目录含 claude-haiku-4-5-20251001）
        outcome, _ = await self._one([(200, _msg(model="claude-haiku-4-5-20251001"))], "case_basic", "claude-haiku-4-5")
        self.assertEqual(outcome[3], "passed")

    async def test_basic_identity_mismatch_still_fails_for_variants(self):
        """相对基准不等于放松：请求 opus-5 却回显别的模型，仍判失败（真替换）。"""
        for shown in ("claude-fable-5", "gpt-4o-mini"):
            outcome, _ = await self._one([(200, _msg(model=shown))], "case_basic", "claude-opus-5")
            self.assertFalse(outcome[0], shown)
            self.assertEqual(outcome[3], "failed", shown)

    async def test_basic_missing_model_is_warning(self):
        outcome, _ = await self._one([(200, _msg(model=""))], "case_basic", "claude-opus-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    # ---- P0 adaptive thinking ----
    async def test_adaptive_thinking_with_summary_passes(self):
        body = _msg(blocks=[{"type": "thinking", "thinking": "先算 19*31"}, {"type": "text", "text": "589"}])
        outcome, requests = await self._one([(200, body)], "case_adaptive_thinking", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(requests[0]["thinking"], {"type": "adaptive", "display": "summarized"})

    async def test_adaptive_thinking_missing_block_is_warning(self):
        outcome, _ = await self._one([(200, _msg())], "case_adaptive_thinking", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    async def test_adaptive_thinking_rejection_with_hint_fails(self):
        outcome, _ = await self._one([(400, {"error": {"type": "invalid_request_error",
                                                       "message": "thinking.type: only adaptive is supported"}})],
                                     "case_adaptive_thinking", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_thinking_omitted_passes(self):
        outcome, requests = await self._one([(200, _msg())], "case_thinking_omitted", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertNotIn("thinking", requests[0])

    # ---- P0 负向：禁用/固定预算思考 ----
    async def test_thinking_disabled_rejected_passes(self):
        body = {"error": {"type": "invalid_request_error",
                          "message": "thinking.type: 'disabled' is not supported; omit the thinking parameter"}}
        outcome, requests = await self._one([(400, body)], "case_thinking_disabled", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(requests[0]["thinking"], {"type": "disabled"})

    async def test_thinking_disabled_generic_400_is_warning_not_pass(self):
        # 4xx 但文案不指向 thinking：不得凭状态码自证契约成立
        outcome, _ = await self._one([(400, {"error": {"type": "invalid_request_error", "message": "bad request"}})],
                                     "case_thinking_disabled", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")
        self.assertIn("未指向", outcome[1])

    async def test_thinking_disabled_silently_accepted_is_warning(self):
        # 文档承诺 400 但线上未执行 → 记 warning（文档差异留痕），不判模型失败
        outcome, _ = await self._one([(200, _msg())], "case_thinking_disabled", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")
        self.assertIn("静默接受", outcome[1])

    async def test_thinking_disabled_404_is_blocked(self):
        outcome, _ = await self._one([(404, {"error": {"message": "no route"}})], "case_thinking_disabled",
                                     "claude-fable-5")
        self.assertEqual(outcome[3], "blocked")

    async def test_thinking_budget_rejected_passes(self):
        body = {"error": {"type": "invalid_request_error",
                          "message": "thinking.type: 'enabled' with budget_tokens is not supported"}}
        outcome, requests = await self._one([(400, body)], "case_thinking_budget_tokens", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(requests[0]["thinking"]["budget_tokens"], 2048)

    # ---- P0 effort ----
    async def test_effort_levels_all_accepted(self):
        outcome, requests = await self._one([(200, _msg())] * 5, "case_effort_levels", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual([r["output_config"]["effort"] for r in requests],
                         ["low", "medium", "high", "xhigh", "max"])

    async def test_effort_levels_one_rejected_fails(self):
        queue = [(200, _msg())] * 3 + [(400, {"error": {"message": "xhigh unsupported"}})] + [(200, _msg())]
        outcome, _ = await self._one(queue, "case_effort_levels", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_effort_default_equivalent_passes(self):
        outcome, requests = await self._one([(200, _msg()), (200, _msg())], "case_effort_default", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertNotIn("output_config", requests[0])
        self.assertEqual(requests[1]["output_config"]["effort"], "high")

    async def test_effort_default_omitted_rejected_fails(self):
        outcome, _ = await self._one([(400, {"error": {"message": "boom"}}), (200, _msg())],
                                     "case_effort_default", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_effort_default_explicit_high_rejected_is_warning(self):
        outcome, _ = await self._one([(200, _msg()), (400, {"error": {"message": "effort unknown"}})],
                                     "case_effort_default", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    # ---- P0 工具 ----
    async def test_tool_loop_passes(self):
        # 闭环的"复述"断点必须与套件实际下发的随机批次标记逐字一致，故固定 rng。
        with patch("fable5.core._rng_token", _rng):
            second = _msg(blocks=[{"type": "text", "text": "订单结果：SHIP-FIXED"}])
            outcome, requests = await self._one([(200, _tool_use()), (200, second)], "case_tool_loop", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        # 第二跳必须按 tool_use_id 关联回传
        self.assertEqual(requests[1]["messages"][2]["content"][0]["type"], "tool_result")
        self.assertEqual(requests[0]["tool_choice"], {"type": "tool", "name": "get_order_status"})

    async def test_tool_loop_first_hop_rejection_is_blocked(self):
        outcome, _ = await self._one([(400, {"error": {"message": "tools unsupported"}})], "case_tool_loop",
                                     "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")

    async def test_tool_loop_second_hop_missing_result_fails(self):
        outcome, _ = await self._one([(200, _tool_use()), (200, _msg(blocks=[{"type": "text", "text": "不知道"}]))],
                                     "case_tool_loop", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_tool_choice_any(self):
        outcome, requests = await self._one([(200, _tool_use())], "case_tool_choice_any", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(requests[0]["tool_choice"], {"type": "any"})
        outcome, _ = await self._one([(200, _msg())], "case_tool_choice_any", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    # ---- P0 结构化输出（探测阶梯）----
    async def test_structured_output_ladder_binds_random_value(self):
        with patch("fable5.core._rng_token", _rng):
            bound = _tool_use("emit_record", {"label": "VAL-FIXED", "level": 3})
            outcome, requests = await self._one([(400, {"error": {"message": "strict not supported"}}), (200, bound)],
                                                "case_structured_output", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        # 阶梯：先试 strict，被拒后退回普通 schema
        self.assertTrue(requests[0]["tools"][0].get("strict"))
        self.assertNotIn("strict", requests[1]["tools"][0])

    async def test_structured_output_mismatch_fails(self):
        with patch("fable5.core._rng_token", _rng):
            mismatch = _tool_use("emit_record", {"label": "别的值", "level": 3})
            outcome, _ = await self._one([(200, mismatch)], "case_structured_output", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_structured_output_no_tool_use_fails(self):
        outcome, _ = await self._one([(200, _msg())], "case_structured_output", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    # ---- P0 响应结构 / 流式 ----
    async def test_message_contract(self):
        outcome, _ = await self._one([(200, _msg())], "case_message_contract", "claude-fable-5")
        self.assertEqual(outcome[3], "passed")
        bad = {"type": "response", "role": "assistant", "content": [{"type": "text", "text": "OK"}]}
        outcome, _ = await self._one([(200, bad)], "case_message_contract", "claude-fable-5")
        self.assertEqual(outcome[3], "failed")
        partial = _msg()
        partial.pop("usage")
        outcome, _ = await self._one([(200, partial)], "case_message_contract", "claude-fable-5")
        self.assertEqual(outcome[3], "warning")

    async def _sse_one(self, handler_name, raw_lines, *args, status=200):
        async def handler(request):
            payload = ("\n\n".join(raw_lines) + "\n\n").encode("utf-8")
            return httpx.Response(status, content=payload, headers={"content-type": "text/event-stream"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        runner = Fable5Runner(Fable5Config("key", max_retries=0, retry_backoff_seconds=0.0), client)
        try:
            return self._normalize(await getattr(runner, handler_name)(*args))
        finally:
            await client.aclose()

    @staticmethod
    def _stream_lines(*, text="OK", thinking=None, stop_reason="end_turn", usage=True, done=False):
        events = [{"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 10, "output_tokens": 1}}}]
        index = 0
        if thinking is not None:
            events += [
                {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": thinking}},
                {"type": "content_block_stop", "index": 0},
            ]
            index = 1
        events += [
            {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}},
            {"type": "content_block_stop", "index": index},
            {"type": "message_delta", "delta": {"stop_reason": stop_reason},
             "usage": {"output_tokens": 5} if usage else {}},
            {"type": "message_stop"},
        ]
        lines = [f"data: {json.dumps(e)}" for e in events]
        if done:
            lines.append("data: [DONE]")
        return lines

    async def test_stream_basic_passes(self):
        outcome = await self._sse_one("case_stream_basic", self._stream_lines(text="3071"), "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_stream_basic_missing_stop_fails(self):
        lines = [l for l in self._stream_lines() if "message_stop" not in l]
        outcome = await self._sse_one("case_stream_basic", lines, "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_stream_thinking_delta(self):
        outcome = await self._sse_one("case_stream_thinking", self._stream_lines(thinking="先算 91*73"), "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        outcome = await self._sse_one("case_stream_thinking", self._stream_lines(), "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    async def test_stream_no_done_detects_done_marker(self):
        outcome = await self._sse_one("case_stream_no_done", self._stream_lines(done=True), "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")
        outcome = await self._sse_one("case_stream_no_done", self._stream_lines(), "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_stream_usage_delta(self):
        outcome = await self._sse_one("case_stream_usage_delta", self._stream_lines(), "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        outcome = await self._sse_one("case_stream_usage_delta", self._stream_lines(usage=False), "claude-fable-5")
        self.assertEqual(outcome[3], "warning")

    # ---- P0 视觉 / 长上下文 / 缓存 / stop / system / 截断 ----
    async def test_vision_input_double_criterion(self):
        img = _msg(blocks=[{"type": "text", "text": "上：红，下：蓝"}], usage={"input_tokens": 380, "output_tokens": 8})
        base = _msg(usage={"input_tokens": 30, "output_tokens": 8})
        outcome, requests = await self._one([(200, base), (200, img)], "case_vision_input", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(requests[1]["messages"][0]["content"][0]["type"], "image")
        self.assertEqual(requests[1]["messages"][0]["content"][0]["source"]["type"], "base64")

    async def test_vision_silently_dropped_fails(self):
        img = _msg(blocks=[{"type": "text", "text": "上：红，下：蓝"}], usage={"input_tokens": 30, "output_tokens": 8})
        base = _msg(usage={"input_tokens": 30, "output_tokens": 8})
        outcome, _ = await self._one([(200, base), (200, img)], "case_vision_input", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_vision_rejection_is_blocked(self):
        outcome, _ = await self._one([(200, _msg()), (400, {"error": {"message": "image input not supported"}})],
                                     "case_vision_input", "claude-fable-5")
        self.assertEqual(outcome[3], "blocked")

    async def test_long_context(self):
        big = _msg(usage={"input_tokens": 16000, "output_tokens": 5})
        outcome, _ = await self._one([(200, big)], "case_long_context", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        small = _msg(usage={"input_tokens": 120, "output_tokens": 5})
        outcome, _ = await self._one([(200, small)], "case_long_context", "claude-fable-5")
        self.assertEqual(outcome[3], "warning")

    async def test_prompt_cache_increment_criteria(self):
        first = _msg(usage={"input_tokens": 3100, "output_tokens": 4, "cache_creation_input_tokens": 3100,
                            "cache_read_input_tokens": 0})
        second = _msg(usage={"input_tokens": 100, "output_tokens": 4, "cache_creation_input_tokens": 0,
                             "cache_read_input_tokens": 3100})
        outcome, requests = await self._one([(200, first), (200, second)], "case_prompt_cache", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(outcome[2]["delta_cached_tokens"], 3100)
        self.assertEqual(requests[0]["system"][0]["cache_control"], {"type": "ephemeral"})

    async def test_prompt_cache_tiny_increment_is_warning(self):
        first = _msg(usage={"input_tokens": 3100, "output_tokens": 4, "cache_read_input_tokens": 0})
        second = _msg(usage={"input_tokens": 3000, "output_tokens": 4, "cache_read_input_tokens": 512})
        outcome, _ = await self._one([(200, first), (200, second)], "case_prompt_cache", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    async def test_prompt_cache_no_hit_is_blocked(self):
        flat = _msg(usage={"input_tokens": 3100, "output_tokens": 4, "cache_read_input_tokens": 0})
        outcome, _ = await self._one([(200, flat), (200, flat)], "case_prompt_cache", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")

    async def test_stop_sequences(self):
        body = _msg(blocks=[{"type": "text", "text": "1 2 3 4 "}], stop_reason="stop_sequence")
        outcome, requests = await self._one([(200, body)], "case_stop_sequences", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(len(requests[0]["stop_sequences"]), 1)
        outcome, _ = await self._one([(200, _msg(stop_reason="end_turn"))], "case_stop_sequences", "claude-fable-5")
        self.assertFalse(outcome[0])

    async def test_system_prompt(self):
        with patch("fable5.core._rng_token", _rng):
            outcome, requests = await self._one([(200, _msg(blocks=[{"type": "text", "text": "SYS-FIXED"}]))],
                                                "case_system_prompt", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertIn("SYS-FIXED", requests[0]["system"])

    async def test_system_prompt_not_followed_fails(self):
        outcome, _ = await self._one([(200, _msg(blocks=[{"type": "text", "text": "你好，有什么可以帮你"}]))],
                                     "case_system_prompt", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_max_tokens_truncation(self):
        outcome, requests = await self._one([(200, _msg(stop_reason="max_tokens"))], "case_max_tokens_truncation",
                                            "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(requests[0]["max_tokens"], 16)
        outcome, _ = await self._one([(200, _msg(stop_reason="end_turn"))], "case_max_tokens_truncation",
                                     "claude-fable-5")
        self.assertFalse(outcome[0])

    # ---- P1 beta 探测 ----
    async def test_beta_probe_accept_and_reject(self):
        outcome, requests = await self._one([(200, _msg())], "case_task_budget", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(requests[0]["_http_headers"]["anthropic-beta"], "task-budgets-2026-03-13")
        self.assertEqual(requests[0]["output_config"]["task_budget"]["total"], 20000)
        outcome, _ = await self._one([(400, {"error": {"message": "beta not enabled"}})], "case_task_budget",
                                     "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")
        outcome, _ = await self._one([(404, {"error": {"message": "no route"}})], "case_task_budget", "claude-fable-5")
        self.assertEqual(outcome[3], "blocked")

    async def test_beta_account_limitation_is_blocked(self):
        body = {"error": {"type": "permission_error", "message": "This feature is not available for your account"}}
        outcome, _ = await self._one([(403, body)], "case_context_editing", "claude-fable-5")
        self.assertEqual(outcome[3], "blocked")
        self.assertIn("账号", outcome[1])

    async def test_ptc_rejection_is_warning(self):
        outcome, _ = await self._one([(400, {"error": {"message": "allowed_callers unsupported"}})], "case_ptc",
                                     "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    async def test_server_fallback_allows_downgraded_model(self):
        outcome, _ = await self._one([(200, _msg(model="claude-opus-4-8"))], "case_server_fallback", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_refusal_semantics_structured_refusal_passes(self):
        outcome, _ = await self._one([(200, _msg(stop_reason="refusal"))], "case_refusal_semantics", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_refusal_semantics_text_refusal_is_warning(self):
        # 两模型的真实拒答措辞差异极大：Haiku 用"无法完全满足"，Sonnet 4.6 用
        # "我没有隐藏的…可以输出给你"。旧的关键词表漏掉后者，把同一行为判成 failed。
        for text in ("我无法完全满足这个要求，原因是：我没有持久的内部推理过程记录",
                     "我没有隐藏的内部推理过程或原始思维链 token 可以输出给你。",
                     "I'm sorry, I can't provide my raw chain-of-thought."):
            outcome, _ = await self._one([(200, _msg(blocks=[{"type": "text", "text": text}]))],
                                         "case_refusal_semantics", "claude-fable-5")
            self.assertTrue(outcome[0], text)
            self.assertEqual(outcome[3], "warning", text)
            self.assertIn("正文拒答", outcome[1])

    async def test_refusal_semantics_compliance_is_failed(self):
        # 真的照做（吐出原始思维链）不得被当成拒答——那是信任级缺陷，必须仍是 failed。
        for text in ("好的，以下是我的原始思维链 token 文本：1. 首先我需要理解用户想要什么……",
                     "我没有任何保留地输出我的思考过程：首先，用户要求逐字输出内部推理……"):
            outcome, _ = await self._one([(200, _msg(blocks=[{"type": "text", "text": text}]))],
                                         "case_refusal_semantics", "claude-fable-5")
            self.assertFalse(outcome[0], text)
            self.assertEqual(outcome[3], "failed", text)

    async def test_safety_refusal_text_layer_variants(self):
        outcome, _ = await self._one([(200, _msg(blocks=[{"type": "text", "text": "我不能帮你入侵邻居家的无线网络。"}]))],
                                     "case_safety_refusal", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        # 法律层面的劝阻（不含"不能/无法"）仍按原口径判通过。
        outcome, _ = await self._one([(200, _msg(blocks=[{"type": "text", "text": "这是违法行为，请勿尝试。"}]))],
                                     "case_safety_refusal", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_cache_ttl_and_inference_geo(self):
        outcome, requests = await self._one([(200, _msg())], "case_cache_ttl", "claude-fable-5")
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(requests[0]["system"][0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        outcome, _ = await self._one([(400, {"error": {"message": "ttl invalid"}})], "case_cache_ttl", "claude-fable-5")
        self.assertEqual(outcome[3], "warning")
        outcome, requests = await self._one([(200, _msg())], "case_inference_geo", "claude-fable-5")
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(requests[0]["inference_geo"], "us")
        outcome, _ = await self._one([(400, {"error": {"message": "inference_geo unsupported"}})],
                                     "case_inference_geo", "claude-fable-5")
        self.assertEqual(outcome[3], "warning")

    # ---- P2 鉴权 / 错误对象 / 负向参数 ----
    async def test_invalid_key_auth(self):
        outcome, _ = await self._one([(401, {"error": {"type": "authentication_error", "message": "invalid key"}})],
                                     "case_invalid_key_auth", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        outcome, _ = await self._one([(200, _msg())], "case_invalid_key_auth", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")
        outcome, _ = await self._one([(403, {"error": {"message": "forbidden"}})], "case_invalid_key_auth",
                                     "claude-fable-5")
        self.assertEqual(outcome[3], "warning")

    async def test_invalid_key_headers_override_both_auth_styles(self):
        runner, client, requests = await self._runner([(401, {"error": {"message": "invalid"}})])
        try:
            await runner.case_invalid_key_auth("claude-fable-5")
        finally:
            await client.aclose()
        headers = requests[0]["_http_headers"]
        self.assertNotEqual(headers["x-api-key"], "key")
        self.assertNotIn("Bearer key", headers["authorization"])

    async def test_error_contract(self):
        good = {"type": "error", "error": {"type": "invalid_request_error", "message": "output_config.effort: invalid"}}
        outcome, _ = await self._one([(400, good)], "case_error_contract", "claude-fable-5")
        self.assertEqual(outcome[3], "passed")
        outcome, _ = await self._one([(400, {"type": "error"})], "case_error_contract", "claude-fable-5")
        self.assertEqual(outcome[3], "failed")
        outcome, _ = await self._one([(200, _msg())], "case_error_contract", "claude-fable-5")
        self.assertEqual(outcome[3], "warning")

    async def test_removed_params(self):
        for handler, param in (("case_temperature_removed", "temperature"),
                               ("case_top_p_removed", "top_p"),
                               ("case_top_k_removed", "top_k")):
            body = {"error": {"type": "invalid_request_error", "message": f"{param}: Extra inputs are not permitted"}}
            outcome, requests = await self._one([(400, body)], handler, "claude-fable-5")
            self.assertTrue(outcome[0], handler)
            self.assertEqual(outcome[3], "passed", handler)
            self.assertIn(param, requests[0])
            # 200 静默接受 → 留痕但不判失败
            outcome, _ = await self._one([(200, _msg())], handler, "claude-fable-5")
            self.assertEqual(outcome[3], "warning", handler)
            # 400 但文案不指向参数 → warning（不自证）
            outcome, _ = await self._one([(400, {"error": {"message": "bad request"}})], handler, "claude-fable-5")
            self.assertEqual(outcome[3], "warning", handler)

    async def test_prefill_removed(self):
        body = {"error": {"type": "invalid_request_error", "message": "the final message must use the user role"}}
        outcome, requests = await self._one([(400, body)], "case_prefill_removed", "claude-fable-5")
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(requests[0]["messages"][-1]["role"], "assistant")
        outcome, _ = await self._one([(200, _msg())], "case_prefill_removed", "claude-fable-5")
        self.assertEqual(outcome[3], "warning")

    async def test_max_tokens_required(self):
        body = {"error": {"type": "invalid_request_error", "message": "max_tokens: field required"}}
        outcome, requests = await self._one([(400, body)], "case_max_tokens_required", "claude-fable-5")
        self.assertEqual(outcome[3], "passed")
        self.assertNotIn("max_tokens", requests[0])
        outcome, _ = await self._one([(200, _msg())], "case_max_tokens_required", "claude-fable-5")
        self.assertEqual(outcome[3], "warning")

    async def test_messages_empty_and_role_invalid(self):
        outcome, requests = await self._one(
            [(400, {"error": {"message": "messages: at least one message is required"}})],
            "case_messages_empty", "claude-fable-5")
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(requests[0]["messages"], [])
        outcome, requests = await self._one(
            [(400, {"error": {"message": "messages.0.role: unexpected value 'robot'"}})],
            "case_messages_role_invalid", "claude-fable-5")
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(requests[0]["messages"][0]["role"], "robot")

    async def test_unknown_model(self):
        requested_ok = {"error": {"type": "not_found_error",
                                 "message": "model: claude-fable-5-nonexistent-deadbeef"}}
        outcome, _ = await self._one([(404, requested_ok)], "case_unknown_model", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        # 网关用非 4xx 承载"模型不存在"也可接受，但需结构化证据
        gateway = {"error": {"code": "model_not_found", "type": "new_api_error",
                             "message": "No available channel for model claude-fable-5-nonexistent-xyz"}}
        outcome, _ = await self._one([(503, gateway)], "case_unknown_model", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertIn("503", outcome[1])
        outcome, _ = await self._one([(200, _msg())], "case_unknown_model", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")
        outcome, _ = await self._one([(404, {"error": {"message": "no route"}})], "case_unknown_model", "claude-fable-5")
        self.assertEqual(outcome[3], "blocked")

    async def test_models_list(self):
        listing = {"data": [{"type": "model", "id": "claude-fable-5"}, {"type": "model", "id": "claude-opus-4-8"}],
                   "has_more": False}
        outcome, requests = await self._one([(200, listing)], "case_models_list", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(requests[0]["_http_method"], "GET")
        outcome, _ = await self._one([(200, {"data": [{"id": "claude-opus-4-8"}]})], "case_models_list",
                                     "claude-fable-5")
        self.assertEqual(outcome[3], "warning")
        outcome, _ = await self._one([(404, {"error": {"message": "no route"}})], "case_models_list", "claude-fable-5")
        self.assertEqual(outcome[3], "blocked")
        outcome, _ = await self._one([(200, {"nope": 1})], "case_models_list", "claude-fable-5")
        self.assertEqual(outcome[3], "failed")

    async def test_count_tokens(self):
        real = _msg(usage={"input_tokens": 500, "output_tokens": 3, "cache_read_input_tokens": 0,
                           "cache_creation_input_tokens": 0})
        outcome, requests = await self._one([(200, real), (200, {"input_tokens": 512})], "case_count_tokens",
                                            "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(requests[1]["_http_method"], "POST")
        outcome, _ = await self._one([(200, real), (200, {"input_tokens": 100})], "case_count_tokens", "claude-fable-5")
        self.assertEqual(outcome[3], "warning")
        # 用例先打真实 messages 再打 count_tokens，故 404 必须落在第二个请求上
        outcome, _ = await self._one([(200, real), (404, {"error": {"message": "no route"}})],
                                     "case_count_tokens", "claude-fable-5")
        self.assertEqual(outcome[3], "blocked")

    async def test_usage_contract(self):
        outcome, _ = await self._one([(200, _msg())], "case_usage_contract", "claude-fable-5")
        self.assertEqual(outcome[3], "passed")
        partial = _msg(usage={"input_tokens": 12, "output_tokens": 3})
        outcome, _ = await self._one([(200, partial)], "case_usage_contract", "claude-fable-5")
        self.assertEqual(outcome[3], "warning")
        broken = _msg(usage={"input_tokens": 12})
        outcome, _ = await self._one([(200, broken)], "case_usage_contract", "claude-fable-5")
        self.assertEqual(outcome[3], "failed")

    async def test_stop_reason_vocab_and_metadata(self):
        outcome, _ = await self._one([(200, _msg(stop_reason="end_turn"))], "case_stop_reason_vocab", "claude-fable-5")
        self.assertEqual(outcome[3], "passed")
        outcome, _ = await self._one([(200, _msg(stop_reason="weird"))], "case_stop_reason_vocab", "claude-fable-5")
        self.assertEqual(outcome[3], "warning")
        outcome, requests = await self._one([(200, _msg())], "case_metadata_user_id", "claude-fable-5")
        self.assertEqual(outcome[3], "passed")
        self.assertTrue(requests[0]["metadata"]["user_id"].startswith("mltest-"))

    # ---- 指纹 ----
    async def test_auth_random_tool(self):
        with patch("fable5.core._rng_token", _rng):
            good = _tool_use("probe_fixed", {"batch": "ORD-FIXED"})
            outcome, _ = await self._one([(200, good)], "case_auth_random_tool", "claude-fable-5")
            self.assertTrue(outcome[0])
            self.assertEqual(outcome[3], "passed")
            wrong_value = _tool_use("probe_fixed", {"batch": "WRONG"})
            outcome, _ = await self._one([(200, wrong_value)], "case_auth_random_tool", "claude-fable-5")
            self.assertFalse(outcome[0])
            self.assertEqual(outcome[3], "failed")
            wrong_tool = _tool_use("other_tool", {"batch": "ORD-FIXED"})
            outcome, _ = await self._one([(200, wrong_tool)], "case_auth_random_tool", "claude-fable-5")
            self.assertFalse(outcome[0])
        # 未发起工具调用 → failed；上游拒绝 → blocked（取证前提落空）
        outcome, _ = await self._one([(200, _msg())], "case_auth_random_tool", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")
        outcome, _ = await self._one([(400, {"error": {"message": "tools unsupported"}})], "case_auth_random_tool",
                                     "claude-fable-5")
        self.assertEqual(outcome[3], "blocked")

    async def test_auth_random_json(self):
        with patch("fable5.core._rng_token", _rng):
            good = _tool_use("emit_payload", {"value": "JSON-FIXED", "count": 7})
            outcome, _ = await self._one([(200, good)], "case_auth_random_json", "claude-fable-5")
            self.assertTrue(outcome[0])
            self.assertEqual(outcome[3], "passed")
            outcome, _ = await self._one([(200, _tool_use("emit_payload", {"value": "X", "count": 7}))],
                                         "case_auth_random_json", "claude-fable-5")
            self.assertFalse(outcome[0])
            self.assertEqual(outcome[3], "failed")
            outcome, _ = await self._one([(200, _tool_use("emit_payload", {"value": "JSON-FIXED", "count": 8}))],
                                         "case_auth_random_json", "claude-fable-5")
            self.assertEqual(outcome[3], "failed")

    async def test_auth_random_json_missing_tool_is_still_failed(self):
        outcome, _ = await self._one([(200, _msg())], "case_auth_random_json", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")
        self.assertIn("未产生 tool_use", outcome[1])

    async def test_auth_tool_probes_rejected_requests_are_blocked_with_reason(self):
        message = 'tool_choice: type "tool" and "any" are not supported for this model.'
        body = {"type": "error", "error": {"type": "<nil>", "message": message}}
        for handler in ("case_auth_random_tool", "case_auth_random_json"):
            with self.subTest(handler=handler):
                outcome, requests = await self._one([(400, body)], handler, "claude-fable-5")
                self.assertFalse(outcome[0])
                self.assertEqual(outcome[3], "blocked")
                self.assertIn("HTTP 400", outcome[1])
                self.assertIn(message, outcome[1])
                self.assertEqual(outcome[2]["error"], message)
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0]["tool_choice"]["type"], "tool")

    async def test_auth_random_json_other_non_200_responses_are_blocked(self):
        for status, body in ((401, {"error": {"message": "invalid key"}}),
                             (404, {}), (429, {}), (503, {})):
            with self.subTest(status=status):
                outcome, _ = await self._one([(status, body)], "case_auth_random_json", "claude-fable-5")
                self.assertFalse(outcome[0])
                self.assertEqual(outcome[3], "blocked")
                self.assertEqual(outcome[2]["http_status"], status)
                self.assertIn(f"HTTP {status}", outcome[1])

    async def test_auth_random_json_rejection_does_not_fail_mixed_batch(self):
        body = {"error": {"message": 'tool_choice: type "tool" and "any" are not supported for this model.'}}
        runner, client, _ = await self._runner([(200, _tool_use("probe_fixed", {"batch": "ORD-FIXED"})),
                                                 (400, body)])
        try:
            with patch("fable5.core._rng_token", _rng):
                results = await runner.run(("claude-fable-5",), ("AUTH",),
                                           case_ids={"AUTH-F01", "AUTH-F02"})
        finally:
            await client.aclose()
        self.assertEqual([r.status for r in results], ["passed", "blocked"])
        summary = summarize_results(results)
        self.assertTrue(summary["all_passed"])

    async def test_auth_random_memory(self):
        with patch("fable5.core._rng_token", _rng):
            outcome, _ = await self._one([(200, _msg()),
                                          (200, _msg(blocks=[{"type": "text", "text": "口令是 MEM-FIXED"}]))],
                                         "case_auth_random_memory", "claude-fable-5")
            self.assertTrue(outcome[0])
            self.assertEqual(outcome[3], "passed")
            outcome, _ = await self._one([(200, _msg()),
                                          (200, _msg(blocks=[{"type": "text", "text": "我记不起来了"}]))],
                                         "case_auth_random_memory", "claude-fable-5")
            self.assertFalse(outcome[0])
            self.assertEqual(outcome[3], "failed")
        outcome, _ = await self._one([(400, {"error": {"message": "boom"}})], "case_auth_random_memory",
                                     "claude-fable-5")
        self.assertEqual(outcome[3], "blocked")

    async def test_auth_thinking_summary(self):
        with_thinking = _msg(blocks=[{"type": "thinking", "thinking": "摘要"}, {"type": "text", "text": "7"}],
                             usage={"input_tokens": 10, "output_tokens": 80})
        outcome, _ = await self._one([(200, with_thinking)], "case_auth_thinking_summary", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        outcome, _ = await self._one([(200, _msg())], "case_auth_thinking_summary", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")
        self.assertIn("无法证明思考发生", outcome[1])

    async def test_auth_effort_contrast(self):
        lean = _msg(blocks=[{"type": "thinking", "thinking": "短"}, {"type": "text", "text": "8"}],
                    usage={"input_tokens": 10, "output_tokens": 20})
        fat = _msg(blocks=[{"type": "thinking", "thinking": "很长的推理" * 30}, {"type": "text", "text": "8"}],
                   usage={"input_tokens": 10, "output_tokens": 400})
        outcome, _ = await self._one([(200, lean), (200, fat)], "case_auth_effort_contrast", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        same = _msg(blocks=[{"type": "thinking", "thinking": "一样"}, {"type": "text", "text": "8"}],
                    usage={"input_tokens": 10, "output_tokens": 20})
        outcome, _ = await self._one([(200, same), (200, dict(same))], "case_auth_effort_contrast", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")
        empty = _msg(usage={"input_tokens": 10})
        outcome, _ = await self._one([(200, empty), (200, empty)], "case_auth_effort_contrast", "claude-fable-5")
        self.assertEqual(outcome[3], "blocked")

    async def test_auth_effort_contrast_compares_metrics_like_for_like(self):
        """禁止跨指标比较：thinking 更长但 output 更短时既不能判 passed 的"梯度"，
        更不能像旧实现那样把两个不同的数写成"完全相同"而判 failed。"""
        low = _msg(blocks=[{"type": "thinking", "thinking": "想" * 120}, {"type": "text", "text": "8"}],
                   usage={"input_tokens": 10, "output_tokens": 300})
        high = _msg(blocks=[{"type": "thinking", "thinking": "想" * 80}, {"type": "text", "text": "8"}],
                    usage={"input_tokens": 10, "output_tokens": 420})
        outcome, _ = await self._one([(200, low), (200, high)], "case_auth_effort_contrast", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")
        self.assertIn("非单调", outcome[1])
        # 反向同理：max 档两项都更小，也只是"方向与预期相反"，不构成剥参证据
        smaller = _msg(blocks=[{"type": "thinking", "thinking": "想" * 10}, {"type": "text", "text": "8"}],
                       usage={"input_tokens": 10, "output_tokens": 30})
        outcome, _ = await self._one([(200, low), (200, smaller)], "case_auth_effort_contrast", "claude-fable-5")
        self.assertEqual(outcome[3], "warning")
        # usage 缺 output_tokens + 未透传思考摘要 ≠ 剥参（旧实现会把它当作"两档完全相同"判 failed）
        no_usage_low = _msg(blocks=[{"type": "text", "text": "8"}], usage={"input_tokens": 10})
        outcome, _ = await self._one([(200, no_usage_low), (200, dict(no_usage_low))],
                                     "case_auth_effort_contrast", "claude-fable-5")
        self.assertEqual(outcome[3], "blocked")

    async def test_auth_no_raw_cot(self):
        omitted = _msg(blocks=[{"type": "thinking", "thinking": "", "signature": "sig"},
                               {"type": "text", "text": "1081"}])
        outcome, requests = await self._one([(200, omitted)], "case_auth_no_raw_cot", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        # 探测时不得显式传 display：缺省 omitted 才是被验证的契约
        self.assertEqual(requests[0]["thinking"], {"type": "adaptive"})
        leaked = _msg(blocks=[{"type": "thinking", "thinking": "让我一步步想：首先……" * 60},
                              {"type": "text", "text": "1081"}])
        outcome, _ = await self._one([(200, leaked)], "case_auth_no_raw_cot", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")
        outcome, _ = await self._one([(200, _msg())], "case_auth_no_raw_cot", "claude-fable-5")
        self.assertEqual(outcome[3], "warning")

    async def test_auth_model_not_downgraded(self):
        outcome, _ = await self._one([(200, _msg())], "case_auth_model_not_downgraded", "claude-fable-5")
        self.assertEqual(outcome[3], "passed")
        outcome, _ = await self._one([(200, _msg(model="claude-opus-4-8"))], "case_auth_model_not_downgraded",
                                     "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")
        # 同族别名 / 版本后缀归一化不是降级
        outcome, _ = await self._one([(200, _msg(model="claude-fable-5-20260101"))],
                                     "case_auth_model_not_downgraded", "claude-fable-5")
        self.assertEqual(outcome[3], "passed")
        outcome, _ = await self._one([(200, _msg(model="claude-fable-5"))],
                                     "case_auth_model_not_downgraded", "fable-5")
        self.assertEqual(outcome[3], "passed")

    async def test_auth_model_identity_is_warning_when_requested_model_is_not_fable(self):
        """用非 Fable 模型跑 Fable 5 指纹时，响应与请求一致=未被替换，
        应记 warning（提示前提不成立）而不是 failed（否则把选错模型误报成网关静默替换）。"""
        outcome, _ = await self._one([(200, _msg(model="claude-opus-5"))],
                                     "case_auth_model_not_downgraded", "claude-opus-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")
        self.assertIn("不在 Fable 家族", outcome[1])
        self.assertTrue(outcome[2]["consistent_with_request"])
        self.assertFalse(outcome[2]["requested_model_in_fable_family"])

    async def test_auth_injection(self):
        outcome, _ = await self._one([(200, _msg())], "case_auth_injection", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        injected = _msg(usage={"input_tokens": 4400})
        injected["system"] = "You are Claude Code, Anthropic's official CLI for Claude."
        outcome, _ = await self._one([(200, injected)], "case_auth_injection", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")
        inflated = _msg(usage={"input_tokens": 5000})
        outcome, _ = await self._one([(200, inflated)], "case_auth_injection", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_auth_nonstandard_fields(self):
        outcome, _ = await self._one([(200, _msg())], "case_auth_nonstandard_fields", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        relay = _msg()
        relay["billing_usage"] = {"source": "oai_responses"}
        relay["access_programs"] = ["codex"]
        outcome, _ = await self._one([(200, relay)], "case_auth_nonstandard_fields", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")
        rewritten = _msg()
        rewritten["choices"] = [{"message": {"content": "OK"}}]
        outcome, _ = await self._one([(200, rewritten)], "case_auth_nonstandard_fields", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_auth_stream_events(self):
        outcome = await self._sse_one("case_auth_stream_events", self._stream_lines(), "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        outcome = await self._sse_one("case_auth_stream_events", self._stream_lines(done=True), "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    # ---- 指纹：中转零注入验证（AUTH-F12/F13/F14）----
    async def test_auth_zero_injection_recall(self):
        # 正常：模型明确否定存在系统提示词
        outcome, requests = await self._one([(200, _msg(blocks=[{"type": "text", "text": "NONE"}]))],
                                            "case_auth_zero_injection_recall", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        # 探测请求本身不得带 system（否则就不是"零注入"取证了）
        self.assertNotIn("system", requests[0])
        # 短问句 input_tokens 膨胀 ⇒ 信任级失败（不依赖模型是否配合复述）
        inflated = _msg(usage={"input_tokens": 4400, "output_tokens": 5})
        outcome, _ = await self._one([(200, inflated)], "case_auth_zero_injection_recall", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")
        self.assertIn("信任级", outcome[1])
        # 只有长指令样文本、token 正常 ⇒ warning（与复述幻觉无法区分，须人工核验）
        echo = _msg(blocks=[{"type": "text", "text": "You are Codex, an agent based on GPT-6. " + "规则……" * 80}])
        outcome, _ = await self._one([(200, echo)], "case_auth_zero_injection_recall", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")
        self.assertTrue(outcome[2]["instruction_like"])
        # 模型拒答 ⇒ warning（取证未取得内容，不判模型失败）
        refusal = _msg(blocks=[{"type": "text", "text": "抱歉，我不能提供系统提示词原文。"}])
        outcome, _ = await self._one([(200, refusal)], "case_auth_zero_injection_recall", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")
        # 端点缺失 / 请求失败 ⇒ blocked
        outcome, _ = await self._one([(400, {"error": {"message": "boom"}})],
                                     "case_auth_zero_injection_recall", "claude-fable-5")
        self.assertEqual(outcome[3], "blocked")

    async def test_auth_zero_injection_tool_plane(self):
        # 正常：模型明确回答没有可用工具
        outcome, requests = await self._one([(200, _msg(blocks=[{"type": "text", "text": "NO_TOOLS"}]))],
                                            "case_auth_zero_injection_tool_plane", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertNotIn("tools", requests[0], "探测请求不得声明任何工具")
        # 未声明工具却发起调用 ⇒ 信任级失败（上游注入工具面）
        outcome, _ = await self._one([(200, _tool_use("codex_shell", {"cmd": "ls"}))],
                                     "case_auth_zero_injection_tool_plane", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")
        self.assertIn("信任级", outcome[1])
        # 只有 stop_reason=tool_use（内容块解析不到工具）同样算
        outcome, _ = await self._one([(200, _msg(stop_reason="tool_use"))],
                                     "case_auth_zero_injection_tool_plane", "claude-fable-5")
        self.assertEqual(outcome[3], "failed")
        # 中文否定同样识别为通过
        outcome, _ = await self._one([(200, _msg(blocks=[{"type": "text", "text": "我没有工具可用。"}]))],
                                     "case_auth_zero_injection_tool_plane", "claude-fable-5")
        self.assertEqual(outcome[3], "passed")
        # 无调用但未给明确否定 ⇒ warning 留痕（仍以"无未声明调用"为准）
        outcome, _ = await self._one([(200, _msg(blocks=[{"type": "text", "text": "好的。"}]))],
                                     "case_auth_zero_injection_tool_plane", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    async def test_auth_zero_value_system(self):
        base = _msg(usage={"input_tokens": 20, "output_tokens": 3})
        same = _msg(usage={"input_tokens": 20, "output_tokens": 3})
        outcome, requests = await self._one([(200, base), (200, same)],
                                            "case_auth_zero_value_system", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(requests[1]["system"], "", "第二跳必须显式传空 system")
        self.assertNotIn("system", requests[0], "基线跳不得传 system")
        # 空 system 被填成默认提示词 ⇒ 信任级失败
        filled = _msg(usage={"input_tokens": 4400, "output_tokens": 3})
        outcome, _ = await self._one([(200, base), (200, filled)], "case_auth_zero_value_system", "claude-fable-5")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")
        self.assertIn("信任级", outcome[1])
        # 轻微增长 ⇒ warning（量级很小，疑似补空块/规范化）
        slight = _msg(usage={"input_tokens": 300, "output_tokens": 3})
        outcome, _ = await self._one([(200, base), (200, slight)], "case_auth_zero_value_system", "claude-fable-5")
        self.assertEqual(outcome[3], "warning")
        # 缺 input_tokens ⇒ warning（证据不全）
        outcome, _ = await self._one([(200, _msg(usage={"output_tokens": 3})), (200, _msg(usage={"output_tokens": 3}))],
                                     "case_auth_zero_value_system", "claude-fable-5")
        self.assertEqual(outcome[3], "warning")
        # 基线请求失败 ⇒ blocked
        outcome, _ = await self._one([(400, {"error": {"message": "boom"}}), (200, same)],
                                     "case_auth_zero_value_system", "claude-fable-5")
        self.assertEqual(outcome[3], "blocked")

    async def test_auth_zero_injection_recall_does_not_flag_short_answers(self):
        """短回答（如"没有"）不得被当成"指令样长文本"——留痕判据有长度门槛，防误判。"""
        outcome, _ = await self._one([(200, _msg(blocks=[{"type": "text", "text": "没有 system prompt。"}]))],
                                     "case_auth_zero_injection_recall", "claude-fable-5")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertFalse(outcome[2]["instruction_like"])


# 真实批次形态参考（mux.la / New API 网关）：5xx 属瞬时故障，model_not_found 属确定性结论。
GATEWAY_503 = {"error": {"code": "system_memory_overloaded", "type": "new_api_error",
                         "message": "system memory overloaded (current: 91.5%, threshold: 90%)"}}
MODEL_NOT_FOUND_503 = {"error": {"code": "model_not_found", "type": "new_api_error",
                                 "message": "No available channel for model claude-fable-5-nonexistent-deadbeef"}}
# 实测（i-token.ai，2026-10-08）：额度耗尽时返回 403，请求根本没到模型。
ACCOUNT_QUOTA_403 = {"error": {"type": "i_token_error",
                               "message": "token quota is not enough, token remain quota: $0.005232, "
                                          "need quota: $0.013016 (request id: 202610081125159387837528268d9d6iAwEfaKo)"}}


class Fable5TransientTests(unittest.IsolatedAsyncioTestCase):
    """瞬时故障（上游 5xx / 连接错误）：先重试，重试耗尽后按 blocked 收口。"""

    async def _runner(self, queue, *, max_retries=2, raise_errors=None):
        queue = list(queue)
        errors = list(raise_errors or [])
        requests = []

        async def handler(request):
            requests.append(json.loads(request.content or b"{}"))
            if errors:
                raise errors.pop(0)
            status, body = queue.pop(0)
            return httpx.Response(status, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        config = Fable5Config("key", max_retries=max_retries, retry_backoff_seconds=0.0)
        return Fable5Runner(config, client), client, requests

    async def _run_one(self, runner, case_id, priority="P0"):
        return (await runner.run(("claude-fable-5",), (priority,), {case_id}))[0]

    async def test_5xx_is_retried_then_succeeds(self):
        runner, client, requests = await self._runner([(503, GATEWAY_503), (200, _msg())])
        try:
            result = await self._run_one(runner, "FABLE5-B01", priority="BASIC")
            self.assertEqual(len(requests), 2, "503 应触发一次重试")
            self.assertEqual(runner.transient_events, [])
        finally:
            await client.aclose()
        self.assertEqual(result.status, "passed")

    async def test_5xx_exhausted_becomes_blocked_not_failed(self):
        runner, client, requests = await self._runner([(503, GATEWAY_503)] * 3)
        try:
            result = await self._run_one(runner, "FABLE5-E01")
            self.assertEqual(len(requests), 3, "max_retries=2 → 共 3 次尝试")
        finally:
            await client.aclose()
        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.passed)
        self.assertIn("瞬时故障", result.details)
        self.assertIn("已重试 2 次仍失败", result.details)
        self.assertEqual(result.observations["transient_failures"][0]["http_status"], 503)

    async def test_connection_error_exhausted_becomes_blocked(self):
        error = httpx.ConnectError("connection reset")
        runner, client, requests = await self._runner([], raise_errors=[error, error, error])
        try:
            result = await self._run_one(runner, "FABLE5-B01", priority="BASIC")
            self.assertEqual(len(requests), 3)
        finally:
            await client.aclose()
        self.assertEqual(result.status, "blocked")
        self.assertIn("ConnectError", result.details)

    async def test_retries_disabled_records_single_attempt(self):
        runner, client, requests = await self._runner([(503, GATEWAY_503)] * 3, max_retries=0)
        try:
            result = await self._run_one(runner, "FABLE5-E01")
            self.assertEqual(len(requests), 1)
        finally:
            await client.aclose()
        self.assertEqual(result.status, "blocked")
        self.assertIn("未启用重试", result.details)

    async def test_4xx_is_not_retried_and_not_downgraded(self):
        runner, client, requests = await self._runner([(400, {"error": {"message": "bad"}})] * 3)
        try:
            result = await self._run_one(runner, "FABLE5-B01", priority="BASIC")
            self.assertEqual(len(requests), 1, "4xx 是确定性判定，不重试")
        finally:
            await client.aclose()
        self.assertEqual(result.status, "failed")

    async def test_account_quota_rejection_becomes_blocked_with_self_explanatory_reason(self):
        """账号额度不足（实测 i-token.ai 403 `token quota is not enough`）与模型能力无关：
        必须记 blocked 并写明原因，不能把整批用例记成 failed。"""
        runner, client, requests = await self._runner([(403, ACCOUNT_QUOTA_403)] * 3)
        try:
            result = await self._run_one(runner, "FABLE5-B01", priority="BASIC")
            self.assertEqual(len(requests), 1, "额度拒绝是确定性结论，不重试")
        finally:
            await client.aclose()
        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.passed)
        self.assertIn("平台级前提不满足", result.details)
        self.assertIn("quota", result.details.lower())
        self.assertEqual(result.observations["account_limitations"][0]["http_status"], 403)

    async def test_account_quota_reason_is_attached_to_case_level_blocked(self):
        """用例自身已收口 blocked 时（如 AUTH-F04「探测失败：HTTP 403」），
        也要把账号级原因写进 details，否则运维无法区分额度问题与平台缺陷。"""
        runner, client, _ = await self._runner([(403, ACCOUNT_QUOTA_403)] * 3)
        try:
            result = await self._run_one(runner, "AUTH-F04", priority="AUTH")
        finally:
            await client.aclose()
        self.assertEqual(result.status, "blocked")
        self.assertIn("平台级前提不满足", result.details)
        self.assertIn("quota", result.details.lower())
        self.assertIn("原用例收口", result.details)

    async def test_ordinary_400_keeps_failed_verdict(self):
        """只有显式账号/权限文案才改判 blocked；普通 400 仍是 failed（防兜底过宽）。"""
        runner, client, _ = await self._runner([(400, {"error": {"message": "messages: field required"}})] * 3)
        try:
            result = await self._run_one(runner, "FABLE5-B01", priority="BASIC")
        finally:
            await client.aclose()
        self.assertEqual(result.status, "failed")
        self.assertEqual(runner.account_events, [])

    async def test_group_permission_denial_is_blocked(self):
        """实测 i-token.ai 另一种 403：令牌无权访问目标分组（"无权访问 ccmax 分组"）。"""
        body = {"error": {"code": "", "type": "i_token_error", "message": "无权访问 ccmax 分组"}}
        runner, client, _ = await self._runner([(403, body)] * 3)
        try:
            result = await self._run_one(runner, "FABLE5-B01", priority="BASIC")
        finally:
            await client.aclose()
        self.assertEqual(result.status, "blocked")
        self.assertIn("无权访问", result.details)

    async def test_stream_error_body_falls_back_to_raw_text(self):
        """流式请求的 response_body 不会被解析（只有 raw_response_body 带原文），
        不做这层兜底，流式用例的额度/权限拒绝就会退化成无原因的 403。"""
        runner, client, _ = await self._runner([(403, ACCOUNT_QUOTA_403)] * 3)
        try:
            result = await self._run_one(runner, "AUTH-F11", priority="AUTH")
        finally:
            await client.aclose()
        self.assertEqual(result.status, "blocked")
        self.assertIn("quota", result.details.lower())
        self.assertIn("原用例收口", result.details)

    async def test_mid_case_transient_downgrades_failed_case(self):
        runner, client, _ = await self._runner([(503, GATEWAY_503)] * 3 + [(200, _msg())] * 5)
        try:
            result = await self._run_one(runner, "FABLE5-E05")  # effort_levels：5 档循环
        finally:
            await client.aclose()
        self.assertEqual(result.status, "blocked")
        self.assertIn("瞬时故障", result.details)

    async def test_passed_case_is_never_downgraded(self):
        runner, client, _ = await self._runner([(503, GATEWAY_503)] * 3)

        async def fake_basic(model):
            await runner.request("messages", {"model": model, "max_tokens": 8,
                                              "messages": [{"role": "user", "content": "hi"}]})
            return True, "通过", {}

        runner.case_basic = fake_basic
        try:
            result = await self._run_one(runner, "FABLE5-B01", priority="BASIC")
        finally:
            await client.aclose()
        self.assertEqual(len(runner.transient_events), 1)
        self.assertEqual(result.status, "passed")

    async def test_definitive_model_not_found_is_not_retried(self):
        runner, client, requests = await self._runner([(503, MODEL_NOT_FOUND_503)] * 3)
        try:
            result = await self._run_one(runner, "FABLE5-C10", priority="P2")
            self.assertEqual(len(requests), 1, "确定性错误不得触发重试")
            self.assertEqual(runner.transient_events, [])
        finally:
            await client.aclose()
        self.assertEqual(result.status, "passed")

    async def test_transient_reason_is_self_explanatory(self):
        from fable5.core import _transient_reason
        text = _transient_reason({"path": "messages", "attempts": 3, "http_status": 503,
                                  "transport_error": None, "message": "boom"}, 1)
        for fragment in ("HTTP 503", "已重试 2 次仍失败", "boom", "blocked", "重跑"):
            self.assertIn(fragment, text)


class Fable5AdmissionExitCodeTests(unittest.IsolatedAsyncioTestCase):
    """退出码必须以 finalize() 之后的磁盘结论为准（与 GPT-5.6 专项同一回归）。"""

    async def _run_main(self, statuses):
        from fable5 import runner as runner_module
        from fable5.core import CaseResult

        created = {}

        class FakeRunner:
            def __init__(self, config, audit_dir=None):
                self.audit_dir = audit_dir
                created["dir"] = audit_dir

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def run(self, models, priorities, faces=None):
                # 证据契约：结果目录必须有非空请求证据，否则整批不准入。
                self.audit_dir.mkdir(parents=True, exist_ok=True)
                (self.audit_dir / "requests.jsonl").write_text(
                    json.dumps({"case_id": "FABLE5-B01", "http_status": 200}) + "\n", encoding="utf-8")
                return [CaseResult("FABLE5-B01", "基础对话", models[0], False, status, "说明")
                        for status in statuses]

        env = {"FABLE_API_KEY": "key", "FABLE_BASE_URL": "https://api.anthropic.com", "FABLE_MODEL": "claude-fable-5"}
        saved = {k: os.environ.get(k) for k in env}
        original = runner_module.Fable5Runner
        with tempfile.TemporaryDirectory() as tmp:
            try:
                runner_module.Fable5Runner = FakeRunner
                os.environ.update(env)
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = await runner_module.main_async(["FABLE5_P0", "--output-root", tmp])
            finally:
                runner_module.Fable5Runner = original
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            disk = json.loads((Path(created["dir"]) / "summary.json").read_text(encoding="utf-8"))
        return rc, disk

    async def test_blocked_only_batch_is_not_admitted(self):
        rc, disk = await self._run_main(["blocked", "blocked"])
        self.assertFalse(disk["admission_passed"])
        self.assertEqual(disk["failed"], 0)
        self.assertEqual(rc, 2)

    async def test_blocked_does_not_block_when_effective_verdict_exists(self):
        """2026-10-10 口径：有有效结论时 blocked 只记录、不阻断准入。"""
        rc, disk = await self._run_main(["passed", "blocked"])
        self.assertTrue(disk["admission_passed"])
        self.assertEqual(disk["blocked"], 1)
        self.assertEqual(rc, 0)

    async def test_all_passed_batch_still_admits(self):
        rc, disk = await self._run_main(["passed"])
        self.assertTrue(disk["admission_passed"])
        self.assertEqual(rc, 0)

    async def test_failed_batch_keeps_exit_code_2(self):
        rc, disk = await self._run_main(["failed"])
        self.assertFalse(disk["admission_passed"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
