import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

import httpx

from gpt56.cases import CASES
from gpt56.core import (
    CASE_SPECS,
    GPT6_MODELS,
    GPT56Config,
    GPT56Runner,
    GPT61_FLAGSHIP,
    MODEL_TIERS,
    MODEL_TOKENS,
    PELICAN_MAX_OUTPUT_TOKENS,
    PELICAN_PROMPT,
    _cache_write_tokens,
    _cached_input_tokens,
    _declared_effort_values,
    _output_text,
    _reasoning_tokens,
    _variant_of,
    summarize_results,
    tier_for_model,
    write_report,
)
from gpt56.runner import config_from_env


class GPT56StaticTests(unittest.TestCase):
    def test_ui_suites(self):
        self.assertEqual(set(CASES), {"GPT56_P0", "GPT56_P1", "GPT56_P2", "GPT56_FINGERPRINT", "GPT56_PELICAN",
                                      "GPT61_P0", "GPT61_P1", "GPT61_P2", "GPT61_FINGERPRINT", "GPT61_PELICAN",
                                      "GPT6_P0", "GPT6_P1", "GPT6_P2", "GPT6_FINGERPRINT", "GPT6_PELICAN"})
        self.assertEqual(CASES["GPT56_P0"].request_count, 22)
        self.assertEqual(CASES["GPT56_P1"].request_count, 9)
        self.assertEqual(CASES["GPT56_P2"].request_count, 15)
        self.assertEqual(CASES["GPT56_FINGERPRINT"].request_count, 11)
        self.assertEqual(CASES["GPT61_P0"].request_count, 19)
        self.assertEqual(CASES["GPT61_P1"].request_count, 6)
        self.assertEqual(CASES["GPT61_P2"].request_count, 11)
        self.assertEqual(CASES["GPT61_FINGERPRINT"].request_count, 12)
        self.assertEqual(CASES["GPT6_P0"].request_count, 19)
        self.assertEqual(CASES["GPT6_P1"].request_count, 6)
        self.assertEqual(CASES["GPT6_P2"].request_count, 11)
        self.assertEqual(CASES["GPT6_FINGERPRINT"].request_count, 12)
        # 产物类：三档各一条，共用 pelican_svg handler（提示词与判据跨档一致）。
        self.assertEqual(CASES["GPT56_PELICAN"].request_count, 1)
        self.assertEqual(CASES["GPT61_PELICAN"].request_count, 1)
        self.assertEqual(CASES["GPT6_PELICAN"].request_count, 1)
        self.assertEqual({spec.priority for spec in CASE_SPECS}, {"P0", "P1", "P2", "AUTH", "PELICAN"})
        self.assertEqual(len(CASE_SPECS), 156)
        self.assertEqual(len({spec.case_id for spec in CASE_SPECS}), 156)
        # 三个档位组的元数据 model_variant 必须与 CASE_SPECS 的 variant 一一对应。
        for name, case in CASES.items():
            expected = "gpt6.1" if name.startswith("GPT61_") else "gpt6" if name.startswith("GPT6_") else "gpt5.6"
            self.assertEqual(case.model_variant, expected, name)

    def test_variant_partition_matches_suite_counts(self):
        """三个档位组的 CASE_SPECS 必须严格分区，条数与 cases.py 元数据一致。"""
        by_variant: dict[str, list[str]] = {}
        for spec in CASE_SPECS:
            by_variant.setdefault(spec.variant, []).append(spec.case_id)
        self.assertEqual(len(by_variant["gpt5.6"]), 58)
        self.assertEqual(len(by_variant["gpt6.1"]), 49)
        self.assertEqual(len(by_variant["gpt6"]), 49)
        # 每个套件的实际 priority 条数与展示元数据 request_count 必须自洽。
        counts = {"gpt5.6": {"P0": 22, "P1": 9, "P2": 15, "AUTH": 11, "PELICAN": 1},
                  "gpt6.1": {"P0": 19, "P1": 6, "P2": 11, "AUTH": 12, "PELICAN": 1},
                  "gpt6": {"P0": 19, "P1": 6, "P2": 11, "AUTH": 12, "PELICAN": 1}}
        for spec in CASE_SPECS:
            counts[spec.variant][spec.priority] -= 1
        for variant, remaining in counts.items():
            self.assertEqual(remaining, {k: 0 for k in remaining}, variant)
        # GPT61-* / AUTH-G61-* → gpt6.1，GPT6-* / AUTH-G6-* → gpt6，且不存在串档。
        for spec in CASE_SPECS:
            self.assertEqual(_variant_of(spec.case_id), spec.variant, spec.case_id)
        # GPT6 家族必须是 GPT61 的**子集复用**：同一套 handler，期望值由档位画像解析。
        gpt6_handlers = {s.handler for s in CASE_SPECS if s.variant == "gpt6"}
        gpt61_handlers = {s.handler for s in CASE_SPECS if s.variant == "gpt6.1"}
        self.assertTrue(gpt6_handlers <= gpt61_handlers,
                        f"GPT6 出现 6.1 未覆盖的 handler：{sorted(gpt6_handlers - gpt61_handlers)}")

    def test_config_urls(self):
        # GPT-5.6 无 beta/anthropic 特殊端点，全部挂在 openai 家族根（.../v1）下。
        runner = GPT56Runner(GPT56Config("key", "https://api.openai.com/v1"))
        self.assertEqual(runner._url("chat/completions"), "https://api.openai.com/v1/chat/completions")
        self.assertEqual(runner._url("responses"), "https://api.openai.com/v1/responses")

    def test_config_from_env_normalizes_host_only_base_url(self, ):
        import os
        env = {"GPT_API_KEY": "key", "GPT_BASE_URL": "https://gw.example.com", "GPT_MODEL": "gpt-5.6"}
        original = {k: os.environ.get(k) for k in env}
        try:
            os.environ.update(env)
            config = config_from_env()
            self.assertEqual(config.base_url, "https://gw.example.com/v1")
        finally:
            for k, v in original.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_config_from_env_keeps_openai_family_protocol_ids(self):
        """按族放行、并保留协议 id —— 否则千帆/火山云会被改写成 .../v1 这个不存在的路径。

        回归靶点：以前是 `if protocol not in FAMILIES: protocol = "openai"`，把 family 集合
        当成 protocol id 集合用。百度千帆的 family 也是 "openai"，但 api root 是 /v2，
        一旦被改写成 "openai"，请求就会打到 https://qianfan.baidubce.com/v1/chat/completions。
        """
        import os
        from gpt56.runner import resolve_protocol
        cases = {
            None: "openai",                      # 缺省
            "qianfan": "qianfan",                # openai 族 + /v2 根，必须原样保留
            "openai_responses": "openai_responses",
            "anthropic": "openai",               # 别族 → 回落 openai
            "no-such-protocol": "openai",        # 未知 id → 回落 openai
        }
        for raw, expected in cases.items():
            self.assertEqual(resolve_protocol(raw), expected)
        env = {"GPT_API_KEY": "key", "GPT_BASE_URL": "https://qianfan.baidubce.com",
               "GPT_MODEL": "gpt-5.6", "GPT_PROTOCOL": "qianfan"}
        original = {k: os.environ.get(k) for k in env}
        try:
            os.environ.update(env)
            self.assertEqual(config_from_env().base_url, "https://qianfan.baidubce.com/v2")
        finally:
            for k, v in original.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_model_tokens_cover_alias_and_tiers(self):
        for token in ("gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
            self.assertIn(token, MODEL_TOKENS)

    def test_extractors(self):
        body = {"output": [
            {"type": "reasoning", "summary": [{"text": "s"}], "encrypted_content": "enc"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "答案"}]},
        ], "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "output_tokens_details": {"reasoning_tokens": 7}, "input_tokens_details": {"cached_tokens": 3}}}
        self.assertEqual(_output_text(body), "答案")
        self.assertEqual(_reasoning_tokens(body), 7)
        self.assertEqual(_cached_input_tokens(body), 3)
        self.assertEqual(_cache_write_tokens({"usage": {"input_tokens_details": {"cache_write_tokens": 9}}}), 9)
        self.assertEqual(_cache_write_tokens({"usage": {}}), 0)

    def test_summary_shape(self):
        from gpt56.core import CaseResult
        summary = summarize_results([CaseResult("1", "a", "m", True)])
        self.assertTrue(summary["all_passed"])
        self.assertEqual(summary["results"][0]["case_id"], "1")


class GPT56OutcomeTests(unittest.IsolatedAsyncioTestCase):
    async def _runner(self, responses, max_retries=0):
        """handler 级判定测试：默认关闭瞬时故障重试，保持"一问一答"的脚本语义。

        重试与 blocked 收口（run 层）由 GPT56TransientTests 专门覆盖。
        """
        queue = list(responses)
        requests = []

        async def handler(request):
            # GET（raw_request）无请求体；队列按"第 n 次请求"语义弹出，与方法无关。
            requests.append(json.loads(request.content or b"{}"))
            status, body = queue.pop(0)
            return httpx.Response(status, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        config = GPT56Config("key", max_retries=max_retries, retry_backoff_seconds=0.0)
        return GPT56Runner(config, client), client, requests

    @staticmethod
    def _normalize(outcome):
        """handler 直接调用返回 3/4 元组；统一成 (passed, details, observations, status)。"""
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

    async def test_basic_identity_token_mismatch_fails(self):
        ok = {"model": "gpt-5.6-sol", "choices": [{"message": {"role": "assistant", "content": "OK"}}]}
        ghost = {"model": "gpt-4o-mini", "choices": [{"message": {"role": "assistant", "content": "OK"}}]}
        outcome, _ = await self._one([(200, ok)], "case_basic", "gpt-5.6")
        self.assertTrue(outcome[0])
        outcome, _ = await self._one([(200, ghost)], "case_basic", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_default_effort_missing_field_is_warning(self):
        runner, client, _ = await self._runner([(200, {"output": [{"type": "message", "content": [{"type": "output_text", "text": "391"}]}]})])
        try:
            passed, _, _, status = await runner.case_default_effort("gpt-5.6")
        finally:
            await client.aclose()
        self.assertTrue(passed)
        self.assertEqual(status, "warning")

    async def test_default_effort_medium_passes(self):
        body = {"reasoning": {"effort": "medium", "context": "auto"}, "output": [], "usage": {}}
        outcome, _ = await self._one([(200, body)], "case_default_effort", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_effort_none_missing_reasoning_tokens_is_warning(self):
        body = {"output": [], "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
        outcome, _ = await self._one([(200, body)], "case_effort_none", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    async def test_effort_none_with_reasoning_items_fails(self):
        body = {"output": [{"type": "reasoning", "summary": []}], "usage": {"output_tokens_details": {"reasoning_tokens": 10}}}
        outcome, _ = await self._one([(200, body)], "case_effort_none", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_cache_hit_blocked_when_not_observed(self):
        empty_usage = {"choices": [{"message": {"role": "assistant", "content": "OK"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        runner, client, _ = await self._runner([(200, empty_usage), (200, empty_usage)])
        try:
            passed, _, _, status = await runner.case_cache_hit("gpt-5.6")
        finally:
            await client.aclose()
        self.assertFalse(passed)
        self.assertEqual(status, "blocked")

    @staticmethod
    def _chat_body(cached: int, written: int = 0):
        return {"choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101,
                          "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written}}}

    async def test_cache_hit_passes_on_increment_over_baseline(self):
        # 首跳自带基线命中（如上游注入的固定前缀），但第二跳增量显著 → 构造前缀确实被缓存
        outcome, _ = await self._one([(200, self._chat_body(3840)), (200, self._chat_body(11000))], "case_cache_hit", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(outcome[2]["delta_cached_tokens"], 7160)

    async def test_cache_hit_baseline_only_is_blocked_not_passed(self):
        # 真实批次回归（mux.la）：两跳 cached_tokens 同为注入前缀的 3840、增量为 0，不得判通过
        outcome, _ = await self._one([(200, self._chat_body(3840)), (200, self._chat_body(3840))], "case_cache_hit", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")
        self.assertEqual(outcome[2]["delta_cached_tokens"], 0)
        self.assertEqual(outcome[2]["baseline_cached_tokens"], 3840)

    async def test_cache_hit_tiny_increment_is_warning(self):
        outcome, _ = await self._one([(200, self._chat_body(0)), (200, self._chat_body(128))], "case_cache_hit", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "warning")

    async def test_unknown_model_routed_success_is_failed(self):
        outcome, _ = await self._one([(200, {"output": [], "usage": {}})], "case_unknown_model", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_unknown_model_structured_404_passes(self):
        body = {"error": {"message": "model not found", "type": "invalid_request_error"}}
        outcome, _ = await self._one([(404, body)], "case_unknown_model", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_unknown_model_gateway_503_model_not_found_passes(self):
        # 真实批次回归（mux.la）：New API 用 503 + code=model_not_found 承载"模型不存在"
        body = {"error": {"code": "model_not_found", "type": "new_api_error",
                          "message": "No available channel for model gpt-5.6-nonexistent-deadbeef under group muxla_company"}}
        outcome, _ = await self._one([(503, body)], "case_unknown_model", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertIn("503", outcome[1])

    async def test_unknown_model_5xx_without_model_evidence_fails(self):
        # 渠道整体故障 / 非结构化错误不得被当成"模型校验成立"
        for body in ({"error": {"message": "upstream timeout", "type": "server_error"}},
                     {"error": {"code": "channel_error", "message": "No available channel for model gpt-5.6-sol"}}):
            outcome, _ = await self._one([(503, body)], "case_unknown_model", "gpt-5.6")
            self.assertFalse(outcome[0])
            self.assertEqual(outcome[3], "failed")

    _ACCOUNT_LIMIT = {"error": {"message": "previous_response_id requires an OpenAI API-key account for HTTP requests",
                                "type": "invalid_request_error"}}

    async def test_previous_response_account_limit_is_blocked(self):
        first = {"id": "resp_1", "output": [{"type": "message", "content": [{"type": "output_text", "text": "已记住"}]}]}
        outcome, _ = await self._one([(200, first), (400, self._ACCOUNT_LIMIT)], "case_previous_response_state", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")
        self.assertIn("账号层", outcome[1])

    async def test_previous_response_plain_400_is_failed(self):
        # 普通 400（无账号层限制特征）不得被兜底成 blocked
        first = {"id": "resp_1", "output": []}
        outcome, _ = await self._one([(200, first), (400, {"error": {"message": "bad request"}})], "case_previous_response_state", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_context_all_turns_account_limit_is_blocked(self):
        first = {"id": "resp_1", "output": []}
        outcome, _ = await self._one([(200, first), (400, self._ACCOUNT_LIMIT)], "case_context_all_turns", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")

    async def test_auth_random_memory_account_limit_is_blocked(self):
        first = {"id": "resp_1", "output": []}
        outcome, _ = await self._one([(200, first), (400, self._ACCOUNT_LIMIT)], "case_auth_random_memory", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")

    async def test_auth_reasoning_summary_without_items_is_blocked(self):
        # 与 GPT56-05 统一口径：200 但拿不到 reasoning 产物 → 指纹取证前提不满足（blocked），不判模型失败
        body = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "391"}]}], "usage": {}}
        outcome, _ = await self._one([(200, body)], "case_auth_reasoning_summary", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")
        self.assertIn("GPT56-05", outcome[1])

    async def test_negative_cases_block_on_gateway_404(self):
        # 404 是链路/配置问题：非法 messages 的"参数被拒绝"不能由 404 冒充
        body = {"error": {"message": "no route", "type": "not_found_error"}}
        runner, client, _ = await self._runner([(404, body)])
        try:
            passed, _, _, status = await runner.case_messages_invalid("gpt-5.6")
        finally:
            await client.aclose()
        self.assertFalse(passed)
        self.assertEqual(status, "blocked")

    async def test_temperature_lenient_200_is_warning(self):
        runner, client, _ = await self._runner([(200, {"choices": [{"message": {"role": "assistant", "content": "OK"}}]})])
        try:
            passed, _, _, status = await runner.case_temperature_invalid("gpt-5.6")
        finally:
            await client.aclose()
        self.assertTrue(passed)
        self.assertEqual(status, "warning")

    async def test_encrypted_replay_blocked_without_encrypted_content(self):
        body = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "已记住"}]}], "usage": {}}
        runner, client, _ = await self._runner([(200, body)])
        try:
            passed, _, _, status = await runner.case_encrypted_replay("gpt-5.6")
        finally:
            await client.aclose()
        self.assertFalse(passed)
        self.assertEqual(status, "blocked")

    async def test_effort_contrast_silent_swallow_is_failed(self):
        body = {"output": [], "usage": {"output_tokens_details": {"reasoning_tokens": 0}}}
        outcome, _ = await self._one([(200, body), (200, body)], "case_auth_effort_contrast", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_tool_loop_blocked_when_upstream_rejects(self):
        runner, client, _ = await self._runner([(400, {"error": {"message": "bad"}})])
        try:
            passed, _, _, status = await runner.case_tool_loop("gpt-5.6")
        finally:
            await client.aclose()
        self.assertFalse(passed)
        self.assertEqual(status, "blocked")

    async def test_truncation_detects_incomplete(self):
        body = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
        runner, client, _ = await self._runner([(200, body)])
        try:
            passed, _, _ = await runner.case_truncation("gpt-5.6")
        finally:
            await client.aclose()
        self.assertTrue(passed)

    async def test_pro_mode_rejection_is_blocked(self):
        runner, client, _ = await self._runner([(400, {"error": {"message": "pro not available"}})])
        try:
            passed, _, _, status = await runner.case_pro_mode("gpt-5.6")
        finally:
            await client.aclose()
        self.assertFalse(passed)
        self.assertEqual(status, "blocked")


# ---------------------------------------------------------------------------
# 缺口补充用例（2026-09-28 实现，设计见 docs/GPT5.6测试缺口分析与补充用例设计.md）
# ---------------------------------------------------------------------------


def _resp(model: str = "gpt-5.6", *, text: str = "OK", usage: dict | None = None, **extra: Any) -> dict:
    body = {"id": "resp_" + "a" * 24, "object": "response", "created_at": 1, "model": model,
            "status": "completed", "output": [{"type": "message", "role": "assistant",
                                               "content": [{"type": "output_text", "text": text}]}],
            "usage": usage or {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11}}
    body.update(extra)
    return body


class GPT56GapTests(unittest.IsolatedAsyncioTestCase):
    """第一批（34/35/41/54/55/56/57/61 + AUTH-G07/G08/G11/G12）与
    第二批（36/37/40/49/53/AUTH-G10）新用例的关键分支。"""

    async def _one(self, responses, handler_name, *args):
        runner, client, _ = await GPT56OutcomeTests._runner(self, responses)
        try:
            return GPT56OutcomeTests._normalize(await getattr(runner, handler_name)(*args))
        finally:
            await client.aclose()

    # ---- GPT56-34 三档模型 ----
    async def test_tier_models_pass(self):
        queue = [(200, _resp(t)) for t in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")]
        outcome = await self._one(queue, "case_tier_models", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_tier_mismatch_is_failed(self):
        # 请求 terra 却返回 sol：档位路由错配，能力与计价口径不符
        queue = [(200, _resp("gpt-5.6-sol")), (200, _resp("gpt-5.6-sol")), (200, _resp("gpt-5.6-luna"))]
        outcome = await self._one(queue, "case_tier_models", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_tier_unavailable_is_warning(self):
        queue = [(200, _resp("gpt-5.6-sol")),
                 (404, {"error": {"code": "model_not_found", "message": "The model `gpt-5.6-terra` does not exist"}}),
                 (200, _resp("gpt-5.6-luna"))]
        outcome = await self._one(queue, "case_tier_models", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    # ---- GPT56-35 别名路由 ----
    async def test_alias_routes_to_flagship(self):
        outcome = await self._one([(200, _resp("gpt-5.6-sol"))], "case_alias_flagship", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_alias_echo_is_warning(self):
        outcome = await self._one([(200, _resp("gpt-5.6"))], "case_alias_flagship", "gpt-5.6")
        self.assertEqual(outcome[3], "warning")

    async def test_alias_to_non_flagship_is_failed(self):
        outcome = await self._one([(200, _resp("gpt-5.6-luna"))], "case_alias_flagship", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    # ---- GPT56-36 图像识别 ----
    async def test_image_recognition_passes_with_token_evidence(self):
        img = _resp(text="上：红，下：蓝", usage={"input_tokens": 900})
        base = _resp(usage={"input_tokens": 130})
        outcome = await self._one([(200, img), (200, base)], "case_image_recognition", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_image_without_token_delta_is_warning(self):
        """识别"对"但 tokens 不高于基线：取证不足 ⇒ warning 留痕，不再判 failed。

        2026-10-10：判定改为"严格高于基线"（原为 >1.5×）。中转会注入固定 instructions 前缀
        （mux.la 实测占 input 的 ≈99%），图片只贡献个位数 token，比例阈值恒不可满足——
        再判 failed 会把"模型确实看图"记成模型缺陷。
        """
        img = _resp(text="上：红，下：蓝", usage={"input_tokens": 130})
        base = _resp(usage={"input_tokens": 130})
        outcome = await self._one([(200, img), (200, base)], "case_image_recognition", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    async def test_image_delta_just_above_baseline_passes(self):
        """旁证只要求"严格高于"：数值差 1 也成立（两条请求共享同一注入前缀，差值只能来自图片）。"""
        img = _resp(text="上：红，下：蓝", usage={"input_tokens": 131})
        base = _resp(usage={"input_tokens": 130})
        outcome = await self._one([(200, img), (200, base)], "case_image_recognition", "gpt-5.6")
        self.assertEqual(outcome[3], "passed")

    async def test_image_wrong_answer_is_failed(self):
        img = _resp(text="请上传图片", usage={"input_tokens": 900})
        base = _resp(usage={"input_tokens": 130})
        outcome = await self._one([(200, img), (200, base)], "case_image_recognition", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_image_rejection_is_blocked(self):
        outcome = await self._one([(400, {"error": {"message": "image input not supported"}})], "case_image_recognition", "gpt-5.6")
        self.assertEqual(outcome[3], "blocked")

    # ---- GPT56-37 detail 档位 ----
    async def test_detail_tiers_pass_with_lenient_invalid(self):
        ok = (200, _resp(text="上：红，下：蓝"))
        outcome = await self._one([ok, ok, ok, ok, (200, _resp(text="上：红，下：蓝"))], "case_image_detail_tiers", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")  # 非法 detail=ultra 被静默接受 → 留痕

    async def test_detail_tier_rejected_is_blocked(self):
        ok = (200, _resp(text="上：红，下：蓝"))
        queue = [ok, ok, (400, {"error": {"message": "invalid detail"}}), ok, ok]
        outcome = await self._one(queue, "case_image_detail_tiers", "gpt-5.6")
        self.assertEqual(outcome[3], "blocked")

    # ---- GPT56-40 后台模式 ----
    async def test_background_poll_to_completed(self):
        queued = {"id": "resp_" + "b" * 24, "object": "response", "status": "queued", "output": []}
        completed = {"id": "resp_" + "b" * 24, "object": "response", "status": "completed",
                     "output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}]}
        outcome = await self._one([(200, queued), (200, completed)], "case_background_mode", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_background_sync_completed_is_warning(self):
        outcome = await self._one([(200, {"id": "resp_x", "status": "completed", "output": []})], "case_background_mode", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    async def test_background_rejected_is_blocked(self):
        outcome = await self._one([(400, {"error": {"message": "background not supported"}})], "case_background_mode", "gpt-5.6")
        self.assertEqual(outcome[3], "blocked")

    # ---- GPT56-41 响应对象结构 ----
    async def test_response_object_contract_passes(self):
        outcome = await self._one([(200, _resp())], "case_response_object_contract", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_response_object_wrong_structure_is_failed(self):
        outcome = await self._one([(200, {"id": "x", "output": []})], "case_response_object_contract", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_response_object_missing_detail_is_warning(self):
        body = _resp()
        body.pop("created_at")
        outcome = await self._one([(200, body)], "case_response_object_contract", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    # ---- GPT56-49 显式缓存断点 ----
    async def test_explicit_breakpoint_passes_on_delta_or_write(self):
        first = {"usage": {"input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 800}}}
        second = {"usage": {"input_tokens_details": {"cached_tokens": 6000, "cache_write_tokens": 0}}}
        outcome = await self._one([(200, first), (200, second)], "case_explicit_cache_breakpoint", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_explicit_breakpoint_no_cache_fields_is_warning(self):
        body = {"usage": {"input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}}}
        outcome = await self._one([(200, body), (200, body)], "case_explicit_cache_breakpoint", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    async def test_explicit_breakpoint_rejected_is_blocked(self):
        outcome = await self._one([(400, {"error": {"message": "explicit cache unsupported"}})], "case_explicit_cache_breakpoint", "gpt-5.6")
        self.assertEqual(outcome[3], "blocked")

    # ---- GPT56-53 multi-agent ----
    async def test_multi_agent_accepted_passes(self):
        outcome = await self._one([(200, _resp())], "case_multi_agent", "gpt-5.6")
        self.assertTrue(outcome[0])

    async def test_multi_agent_rejected_is_blocked(self):
        outcome = await self._one([(403, {"error": {"message": "beta required"}})], "case_multi_agent", "gpt-5.6")
        self.assertEqual(outcome[3], "blocked")

    # ---- GPT56-54 鉴权契约 ----
    async def test_invalid_key_401_passes(self):
        outcome = await self._one([(401, {"error": {"message": "Incorrect API key provided", "type": "invalid_request_error"}})], "case_invalid_key_auth", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_invalid_key_200_is_failed(self):
        # 鉴权完全失效是严重项，必须判 failed
        outcome = await self._one([(200, _resp())], "case_invalid_key_auth", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_invalid_key_403_is_warning(self):
        outcome = await self._one([(403, {"error": {"message": "forbidden"}})], "case_invalid_key_auth", "gpt-5.6")
        self.assertEqual(outcome[3], "warning")

    # ---- GPT56-55 错误对象结构 ----
    async def test_error_object_four_fields_passes(self):
        err = {"error": {"message": "Invalid value", "type": "invalid_request_error", "param": "text.verbosity", "code": "invalid_value"}}
        outcome = await self._one([(400, err)], "case_error_object_contract", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_error_object_missing_param_code_is_warning(self):
        err = {"error": {"message": "Invalid value", "type": "invalid_request_error"}}
        outcome = await self._one([(400, err)], "case_error_object_contract", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    async def test_error_object_unstructured_is_failed(self):
        outcome = await self._one([(400, "plain gateway error")], "case_error_object_contract", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    # ---- GPT56-56 /models ----
    async def test_models_list_with_tiers_passes(self):
        body = {"object": "list", "data": [{"id": t} for t in ("gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")]}
        outcome = await self._one([(200, body)], "case_models_list", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_models_list_missing_tier_is_warning(self):
        body = {"object": "list", "data": [{"id": "gpt-5.6-sol"}]}
        outcome = await self._one([(200, body)], "case_models_list", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    async def test_models_list_404_is_blocked(self):
        outcome = await self._one([(404, {"error": {"message": "unknown path"}})], "case_models_list", "gpt-5.6")
        self.assertEqual(outcome[3], "blocked")

    # ---- GPT56-57 input_tokens 端点 ----
    async def test_input_tokens_endpoint_cross_validated(self):
        count = {"input_tokens": 500}
        real = _resp(usage={"input_tokens": 520, "output_tokens": 1, "total_tokens": 521})
        outcome = await self._one([(200, count), (200, real)], "case_input_tokens_endpoint", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_input_tokens_endpoint_404_is_blocked(self):
        outcome = await self._one([(404, {"error": {"message": "not found"}})], "case_input_tokens_endpoint", "gpt-5.6")
        self.assertEqual(outcome[3], "blocked")

    async def test_input_tokens_endpoint_deviation_is_warning(self):
        count = {"input_tokens": 100}
        real = _resp(usage={"input_tokens": 500, "output_tokens": 1, "total_tokens": 501})
        outcome = await self._one([(200, count), (200, real)], "case_input_tokens_endpoint", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    # ---- AUTH-G07 注入识别 ----
    async def test_auth_injection_clean_passes(self):
        outcome = await self._one([(200, {"instructions": None, "usage": {"input_tokens": 130}})], "case_auth_injection", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_auth_injection_instructions_filled_is_failed(self):
        # 真实批次回归：mux.la 注入 Codex 系统提示词
        body = {"instructions": "You are Codex, a coding agent based on GPT-5...", "usage": {"input_tokens": 4400}}
        outcome = await self._one([(200, body)], "case_auth_injection", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_auth_injection_inflated_tokens_is_failed(self):
        outcome = await self._one([(200, {"instructions": None, "usage": {"input_tokens": 5000}})], "case_auth_injection", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_auth_injection_missing_field_is_warning(self):
        outcome = await self._one([(200, {"usage": {"input_tokens": 130}})], "case_auth_injection", "gpt-5.6")
        self.assertEqual(outcome[3], "warning")

    # ---- AUTH-G08 非标准字段 ----
    async def test_auth_fields_clean_passes(self):
        outcome = await self._one([(200, _resp())], "case_auth_nonstandard_fields", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_auth_fields_relay_markers_fail(self):
        # 真实批次回归：mux.la 响应顶层出现 billing_usage / access_programs
        body = _resp()
        body["billing_usage"] = {"source": "oai_responses"}
        body["access_programs"] = ["codex"]
        outcome = await self._one([(200, body)], "case_auth_nonstandard_fields", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    # ---- AUTH-G10 档位梯度 ----
    async def test_tier_gradient_observable_passes(self):
        luna = _resp("gpt-5.6-luna", text="7006652", usage={"input_tokens": 10, "output_tokens_details": {"reasoning_tokens": 16}})
        sol = _resp("gpt-5.6-sol", text="7006652", usage={"input_tokens": 10, "output_tokens_details": {"reasoning_tokens": 48}})
        outcome = await self._one([(200, luna), (200, sol)], "case_auth_tier_gradient", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_tier_gradient_identical_is_failed(self):
        # 两档 tokens 逐一相等且输出逐字相同 = 疑似同一后端
        same = _resp(text="7006652", usage={"input_tokens": 10, "output_tokens_details": {"reasoning_tokens": 16}})
        outcome = await self._one([(200, same), (200, dict(same))], "case_auth_tier_gradient", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    # ---- AUTH-G11 response id 真实性 ----
    async def test_response_id_authentic_passes(self):
        rid = "resp_" + "0123abcd" * 3  # 24 位 hex
        outcome = await self._one([(200, {"id": rid}), (200, {"id": rid})], "case_auth_response_id_authenticity", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_response_id_chatcmpl_is_failed(self):
        outcome = await self._one([(200, {"id": "chatcmpl-abc123"})], "case_auth_response_id_authenticity", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_response_id_retrieval_404_is_blocked(self):
        rid = "resp_" + "0123abcd" * 3
        outcome = await self._one([(200, {"id": rid}), (404, {"error": {"message": "no response"}})], "case_auth_response_id_authenticity", "gpt-5.6")
        self.assertEqual(outcome[3], "blocked")

    async def test_response_id_bad_format_is_warning(self):
        outcome = await self._one([(200, {"id": "resp_short"})], "case_auth_response_id_authenticity", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    # ---- AUTH-G12 Chat 未被改写 ----
    async def test_chat_contract_clean_passes(self):
        body = {"object": "chat.completion", "id": "chatcmpl-abc", "choices": [{"message": {"role": "assistant", "content": "OK"}}]}
        outcome = await self._one([(200, body)], "case_auth_chat_not_translated", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_chat_translated_is_failed(self):
        # 真实批次回归：mux.la 把 Chat 翻译成 Responses（stop 丢参的根因）
        body = {"object": "response", "id": "resp_" + "a" * 24, "usage": {"billing_usage": {"source": "oai_responses"}}}
        outcome = await self._one([(200, body)], "case_auth_chat_not_translated", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    # ---- GPT56-61 流式语义事件（SSE）----
    async def _sse_one(self, raw_lines):
        lines = list(raw_lines)
        requests = []

        async def handler(request):
            requests.append(json.loads(request.content or b"{}"))
            payload = ("\n\n".join(lines) + "\n\n").encode("utf-8")
            return httpx.Response(200, content=payload, headers={"content-type": "text/event-stream"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        runner = GPT56Runner(GPT56Config("key", max_retries=0, retry_backoff_seconds=0.0), client)
        try:
            return GPT56OutcomeTests._normalize(await runner.case_stream_semantic_events("gpt-5.6"))
        finally:
            await client.aclose()

    async def test_stream_semantic_passes_without_done(self):
        events = [
            {"type": "response.created", "sequence_number": 0},
            {"type": "response.output_text.delta", "delta": "731", "sequence_number": 1},
            {"type": "response.completed", "sequence_number": 2, "response": {"id": "resp_x", "usage": {"total_tokens": 5}}},
        ]
        outcome = await self._sse_one([f"data: {json.dumps(e)}" for e in events])
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_stream_with_done_marker_is_failed(self):
        events = [
            {"type": "response.created", "sequence_number": 0},
            {"type": "response.output_text.delta", "delta": "731", "sequence_number": 1},
            {"type": "response.completed", "sequence_number": 2, "response": {"id": "resp_x"}},
        ]
        lines = [f"data: {json.dumps(e)}" for e in events] + ["data: [DONE]"]
        outcome = await self._sse_one(lines)
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_stream_non_monotonic_sequence_is_failed(self):
        events = [
            {"type": "response.created", "sequence_number": 2},
            {"type": "response.output_text.delta", "delta": "731", "sequence_number": 1},
            {"type": "response.completed", "sequence_number": 3, "response": {"id": "resp_x"}},
        ]
        outcome = await self._sse_one([f"data: {json.dumps(e)}" for e in events])
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")


# 真实批次证据（2026-09-24，results/GPT56_P0-20260924-153058/requests.jsonl）：
# mux.la 中转对 GPT56-12 / GPT56-16 返回的网关自身故障。
MEMORY_OVERLOADED_503 = {"error": {"message": "system memory overloaded (current: 91.5%, threshold: 90%)",
                                   "type": "new_api_error", "param": "", "code": "system_memory_overloaded"}}
OK_CHAT = {"model": "gpt-5.6-sol", "choices": [{"message": {"role": "assistant", "content": "OK"}}]}
MODEL_NOT_FOUND_503 = {"error": {"code": "model_not_found", "type": "new_api_error",
                                 "message": "No available channel for model gpt-5.6-nonexistent-deadbeef under group muxla_company"}}


class GPT56TransientTests(unittest.IsolatedAsyncioTestCase):
    """瞬时故障（上游 5xx / 连接错误）：先重试，重试耗尽后按 blocked 收口。"""

    async def _runner(self, queue, *, max_retries=2, raise_errors=None):
        queue = list(queue)
        errors = list(raise_errors or [])
        requests = []

        async def handler(request):
            requests.append(json.loads(request.content))
            if errors:
                raise errors.pop(0)
            status, body = queue.pop(0)
            return httpx.Response(status, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        config = GPT56Config("key", max_retries=max_retries, retry_backoff_seconds=0.0)
        return GPT56Runner(config, client), client, requests

    async def _run_one(self, runner, case_id, priority="P0"):
        return (await runner.run(("gpt-5.6",), (priority,), {case_id}))[0]

    async def test_5xx_is_retried_then_succeeds(self):
        """首跳 503、重试命中 200：用例应正常判定，且不留下瞬时故障记录。"""
        runner, client, requests = await self._runner([(503, MEMORY_OVERLOADED_503), (200, OK_CHAT)])
        try:
            result = await self._run_one(runner, "GPT56-01")
            self.assertEqual(len(requests), 2, "503 应触发一次重试")
            self.assertEqual(runner.transient_events, [])
        finally:
            await client.aclose()
        self.assertEqual(result.status, "passed")

    async def test_5xx_exhausted_retries_becomes_blocked_not_failed(self):
        """重试耗尽仍 503：原判 failed 的用例改判 blocked，理由需自解释。"""
        runner, client, requests = await self._runner([(503, MEMORY_OVERLOADED_503)] * 3)
        try:
            result = await self._run_one(runner, "GPT56-16")
            self.assertEqual(len(requests), 3, "max_retries=2 → 共 3 次尝试")
        finally:
            await client.aclose()
        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.passed)
        self.assertIn("瞬时故障", result.details)
        self.assertIn("已重试 2 次仍失败", result.details)
        self.assertIn("system memory overloaded", result.details)
        self.assertEqual(result.observations["transient_failures"][0]["http_status"], 503)

    async def test_connection_error_exhausted_becomes_blocked(self):
        """连接错误（ConnectError）同样重试，耗尽后记 blocked。"""
        error = httpx.ConnectError("connection reset")
        runner, client, requests = await self._runner([], raise_errors=[error, error, error])
        try:
            result = await self._run_one(runner, "GPT56-01")
            self.assertEqual(len(requests), 3)
        finally:
            await client.aclose()
        self.assertEqual(result.status, "blocked")
        self.assertIn("ConnectError", result.details)

    async def test_retries_disabled_records_single_attempt(self):
        runner, client, requests = await self._runner([(503, MEMORY_OVERLOADED_503)] * 3, max_retries=0)
        try:
            result = await self._run_one(runner, "GPT56-16")
            self.assertEqual(len(requests), 1)
        finally:
            await client.aclose()
        self.assertEqual(result.status, "blocked")
        self.assertIn("未启用重试", result.details)

    async def test_definitive_model_not_found_is_not_retried(self):
        """503 + code=model_not_found 是确定性结论，不是网关抖动：不重试，用例照常判通过。"""
        runner, client, requests = await self._runner([(503, MODEL_NOT_FOUND_503)] * 3)
        try:
            result = await self._run_one(runner, "GPT56-23", priority="P1")
            self.assertEqual(len(requests), 1, "确定性错误不得触发重试")
            self.assertEqual(runner.transient_events, [])
        finally:
            await client.aclose()
        self.assertEqual(result.status, "passed")

    async def test_4xx_is_not_retried_and_not_downgraded(self):
        runner, client, requests = await self._runner([(400, {"error": {"message": "bad request"}})] * 3)
        try:
            result = await self._run_one(runner, "GPT56-01")
            self.assertEqual(len(requests), 1, "4xx 是确定性判定，不重试")
        finally:
            await client.aclose()
        self.assertEqual(result.status, "failed")

    async def test_mid_case_transient_downgrades_failed_case(self):
        """故障发生在用例中途（最后一跳已 200）也要按 blocked 收口。"""
        runner, client, _ = await self._runner([(503, MEMORY_OVERLOADED_503)] * 3 + [(200, OK_CHAT)] * 2)
        try:
            result = await self._run_one(runner, "GPT56-04")  # effort_levels：5 档循环
        finally:
            await client.aclose()
        self.assertEqual(result.status, "blocked")
        self.assertIn("瞬时故障", result.details)

    async def test_passed_case_is_never_downgraded(self):
        """只有 failed/error 才降级：通过用例即便出现过瞬时故障也不得被改成 blocked。"""
        runner, client, _ = await self._runner([(503, MEMORY_OVERLOADED_503)])

        async def fake_basic(model):
            await runner.request("chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}]})
            return True, "通过", {}

        runner.case_basic = fake_basic  # 模拟"末跳瞬时故障但用例仍通过"的极端组合
        try:
            result = await self._run_one(runner, "GPT56-01")
        finally:
            await client.aclose()
        self.assertEqual(runner.transient_events and len(runner.transient_events), 1)
        self.assertEqual(result.status, "passed")

    async def test_transient_reason_is_self_explanatory(self):
        from gpt56.core import _transient_reason
        text = _transient_reason({"path": "responses", "attempts": 3, "http_status": 503,
                                  "transport_error": None, "message": "boom"}, 1)
        for fragment in ("HTTP 503", "已重试 2 次仍失败", "boom", "blocked", "重跑"):
            self.assertIn(fragment, text)


RESP_OK_61 = {"id": "resp_" + "a" * 24, "object": "response", "status": "completed", "model": "gpt-6.1-sol",
              "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "OK"}]}],
              "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 3}}}
CHAT_OK_61 = {"id": "chatcmpl-" + "b" * 16, "object": "chat.completion", "model": "gpt-6.1-sol",
              "choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}]}
EFFORT_NONE_ERROR = {"error": {
    "message": "Unsupported value: 'none' is not supported with the 'gpt-6.1-sol' model. "
               "Supported values are: 'low', 'medium', 'high', 'xhigh', and 'max'.",
    "type": "invalid_request_error", "param": "reasoning.effort", "code": "unsupported_value"}}
# 中转层常见的"与模型无关的统一七档白名单"：连 6.1/Astra 官方已移除的 none/minimal 也在列
# （2026-10-10 在 mux.la 实测：gpt-6-sol 与 gpt-6.1-sol 返回同一份 Supported values）。
ENUM_UNIFIED_ERROR = {"error": {
    "message": "Invalid value: '__probe_effort__'. Supported values are: 'none', 'minimal', "
               "'low', 'medium', 'high', 'xhigh', and 'max'.",
    "type": "new_api_error", "param": "reasoning.effort", "code": "invalid_value"}}
RESP_OK_6SOL = {**RESP_OK_61, "model": "gpt-6-sol"}
RESP_OK_6ASTRA = {**RESP_OK_61, "model": "gpt-6-astra"}


def _chat_with_tool_call(model: str, *, translated: bool = False) -> dict[str, Any]:
    """Chat 通道的 tool_calls 响应；translated=True 时附协议改写证据（billing_usage.source）。"""
    body: dict[str, Any] = {
        "id": "chatcmpl-" + "c" * 16, "object": "chat.completion", "model": model,
        "choices": [{"message": {"role": "assistant", "content": None,
                                 "tool_calls": [{"id": "c1", "type": "function",
                                                 "function": {"name": "get_order_status", "arguments": "{}"}}]},
                     "finish_reason": "tool_calls"}],
    }
    if translated:
        body["usage"] = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                         "billing_usage": {"source": "oai_responses"}}
    return body


class GPT6TierMatrixTests(unittest.TestCase):
    """档位能力矩阵：GPT-6 家族三档的契约画像必须与官方口径一致。"""

    def test_sol_and_luna_are_six_tier(self):
        for model in ("gpt-6-sol", "gpt-6-luna"):
            tier = tier_for_model(model)
            self.assertIsNotNone(tier, model)
            self.assertIn("none", tier.effort_levels, model)
            self.assertEqual(tier.rejected_efforts, (), model)
            self.assertEqual(tier.chat_tools, "none_only", model)

    def test_astra_and_61_reject_floor_and_chat_tools(self):
        for model in ("gpt-6-astra", "gpt-6.1-sol"):
            tier = tier_for_model(model)
            self.assertNotIn("none", tier.effort_levels, model)
            self.assertEqual(tier.rejected_efforts, ("none", "minimal"), model)
            self.assertEqual(tier.chat_tools, "rejected", model)

    def test_luna_kept_but_marked_unavailable(self):
        """luna 留位不可选：仍在矩阵里，但 available=False 且不在可选型号列表。"""
        tier = tier_for_model("gpt-6-luna")
        self.assertFalse(tier.available)
        self.assertNotIn("gpt-6-luna", GPT6_MODELS)
        self.assertIn("gpt-6-sol", GPT6_MODELS)
        self.assertIn("gpt-6-astra", GPT6_MODELS)

    def test_tier_resolution_is_strict(self):
        self.assertIsNone(tier_for_model("gpt-6-1-sol"))
        self.assertIsNone(tier_for_model("gpt-61-sol"))
        self.assertIsNone(tier_for_model(""))
        self.assertEqual(tier_for_model("gpt-6.1-sol-preview").model_id, "gpt-6.1-sol")
        self.assertEqual(len(MODEL_TIERS), 4)

    def test_declared_effort_values_parsing(self):
        self.assertEqual(_declared_effort_values(ENUM_UNIFIED_ERROR),
                         ("none", "minimal", "low", "medium", "high", "xhigh", "max"))
        self.assertEqual(_declared_effort_values(EFFORT_NONE_ERROR),
                         ("low", "medium", "high", "xhigh", "max"))
        self.assertIsNone(_declared_effort_values({"error": {"message": "boom"}}))


class GPT6TierOutcomeTests(unittest.IsolatedAsyncioTestCase):
    """GPT-6 家族：同一套 handler 必须按被测档位解析期望值（sol 与 astra 方向相反）。"""

    async def _runner(self, responses, *, variant="gpt6", max_retries=0):
        queue = list(responses)
        requests: list[dict[str, Any]] = []

        async def handler(request):
            requests.append(json.loads(request.content or b"{}"))
            status, body = queue.pop(0)
            return httpx.Response(status, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        runner = GPT56Runner(GPT56Config("key", max_retries=max_retries, retry_backoff_seconds=0.0), client)
        runner.variant = variant
        return runner, client, requests

    async def _one(self, responses, handler_name, *args, variant="gpt6"):
        runner, client, requests = await self._runner(responses, variant=variant)
        try:
            outcome = await getattr(runner, handler_name)(*args)
            if len(outcome) == 4:
                return outcome, requests
            passed, details, observations = outcome
            return (passed, details, observations, "passed" if passed else "failed"), requests
        finally:
            await client.aclose()

    async def test_sol_accepts_none_and_minimal(self):
        outcome, _ = await self._one([(200, RESP_OK_6SOL)], "case_g61_effort_none_rejected", "gpt-6-sol")
        self.assertEqual(outcome[3], "passed")
        outcome, _ = await self._one([(200, RESP_OK_6SOL)], "case_g61_effort_minimal_rejected", "gpt-6-sol")
        self.assertEqual(outcome[3], "passed")

    async def test_sol_rejecting_none_is_failed(self):
        """sol 官方支持 none：被拒说明后端档位集与文档不符。"""
        outcome, _ = await self._one([(400, EFFORT_NONE_ERROR)], "case_g61_effort_none_rejected", "gpt-6-sol")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_astra_rejecting_none_passes(self):
        outcome, _ = await self._one([(400, EFFORT_NONE_ERROR)], "case_g61_effort_none_rejected", "gpt-6-astra")
        self.assertEqual(outcome[3], "passed")

    async def test_astra_accepting_none_with_unified_enum_is_blocked(self):
        outcome, requests = await self._one([(200, RESP_OK_6ASTRA), (400, ENUM_UNIFIED_ERROR)],
                                            "case_g61_effort_none_rejected", "gpt-6-astra")
        self.assertEqual(outcome[3], "blocked")
        self.assertEqual(requests[1]["reasoning"]["effort"], "__probe_effort__")

    async def test_astra_accepting_none_without_relay_evidence_is_failed(self):
        outcome, _ = await self._one([(200, RESP_OK_6ASTRA), (400, EFFORT_NONE_ERROR)],
                                     "case_g61_effort_none_rejected", "gpt-6-astra")
        self.assertEqual(outcome[3], "failed")

    async def test_identity_uses_tier_token(self):
        outcome, _ = await self._one([(200, RESP_OK_6SOL)], "case_g61_identity", "gpt-6-sol")
        self.assertEqual(outcome[3], "passed")
        outcome, _ = await self._one([(200, RESP_OK_61)], "case_g61_identity", "gpt-6-sol")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_effort_levels_exclude_floor(self):
        outcome, requests = await self._one([(200, RESP_OK_6SOL)] * 5, "case_g61_effort_levels", "gpt-6-sol")
        self.assertEqual(outcome[3], "passed")
        self.assertEqual([r["reasoning"]["effort"] for r in requests],
                         ["low", "medium", "high", "xhigh", "max"])

    async def test_chat_tools_none_only_both_directions(self):
        queue = [(200, _chat_with_tool_call("gpt-6-sol")), (400, EFFORT_NONE_ERROR)]
        outcome, requests = await self._one(queue, "case_g61_chat_tools_rejected", "gpt-6-sol")
        self.assertEqual(outcome[3], "passed")
        self.assertEqual([r.get("reasoning_effort") for r in requests], ["none", "medium"])

    async def test_chat_tools_none_only_violation_is_failed(self):
        queue = [(200, _chat_with_tool_call("gpt-6-sol")), (200, _chat_with_tool_call("gpt-6-sol"))]
        outcome, _ = await self._one(queue, "case_g61_chat_tools_rejected", "gpt-6-sol")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_chat_tools_translation_is_flagged_as_protocol_rewrite(self):
        """带协议改写证据时归因于网关链路，不得表述为「后端不是该档位」。"""
        queue = [(200, _chat_with_tool_call("gpt-6.1-sol", translated=True))]
        outcome, _ = await self._one(queue, "case_g61_chat_tools_rejected", "gpt-6.1-sol", variant="gpt6.1")
        self.assertFalse(outcome[0])
        self.assertIn("协议改写", outcome[1])
        # 归因必须落在网关链路上，不得写成"此通道支持工具 ⇒ 后端不是该档位"。
        self.assertNotIn("即说明后端不是该档位", outcome[1])
        self.assertIn("归因于网关链路", outcome[1])

    async def test_sampling_none_only_violation_is_failed(self):
        outcome, requests = await self._one([(200, RESP_OK_61), (200, RESP_OK_61)],
                                            "case_g61_sampling_params_rejected", "gpt-6-sol")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")
        self.assertEqual(requests[1].get("reasoning_effort"), "none")

    async def test_sampling_lenient_acceptance_is_warning_for_no_none_tier(self):
        outcome, _ = await self._one([(200, RESP_OK_6ASTRA)], "case_g61_sampling_params_rejected", "gpt-6-astra")
        self.assertEqual(outcome[3], "warning")

    async def test_astra_default_effort_not_documented_is_not_vetoed(self):
        body = {**RESP_OK_6ASTRA, "reasoning": {"effort": "high"}}
        outcome, _ = await self._one([(200, body)], "case_g61_default_effort", "gpt-6-astra")
        self.assertEqual(outcome[3], "warning")

    async def test_astra_default_effort_falling_to_removed_level_is_failed(self):
        body = {**RESP_OK_6ASTRA, "reasoning": {"effort": "minimal"}}
        outcome, _ = await self._one([(200, body)], "case_g61_default_effort", "gpt-6-astra")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_unknown_model_falls_back_to_suite_tier(self):
        """网关自定义模型名无法识别时，按当前套件权威档位（sol）执行，判据不悬空。"""
        outcome, _ = await self._one([(200, RESP_OK_61)], "case_g61_effort_none_rejected", "gw-custom-model")
        self.assertEqual(outcome[3], "passed")

    async def test_luna_batch_is_blocked_with_single_probe(self):
        """luna 无通道：整批 blocked，只打一次端点留证，不逐条发请求。"""
        runner, client, requests = await self._runner([(200, {"object": "list", "data": [{"id": "gpt-6-sol"}]})])
        try:
            results = await runner.run(("gpt-6-luna",), ("P0",), None, variant="gpt6",
                                       faces=("model", "verify", "channel"))
        finally:
            await client.aclose()
        self.assertEqual(len(results), 19)
        self.assertTrue(all(r.status == "blocked" for r in results))
        self.assertTrue(all("无 gpt-6-luna 通道" in r.details for r in results))
        self.assertEqual(len(requests), 1)

    async def test_run_isolates_gpt6_variant(self):
        runner, client, _ = await self._runner([])
        try:
            self.assertEqual(await runner.run(("gpt-6-sol",), ("P0",), {"GPT61-01"}, variant="gpt6"), [])
            self.assertEqual(await runner.run(("gpt-6.1-sol",), ("P0",), {"GPT6-01"}, variant="gpt6.1"), [])
        finally:
            await client.aclose()


class GPT61OutcomeTests(unittest.IsolatedAsyncioTestCase):
    """GPT-6.1 的判定必须贴合 6.1 自身契约（none/minimal 被移除、工具面只在 Responses）。"""

    async def _runner(self, responses, max_retries=0):
        queue = list(responses)
        requests = []

        async def handler(request):
            requests.append(json.loads(request.content or b"{}"))
            status, body = queue.pop(0)
            return httpx.Response(status, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        config = GPT56Config("key", max_retries=max_retries, retry_backoff_seconds=0.0)
        return GPT56Runner(config, client), client, requests

    async def _one(self, responses, handler_name, *args):
        runner, client, requests = await self._runner(responses)
        try:
            outcome = await getattr(runner, handler_name)(*args)
            if len(outcome) == 4:
                return outcome, requests
            passed, details, observations = outcome
            return (passed, details, observations, "passed" if passed else "failed"), requests
        finally:
            await client.aclose()

    async def test_effort_none_rejected_with_evidence_passes(self):
        outcome, _ = await self._one([(400, EFFORT_NONE_ERROR)], "case_g61_effort_none_rejected", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "passed")

    async def test_effort_none_accepted_is_failed(self):
        """网关自身的档位集已排除 none ⇒ 接受 none 只能解释为后端档位集不符。"""
        outcome, requests = await self._one([(200, RESP_OK_61), (400, EFFORT_NONE_ERROR)],
                                            "case_g61_effort_none_rejected", "gpt-6.1-sol")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")
        # 第二次请求是档位白名单探测（必然非法值），用于区分模型口径与网关口径。
        self.assertEqual(requests[1]["reasoning"]["effort"], "__probe_effort__")

    async def test_effort_none_accepted_with_relay_enums_is_blocked(self):
        """网关声明统一七档（含 none）⇒ 该发现不能用于判定模型身份，记 blocked 而非 failed。"""
        outcome, _ = await self._one([(200, RESP_OK_61), (400, ENUM_UNIFIED_ERROR)],
                                     "case_g61_effort_none_rejected", "gpt-6.1-sol")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")
        self.assertIn("统一白名单", outcome[1])

    async def test_effort_none_accepted_without_probe_evidence_stays_failed(self):
        """探测不到网关档位集（探测请求非 4xx）时不得放宽判据。"""
        outcome, _ = await self._one([(200, RESP_OK_61), (200, RESP_OK_61)],
                                     "case_g61_effort_none_rejected", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "failed")

    async def test_effort_minimal_rejected(self):
        err = {"error": {"message": "unsupported value: 'minimal'", "type": "invalid_request_error",
                         "param": "reasoning.effort", "code": "unsupported_value"}}
        outcome, _ = await self._one([(400, err)], "case_g61_effort_minimal_rejected", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "passed")

    async def test_identity_token_mismatch_is_failed(self):
        wrong = {**RESP_OK_61, "model": "gpt-5.6-sol"}
        outcome, _ = await self._one([(200, wrong)], "case_g61_identity", "gpt-6.1-sol")
        self.assertFalse(outcome[0])
        outcome, _ = await self._one([(200, RESP_OK_61)], "case_g61_identity", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "passed")

    async def test_chat_tools_rejected_is_passed(self):
        err = {"error": {"message": "Function tools with reasoning_effort are not supported for gpt-6.1-sol "
                                    "in /v1/chat/completions. To use function tools, use /v1/responses",
                         "type": "invalid_request_error", "code": "unsupported_parameter"}}
        outcome, _ = await self._one([(400, err)], "case_g61_chat_tools_rejected", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "passed")

    async def test_chat_tools_succeeding_is_failed(self):
        """Chat+tools 竟成功发起工具调用 ⇒ 后端支持 5.6/6.0 工具面，不是 6.1。"""
        body = {**CHAT_OK_61, "choices": [{"message": {"role": "assistant", "content": None,
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "get_order_status", "arguments": "{}"}}]}}]}
        outcome, _ = await self._one([(200, body)], "case_g61_chat_tools_rejected", "gpt-6.1-sol")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_sampling_params_rejected_is_passed(self):
        outcome, requests = await self._one([(400, EFFORT_NONE_ERROR)], "case_g61_sampling_params_rejected", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(requests[0]["temperature"], 0.7)

    async def test_auth_effort_floor_accepting_none_with_relay_enums_is_blocked(self):
        """网关统一白名单（含 none/minimal）⇒ 档位指纹失效，记 blocked 而非 failed。"""
        outcome, _ = await self._one([(200, RESP_OK_61), (200, RESP_OK_61), (400, ENUM_UNIFIED_ERROR)],
                                     "case_g61_auth_effort_floor", "gpt-6.1-sol")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")

    async def test_auth_effort_floor_accepting_none_without_relay_evidence_is_failed(self):
        outcome, _ = await self._one([(200, RESP_OK_61), (200, RESP_OK_61), (400, EFFORT_NONE_ERROR)],
                                     "case_g61_auth_effort_floor", "gpt-6.1-sol")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_auth_effort_floor_rejecting_both_passes(self):
        outcome, _ = await self._one([(400, EFFORT_NONE_ERROR), (400, EFFORT_NONE_ERROR)],
                                     "case_g61_auth_effort_floor", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "passed")

    async def test_model_id_strictness(self):
        outcome, requests = await self._one([(400, EFFORT_NONE_ERROR)] * 4,
                                            "case_g61_model_id_strictness", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(len(requests), 4)
        self.assertNotIn("gpt-6.1-sol", {request["model"] for request in requests})
        outcome, _ = await self._one([(200, RESP_OK_61)] * 4,
                                     "case_g61_model_id_strictness", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "warning")

    async def test_builtin_tools_all_rejected_is_blocked(self):
        from gpt56.core import GPT61_BUILTIN_TOOLS
        outcome, _ = await self._one([(400, EFFORT_NONE_ERROR)] * len(GPT61_BUILTIN_TOOLS),
                                     "case_g61_builtin_tools", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "blocked")

    async def test_builtin_tools_partial_is_warning(self):
        from gpt56.core import GPT61_BUILTIN_TOOLS
        queue = [(400, EFFORT_NONE_ERROR)] * (len(GPT61_BUILTIN_TOOLS) - 1) + [(200, RESP_OK_61)]
        outcome, _ = await self._one(queue, "case_g61_builtin_tools", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "warning")

    async def test_truncation_detects_incomplete(self):
        body = {**RESP_OK_61, "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
        outcome, _ = await self._one([(200, body)], "case_g61_truncation", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "passed")

    async def test_truncation_without_transmitted_limit_is_warning(self):
        """中转未透传 max_output_tokens（回显 null、实际输出超限）⇒ warning，不判模型失败。

        2026-10-10 mux.la 实测：请求 16，响应 status=completed、max_output_tokens=null、
        output_tokens=225——上限根本没被应用，截断信号无从产生，与 GPT61-31 同一根因。
        """
        body = {**RESP_OK_61, "status": "completed", "incomplete_details": None, "max_output_tokens": None,
                "usage": {**RESP_OK_61["usage"], "output_tokens": 225, "total_tokens": 235}}
        outcome, _ = await self._one([(200, body)], "case_g61_truncation", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "warning")
        self.assertIn("未透传", outcome[1])

    async def test_truncation_with_effective_limit_but_no_signal_is_failed(self):
        """上限已生效（回显 16 且未超产）却仍无 incomplete 信号 ⇒ 真缺陷，判 failed。"""
        body = {**RESP_OK_61, "status": "completed", "incomplete_details": None, "max_output_tokens": 16}
        outcome, _ = await self._one([(200, body)], "case_g61_truncation", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "failed")

    async def test_allowed_tools_uses_nested_tool_choice(self):
        """工具开关必须嵌在 tool_choice 下；顶层 allowed_tools 会被网关丢弃（2026-10-10 实测）。"""
        call = {"type": "function_call", "name": "get_order_status", "arguments": "{\"order_id\":\"u1\"}"}
        outcome, requests = await self._one([(200, {**RESP_OK_61, "output": [call]})],
                                            "case_g61_allowed_tools", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "passed")
        self.assertNotIn("allowed_tools", requests[0])
        tool_choice = requests[0]["tool_choice"]
        self.assertEqual(tool_choice["type"], "allowed_tools")
        self.assertEqual([t["name"] for t in tool_choice["tools"]], ["get_order_status"])

    async def test_allowed_tools_ignored_is_failed(self):
        call = {"type": "function_call", "name": "delete_account", "arguments": "{\"order_id\":\"u1\"}"}
        outcome, _ = await self._one([(200, {**RESP_OK_61, "output": [call]})],
                                     "case_g61_allowed_tools", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "failed")

    def _image_pair(self, img_tokens: int, baseline_tokens: int, *, img_text="上面是红色，下面是蓝色。"):
        def body(text, tokens):
            return {**RESP_OK_61,
                    "output": [{"type": "message", "role": "assistant",
                                "content": [{"type": "output_text", "text": text}]}],
                    "usage": {**RESP_OK_61["usage"], "input_tokens": tokens, "total_tokens": tokens + 5}}
        return [(200, body(img_text, img_tokens)), (200, body("请上传图片，我才能判断。", baseline_tokens))]

    async def test_image_recognition_uses_delta_not_ratio(self):
        """旁证只要求带图 input_tokens 严格高于基线——中转注入前缀会淹没图片 token，比例阈值恒不可过。"""
        outcome, _ = await self._one(self._image_pair(4247, 4242),
                                     "case_g61_image_recognition", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "passed")

    async def test_image_recognition_without_token_delta_is_warning(self):
        outcome, _ = await self._one(self._image_pair(4242, 4242),
                                     "case_g61_image_recognition", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "warning")

    async def test_image_recognition_wrong_answer_is_failed(self):
        outcome, _ = await self._one(self._image_pair(4247, 4242, img_text="请上传图片。"),
                                     "case_g61_image_recognition", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "failed")

    async def test_config_update_preserves_cache(self):
        first = {**RESP_OK_61, "usage": {"input_tokens": 3000, "output_tokens": 4, "total_tokens": 3004,
                                         "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 1500}}}
        second = {**RESP_OK_61, "usage": {"input_tokens": 3000, "output_tokens": 4, "total_tokens": 3004,
                                          "input_tokens_details": {"cached_tokens": 2800, "cache_write_tokens": 0}}}
        outcome, requests = await self._one([(200, first), (200, second)], "case_g61_config_update_cache", "gpt-6.1-sol")
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(requests[1]["input"][0]["type"], "configuration_update")

    async def test_run_isolates_variants(self):
        """variant 过滤必须生效：5.6 档位不得执行 6.1 用例，反之亦然。"""
        runner, client, _ = await self._runner([(400, EFFORT_NONE_ERROR), (400, EFFORT_NONE_ERROR)])
        try:
            self.assertEqual(await runner.run(("gpt-5.6",), ("P0",), {"GPT61-01"}), [])
        finally:
            await client.aclose()
        runner, client, _ = await self._runner([(200, RESP_OK_61)])
        try:
            results = await runner.run(("gpt-6.1-sol",), ("P0",), {"GPT61-01"}, variant="gpt6.1")
            self.assertEqual([r.case_id for r in results], ["GPT61-01"])
            self.assertEqual(results[0].status, "passed")
        finally:
            await client.aclose()


class GPT56AdmissionExitCodeTests(unittest.IsolatedAsyncioTestCase):
    """退出码必须以 finalize() 之后的磁盘结论为准。

    回归：收尾前内存里的 summary 由 ``all_passed``（只算 failed+error）推出，
    "整批只有 blocked、没有 failed"时是 True，于是 main_async 曾返回 0，
    而磁盘 summary / Web 显示不准入（本轮瞬时故障场景实测 exit=0）。
    """

    async def _run_main(self, statuses):
        from gpt56 import runner as runner_module
        from gpt56.core import CaseResult

        created = {}

        class FakeRunner:
            def __init__(self, config, audit_dir=None):
                self.audit_dir = audit_dir
                created["dir"] = audit_dir

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def run(self, models, priorities, case_ids=None, variant="gpt5.6", faces=None):
                # 证据契约：结果目录必须有非空请求证据，否则整批不准入（与本用例无关，但要满足）。
                self.audit_dir.mkdir(parents=True, exist_ok=True)
                (self.audit_dir / "requests.jsonl").write_text(
                    json.dumps({"case_id": "GPT56-01", "http_status": 200}) + "\n", encoding="utf-8")
                return [CaseResult("GPT56-01", "基础对话", models[0], False, status, "说明")
                        for status in statuses]

        env = {"GPT_API_KEY": "key", "GPT_BASE_URL": "https://gw.example.com", "GPT_MODEL": "gpt-5.6-sol"}
        saved = {k: os.environ.get(k) for k in env}
        original = runner_module.GPT56Runner
        with tempfile.TemporaryDirectory() as tmp:
            try:
                runner_module.GPT56Runner = FakeRunner
                os.environ.update(env)
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = await runner_module.main_async(["GPT56_P0", "--output-root", tmp])
            finally:
                runner_module.GPT56Runner = original
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            disk = json.loads((Path(created["dir"]) / "summary.json").read_text(encoding="utf-8"))
        return rc, disk

    async def test_blocked_only_batch_is_not_admitted(self):
        """只有 blocked、没有 failed：磁盘不准入，退出码必须是 2（修复前误为 0）。"""
        rc, disk = await self._run_main(["blocked", "blocked"])
        self.assertFalse(disk["admission_passed"])
        self.assertEqual(disk["failed"], 0)  # 收尾前 all_passed 正是被这一项骗过
        self.assertEqual(rc, 2)

    async def test_blocked_does_not_block_when_effective_verdict_exists(self):
        """2026-10-10 口径：有有效结论（passed/warning）时，blocked 只记录、不阻断准入。"""
        rc, disk = await self._run_main(["passed", "blocked", "blocked"])
        self.assertTrue(disk["admission_passed"])
        self.assertEqual(disk["blocked"], 2)  # 仍如实记录，只是不否决
        self.assertEqual(rc, 0)

    async def test_warning_plus_blocked_batch_admits(self):
        """warning 也是有效结论：warning + blocked 的组合应准入。"""
        rc, disk = await self._run_main(["warning", "blocked"])
        self.assertTrue(disk["admission_passed"])
        self.assertEqual(rc, 0)

    async def test_all_passed_batch_still_admits(self):
        rc, disk = await self._run_main(["passed"])
        self.assertTrue(disk["admission_passed"])
        self.assertEqual(rc, 0)

    async def test_failed_batch_keeps_exit_code_2(self):
        rc, disk = await self._run_main(["failed"])
        self.assertFalse(disk["admission_passed"])
        self.assertEqual(rc, 2)


class GPT56PelicanSvgTests(unittest.IsolatedAsyncioTestCase):
    """鹈鹕骑车 SVG（产物类用例）：提示词逐字冻结 + 产物落盘 + 报告「产物」小节可点链接。

    需求方要求"提示词一点都不要动"，故除行为断言外**额外钉死提示词字面量**：文案一旦被
    顺手"优化"（补空格、换标点），历史批次与新批次的结论就不再可比。
    """

    PROMPT = "你给我画一个鹈鹕骑自行车的SVG，直接输出完整SVG，不要read，不要调用imagegen"
    SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
           '<circle cx="32" cy="32" r="16"/></svg>')

    def _body(self, content: str, finish_reason: str = "stop") -> dict[str, Any]:
        return {"model": "gpt-5.6-sol",
                "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}]}

    async def _run(self, body: dict[str, Any], *, audit_dir: bool = True, case_id: str = "GPT56-62"):
        requests: list[dict[str, Any]] = []

        async def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        directory = None
        if audit_dir:
            holder = tempfile.TemporaryDirectory()
            self.addCleanup(holder.cleanup)
            directory = Path(holder.name) / "GPT56_PELICAN-20261012-140000"
        runner = GPT56Runner(GPT56Config("key", retry_backoff_seconds=0.0), client, directory)
        try:
            results = await runner.run(("gpt-5.6",), ("PELICAN",), {case_id})
        finally:
            await client.aclose()
        return results[0], requests, directory

    async def test_prompt_is_sent_verbatim(self):
        """提示词逐字发送：不做任何 trim/补空格/换标点。"""
        self.assertEqual(PELICAN_PROMPT, self.PROMPT)
        # 三个档位的展示元数据里也必须原样出现同一句提示词。
        for name in ("GPT56_PELICAN", "GPT61_PELICAN", "GPT6_PELICAN"):
            self.assertIn(self.PROMPT, CASES[name].description, name)
        result, requests, _ = await self._run(self._body(f"```svg\n{self.SVG}\n```"))
        self.assertEqual(requests[0]["messages"], [{"role": "user", "content": self.PROMPT}])
        self.assertEqual(requests[0]["max_completion_tokens"], PELICAN_MAX_OUTPUT_TOKENS)
        self.assertEqual(result.status, "passed")

    async def test_complete_svg_is_passed_and_viewable_artifacts_are_written(self):
        """完整 SVG：判通过，并把渲染页 / 原始文件 / 响应原文都落到结果目录。"""
        result, _, directory = await self._run(self._body("这是鹈鹕：\n```svg\n" + self.SVG + "\n```\n骑自行车。"))
        self.assertTrue(result.passed)
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.observations["svg_chars"], len(self.SVG))
        self.assertEqual((directory / "pelican.svg").read_text(encoding="utf-8").strip(), self.SVG)
        self.assertIn(self.SVG, (directory / "pelican.html").read_text(encoding="utf-8"))
        self.assertIn("骑自行车", (directory / "pelican_response.md").read_text(encoding="utf-8"))
        names = [item["name"] for item in result.observations["artifacts"]]
        self.assertEqual(names, ["pelican.html", "pelican.svg", "pelican_response.md"])

    async def test_plain_text_without_svg_fails(self):
        """只有文字、没有 SVG：判失败（未按要求直出），且不留半成品产物。"""
        result, _, directory = await self._run(self._body("抱歉，我不能画矢量图。"))
        self.assertFalse(result.passed)
        self.assertEqual(result.status, "failed")
        self.assertIn("未包含 SVG", result.details)
        self.assertFalse((directory / "pelican.html").exists())
        self.assertEqual([item["name"] for item in result.observations["artifacts"]], ["pelican_response.md"])

    async def test_unclosed_svg_is_failed_and_cites_token_truncation(self):
        """输出被上限截断（未闭合的 <svg> + finish_reason=length）：判失败并说明截断成因。"""
        result, _, _ = await self._run(self._body("<svg xmlns=\"http://www.w3.org/2000/svg\"><circle", "length"))
        self.assertFalse(result.passed)
        self.assertEqual(result.status, "failed")
        self.assertIn("截断", result.details)
        self.assertEqual(result.observations["finish_reason"], "length")
        self.assertFalse(result.observations["svg_found"])

    async def test_artifacts_absent_without_result_dir_is_recorded_not_failed(self):
        """未指定结果目录（裸 CLI 直跑）：不落盘但也不误判——只在 observations 里留痕。"""
        result, _, _ = await self._run(self._body(self.SVG), audit_dir=False)
        self.assertTrue(result.passed)
        self.assertEqual(result.observations["artifacts"], [])
        self.assertIn("未落盘", result.observations["artifact_error"])

    async def test_report_lists_clickable_artifact_links(self):
        """报告「产物」小节：带记录 ID 时给站内可点链接，无记录 ID 时退回文件名。"""
        result, _, directory = await self._run(self._body(self.SVG))
        summary = summarize_results([result])
        summary.update({"record_id": "run-abc", "test_kind": "gpt", "case": "GPT56_PELICAN",
                        "model": "gpt-5.6", "model_variant": "gpt5.6",
                        "admission_passed": summary["all_passed"]})
        write_report(directory, summary)
        report = (directory / "report.md").read_text(encoding="utf-8")
        self.assertIn("## 产物", report)
        self.assertIn("[鹈鹕骑车 SVG（渲染视图，浏览器直接打开）](/api/runs/run-abc/artifact/pelican.html)", report)
        self.assertIn("/api/runs/run-abc/artifact/pelican.svg", report)

        summary["record_id"] = None
        write_report(directory, summary)
        self.assertNotIn("/api/runs/", (directory / "report.md").read_text(encoding="utf-8"))

    def test_three_variants_share_one_pelican_handler(self):
        """三档共用同一个 handler 与同一套判据（提示词跨档一致，不存在档位期望值差异）。"""
        specs = {spec.variant: spec for spec in CASE_SPECS if spec.priority == "PELICAN"}
        self.assertEqual(set(specs), {"gpt5.6", "gpt6.1", "gpt6"})
        for spec in specs.values():
            self.assertEqual(spec.handler, "pelican_svg")
        self.assertEqual([spec.case_id for spec in specs.values()], ["GPT56-62", "GPT61-37", "GPT6-37"])


class GPT56PelicanMultiRoundTests(unittest.IsolatedAsyncioTestCase):
    """鹈鹕骑车 SVG · 一次跑多轮（请求轮数 × 并发数）。

    多轮只增加样本量：**提示词与判据跨轮完全一致**，所以这里既钉死"每轮都发同一句提示词"，
    也钉死两类失败的归因——"某轮拿到 200 却没产出完整 SVG"是模型侧证据（failed），
    "某轮上游 5xx"是平台侧（blocked）。两者同时出现时不得互相压平：
    真缺陷轮必须保住 failed（``keeps_failure_verdict``），否则一批里明明有缺陷却被记成 blocked。
    """

    PROMPT = "你给我画一个鹈鹕骑自行车的SVG，直接输出完整SVG，不要read，不要调用imagegen"
    SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
           '<circle cx="32" cy="32" r="16"/></svg>')

    def _body(self, content: str) -> dict[str, Any]:
        return {"model": "gpt-5.6-sol",
                "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}

    async def _run(self, plan, *, rounds: int, concurrency: int, max_retries: int = 0, audit_dir: bool = True):
        """`plan(序号)` 返回 "ok" / "no-svg" / "5xx"；序号即第几个**请求**（重试会占号，故默认关重试）。"""
        requests: list[dict[str, Any]] = []

        async def handler(request):
            requests.append(json.loads(request.content))
            kind = plan(len(requests))
            if kind == "5xx":
                return httpx.Response(503, json={"error": {"message": "upstream busy"}})
            content = "抱歉，我画不了。" if kind == "no-svg" else f"第{len(requests)}只鹈鹕\n```svg\n{self.SVG}\n```"
            return httpx.Response(200, json=self._body(content))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        directory = None
        if audit_dir:
            holder = tempfile.TemporaryDirectory()
            self.addCleanup(holder.cleanup)
            directory = Path(holder.name) / "GPT56_PELICAN-20261012-150000"
        runner = GPT56Runner(GPT56Config("key", max_retries=max_retries, retry_backoff_seconds=0.0),
                             client, directory)
        runner.pelican_rounds = rounds
        runner.pelican_concurrency = concurrency
        try:
            results = await runner.run(("gpt-5.6",), ("PELICAN",), {"GPT56-62"})
        finally:
            await client.aclose()
        return results[0], requests, directory

    @staticmethod
    def _ok(_index: int) -> str:
        return "ok"

    async def test_multi_round_repeats_the_frozen_prompt_and_writes_overview(self):
        """5 轮 / 并发 3：每轮都发同一句提示词，产物是总览页 + 逐轮原始 SVG + 分段响应原文。"""
        result, requests, directory = await self._run(self._ok, rounds=5, concurrency=3)
        self.assertEqual(requests[0]["messages"], [{"role": "user", "content": self.PROMPT}])
        self.assertTrue(all(item["messages"][0]["content"] == PELICAN_PROMPT for item in requests),
                        "多轮必须逐轮原样发送冻结提示词")
        self.assertEqual(len(requests), 5)
        self.assertTrue(result.passed)
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.observations["rounds_total"], 5)
        self.assertEqual(result.observations["rounds_passed"], 5)
        self.assertEqual(result.observations["concurrency"], 3)
        self.assertEqual([item["round"] for item in result.observations["rounds"]], [1, 2, 3, 4, 5])
        # 多轮不再写 pelican.svg（它只代表"某一轮"，会误导）；总览页与响应原文是入口。
        self.assertEqual([item["name"] for item in result.observations["artifacts"]],
                         ["pelican.html", "pelican_response.md"])
        self.assertFalse((directory / "pelican.svg").exists())
        for index in range(1, 6):
            self.assertIn(self.SVG, (directory / f"pelican-{index}.svg").read_text(encoding="utf-8"))
        overview = (directory / "pelican.html").read_text(encoding="utf-8")
        self.assertEqual(overview.count("<section>"), 5)
        self.assertEqual(overview.count("viewBox"), 5)          # 每轮渲染视图都内联在总览页里
        self.assertIn('href="pelican-1.svg"', overview)          # 总览页指向各轮原始文件
        transcript = (directory / "pelican_response.md").read_text(encoding="utf-8")
        self.assertEqual([line for line in transcript.splitlines() if line.startswith("## ")],
                         [f"## 第 {index} 轮" for index in range(1, 6)])
        self.assertIn("第1只鹈鹕", transcript)

    async def test_single_round_keeps_legacy_behavior(self):
        """默认（不传轮数）仍是单轮：产物文件名与引入多轮之前完全一致。"""
        result, requests, directory = await self._run(self._ok, rounds=1, concurrency=1)
        self.assertEqual(len(requests), 1)
        self.assertEqual(result.observations["rounds_total"], 1)
        self.assertEqual(result.observations["concurrency"], 1)
        self.assertEqual([item["name"] for item in result.observations["artifacts"]],
                         ["pelican.html", "pelican.svg", "pelican_response.md"])
        self.assertEqual((directory / "pelican.svg").read_text(encoding="utf-8").strip(), self.SVG)
        self.assertFalse(list(directory.glob("pelican-*.svg")), "单轮不应产生逐轮文件")

    async def test_any_defect_round_fails_even_when_another_round_was_transient(self):
        """真缺陷轮 + 上游 5xx 轮同批：必须保持 failed（不得被瞬时故障压成 blocked）。"""
        result, requests, _ = await self._run(
            lambda index: {2: "5xx", 4: "no-svg"}.get(index, "ok"), rounds=4, concurrency=4)
        self.assertEqual(len(requests), 4)          # 关掉重试：4 轮各 1 次请求
        self.assertFalse(result.passed)
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.observations["keeps_failure_verdict"])
        self.assertEqual(result.observations["rounds_passed"], 2)
        self.assertEqual(result.observations["rounds_transient"], 1)
        self.assertIn("第 4 轮", result.details)
        self.assertIn("上游瞬时故障", result.details)

    async def test_transient_only_rounds_are_blocked_not_failed(self):
        """全部轮次都是上游 5xx：没有有效响应 ⇒ blocked（不冤枉模型），observation 留证据。"""
        result, _, _ = await self._run(lambda _index: "5xx", rounds=3, concurrency=3)
        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.passed)
        self.assertEqual(result.observations["rounds_transient"], 3)
        self.assertIn("blocked", result.details)

    async def test_success_plus_transient_round_yields_no_conclusion(self):
        """部分轮成功、部分轮瞬时故障：没有真缺陷也不能记 passed（批次不完整）⇒ blocked。"""
        result, _, _ = await self._run(
            lambda index: "5xx" if index == 2 else "ok", rounds=3, concurrency=1)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.observations["rounds_passed"], 2)
        self.assertEqual(result.observations["rounds_transient"], 1)
        self.assertNotIn("keeps_failure_verdict", result.observations)

    async def test_concurrency_is_capped_at_the_round_count(self):
        """并发数大于轮数时收敛到轮数：多出来的 worker 只会让"并发数"骗人。"""
        result, requests, _ = await self._run(self._ok, rounds=2, concurrency=8)
        self.assertEqual(len(requests), 2)
        self.assertEqual(result.observations["concurrency"], 2)
        self.assertEqual(result.status, "passed")

    async def test_report_declares_round_scale(self):
        """报告头部必须自证执行规模，否则"9/10 轮成功"这类结论无从解释。"""
        result, _, directory = await self._run(self._ok, rounds=3, concurrency=2)
        summary = summarize_results([result])
        summary.update({"record_id": "run-multi", "test_kind": "gpt", "case": "GPT56_PELICAN",
                        "model": "gpt-5.6", "model_variant": "gpt5.6",
                        "pelican_rounds": 3, "pelican_concurrency": 2,
                        "admission_passed": summary["all_passed"]})
        write_report(directory, summary)
        report = (directory / "report.md").read_text(encoding="utf-8")
        self.assertIn("- 请求轮数：3（并发 2）", report)
        self.assertIn("/api/runs/run-multi/artifact/pelican.html", report)
        # 多轮不写 pelican.svg：报告也不应出现指向它的死链接。
        self.assertNotIn("artifact/pelican.svg", report)


class GPT56PelicanRoundCliTests(unittest.IsolatedAsyncioTestCase):
    """Web → CLI 的轮数/并发透传：只有产物类用例接受这两个参数，且值域受限。"""

    async def _run_main(self, argv: list[str]):
        from gpt56 import runner as runner_module
        from gpt56.core import CaseResult

        seen: dict[str, Any] = {}

        class FakeRunner:
            def __init__(self, config, audit_dir=None):
                self.audit_dir = audit_dir
                seen["dir"] = audit_dir

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def run(self, models, priorities, case_ids=None, variant="gpt5.6", faces=None):
                seen["rounds"] = self.pelican_rounds
                seen["concurrency"] = self.pelican_concurrency
                self.audit_dir.mkdir(parents=True, exist_ok=True)
                (self.audit_dir / "requests.jsonl").write_text(
                    json.dumps({"case_id": "GPT56-62", "http_status": 200}) + "\n", encoding="utf-8")
                return [CaseResult("GPT56-62", "鹈鹕骑车 SVG 生成", models[0], True, "passed", "说明")]

        env = {"GPT_API_KEY": "key", "GPT_BASE_URL": "https://gw.example.com", "GPT_MODEL": "gpt-5.6-sol"}
        saved = {key: os.environ.get(key) for key in env}
        original = runner_module.GPT56Runner
        with tempfile.TemporaryDirectory() as tmp:
            try:
                runner_module.GPT56Runner = FakeRunner
                os.environ.update(env)
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = await runner_module.main_async([*argv, "--output-root", tmp])
            finally:
                runner_module.GPT56Runner = original
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            disk = json.loads((Path(seen["dir"]) / "summary.json").read_text(encoding="utf-8"))
        return rc, disk, seen

    async def test_cli_forwards_rounds_and_concurrency(self):
        rc, disk, seen = await self._run_main(["GPT56_PELICAN", "--requests", "5", "--concurrency", "2"])
        self.assertEqual(rc, 0)
        self.assertEqual((seen["rounds"], seen["concurrency"]), (5, 2))
        self.assertEqual((disk["pelican_rounds"], disk["pelican_concurrency"]), (5, 2))

    async def test_cli_defaults_to_single_round(self):
        _, disk, seen = await self._run_main(["GPT56_PELICAN"])
        self.assertEqual((seen["rounds"], seen["concurrency"]), (1, 1))
        self.assertEqual((disk["pelican_rounds"], disk["pelican_concurrency"]), (1, 1))

    async def test_cli_rejects_rounds_for_non_pelican_case(self):
        with self.assertRaises(ValueError) as caught:
            await self._run_main(["GPT56_P0", "--requests", "3"])
        self.assertIn("只对产物类用例", str(caught.exception))

    async def test_cli_rejects_out_of_range_values(self):
        for argv in (["GPT56_PELICAN", "--requests", "0"], ["GPT56_PELICAN", "--concurrency", "99"]):
            with self.assertRaises(ValueError, msg=argv):
                await self._run_main(argv)


if __name__ == "__main__":
    unittest.main()
