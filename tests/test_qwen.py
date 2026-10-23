"""千问（Qwen 3.6/3.7/3.8）套件回归。

重点不是「跑通」，而是**裁定口径**：
- 端点拒绝参数（4xx）/ 平台侧故障（429、5xx、连接错）只记 blocked，不判 failed；
- 能力级不符（思考开关未生效、预算超限、并行工具退化）记 warning；
- 信任级问题（响应模型异族替换、不存在的模型被静默放行）才 failed / 不准入。

mock 一律用 ``httpx.MockTransport`` 回放官方响应形状（含 usage.completion_tokens_details）。
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx

import qwen.core as qwen_core
from qwen.cases import (CASES, QWEN_DEFAULT_MODEL, QWEN_MODEL_IDS, QWEN_MODELS,
                        PRIORITY_THINKING, PRIORITY_TIER, PRIORITY_VISION, model_spec)
from qwen.core import QwenConfig, QwenRunner, summarize_results, write_report

BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _chat(content: str = "OK", model: str = "qwen3.8-max", *, reasoning: str | None = None,
          reasoning_tokens: int | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    usage: dict = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return {"model": model, "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": usage}


def _sse(chunks: list[dict], usage: dict | None = None) -> bytes:
    lines = [f"data: {json.dumps(chunk, ensure_ascii=False)}" for chunk in chunks]
    if usage is not None:
        lines.append(f"data: {json.dumps({'choices': [], 'usage': usage}, ensure_ascii=False)}")
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode("utf-8")


async def _run(handler, priorities, model: str = "qwen3.8-max", audit_dir: Path | None = None):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    runner = QwenRunner(QwenConfig("sk-test", BASE), audit_dir=audit_dir, client=client)
    try:
        async with runner:
            return await runner.run(model, priorities, faces=("model", "verify", "channel"))
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# 登记与矩阵
# ---------------------------------------------------------------------------

class QwenRegistrationTests(unittest.TestCase):
    def test_five_cards_and_priorities(self):
        self.assertEqual(set(CASES), {"QWEN_CORE", "QWEN_THINKING", "QWEN_AGENT", "QWEN_TIER", "QWEN_VISION", "QWEN_CAPABILITY", "QWEN_VERIFY", "QWEN_FULL"})
        self.assertEqual([CASES[name].request_count for name in CASES], [9, 6, 6, 10, 5, 12, 6, 54])
        for case in CASES.values():
            self.assertEqual(case.concurrency, 1)
            self.assertEqual(case.protocols, ("openai",))
            self.assertTrue(case.admission, f"{case.id} 缺少准入口径文案")
        self.assertTrue(CASES["QWEN_TIER"].model_scope)
        self.assertTrue(CASES["QWEN_VISION"].model_scope)
        self.assertFalse(CASES["QWEN_CORE"].model_scope)

    def test_case_specs_cover_every_priority_and_handler(self):
        priorities = {spec.priority for spec in qwen_core.CASE_SPECS}
        for case in CASES.values():
            self.assertTrue(set(case.priorities) <= priorities, f"{case.id} 没有对应用例")
        self.assertEqual(len({spec.case_id for spec in qwen_core.CASE_SPECS}), len(qwen_core.CASE_SPECS))
        for spec in qwen_core.CASE_SPECS:
            self.assertTrue(hasattr(QwenRunner, "case_" + spec.handler), spec.handler)
        self.assertEqual(len([s for s in qwen_core.CASE_SPECS if s.priority == PRIORITY_TIER]), 6)
        self.assertEqual(len([s for s in qwen_core.CASE_SPECS if s.priority == PRIORITY_VISION]), 1)
        self.assertEqual(len([s for s in qwen_core.CASE_SPECS if s.priority == PRIORITY_THINKING]), 6)

    def test_six_models_in_matrix(self):
        self.assertEqual(QWEN_MODEL_IDS,
                         ("qwen3.8-max", "qwen3.8-flash", "qwen3.7-max", "qwen3.7-plus",
                          "qwen3.6-plus", "qwen3.6-flash"))
        self.assertIn(QWEN_DEFAULT_MODEL, QWEN_MODEL_IDS)
        # 3.6/3.7/3.8 系列默认开启思考：矩阵必须一致（套件的 enable_thinking 关闭口径依赖它）。
        self.assertTrue(all(model.thinking_default for model in QWEN_MODELS))
        self.assertEqual(model_spec("qwen/qwen3.8-max").generation, "3.8")
        self.assertIsNone(model_spec("gpt-5.6"))

    def test_qwen_protocol_uses_compatible_mode_root(self):
        from volcperf.protocols import PROTOCOLS, api_root
        spec = PROTOCOLS["qwen"]
        self.assertEqual(spec.family, "openai")
        self.assertIn("qwen", spec.test_kinds)
        self.assertEqual(api_root("qwen", "https://dashscope.aliyuncs.com"),
                         "https://dashscope.aliyuncs.com/compatible-mode/v1")
        # 已带路径的地址原样返回，不会重复拼接。
        self.assertEqual(api_root("qwen", BASE), BASE)


# ---------------------------------------------------------------------------
# 响应解析
# ---------------------------------------------------------------------------

class QwenParsingTests(unittest.TestCase):
    def test_reasoning_carriers(self):
        self.assertEqual(qwen_core.reasoning_text(_chat(reasoning="先算乘法")), "先算乘法")
        body = _chat()
        body["choices"][0]["message"]["reasoning_details"] = [{"type": "reasoning.text", "text": "ab"},
                                                             {"text": "cd"}]
        self.assertEqual(qwen_core.reasoning_text(body), "abcd")
        self.assertEqual(qwen_core.reasoning_text(_chat()), "")

    def test_reasoning_tokens_only_from_completion_details(self):
        self.assertEqual(qwen_core.reasoning_tokens(_chat(reasoning_tokens=128)), 128)
        self.assertIsNone(qwen_core.reasoning_tokens(_chat()))

    def test_out_of_range_negative_excludes_404(self):
        self.assertTrue(qwen_core._unsupported(400, {"error": {"message": "InvalidParameter: temperature"}}))
        self.assertFalse(qwen_core._unsupported(404, {"error": {"message": "invalid parameter"}}))
        self.assertTrue(qwen_core._blocked_status(429))
        self.assertTrue(qwen_core._blocked_status(503))
        self.assertFalse(qwen_core._blocked_status(200))


# ---------------------------------------------------------------------------
# 思考模式：参数未生效只记 warning
# ---------------------------------------------------------------------------

class QwenThinkingTests(unittest.IsolatedAsyncioTestCase):
    def test_thinking_off_but_reasoning_present_is_warning(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_chat("结果 750000000000", reasoning="我先拆解乘法", reasoning_tokens=90))

        results = asyncio.run(_run(handler, [PRIORITY_THINKING]))
        result = {item.case_id: item for item in results}["QWEN-T02"]
        self.assertEqual(result.status, "warning")
        self.assertFalse(result.passed)
        self.assertIsNone(result.admission)

    def test_budget_enforced_vs_exceeded(self):
        for tokens, expected in ((120, "passed"), (900, "warning")):
            def handler(request: httpx.Request, tokens=tokens) -> httpx.Response:
                return httpx.Response(200, json=_chat("5050", reasoning="推理", reasoning_tokens=tokens))

            results = asyncio.run(_run(handler, [PRIORITY_THINKING]))
            result = {item.case_id: item for item in results}["QWEN-T03"]
            self.assertEqual(result.status, expected, tokens)

    def test_missing_usage_cannot_verify_budget(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_chat("5050", reasoning="推理"))

        results = asyncio.run(_run(handler, [PRIORITY_THINKING]))
        result = {item.case_id: item for item in results}["QWEN-T03"]
        self.assertEqual(result.status, "blocked")

    def test_streaming_reasoning_and_usage(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = _sse([
                {"model": "qwen3.8-max", "choices": [{"delta": {"reasoning_content": "先算"}}]},
                {"model": "qwen3.8-max", "choices": [{"delta": {"content": str(918273*817263)}, "finish_reason":"stop"}]},
            ], usage={"prompt_tokens": 10, "completion_tokens": 30, "total_tokens": 40,
                      "completion_tokens_details": {"reasoning_tokens": 12}})
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        results = asyncio.run(_run(handler, [PRIORITY_THINKING]))
        result = {item.case_id: item for item in results}["QWEN-T04"]
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.observations["reasoning_tokens"], 12)


# ---------------------------------------------------------------------------
# 档位指纹：信任级 vs 能力级
# ---------------------------------------------------------------------------

class QwenTierTests(unittest.IsolatedAsyncioTestCase):
    def test_other_family_model_echo_fails(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_chat("OK", model="glm-5.3"))

        results = asyncio.run(_run(handler, [PRIORITY_TIER]))
        result = {item.case_id: item for item in results}["QWEN-F01"]
        self.assertEqual(result.status, "failed")

    def test_same_family_alias_is_warning(self):
        def handler(request: httpx.Request) -> httpx.Response:
            # 同族不同档：请求 3.8-max 却返回 3.7-plus，属「档位替换」而非异族替换。
            return httpx.Response(200, json=_chat("OK", model="qwen3.7-plus"))

        results = asyncio.run(_run(handler, [PRIORITY_TIER]))
        result = {item.case_id: item for item in results}["QWEN-F01"]
        self.assertEqual(result.status, "warning")
        self.assertIsNone(result.admission)

    def test_silent_fallback_blocks_admission(self):
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/qwen-probe-not-found" in url:
                return httpx.Response(404, json={"error": {"message": "not found"}})
            raw = request.read()
            try:
                body = json.loads(raw)
            except ValueError:
                return httpx.Response(400, json={"error": {"message": "bad json"}})
            # 不存在的模型也返回 200：疑似静默回退。
            return httpx.Response(200, json=_chat("OK"))

        results = asyncio.run(_run(handler, [PRIORITY_TIER]))
        result = {item.case_id: item for item in results}["QWEN-F05"]
        self.assertTrue(result.observations["silent_fallback"])
        self.assertIs(result.admission, False)
        self.assertEqual(result.status, "warning")
        summary = summarize_results(results)
        self.assertFalse(summary["all_passed"])
        self.assertIn("QWEN-F05", summary["admission_blocked"])

    def test_high_risk_leak_fails(self):
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/qwen-probe-not-found" in url:
                return httpx.Response(404, text="not found")
            return httpx.Response(400, text='Traceback (most recent call last): File "/app/main.py", line 42, in handler')

        results = asyncio.run(_run(handler, [PRIORITY_TIER]))
        result = {item.case_id: item for item in results}["QWEN-F05"]
        self.assertEqual(result.status, "failed")
        self.assertIs(result.admission, False)

    def test_platform_failures_are_blocked_not_failed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})

        results = asyncio.run(_run(handler, [PRIORITY_THINKING]))
        self.assertTrue(results)
        self.assertEqual({item.status for item in results}, {"blocked"})


# ---------------------------------------------------------------------------
# 视觉
# ---------------------------------------------------------------------------

def _vision_answer(prompt):
    rng = QwenRunner(QwenConfig("offline", BASE)).probe_rng("vision")
    digits = str(rng.randrange(1000, 10000))
    right_yellow = bool(rng.getrandbits(1))
    if "四位数字" in prompt: return digits
    if "右半部分" in prompt: return "黄色" if right_yellow else "蓝色"
    return "32" if "深色格子" in prompt else "黄色"


class QwenVisionTests(unittest.IsolatedAsyncioTestCase):
    def test_all_correct_passes(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.read())
            prompt = body["messages"][0]["content"][0]["text"]
            answer = _vision_answer(prompt)
            return httpx.Response(200, json=_chat(answer))

        results = asyncio.run(_run(handler, [PRIORITY_VISION]))
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.observations["levels_correct"], 5)
        self.assertEqual(len(result.observations["rows"]), 5)

    def test_three_of_five_is_warning(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.read())
            prompt = body["messages"][0]["content"][0]["text"]
            answer = "wrong" if "深色格子" in prompt or "四位数字" in prompt else _vision_answer(prompt)
            return httpx.Response(200, json=_chat(answer))

        results = asyncio.run(_run(handler, [PRIORITY_VISION]))
        result = results[0]
        self.assertEqual(result.status, "warning")
        self.assertEqual(result.observations["levels_correct"], 3)
        # 能力打折不否决准入。
        self.assertIsNone(result.admission)
        summary = summarize_results(results)
        self.assertTrue(summary["all_passed"])

    def test_zero_of_five_fails(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.read())
            prompt = body["messages"][0]["content"][0]["text"]
            if "深色格子" in prompt:
                return httpx.Response(200, json=_chat("31"))
            if "背景" in prompt:
                return httpx.Response(200, json=_chat("蓝色"))
            return httpx.Response(200, json=_chat("红色"))

        results = asyncio.run(_run(handler, [PRIORITY_VISION]))
        self.assertEqual(results[0].status, "failed")
        self.assertEqual(results[0].observations["levels_correct"], 0)

    def test_unsupported_image_input_is_blocked(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"type": "invalid_request_error",
                                                       "message": "image input not supported"}})

        results = asyncio.run(_run(handler, [PRIORITY_VISION]))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "blocked")
        self.assertIsNone(results[0].admission)


# ---------------------------------------------------------------------------
# 汇总与报告
# ---------------------------------------------------------------------------

class QwenSummaryTests(unittest.IsolatedAsyncioTestCase):
    def test_report_expands_sub_rows_and_keeps_warning_admitted(self):
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/qwen-probe-not-found" in url:
                return httpx.Response(404, json={"error": {"message": "not found"}})
            raw = request.read()
            try:
                json.loads(raw)
            except ValueError:
                return httpx.Response(400, json={"error": {"message": "bad json"}})
            if b"nonexistent" in raw:
                return httpx.Response(400, json={"error": {"message": "model not found"}})
            return httpx.Response(200, json=_chat("OK"))

        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp)
            results = asyncio.run(_run(handler, [PRIORITY_TIER], audit_dir=audit))
            summary = summarize_results(results)
            self.assertTrue(summary["all_passed"], summary["admission_blocked"])
            self.assertEqual(summary["tier_assessment"]["tested"], 5)
            write_report(audit, summary)
            report = (audit / "report.md").read_text(encoding="utf-8")
            self.assertIn("QWEN-F01", report)
            self.assertIn("↳ 畸形 JSON", report)
            self.assertIn("↳ 不存在的路径", report)
            self.assertIn("档位指纹结论", report)
            self.assertTrue((audit / "requests.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(json.loads((audit / "summary.json").read_text(encoding="utf-8"))["total"], 6)

    def test_all_passed_ignores_warning_without_admission_veto(self):
        results = [qwen_core.CaseResult("QWEN-F02", "档位与代际自述", "qwen3.8-max", False,
                                        "warning", "自述不一致（仅留痕）", 0.0, {})]
        summary = summarize_results(results)
        self.assertTrue(summary["all_passed"])
        self.assertEqual(summary["admission_blocked"], [])
        self.assertEqual(summary["tier_assessment"]["level"], "部分指纹不一致，不能认定同款")


if __name__ == "__main__":
    unittest.main()


class QwenRelayProfileTests(unittest.TestCase):
    """中转接入口径：核心面/信任面阻断，高级面缺口只记录。"""

    def _summary(self, items, model="qwen3.8-max"):
        results = [qwen_core.CaseResult(case_id, case_id, model, passed, status, "detail", 0.0, {})
                   for case_id, status, passed in items]
        return summarize_results(results, model)

    def test_every_case_has_a_face(self):
        from qwen.relay_profile import CASE_FACES
        from qwen.core import CASE_SPECS
        self.assertEqual({spec.case_id for spec in CASE_SPECS}, set(CASE_FACES))

    def test_core_failure_blocks_admission(self):
        summary = self._summary([("QWEN-01", "failed", False), ("QWEN-02", "passed", True)])
        self.assertFalse(summary["relay_verdict"]["admission_passed"])
        self.assertEqual([row["case_id"] for row in summary["relay_verdict"]["blocking"]], ["QWEN-01"])

    def test_trust_level_model_substitution_blocks(self):
        summary = self._summary([("QWEN-01", "passed", True), ("QWEN-F01", "failed", False)])
        self.assertFalse(summary["relay_verdict"]["admission_passed"])

    def test_strict_tool_gap_does_not_block(self):
        """A02（tool_choice=required 不强制）在中转侧只记 relay_gap。"""
        summary = self._summary([("QWEN-01", "passed", True), ("QWEN-A02", "failed", False)])
        verdict = summary["relay_verdict"]
        self.assertTrue(verdict["admission_passed"])
        self.assertEqual([row["case_id"] for row in verdict["relay_gap"]], ["QWEN-A02"])
        self.assertIn("强制工具语义", verdict["summary_line"])

    def test_declared_vision_rejection_is_not_blocking(self):
        """声明无视觉的型号（qwen3.7-max）视觉被拒 ⇒ vision_undeclared，不阻断。"""
        summary = self._summary([("QWEN-V01", "blocked", False)], model="qwen3.7-max")
        verdict = summary["relay_verdict"]
        self.assertTrue(verdict["admission_passed"])
        self.assertEqual(verdict["vision_undeclared"], ["QWEN-V01"])
        self.assertIn("视觉未开放", verdict["summary_line"])

    def test_declared_vision_all_blocked_has_no_verdict(self):
        """声明支持视觉的型号整批被拒 ⇒ 无有效结论，不能记准入。"""
        summary = self._summary([("QWEN-V01", "blocked", False)], model="qwen3.8-max")
        self.assertFalse(summary["relay_verdict"]["admission_passed"])

    def test_vision_partially_wrong_still_admits(self):
        summary = self._summary([("QWEN-V01", "warning", False)], model="qwen3.8-max")
        self.assertTrue(summary["relay_verdict"]["admission_passed"])

    def test_official_verdict_is_still_recorded(self):
        summary = self._summary([("QWEN-01", "passed", True), ("QWEN-A02", "failed", False)])
        self.assertFalse(summary["official_all_passed"])
        self.assertTrue(summary["all_passed"] is False)   # 官方口径字段保持原义，可追溯
        self.assertTrue(summary["relay_verdict"]["admission_passed"])


class QwenNonJsonGuardTests(unittest.IsolatedAsyncioTestCase):
    """网关把未知路径兜成前端 index.html（200）时，必须记 blocked 而不是「空回答」。"""

    def test_spa_html_200_is_blocked(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-type": "text/html"},
                                  text="<!doctype html><html><head><title>AinoWork</title></head></html>")

        results = asyncio.run(_run(handler, ["P0"]))
        self.assertEqual({item.status for item in results}, {"blocked"},
                         [(item.case_id, item.status, item.details) for item in results])
        summary = summarize_results(results, "qwen3.8-max")
        self.assertFalse(summary["relay_verdict"]["admission_passed"])

    def test_html_guard_does_not_touch_json(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_chat("OK"))

        results = asyncio.run(_run(handler, [PRIORITY_VISION]))
        self.assertEqual({item.status for item in results}, {"failed"},
                         "纯色夹具靠模型回答判定，答 OK 应 failed 而非 blocked")


class QwenRelayFinalizeTests(unittest.TestCase):
    """收尾层必须尊重 relay 判据契约：高级面 failed 不能被通用「有 failed 即不准入」推翻。"""

    def test_finalize_keeps_relay_admission(self):
        from common.suite_runtime import finalize
        results = [qwen_core.CaseResult("QWEN-01", "基础对话", "qwen3.7-max", True, "passed", "ok", 0.0, {}),
                   qwen_core.CaseResult("QWEN-A02", "tool_choice=required", "qwen3.7-max", False,
                                        "failed", "强制调用=未命中", 0.0, {})]
        summary = summarize_results(results, "qwen3.7-max")
        summary.update({"test_kind": "qwen", "case": "QWEN_AGENT", "model": "qwen3.7-max",
                        "admission_passed": summary["relay_verdict"]["admission_passed"]})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "requests.jsonl").write_text(json.dumps({"case_id": "QWEN-A02", "http_status": 200}) + "\n",
                                                encoding="utf-8")
            write_report(out, summary)
            final = finalize(out)
            self.assertTrue(final["admission_passed"], final["validation"])
            self.assertIn("relay", final["verdict_contract"])
            self.assertEqual(final["validation"]["original_admission_passed"], True)


class QwenVisionDeclarationTests(unittest.IsolatedAsyncioTestCase):
    """型号矩阵声明的图像能力是权威：无视觉型号的视觉卡片判「不适用」，不是失败也不是缺口。"""

    def test_only_3_7_max_lacks_vision(self):
        no_vision = [model.model_id for model in QWEN_MODELS if not model.image_input]
        self.assertEqual(no_vision, ["qwen3.7-max"],
                         "3.7-max 是纯文本旗舰（已确认）；3.6-plus 实测 3/3 读图正确，不能按同代推定无视觉")

    def test_every_other_model_declares_vision(self):
        for model in QWEN_MODELS:
            if model.model_id != "qwen3.7-max":
                self.assertTrue(model.image_input, model.model_id)

    def test_rejection_on_text_only_model_reads_as_not_applicable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"type": "invalid_request_error",
                                                       "code": "InvalidParameter",
                                                       "message": "Unexpected item type in content"}})

        results = asyncio.run(_run(handler, [PRIORITY_VISION], model="qwen3.7-max"))
        result = results[0]
        self.assertEqual(result.status, "blocked")
        self.assertIn("不提供图像输入", result.details)
        self.assertIn("不适用", result.details)
        self.assertFalse(result.observations["applicable"])
        self.assertFalse(summarize_results(results, "qwen3.7-max")["relay_verdict"]["blocking"])

    def test_rejection_on_vision_model_reads_as_gateway_gap(self):
        """声明支持视觉却被拒 ⇒ 措辞必须指向网关/上游缺口，不能被误读成型号边界。"""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"type": "invalid_request_error",
                                                       "message": "Unexpected item type in content"}})

        results = asyncio.run(_run(handler, [PRIORITY_VISION], model="qwen3.8-max"))
        result = results[0]
        self.assertEqual(result.status, "blocked")
        self.assertIn("网关/上游能力缺口", result.details)
        self.assertTrue(result.observations["applicable"])
