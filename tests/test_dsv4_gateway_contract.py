"""中转网关口径（gateway verdict）回归测试。

背景：`dsv4/` 用例按 **DeepSeek 官方文档** 设计，实际常打在中转/托管网关（如百度千帆）上。
网关侧限制——端点缺失、限流、账号/平台前提未满足、不收某类内容、自有取值域、模型本就不提供的
能力面（如 Pro 的视觉）——**不得判成「模型能力缺陷」**。判定层在 `dsv4/gateway_contract.py`，
用例内收口在 `_gateway_verdict()`，统一兜底在 `run()` 的 `_downgrade_gateway_failure()`。

对应真实批次证据：`results/DSV4-TRIAGE-20261014.md`（10 条 failed 无一可归因模型缺陷）。
"""
import inspect
import json
import unittest

import httpx

from dsv4.core import DeepSeekConfig, DeepSeekV4Runner
from dsv4.gateway_contract import (
    CAPABILITY_VISION,
    classify_gateway_issue,
    model_supports_capability,
)


class GatewayContractUnitTests(unittest.TestCase):
    def test_endpoint_missing_is_blocked(self):
        issue = classify_gateway_issue(404, '{"code":"ResourceNotFound","message":"404 page not found"}')
        self.assertEqual(issue.status, "blocked")
        self.assertEqual(issue.kind, "endpoint_missing")

    def test_rate_limit_is_blocked(self):
        issue = classify_gateway_issue(429, '{"code":"OverRateLimit","message":"Over rate limit."}')
        self.assertEqual((issue.kind, issue.status), ("rate_limited", "blocked"))

    def test_account_precondition_on_5xx_is_blocked(self):
        issue = classify_gateway_issue(
            500, "upload_file_error: [Code: OptInRequired; Message: You must active bos service.]")
        self.assertEqual((issue.kind, issue.status), ("account_precondition", "blocked"))

    def test_capability_not_in_model_vs_gateway_unsupported(self):
        message = "Invalid content type. image_url is only supported by certain models"
        # 被测模型未声明该能力面（Pro 不带视觉）⇒ blocked
        pro = classify_gateway_issue(400, message, model="deepseek-v4-pro-0813", capability=CAPABILITY_VISION)
        self.assertEqual((pro.kind, pro.status), ("capability_not_in_model", "blocked"))
        # 模型声明支持、网关整体不收 ⇒ warning（功能可用性不受影响）
        flash = classify_gateway_issue(400, message, model="deepseek-flash", capability=CAPABILITY_VISION)
        self.assertEqual((flash.kind, flash.status), ("capability_unsupported_by_gateway", "warning"))

    def test_param_domain_narrowed_is_warning(self):
        for message in ("reasoning_effort must be none or low or medium or high or xhigh or max",
                        "invalid_value: Supported values are: function, mcp, knowledge_search, namespace"):
            issue = classify_gateway_issue(400, message)
            self.assertEqual((issue.kind, issue.status), ("param_domain_narrowed", "warning"), message)

    def test_model_side_errors_stay_failed(self):
        """模型/协议自身的参数错误不带网关特征 ⇒ 不降级（仍是 failed）。"""
        for status, message in ((400, "messages must be an array"),
                                (400, "temperature must be a number"),
                                (401, "invalid api key")):
            self.assertIsNone(classify_gateway_issue(status, message))

    def test_vision_capability_follows_official_docs(self):
        # 官方 Vision / Files API / 定价页：图片输入归属 deepseek-flash（含退休视觉名）
        self.assertTrue(model_supports_capability("deepseek-flash", CAPABILITY_VISION))
        self.assertTrue(model_supports_capability("deepseek-v4-flash-vision-exp", CAPABILITY_VISION))
        self.assertFalse(model_supports_capability("deepseek-v4-pro", CAPABILITY_VISION))
        self.assertFalse(model_supports_capability("deepseek-v4-pro-0813", CAPABILITY_VISION))

    def test_vision_capability_tracks_model_version_not_name_prefix(self):
        """名字含 flash ≠ 有视觉：V4-Flash-0731 早于视觉开放，属纯文本版本。

        2026-10-21 实测：0731 与 pro 收图片块都返回 200 且 prompt_tokens 同样 +75，
        但都答不出图 ⇒ 只能按官方「模型版本」判能力面，不能按 `flash` 子串。
        """
        for text_only in ("deepseek-v4-flash-0731", "deepseek-v4-flash-ga-260731",
                          "deepseek-v4-flash-vision-exp-0731"):
            self.assertFalse(model_supports_capability(text_only, CAPABILITY_VISION), text_only)
        for vision in ("deepseek-v4.1-flash-1017", "deepseek-v4.1-flash-ark",
                       "deepseek-v4-1-flash-261017", "deepseek-v4-flash"):
            self.assertTrue(model_supports_capability(vision, CAPABILITY_VISION), vision)
        # 未知/空模型名不得默认成「有视觉」
        self.assertFalse(model_supports_capability("", CAPABILITY_VISION))
        self.assertFalse(model_supports_capability(None, CAPABILITY_VISION))
        # 其他能力面不受影响
        self.assertTrue(model_supports_capability("deepseek-v4-flash-0731", "thinking"))


class GatewayVerdictCaseTests(unittest.IsolatedAsyncioTestCase):
    async def _runner(self, responses, base_url="https://api.deepseek.com"):
        queue = list(responses)

        async def handler(request):
            body = None
            if request.content:
                try: body = json.loads(request.content.decode("utf-8", "replace"))
                except json.JSONDecodeError: body = {"_raw": "multipart"}
            item = queue.pop(0)
            if callable(item): return await item(request)
            return httpx.Response(item[0], json=item[1])
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return DeepSeekV4Runner(DeepSeekConfig("key", base_url), client), client

    @staticmethod
    def _chat(content="OK"):
        return {"choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}}

    async def test_image_input_on_non_vision_model_is_blocked(self):
        """Pro 不提供视觉面：网关拒收 image_url ⇒ blocked，不得判「模型不会看图」。"""
        body = {"error": {"message": "Invalid content type. image_url is only supported by certain models"}}
        runner, client = await self._runner([(200, self._chat()), (400, body)])
        try: outcome = await runner.case_image_input("deepseek-v4-pro-0813")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")
        self.assertIn("未声明支持视觉", outcome[1])

    async def test_image_wrong_role_4xx_from_gateway_is_blocked_not_false_pass(self):
        """DS-V4-42 假通过防护：整体拒收图片时的 4xx 与「role 放错」无关 ⇒ blocked。"""
        body = {"error": {"message": "Invalid content type. image_url is only supported by certain models"}}
        runner, client = await self._runner([(400, body)])
        try: outcome = await runner.case_image_wrong_role("deepseek-v4-pro-0813")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")
        # 视觉模型上、错误文案确指 role 时仍应判通过
        runner, client = await self._runner([(400, {"error": {"message": "Images are supported in user messages only"}})])
        try: outcome = await runner.case_image_wrong_role("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(len(outcome), 3)
        # 千帆对 v4.1-flash 的实测文案：虽含 "not supported"，但明确点名「system messages」
        # ⇒ 拒绝可归因到 role 放错（文档承诺的负向行为），应判 passed 而非 blocked。
        runner, client = await self._runner([(400, {"error": {"message": "Images in system messages are not supported. Request id: 0217"}})])
        try: outcome = await runner.case_image_wrong_role("deepseek-v4.1-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(len(outcome), 3)
        self.assertIn("明确指向放置位置", outcome[1])

    async def test_models_listing_rate_limited_is_blocked(self):
        runner, client = await self._runner([(429, {"code": "OverRateLimit", "message": "Over rate limit."}),
                                             (429, {"code": "OverRateLimit", "message": "Over rate limit."})])
        try: outcome = await runner.case_models_listing("deepseek-v4-pro-0813")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")

    async def test_files_upload_account_precondition_is_blocked(self):
        body = {"error": {"message": "upload_file_error: [Code: OptInRequired; Message: You must active bos service.]"}}
        runner, client = await self._runner([(500, body)])
        try: outcome = await runner.case_files_lifecycle("deepseek-v4-pro-0813")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")

    async def test_logprobs_not_relayed_is_warning(self):
        """官方支持 logprobs，但网关不透传 ⇒ 不影响使用，记 warning 而非 failed。"""
        empty = {"choices": [{"message": {"role": "assistant", "content": "OK"}, "logprobs": {"content": []}}]}
        runner, client = await self._runner([(200, empty), (200, empty)])
        try: outcome = await runner.case_logprobs_structure("deepseek-v4-pro-0813")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")
        self.assertIn("不透传", outcome[1])

    async def test_responses_strict_validation_is_warning(self):
        """官方承诺「不支持字段静默忽略」，网关改为强校验 ⇒ 更严格但可用，记 warning。"""
        body = {"error": {"message": "previous response id resp_nonexistent not found"}}
        runner, client = await self._runner([(400, body)])
        try: outcome = await runner.case_responses_stateless("deepseek-v4-pro-0813")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")
        self.assertIn("强校验", outcome[1])

    async def test_cache_not_observed_is_blocked(self):
        """官方缓存是 best-effort：多跳都没命中 ⇒ blocked（未观测到前提），不判失败。"""
        empty = {"choices": [{"message": {"role": "assistant", "content": "一"}}],
                 "usage": {"prompt_tokens": 7200, "completion_tokens": 2, "total_tokens": 7202}}
        runner, client = await self._runner([(200, empty), (200, empty), (200, empty)])
        try: outcome = await runner.case_cache_hit("deepseek-v4-pro-0813")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")
        self.assertIn("best-effort", outcome[1])

    async def test_downgrade_only_kicks_in_for_gateway_signatures(self):
        """run() 的单点降级：failed + 网关特征 ⇒ blocked/warning；无特征 ⇒ 保持 failed。"""
        runner, client = await self._runner([(429, {"code": "OverRateLimit", "message": "Over rate limit."})])
        try:
            runner.current_model = "deepseek-v4-pro-0813"
            runner._case_gateway_issues = []
            await runner.request("chat/completions", {"model": "deepseek-v4-pro-0813", "messages": []})
            status, details = runner._downgrade_gateway_failure("failed", "某正向断言不满足")
            self.assertEqual(status, "blocked")
            self.assertIn("网关口径降级", details)
            runner._case_gateway_issues = []
            status, details = runner._downgrade_gateway_failure("failed", "模型自身能力不达标")
            self.assertEqual(status, "failed")
            self.assertEqual(details, "模型自身能力不达标")
        finally:
            await client.aclose()


if __name__ == "__main__":
    unittest.main()
