import json
import unittest
from unittest.mock import patch
from common.model_contracts import PROFILES, json_object, validate_identity, validate_usage, model_matches_family

class ContractTests(unittest.TestCase):
    def test_k3_gate_uses_builtin_family_when_optional_config_is_missing(self):
        with patch("common.model_contracts.load_profile_data", return_value={}):
            for model in ("kimi-k3", "Kimi-K3", "k3", "vendor/kimi-k3"):
                self.assertEqual(model_matches_family("kvv", model), (True, ""))
            for model in ("gpt-5.6-sol", "deepseek-v4", "glm-5.3", "", "  "):
                self.assertFalse(model_matches_family("kvv", model)[0])

    def test_explicit_vendor_alias_remains_supported(self):
        with patch("common.model_contracts.load_profile_data", return_value={"model_tokens": ["qianfan-vl-v1"]}):
            self.assertTrue(model_matches_family("kvv", "qianfan-vl-v1")[0])

    def test_profiles_are_model_specific(self):
        self.assertEqual(set(PROFILES), {"kvv", "deepseek", "glm"})
        self.assertNotEqual(PROFILES["kvv"].answer_marker, PROFILES["deepseek"].answer_marker)

    def test_identity_requires_expected_model(self):
        ok, obs = validate_identity(200, {"model": "deepseek-v4-pro"}, ("deepseek-v4",))
        self.assertTrue(ok); self.assertEqual(obs["response_model"], "deepseek-v4-pro")
        self.assertFalse(validate_identity(200, {"model": "other"}, ("deepseek-v4",))[0])

    def test_usage_contract(self):
        self.assertTrue(validate_usage({"usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}))
        self.assertFalse(validate_usage({"usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 6}}))

    def test_json_object(self):
        self.assertEqual(json_object('{"answer":"OK"}') ["answer"], "OK")
        self.assertIsNone(json_object("not-json"))


class K3ContractRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_k3_passes_gate_and_runs_all_probes_with_supported_params(self):
        import importlib.util
        import tempfile
        from pathlib import Path
        import httpx

        path = Path(__file__).resolve().parents[1] / "scripts/run_contracts.py"
        spec = importlib.util.spec_from_file_location("kvv_contract_runner_test", path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        seen = []

        def handler(request):
            body = json.loads(request.content)
            seen.append(body)
            self.assertEqual(body["model"], "kimi-k3")
            self.assertNotEqual(body.get("thinking", {}).get("type"), "disabled")
            if body.get("temperature") == 2.1:
                return httpx.Response(400, json={"error": {"message": "invalid temperature"}})
            if body.get("stream"):
                chunk = {"choices": [{"index": 0, "delta": {"content": "3071"}, "finish_reason": "stop"}]}
                return httpx.Response(200, text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n", headers={"Content-Type": "text/event-stream"})
            message = {"role": "assistant", "content": "OK", "reasoning_content": "reasoning"}
            if body["messages"][-1]["role"] == "tool":
                message["content"] = body["messages"][-1]["content"]
            elif "reasoning_effort" in body:
                message["content"] = str(918273*817263)
            elif body.get("tools"):
                message["tool_calls"] = [{"id": "call_1", "type": "function", "function": {"name": "lookup_order", "arguments": '{"order_id":"ORD-000731"}'}}]
            return httpx.Response(200, json={"model": "kimi-k3", "choices": [{"message": message, "finish_reason": "stop"}]})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with tempfile.TemporaryDirectory() as td, \
                patch.object(runner.sys, "argv", ["run_contracts.py", "kvv", "--model", "kimi-k3", "--base-url", "https://example.com/v1", "--api-key", "offline", "--output-root", td]), \
                patch.object(runner.httpx, "AsyncClient", return_value=client), \
                patch.object(runner, "finalize"), patch("builtins.print"):
            self.assertEqual(await runner.main(), 0)
            summary = json.loads(next(Path(td).glob("*/summary.json")).read_text())
            self.assertEqual((summary["total"], summary["passed"]), (5, 5))
        self.assertEqual(len(seen), 8)
        effort_requests = [body for body in seen if "reasoning_effort" in body]
        self.assertEqual([body["reasoning_effort"] for body in effort_requests], ["low", "high", "max"])
        for body in effort_requests:
            self.assertEqual(body["thinking"], {"type": "enabled", "keep": "all"})
