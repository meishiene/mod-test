"""fable5.model_name_check（模型名称两步校验）的离线测试。

全部走 httpx.MockTransport，不连网。钉住四件事：

1. **变异本身**：只改一个字符、长度不变、确定性可复现，且替换字符不在原名里；
2. **MN-01 前提**：真实模型名 200 ⇒ passed；4xx / 5xx / 传输错 ⇒ blocked，
   且第二步记 blocked（未发送变异请求）；
3. **MN-02 判定矩阵**：明确指向模型的 4xx 或 model_not_found 型错误 ⇒ passed；
   HTTP 200 放行 ⇒ failed；泛化 4xx / 限流 / 5xx / 传输错 ⇒ blocked；
   变异名恰在 /v1/models 声明列表中 ⇒ blocked（变异无效，不冤枉端点）；
   第二步负向请求不得重试（瞬时 5xx 被重试成 200 会造出假失败）；
4. **volcperf 证据契约**：summary.json 顶层键、requests.jsonl 齐备。
"""
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import httpx

from fable5 import model_name_check as mnc
from fable5.model_name_check import is_model_rejection, mutate_model_name
from fable5.proto_probe import BLOCKED, Client, FAILED, PASSED, Resp

MODEL = "claude-fable-5-1"
MUTATED, MUTATED_POS = mutate_model_name(MODEL)


def _msg(text: str = "OK", model: str = MODEL) -> dict:
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 2}}


def _client(handler, **kwargs) -> Client:
    kwargs.setdefault("backoff", 0.0)
    return Client("https://changhuai.ai", "sk-x", model=MODEL,
                  transport=httpx.MockTransport(handler), **kwargs)


def _models_probe(status: int = 200, ids: list[str] | None = None):
    """GET /v1/models 的响应（默认声明列表不含变异名）。"""
    if status != 200:
        return httpx.Response(status, json={"error": {"type": "not_found_error",
                                                      "message": "Not Found"}})
    return httpx.Response(200, json={"data": [{"id": item} for item in (ids or [MODEL, "claude-opus-5"])]})


def _routed(messages: list, *, models=None, models_status: int = 200):
    """按 method 路由：POST /v1/messages 按序取预设，GET /v1/models 返回声明列表。"""
    queue = list(messages)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return (_models_probe(models_status) if models is None
                    else httpx.Response(200, json=models))
        if not queue:
            return httpx.Response(500, json={"type": "error",
                                             "error": {"type": "api_error", "message": "over"}})
        item = queue.pop(0)
        if item.get("status") != 200:
            error = {"type": item.get("type", "invalid_request_error"),
                     "message": item.get("message", "x")}
            if item.get("code"):
                error["code"] = item["code"]
            return httpx.Response(item["status"], json={"type": "error", "error": error})
        return httpx.Response(200, json=item.get("body") or _msg(item.get("text", "OK")))

    return _client(handler)


def _always(status: int, body: dict) -> Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return _models_probe()
        return httpx.Response(status, json=body)

    return _client(handler)


class MutationTests(unittest.TestCase):
    def test_single_character_change_keeps_length(self):
        changed, position = mutate_model_name(MODEL)
        self.assertEqual(len(changed), len(MODEL))
        self.assertEqual(sum(a != b for a, b in zip(MODEL, changed)), 1)
        self.assertNotEqual(changed[position], MODEL[position])
        self.assertNotIn(changed[position], MODEL.lower())

    def test_mutation_is_deterministic_and_interior(self):
        self.assertEqual(mutate_model_name(MODEL), mutate_model_name(MODEL))
        self.assertGreater(mutate_model_name(MODEL)[1], 0)
        self.assertLess(mutate_model_name(MODEL)[1], len(MODEL) - 1)

    def test_unmutable_name_raises(self):
        with self.assertRaises(ValueError):
            mutate_model_name("ab")

    def test_rejection_requires_model_related_error(self):
        def resp(status: int, body: dict) -> Resp:
            return Resp(status, {}, json.dumps(body).encode(), 0.0)

        self.assertTrue(is_model_rejection(resp(404, {"error": {
            "type": "not_found_error", "message": "model: claude-qable-5-1 not found"}})))
        self.assertTrue(is_model_rejection(resp(503, {"error": {
            "type": "model_not_found", "message": "No available accounts"}})))
        self.assertFalse(is_model_rejection(resp(429, {"error": {
            "type": "rate_limit_error", "message": "slow down"}})))
        self.assertFalse(is_model_rejection(resp(400, {"error": {
            "type": "invalid_request_error", "message": "max_tokens: field required"}})))
        self.assertFalse(is_model_rejection(resp(404, {"error": {
            "type": "not_found_error", "message": "Not Found"}})))

    def test_new_api_error_code_field_is_read(self):
        """2026-10-13 实批回归：New API 把码放在 error.code，type 只是 new_api_error。

        真实响应体逐字回放（180.178.56.190，claude-fable-5 → claude-qable-5）：旧判据
        只看 error.type ⇒ 503 落进"错误未指向模型"分支被误记 blocked。
        """
        real = Resp(503, {}, json.dumps({"error": {
            "code": "model_not_found",
            "type": "new_api_error",
            "message": "No available channel for model claude-qable-5 under group claude-cc "
                       "(distributor) (request id: 202610130158286689395408268d9d6Ub5uKmUb)",
        }}).encode(), 0.0)
        self.assertTrue(is_model_rejection(real, requested_model="claude-qable-5"))
        # 同一信封若换成"渠道整体不可用"（未点名模型）⇒ 不得当作模型名校验成功
        generic = Resp(503, {}, json.dumps({"error": {
            "type": "new_api_error",
            "message": "No available accounts: no available accounts (request id: abc)",
        }}).encode(), 0.0)
        self.assertFalse(is_model_rejection(generic, requested_model="claude-qable-5"))


class Mn01Tests(unittest.TestCase):
    def test_normal_model_name_is_premise(self):
        verdicts, evidence = mnc.run_all(_routed([
            {"status": 200},
            {"status": 200, "text": "OK"},
        ]))
        self.assertEqual(verdicts[0].status, PASSED)
        self.assertIn("正常模型名", verdicts[0].detail)
        self.assertEqual(evidence["mutation"]["changed_characters"], 1)

    def test_endpoint_rejection_blocks_both_steps(self):
        verdicts, _ = mnc.run_all(_routed([
            {"status": 400, "message": "bad"},
        ]))
        self.assertEqual([v.status for v in verdicts], [BLOCKED, BLOCKED])
        self.assertIn("未发送变异模型名请求", verdicts[1].detail)

    def test_transient_5xx_retried_then_blocked(self):
        verdicts, evidence = mnc.run_all(_always(503, {"error": {
            "type": "api_error", "message": "No available accounts: no available accounts"}}))
        self.assertEqual(verdicts[0].status, BLOCKED)
        self.assertIn("No available accounts", verdicts[0].detail)
        self.assertEqual(evidence["requests"][0]["attempts"], 3)  # 默认重试 2 次

    def test_transport_failure_is_blocked(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        verdicts, _ = mnc.run_all(_client(handler))
        self.assertEqual(verdicts[0].status, BLOCKED)
        self.assertIn("ConnectError", verdicts[0].detail)

    def test_non_json_body_is_failed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return _models_probe()
            return httpx.Response(200, content=b"<html>not json</html>")

        verdicts, _ = mnc.run_all(_client(handler))
        self.assertEqual(verdicts[0].status, FAILED)
        self.assertIn("不是合法的 Messages JSON", verdicts[0].detail)


class Mn02Tests(unittest.TestCase):
    def test_explicit_model_rejection_passes(self):
        verdicts, evidence = mnc.run_all(_routed([
            {"status": 200},
            {"status": 400, "type": "invalid_request_error",
             "message": f"model: {MUTATED} not found"},
        ]))
        self.assertEqual([v.status for v in verdicts], [PASSED, PASSED])
        self.assertEqual(evidence["mutation"]["to"], MUTATED)
        row = evidence["requests"][-1]
        self.assertEqual(row["request"]["model"], MUTATED)
        self.assertEqual(row["phase"], "mutated_model_request")

    def test_503_model_not_found_counts_as_rejection(self):
        # 已知上游形态（i-token.ai）：未知模型 ⇒ 503 + model_not_found，属确定性拒绝
        verdicts, _ = mnc.run_all(_routed([
            {"status": 200},
            {"status": 503, "type": "model_not_found", "message": "model not found"},
        ]))
        self.assertEqual([v.status for v in verdicts], [PASSED, PASSED])

    def test_new_api_code_field_end_to_end(self):
        """2026-10-13 实批形态（New API：码在 error.code）必须判 passed 而非 blocked。"""
        verdicts, evidence = mnc.run_all(_routed([
            {"status": 200},
            {"status": 503, "type": "new_api_error", "code": "model_not_found",
             "message": f"No available channel for model {MUTATED} under group claude-cc (distributor)"},
        ]))
        self.assertEqual([v.status for v in verdicts], [PASSED, PASSED])
        obs = verdicts[1].observations
        self.assertEqual(obs["error_code"], "model_not_found")
        self.assertEqual(obs["error_type"], "new_api_error")
        self.assertIn("model_not_found", verdicts[1].detail)
        self.assertEqual(evidence["requests"][-1]["attempts"], 1)

    def test_200_acceptance_is_failed(self):
        verdicts, evidence = mnc.run_all(_routed([
            {"status": 200},
            {"status": 200, "text": "OK"},
        ]))
        self.assertEqual(verdicts[1].status, FAILED)
        self.assertIn("未被拒绝", verdicts[1].detail)
        self.assertTrue(evidence["declared_models_known"])
        # 模型列表探测必须留痕（佐证"声明列表里没有这个变异名"）
        self.assertEqual(evidence["requests"][-2]["phase"], "models_probe")

    def test_200_with_rewritten_model_notes_mapping(self):
        verdicts, _ = mnc.run_all(_routed([
            {"status": 200},
            {"status": 200, "body": _msg(model="claude-sonnet-5")},
        ]))
        self.assertEqual(verdicts[1].status, FAILED)
        self.assertIn("静默改写", verdicts[1].detail)
        self.assertEqual(verdicts[1].observations["response_model"], "claude-sonnet-5")

    def test_mutation_colliding_with_declared_model_is_blocked(self):
        # 变异名恰好是端点声明的可用模型 ⇒ 变异无效，取证前提落空，不判端点 failed
        verdicts, evidence = mnc.run_all(_routed(
            [{"status": 200}, {"status": 400, "message": "should not be sent"}],
            models={"data": [{"id": MODEL}, {"id": MUTATED}]}))
        self.assertEqual(verdicts[1].status, BLOCKED)
        self.assertIn("变异无效", verdicts[1].detail)
        # 只在 /v1/models 命中时都不发变异请求
        self.assertEqual([row["phase"] for row in evidence["requests"]],
                         ["initial", "models_probe"])

    def test_generic_4xx_is_blocked_not_passed(self):
        verdicts, _ = mnc.run_all(_routed([
            {"status": 200},
            {"status": 400, "type": "invalid_request_error", "message": "max_tokens: field required"},
        ]))
        self.assertEqual(verdicts[1].status, BLOCKED)
        self.assertIn("未提及模型", verdicts[1].detail)

    def test_rate_limit_is_blocked(self):
        verdicts, _ = mnc.run_all(_routed([
            {"status": 200},
            {"status": 429, "type": "rate_limit_error", "message": "slow down"},
        ]))
        self.assertEqual(verdicts[1].status, BLOCKED)

    def test_negative_request_does_not_retry_5xx(self):
        # 关键负控：瞬时 5xx 若被重试成 200，会把"该拒未拒"误判成 failed（假失败）
        calls = {"post": 0}
        queue = [{"status": 200}, {"status": 503, "type": "api_error", "message": "temporary"}]

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return _models_probe()
            calls["post"] += 1
            item = queue.pop(0) if queue else {"status": 200}
            if item["status"] != 200:
                return httpx.Response(item["status"], json={"error": {"type": item["type"],
                                                                     "message": item["message"]}})
            return httpx.Response(200, json=_msg())

        verdicts, evidence = mnc.run_all(_client(handler, max_retries=2))
        self.assertEqual(verdicts[1].status, BLOCKED)
        self.assertEqual(calls["post"], 2)  # 第一步 1 + 第二步 1（未重试）
        self.assertEqual(evidence["requests"][-1]["attempts"], 1)

    def test_models_probe_failure_still_judges(self):
        # /v1/models 不可用只影响佐证，不影响判定（此时仍按 200 放行 ⇒ failed）
        verdicts, evidence = mnc.run_all(_routed(
            [{"status": 200}, {"status": 200}], models_status=404))
        self.assertEqual(verdicts[1].status, FAILED)
        self.assertFalse(evidence["declared_models_known"])
        self.assertIn("未能取得端点 /v1/models 声明列表佐证", verdicts[1].detail)


class OutputTests(unittest.TestCase):
    def test_outputs_carry_volcperf_keys_and_evidence(self):
        out = Path(tempfile.mkdtemp()) / "mnc"
        with _routed([{"status": 200}, {"status": 400, "message": f"model {MUTATED} not found"}]) as client:
            verdicts, evidence = mnc.run_all(client)
            summary = mnc.write_outputs(out, MODEL, client.root, verdicts, evidence)
        self.assertTrue(summary["admission_passed"])
        payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "volcperf.result.v2")
        self.assertEqual(payload["case"], "MODEL_NAME_CHECK")
        self.assertEqual(payload["script"], "fable5.model_name_check")
        self.assertEqual(payload["total"], 2)
        self.assertEqual({row["case_id"] for row in payload["results"]}, {"MN-01", "MN-02"})
        for row in payload["results"]:
            self.assertIn(row["status"], {"passed", "failed", "warning", "blocked"})
            self.assertIsInstance(row["passed"], bool)
        self.assertTrue((out / "requests.jsonl").read_text(encoding="utf-8").strip())
        self.assertTrue((out / "mutation.json").exists())
        self.assertIn("模型名称两步校验", (out / "report.md").read_text(encoding="utf-8"))

    def test_blocked_or_failed_batch_not_admitted(self):
        self.assertFalse(mnc.summarize([mnc._verdict(1, PASSED, "d"),
                                        mnc._verdict(2, BLOCKED, "d")])["admission_passed"])
        self.assertFalse(mnc.summarize([mnc._verdict(1, PASSED, "d"),
                                        mnc._verdict(2, FAILED, "d")])["admission_passed"])


class SelfTestTests(unittest.TestCase):
    def test_self_test_runs_offline(self):
        import io
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = mnc.main(["--self-test"])
        self.assertEqual(code, 0)
        self.assertIn("变异确定性通过", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
