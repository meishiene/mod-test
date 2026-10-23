"""fable5.zero_injection_check（0 注入中转判定组）的离线测试。

全部走 httpx.MockTransport，不连网。钉住四件事：

1. **纯函数口径**：金丝雀回显大小写不敏感、token 差分只看增量、模型族只做同族/异族判断、
   签名拒绝必须指向签名、thinking 模式按模型名解析；
2. **ZI-01 分支**：system 金丝雀回显 ⇒ passed（且不再发对照请求）；system 未回显而 user
   对照回显 ⇒ failed（system 被剥离）；两者都未回显 ⇒ warning（不冤wang端点）；
   对照请求本身失败 ⇒ blocked；
3. **ZI-02 / ZI-03 / ZI-04 判定矩阵**：缺省 system 的 token 增量超阈值 ⇒ failed；usage 缺失 ⇒ blocked；
   未声明工具却出现 tool_use ⇒ failed；异族模型 ⇒ failed；有 thinking 无签名 ⇒ failed；
   无 thinking 块 / 原签名回传遇非签名错误 ⇒ blocked；原签名被拒 ⇒ failed；
4. **volcperf 证据契约与准入口径**：summary.json 顶层键、requests.jsonl 非空、
   failed ⇒ 不准入、纯 blocked ⇒ 不准入、passed+blocked ⇒ 准入（blocked 只记录不阻断）。
"""
import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import httpx

from fable5 import zero_injection_check as zic
from fable5.proto_probe import BLOCKED, Client, FAILED, PASSED, Resp, WARNING
from fable5.signature_structure import synthetic_signature

MODEL = "claude-opus-5"
CANARY_RE = re.compile(r"CANARY-[0-9A-F]+")


def _msg(model: str = MODEL, text: str = "OK", *, content=None, usage=None,
         stop_reason: str = "end_turn") -> dict:
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": model,
            "content": content if content is not None else [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "usage": usage or {"input_tokens": 12, "output_tokens": 2}}


def _body(request: httpx.Request) -> dict:
    return json.loads(request.content.decode("utf-8"))


def _client(handler, model: str = MODEL, **kwargs) -> Client:
    kwargs.setdefault("backoff", 0.0)
    return Client("https://changhuai.ai", "sk-x", model=model,
                  transport=httpx.MockTransport(handler), **kwargs)


def _always(payload):
    """固定应答的 handler：dict 按 Messages 返回，tuple(status, body) 返回错误。"""
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(payload, tuple):
            return httpx.Response(payload[0], json=payload[1])
        return httpx.Response(200, json=payload)

    return handler


def _run(handler, model: str = MODEL):
    with _client(handler, model=model) as client:
        return zic.run_all(client)


def _verdict_by_id(verdicts, case_id: str):
    return next(v for v in verdicts if v.case_id == case_id)


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------

class PureFunctionTests(unittest.TestCase):
    def test_canary_is_random_and_echo_is_case_insensitive(self):
        first, second = zic.new_canary(), zic.new_canary()
        self.assertTrue(first.startswith("CANARY-") and len(first) == 19)
        self.assertNotEqual(first, second)
        self.assertTrue(zic.canary_echoed(f"口令是 {first.lower()}", first))
        self.assertFalse(zic.canary_echoed("我不清楚口令。", first))
        self.assertFalse(zic.canary_echoed("", first))

    def test_token_delta_helpers(self):
        def resp(body: dict, status: int = 200) -> Resp:
            return Resp(status, {}, json.dumps(body).encode(), 0.0)

        self.assertEqual(zic.input_tokens(resp({"usage": {"input_tokens": 12}})), 12)
        self.assertIsNone(zic.input_tokens(resp({"usage": {"input_tokens": True}})))
        self.assertIsNone(zic.input_tokens(resp({"usage": {}})))
        self.assertIsNone(zic.input_tokens(resp({})))

    def test_model_family_only_compares_family_tokens(self):
        self.assertEqual(zic.model_families("claude-opus-5"), {"opus"})
        self.assertEqual(zic.model_families("relay-claude-1"), set())
        self.assertFalse(zic.model_families("claude-opus-5") & zic.model_families("claude-sonnet-5"))
        # 同族不同版本不算异族（属别名，只留痕）
        self.assertTrue(zic.model_families("claude-fable-5") & zic.model_families("claude-fable-5-1"))

    def test_thinking_mode_resolved_by_model_name(self):
        # 档位按模型族三分（唯一权威 fable5/model_profile）：4.5/4.6 系 enabled、
        # 5 系 adaptive+output_config.effort、fable 系沿用 adaptive。
        self.assertEqual(zic.thinking_mode_for("claude-haiku-4-5"), "enabled")
        self.assertEqual(zic.thinking_mode_for("claude-haiku-4-5-20251001"), "enabled")
        self.assertEqual(zic.thinking_mode_for("claude-opus-4-5-20251101"), "enabled")
        self.assertEqual(zic.thinking_mode_for("claude-sonnet-4-6"), "enabled")
        self.assertEqual(zic.thinking_mode_for("claude-fable-5"), "adaptive")
        self.assertEqual(zic.thinking_mode_for("claude-sonnet-5"), "adaptive_effort")
        self.assertEqual(zic.thinking_mode_for("claude-opus-5-5"), "adaptive_effort")
        # 中转别名识别不出族时回退 adaptive（派发侧再用 claude_variant 兜底）。
        self.assertEqual(zic.thinking_mode_for("relay-haiku"), "adaptive")

    def test_adaptive_effort_sends_output_config(self):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
                "content": [{"type": "thinking", "thinking": "x", "signature": synthetic_signature()},
                            {"type": "text", "text": "2"}], "stop_reason": "end_turn"})

        with Client("https://example.com", "secret", model="claude-sonnet-5",
                    transport=httpx.MockTransport(handler)) as client:
            zic._zi04(client, {"requests": []}, "adaptive_effort")
        self.assertEqual(seen[0]["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertEqual(seen[0]["output_config"], {"effort": "high"})

    def test_blocks_and_signature_helpers(self):
        body = {"content": [{"type": "thinking", "thinking": "x", "signature": "QUJD"},
                            {"type": "text", "text": "A"}]}
        self.assertEqual(len(zic.signed_thinking_blocks(body)), 1)
        self.assertEqual(zic.block_types(body), ["thinking", "text"])
        self.assertEqual(zic.response_text(body), "A")
        self.assertEqual(zic.signed_thinking_blocks(
            {"content": [{"type": "thinking", "signature": ""}]}), [])

    def test_signature_rejection_must_point_at_signature(self):
        def resp(status: int, message: str) -> Resp:
            return Resp(status, {}, json.dumps({"error": {"message": message}}).encode(), 0.0)

        self.assertTrue(zic.is_signature_rejection(resp(400, "Invalid signature in thinking block")))
        self.assertTrue(zic.is_signature_rejection(resp(422, "thinking block 签名无效")))
        self.assertFalse(zic.is_signature_rejection(resp(400, "signature not supported")))
        self.assertFalse(zic.is_signature_rejection(resp(400, "max_tokens: field required")))
        self.assertFalse(zic.is_signature_rejection(resp(400, "invalid api key")))
        self.assertTrue(zic.mentions_thinking(resp(400, "thinking.budget_tokens: unsupported")))
        self.assertFalse(zic.mentions_thinking(resp(400, "model: not found")))


# ---------------------------------------------------------------------------
# ZI-01 system 金丝雀透传
# ---------------------------------------------------------------------------

class Zi01Tests(unittest.TestCase):
    def test_system_canary_echo_passes_without_control_request(self):
        def handler(request: httpx.Request) -> httpx.Response:
            match = CANARY_RE.search(_body(request).get("system") or "")
            if match is None:
                # 其余三条用例的中性应答（本测试只关心 ZI-01 是否走对照分支）
                return httpx.Response(200, json=_msg(text="OK"))
            return httpx.Response(200, json=_msg(text=f"口令是 {match.group(0)}。"))

        verdicts, evidence = _run(handler)
        case = _verdict_by_id(verdicts, "ZI-01")
        self.assertEqual(case.status, PASSED)
        self.assertTrue(case.observations["echoed_in_system"])
        # 已回显即无需求发对照请求，避免浪费
        phases = [row["phase"] for row in evidence["requests"]]
        self.assertEqual(phases.count("canary_in_system"), 1)
        self.assertNotIn("canary_in_user_control", phases)

    def test_system_stripped_but_user_canary_echoed_is_failed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = _body(request)
            if body.get("system"):
                # 网关把 system 剥掉了：模型看不到金丝雀
                return httpx.Response(200, json=_msg(text="抱歉，我不知道什么口令。"))
            match = CANARY_RE.search(json.dumps(body, ensure_ascii=False))
            if match is None:
                return httpx.Response(200, json=_msg(text="OK"))
            return httpx.Response(200, json=_msg(text=f"口令是 {match.group(0)}。"))

        verdicts, _ = _run(handler)
        case = _verdict_by_id(verdicts, "ZI-01")
        self.assertEqual(case.status, FAILED)
        self.assertFalse(case.observations["echoed_in_system"])
        self.assertTrue(case.observations["echoed_in_user"])
        self.assertIn("输入侧注入", case.detail)

    def test_both_channels_missing_is_warning(self):
        verdicts, _ = _run(_always(_msg(text="抱歉，我不清楚。")))
        case = _verdict_by_id(verdicts, "ZI-01")
        self.assertEqual(case.status, WARNING)
        self.assertIn("无法确证", case.detail)

    def test_control_request_failure_is_blocked(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(200, json=_msg(text="不知道。"))
            return httpx.Response(401, json={"error": {"type": "authentication_error",
                                                       "message": "invalid api key"}})

        verdicts, _ = _run(handler)
        case = _verdict_by_id(verdicts, "ZI-01")
        self.assertEqual(case.status, BLOCKED)
        self.assertIn("对照请求失败", case.detail)

    def test_first_request_transport_failure_is_blocked(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        verdicts, _ = _run(handler)
        case = _verdict_by_id(verdicts, "ZI-01")
        self.assertEqual(case.status, BLOCKED)
        self.assertIn("ConnectError", case.detail)


# ---------------------------------------------------------------------------
# ZI-02 缺省 system 差分
# ---------------------------------------------------------------------------

class Zi02Tests(unittest.TestCase):
    def _by_system(self, omitted_tokens: int, empty_tokens: int):
        def handler(request: httpx.Request) -> httpx.Response:
            body = _body(request)
            tokens = empty_tokens if "system" in body else omitted_tokens
            return httpx.Response(200, json=_msg(usage={"input_tokens": tokens,
                                                        "output_tokens": 2}))

        return handler

    def test_equal_tokens_passes(self):
        verdicts, _ = _run(self._by_system(12, 12))
        case = _verdict_by_id(verdicts, "ZI-02")
        self.assertEqual(case.status, PASSED)
        self.assertEqual(case.observations["delta_tokens"], 0)

    def test_small_jitter_within_threshold_passes(self):
        verdicts, _ = _run(self._by_system(12, 12 + zic.INJECTION_TOKEN_THRESHOLD))
        self.assertEqual(_verdict_by_id(verdicts, "ZI-02").status, PASSED)

    def test_hidden_system_prompt_is_failed(self):
        verdicts, _ = _run(self._by_system(12, 12 + zic.INJECTION_TOKEN_THRESHOLD + 1))
        case = _verdict_by_id(verdicts, "ZI-02")
        self.assertEqual(case.status, FAILED)
        self.assertIn("自行追加", case.detail)

    def test_missing_usage_is_blocked(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = _msg(usage={})
            body.pop("usage")
            return httpx.Response(200, json=body)

        verdicts, _ = _run(handler)
        self.assertEqual(_verdict_by_id(verdicts, "ZI-02").status, BLOCKED)

    def test_request_failure_is_blocked(self):
        verdicts, _ = _run(_always((500, {"error": {"type": "api_error", "message": "temporary"}})))
        self.assertEqual(_verdict_by_id(verdicts, "ZI-02").status, BLOCKED)


# ---------------------------------------------------------------------------
# ZI-03 未声明工具面
# ---------------------------------------------------------------------------

class Zi03Tests(unittest.TestCase):
    def test_plain_text_passes_and_no_tools_are_sent(self):
        sent: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(_body(request))
            return httpx.Response(200, json=_msg(text="杭州是浙江省的省会。"))

        verdicts, _ = _run(handler)
        self.assertEqual(_verdict_by_id(verdicts, "ZI-03").status, PASSED)
        self.assertNotIn("tools", sent[0])
        self.assertNotIn("tool_choice", sent[0])

    def test_undeclared_tool_use_is_failed(self):
        body = _msg(content=[{"type": "tool_use", "id": "t1", "name": "web_search",
                              "input": {}}], stop_reason="tool_use")
        verdicts, _ = _run(_always(body))
        case = _verdict_by_id(verdicts, "ZI-03")
        self.assertEqual(case.status, FAILED)
        self.assertEqual(case.observations["stop_reason"], "tool_use")

    def test_stop_reason_tool_use_alone_is_failed(self):
        verdicts, _ = _run(_always(_msg(text="好的", stop_reason="tool_use")))
        self.assertEqual(_verdict_by_id(verdicts, "ZI-03").status, FAILED)

    def test_unexpected_non_tool_block_is_warning(self):
        verdicts, _ = _run(_always(_msg(content=[{"type": "text", "text": "杭州"},
                                                 {"type": "citation", "text": "x"}])))
        self.assertEqual(_verdict_by_id(verdicts, "ZI-03").status, WARNING)

    def test_empty_text_is_warning(self):
        verdicts, _ = _run(_always(_msg(text="")))
        self.assertEqual(_verdict_by_id(verdicts, "ZI-03").status, WARNING)

    def test_request_failure_is_blocked(self):
        verdicts, _ = _run(_always((401, {"error": {"type": "authentication_error",
                                                    "message": "invalid api key"}})))
        self.assertEqual(_verdict_by_id(verdicts, "ZI-03").status, BLOCKED)


# ---------------------------------------------------------------------------
# ZI-04 签名原生性
# ---------------------------------------------------------------------------

SIGNED_CONTENT = [{"type": "thinking", "thinking": "17×23=391。", "signature": "QUJDREVG"},
                  {"type": "text", "text": "结果是 391。"}]


class Zi04Tests(unittest.TestCase):
    def _client_for(self, first, second=None, model: str = MODEL) -> Client:
        """按请求体路由：thinking 请求返回 first，带 assistant 轮次的回传请求返回 second。

        其余三条用例（金丝雀 / 差分 / 工具面）给中性应答，避免用「按序取预设」的队列
        把 ZI-04 的响应喂给别的用例。
        """
        def handler(request: httpx.Request) -> httpx.Response:
            body = _body(request)
            replay = any(isinstance(message, dict) and message.get("role") == "assistant"
                         for message in body.get("messages") or [])
            if replay:
                item = second
            elif body.get("thinking"):
                item = first
            else:
                return httpx.Response(200, json=_msg(text="OK"))
            if isinstance(item, dict) and "status" in item:
                return httpx.Response(item["status"], json={"error": {"message": item["message"]}})
            return httpx.Response(200, json=item if isinstance(item, dict) else _msg())

        return _client(handler, model=model)

    def test_native_signature_passes(self):
        with self._client_for(_msg(content=SIGNED_CONTENT)) as client:
            verdicts, evidence = zic.run_all(client)
        case = _verdict_by_id(verdicts, "ZI-04")
        self.assertEqual(case.status, PASSED)
        self.assertEqual(case.observations["signature_length"], 8)
        self.assertEqual(len(case.observations["signature_sha256"]), 64)
        signature_request = next(
            row["request"] for row in evidence["requests"]
            if row["phase"] == "signed_thinking_request"
        )
        self.assertEqual(
            signature_request["messages"][0]["content"],
            "请解释快速排序的平均时间复杂度为什么是 O(n log n)，给出递推推导过程。",
        )
        # 回传请求必须原样带上官方 thinking 块
        replay = evidence["requests"][-1]["request"]["messages"][-2]["content"]
        self.assertEqual(replay[0]["signature"], "QUJDREVG")

    def test_cross_family_model_is_failed(self):
        with self._client_for(_msg(model="claude-sonnet-5", content=SIGNED_CONTENT)) as client:
            verdicts, _ = zic.run_all(client)
        case = _verdict_by_id(verdicts, "ZI-04")
        self.assertEqual(case.status, FAILED)
        self.assertIn("异族", case.detail)

    def test_same_family_alias_is_not_failed(self):
        with self._client_for(_msg(model="claude-opus-5-1", content=SIGNED_CONTENT)) as client:
            verdicts, _ = zic.run_all(client)
        case = _verdict_by_id(verdicts, "ZI-04")
        self.assertEqual(case.status, PASSED)
        self.assertIn("model_alias_note", case.observations)

    def test_thinking_without_signature_is_failed(self):
        content = [{"type": "thinking", "thinking": "17×23=391。", "signature": ""},
                   {"type": "text", "text": "391"}]
        with self._client_for(_msg(content=content)) as client:
            verdicts, _ = zic.run_all(client)
        case = _verdict_by_id(verdicts, "ZI-04")
        self.assertEqual(case.status, FAILED)
        self.assertIn("没有非空 signature", case.detail)

    def test_no_thinking_block_is_blocked(self):
        with self._client_for(_msg(text="391")) as client:
            verdicts, _ = zic.run_all(client)
        self.assertEqual(_verdict_by_id(verdicts, "ZI-04").status, BLOCKED)

    def test_signature_replay_rejected_is_failed(self):
        with self._client_for(_msg(content=SIGNED_CONTENT),
                              {"status": 400, "message": "Invalid signature in thinking block"}) as client:
            verdicts, _ = zic.run_all(client)
        case = _verdict_by_id(verdicts, "ZI-04")
        self.assertEqual(case.status, FAILED)
        self.assertIn("原样回传的官方签名被拒绝", case.detail)

    def test_replay_other_error_is_blocked(self):
        with self._client_for(_msg(content=SIGNED_CONTENT),
                              {"status": 401, "message": "invalid api key"}) as client:
            verdicts, _ = zic.run_all(client)
        self.assertEqual(_verdict_by_id(verdicts, "ZI-04").status, BLOCKED)

    def test_thinking_unsupported_is_blocked_with_note(self):
        with self._client_for({"status": 400, "message": "thinking.budget_tokens: unsupported"}) as client:
            verdicts, _ = zic.run_all(client)
        case = _verdict_by_id(verdicts, "ZI-04")
        self.assertEqual(case.status, BLOCKED)
        self.assertIn("未开放 thinking", case.detail)

    def test_haiku_uses_enabled_thinking_mode(self):
        with self._client_for(_msg(content=SIGNED_CONTENT, model="claude-haiku-4-5"),
                              model="claude-haiku-4-5") as client:
            verdicts, _ = zic.run_all(client, thinking_mode=None)
        self.assertEqual(_verdict_by_id(verdicts, "ZI-04").status, PASSED)
        self.assertEqual(_verdict_by_id(verdicts, "ZI-04").observations["thinking_mode"], "enabled")

    def test_explicit_thinking_mode_wins(self):
        with self._client_for(_msg(content=SIGNED_CONTENT)) as client:
            verdicts, _ = zic.run_all(client, thinking_mode="enabled")
        self.assertEqual(_verdict_by_id(verdicts, "ZI-04").observations["thinking_mode"], "enabled")


# ---------------------------------------------------------------------------
# 汇总 / 准入 / 证据契约
# ---------------------------------------------------------------------------

class OutputTests(unittest.TestCase):
    def test_all_passed_is_admitted_and_summary_contract_holds(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = _body(request)
            if body.get("system") or CANARY_RE.search(json.dumps(body, ensure_ascii=False)):
                canary = CANARY_RE.search(json.dumps(body, ensure_ascii=False)).group(0)
                return httpx.Response(200, json=_msg(text=f"口令是 {canary}。"))
            if body.get("thinking"):
                return httpx.Response(200, json=_msg(content=SIGNED_CONTENT))
            return httpx.Response(200, json=_msg(text="杭州是浙江省的省会。"))

        out = Path(tempfile.mkdtemp()) / "zi"
        with _client(handler) as client:
            verdicts, evidence = zic.run_all(client)
            summary = zic.write_outputs(out, MODEL, client.root, verdicts, evidence)
        self.assertTrue(summary["admission_passed"])
        self.assertEqual([v.status for v in verdicts], [PASSED, PASSED, PASSED, PASSED])
        payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "volcperf.result.v2")
        self.assertEqual(payload["case"], "ZERO_INJECTION_CHECK")
        self.assertEqual(payload["script"], "fable5.zero_injection_check")
        self.assertEqual(payload["test_kind"], "fable")
        self.assertEqual(payload["total"], 4)
        self.assertGreater(payload["sent"], 0)
        self.assertEqual({row["case_id"] for row in payload["results"]},
                         {"ZI-01", "ZI-02", "ZI-03", "ZI-04"})
        for row in payload["results"]:
            self.assertIn(row["status"], {"passed", "failed", "warning", "blocked"})
            self.assertIsInstance(row["passed"], bool)
        self.assertTrue((out / "requests.jsonl").read_text(encoding="utf-8").strip())
        self.assertTrue((out / "observations.json").exists())
        report = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("0 注入中转判定组", report)
        self.assertIn("覆盖边界", report)
        # 证据行必须齐备（平台契约）
        row = json.loads((out / "requests.jsonl").read_text(encoding="utf-8").splitlines()[0])
        for key in ("phase", "url", "request", "http_status", "response", "elapsed_seconds",
                    "attempts", "timing", "transport_error"):
            self.assertIn(key, row)

    def test_admission_rule(self):
        def verdict(case_id: str, status: str):
            return zic._verdict(case_id, "t", "input", status, "d")

        passed = verdict("ZI-01", PASSED)
        self.assertFalse(zic.summarize([verdict("ZI-01", FAILED)])["admission_passed"])
        self.assertFalse(zic.summarize([verdict("ZI-01", BLOCKED)])["admission_passed"])
        self.assertFalse(zic.summarize([verdict("ZI-01", WARNING),
                                        verdict("ZI-02", FAILED)])["admission_passed"])
        # blocked 只记录不阻断：只要有有效结论就准入
        self.assertTrue(zic.summarize([passed, verdict("ZI-02", BLOCKED)])["admission_passed"])
        self.assertTrue(zic.summarize([verdict("ZI-01", WARNING)])["admission_passed"])

    def test_failed_case_blocks_end_to_end(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = _body(request)
            if body.get("thinking"):
                return httpx.Response(200, json=_msg(content=SIGNED_CONTENT))
            if body.get("system"):
                # system 金丝雀被网关剥掉：模型看不到口令
                return httpx.Response(200, json=_msg(text="不知道。"))
            if CANARY_RE.search(json.dumps(body, ensure_ascii=False)):
                canary = CANARY_RE.search(json.dumps(body, ensure_ascii=False)).group(0)
                return httpx.Response(200, json=_msg(text=f"口令是 {canary}。"))
            # 缺省 system 时网关自行补了系统提示词 ⇒ ZI-02 必须判 failed
            tokens = 100 if "system" in body else 12
            return httpx.Response(200, json=_msg(text="杭州",
                                                 usage={"input_tokens": tokens, "output_tokens": 2}))

        out = Path(tempfile.mkdtemp()) / "zi2"
        with _client(handler) as client:
            verdicts, evidence = zic.run_all(client)
            summary = zic.write_outputs(out, MODEL, client.root, verdicts, evidence)
        self.assertEqual(_verdict_by_id(verdicts, "ZI-02").status, FAILED)
        self.assertFalse(summary["admission_passed"])


class SelfTestTests(unittest.TestCase):
    def test_self_test_runs_offline(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = zic.main(["--self-test"])
        self.assertEqual(code, 0)
        self.assertIn("全部通过", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
