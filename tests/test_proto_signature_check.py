import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from common.finalizer import finalize_result_dir
from fable5 import proto_signature_check as psc
from fable5.proto_probe import Client
from fable5.signature_structure import synthetic_signature


def message(model="claude-opus-5", signature=synthetic_signature()):
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "thinking", "thinking": "原始思考内容", "signature": signature},
                        {"type": "text", "text": "391"}], "stop_reason": "end_turn"}


# 实测形态（results/PROTOBUF_SIGNATURE_CHECK-20261015-134446-126000）：非流式请求收到
# 200，但包体是网关自造的 SSE 空壳——id 里嵌响应时刻时间戳、content 空、stop_reason 空、
# usage 全 0。它不是 Messages JSON，也不是模型的任何产出。
GATEWAY_SHELL = (
    'event: message_start\n'
    'data: {"type":"message_start","message":{"id":"msg_fallback_1792043113",'
    '"type":"message","role":"assistant","model":"claude-sonnet-5-5","content":[],'
    '"stop_reason":null,"usage":{"input_tokens":0,"output_tokens":0}}}\n\n'
    'event: message_stop\n\n'
)


def run_sequence(responses, model="claude-opus-5", mode="adaptive", max_retries=0, backoff=0.0,
                 max_tokens=4096):
    """``responses`` 每项为 ``(status, body)``：``body`` 是 dict ⇒ 按 JSON 返回；
    是 str ⇒ 按原样字节返回（模拟网关 SSE 空壳等非 Messages JSON 包体）；
    是 Exception 实例 ⇒ 直接抛出（模拟断链）。"""
    requests = []
    queue = list(responses)

    def handler(request):
        requests.append(json.loads(request.content))
        status, body = queue.pop(0)
        if isinstance(body, Exception):
            raise body
        if isinstance(body, str):
            return httpx.Response(status, content=body.encode("utf-8"),
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(status, json=body)

    with Client("https://example.com", "secret", model=model, max_retries=max_retries,
                backoff=backoff, transport=httpx.MockTransport(handler)) as client:
        verdicts, evidence = psc.run_all(client, thinking_mode=mode, max_tokens=max_tokens)
    return verdicts, evidence, requests


class SignatureCheckTests(unittest.TestCase):
    def test_real_signature_replay_and_single_character_mutation(self):
        for model in ("claude-fable-5", "claude-opus-5", "claude-sonnet-4-6", "relay-custom"):
            with self.subTest(model=model):
                first = message(model)
                vs, evidence, requests = run_sequence([
                    (200, first), (200, message(model)),
                    (400, {"error": {"type": "invalid_request_error", "message": "Invalid signature in thinking block"}}),
                ], model=model)
                self.assertEqual([v.status for v in vs], ["passed", "passed", "passed"])
                self.assertEqual(len(requests), 3)
                self.assertTrue(all(r["model"] == model for r in requests))
                self.assertEqual(requests[1]["messages"][1]["content"], first["content"])
                original = requests[1]["messages"][1]["content"][0]["signature"]
                changed = requests[2]["messages"][1]["content"][0]["signature"]
                self.assertEqual(len(original), len(changed))
                self.assertEqual(sum(a != b for a, b in zip(original, changed)), 1)
                restored = copy.deepcopy(requests[2])
                restored["messages"][1]["content"][0]["signature"] = original
                self.assertEqual(restored, requests[1])
                # 不留本地 protobuf 快照：契约校验直接读真实 Messages JSON 字段。
                self.assertNotIn("original_protobuf", evidence)
                self.assertNotIn("mutated_protobuf", evidence)
                self.assertEqual(evidence["original_response"]["model"], model)
                self.assertEqual(evidence["original_response"]["id"], "msg_test")
                self.assertEqual(evidence["mutation"]["changed_characters"], 1)
                self.assertTrue(evidence["mutation"]["model_unchanged"])
                self.assertEqual(first, message(model))

    def test_missing_signature_blocks_without_fabrication(self):
        first = message()
        first["content"][0].pop("signature")
        second = message()
        second["content"][0].pop("signature")
        vs, evidence, requests = run_sequence([(200, first), (200, second)])
        self.assertEqual([v.status for v in vs], ["blocked", "blocked", "blocked"])
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["messages"][0]["content"], psc.BALL_PROMPT)
        self.assertEqual(requests[1]["messages"][0]["content"], psc.QUICKSORT_PROMPT)
        self.assertNotIn("mutation", evidence)
        self.assertNotIn("signature_structure", evidence)

    def test_falls_back_to_quicksort_prompt_when_ball_prompt_has_no_signature(self):
        first = message()
        first["content"][0].pop("signature")
        second = message()
        vs, evidence, requests = run_sequence([
            (200, first), (200, second), (200, message()),
            (400, {"error": {"message": "Invalid signature in thinking block"}}),
        ])
        self.assertEqual([v.status for v in vs], ["passed", "passed", "passed"])
        self.assertEqual(len(requests), 4)
        self.assertEqual(requests[0]["messages"][0]["content"], psc.BALL_PROMPT)
        self.assertEqual(requests[1]["messages"][0]["content"], psc.QUICKSORT_PROMPT)
        self.assertEqual(requests[2]["messages"][0]["content"], psc.QUICKSORT_PROMPT)
        self.assertEqual(requests[3]["messages"][0]["content"], psc.QUICKSORT_PROMPT)
        self.assertEqual(evidence["signature_prompt"], {"name": "quicksort", "text": psc.QUICKSORT_PROMPT})

    def test_other_thinking_signatures_and_content_are_preserved(self):
        first = message()
        first["content"].insert(1, {"type": "thinking", "thinking": "第二块思考", "signature": "U0VDT05E"})
        first["content"].insert(2, {"type": "redacted_thinking", "data": "opaque-data"})
        vs, _, requests = run_sequence([
            (200, first), (200, message()),
            (400, {"error": {"message": "Invalid signature in thinking block"}}),
        ])
        self.assertEqual([v.status for v in vs], ["passed", "passed", "passed"])
        self.assertEqual(requests[2]["messages"][1]["content"][1:], first["content"][1:])

    def test_initial_transport_and_unsupported_thinking_are_blocked(self):
        vs, _, requests = run_sequence([(400, {"error": {"message": "thinking not supported"}})])
        self.assertEqual([v.status for v in vs], ["blocked", "blocked", "blocked"])
        self.assertEqual(len(requests), 1)
        def unreachable(request):
            raise httpx.ConnectError("unreachable", request=request)
        with Client("https://example.com", "secret", model="claude-opus-5", max_retries=0,
                    transport=httpx.MockTransport(unreachable)) as client:
            vs, evidence = psc.run_all(client)
        self.assertEqual([v.status for v in vs], ["blocked", "blocked", "blocked"])
        self.assertEqual(len(evidence["requests"]), 1)

    def test_response_contract_failures_stop_replay(self):
        for mutate in (lambda m: m.update(model="other-model"),
                       lambda m: m.update(id=""),
                       lambda m: m["content"][1].update(text=""),
                       lambda m: m.update(content=[])):
            first = message()
            mutate(first)
            vs, _, requests = run_sequence([(200, first)])
            self.assertEqual([v.status for v in vs], ["failed", "blocked", "blocked"])
            self.assertEqual(len(requests), 1)

    def test_original_replay_rejection_does_not_pass_negative_control(self):
        vs, _, requests = run_sequence([(200, message()), (400, {"error": {"message": "Invalid signature"}})])
        # 原签名被拒 ⇒ 两步契约不成立；但签名已取得，SIG-03 仍能独立读出结构。
        self.assertEqual([v.status for v in vs], ["blocked", "blocked", "passed"])
        self.assertEqual(len(requests), 2)

    def test_mutation_error_must_be_specific(self):
        for status, body, expected in (
            (200, message(), "failed"),
            (400, {"error": {"message": "max_tokens too small"}}, "blocked"),
            (400, {"error": {"message": "signature field is not supported"}}, "blocked"),
            (401, {"error": {"message": "Invalid signature"}}, "blocked"),
            (429, {"error": {"message": "Invalid signature"}}, "blocked"),
            (500, {"error": {"message": "Invalid signature"}}, "blocked"),
            (422, {"error": {"message": "thinking.signature: verification failed"}}, "passed"),
        ):
            vs, _, _ = run_sequence([(200, message()), (200, message()), (status, body)])
            self.assertEqual(vs[1].status, expected, (status, body))

    def test_gateway_signature_error_with_backticks_is_recognized(self):
        gateway_error = {"error": {"type": "<nil>", "message":
            "***.***.content.0: Invalid `signature` in `thinking` block (request id: test-request)"}}
        vs, _, requests = run_sequence([(200, message()), (200, message()), (400, gateway_error)])
        self.assertEqual([v.status for v in vs], ["passed", "passed", "passed"])
        self.assertEqual(vs[1].observations["error"], gateway_error["error"])
        self.assertEqual(len(requests), 3)

    def test_transient_disconnect_is_retried_before_blocking(self):
        """一次断链不再直接判 blocked：先按 max_retries 重发，重发成功即正常走完两步。"""
        vs, evidence, requests = run_sequence([
            (0, httpx.RemoteProtocolError("Server disconnected without sending a response")),
            (200, message()), (200, message()),
            (400, {"error": {"message": "Invalid signature in thinking block"}}),
        ], max_retries=2)
        self.assertEqual([v.status for v in vs], ["passed", "passed", "passed"])
        self.assertEqual(len(requests), 4, "断链 1 次 + 重发 1 次 + 回传 + 变异")
        self.assertEqual(evidence["requests"][0]["attempts"], 2, "同一次逻辑请求内重试 2 次尝试")
        self.assertIsNone(evidence["requests"][0]["transport_error"])
        self.assertEqual(sum(row["attempts"] for row in evidence["requests"]), 4)

    def test_gateway_shell_is_retried_then_passes(self):
        """HTTP 200 + 网关自造 SSE 空壳属通道侧一次性异常，重发即可恢复正常结论。"""
        vs, evidence, requests = run_sequence([
            (200, GATEWAY_SHELL), (200, message()), (200, message()),
            (400, {"error": {"message": "Invalid signature in thinking block"}}),
        ], max_retries=1)
        self.assertEqual([v.status for v in vs], ["passed", "passed", "passed"])
        self.assertEqual(len(requests), 4, "空壳 1 次 + 重发 1 次 + 回传 + 变异")
        self.assertEqual(evidence["requests"][0]["http_status"], 200)
        self.assertIsNone(evidence["requests"][0]["response"])
        self.assertIn("msg_fallback_", evidence["requests"][0]["raw_response"])
        self.assertEqual(evidence["requests"][0]["content_type"], "text/event-stream")

    def test_gateway_shell_after_retries_is_blocked_not_failed(self):
        """重试耗尽仍只有空壳 ⇒ 通道侧异常，记 blocked（无依据），绝不记 failed。"""
        vs, evidence, requests = run_sequence([(200, GATEWAY_SHELL), (200, GATEWAY_SHELL)],
                                              max_retries=1)
        self.assertEqual([v.status for v in vs], ["blocked", "blocked", "blocked"])
        self.assertEqual(len(requests), 2, "重试耗尽即停，不发送回传/变异请求")
        self.assertEqual(vs[0].observations["shell_retries"], 1)
        self.assertIn("msg_fallback_", vs[0].observations["raw_response_head"])
        self.assertEqual(vs[0].observations["content_type"], "text/event-stream")
        self.assertIn("通道侧异常", vs[0].detail)
        self.assertNotIn("mutation", evidence)

    def test_negative_mutation_is_never_retried_on_5xx(self):
        """负向变异请求一次定型：5xx 被重试成 200 会被读成「该拒未拒」的假失败。"""
        vs, evidence, requests = run_sequence([
            (200, message()), (200, message()), (500, {"error": {"message": "boom"}}),
        ], max_retries=2)
        self.assertEqual([v.status for v in vs], ["passed", "blocked", "passed"])
        mutated = evidence["requests"][-1]
        self.assertEqual(mutated["phase"], "mutated_signature_replay")
        self.assertEqual(mutated["attempts"], 1)
        self.assertEqual(len(requests), 3)

    def test_negative_mutation_resends_after_connection_error(self):
        """纯连接错可以重发（结论语义不变），不把一次闪断变成整批 blocked。"""
        vs, evidence, requests = run_sequence([
            (200, message()), (200, message()),
            (0, httpx.RemoteProtocolError("Server disconnected without sending a response")),
            (400, {"error": {"message": "Invalid signature in thinking block"}}),
        ], max_retries=2)
        self.assertEqual([v.status for v in vs], ["passed", "passed", "passed"])
        mutated = evidence["requests"][-1]
        self.assertEqual(mutated["attempts"], 2)
        self.assertEqual(len(requests), 4)

    def test_enabled_thinking_for_haiku(self):
        model = "claude-haiku-4-5"
        vs, _, requests = run_sequence([
            (200, message(model)), (200, message(model)),
            (400, {"error": {"message": "Invalid signature in thinking block"}}),
        ], model=model, mode="enabled")
        self.assertEqual(requests[0]["thinking"], {"type": "enabled", "budget_tokens": 1024})
        self.assertEqual([v.status for v in vs], ["passed", "passed", "passed"])

    def test_adaptive_effort_sends_output_config(self):
        # 5 系只认 adaptive + output_config.effort：两个字段都必须在请求体里。
        model = "claude-sonnet-5"
        vs, _, requests = run_sequence([
            (200, message(model)), (200, message(model)),
            (400, {"error": {"message": "Invalid signature in thinking block"}}),
        ], model=model, mode="adaptive_effort")
        self.assertEqual(requests[0]["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertEqual(requests[0]["output_config"], {"effort": "high"})
        # 回传请求复用初次请求体：effort 必须一起带回去，否则上游按另一档重解会拒。
        self.assertEqual(requests[1]["output_config"], {"effort": "high"})
        self.assertEqual([v.status for v in vs], ["passed", "passed", "passed"])

    def test_adaptive_payload_has_no_output_config(self):
        vs, _, requests = run_sequence([
            (200, message("claude-fable-5")), (200, message("claude-fable-5")),
            (400, {"error": {"message": "Invalid signature in thinking block"}}),
        ], model="claude-fable-5", mode="adaptive")
        self.assertNotIn("output_config", requests[0])
        self.assertEqual([v.status for v in vs], ["passed", "passed", "passed"])

    def test_bedrock_model_spelling_passes_and_is_recorded(self):
        # 实测形态（results/rerun-a3b2/）：Bedrock 通道把 model 写成
        # anthropic.claude-haiku-4-5-20251001-v1:0，与请求名是同一模型的两种写法。
        # 这不该判 failed（响应有签名有正文），但必须在 detail 与 observations 留下原样写法。
        model = "claude-haiku-4-5-20251001"
        bedrock = "anthropic.claude-haiku-4-5-20251001-v1:0"
        vs, evidence, _ = run_sequence([
            (200, message(bedrock)), (200, message(bedrock)),
            (400, {"error": {"message": "Invalid signature in thinking block"}}),
        ], model=model, mode="enabled")
        self.assertEqual([v.status for v in vs], ["passed", "passed", "passed"])
        self.assertIn(bedrock, vs[0].detail)
        self.assertIn("归一等价", vs[0].detail)
        self.assertEqual(vs[0].observations["response_model"], bedrock)
        self.assertEqual(vs[0].observations["expected_model"], model)
        self.assertIn("model_field_alias", vs[0].observations)

    def test_bedrock_spelling_of_another_model_still_fails(self):
        # 归一不能变成"什么都放行"：请求 haiku 回 opus 的 Bedrock 写法必须仍判 failed。
        vs, _, requests = run_sequence([
            (200, message("anthropic.claude-opus-4-5-20251101-v1:0")),
        ], model="claude-haiku-4-5-20251001", mode="enabled")
        self.assertEqual([v.status for v in vs], ["failed", "blocked", "blocked"])
        self.assertEqual(len(requests), 1)

    def test_max_tokens_budget_is_honoured(self):
        # 预算给少了思考会吃满上限、正文为零 ⇒ 假 failed；给足是前提，必须能调。
        vs, evidence, requests = run_sequence([
            (200, message()), (200, message()),
            (400, {"error": {"message": "Invalid signature in thinking block"}}),
        ], max_tokens=16000)
        self.assertEqual(requests[0]["max_tokens"], 16000)
        self.assertEqual(evidence["max_tokens"], 16000)
        self.assertEqual([v.status for v in vs], ["passed", "passed", "passed"])

    def test_cli_resolves_thinking_mode_and_budget_by_model(self):
        """不显式给 --thinking-mode 时按模型名解析档位；预算默认 16000。

        这是「界面不再手选档位」的前提：CLI 必须自己认得出模型族，
        否则又回到「4.5 系发 adaptive ⇒ 400」的老坑。
        """
        cases = [("claude-haiku-4-5-20251001", "enabled"),
                 ("claude-opus-4-5-20251101", "enabled"),
                 ("claude-sonnet-4-6", "enabled"),
                 ("claude-sonnet-5", "adaptive_effort"),
                 ("claude-opus-5-5", "adaptive_effort"),
                 ("claude-fable-5", "adaptive")]

        class FakeClient:
            method = "POST"

            def __init__(self, *args, **kwargs):
                self.root = "https://example.com"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        for model, expected_mode in cases:
            seen = {}

            def fake_run_all(client, thinking_mode="adaptive", *, allow_prompt_fallback=True,
                             max_tokens=4096):
                seen["mode"] = thinking_mode
                seen["max_tokens"] = max_tokens
                return [], {"requests": []}

            with self.subTest(model=model), tempfile.TemporaryDirectory() as folder:
                with patch.object(psc, "Client", FakeClient), \
                     patch.object(psc, "run_all", fake_run_all):
                    psc.main(["--model", model, "--base-url", "https://example.com",
                              "--api-key", "k", "--out", folder])
                self.assertEqual(seen["mode"], expected_mode)
                self.assertEqual(seen["max_tokens"], 16000)

    def test_output_and_finalizer_require_both_steps(self):
        for last_status, last_body, expected in (
            (400, {"error": {"message": "Invalid signature in thinking block"}}, True),
            (400, {"error": {"message": "signature not supported"}}, False),
        ):
            vs, evidence, _ = run_sequence([(200, message()), (200, message()), (last_status, last_body)])
            with tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                psc.write_outputs(out, "claude-opus-5", "https://example.com/v1", vs, evidence)
                summary = finalize_result_dir(out)
                self.assertEqual(summary["admission_passed"], expected)
                self.assertEqual(summary["case"], "PROTOBUF_SIGNATURE_CHECK")
                self.assertEqual(summary["total"], 3)
                self.assertEqual(summary["sent"], 3)
                self.assertTrue((out / "response.json").is_file())
                self.assertTrue((out / "signature_structure.json").is_file())
                # 响应不再落本地 protobuf 快照（2026-10-15 按需求移除）：目录里必须没有 .pb。
                self.assertEqual(sorted(out.glob("*.pb")), [])
                self.assertNotIn("protobuf_source", summary)
                self.assertNotIn("secret", (out / "requests.jsonl").read_text())

    def test_signature_structure_reads_protobuf_without_extra_requests(self):
        """SIG-03 只读签名：解码 → 严格解析 → 三项核对，不额外发请求。"""
        vs, evidence, requests = run_sequence([
            (200, message()), (200, message()),
            (400, {"error": {"message": "Invalid signature in thinking block"}}),
        ])
        self.assertEqual([v.status for v in vs], ["passed", "passed", "passed"])
        self.assertEqual(len(requests), 3, "结构层核对不发请求")
        self.assertEqual(vs[2].case_id, "SIG-03")
        entry = evidence["signature_structure"]
        self.assertTrue(entry["strictly_parseable"])
        self.assertTrue(entry["reencoded_equal"])
        self.assertEqual(entry["byte_length"], 120)
        self.assertEqual(entry["version"], 4)
        self.assertEqual(entry["top_fields"], [1, 2, 3])
        self.assertTrue(entry["known_layout"])
        self.assertEqual(entry["block_type"], "thinking")
        self.assertEqual(entry["block_type_matches"], True)
        self.assertEqual(entry["inner_field_numbers"], [1, 2, 3, 4, 5])
        self.assertEqual(entry["fixed_lengths"], {"2": 12, "3": 12, "4": 48, "5": 16})
        self.assertEqual(entry["cipher_length"], 16)
        self.assertIsNone(entry["model_identifier"], "合成样本不含模型标识明文")
        self.assertEqual(entry["failures"], [])
        self.assertIn({"path": ".2.1.8", "text": "thinking"}, entry["readable_strings"])
        self.assertEqual(vs[2].observations["structure_failures"], [])
        self.assertNotIn("readable_strings", vs[2].observations)

    def test_unparseable_signature_is_warning_and_does_not_block_admission(self):
        """签名不是合法 protobuf ⇒ SIG-03 warning 留痕；准入只认两步回传契约。"""
        junk = "QUJDREVGR0hJSktMTU5PUA=="
        vs, evidence, _ = run_sequence([
            (200, message(signature=junk)), (200, message(signature=junk)),
            (400, {"error": {"message": "Invalid signature in thinking block"}}),
        ])
        self.assertEqual([v.status for v in vs], ["passed", "passed", "warning"])
        self.assertFalse(evidence["signature_structure"]["strictly_parseable"])
        self.assertTrue(vs[2].observations["structure_failures"])
        self.assertIn("不是合法 protobuf", vs[2].detail)
        self.assertIn("不判 failed", vs[2].detail)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            psc.write_outputs(out, "claude-opus-5", "https://example.com/v1", vs, evidence)
            summary = finalize_result_dir(out)
        self.assertTrue(summary["admission_passed"], "warning 留痕不阻断两步契约的准入")
        self.assertEqual(summary["warning"], 1)
        self.assertEqual(summary["passed"], 2)

    def test_report_lists_decoded_signature_plaintext(self):
        """报告要如实列出解码后的可读明文：字段路径 + 内容 + 归因。

        签名内写 `claude-opus-5`、请求 `claude-fable-5-1` ⇒ 标识不一致：SIG-03 记 warning，
        报告里逐段标出不一致并写明是取证项。
        """
        signature = synthetic_signature(model_identifier="claude-opus-5")
        model = "claude-fable-5-1"
        vs, evidence, _ = run_sequence([
            (200, message(model, signature)), (200, message(model, signature)),
            (400, {"error": {"message": "Invalid signature in thinking block"}}),
        ], model=model)
        self.assertEqual([v.status for v in vs], ["passed", "passed", "warning"])
        entry = evidence["signature_structure"]
        self.assertIsNone(entry["model_identifier"], "标识不等于请求模型 ⇒ 不当作匹配")
        self.assertEqual(entry["model_identifiers"], ["claude-opus-5"])
        self.assertEqual(entry["expected_model"], model)
        self.assertIs(entry["model_match"], False)
        self.assertIn("模型标识", vs[2].detail)
        self.assertIn("不阻断准入", vs[2].detail)
        self.assertEqual(vs[2].observations["model_consistency_failures"],
                         [f"签名内模型标识 claude-opus-5 与请求模型 '{model}' 不一致"])
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            psc.write_outputs(out, model, "https://example.com/v1", vs, evidence)
            report = (out / "report.md").read_text()
        self.assertIn("## 签名解码明文", report)
        self.assertIn("| `.2.1.6` | `claude-opus-5` |", report)
        self.assertIn("| `.2.1.8` | `thinking` |", report)
        self.assertIn("与请求模型 `claude-fable-5-1` 不一致", report)
        self.assertIn("记 warning", report)
        self.assertIn("块类型描述符", report)
        self.assertIn("密文", report)

    def test_signature_model_identifier_match_passes(self):
        """签名内的模型标识与请求模型一致 ⇒ SIG-03 passed，并在结论里写明命中哪个标识。"""
        model = "claude-opus-5"
        signature = synthetic_signature(model_identifier=model)
        vs, evidence, _ = run_sequence([
            (200, message(model, signature)), (200, message(model, signature)),
            (400, {"error": {"message": "Invalid signature in thinking block"}}),
        ], model=model)
        self.assertEqual([v.status for v in vs], ["passed", "passed", "passed"])
        entry = evidence["signature_structure"]
        self.assertEqual(entry["model_identifier"], model)
        self.assertEqual(entry["model_identifiers"], [model])
        self.assertIs(entry["model_match"], True)
        self.assertEqual(entry["failures"], [])
        self.assertEqual(vs[2].observations["model_consistency_failures"], [])
        self.assertIn(f"签名内含模型标识 {model}，与请求模型 {model} 一致", vs[2].detail)

    def test_signature_model_identifier_alias_is_not_a_warning(self):
        """按 `-` 边界的别名/版本写法算一致：签名 `claude-fable-5`、请求 `claude-fable-5-1`。"""
        signature = synthetic_signature(model_identifier="claude-fable-5")
        model = "claude-fable-5-1"
        vs, evidence, _ = run_sequence([
            (200, message(model, signature)), (200, message(model, signature)),
            (400, {"error": {"message": "Invalid signature in thinking block"}}),
        ], model=model)
        self.assertEqual([v.status for v in vs], ["passed", "passed", "passed"])
        entry = evidence["signature_structure"]
        self.assertIs(entry["model_match"], True)
        self.assertIsNone(entry["model_identifier"], "别名写法不算逐字命中")
        self.assertIn("视为同一型号", vs[2].detail)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            psc.write_outputs(out, model, "https://example.com/v1", vs, evidence)
            report = (out / "report.md").read_text()
        self.assertIn("别名/版本写法", report)
        self.assertNotIn("不一致 ⇒ 记 warning", report)

    def test_signature_without_model_identifier_has_no_conclusion(self):
        """签名里没有模型标识明文（实测 fable 系列如此）⇒ 标识一致性无结论，不构成 warning。"""
        model = "claude-fable-5"
        vs, evidence, _ = run_sequence([
            (200, message(model)), (200, message(model)),
            (400, {"error": {"message": "Invalid signature in thinking block"}}),
        ], model=model)
        self.assertEqual([v.status for v in vs], ["passed", "passed", "passed"])
        entry = evidence["signature_structure"]
        self.assertEqual(entry["model_identifiers"], [])
        self.assertIsNone(entry["model_match"])
        self.assertIn("标识一致性无结论", vs[2].detail)
        self.assertEqual(vs[2].observations["model_consistency_failures"], [])

    def test_model_identifier_mismatch_does_not_block_admission(self):
        """标识不一致只记 warning：准入仍由两步回传契约决定。"""
        signature = synthetic_signature(model_identifier="claude-sonnet-5")
        model = "claude-opus-5"
        vs, evidence, _ = run_sequence([
            (200, message(model, signature)), (200, message(model, signature)),
            (400, {"error": {"message": "Invalid signature in thinking block"}}),
        ], model=model)
        self.assertEqual([v.status for v in vs], ["passed", "passed", "warning"])
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            psc.write_outputs(out, model, "https://example.com/v1", vs, evidence)
            summary = finalize_result_dir(out)
        self.assertTrue(summary["admission_passed"])
        self.assertEqual(summary["warning"], 1)
        self.assertEqual(summary["passed"], 2)

    def test_report_shows_no_plaintext_when_signature_missing(self):
        """未取得签名时报告明确写「没有可解码的明文内容」，不编造明文。"""
        first = message()
        first["content"][0].pop("signature")
        second = message()
        second["content"][0].pop("signature")
        vs, evidence, _ = run_sequence([(200, first), (200, second)])
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            psc.write_outputs(out, "claude-opus-5", "https://example.com/v1", vs, evidence)
            report = (out / "report.md").read_text()
        self.assertIn("## 签名解码明文", report)
        self.assertIn("未取得签名，没有可解码的明文内容", report)
        self.assertNotIn("| `.2.1.8` |", report)

    def test_block_type_mismatch_is_reported(self):
        """签名声明的块类型与响应块不一致 ⇒ 结构层记问题（仍不判 failed）。"""
        mismatched = synthetic_signature(block_type="text")
        vs, evidence, _ = run_sequence([
            (200, message(signature=mismatched)), (200, message(signature=mismatched)),
            (400, {"error": {"message": "Invalid signature in thinking block"}}),
        ])
        self.assertEqual([v.status for v in vs], ["passed", "passed", "warning"])
        self.assertEqual(evidence["signature_structure"]["block_type"], "text")
        self.assertFalse(evidence["signature_structure"]["block_type_matches"])
        self.assertIn("块类型", vs[2].detail)
        self.assertEqual(evidence["requests"][2]["phase"], "mutated_signature_replay")
        self.assertEqual(
            vs[1].observations["mutated_signature_still_parseable"],
            psc.read_signature(
                evidence["requests"][2]["request"]["messages"][1]["content"][0]["signature"],
                expected_block_type="thinking").strictly_parseable)
