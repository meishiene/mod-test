"""测试面分层（model / verify / channel）的口径测试。

三件事必须钉死，否则「默认不跑通道面」会静默退化成「默认什么都不跑」或
「通道面照跑但把准入判死」：

1. **登记完整性**：每个套件的 ``CASE_SPECS`` 都要在 ``<suite>/faces.py`` 里有面归属；
2. **与判据同向**：已在 relay_profile 里认定为「网关缺口」的用例，必须在 faces 里是
   ``channel``（否则默认执行 + 失败阻断，会把网关问题读成模型问题）；
3. **判定**：只有 model/verify 面的硬失败阻断准入，通道面失败一律只记 ``channel_gap``。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from common.faces import (ALL_FACES, DEFAULT_FACES, FACE_CHANNEL, FACE_MODEL,
                          FACE_VERIFY, FaceSelectionError, annotate_summary,
                          face_of, face_verdict, filter_specs, parse_faces,
                          selected_faces)
from common.finalizer import finalize_result_dir
from common.result_adapter import normalize_summary


class _Row:
    def __init__(self, case_id: str, status: str, name: str = ""):
        self.case_id = case_id
        self.status = status
        self.name = name or case_id
        self.details = ""


class _Spec:
    def __init__(self, case_id: str):
        self.case_id = case_id


class FaceParsingTests(unittest.TestCase):
    def test_default_is_model_and_verify(self):
        self.assertEqual(parse_faces(None), DEFAULT_FACES)
        self.assertEqual(parse_faces(""), DEFAULT_FACES)
        self.assertNotIn(FACE_CHANNEL, DEFAULT_FACES)

    def test_all_expands_to_every_face(self):
        self.assertEqual(parse_faces("all"), ALL_FACES)
        self.assertEqual(parse_faces("*"), ALL_FACES)

    def test_unknown_face_is_rejected_instead_of_silently_dropped(self):
        # 拼错面名会让整批静默空跑，必须显式拦回。
        with self.assertRaises(FaceSelectionError):
            parse_faces("modle")
        with self.assertRaises(FaceSelectionError):
            parse_faces("model,verify,channel,vision")

    def test_order_is_stable(self):
        self.assertEqual(parse_faces("channel,model"), (FACE_MODEL, FACE_CHANNEL))

    def test_selected_faces_reads_env(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"VOLC_FACES": "all"}):
            self.assertEqual(selected_faces(), ALL_FACES)
        with patch.dict(os.environ, {"VOLC_FACES": ""}):
            self.assertEqual(selected_faces(), DEFAULT_FACES)

    def test_unregistered_case_defaults_to_model(self):
        self.assertEqual(face_of({}, "NEVER-REGISTERED"), FACE_MODEL)


class RegistryCompletenessTests(unittest.TestCase):
    """每个套件的每条 CASE_SPEC 都必须有面归属。"""

    def _check(self, specs, table, label: str):
        ids = {spec.case_id for spec in specs}
        self.assertEqual(ids - set(table), set(), f"{label}: 有用例未登记测试面")
        self.assertEqual(set(table) - ids, set(), f"{label}: 登记了不存在的用例")
        self.assertTrue(set(table.values()) <= set(ALL_FACES), f"{label}: 出现未知面")

    def test_dsv4(self):
        from dsv4.core import CASE_SPECS
        from dsv4.faces import CASE_FACES
        self._check(CASE_SPECS, CASE_FACES, "dsv4")

    def test_qwen(self):
        from qwen.core import CASE_SPECS
        from qwen.faces import CASE_FACES
        self._check(CASE_SPECS, CASE_FACES, "qwen")

    def test_glm53(self):
        from glm53.core import CASE_SPECS
        from glm53.faces import CASE_FACES
        self._check(CASE_SPECS, CASE_FACES, "glm53")

    def test_fable5(self):
        from fable5.core import CASE_SPECS
        from fable5.faces import CASE_FACES
        self._check(CASE_SPECS, CASE_FACES, "fable5")

    def test_gpt56_all_variants(self):
        from gpt56.core import CASE_SPECS
        from gpt56.faces import CASE_FACES, UNREGISTERED
        self._check(CASE_SPECS, CASE_FACES, "gpt56")
        # handler 未登记会被兜底成 model（不静默漏跑），但必须能在测试里看见。
        self.assertEqual(UNREGISTERED, ())

    def test_gemini3(self):
        from gemini3.core import CASE_SPECS
        from gemini3.faces import CASE_FACES
        self._check(CASE_SPECS, CASE_FACES, "gemini3")

    def test_kvv_cards(self):
        from kvv.cases import CASES as KVV_CASES, TEMP_CASE_ID
        from kvv.faces import CASE_FACES
        self.assertEqual(set(KVV_CASES) | {TEMP_CASE_ID}, set(CASE_FACES))

    def test_default_selection_is_strictly_smaller(self):
        """默认执行集必须真的比全量小：否则「减少通道面」等于没做。"""
        for module, specs_module in (
            ("dsv4.faces", "dsv4.core"), ("qwen.faces", "qwen.core"),
            ("glm53.faces", "glm53.core"), ("fable5.faces", "fable5.core"),
            ("gpt56.faces", "gpt56.core"), ("gemini3.faces", "gemini3.core"),
        ):
            import importlib
            table = importlib.import_module(module).CASE_FACES
            specs = importlib.import_module(specs_module).CASE_SPECS
            default = filter_specs(specs, table, DEFAULT_FACES)
            self.assertLess(len(default), len(specs), f"{module}: 默认集没有缩小")

    def test_gpt_channel_faces_are_the_same_across_variants(self):
        """三档共用 handler 池：同名 handler 的面归属必须一致（不得按档位漂移）。"""
        from gpt56 import faces as gpt_faces
        from gpt56.core import CASE_SPECS
        by_handler: dict[str, set[str]] = {}
        for spec in CASE_SPECS:
            by_handler.setdefault(spec.handler, set()).add(gpt_faces.CASE_FACES[spec.case_id])
        drifted = {h: f for h, f in by_handler.items() if len(f) > 1}
        self.assertEqual(drifted, {})


class RelayConsistencyTests(unittest.TestCase):
    """已判为「网关缺口」的用例，默认执行集里不能再出现（否则失败会被判死）。"""

    def test_dsv4_relay_gaps_are_channel(self):
        from dsv4.faces import CASE_FACES
        from dsv4.relay_profile import CASE_TO_FACE
        missing = [cid for cid in CASE_TO_FACE if CASE_FACES.get(cid) != FACE_CHANNEL]
        self.assertEqual(missing, [], f"这些 relay_gap 用例没标成 channel：{missing}")

    def test_qwen_blocking_faces_are_model_or_verify(self):
        from qwen.faces import CASE_FACES
        from qwen.relay_profile import BLOCKING_FACES, CASE_FACES as RELAY_FACES
        wrong = [cid for cid, relay_face in RELAY_FACES.items()
                 if relay_face in BLOCKING_FACES
                 and CASE_FACES.get(cid) not in {FACE_MODEL, FACE_VERIFY}]
        self.assertEqual(wrong, [], f"这些阻断面用例被错标成 channel：{wrong}")


class FaceVerdictTests(unittest.TestCase):
    def setUp(self):
        self.table = {"M-1": FACE_MODEL, "V-1": FACE_VERIFY, "C-1": FACE_CHANNEL}

    def test_channel_failure_is_recorded_not_blocking(self):
        rows = [_Row("M-1", "passed"), _Row("C-1", "failed"), _Row("C-1", "error")]
        verdict = face_verdict(rows, self.table, DEFAULT_FACES)
        self.assertTrue(verdict["admission_passed"])
        self.assertEqual(verdict["blocking"], [])
        self.assertEqual(len(verdict["channel_gap"]), 2)
        self.assertIn("不阻断准入", verdict["summary_line"])

    def test_model_failure_blocks(self):
        rows = [_Row("M-1", "failed"), _Row("C-1", "passed")]
        verdict = face_verdict(rows, self.table, ALL_FACES)
        self.assertFalse(verdict["admission_passed"])
        self.assertEqual([r["case_id"] for r in verdict["blocking"]], ["M-1"])

    def test_verify_failure_blocks(self):
        rows = [_Row("V-1", "error"), _Row("C-1", "passed")]
        verdict = face_verdict(rows, self.table, ALL_FACES)
        self.assertFalse(verdict["admission_passed"])

    def test_blocked_model_case_does_not_block_but_needs_effective_evidence(self):
        # 与既有口径一致：blocked 不阻断，但整批没有 passed/warning 时仍不准入。
        only_blocked = face_verdict([_Row("M-1", "blocked")], self.table, ALL_FACES)
        self.assertFalse(only_blocked["admission_passed"])
        mixed = face_verdict([_Row("M-1", "blocked"), _Row("C-1", "passed")], self.table, ALL_FACES)
        self.assertTrue(mixed["admission_passed"])

    def test_channel_only_batch_does_not_claim_model_verdict(self):
        rows = [_Row("C-1", "passed")]
        verdict = face_verdict(rows, self.table, (FACE_CHANNEL,))
        self.assertTrue(verdict["admission_passed"])
        self.assertIn("不构成模型能力结论", verdict["summary_line"])

    def test_annotate_summary_records_faces(self):
        rows = [_Row("M-1", "passed"), _Row("C-1", "failed")]
        summary = annotate_summary({}, rows, self.table, DEFAULT_FACES)
        self.assertEqual(summary["executed_faces"], list(DEFAULT_FACES))
        self.assertEqual(summary["case_faces"], {"M-1": FACE_MODEL, "C-1": FACE_CHANNEL})
        self.assertTrue(summary["face_verdict"]["admission_passed"])

    def test_annotate_summary_can_skip_verdict(self):
        summary = annotate_summary({}, [_Row("M-1", "passed")], self.table, DEFAULT_FACES,
                                   with_verdict=False)
        self.assertNotIn("face_verdict", summary)
        self.assertIn("case_faces", summary)


class VerdictChainTests(unittest.TestCase):
    """收尾链必须认 face_verdict，否则通用「有 failed 即不准入」会推翻面结论。"""

    def test_normalize_summary_uses_face_verdict(self):
        summary = {
            "results": [{"case_id": "M-1", "status": "passed"}, {"case_id": "C-1", "status": "failed"}],
            "face_verdict": {"contract": "faces", "admission_passed": True, "blocking": [],
                             "channel_gap": [{"case_id": "C-1"}], "evaluated": 1},
        }
        out = normalize_summary(summary)
        self.assertTrue(out["admission_passed"])
        self.assertIn("faces", out["verdict_contract"])

    def test_normalize_summary_still_blocks_without_face_verdict(self):
        summary = {"results": [{"case_id": "M-1", "status": "failed"}]}
        out = normalize_summary(summary)
        self.assertFalse(out["admission_passed"])

    def test_finalize_keeps_admission_when_only_channel_failed(self):
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            (out / "requests.jsonl").write_text('{"status": 200}\n', encoding="utf-8")
            summary = {
                "admission_passed": True,
                "results": [
                    {"case_id": "M-1", "name": "核心", "status": "passed"},
                    {"case_id": "C-1", "name": "网关端点", "status": "failed"},
                ],
                "face_verdict": {
                    "contract": "faces", "admission_passed": True,
                    "blocking": [], "channel_gap": [{"case_id": "C-1"}],
                    "evaluated": 1,
                    "summary_line": "模型能力与验真面无失败；通道面记录 1 项缺口（网关端点/严格语义未实现），不阻断准入",
                },
            }
            (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
            finalized = finalize_result_dir(out)
        checks = {c["name"]: c for c in finalized["validation"]["checks"]}
        self.assertTrue(checks["case_verdicts_complete"]["passed"])
        self.assertTrue(finalized["admission_passed"])


class WebFaceSurfaceTests(unittest.TestCase):
    """/api/cases 必须把面信息带给前端，且纯通道卡默认不可选。"""

    def test_case_face_breakdown_covers_suites(self):
        from web import server
        deepseek = server.case_face_breakdown("deepseek", "DS_V4_P1")
        self.assertEqual(deepseek["channel_count"], 16)
        self.assertEqual(deepseek["default"], list(DEFAULT_FACES))
        kvv = server.case_face_breakdown("kvv", "KVV_TOOL_SCHEMA")
        self.assertEqual(kvv["channel_count"], 1)
        self.assertNotIn("model", kvv["counts"])
        self.assertIsNone(server.case_face_breakdown("seedance", "SEEDANCE_TEXT"))


if __name__ == "__main__":
    unittest.main()
