"""dsv4.relay_profile 中转接入口径的判定测试。"""
from __future__ import annotations

import unittest

from dsv4.relay_profile import (
    classify_case,
    evaluate_relay_admission,
    merge_case_records,
)


def case(case_id: str, status: str, details: str = "") -> dict:
    return {"case_id": case_id, "name": case_id, "status": status, "details": details}


class ClassifyCaseTests(unittest.TestCase):
    def test_optional_face_failure_is_relay_gap(self) -> None:
        for case_id in ("DS-V4-19", "DS-V4-23", "DS-V4-35", "DS-V4-41", "DS-V4-43", "DS-V4-55", "DS-V4-60"):
            self.assertEqual(classify_case(case_id, "failed"), "relay_gap", case_id)

    def test_core_failure_is_blocking(self) -> None:
        for case_id in ("DS-V4-01", "DS-V4-06", "DS-V4-11", "DS-V4-16", "DS-V4-33", "DS-V4-46"):
            self.assertEqual(classify_case(case_id, "failed"), "blocking", case_id)

    def test_vision_failure_depends_on_model_declaration(self) -> None:
        # 视觉面只在官方声明的模型版本上具备：V4.1-Flash 系失败才算模型缺陷
        self.assertEqual(
            classify_case("DS-V4-36", "failed", model="deepseek-v4.1-flash-ark"), "blocking"
        )
        # V4-Flash-0731 是视觉开放前的纯文本版本，答不出图属「该模型不提供该能力面」
        self.assertEqual(
            classify_case("DS-V4-36", "failed", model="deepseek-v4-flash-0731"), "vision_undeclared"
        )
        self.assertEqual(
            classify_case("DS-V4-53", "failed", model="deepseek-v4-flash-ga-260731"), "vision_undeclared"
        )
        self.assertEqual(
            classify_case("DS-V4-36", "failed", model="deepseek-v4-pro-0813"), "vision_undeclared"
        )

    def test_non_failure_statuses_are_informational(self) -> None:
        self.assertEqual(classify_case("DS-V4-01", "warning"), "informational")
        self.assertEqual(classify_case("DS-V4-01", "blocked"), "informational")
        self.assertEqual(classify_case("DS-V4-01", "passed"), "informational")


class EvaluateTests(unittest.TestCase):
    def test_relay_gaps_do_not_block_admission(self) -> None:
        verdict = evaluate_relay_admission(
            [
                case("DS-V4-01", "passed"),
                case("DS-V4-19", "failed", "Strict 未强制 schema"),
                case("DS-V4-41", "failed", "file_id 空"),
                case("DS-V4-39", "warning", "max_tokens 越界未拒"),
                case("DS-V4-37", "blocked", "退休别名 503"),
            ],
            model="deepseek-v4.1-flash-ark",
        )
        self.assertTrue(verdict.admission_passed)
        self.assertEqual(len(verdict.relay_gaps), 2)
        self.assertEqual({c["face"] for c in verdict.relay_gaps}, {"strict_tool", "files"})
        self.assertEqual(len(verdict.recorded), 2)
        self.assertEqual(verdict.stats["failed"], 2)

    def test_core_failure_blocks_admission(self) -> None:
        verdict = evaluate_relay_admission(
            [case("DS-V4-16", "failed", "跨 user_id 命中")],
            model="deepseek-v4.1-flash-ark",
        )
        self.assertFalse(verdict.admission_passed)
        self.assertEqual([c["case_id"] for c in verdict.blocking], ["DS-V4-16"])

    def test_vision_undeclared_not_blocking(self) -> None:
        verdict = evaluate_relay_admission(
            [case("DS-V4-36", "failed"), case("DS-V4-53", "failed")],
            model="deepseek-v4-pro-0813",
        )
        self.assertTrue(verdict.admission_passed)
        self.assertEqual(len(verdict.relay_gaps), 0)
        self.assertEqual(len(verdict.recorded), 2)

    def test_case_status_fallback_field(self) -> None:
        verdict = evaluate_relay_admission(
            [{"case_id": "DS-V4-21", "case_status": "failed"}],
            model="deepseek-v4.1-flash-ark",
        )
        self.assertTrue(verdict.admission_passed)  # chat_prefix 属中转未承诺面
        self.assertEqual(len(verdict.relay_gaps), 1)


class MergeTests(unittest.TestCase):
    def test_latest_evidence_wins(self) -> None:
        merged = merge_case_records(
            [
                {"generated_at": "2026-10-21T12:58:51", "results": [case("DS-V4-41", "failed", "旧证据")]},
                {"generated_at": "2026-10-21T14:00:00", "results": [case("DS-V4-41", "passed", "复测通过")]},
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["status"], "passed")
        self.assertEqual(merged[0]["details"], "复测通过")

    def test_missing_cases_kept_from_baseline(self) -> None:
        merged = merge_case_records(
            [
                {"generated_at": "t1", "results": [case("DS-V4-01", "passed"), case("DS-V4-16", "failed")]},
                {"generated_at": "t2", "results": [case("DS-V4-16", "passed")]},
            ]
        )
        by_id = {c["case_id"]: c for c in merged}
        self.assertEqual(by_id["DS-V4-01"]["status"], "passed")
        self.assertEqual(by_id["DS-V4-16"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
