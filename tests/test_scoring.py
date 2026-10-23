"""120 分评分：模型 100、通道 20；保留证据与原始用例状态。"""
from __future__ import annotations

import unittest

from common.score_registry import (
    FACE_WEIGHTS,
    SCORING_VERSION,
    VETO_FACE,
    face_for,
    inventory,
    not_applicable,
)
from common.faces import DEFAULT_FACES, skipped_case_ids, suite_case_faces
from common.scoring import headline, render_markdown, score_batch, score_rows, score_summary


def _row(case_id: str, status: str, **extra):
    row = {"case_id": case_id, "name": case_id, "status": status,
           "case_status": status, "case_passed": status == "passed"}
    row.update(extra)
    return row


def _all(case_ids, status="passed"):
    return [_row(case_id, status) for case_id in case_ids]


class FullPassTests(unittest.TestCase):
    def test_all_passed_is_exactly_120(self):
        inv = inventory("qwen", "qwen3.6-flash")
        report = score_rows(_all(inv), test_kind="qwen", model="qwen3.6-flash")
        self.assertEqual(report["score"], 120.0)
        self.assertEqual(report["grade"], "官方")
        self.assertEqual(report["gap_to_official"], 0.0)
        self.assertEqual(report["coverage"], 1.0)
        self.assertFalse(report["partial"])

    def test_score_is_120_only_when_every_face_is_clean(self):
        inv = inventory("qwen", "qwen3.6-flash")
        rows = _all(inv)
        rows[0] = _row(rows[0]["case_id"], "warning")
        self.assertLess(score_rows(rows, test_kind="qwen")["score"], 120.0)


class WeightShareTests(unittest.TestCase):
    def score(self, model_status="passed", channel_status="passed"):
        from common.score_registry import CaseMeta
        inv = {"M0": CaseMeta("M0", "模型核心", "P0", "core"),
               "M1": CaseMeta("M1", "模型补充", "P1", "core"),
               "C0": CaseMeta("C0", "通道", "P0", "core")}
        return score_rows([_row("M0", model_status), _row("M1", "passed"), _row("C0", channel_status)],
                          test_kind="qwen", inventory_override=inv, case_faces={"C0":"channel"})

    def test_model_priority_three_to_two_splits_100_into_60_and_40(self):
        report = self.score(model_status="failed")
        self.assertEqual(report["model_score"], 40)
        self.assertEqual(report["channel_score"], 20)
        self.assertEqual(report["score"], 60)
        self.assertEqual(report["grade"], "可用")
        self.assertTrue(report["rating_passed"])

    def test_warning_earns_half_of_its_weight(self):
        report = self.score(model_status="warning")
        self.assertEqual(report["model_score"], 70)
        self.assertEqual(report["score"], 90)

    def test_channel_failure_never_reduces_model_score(self):
        report = self.score(channel_status="failed")
        self.assertEqual(report["model_score"], 100)
        self.assertEqual(report["channel_score"], 0)
        self.assertEqual(report["score"], 100)
        self.assertEqual(report["grade"], "十分优秀")
        self.assertEqual(report["deductions"][0]["side"], "channel")

    def test_model_failure_never_reduces_channel_score(self):
        self.assertEqual(self.score(model_status="failed")["channel_score"], 20)


class EvidenceGapTests(unittest.TestCase):
    def test_blocked_is_a_gap_and_does_not_earn_unverified_points(self):
        inv = inventory("qwen", "qwen3.6-flash")
        clean = score_rows(_all(inv), test_kind="qwen")
        rows = [_row(cid, "blocked" if cid == "QWEN-05" else "passed") for cid in inv]
        report = score_rows(rows, test_kind="qwen")
        self.assertLess(report["score"], clean["score"])
        self.assertEqual(report["range"][1], 120)
        self.assertFalse(report["deductions"])
        self.assertLess(report["coverage"], clean["coverage"])

    def test_unobserved_cases_are_counted_as_coverage_gap(self):
        inv = inventory("qwen", "qwen3.6-flash")
        core_only = [cid for cid, meta in inv.items() if meta.face == "core"]
        report = score_rows(_all(core_only), test_kind="qwen")
        self.assertTrue(report["partial"])
        self.assertIsNone(report["grade"])
        self.assertLess(report["coverage"], 0.5)
        self.assertIn("思考与推理控制", "".join(report["notes"]))

    def test_all_blocked_has_no_score(self):
        inv = inventory("qwen", "qwen3.6-flash")
        report = score_rows(_all(inv, "blocked"), test_kind="qwen")
        self.assertIsNone(report["score"])
        self.assertIsNone(report["grade"])

    def test_empty_rows_return_none(self):
        self.assertIsNone(score_rows([], test_kind="qwen"))

    def test_unknown_test_kind_returns_none(self):
        self.assertIsNone(score_rows(_all(["X-1"]), test_kind="不存在的套件"))


class FaceScopeTests(unittest.TestCase):
    """通道面默认不跑时，评分分母必须同步收窄。

    这套口径的理由：中转接场景下通道契约面（第二协议/端点/参数校验/错误信封）必然
    大面积不通过，所以默认不执行。此时用例「没跑」是**执行范围**问题，不是证据不足；
    若还留在分母里，所有默认批次都会被判「覆盖不足 ⇒ 不出等级」，模型能力结论就被
    通道噪声废掉了。反过来，勾选了通道面（或没记录执行面）时口径一律回到全量。
    """

    @staticmethod
    def _default_run(kind, model=None):
        inv = inventory(kind, model)
        table = suite_case_faces(kind)
        skipped = skipped_case_ids(inv, kind, DEFAULT_FACES)
        rows = [_row(cid, "passed") for cid in inv if cid not in skipped]
        report = score_rows(rows, test_kind=kind, model=model,
                            executed_faces=DEFAULT_FACES, case_faces=table)
        return report, rows, skipped

    def test_default_face_run_still_earns_a_grade(self):
        report, _rows, skipped = self._default_run("deepseek")
        self.assertTrue(skipped, "deepseek 通道面用例集不能为空")
        self.assertFalse(report["partial"])
        self.assertEqual(report["grade"], "十分优秀")
        self.assertEqual(report["model_score"], 100.0)
        self.assertIsNone(report["channel_score"])
        self.assertEqual(report["components"]["channel"]["status"], "not_selected")
        # 矩阵声明不适用（无视觉型号的 vision 用例）的通道面用例本来就不在分母里。
        expected = {cid for cid in skipped if cid not in not_applicable("deepseek", None)}
        self.assertEqual(report["skipped_cases"], len(expected))
        self.assertLess(report["executed_weight"], report["scope_weight"])
        self.assertTrue(any("本次未选" in note for note in report["notes"]))

    def test_same_rows_without_face_info_stay_partial(self):
        # 没有执行面信息（老结果、或调用方没传）⇒ 回到全量分母，不偷偷放水。
        report, rows, _skipped = self._default_run("deepseek")
        legacy = score_rows(rows, test_kind="deepseek")
        self.assertTrue(legacy["partial"])
        self.assertEqual(legacy["coverage"], report["coverage"])
        self.assertIsNone(legacy["grade"])
        self.assertEqual(legacy["components"]["channel"]["status"], "unmeasured")

    def test_fully_unselected_face_is_reported_as_not_selected(self):
        # deepseek 的 envelope 面整面属通道契约：默认不跑时应是「本次未选」而非「无证据」。
        report, _rows, _skipped = self._default_run("deepseek")
        envelope = next(face for face in report["faces"] if face["face"] == "envelope")
        self.assertEqual(envelope["status"], "not_selected")
        self.assertTrue(any("未选" in note for note in report["notes"]))

    def test_executing_channel_face_restores_the_full_denominator(self):
        inv = inventory("deepseek")
        report = score_rows([_row(cid, "passed") for cid in inv], test_kind="deepseek",
                            executed_faces=("model", "verify", "channel"))
        self.assertEqual(report["skipped_cases"], 0)
        self.assertEqual(report["executed_weight"], report["scope_weight"])

    def test_summary_carries_the_scope_into_scoring(self):
        report, rows, skipped = self._default_run("qwen", "qwen3.6-flash")
        summary = {"test_kind": "qwen", "model": "qwen3.6-flash", "results": rows,
                   "executed_faces": list(DEFAULT_FACES), "case_faces": suite_case_faces("qwen")}
        scored = score_summary(summary)
        self.assertEqual(scored["skipped_cases"], len(skipped))
        self.assertFalse(scored["partial"])

    def test_channel_only_rows_produce_no_score(self):
        # 只跑了通道面用例 ⇒ 可评测分面一条证据都没有，不出分（不能拿 0 分冒充「全错」）。
        inv = inventory("deepseek")
        only_channel = [cid for cid in inv if suite_case_faces("deepseek").get(cid) == "channel"]
        report = score_rows([_row(cid, "passed") for cid in only_channel], test_kind="deepseek",
                            executed_faces=("model", "verify"), case_faces=suite_case_faces("deepseek"))
        self.assertIsNone(report["score"])
        self.assertIsNone(report["grade"])

    def test_nothing_measurable_left_returns_none(self):
        # 唯一用例属未执行的面 ⇒ 执行范围权重为 0，整批不出分。
        from common.score_registry import CaseMeta

        report = score_rows([_row("X-1", "passed")], test_kind="qwen",
                            inventory_override={"X-1": CaseMeta("X-1", "x", "P0", "core")},
                            executed_faces=("model", "verify"), case_faces={"X-1": "channel"})
        self.assertIsNone(report)


class DeclarationTests(unittest.TestCase):
    def test_vision_not_applicable_for_text_only_model(self):
        # qwen3.7-max 矩阵声明无图像输入 ⇒ 视觉移出权重池，既不扣分也不算缺口。
        excluded = not_applicable("qwen", "qwen3.7-max")
        self.assertIn("QWEN-V01", excluded)
        inv = inventory("qwen", "qwen3.7-max")
        rows = [_row(cid, "failed" if cid == "QWEN-V01" else "passed") for cid in inv]
        report = score_rows(rows, test_kind="qwen", model="qwen3.7-max")
        self.assertEqual(report["score"], 120.0)
        self.assertEqual(report["coverage"], 1.0)
        self.assertEqual(report["counts"].get("not_applicable"), 1)

    def test_vision_counts_for_vision_model(self):
        self.assertEqual(not_applicable("qwen", "qwen3.6-plus"), set())
        inv = inventory("qwen", "qwen3.6-plus")
        rows = [_row(cid, "failed" if cid == "QWEN-V01" else "passed") for cid in inv]
        report = score_rows(rows, test_kind="qwen", model="qwen3.6-plus")
        self.assertLess(report["model_score"], 100.0)


class VetoAndCapTests(unittest.TestCase):
    def test_trust_failure_invalidates_the_score(self):
        inv = inventory("qwen", "qwen3.6-flash")
        rows = [_row(cid, "failed" if cid == "QWEN-F01" else "passed") for cid in inv]
        report = score_rows(rows, test_kind="qwen")
        self.assertFalse(report["valid"])
        self.assertIsNone(report["score"])
        self.assertIn("QWEN-F01", report["invalid_reason"])

    def test_core_failure_deducts_weight_without_legacy_70_point_cap(self):
        inv = inventory("qwen", "qwen3.6-flash")
        rows = [_row(cid, "failed" if cid == "QWEN-01" else "passed") for cid in inv]
        report = score_rows(rows, test_kind="qwen")
        self.assertEqual(report["model_score"], 96.1111)
        self.assertEqual(report["score"], 116.1111)
        self.assertFalse(report["caps"])


class RegistryTests(unittest.TestCase):
    def test_face_weights_sum_to_one_hundred(self):
        self.assertEqual(sum(FACE_WEIGHTS.values()), 100.0)
        self.assertEqual(SCORING_VERSION, "conformance-2.1")

    def test_every_suite_case_is_registered(self):
        """注册表必须覆盖套件定义里的**全部**用例，否则漏测就能刷分。"""
        for kind, model in (("deepseek", None), ("qwen", None), ("glm", None),
                            ("fable", None), ("gemini", None), ("kvv", None),
                            ("gpt", "gpt-6.1-sol"), ("gpt", "gpt-6-sol"), ("gpt", "gpt-5.6")):
            inv = inventory(kind, model)
            self.assertTrue(inv, f"{kind} 清单为空")
            unregistered = [cid for cid in inv if face_for(kind, cid) == "unknown"]
            self.assertEqual(unregistered, [], f"{kind} 有未登记用例：{unregistered}")

    def test_gpt_inventory_is_partitioned_by_variant(self):
        counted = {variant: len(inventory("gpt", model)) for variant, model in
                   (("gpt5.6", "gpt-5.6"), ("gpt6.1", "gpt-6.1-sol"), ("gpt6", "gpt-6-sol"))}
        self.assertEqual(counted, {"gpt5.6": 58, "gpt6.1": 49, "gpt6": 49})
        ids = {model: set(inventory("gpt", model)) for model in ("gpt-6.1-sol", "gpt-6-sol")}
        self.assertFalse(ids["gpt-6.1-sol"] & ids["gpt-6-sol"])

    def test_performance_cases_do_not_take_weight(self):
        inv = inventory("kvv")
        ttft = [cid for cid in inv if cid.startswith("K3-TTFT")]
        self.assertTrue(ttft)
        report = score_rows(_all(inv), test_kind="kvv")
        scored = {face["face"] for face in report["faces"] if face.get("weight")}
        self.assertNotIn("performance", scored)


class BatchTests(unittest.TestCase):
    def test_batch_merges_cards_and_groups_by_model(self):
        items = [
            {"model": "m-a", "summary": {"test_kind": "qwen", "model": "m-a", "results": _all(
                [cid for cid, meta in inventory("qwen", "m-a").items() if meta.face == "core"])}},
            {"model": "m-a", "summary": {"test_kind": "qwen", "model": "m-a", "results": _all(
                [cid for cid, meta in inventory("qwen", "m-a").items() if meta.face == "tools"])}},
            {"model": "m-b", "summary": {"test_kind": "qwen", "model": "m-b", "results": _all(
                [cid for cid, meta in inventory("qwen", "m-b").items() if meta.face == "core"])}},
        ]
        reports = score_batch(items)
        self.assertEqual([report["model"] for report in reports], ["m-a", "m-b"])
        merged = reports[0]
        self.assertGreater(merged["coverage"], reports[1]["coverage"])

    def test_summary_without_results_is_not_scored(self):
        self.assertIsNone(score_summary({"test_kind": "qwen", "model": "x", "results": []}))
        self.assertIsNone(score_summary({}))


class CrossSuiteProbeTests(unittest.TestCase):
    """跨套件通用契约探针（CONTRACT-0x）必须按面登记，否则会被当成未知用例丢掉。"""

    def test_contract_probes_have_faces(self):
        expected = {"CONTRACT-01": "trust", "CONTRACT-02": "envelope", "CONTRACT-03": "core",
                    "CONTRACT-04": "tools", "CONTRACT-05": "thinking"}
        for case_id, face in expected.items():
            self.assertEqual(face_for("deepseek", case_id), face)

    def test_extra_probe_cases_enter_the_denominator(self):
        inv = inventory("qwen", "qwen3.6-flash")
        rows = [_row(cid, "passed") for cid in inv] + [_row("CONTRACT-03", "passed")]
        report = score_rows(rows, test_kind="qwen")
        self.assertEqual(report["counts"]["extra_cases"], 1)
        self.assertTrue(any("跨套件探针" in note for note in report["notes"]))

    def test_relay_optional_case_is_attributed_to_the_channel(self):
        # Strict Tool / FIM / Prefix 属 DeepSeek 中转可选能力面：失败扣分照算，但归因是通道侧。
        inv = inventory("deepseek")
        rows = [_row(cid, "failed" if cid == "DS-V4-19" else "passed") for cid in inv]
        report = score_rows(rows, test_kind="deepseek", model="deepseek-v4.1-flash-ark")
        deduction = next(item for item in report["deductions"] if item["case_id"] == "DS-V4-19")
        self.assertEqual(deduction["attribution"], "channel")

    def test_core_case_failure_is_attributed_to_the_model(self):
        inv = inventory("deepseek")
        rows = [_row(cid, "failed" if cid == "DS-V4-26" else "passed") for cid in inv]
        report = score_rows(rows, test_kind="deepseek")
        deduction = next(item for item in report["deductions"] if item["case_id"] == "DS-V4-26")
        self.assertEqual(deduction["attribution"], "model")


class PresentationTests(unittest.TestCase):
    """呈现口径：证据不足时必须显式说明「只测了一部分面」，不能把局部分当 100 摆出来。"""

    def _partial(self):
        inv = inventory("qwen", "qwen3.6-flash")
        core_only = [cid for cid, meta in inv.items() if meta.face == "core"]
        return score_rows(_all(core_only), test_kind="qwen", model="qwen3.6-flash")

    def test_partial_headline_says_it_is_partial_and_still_shows_the_number(self):
        text = headline(self._partial())
        self.assertIn("仅已测面", text)
        self.assertIn("不出等级", text)
        self.assertIn("证据覆盖", text)
        self.assertIn("100", text)  # 已测面满分也要如实给出分数，但不能给等级

    def test_partial_headline_prints_the_full_range(self):
        report = self._partial()
        text = headline(report)
        low, high = report["range"]
        self.assertIn(f"{low:g}–{high:g}", text)

    def test_full_headline_states_the_gap_to_maximum(self):
        inv = inventory("qwen", "qwen3.6-flash")
        rows = [_row(cid, "warning" if cid == "QWEN-05" else "passed") for cid in inv]
        report = score_rows(rows, test_kind="qwen", model="qwen3.6-flash")
        text = headline(report)
        self.assertIn(report["grade"], text)
        self.assertIn(f"距满分 {report['gap_to_official']}", text)

    def test_markdown_section_records_coverage_and_version(self):
        text = render_markdown(self._partial())
        self.assertIn("## 评分", text)
        self.assertIn("证据覆盖", text)
        self.assertIn(SCORING_VERSION, text)


class SummaryIntegrationTests(unittest.TestCase):
    def test_score_summary_reads_relay_attribution(self):
        inv = inventory("qwen", "qwen3.6-flash")
        rows = [_row(cid, "failed" if cid == "QWEN-T01" else "passed") for cid in inv]
        summary = {"test_kind": "qwen", "model": "qwen3.6-flash", "results": rows,
                   "relay_verdict": {"contract": "relay", "admission_passed": True,
                                     "blocking": [], "relay_gap": [{"case_id": "QWEN-T01"}],
                                     "evaluated": 25}}
        report = score_summary(summary)
        self.assertEqual(report["deductions"][0]["attribution"], "channel")


if __name__ == "__main__":
    unittest.main()


def test_kimi_schema_results_are_excluded_from_scores_and_denominator():
    inv = inventory('kvv', 'kimi-k3')
    assert 'KVV_TOOL_SCHEMA' not in inv
    cid = next(cid for cid, meta in inv.items() if meta.face in FACE_WEIGHTS)
    baseline = {'test_kind': 'kvv', 'model': 'kimi-k3', 'case': 'K3_TEMP_SUITE', 'results': [_row(cid, 'passed')]}
    # Even an imported historical report carrying otherwise scorable rows must
    # not let the explicitly excluded suite alter the score or its denominator.
    schema = {**baseline, 'case': 'KVV_TOOL_SCHEMA', 'results': [_row(cid, 'failed')]}
    assert score_summary(schema) is None
    assert score_batch([{'summary': schema}]) == []
    assert score_batch([{'summary': baseline}, {'summary': schema}]) == score_batch([{'summary': baseline}])
