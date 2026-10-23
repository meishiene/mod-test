"""单项测试报告统一为聚合版式的回归测试（图片类套件不参与）。"""
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from common.finalizer import finalize_result_dir
from common.report_metrics import build_item_metrics
from common.report_style import METRICS_COLUMNS, case_table_report, is_image_summary, unify_report


def _fable_report() -> str:
    results = [
        {"case_id": "FABLE5-E01", "name": "adaptive thinking 接受与产出", "status": "passed", "model": "claude-sonnet-4-6", "duration_seconds": 3.72, "details": "接受且回传非空摘要块"},
        {"case_id": "FABLE5-E03", "name": "thinking=disabled 应被拒绝", "status": "warning", "model": "claude-sonnet-4-6", "duration_seconds": 1.29, "details": "被 200 静默接受（留痕）"},
    ]
    rows = [[f"{r['case_id']} {r['name']}", r["model"], r["status"], f"{r['duration_seconds']:.2f}", r["details"]] for r in results]
    return case_table_report("Fable 5 专项测试报告", total=2, passed=1, failed=0, warning=1,
                             extra_stats=["指纹结论：部分能力受限"], rows=rows)


def _summary(**extra):
    summary = {"case": "FABLE5_P0", "test_kind": "fable", "model": "claude-sonnet-4-6", "protocol": "anthropic",
               "record_id": "abc123", "total": 2, "passed": 1, "failed": 0, "warning": 1, "pass_rate": 1.0,
               "admission_passed": False,
               "results": [{"case_id": "FABLE5-E01", "name": "adaptive", "status": "passed", "duration_seconds": 1.0, "details": "ok"},
                           {"case_id": "FABLE5-E03", "name": "disabled", "status": "warning", "duration_seconds": 2.0, "details": "warn"}]}
    summary.update(extra)
    return summary


class UnifiedReportStyleTests(unittest.TestCase):
    def test_header_matches_aggregate_style_and_keeps_domain_bullets(self):
        text = unify_report(_fable_report(), _summary())
        for bullet in ("- 记录 ID：`abc123`", "- 协议：`anthropic`", "- 模型：`claude-sonnet-4-6`", "- 总结论：未通过"):
            self.assertIn(bullet, text)
        self.assertIn("- 通过：1/2", text)
        # 旧统计要点被新的聚合要点替换，领域要点保留
        self.assertNotIn("- 总计：", text)
        self.assertNotIn("- 已评估用例通过率：", text)
        self.assertIn("指纹结论：部分能力受限", text)
        # 用例表保留并归入“用例结果”章节
        self.assertIn("## 用例结果", text)
        self.assertIn("FABLE5-E01 adaptive thinking 接受与产出", text)
        self.assertNotIn("<small>", text)

    def test_metrics_table_uses_aggregate_columns(self):
        text = unify_report(_fable_report(), _summary())
        self.assertIn("| " + " | ".join(METRICS_COLUMNS) + " |", text)
        self.assertIn("| FABLE5_P0 | 已完成 | 未通过 |", text)

    def test_metrics_fall_back_to_request_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "KVV-SUITE-20261010-120000"
            out.mkdir()
            (out / "requests.jsonl").write_text("\n".join(
                json.dumps({"http_status": 200, "timing": {"ttft_ms": 1000, "ttfc_ms": 2000, "e2e_ms": 5000}}) for _ in range(5)
            ) + "\n", encoding="utf-8")
            summary = _summary(sent=None, success_rate=None)
            text = unify_report(_fable_report(), summary, out)
            self.assertIn("| 5 | 100.0% |", text)
            self.assertIn("1.000s", text)
            self.assertIn("2.000s", text)

    def test_idempotent_and_legacy_metrics_section_renamed(self):
        legacy = (
            "# T0 测试报告\n\n- 总计：0\n- 通过：0\n- 本轮结论：未通过/数据不足\n- 负载画像：mixed-business-v1\n\n"
            "## 指标\n\n| 指标 | 数值 |\n|---|---:|\n| TTFT P50/P90/P99（秒） | - / - / - |\n\n## 判定\n\n| 检查项 | 通过 |\n|---|---|\n| connectivity | 否 |\n"
        )
        summary = {"case": "T0", "test_kind": "load", "model": "m", "admission_passed": False, "sent": 0, "success_rate": 0.0}
        first = unify_report(legacy, summary)
        self.assertIn("## 指标明细", first)
        self.assertEqual(first.count("## 指标\n"), 1)
        self.assertNotIn("- 总计：", first)
        self.assertIn("负载画像：mixed-business-v1", first)
        self.assertIn("## 判定", first)
        self.assertEqual(unify_report(first, summary), first)

    def test_empty_report_is_synthesized_from_summary(self):
        text = unify_report("", _summary())
        self.assertIn("# FABLE5_P0 测试报告", text)
        self.assertIn("## 用例结果", text)
        self.assertIn("FABLE5-E01", text)

    def test_image_summaries_are_excluded(self):
        self.assertTrue(is_image_summary({"test_kind": "gpt-image"}))
        self.assertTrue(is_image_summary({"test_kind": "gemini-image"}))
        self.assertTrue(is_image_summary({"images_saved": 3}))
        self.assertFalse(is_image_summary(_summary()))

    def test_finalize_unifies_report_and_keeps_image_report_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "FABLE5_P0-20261010-120000"
            out.mkdir()
            (out / "requests.jsonl").write_text(json.dumps({"http_status": 200, "timing": {"ttft_ms": 900, "ttfc_ms": 1200, "e2e_ms": 3000}}) + "\n", encoding="utf-8")
            (out / "summary.json").write_text(json.dumps(_summary(), ensure_ascii=False), encoding="utf-8")
            (out / "report.md").write_text(_fable_report(), encoding="utf-8")
            finalize_result_dir(out)
            unified = (out / "report.md").read_text(encoding="utf-8")
            self.assertIn("- 记录 ID：`abc123`", unified)
            self.assertIn("| " + " | ".join(METRICS_COLUMNS) + " |", unified)

            image_dir = Path(td) / "GPT_IMAGE-20261010-120000"
            image_dir.mkdir()
            original = "# GPT 图片压测报告\n\n- 图片清单：`images/a.png`\n"
            (image_dir / "report.md").write_text(original, encoding="utf-8")
            (image_dir / "summary.json").write_text(json.dumps({"case": "GPT_IMAGE", "test_kind": "gpt-image", "sent": 2, "successful": 2, "images_saved": 2, "admission_passed": True}), encoding="utf-8")
            finalize_result_dir(image_dir)
            self.assertEqual((image_dir / "report.md").read_text(encoding="utf-8"), original)

    def test_single_report_and_aggregation_share_one_metrics_implementation(self):
        from web import aggregation
        self.assertIs(aggregation.build_item_metrics, build_item_metrics)

    def test_frontend_replaces_static_metrics_section(self):
        js = (ROOT / "web/static/app.js").read_text(encoding="utf-8")
        self.assertIn("function renderSingleReport", js)
        self.assertIn("sec.heading === '指标'", js)
        self.assertIn("hasMetrics", js)

    def test_frontend_groups_batch_case_sections_into_suite_table(self):
        """批次报告的「各轮结果」/「子用例结果」都必须替换成分组套件表，不得静态平铺。

        背景：dsv4 等专项批次的 report.md 章节名是「子用例结果」（fable5 批次是
        「各轮结果」），漏掉任何一个都会退回静态 markdown 大平铺表。
        """
        js = (ROOT / "web/static/app.js").read_text(encoding="utf-8")
        # 聚合报告：两个章节名都要走 batchSuiteTable
        self.assertIn("sec.heading === '各轮结果' || sec.heading === '子用例结果'", js)
        # 单项报告：「指标」与「子用例结果」的静态表同样交给套件表
        self.assertIn("sec.heading === '指标' || sec.heading === '子用例结果'", js)


if __name__ == "__main__":
    unittest.main()
