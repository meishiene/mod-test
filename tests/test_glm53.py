"""GLM 5.3 套件回归：既有契约（用例分组 / usage / 指纹结论）与多模型聚合报告。

三条取证探针卡片（GLM_53_IDENTITY / VISION / ENVELOPE）已于 2026-10-21 下线：
跨模型取证统一走「中转验真」模块（权威实现 forensics 包），GLM 套件不再自带适配层。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import glm53.core as glm_core
from glm53.cases import CASES
from glm53.core import GLM53Runner, CaseResult, summarize_results, write_report


class GLM53BaselineTests(unittest.TestCase):
    def test_case_groups_removed(self):
        self.assertNotIn("GLM_53_ALL", CASES)
        self.assertIn("GLM_53_CORE", CASES)
        self.assertIn("GLM_53_AGENT", CASES)
        self.assertIn("GLM_53_CACHE", CASES)

    def test_forensics_probe_cards_removed(self):
        """三条取证探针卡已下线；跨模型取证统一走 forensics（中转验真）模块。"""
        for name in ("GLM_53_IDENTITY", "GLM_53_VISION", "GLM_53_ENVELOPE"):
            self.assertNotIn(name, CASES)
        specs = {spec.case_id for spec in glm_core.CASE_SPECS}
        self.assertNotIn("GLM-53-I01", specs)
        self.assertNotIn("GLM-53-V01", specs)
        self.assertNotIn("GLM-53-E01", specs)
        for handler in ("case_identity_probe", "case_vision_probe", "case_envelope_probe"):
            self.assertFalse(hasattr(GLM53Runner, handler))

    def test_usage_contract(self):
        self.assertTrue(glm_core.usage_ok({"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}))
        self.assertFalse(glm_core.usage_ok({"usage": {"prompt_tokens": 1}}))

    def test_fingerprint_assessment_levels(self):
        results = [CaseResult(f"GLM-53-F0{i}", "probe", "glm-5.3", True) for i in range(1, 8)]
        result = summarize_results(results)
        self.assertEqual(result["fingerprint_assessment"]["level"], "已测接口指纹相容（不证明同款或满血）")


class CapabilityBatchReportTests(unittest.TestCase):
    """GLM 多模型批次（glm-5.3 / glm-5.3-flash）的聚合报告要能分清模型。"""

    def _child(self, root: Path, index: int, model: str) -> dict:
        summary_path = root / f"child{index}.json"
        summary_path.write_text(json.dumps({"results": [
            {"case_id": "GLM-53-01", "name": "基础对话与模型标识", "status": "passed", "details": "HTTP 200"}]},
            ensure_ascii=False), encoding="utf-8")
        return {"id": f"child{index}", "case_name": "GLM_53_CORE", "model": model, "status": "completed",
                "passed": True, "summary_path": str(summary_path), "result_dir": None,
                "report_path": None, "log_path": None, "error": None}

    def test_multi_model_capability_report_adds_model_column(self):
        from web.aggregation import aggregate_batch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            children = [self._child(root, 1, "glm-5.3"), self._child(root, 2, "glm-5.3-flash")]
            batch = {"id": "b1", "test_kind": "glm", "protocol": "openai",
                     "base_url": "https://example.com", "model": "2 个模型"}
            aggregate = aggregate_batch(out, batch, children)
            report = (out / "report.md").read_text(encoding="utf-8")
            self.assertIn("| 用例 | 子记录 | 模型 | 状态 | 结论 | 测试结果/失败原因 |", report)
            self.assertIn("模型范围：glm-5.3、glm-5.3-flash", report)
            self.assertIn("glm-5.3-flash", report.split("## 子用例结果", 1)[1])
            self.assertEqual(len(aggregate["results"]), 2)

    def test_single_model_capability_report_keeps_old_layout(self):
        from web.aggregation import aggregate_batch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            children = [self._child(root, 1, "glm-5.3")]
            batch = {"id": "b2", "test_kind": "glm", "protocol": "openai",
                     "base_url": "https://example.com", "model": "glm-5.3"}
            aggregate_batch(out, batch, children)
            report = (out / "report.md").read_text(encoding="utf-8")
            self.assertIn("| 用例 | 子记录 | 状态 | 结论 | 测试结果/失败原因 |", report)
            self.assertNotIn("模型范围", report)


if __name__ == "__main__":
    unittest.main()
