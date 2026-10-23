"""`scripts/probe_qianfan_params.py` 的判定口径与产物形状。

脚本不接入平台登记，但它的判定是「官方契约」这类结论的直接证据来源，
所以口径本身要有测试：期望 200 的探测不许被 400 放行，期望 400 的探测
不许把 200 记成 failed（那是「网关更宽松」，属 warning，需人工定性）。
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

from scripts.probe_qianfan_params import ROOT, build_probes, judge

ERROR_400 = {"error": {"type": "invalid_request_error",
                       "message": "invalid temperature: only 1 is allowed for this model"}}
OK_200 = {"choices": [{"message": {"content": "OK"}}], "usage": {"prompt_tokens": 9, "completion_tokens": 2}}


def probe(probe_id):
    return next(item for item in build_probes() if item["id"] == probe_id)


class ProbeDefinitionTests(unittest.TestCase):
    def test_covers_the_three_missing_evidence_items(self):
        ids = {item["id"] for item in build_probes()}
        for expected in ("PARAM-TEMP1", "PARAM-TOPP09", "THINK-OFFICIAL"):
            self.assertIn(expected, ids)

    def test_temperature_negatives_stay_inside_legal_range(self):
        # 0.6 在 [0,2] 区间内 ⇒ 被拒只能解释为「取值域锁死」，不是越界校验。
        self.assertEqual(probe("PARAM-TEMP06")["params"]["temperature"], 0.6)

    def test_thinking_positive_uses_official_shape(self):
        self.assertEqual(probe("THINK-OFFICIAL")["params"]["thinking"],
                         {"type": "enabled", "keep": "all"})


class JudgeTests(unittest.TestCase):
    def test_fixed_value_accepted(self):
        self.assertEqual(judge(probe("PARAM-TEMP1"), 200, OK_200, "")[0], "passed")

    def test_fixed_value_rejected_is_failed(self):
        self.assertEqual(judge(probe("PARAM-TEMP1"), 400, ERROR_400, "")[0], "failed")

    def test_non_fixed_value_rejected(self):
        self.assertEqual(judge(probe("PARAM-TEMP06"), 400, ERROR_400, "")[0], "passed")

    def test_non_fixed_value_accepted_is_warning_not_failure(self):
        # 网关比原厂宽松：不能判模型失败，要留痕交人工定性。
        self.assertEqual(judge(probe("PARAM-TEMP06"), 200, OK_200, "")[0], "warning")

    def test_top_p_probe_accepts_both_outcomes(self):
        self.assertEqual(judge(probe("PARAM-TOPP09"), 200, OK_200, "")[0], "passed")
        self.assertEqual(judge(probe("PARAM-TOPP09"), 400, ERROR_400, "")[0], "passed")

    def test_thinking_disabled_accepts_rejection_or_thinking(self):
        self.assertEqual(judge(probe("THINK-DISABLED"), 400, ERROR_400, "")[0], "passed")
        with_reasoning = {"choices": [{"message": {"reasoning_content": "想一下…", "content": ""}}]}
        self.assertEqual(judge(probe("THINK-DISABLED"), 200, with_reasoning, "")[0], "passed")

    def test_thinking_disabled_without_reasoning_fails(self):
        self.assertEqual(judge(probe("THINK-DISABLED"), 200, OK_200, "")[0], "failed")

    def test_transport_failure_is_blocked_not_failed(self):
        self.assertEqual(judge(probe("PARAM-TEMP1"), None, None, "ConnectError: refused")[0], "blocked")
        self.assertEqual(judge(probe("PARAM-TEMP1"), 503, None, "")[0], "blocked")


class ScriptSmokeTests(unittest.TestCase):
    def test_requires_key_and_writes_evidence(self):
        env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"}
        missing = subprocess.run([sys.executable, "scripts/probe_qianfan_params.py"],
                                 cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(missing.returncode, 1)
        self.assertIn("QIANFAN_API_KEY", missing.stderr)

        out = Path(ROOT / ".pytest-tmp" / "params-probe-smoke")
        run = subprocess.run(
            [sys.executable, "scripts/probe_qianfan_params.py",
             "--out", str(out), "--base-url", "http://127.0.0.1:9", "--no-proxy",
             "--timeout", "2"],
            cwd=ROOT, env={**env, "QIANFAN_API_KEY": "bce-v3/fake"},
            capture_output=True, text=True,
        )
        # 连不上 = 取证失败（blocked），不是模型失败 ⇒ 退出码 2 且不含 failed。
        self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["blocked"], summary["total"])
        self.assertEqual(summary["conclusion"], "存在未达标/待定性项")
        self.assertTrue((out / "requests.jsonl").read_text(encoding="utf-8").strip())
        self.assertIn("K3 官方契约取值域闭环探测", (out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
