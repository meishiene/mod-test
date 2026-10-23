import unittest
from volcperf.cases import CASES
from volcperf.metrics import percentile, summarize


def row(**changes):
    value = {"counted": True, "success": True, "prompt_tokens": 7500, "completion_tokens": 500,
             "visible_completion_tokens": 500, "finish_reason": "stop", "ttfc_seconds": 1,
             "first_sse_seconds": .1, "first_reasoning_seconds": .2, "ttft_seconds": .2, "e2e_seconds": 5,
             "tpot_ms_per_visible_token": 8, "tpot_token_basis": "visible", "send_delay_ms": None}
    value.update(changes); return value


class MetricTests(unittest.TestCase):
    def test_percentile(self):
        self.assertEqual(percentile([1,2,3,4], .5), 2.5)
        self.assertAlmostEqual(percentile([1,2,3,4], .9), 3.7)

    def test_t0_exact_success(self):
        self.assertTrue(summarize(CASES["T0"], [row() for _ in range(10)])["admission_passed"])

    def test_ttft_prefers_first_effective_token(self):
        rows=[row(ttft_seconds=None,first_reasoning_seconds=.4,ttfc_seconds=2.0),
              row(ttft_seconds=None,first_reasoning_seconds=None,ttfc_seconds=1.0)]
        result=summarize(CASES["B1"],rows)
        self.assertAlmostEqual(result["ttft_p50_seconds"],.7)
        self.assertAlmostEqual(result["ttft_p90_seconds"],.94)

    def test_b1_has_four_requests_per_size(self):
        rows=[]
        for dataset,tokens in (("I1",7500),("I2",12000),("I3",24000),("I4",48000)):
            rows += [row(dataset_id=f"{dataset}-{i:05d}", prompt_tokens=tokens) for i in range(4)]
        result=summarize(CASES["B1"], rows)
        self.assertTrue(result["admission_passed"])
        self.assertEqual({k:v["sent"] for k,v in result["baselines_by_dataset"].items()}, {"I1":4,"I2":4,"I3":4,"I4":4})

    def test_l1_all_requirements(self):
        self.assertTrue(summarize(CASES["L1"], [row() for _ in range(CASES["L1"].samples)], baseline_p90=1)["admission_passed"])

    def test_failure_not_hidden(self):
        # 4/200 = 0.98 < 0.99：少量失败也必须把准入拉下来，不能被样本数掩盖。
        rows = [row() for _ in range(CASES["L1"].samples - 4)] + [row(success=False, error_type="timeout") for _ in range(4)]
        self.assertFalse(summarize(CASES["L1"], rows, baseline_p90=1)["admission_passed"])

    def test_short_outputs_excluded_from_tpot(self):
        result = summarize(CASES["L1"], [row(visible_completion_tokens=100) for _ in range(CASES["L1"].samples)], baseline_p90=1)
        self.assertEqual(result["data_quality"]["tpot_valid_samples"], 0)
        self.assertFalse(result["checks"]["tpot_p90"]["passed"])

    def test_sample_count_requirement_follows_the_case_definition(self):
        """样本数要求必须读 case.samples（L1–L5 与 O1 同一口径），不能任何一处写死。"""
        fewer = summarize(CASES["L1"], [row() for _ in range(CASES["L1"].samples - 1)], baseline_p90=1)
        self.assertFalse(fewer["checks"]["sample_count"]["passed"])
        self.assertEqual(fewer["checks"]["sample_count"]["minimum"], CASES["L1"].samples)
        o1 = summarize(CASES["O1"], [row(visible_completion_tokens=500) for _ in range(CASES["O1"].samples - 1)])
        self.assertFalse(o1["checks"]["sample_count"]["passed"])
        self.assertEqual(o1["checks"]["sample_count"]["minimum"], CASES["O1"].samples)

    def test_c1_three_windows(self):
        start, rows = 1_000_000.0, []
        for i in range(1890):
            rows.append(row(prompt_tokens=9500, completion_tokens=500, total_tokens=10000,
                            planned_epoch=start+i/10.5, phase="steady", send_delay_ms=1))
        result = summarize(CASES["C1"], rows)
        self.assertEqual(len(result["capacity_windows"]), 3)
        self.assertTrue(result["admission_passed"])

if __name__ == "__main__": unittest.main()
