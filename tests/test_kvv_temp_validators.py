"""K3 临时案例集的判定口径（2026-10-14 按官方契约修订的部分）。

钉住三件事，避免回归：

1. `K3-PARAM-002` 是**负向**用例：非固定值（0.6 / 0.8）必须被拒（K3 原厂
   `temperature` 固定 1.0），而不是像修订前那样断言 200。
2. `K3-THINK-002` 是**观察项**：400 拒绝 / 200 仍思考都算达标，只有
   「200 且没有产生 reasoning」才判失败；非 400/200 记 not_evaluable。
3. 缓存按**聚合口径**判定：单次 `cached_tokens / prompt_tokens ≥80%` 记为一次
   命中，命中 probe 占比 ≥80% 才算达标（逐条严判会把正常批次误判为失败）。
"""
import unittest

from common.suite import ManifestCase
from common.test_framework import ExecutionRecord, validate_case
from kvv.temp_runner import (CACHE_HIT_PROBE_RATE, CACHE_HIT_RATIO, _cache_ratio,
                             _cache_sample_check, _thinking_not_closable, aggregate_cache,
                             validators_for)
from kvv.temp_suite import find_case, load_manifest, manifest_cases, merge_cases

BY_ID = {item["id"]: item for item in manifest_cases(load_manifest())}
MERGED = merge_cases(manifest_cases(load_manifest()))

TEMPERATURE_ERROR = {
    "error": {"type": "invalid_request_error",
              "message": "invalid temperature: only 1 is allowed for this model"}
}


def record(*, status_code=200, usage_value=None, reasoning="", content="", response_body=None):
    return ExecutionRecord(
        case_id="K3-PROBE", request_id="req-1", request={}, request_headers={},
        started_at="2026-10-14T00:00:00+08:00", started_epoch=0.0,
        http_status=status_code,
        response_body=response_body if response_body is not None else {"choices": [{"message": {"content": content}}]},
        content=content, reasoning=reasoning, usage=usage_value,
    )


def cached_usage(prompt, cached):
    value = {"prompt_tokens": prompt, "completion_tokens": 8}
    if cached is not None:
        value["prompt_tokens_details"] = {"cached_tokens": cached}
    return value


class ParamCaseContractTests(unittest.TestCase):
    def test_param_001_uses_official_fixed_values(self):
        case = ManifestCase(BY_ID["K3-PARAM-001"])
        self.assertEqual(case.request["temperature"], 1.0)
        self.assertEqual(case.request["top_p"], 0.95)
        passed = validate_case(record(content="PARAM-OK", usage_value=cached_usage(20, None)),
                               validators_for(case))
        self.assertEqual(passed.status, "passed", passed.reasons)

    def test_param_002_rejects_non_fixed_values(self):
        case = ManifestCase(BY_ID["K3-PARAM-002"])
        self.assertEqual(case.request["temperature"], 0.6)
        validators = validators_for(case)
        rejected = validate_case(record(status_code=400, response_body=TEMPERATURE_ERROR), validators)
        self.assertEqual(rejected.status, "passed", rejected.reasons)
        # 修订前这里断言的是 200 —— 现在 200 必须判失败（非固定值被静默接受）。
        accepted = validate_case(record(), validators)
        self.assertEqual(accepted.status, "failed")

    def test_thinking_001_uses_official_shape(self):
        self.assertEqual(ManifestCase(BY_ID["K3-THINK-001"]).request["thinking"],
                         {"type": "enabled", "keep": "all"})

    def test_content_cases_have_output_budget_for_thinking(self):
        # K3 思考常开且思考 token 计入输出上限：要可见正文就得给足预算。
        for case_id in ("K3-CONN-001", "K3-CONN-002", "K3-THINK-002", "K3-STREAM-002",
                        "K3-JSON-001", "K3-OBS-001"):
            with self.subTest(case=case_id):
                self.assertGreaterEqual(ManifestCase(BY_ID[case_id]).request["max_tokens"], 512)


class ThinkingNotClosableTests(unittest.TestCase):
    def setUp(self):
        self.validators = [_thinking_not_closable()]

    def test_400_is_acceptable(self):
        self.assertEqual(validate_case(record(status_code=400), self.validators).status, "passed")

    def test_200_with_reasoning_is_acceptable(self):
        result = validate_case(record(reasoning="先算 918273×817263 的中段…",
                                      usage_value=cached_usage(100, None)), self.validators)
        self.assertEqual(result.status, "passed", result.reasons)

    def test_200_without_reasoning_fails(self):
        result = validate_case(record(usage_value=cached_usage(100, None)), self.validators)
        self.assertEqual(result.status, "failed")
        self.assertIn("思考被真的关掉了", " ".join(result.reasons))

    def test_unexpected_status_is_not_evaluable(self):
        self.assertEqual(validate_case(record(status_code=503), self.validators).status, "not_evaluable")


class CacheAggregateTests(unittest.TestCase):
    def _plan(self, case_id):
        plan = find_case(MERGED, case_id)
        self.assertIsNotNone(plan)
        return plan

    def _probes(self, ratios):
        return [{"case_id": f"K3-{index}", "status": "passed", "passed": True, "checks": [],
                 "failure_reasons": [], "request_id": "r", "cache_ratio": ratio}
                for index, ratio in enumerate(ratios, 1)]

    def test_probe_ratio_reads_nested_cached_tokens(self):
        self.assertAlmostEqual(_cache_ratio(record(usage_value=cached_usage(1000, 900))), 0.9)
        self.assertIsNone(_cache_ratio(record(usage_value=cached_usage(1000, None))))

    def test_sample_check_never_fails_but_records(self):
        assertion = _cache_sample_check(record(usage_value=cached_usage(1000, None)))
        self.assertTrue(assertion.passed)
        self.assertIsNone(assertion.actual)

    def test_hit_rate_above_threshold_passes(self):
        # 6/7 = 85.7% ≥ 80%：修订前逐条严判会把这条判成 failed。
        row = aggregate_cache(self._plan("CACHE-HIT-7"),
                              self._probes([0.99, 0.99, 0.99, 0.99, 0.99, 0.99, None]))
        self.assertTrue(row["passed"], row["failure_reasons"])
        self.assertEqual((row["cache_hits"], row["cache_probes"]), (6, 7))

    def test_hit_rate_below_threshold_fails(self):
        row = aggregate_cache(self._plan("CACHE-HIT-7"),
                              self._probes([0.99, 0.99, 0.99, None, None, None, None]))
        self.assertFalse(row["passed"])
        self.assertLess(row["cache_hit_rate"], CACHE_HIT_PROBE_RATE)
        self.assertIn("缓存命中率 3/7", " ".join(row["failure_reasons"]))

    def test_transport_failure_is_not_masked_by_hit_rate(self):
        probes = self._probes([0.99, 0.99, 0.99])
        probes[1] = {**probes[1], "status": "failed", "passed": False,
                     "failure_reasons": ["HTTP 状态码不符合预期"]}
        row = aggregate_cache(self._plan("CACHE-TTL-3"), probes)
        self.assertFalse(row["passed"], "运输/取证失败必须保留，不能被命中率盖过")

    def test_thresholds_are_single_sourced(self):
        self.assertEqual((CACHE_HIT_RATIO, CACHE_HIT_PROBE_RATE), (0.80, 0.80))

    def test_manifest_cache_assertions_speak_of_rate(self):
        for item in manifest_cases(load_manifest()):
            if item["id"].startswith("K3-CACHE-HIT"):
                self.assertTrue(any("占比" in text for text in item["assertions"]), item["id"])

    def test_cache_batch_name_matches_probe_count(self):
        # batch 名要包含次数（temp_suite 的合并键语义）：CACHE-HIT-7 ↔ 7 条原子案例。
        plan = self._plan("CACHE-HIT-7")
        self.assertEqual(plan.repetitions, 7)


class ManifestDocSyncTests(unittest.TestCase):
    """md 是清单的渲染产物：手改 md 或改清单后忘记重生成，都会在这里被拦下。"""

    def test_md_matches_manifest(self):
        from scripts.gen_k3_temp_cases_md import MD_PATH, render

        self.assertEqual(
            MD_PATH.read_text(encoding="utf-8"), render(load_manifest()),
            "kvv/kimi_k3_temp_cases.md 与 kvv/kimi_k3_temp_cases.json 不一致："
            "跑 `.venv/bin/python scripts/gen_k3_temp_cases_md.py` 重新生成",
        )

    def test_md_has_no_unrendered_placeholder(self):
        from scripts.gen_k3_temp_cases_md import MD_PATH

        text = MD_PATH.read_text(encoding="utf-8")
        self.assertNotIn("{d[", text, "md 里残留模板占位符（历史上出现过 `{d['version']}`）")


if __name__ == "__main__":
    unittest.main()
