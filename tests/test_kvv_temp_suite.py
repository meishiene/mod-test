import unittest
from collections import Counter

from kvv.cases import TEMP_CASE_ID
from kvv.temp_suite import find_case, load_manifest, manifest_cases, merge_cases
from web import server


class KvvTempSuiteMergeTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest()
        self.atomic = manifest_cases(self.manifest)
        self.merged = merge_cases(self.atomic)

    def test_manifest_keeps_atomic_rows(self):
        # 清单本身仍是逐条原子案例：合并只发生在案例集与汇总层。
        self.assertEqual(len(self.atomic), 68)

    def test_same_category_and_batch_are_merged(self):
        self.assertEqual(len(self.merged), 24)
        counts = Counter(case.repetitions for case in self.merged)
        self.assertEqual(counts, Counter({1: 18, 10: 4, 7: 1, 3: 1}))

    def test_merged_ids_use_batch_name(self):
        ids = {case.id for case in self.merged}
        for expected in ("16K-10", "64K-10", "100K-10", "TPOT-10", "CACHE-HIT-7", "CACHE-TTL-3"):
            self.assertIn(expected, ids)
        # 没有 batch 的单条场景保持原子 id，且不与他条合并。
        self.assertIn("K3-CONN-001", ids)
        self.assertIn("K3-OBS-001", ids)

    def test_merged_case_expands_back_to_atomic_order(self):
        case = find_case(self.merged, "16K-10")
        self.assertIsNotNone(case)
        self.assertTrue(case.merged)
        self.assertEqual(case.category, "TTFT")
        self.assertEqual(list(case.case_ids), [f"K3-TTFT-{index:02d}" for index in range(1, 11)])
        # 每条原子请求都在清单里真实存在，合并不会吞掉请求。
        self.assertTrue(set(case.case_ids) <= {item["id"] for item in self.atomic})

    def test_merged_name_drops_repetition_suffix(self):
        case = find_case(self.merged, "TPOT-10")
        self.assertEqual(case.name, "流式生成 TPOT")

    def test_select_by_merged_or_atomic_id(self):
        by_merged = find_case(self.merged, "64K-10")
        by_atomic = find_case(self.merged, "K3-TTFT-12")
        self.assertIsNotNone(by_merged)
        self.assertIs(by_merged, by_atomic)
        self.assertIsNone(find_case(self.merged, "K3-NOT-EXIST"))
        self.assertIsNone(find_case(self.merged, None))

    def test_merge_is_pure(self):
        # 合并只读清单，不修改原始条目（执行器仍要拿到原始 request）。
        snapshot = [dict(item) for item in self.atomic]
        merge_cases(self.atomic)
        self.assertEqual(self.atomic, snapshot)

    def test_web_payload_reports_merged_count(self):
        payload = server.kvv_temp_case_payload()
        k3_group = server.BILLING_PROBE_GROUPS["k3"]
        k3_options = [item for item in payload["case_options"] if item.get("group") == k3_group]
        self.assertEqual(len(k3_options), 24)
        self.assertEqual(payload["case_count"], 24)
        self.assertEqual(payload["atomic_case_count"], 68)
        # K3 临时案例集已并入「Kimi 测试」（test_kind=kvv）模块，入口 id 保持不变。
        self.assertEqual(payload["id"], TEMP_CASE_ID)
        self.assertEqual(payload["test_kind"], "kvv")
        self.assertEqual(payload["executor"], "scripts/run_kvv_temp.py")
        options = {option["id"]: option for option in payload["case_options"]}
        self.assertEqual(options["16K-10"]["repetitions"], 10)
        self.assertTrue(options["16K-10"]["merged"])
        self.assertFalse(options["K3-CONN-001"]["merged"])
        # 计费探针已迁至「Claude 测试」模块，K3 案例集回归纯 K3 面。
        self.assertNotIn(server.BILLING_PROBE_GROUPS["billing"],
                         {option.get("group") for option in payload["case_options"]})
        self.assertNotIn("anthropic", payload["protocols"])

    def test_billing_probe_payload_lives_under_claude_module(self):
        payload = server.billing_probe_payload()
        billing_group = server.BILLING_PROBE_GROUPS["billing"]
        self.assertEqual(payload["id"], server.BILLING_PROBE_CASE_ID)
        # 挂在 Claude 测试（fable）模块下，且只在 Anthropic Messages 面取证。
        self.assertEqual(payload["test_kind"], "fable")
        self.assertEqual(payload["protocols"], ["anthropic"])
        # 常驻：不随 Fable 5 / Opus 5 / Sonnet 5 变体切换而消失。
        self.assertTrue(payload["always_visible"])
        self.assertTrue(payload["billing_probe"])
        billing_options = [item for item in payload["case_options"] if item.get("group") == billing_group]
        # 计费组 = 整组入口（BILLING-ALL）+ 11 条探针
        self.assertEqual(len(billing_options), 12)
        self.assertEqual(billing_options[0]["id"], server.BILLING_PROBE_ALL)
        self.assertTrue(billing_options[0]["aggregate"])
        self.assertEqual(billing_options[1]["id"], "BILLING-S01")
        self.assertEqual(payload["case_count"], 12)


class KvvTempBillingProbeTests(unittest.TestCase):
    """中转计费探针的清单接线（现作为 Claude 测试模块的常驻案例组）。"""

    def test_catalog_is_loaded_from_manifest(self):
        cases = server.billing_probe_cases()
        self.assertEqual(len(cases), 11)
        ids = [case.id for case in cases]
        self.assertEqual(len(set(ids)), 11)
        self.assertTrue(all(case.category for case in cases))
        # K3 临时案例集只含 K3 合并案例；计费探针不再挂在它下面。
        self.assertEqual(len(server.kvv_temp_all_cases()), 24)
        self.assertIsNone(find_case(server.kvv_temp_all_cases(), "BILLING-U01"))

    def test_selection_maps_all_and_single(self):
        self.assertIsNone(server.billing_probe_selection(""))
        self.assertIsNone(server.billing_probe_selection(None))
        self.assertIsNone(server.billing_probe_selection("CACHE-HIT-7"))
        self.assertIsNone(server.billing_probe_selection("BILLING-NOPE"))
        self.assertEqual(server.billing_probe_selection("BILLING-U01"), ("BILLING-U01",))
        all_ids = server.billing_probe_selection(server.BILLING_PROBE_ALL)
        self.assertEqual(all_ids, server.billing_probe_ids())
        self.assertEqual(len(all_ids), 11)
        self.assertTrue(server.is_billing_probe("BILLING-ALL"))
        self.assertFalse(server.is_billing_probe(None))

    def test_case_title_covers_billing_options(self):
        self.assertEqual(server.billing_probe_title("BILLING-U03"), "思考余量指纹（output−thinking 与正文矛盾）")
        self.assertEqual(server.billing_probe_title("BILLING-ALL"), "中转计费探针 · 全部 11 条")
        self.assertEqual(server.billing_probe_title(server.BILLING_PROBE_CASE_ID), "中转计费探针")
        self.assertIsNone(server.billing_probe_title("FABLE5_BASIC"))
        # K3 案例集不再认计费项，K3 项照旧。
        self.assertIsNone(server.kvv_temp_case_title("BILLING-U03"))
        self.assertEqual(server.kvv_temp_case_title("CACHE-TTL-3"), "30 秒 TTL 命中探测")
        self.assertIsNone(server.kvv_temp_case_title(TEMP_CASE_ID))
        # 官方 KVV 套件名不会被误读成 K3 合并案例。
        self.assertIsNone(server.kvv_temp_case_title("KVV_PARAMS"))


if __name__ == "__main__":
    unittest.main()
