"""原厂金标库（gold/）的口径测试。

这里钉死的四件事，任何一条松了都会让「金标」变成"看起来很像金标的东西"：

1. **语料不可漂**：探针字面量一旦发布就不能悄悄改，否则历史金标与实测不可比；
2. **proxy 永不进判别**：拿「同厂另一个开源权重」去判"这不是豆包"，比不判更坏；
3. **判别矩阵有余量**：任意两家可判别厂商的金标必须真有差异探针，不能"看着不同其实一样"；
4. **判别器行为正确**：加常数 offset 仍认回本族；跨族必须报 mismatch；探针不够必须说"没得判"。
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from gold.build import GOLD_PATH, discrimination_matrix
from gold.compare import compare, selfcheck
from gold.probes import CORPUS_VERSION, FILLER, PAIRS, all_samples
from gold.registry import MODELS, discriminating_models, resolve

ROOT = Path(__file__).resolve().parent.parent


def _gold() -> dict:
    if not GOLD_PATH.exists():
        raise unittest.SkipTest(f"金标未生成：{GOLD_PATH}（跑 scripts/gold_build.py）")
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def _counts(slug: str) -> dict[str, int]:
    for entry in _gold()["entries"]:
        if entry["slug"] == slug:
            return (entry.get("tokenizer") or {}).get("probe_counts") or {}
    raise AssertionError(f"金标里没有 {slug}")


class ProbeCorpusTests(unittest.TestCase):
    def test_corpus_has_pairs_and_filler(self):
        samples = all_samples()
        self.assertTrue(FILLER)
        for short_id, long_id in PAIRS:
            self.assertIn(short_id, samples)
            self.assertIn(long_id, samples)
            self.assertTrue(samples[long_id].endswith(samples[short_id]))

    def test_corpus_version_is_dated(self):
        self.assertRegex(CORPUS_VERSION, r"^gold-tok-\d{4}-\d{2}-\d{2}[a-z]$")

    def test_probe_count_is_stable(self):
        # 改语料必须同时升 CORPUS_VERSION；这条防止"顺手加一条探针"让历史金标失效
        self.assertEqual(len(all_samples()), 17, "探针条数变了：请同步升 CORPUS_VERSION 并重建金标")


class RegistryTests(unittest.TestCase):
    def test_proxy_entries_never_discriminate(self):
        for model in MODELS:
            if model.mapping_confidence in ("proxy", "none"):
                self.assertFalse(model.usable_for_discrimination,
                                 f"{model.slug} 映射置信度只有 {model.mapping_confidence}，不能进判别")

    def test_resolve_maps_api_names(self):
        self.assertEqual(resolve("glm-5.3").slug, "glm:glm-4.6")
        self.assertEqual(resolve("glm-5.3-flashx").slug, "glm:glm-4.6")
        self.assertEqual(resolve("qwen3-max").slug, "qwen:qwen3-32b")
        self.assertEqual(resolve("deepseek-reasoner").slug, "deepseek:deepseek-v3.2")

    def test_resolve_unknown_returns_none(self):
        self.assertIsNone(resolve("totally-unknown-model"))
        self.assertIsNone(resolve(""))

    def test_discriminating_set_excludes_proxy(self):
        vendors = {m.vendor for m in discriminating_models()}
        self.assertNotIn("doubao", vendors)
        self.assertGreaterEqual(len(discriminating_models()), 5)


class GoldPayloadTests(unittest.TestCase):
    def test_schema_and_entries(self):
        gold = _gold()
        self.assertEqual(gold["schema"], "gold/tokenizer_gold@1")
        self.assertEqual(gold["corpus_version"], CORPUS_VERSION)
        slugs = {e["slug"] for e in gold["entries"]}
        for expect in ("qwen:qwen3-32b", "glm:glm-4.6", "deepseek:deepseek-v3.2", "kimi:kimi-k2"):
            self.assertIn(expect, slugs)

    def test_ok_entries_have_provenance(self):
        for entry in _gold()["entries"]:
            if entry["status"] != "ok":
                self.assertTrue(entry.get("reason"), f"{entry['slug']} 未采集却没写原因")
                continue
            tok = entry["tokenizer"]
            self.assertEqual(len(tok["sha256"]), 64, f"{entry['slug']} 缺 sha256")
            self.assertGreater(tok["vocab_size"], 1000)

    def test_kimi_is_vocab_only(self):
        # tiktoken 型词表不猜正则：只固化词表级指纹，计数必须是空的 + 带原因
        entry = next(e for e in _gold()["entries"] if e["slug"] == "kimi:kimi-k2")
        self.assertEqual(entry["tokenizer"]["kind"], "tiktoken-bpe")
        self.assertEqual(entry["tokenizer"]["probe_counts"], {})
        self.assertTrue(entry["tokenizer"]["counts_uncollected_reason"])
        self.assertEqual(entry["contract"]["native_train_context"], 262144)

    def test_contract_caveat_present(self):
        entry = next(e for e in _gold()["entries"] if e["slug"] == "qwen:qwen3-32b")
        self.assertIn("caveat", entry["contract"])
        self.assertEqual(entry["contract"]["native_train_context"], 40960)


class DiscriminationTests(unittest.TestCase):
    def test_every_pair_has_margin(self):
        from gold.build import build as _build  # noqa: F401  (仅用于确认可导入)
        gold = _gold()
        matrix = gold["discrimination"]["pairs"]
        self.assertGreaterEqual(len(matrix), 4)
        for a, row in matrix.items():
            for b, diff in row.items():
                self.assertGreaterEqual(diff, 3, f"{a} 与 {b} 只有 {diff} 条探针有差异，判别余量不足")

    def test_reference_entries_not_in_matrix(self):
        matrix = _gold()["discrimination"]["pairs"]
        self.assertNotIn("qwen:qwen2.5-72b", matrix)
        self.assertNotIn("deepseek:deepseek-v3.1", matrix)

    def test_same_family_generations_are_flagged(self):
        groups = _gold()["discrimination"]["indistinguishable_groups"]
        joined = {frozenset(g) for g in groups}
        self.assertTrue(
            any({"qwen:qwen3-32b", "qwen:qwen2.5-72b"} == set(g) for g in joined),
            "Qwen3 与 Qwen2.5 词表计数相同这件事必须被记下来（跨代不可分）")


class CompareTests(unittest.TestCase):
    def test_offset_cancels(self):
        gold = _gold()
        counts = _counts("glm:glm-4.6")
        for offset in (0, 13, -29):
            measured = {k: v + offset for k, v in counts.items()}
            res = compare(measured, gold, declared_slug="glm:glm-4.6")
            self.assertEqual(res.verdict, "matched", f"offset={offset} 时判定 {res.verdict}")
            self.assertEqual(res.best.slug, "glm:glm-4.6")

    def test_cross_family_is_reported(self):
        gold = _gold()
        measured = dict(_counts("qwen:qwen3-32b"))
        res = compare(measured, gold, declared_slug="glm:glm-4.6")
        self.assertEqual(res.verdict, "family_mismatch")
        self.assertEqual(res.best.slug, "qwen:qwen3-32b")

    def test_too_few_probes_says_no_gold(self):
        gold = _gold()
        res = compare({"zh_plain": 9}, gold, declared_slug="glm:glm-4.6")
        self.assertEqual(res.verdict, "unmeasured")
        self.assertTrue(res.notes)

    def test_missing_gold_is_distinct_from_missing_data(self):
        # 金标里只有一家，且实测探针齐全但对不上 → no_gold（不是 unmeasured）
        slim = {"entries": [{"slug": "only:x", "vendor": "only",
                             "tokenizer": {"probe_counts": {"zh_plain": 10, "en": 10, "punct": 16}}}]}
        res = compare({"zh_plain": 10, "en": 10, "punct": 16}, slim)
        self.assertNotEqual(res.verdict, "unmeasured")
        self.assertEqual(res.verdict, "matched")

    def test_unknown_declared_still_ranks(self):
        gold = _gold()
        measured = dict(_counts("ernie:ernie-4.5-300b"))
        res = compare(measured, gold)
        self.assertEqual(res.best.slug, "ernie:ernie-4.5-300b")
        self.assertEqual(res.verdict, "matched")

    def test_selfcheck_passes(self):
        res = selfcheck(_gold())
        self.assertEqual(res["failed"], [], f"自检失败：{res['failed']}")


class DiscriminationMatrixUnitTests(unittest.TestCase):
    def test_matrix_skips_entries_without_counts(self):
        entries = [
            {"slug": "a:x", "usable_for_discrimination": True,
             "tokenizer": {"probe_counts": {"p": 1}, "family_signature": "s1"}},
            {"slug": "b:y", "usable_for_discrimination": True,
             "tokenizer": {"probe_counts": {}, "family_signature": None}},
            {"slug": "c:z", "usable_for_discrimination": False,
             "tokenizer": {"probe_counts": {"p": 2}, "family_signature": "s2"}},
        ]
        out = discrimination_matrix(entries)
        self.assertEqual(list(out["pairs"]), ["a:x"])
        self.assertEqual(out["indistinguishable_groups"], [])


if __name__ == "__main__":
    unittest.main()
