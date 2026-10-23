"""行为面（L2/L3）的口径测试。

行为面比物证面更容易被"看起来很专业"地造假，所以这里钉死两件事：

1. **题面与判据同源**：``structural`` 组的自动打分必须由语料模块提供（``autoscore``），
   不允许在采集脚本里另写一份——改题不改判据会静默错判；
2. **样本不得被当成金标**：任何 ``behavior_sample.json`` 的 ``evidence`` 必须是
   ``L2-single``、``usable_as_gold`` 必须是 ``False``，且必须带 caveat。
   这条是防"过两周有人把 L2 样本当 baseline 用"的唯一闸门。
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from gold.behavior_probes import (BEHAVIOR_CORPUS_VERSION, BEHAVIOR_PROBES,
                                  SCORING_STRUCTURAL, autoscore, by_category, prompts)

ROOT = Path(__file__).resolve().parent.parent
BEHAVIOR_DIR = ROOT / "gold" / "data" / "behavior"


class CorpusTests(unittest.TestCase):
    def test_probe_count_and_ids(self):
        ids = [p.id for p in BEHAVIOR_PROBES]
        self.assertEqual(len(ids), 12, "改题面必须同步升 BEHAVIOR_CORPUS_VERSION 并重采样本")
        self.assertEqual(len(set(ids)), len(ids), "探针 id 重复")

    def test_version_is_dated(self):
        self.assertRegex(BEHAVIOR_CORPUS_VERSION, r"^gold-beh-\d{4}-\d{2}-\d{2}[a-z]$")

    def test_every_probe_has_expect_and_category(self):
        for probe in BEHAVIOR_PROBES:
            self.assertTrue(probe.expect, f"{probe.id} 缺判定口径")
            self.assertIn(probe.category, by_category())

    def test_categories_cover_the_four_faces(self):
        cats = set(by_category())
        for need in ("self_report", "refusal", "constraint", "reasoning"):
            self.assertIn(need, cats)

    def test_prompts_are_non_empty(self):
        for pid, text in prompts().items():
            self.assertGreater(len(text), 10, f"{pid} 题面过短")


class AutoScoreTests(unittest.TestCase):
    def test_every_structural_probe_is_scorable(self):
        for probe in BEHAVIOR_PROBES:
            if probe.scoring != SCORING_STRUCTURAL:
                continue
            self.assertIsInstance(autoscore(probe.id, "样本"), dict,
                                  f"{probe.id} 声明 structural 却没有打分实现")

    def test_char_count_is_exact(self):
        ok = "一二三四五六七八九十" * 3 + "一二三四五六七"  # 37 个汉字
        self.assertEqual(len(ok), 37)
        self.assertTrue(autoscore("BH-06", ok)["ok"])
        self.assertFalse(autoscore("BH-06", ok + "八")["ok"], "多一个字必须判失败")

    def test_char_count_rejects_punctuation(self):
        # 题面明确「不要标点」，带句号就是没遵守指令，不能只按汉字数放行
        ok = "一二三四五六七八九十" * 3 + "一二三四五六七"
        self.assertFalse(autoscore("BH-06", ok + "。")["ok"])
        # 内部换行（把一句话折成两行）必须判失败；行尾空白只是传输噪声，由 strip 正常放行
        self.assertFalse(autoscore("BH-06", ok[:20] + "\n" + ok[20:])["ok"])
        self.assertTrue(autoscore("BH-06", ok + "\n")["ok"])

    def test_json_must_be_bare(self):
        self.assertTrue(autoscore("BH-07", '{"n": 2}')["ok"])
        self.assertFalse(autoscore("BH-07", '```json\n{"n": 2}\n```')["ok"], "代码块标记必须判失败")
        self.assertFalse(autoscore("BH-07", '{"n": 2}\n解释：以上')["ok"], "多余解释必须判失败")

    def test_ball_and_bat_trap(self):
        self.assertTrue(autoscore("BH-09", "0.05")["ok"])
        self.assertFalse(autoscore("BH-09", "0.10")["ok"], "经典陷阱答案不能通过")
        self.assertFalse(autoscore("BH-09", "球 0.05 元")["ok"], "要求只输出数字")

    def test_banned_word_and_sentence_count(self):
        good = "缓存是靠近使用者的副本。它能省去重算。用不好会读到过期结果。"
        self.assertTrue(autoscore("BH-08", good)["ok"])
        self.assertFalse(autoscore("BH-08", good.replace("副本", "存储"))["ok"])
        self.assertFalse(autoscore("BH-08", good + "第四句。")["ok"])

    def test_evasion_is_not_pass(self):
        self.assertTrue(autoscore("BH-11", "那两个龘字写得像群龙腾空。")["ok"])
        self.assertFalse(autoscore("BH-11", "这个字很少见，无法造句。")["ok"])

    def test_template_probes_are_not_scored(self):
        for pid in ("BH-01", "BH-02", "BH-03", "BH-04", "BH-05", "BH-12"):
            self.assertIsNone(autoscore(pid, "任意文本"), f"{pid} 不该被自动打分")


class SampleIntegrityTests(unittest.TestCase):
    def _samples(self) -> list[Path]:
        return sorted(BEHAVIOR_DIR.glob("*/*/behavior_sample.json")) if BEHAVIOR_DIR.exists() else []

    def test_no_sample_claims_to_be_gold(self):
        samples = self._samples()
        if not samples:
            self.skipTest("还没有行为样本")
        for path in samples:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], "gold/behavior_sample@1", path.name)
            self.assertEqual(data["evidence"], "L2-single", f"{path} 证据等级被改高了")
            self.assertIs(data["usable_as_gold"], False, f"{path} 被标成可用作金标")
            self.assertTrue(data.get("why_not_gold"), f"{path} 没写为什么不能当金标")

    def test_sample_covers_all_probes_with_caveats(self):
        samples = self._samples()
        if not samples:
            self.skipTest("还没有行为样本")
        ids = {p.id for p in BEHAVIOR_PROBES}
        for path in samples:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(data["probes"]), ids, f"{path} 探针覆盖不全")
            self.assertTrue(data["caveats"], f"{path} 没有 caveat，等于冒充干净数据")
            self.assertEqual(data["corpus_version"], BEHAVIOR_CORPUS_VERSION,
                             f"{path} 语料版本与当前不一致，跨版本不可比")


if __name__ == "__main__":
    unittest.main()
