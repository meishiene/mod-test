"""fable5.model_profile：thinking 档位与 Bedrock modelId 归一的离线测试。

钉住三件事（2026-10-20 实测于 New API 中转 → AWS Bedrock）：

1. **档位按模型族三分**：4.5 系与 4.6 系只认 enabled、5 系只认 adaptive+effort，
   fable 系与未识别模型回退 adaptive（与引入本模块前的默认一致）；
2. **Bedrock modelId 归一**：``anthropic.X-v1:0`` ↔ ``X`` 视为同一模型，日期后缀不被剥；
   但不做前缀/别名模糊匹配——请求 haiku 回 opus 必须不等；
3. **请求片段形状**：enabled 带 budget_tokens、adaptive_effort 带 output_config.effort、
   adaptive 不带 output_config。
"""
import unittest

from fable5 import model_profile as mp


class ThinkingModeTests(unittest.TestCase):
    def test_four_five_family_uses_enabled(self):
        # 该通道对 4.5 系发 adaptive 直接 400 "adaptive thinking is not supported".
        for model in ("claude-haiku-4-5", "claude-haiku-4-5-20251001",
                      "claude-opus-4-5", "claude-opus-4-5-20251101",
                      "claude-sonnet-4-5-20250929"):
            with self.subTest(model=model):
                self.assertEqual(mp.resolve_thinking_mode(model), "enabled")

    def test_four_six_family_uses_enabled(self):
        # 4.6 系两档都通，但 adaptive 不透传 thinking 块（取不到签名）。
        for model in ("claude-opus-4-6", "claude-sonnet-4-6"):
            with self.subTest(model=model):
                self.assertEqual(mp.resolve_thinking_mode(model), "enabled")

    def test_five_family_uses_adaptive_effort(self):
        # 5 系发 enabled 直接 400 并要求 output_config.effort.
        for model in ("claude-sonnet-5", "claude-opus-5-5"):
            with self.subTest(model=model):
                self.assertEqual(mp.resolve_thinking_mode(model), "adaptive_effort")

    def test_fable_and_unknown_fall_back_to_adaptive(self):
        for model in ("claude-fable-5", "claude-opus-4-1", "claude-opus-4",
                      "relay-haiku", "gpt-5", "", None):
            with self.subTest(model=model):
                self.assertEqual(mp.resolve_thinking_mode(model), "adaptive")

    def test_bedrock_spelling_resolves_the_same_family(self):
        self.assertEqual(mp.resolve_thinking_mode("anthropic.claude-haiku-4-5-20251001-v1:0"),
                         "enabled")
        self.assertEqual(mp.resolve_thinking_mode("ANTHROPIC.Claude-Sonnet-5-v1:0"),
                         "adaptive_effort")


class ModelIdTests(unittest.TestCase):
    def test_bedrock_prefix_and_version_are_stripped(self):
        self.assertEqual(mp.normalize_model_id("anthropic.claude-haiku-4-5-20251001-v1:0"),
                         "claude-haiku-4-5-20251001")
        self.assertEqual(mp.normalize_model_id("claude-opus-4-6-v1"), "claude-opus-4-6")
        # 幂等：归一再归一不变。
        once = mp.normalize_model_id("anthropic.claude-sonnet-5-v1:0")
        self.assertEqual(mp.normalize_model_id(once), once)

    def test_date_suffix_is_not_a_version(self):
        # -20251001 是快照日期，不是 -v1 形态的版本号，必须保留。
        self.assertEqual(mp.normalize_model_id("anthropic.claude-haiku-4-5-20251001-v1:0"),
                         "claude-haiku-4-5-20251001")

    def test_same_model_id_only_normalizes(self):
        self.assertTrue(mp.same_model_id("anthropic.claude-haiku-4-5-20251001-v1:0",
                                         "claude-haiku-4-5-20251001"))
        self.assertTrue(mp.same_model_id("claude-opus-4-6", "Claude-Opus-4-6"))
        # 真换模型必须不等：不做前缀/别名模糊匹配。
        self.assertFalse(mp.same_model_id("anthropic.claude-opus-4-5-20251101-v1:0",
                                          "claude-haiku-4-5-20251001"))
        # 版本差异也不算同一模型（与签名内标识的前缀兼容口径区分开）。
        self.assertFalse(mp.same_model_id("claude-fable-5-1", "claude-fable-5"))
        self.assertFalse(mp.same_model_id("", "claude-opus-4-6"))
        self.assertFalse(mp.same_model_id(None, None))


class PayloadTests(unittest.TestCase):
    def test_enabled_payload_carries_budget(self):
        thinking, output_config = mp.thinking_payload("enabled", budget=2048)
        self.assertEqual(thinking, {"type": "enabled", "budget_tokens": 2048})
        self.assertIsNone(output_config)

    def test_adaptive_effort_payload_carries_effort(self):
        thinking, output_config = mp.thinking_payload("adaptive_effort")
        self.assertEqual(thinking, {"type": "adaptive", "display": "summarized"})
        self.assertEqual(output_config, {"effort": "high"})

    def test_adaptive_payload_has_no_output_config(self):
        thinking, output_config = mp.thinking_payload("adaptive")
        self.assertEqual(thinking, {"type": "adaptive", "display": "summarized"})
        self.assertIsNone(output_config)

    def test_every_resolved_mode_has_a_payload(self):
        # 解析出的每个档位都必须能构造出请求片段（防止档位名与构造分支脱节）。
        for mode in (mp.ENABLED, mp.ADAPTIVE, mp.ADAPTIVE_EFFORT,
                     mp.resolve_thinking_mode("claude-haiku-4-5"),
                     mp.resolve_thinking_mode("claude-sonnet-5")):
            with self.subTest(mode=mode):
                thinking, _ = mp.thinking_payload(mode)
                self.assertIn(thinking["type"], ("enabled", "adaptive"))


if __name__ == "__main__":
    unittest.main()
