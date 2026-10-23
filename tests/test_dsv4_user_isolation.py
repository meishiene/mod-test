"""DS-V4-16（user_id 缓存隔离）判定回归：命中来源三分法。

背景（2026-10-21）：该用例曾把「不同 user_id 也命中」直接判成模型侧数据隔离失效。
但同一现象在三种链路上含义完全不同：

1. DeepSeek 原生 KV 缓存（响应带 `prompt_cache_hit_tokens`）跨 user 命中 ⇒ 真问题，failed；
2. 网关整段响应回放（不同 user 拿到同一个响应 `id`）⇒ 网关侧行为，blocked；
3. 网关/上游服务层的 OpenAI 形状前缀缓存（只有 `prompt_tokens_details.cached_tokens`、
   从不出原生字段、响应 `id` 每次都新）⇒ 无法归因到 DeepSeek 原生 KV，blocked。

真实证据：`results/DS_CACHE_PROBE-20261021-144307/cache_probe.json`（三款模型都是第 3 种，
`cached_tokens` 随前缀长度 64 对齐增长 384/896/1920/4096，换 user_id 数值不变）。
"""
import unittest
from unittest import mock

import httpx

from dsv4.core import DeepSeekConfig, DeepSeekV4Runner


def _chat(*, hit: int | None = None, cached: int | None = None, rid: str = "id-1", content: str = "OK") -> dict:
    usage: dict = {"prompt_tokens": 4213, "completion_tokens": 8, "total_tokens": 4221}
    if hit is not None:
        usage["prompt_cache_hit_tokens"] = hit
        usage["prompt_cache_miss_tokens"] = 4213 - hit
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"id": rid, "created": 1792564000, "choices": [{"message": {"role": "assistant", "content": content}}], "usage": usage}


class UserIsolationCaseTests(unittest.IsolatedAsyncioTestCase):
    async def _run_case(self, responses: list[dict]):
        queue = list(responses)

        async def handler(request):
            return httpx.Response(200, json=queue.pop(0))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        runner = DeepSeekV4Runner(DeepSeekConfig("key", "https://api.deepseek.com"), client)
        try:
            with mock.patch("dsv4.core.asyncio.sleep", new=mock.AsyncMock()):
                return await runner.case_user_isolation("deepseek-v4.1-flash-ark")
        finally:
            await client.aclose()

    async def test_native_fields_across_users_is_failed(self):
        """原生 prompt_cache_hit_tokens 换 user 仍命中 ⇒ 真的跨 user 命中，判 failed。"""
        outcome = await self._run_case([
            _chat(hit=0, rid="cold"),
            _chat(hit=1536, rid="same"),
            _chat(hit=1536, rid="other"),
        ])
        self.assertFalse(outcome[0])
        self.assertEqual(len(outcome), 3)  # 无 status ⇒ 由 run() 落成 failed
        self.assertIn("原生字段", outcome[1])
        self.assertEqual(outcome[2]["cache_field_shape"], "deepseek_native")

    async def test_openai_shape_only_is_blocked_not_failed(self):
        """只有 OpenAI 形状 cached_tokens、id 每次新 ⇒ 命中来源不可归因，blocked。"""
        outcome = await self._run_case([
            _chat(cached=0, rid="cold"),
            _chat(cached=4096, rid="same"),
            _chat(cached=4096, rid="other"),
        ])
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")
        self.assertIn("无法归因到 DeepSeek 原生 KV", outcome[1])
        self.assertEqual(outcome[2]["cache_field_shape"], "openai_cached_tokens")
        self.assertFalse(outcome[2]["gateway_replayed_response"])

    async def test_replayed_response_id_is_blocked(self):
        """不同 user 拿到同一个响应 id ⇒ 网关整段响应回放，blocked。"""
        outcome = await self._run_case([
            _chat(cached=0, rid="cold"),
            _chat(cached=4096, rid="replay-me"),
            _chat(cached=4096, rid="replay-me"),
        ])
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")
        self.assertTrue(outcome[2]["gateway_replayed_response"])
        self.assertIn("响应回放", outcome[1])

    async def test_isolation_holds_passes(self):
        outcome = await self._run_case([
            _chat(hit=0, rid="cold"),
            _chat(hit=1536, rid="same"),
            _chat(hit=0, rid="other"),
        ])
        self.assertTrue(outcome[0])
        self.assertEqual(len(outcome), 3)

    async def test_no_hit_is_blocked_best_effort(self):
        """同 user 都没命中（best-effort 缓存）⇒ 无法判定，blocked。"""
        outcome = await self._run_case([
            _chat(cached=0, rid="a"),
            _chat(cached=0, rid="b"),
            _chat(cached=0, rid="c"),
        ])
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "blocked")
