"""中转计费探针的判定测试。

重点不是「代码能跑」，而是**判定能不能分辨口径**：同一份实现要能分别判出
「按上限扣款」与「伪造 usage」，所以用 `billing_fake_relay` 的两个正交轴
（计费口径 × 用量真实性）交叉验证。
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
for candidate in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import run_relay_billing_probe as probe_mod  # noqa: E402
from billing_fake_relay import (  # noqa: E402
    MODE_BOTH,
    MODE_CAP_CHARGE,
    MODE_CHANNEL_DOWN,
    MODE_FAKE_USAGE,
    MODE_HONEST,
    MODE_NO_ORACLE,
    FakeRelay,
)
from common.test_framework import HttpExecutor, ResultLog  # noqa: E402


class ProbeHarness(unittest.IsolatedAsyncioTestCase):
    """统一的探针装配：MockTransport 挂假上游 + 临时证据目录。"""

    async def build(self, relay, *, protocol: str = "anthropic", **overrides):
        relay.protocol = protocol
        # 生产默认 (0,3,8) 秒是为了捕捉「先占用后释放」的回弹时延；单测只需覆盖
        # 「多次回读」这个形状，用秒级间隔即可，避免每条用例白等 11 秒。
        overrides.setdefault("release_delays", (0, 1, 1))
        config = probe_mod.ProbeConfig(
            base_url="https://relay.test", api_key="sk-test", model="claude-fable-5",
            protocol=protocol, **overrides,
        )
        client = httpx.AsyncClient(transport=httpx.MockTransport(relay.as_httpx_handler()))
        executor = HttpExecutor(client=client)
        temp = Path(tempfile.mkdtemp(prefix="billing-probe-"))
        log = ResultLog(temp / "requests.jsonl")
        instance = probe_mod.BillingProbe(config, executor, log)
        self.addAsyncCleanup(client.aclose)
        self.addCleanup(log.close)
        return instance

    async def run_case(self, instance, case_id: str):
        spec = next(item for item in probe_mod.PROBES if item[0] == case_id)
        _, name, category, handler = spec
        return await getattr(instance, handler)(case_id, name, category)


# --------------------------------------------------------------------------- #
# 用量真实性轴
# --------------------------------------------------------------------------- #
class UsageTruthAxisTests(ProbeHarness):
    async def test_honest_relay_passes_usage_probes(self):
        instance = await self.build(FakeRelay(MODE_HONEST))
        for case_id in ("BILLING-U01", "BILLING-U02", "BILLING-U04", "BILLING-U05", "BILLING-U06"):
            outcome = await self.run_case(instance, case_id)
            self.assertEqual(outcome.status, "passed", f"{case_id}: {outcome.details}")

    async def test_faked_usage_tracking_max_tokens_is_a_red_line(self):
        instance = await self.build(FakeRelay(MODE_FAKE_USAGE))
        outcome = await self.run_case(instance, "BILLING-U01")
        self.assertEqual(outcome.status, "failed")
        self.assertIn("output_tokens 精确等于 max_tokens", outcome.details)
        self.assertEqual(outcome.observations["ladder"][0]["output_tokens"], [64, 64])

    async def test_faked_usage_ignoring_answer_length_is_a_red_line(self):
        instance = await self.build(FakeRelay(MODE_FAKE_USAGE))
        outcome = await self.run_case(instance, "BILLING-U02")
        self.assertEqual(outcome.status, "failed")
        self.assertIn("与真实输出长度无关", outcome.details)

    async def test_honest_usage_under_cap_charging_still_passes_usage_probes(self):
        """计费按下限收 ≠ 用量伪造——两个轴必须能分开判。"""
        instance = await self.build(FakeRelay(MODE_CAP_CHARGE))
        for case_id in ("BILLING-U01", "BILLING-U02"):
            outcome = await self.run_case(instance, case_id)
            self.assertEqual(outcome.status, "passed", f"{case_id}: {outcome.details}")

    async def test_truncated_long_answer_does_not_trigger_item_lower_bound(self):
        """长答案被上限截断时正文本身不完整，「条目数下界」不能用——否则冤枉通道。"""
        instance = await self.build(FakeRelay(MODE_HONEST), ladder=(256, 2048))
        outcome = await self.run_case(instance, "BILLING-U02")
        self.assertEqual(outcome.status, "passed", outcome.details)
        self.assertEqual(outcome.observations["long"]["stop_reason"], "max_tokens")

    async def test_frozen_input_tokens_detected(self):
        instance = await self.build(FakeRelay(MODE_BOTH))
        outcome = await self.run_case(instance, "BILLING-U04")
        self.assertEqual(outcome.status, "failed")
        self.assertIn("input_tokens 与输入长度脱钩", outcome.details.replace("输入侧账目", "input_tokens 与输入长度脱钩"))

    async def test_thinking_residual_fingerprint_flags_contradiction(self):
        instance = await self.build(FakeRelay(MODE_FAKE_USAGE))
        await self.run_case(instance, "BILLING-U02")   # 先攒出长正文的响应
        outcome = await self.run_case(instance, "BILLING-U03")
        self.assertEqual(outcome.status, "failed")
        self.assertIn("账目自相矛盾", outcome.details)
        self.assertGreater(outcome.observations["residual_le_2_count"], 0)

    async def test_thinking_residual_passes_when_accounting_self_consistent(self):
        instance = await self.build(FakeRelay(MODE_HONEST))
        await self.run_case(instance, "BILLING-U01")
        outcome = await self.run_case(instance, "BILLING-U03")
        self.assertEqual(outcome.status, "passed", outcome.details)

    async def test_thinking_switch_honored_passes_and_ignored_warns(self):
        instance = await self.build(FakeRelay(MODE_HONEST))
        self.assertEqual((await self.run_case(instance, "BILLING-U06")).status, "passed")
        instance = await self.build(FakeRelay(MODE_BOTH))
        outcome = await self.run_case(instance, "BILLING-U06")
        self.assertEqual(outcome.status, "warning")
        self.assertIn("思考不可关闭", outcome.details)


# --------------------------------------------------------------------------- #
# 计费口径轴
# --------------------------------------------------------------------------- #
class BillingAxisTests(ProbeHarness):
    async def test_cap_charging_is_convicted(self):
        instance = await self.build(FakeRelay(MODE_CAP_CHARGE))
        outcome = await self.run_case(instance, "BILLING-B01")
        self.assertEqual(outcome.status, "failed")
        self.assertIn("预扣未按实际用量退回", outcome.details)
        self.assertIn("oracle", outcome.observations)
        self.assertEqual(outcome.observations["oracle"], "dashboard")

    async def test_usage_charging_is_absolved(self):
        """用量被伪造、但扣款按真实用量结算时，计费口径这条不能冤枉通道。"""
        instance = await self.build(FakeRelay(MODE_FAKE_USAGE))
        outcome = await self.run_case(instance, "BILLING-B01")
        self.assertEqual(outcome.status, "passed", outcome.details)
        self.assertGreater(outcome.observations["content_factor_delta_usd"], 0)

    async def test_precharge_without_refund_is_convicted(self):
        instance = await self.build(FakeRelay(MODE_CAP_CHARGE))
        outcome = await self.run_case(instance, "BILLING-B02")
        self.assertEqual(outcome.status, "failed")
        self.assertIn("未退回", outcome.details)
        self.assertGreaterEqual(len(outcome.observations["readings"]), 3)

    async def test_settlement_by_real_usage_is_absolved(self):
        instance = await self.build(FakeRelay(MODE_HONEST))
        outcome = await self.run_case(instance, "BILLING-B02")
        self.assertEqual(outcome.status, "passed", outcome.details)

    async def test_billing_probes_block_when_no_balance_oracle(self):
        """没有计量口径时只能 blocked——缺取证能力不等于通道有问题。"""
        instance = await self.build(FakeRelay(MODE_NO_ORACLE))
        for case_id in ("BILLING-B01", "BILLING-B02", "BILLING-B03"):
            outcome = await self.run_case(instance, case_id)
            self.assertEqual(outcome.status, "blocked", f"{case_id}: {outcome.details}")
        self.assertIn("无法取证", (await self.run_case(instance, "BILLING-B01")).details)

    async def test_oracle_falls_back_to_403_remain_quota(self):
        """计费端点缺失但余额偏低时，超额探针能顶上来做精确度量。"""
        relay = FakeRelay(MODE_CAP_CHARGE, initial_balance=0.03, dashboard=False)
        instance = await self.build(relay, ladder=(16, 64), balance_probe_mt=2048)
        outcome = await self.run_case(instance, "BILLING-B01")
        self.assertEqual(instance.oracle_kind, "403_remain")
        self.assertEqual(outcome.status, "failed", outcome.details)
        self.assertEqual(instance.resolution, probe_mod.ORACLE_RESOLUTION["403_remain"])

    async def test_resolution_guard_downgrades_to_warning(self):
        """粗口径（官方计费端点只到分）分辨不出小档扣款时必须降级为 warning，不能硬判。"""
        instance = await self.build(FakeRelay(MODE_CAP_CHARGE), ladder=(16, 64))
        outcome = await self.run_case(instance, "BILLING-B01")
        self.assertEqual(outcome.status, "warning")
        self.assertIn("分辨率", outcome.details)

    async def test_identical_request_charges_are_repeatable(self):
        instance = await self.build(FakeRelay(MODE_CAP_CHARGE))
        outcome = await self.run_case(instance, "BILLING-B03")
        self.assertEqual(outcome.status, "passed", outcome.details)

    async def test_quota_floor_flags_held_reserve(self):
        """余额远高于本次冻结额却仍被拒 → 此前冻结未释放。"""

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            body = json.loads(request.content.decode()) if request.content else {}
            if path.endswith("/dashboard/billing/subscription"):
                return httpx.Response(200, json={"hard_limit_usd": 5.0})
            if path.endswith("/dashboard/billing/usage"):
                return httpx.Response(200, json={"total_usage": 10.0})
            if path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "claude-fable-5"}]})
            if int(body.get("max_tokens") or 0) < 128:
                return httpx.Response(403, json={"type": "error", "error": {
                    "type": "i_token_error", "message": "无权访问 ccmax 分组"}})
            return httpx.Response(200, json={
                "id": "m", "type": "message", "role": "assistant", "model": "claude-fable-5",
                "content": [{"type": "text", "text": "OK"}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 2}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        executor = HttpExecutor(client=client)
        temp = Path(tempfile.mkdtemp(prefix="billing-probe-"))
        log = ResultLog(temp / "requests.jsonl")
        self.addAsyncCleanup(client.aclose)
        self.addCleanup(log.close)
        config = probe_mod.ProbeConfig(base_url="https://relay.test", api_key="sk-test", model="claude-fable-5")
        instance = probe_mod.BillingProbe(config, executor, log)
        outcome = await self.run_case(instance, "BILLING-B04")
        self.assertEqual(outcome.status, "failed")
        self.assertIn("未释放", outcome.details)


# --------------------------------------------------------------------------- #
# 其它
# --------------------------------------------------------------------------- #
class ProbeMiscTests(ProbeHarness):
    async def test_quota_floor_blocks_when_balance_cannot_cover_the_tiny_reserve(self):
        """余额连本次冻结额都覆盖不了时被拒属**正常预扣检查**，不能据此指控「冻结未释放」。"""

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            body = json.loads(request.content.decode()) if request.content else {}
            if path.endswith("/dashboard/billing/subscription"):
                return httpx.Response(200, json={"hard_limit_usd": 1.0})
            if path.endswith("/dashboard/billing/usage"):
                return httpx.Response(200, json={"total_usage": 99.95})   # 余额 $0.0005
            if path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "claude-fable-5"}]})
            return httpx.Response(403, json={"type": "error", "error": {
                "type": "i_token_error",
                "message": "token quota is not enough, token remain quota: $0.000500, need quota: $0.000800"}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        executor = HttpExecutor(client=client)
        temp = Path(tempfile.mkdtemp(prefix="billing-probe-"))
        log = ResultLog(temp / "requests.jsonl")
        self.addAsyncCleanup(client.aclose)
        self.addCleanup(log.close)
        config = probe_mod.ProbeConfig(base_url="https://relay.test", api_key="sk-test", model="claude-fable-5")
        instance = probe_mod.BillingProbe(config, executor, log)
        outcome = await self.run_case(instance, "BILLING-B04")
        self.assertEqual(outcome.status, "blocked", outcome.details)
        self.assertIn("正常的预扣检查", outcome.details)
        self.assertIn("花不出去", outcome.details)

    async def test_channel_down_blocks_every_channel_verdict(self):
        instance = await self.build(FakeRelay(MODE_CHANNEL_DOWN))
        outcome = await self.run_case(instance, "BILLING-U01")
        self.assertEqual(outcome.status, "blocked")
        self.assertIn("平台级前提不满足", outcome.details)

    async def test_channel_probe_warns_when_model_not_in_catalog(self):
        relay = FakeRelay(MODE_HONEST)
        relay.model = "claude-fable-5"
        instance = await self.build(relay)
        instance.cfg.model = "claude-sonnet-9"
        outcome = await self.run_case(instance, "BILLING-S01")
        self.assertEqual(outcome.status, "warning")
        self.assertIn("未在", outcome.details)

    async def test_openai_protocol_usage_is_normalized(self):
        instance = await self.build(FakeRelay(MODE_FAKE_USAGE), protocol="openai")
        outcome = await self.run_case(instance, "BILLING-U01")
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.observations["ladder"][0]["output_tokens"], [64, 64])

    async def test_every_probe_yields_a_result_and_summary_contract_holds(self):
        """整轮跑通：每条探针都要有结论，汇总/报告的契约字段齐全。"""
        instance = await self.build(FakeRelay(MODE_BOTH))
        results = [await self.run_case(instance, spec[0]) for spec in probe_mod.PROBES]
        self.assertEqual(len(results), len(probe_mod.PROBES))
        self.assertTrue(all(item.status in {"passed", "failed", "warning", "blocked"} for item in results))
        self.assertTrue(all(item.details for item in results))
        summary = probe_mod.summarize(results, instance.cfg)
        self.assertEqual(summary["total"], len(probe_mod.PROBES))
        self.assertEqual(summary["test_kind"], "billing-probe")
        self.assertFalse(summary["admission_passed"])          # 复刻 i-token：有 failed → 不准入
        self.assertEqual(summary["failed"] + summary["passed"] + summary["warning"] + summary["blocked"],
                         len(probe_mod.PROBES))
        self.assertTrue(all("case_id" in item and "status" in item for item in summary["results"]))

    async def test_admission_passes_on_honest_relay(self):
        instance = await self.build(FakeRelay(MODE_HONEST))
        results = [await self.run_case(instance, spec[0]) for spec in probe_mod.PROBES]
        summary = probe_mod.summarize(results, instance.cfg)
        self.assertTrue(summary["admission_passed"],
                        [f"{r.case_id}={r.status}:{r.details[:80]}" for r in results if r.status != "passed"])

    async def test_report_written_with_every_case(self):
        instance = await self.build(FakeRelay(MODE_HONEST))
        results = [await self.run_case(instance, spec[0]) for spec in probe_mod.PROBES]
        summary = probe_mod.summarize(results, instance.cfg)
        temp = Path(tempfile.mkdtemp(prefix="billing-report-"))
        probe_mod.write_report(temp, summary, instance.cfg)
        text = (temp / "report.md").read_text(encoding="utf-8")
        for spec in probe_mod.PROBES:
            self.assertIn(spec[0], text)


class ThinkingAndTransientTests(ProbeHarness):
    """2026-10-09 sonnet-5 批次暴露的两类判定缺陷：thinking 字段值 0 被误读、5xx 不打重试。"""

    # ---------- thinking_tokens 的 key 存在 ≠ 真的在思考 ---------- #
    async def test_zero_thinking_field_is_not_read_as_thinking_charged(self):
        """复刻 sonnet-5 批次：两档 thinking_tokens 均为 0（但 key 在）→ 不能判「思考不可关闭」。"""
        instance = await self.build(FakeRelay(MODE_HONEST, zero_thinking=True))
        outcome = await self.run_case(instance, "BILLING-U06")
        self.assertEqual(outcome.status, "warning", outcome.details)
        self.assertNotIn("思考不可关闭", outcome.details)
        self.assertIn("opt-in", outcome.details)
        self.assertEqual(outcome.observations["default"]["thinking_tokens"], 0)

    async def test_thinking_residual_reports_no_power_when_field_is_zero(self):
        """余量指纹在 thinking 恒 0 时是空断言，必须留痕而不是报「自洽」。"""
        instance = await self.build(FakeRelay(MODE_HONEST, zero_thinking=True))
        await self.run_case(instance, "BILLING-U01")
        outcome = await self.run_case(instance, "BILLING-U03")
        self.assertEqual(outcome.status, "warning", outcome.details)
        self.assertIn("无判别力", outcome.details)
        self.assertEqual(outcome.observations["responses_with_thinking_tokens"], 0)
        self.assertGreater(outcome.observations["responses_thinking_zero"], 0)

    async def test_real_thinking_still_detected_as_unclosable(self):
        """回归：真的带思考用量时，原「思考不可关闭」结论不能被改掉。"""
        instance = await self.build(FakeRelay(MODE_BOTH))
        outcome = await self.run_case(instance, "BILLING-U06")
        self.assertEqual(outcome.status, "warning")
        self.assertIn("思考不可关闭", outcome.details)

    # ---------- 瞬时故障：先重试，再 blocked，绝不 failed ---------- #
    async def test_transient_5xx_is_retried_and_recovers(self):
        relay = FakeRelay(MODE_HONEST, transient_5xx=1)
        instance = await self.build(relay, max_retries=1, retry_backoff=0)
        outcome = await self.run_case(instance, "BILLING-S01")
        self.assertEqual(outcome.status, "passed", outcome.details)
        self.assertEqual(len(instance.transient_failures), 1)
        self.assertIn("HTTP 502", instance.transient_failures[0])

    async def test_persistent_5xx_blocks_instead_of_failing_and_names_the_gateway(self):
        """网关成片 502 时：U 系列收口 blocked（不得 failed），且文案必须写明是网关 502。"""
        relay = FakeRelay(MODE_CAP_CHARGE, transient_5xx=999)
        instance = await self.build(relay, max_retries=1, retry_backoff=0)
        usage = await self.run_case(instance, "BILLING-U01")
        self.assertEqual(usage.status, "blocked", usage.details)
        self.assertIn("瞬时", usage.details)
        billing = await self.run_case(instance, "BILLING-B02")
        self.assertEqual(billing.status, "blocked", billing.details)
        self.assertIn("502", billing.details)
        self.assertIn("网关", billing.details)

    async def test_blocked_reason_names_the_failing_endpoint(self):
        """余额 oracle 不可用时，blocked 文案要带上是哪个端点、什么状态，别只写「无可用余额读数」。"""
        instance = await self.build(FakeRelay(MODE_NO_ORACLE), max_retries=0)
        outcome = await self.run_case(instance, "BILLING-B02")
        self.assertEqual(outcome.status, "blocked", outcome.details)
        self.assertIn("读数失败原因", outcome.details)
        # 两条路径都要留痕：计费端点 404 + 超额探针未被拒（200）
        self.assertIn("dashboard/billing/subscription HTTP 404", outcome.details)
        self.assertIn("未被拒", outcome.details)


class RateCalibrationTests(ProbeHarness):
    """单价自动反推：这是「同一套探针跨模型通用」的关键——不必按模型手填 --rate-per-m。"""

    async def test_rate_is_calibrated_from_403_need_quota(self):
        relay = FakeRelay(MODE_CAP_CHARGE, rate_per_m=15.0, initial_balance=0.03, dashboard=False)
        instance = await self.build(relay, ladder=(16, 64), balance_probe_mt=2048)
        outcome = await self.run_case(instance, "BILLING-B01")
        self.assertEqual(instance.cfg.rate_source, "calibrated_from_403")
        self.assertAlmostEqual(instance.cfg.rate_per_m, 15.0, places=6)
        self.assertTrue(instance.cfg.rate_samples)
        # 冻结额换算跟着标定值走，而不是默认的 50
        self.assertAlmostEqual(outcome.observations["A_small_cap_long_answer"]["reserve_usd"],
                               16 * 15.0 / 1e6, places=9)
        self.assertEqual(outcome.status, "failed", outcome.details)

    async def test_explicit_rate_wins_and_is_noted(self):
        relay = FakeRelay(MODE_CAP_CHARGE, rate_per_m=15.0, initial_balance=0.03, dashboard=False)
        instance = await self.build(relay, ladder=(16, 64), balance_probe_mt=2048,
                                    rate_per_m=99.0, rate_explicit=True, rate_source="configured")
        await self.run_case(instance, "BILLING-B01")
        self.assertEqual(instance.cfg.rate_per_m, 99.0)
        self.assertEqual(instance.cfg.rate_source, "configured")
        self.assertTrue(any("不符" in note for note in instance.notes), instance.notes)

    async def test_low_precision_probe_does_not_calibrate(self):
        """上限 16 的 403 只够 6 位小数的量化噪声，不能当标定基准。"""
        relay = FakeRelay(MODE_CAP_CHARGE, initial_balance=0.0005, dashboard=True)
        instance = await self.build(relay)
        outcome = await self.run_case(instance, "BILLING-B04")
        self.assertEqual(outcome.status, "blocked", outcome.details)
        self.assertEqual(instance.cfg.rate_source, "default")
        self.assertEqual(instance.cfg.rate_samples, ())

    async def test_summary_and_report_carry_rate_provenance(self):
        relay = FakeRelay(MODE_CAP_CHARGE, rate_per_m=15.0, initial_balance=0.03, dashboard=False)
        instance = await self.build(relay, ladder=(16, 64), balance_probe_mt=2048)
        results = [await self.run_case(instance, "BILLING-B01")]
        summary = probe_mod.summarize(results, instance.cfg, instance.notes)
        self.assertEqual(summary["rate_per_m_source"], "calibrated_from_403")
        self.assertTrue(summary["rate_per_m_samples"])
        temp = Path(tempfile.mkdtemp(prefix="billing-report-"))
        probe_mod.write_report(temp, summary, instance.cfg)
        text = (temp / "report.md").read_text(encoding="utf-8")
        self.assertIn("反推标定", text)
        self.assertIn("预扣单价", text)

    def test_default_rate_config_is_not_marked_explicit(self):
        """默认构造（无显式单价）必须允许被标定覆盖。"""
        config = probe_mod.ProbeConfig(base_url="https://relay.test", api_key="sk-test",
                                       model="claude-fable-5")
        self.assertFalse(config.rate_explicit)
        self.assertEqual(config.rate_source, "default")
        self.assertEqual(config.rate_samples, ())


class ProbeSelectionTests(ProbeHarness):
    """临时案例集按条选择：清单、依赖展开与只跑选中项。"""

    def test_manifest_matches_runner_catalog(self):
        manifest = json.loads((ROOT / "test_cases" / "relay_billing_cases.json").read_text(encoding="utf-8"))
        cases = [item for item in manifest["cases"] if item.get("id")]
        # Web 端选项 id 必须与执行器目录完全一致，否则选中项会被执行器当未知项拒绝。
        self.assertEqual(tuple(item["id"] for item in cases), probe_mod.PROBE_ORDER)
        names = {item["id"]: item["name"] for item in cases}
        for case_id, name, category, _ in probe_mod.PROBES:
            self.assertEqual(names[case_id], name)
            self.assertEqual(next(item["category"] for item in cases if item["id"] == case_id), category)

    def test_expand_selection_handles_all_single_and_unknown(self):
        self.assertEqual(probe_mod.expand_probe_selection([]), probe_mod.PROBE_ORDER)
        self.assertEqual(probe_mod.expand_probe_selection(["  "]), probe_mod.PROBE_ORDER)
        self.assertEqual(probe_mod.expand_probe_selection(["BILLING-B02"]), ("BILLING-B02",))
        # 顺序按清单稳定排序，不随用户输入顺序变化
        self.assertEqual(probe_mod.expand_probe_selection(["BILLING-B02", "BILLING-S01"]),
                         ("BILLING-S01", "BILLING-B02"))
        with self.assertRaises(ValueError):
            probe_mod.expand_probe_selection(["BILLING-NOPE"])

    def test_thinking_residual_selection_pulls_dependencies(self):
        # U03 是对样本的汇总分析，单跑会只剩「样本不足」，必须自动补样本来源。
        selected = probe_mod.expand_probe_selection(["BILLING-U03"])
        self.assertEqual(selected, ("BILLING-U01", "BILLING-U02", "BILLING-U03"))

    async def test_subset_run_records_only_selected_probes(self):
        selected = probe_mod.expand_probe_selection(["BILLING-B02"])
        instance = await self.build(FakeRelay(MODE_HONEST), probes=selected)
        results = [await self.run_case(instance, spec[0]) for spec in probe_mod.PROBES if spec[0] in instance.cfg.probes]
        self.assertEqual([item.case_id for item in results], ["BILLING-B02"])
        summary = probe_mod.summarize(results, instance.cfg)
        self.assertEqual(summary["selected_probes"], ["BILLING-B02"])
        self.assertEqual(summary["probe_catalog"], list(probe_mod.PROBE_ORDER))
        self.assertEqual(summary["total"], 1)

    async def test_thinking_residual_flags_contradiction_after_dependencies(self):
        selected = probe_mod.expand_probe_selection(["BILLING-U03"])
        instance = await self.build(FakeRelay(MODE_FAKE_USAGE), probes=selected)
        results = [await self.run_case(instance, spec[0]) for spec in probe_mod.PROBES if spec[0] in instance.cfg.probes]
        self.assertEqual([item.case_id for item in results], ["BILLING-U01", "BILLING-U02", "BILLING-U03"])
        # 补上前置样本后，U03 才能给出账目矛盾结论，而不是「样本不足」。
        self.assertEqual(results[-1].status, "failed", results[-1].details)
        self.assertIn("自相矛盾", results[-1].details)


if __name__ == "__main__":
    unittest.main()
