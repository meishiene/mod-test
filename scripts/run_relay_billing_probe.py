#!/usr/bin/env python3
"""中转计费临时探针（一次性用例，不并入常驻套件）。

只回答一个问题：**这条中转通道的计费是否可信**。具体四问：

1. 预扣（下单时按 `max_tokens × 单价` 冻结的额度）在结算时**是否按实际用量退回**；
2. 上报的 `usage` **是否真实**，还是由 `max_tokens` 推导/夹取；
3. 计费的驱动因子是**下单上限**还是**真实生成量**；
4. 关掉 thinking **是否真能省钱**（成本可控性）。

明确划出范围外（用户约定：与官方文档不一致但不影响中转使用者，暂不管）：
文档承诺的 400（`temperature`/`top_p`、非法 role、缺 `max_tokens`、prefill、
非法 `effort` 等）、端点命名差异、字段归一化、错误文案措辞。这些只在
`observations` 里留痕，**不参与判定**。

设计要点：
- **余额 oracle 优先用官方计费端点**（New API 系 `/dashboard/billing/{subscription,usage}`，
  API-key 鉴权），拿不到才回退到「用一次必然被拒的超额请求，读错误文案里的
  `token remain quota`」——实测该类通道被 403 拒绝的请求**不扣款**，故该探针免费。
- **先标定单价与驱动因子，再谈用量真实性**：只有两件都测，才能区分
  「按上限扣款」与「按伪造的上报量扣款」。
- 判定分四档 passed / failed / warning / blocked；**无可用余额 oracle 时，
  计费类用例一律 blocked 而不是 failed**——没取证能力不等于通道有问题。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.finalizer import finalize_result_dir  # noqa: E402
from common.test_framework import ExecutionRecord, HttpExecutor, ResultLog  # noqa: E402
from volcperf.protocols import api_root  # noqa: E402

ANTHROPIC_VERSION = "2023-06-01"

# 探针文案。短答案用于「上限因子」，长答案用于「内容因子」。长答案必须足够长，
# 才能在**按用量计费**的通道上让余额差分大过读数分辨率（否则计费口径无法分辨）。
SHORT_PROMPT = "只回复 OK，不要输出任何其他内容。"
LONG_PROMPT = "用空格分隔依次输出数字 1 到 300，不要输出任何其他文字。"
PAD_UNIT = "这是一段用于撑大输入的填充文本，请忽略它的内容。"

# 各 oracle 的可分辨粒度（美元）。官方计费端点按「分」给累计值，超额探针文案给 6 位小数。
ORACLE_RESOLUTION = {"dashboard": 0.01, "403_remain": 0.000001}

# 预扣单价的来源口径（写进报告，避免把「默认值」误读成实测值）。
RATE_SOURCE_TEXT = {
    "default": "默认值 50，本次未取得可反推的单价观测",
    "configured": "调用方显式指定",
    "calibrated_from_403": "由 403 预扣文案反推标定（`need quota ÷ max_tokens`）",
}

# 反推单价的最小上限：`need quota` 只保留 6 位小数，上限太小则相对误差被放大，
# 不足以作为标定基准（例如上限 16 时 1e-6 美元已相当于 0.06 $/M 的量化噪声）。
MIN_CALIBRATION_MAX_TOKENS = 256

# 瞬时故障口径（对齐平台纪律：先重试、再收口，绝不记 failed）。
# 依据 2026-10-09 claude-sonnet-5 批次：11:17:42 起 12 连 502（`server: Caddy`、body 空），
# 一次抖动就把 4 条计费用例全部打成"无可用余额读数"。
TRANSIENT_STATUSES = frozenset({500, 502, 503, 504, 520, 521, 522, 523, 524})


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class ProbeConfig:
    base_url: str
    api_key: str
    model: str
    protocol: str = "anthropic"
    rate_per_m: float = 50.0          # 预扣单价 $ / 1M token（未显式给定时由 403 文案自动标定）
    rate_explicit: bool = False       # 用户是否显式指定单价；显式指定时自动标定只留痕、不覆盖
    rate_source: str = "default"      # default | configured | calibrated_from_403
    rate_samples: tuple[float, ...] = ()   # 403 文案反推的单价观测（用于一致性留痕）
    timeout: float = 120.0
    verify_ssl: bool = True
    trust_env: bool = False
    ladder: tuple[int, ...] = (1024, 8192)        # (小上限, 大上限)
    balance_probe_mt: int = 2048                  # 403 余额探针的下单上限
    escalate_balance_probe: bool = False          # 探针 200 时是否继续放大（会产生真实消耗）
    balance_probe_ceiling: int = 524288
    release_delays: tuple[int, ...] = (0, 3, 8)   # 结算后回读余额的时刻（秒）
    resolution: float = 0.005                     # 余额读数分辨率（美元）
    probes: tuple[str, ...] = ()                  # 实际执行的探针（空 = 全部）
    max_retries: int = 2                          # 5xx/连接错误的额外重试次数（瞬时故障）
    retry_backoff: float = 1.0                    # 重试退避基数（秒），第 n 次等 n×该值

    @property
    def root(self) -> str:
        return api_root(self.protocol, self.base_url)

    def reserve(self, max_tokens: int) -> float:
        """按上限冻结的额度（美元）。"""
        return max_tokens * self.rate_per_m / 1_000_000


# --------------------------------------------------------------------------- #
# 结果模型
# --------------------------------------------------------------------------- #
@dataclass
class CaseResult:
    case_id: str
    name: str
    category: str
    status: str
    passed: bool
    details: str
    observations: dict[str, Any] = field(default_factory=dict)


@dataclass
class Norm:
    """归一化后的响应（anthropic / openai 两种面统一到同一套字段）。"""

    tag: str
    status: int
    text: str = ""
    in_tokens: float | None = None
    out_tokens: float | None = None
    think_tokens: float | None = None
    stop_reason: str | None = None
    max_tokens: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    error_text: str = ""

    @property
    def ok(self) -> bool:
        return self.status == 200


@dataclass
class Reading:
    """一次余额读数。`spent` 为累计消耗（美元），`balance` 为剩余额度（美元）。"""

    kind: str | None
    spent: float | None = None
    balance: float | None = None
    raw: str = ""


# --------------------------------------------------------------------------- #
# 探针清单（顺序 = 执行顺序；U03 汇总分析放在最后）
# --------------------------------------------------------------------------- #
PROBES: tuple[tuple[str, str, str, str], ...] = (
    ("BILLING-S01", "通道可用性与模型目录", "基线", "case_channel"),
    ("BILLING-U01", "usage 是否随 max_tokens 变化（上限因子）", "用量真实性", "case_usage_cap_factor"),
    ("BILLING-U02", "usage 是否随真实输出长度变化（内容因子）", "用量真实性", "case_usage_content_factor"),
    ("BILLING-U04", "input_tokens 是否随输入增长", "用量真实性", "case_input_ladder"),
    ("BILLING-U05", "同请求重复调用 usage 是否稳定", "用量真实性", "case_usage_repeatability"),
    ("BILLING-U06", "关思考是否降低用量（成本可控性）", "成本可控", "case_thinking_off"),
    ("BILLING-B01", "计费驱动因子标定（上限因子 vs 内容因子）", "计费口径", "case_billing_factors"),
    ("BILLING-B02", "预扣结算口径与释放时延（是否退回）", "计费口径", "case_precharge_settlement"),
    ("BILLING-B03", "同请求计费可重复性", "计费口径", "case_billing_repeatability"),
    ("BILLING-B04", "额度不足时的门槛与小额请求可用性", "计费口径", "case_quota_floor"),
    ("BILLING-U03", "思考余量指纹（output−thinking 与正文矛盾）", "用量真实性", "case_thinking_residual"),
)

# U03 是**对样本的汇总分析**（读 pool 里已成功的响应），本身不下单。
# 单独选中它时必须补上样本来源，否则只会得到「样本不足」的 warning。
PROBE_DEPS: dict[str, tuple[str, ...]] = {
    "BILLING-U03": ("BILLING-U01", "BILLING-U02"),
}
PROBE_ORDER: tuple[str, ...] = tuple(case_id for case_id, *_ in PROBES)


def expand_probe_selection(selected: Iterable[str]) -> tuple[str, ...]:
    """把用户选择的探针展开成实际执行清单（含依赖前置，按清单顺序稳定排序）。

    空选择表示执行全部探针；未知 id 直接报错，避免静默跑成空集后
    被误读为「无问题」。
    """
    wanted_items = [str(item).strip() for item in selected if str(item).strip()]
    if not wanted_items:
        return PROBE_ORDER
    unknown = sorted({item for item in wanted_items if item not in PROBE_ORDER})
    if unknown:
        raise ValueError(f"未知探针: {', '.join(unknown)}")
    wanted: set[str] = set()
    stack = list(wanted_items)
    while stack:
        item = stack.pop()
        if item in wanted:
            continue
        wanted.add(item)
        stack.extend(PROBE_DEPS.get(item, ()))
    return tuple(case_id for case_id in PROBE_ORDER if case_id in wanted)


# --------------------------------------------------------------------------- #
# 运行器
# --------------------------------------------------------------------------- #
class BillingProbe:
    def __init__(self, cfg: ProbeConfig, executor: HttpExecutor, log: ResultLog):
        self.cfg = cfg
        self.executor = executor
        self.log = log
        self.pool: list[Norm] = []
        self.oracle_kind: str | None = None
        self._spent_anchor: float | None = None
        self.notes: list[str] = []
        self.transient_failures: list[str] = []   # 已重试的瞬时故障（5xx/连接错误）
        self.read_errors: list[str] = []          # 本轮余额读数失败原因（dashboard / 超额探针各记一条）
        self._rate_best: tuple[int, float] | None = None   # (max_tokens, 反推单价)，取精度最高者
        self._rate_wobble_noted = False
        self._rate_mismatch_noted = False
        # 读数分辨率随 oracle 变化：官方计费端点只到「分」，超额探针文案保留 6 位小数。
        # 不区分会把精确 oracle 白白限制在 $0.01 上，小上限档的扣款将无法分辨。
        self.resolution = cfg.resolution

    # ---------------- 传输 ---------------- #
    def _headers(self, tag: str) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": f"{tag}-{uuid.uuid4().hex[:10]}",
        }
        if self.cfg.protocol == "anthropic":
            headers["x-api-key"] = self.cfg.api_key
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
            headers["anthropic-version"] = ANTHROPIC_VERSION
        else:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        return headers

    def _message_url(self) -> str:
        return f"{self.cfg.root}/messages" if self.cfg.protocol == "anthropic" else f"{self.cfg.root}/chat/completions"

    def _payload(self, max_tokens: int, prompt: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if extra:
            payload.update(extra)
        return payload

    def _normalize(self, rec: ExecutionRecord, tag: str, max_tokens: int) -> Norm:
        body = rec.response_body if isinstance(rec.response_body, dict) else {}
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        norm = Norm(tag=tag, status=rec.http_status or 0, max_tokens=max_tokens, usage=dict(usage))
        if self.cfg.protocol == "anthropic":
            blocks = body.get("content") if isinstance(body.get("content"), list) else []
            norm.text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
            norm.in_tokens = usage.get("input_tokens")
            norm.out_tokens = usage.get("output_tokens")
            details = usage.get("output_tokens_details")
            if isinstance(details, dict):
                norm.think_tokens = details.get("thinking_tokens")
            norm.stop_reason = body.get("stop_reason")
        else:
            choices = body.get("choices") if isinstance(body.get("choices"), list) else []
            first = choices[0] if choices and isinstance(choices[0], dict) else {}
            message = first.get("message") if isinstance(first.get("message"), dict) else {}
            norm.text = message.get("content") or (first.get("delta") or {}).get("content") or ""
            if not isinstance(norm.text, str):
                norm.text = ""
            norm.in_tokens = usage.get("prompt_tokens")
            norm.out_tokens = usage.get("completion_tokens")
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict):
                norm.think_tokens = details.get("reasoning_tokens")
            norm.stop_reason = first.get("finish_reason")
        err = body.get("error") if isinstance(body.get("error"), dict) else {}
        norm.error_text = f"{err.get('type') or ''} {err.get('code') or ''} {err.get('message') or ''}".strip()
        if not norm.error_text:
            norm.error_text = (rec.response_text or "")[:400]
        if norm.ok:
            self.pool.append(norm)
        return norm

    @staticmethod
    def _is_transient(rec: ExecutionRecord) -> bool:
        """瞬时故障判定：连接错误、或 5xx（结构化 `model_not_found` 例外——那是确定性结论）。"""
        if (rec.transport_status or "") == "error":
            return True
        if (rec.http_status or 0) not in TRANSIENT_STATUSES:
            return False
        body = rec.response_body if isinstance(rec.response_body, dict) else {}
        blob = json.dumps(body, ensure_ascii=False)[:400]
        return "model_not_found" not in blob

    @staticmethod
    def _failure_label(rec: ExecutionRecord) -> str:
        if (rec.transport_status or "") == "error":
            err = rec.transport_error or {}
            return f"连接错误 {err.get('type') or ''}".strip()
        return f"HTTP {rec.http_status}"

    async def _with_retry(self, label: str, factory):
        """5xx/连接错误先重试，重试耗尽才把最后一次结果交回。

        与项目纪律一致：瞬时故障**不得**被写成 failed；这里额外把故障记进
        `transient_failures`，让 blocked 文案能自解释"是网关抖动，不是通道没有计费端点"。
        """
        attempt = 0
        while True:
            rec = await factory()
            if not self._is_transient(rec) or attempt >= self.cfg.max_retries:
                return rec
            self.transient_failures.append(f"{label} {self._failure_label(rec)}")
            attempt += 1
            await asyncio.sleep(self.cfg.retry_backoff * attempt)

    def _oracle_blind_suffix(self) -> str:
        """余额读数不可得时给 blocked/warning 文案补上真实失败原因。"""
        parts: list[str] = []
        if self.read_errors:
            uniq = list(dict.fromkeys(self.read_errors))
            parts.append("读数失败原因：" + "；".join(uniq[:3]))
        if self.transient_failures:
            uniq = list(dict.fromkeys(self.transient_failures))
            parts.append(f"网关 5xx/连接错误共 {len(self.transient_failures)} 次"
                         f"（已各重试 {self.cfg.max_retries} 次），示例：{'、'.join(uniq[:3])}")
        return ("；" + "；".join(parts)) if parts else ""

    async def send(self, tag: str, max_tokens: int, prompt: str, extra: dict[str, Any] | None = None) -> Norm:
        rec = await self._with_retry(tag, lambda: self.executor.execute(
            case_id=tag,
            url=self._message_url(),
            payload=self._payload(max_tokens, prompt, extra),
            headers=self._headers(tag),
            stream=False,
        ))
        await self.log.write(rec)
        norm = self._normalize(rec, tag, max_tokens)
        # 被拒的请求里若有 `need quota`，顺手反推单价——这一程本就要走，标定不额外花钱。
        if norm.status in (402, 403):
            self._calibrate_rate(self._quota_amount(norm.error_text, "need quota"), max_tokens)
        return norm

    # ---------------- 余额 oracle ---------------- #
    @staticmethod
    def _as_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    async def _dashboard_reading(self, tag: str) -> Reading | None:
        """New API 系标准计费面：subscription(上限) + usage(累计消耗，单位美分)。"""
        headers = self._headers(tag)
        sub_rec = await self._with_retry(f"{tag}-subscription", lambda: self.executor.execute_raw(
            case_id=f"{tag}-subscription", method="GET",
            url=f"{self.cfg.root}/dashboard/billing/subscription", headers=headers,
        ))
        await self.log.write(sub_rec)
        if (sub_rec.http_status or 0) != 200 or not isinstance(sub_rec.response_body, dict):
            self.read_errors.append(f"GET /dashboard/billing/subscription {self._failure_label(sub_rec)}")
            return None
        limit = self._as_float(sub_rec.response_body.get("hard_limit_usd"))

        today = time.strftime("%Y-%m-%d")
        usage_rec = await self._with_retry(f"{tag}-usage", lambda: self.executor.execute_raw(
            case_id=f"{tag}-usage", method="GET",
            url=f"{self.cfg.root}/dashboard/billing/usage", headers=headers,
            params={"start_date": today, "end_date": today},
        ))
        await self.log.write(usage_rec)
        total = None
        if (usage_rec.http_status or 0) == 200 and isinstance(usage_rec.response_body, dict):
            cents = self._as_float(usage_rec.response_body.get("total_usage"))
            total = None if cents is None else cents / 100.0
        if total is None:
            self.read_errors.append(f"GET /dashboard/billing/usage {self._failure_label(usage_rec)}"
                                    f"（未取到 total_usage）")
            return None
        balance = None if limit is None else limit - total
        return Reading(kind="dashboard", spent=total, balance=balance,
                       raw=f"hard_limit_usd={limit} total_usage_usd={total}")

    @staticmethod
    def _quota_amount(text: str, label: str) -> float | None:
        """从 403 文案里取 `remain quota` / `need quota` 的金额（美元）。

        原文形如 `token remain quota: $0.913022, need quota: $1.600080`；`need quota`
        恒等于 `max_tokens × 输出单价`，所以它能反推出**该模型在该通道上的真实预扣单价**，
        无需人工按模型填 `--rate-per-m`。
        """
        match = re.search(rf"{re.escape(label)}\s*[:：]\s*[$＄]?\s*([0-9]+(?:\.[0-9]+)?)", text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _calibrate_rate(self, need_usd: float | None, max_tokens: int) -> None:
        """用 403 预扣文案反标定单价：`need quota ÷ max_tokens × 1e6`。

        `need quota` 只保留 6 位小数，上限越小相对误差越大，故始终采用**上限最大**的那次
        观测（精度最高），并对其余观测做一致性留痕。显式指定的单价不被覆盖，只留 note。
        """
        if need_usd is None or need_usd <= 0 or max_tokens < MIN_CALIBRATION_MAX_TOKENS:
            return
        rate = need_usd * 1_000_000 / max_tokens
        self.cfg.rate_samples = (*self.cfg.rate_samples, rate)
        if self._rate_best is None:
            self._rate_best = (max_tokens, rate)
        elif max_tokens > self._rate_best[0]:
            previous_mt, previous_rate = self._rate_best
            self._rate_best = (max_tokens, rate)
            if abs(rate - previous_rate) > max(0.01 * previous_rate, 1e-6) and not self._rate_wobble_noted:
                self._rate_wobble_noted = True
                self.notes.append(
                    f"通道预扣单价随上限变化：上限 {previous_mt} 反推 ${previous_rate:.4f}/M，"
                    f"上限 {max_tokens} 反推 ${rate:.4f}/M（已采用后者）")
        anchor_mt, anchor_rate = self._rate_best
        if self.cfg.rate_explicit:
            if abs(anchor_rate - self.cfg.rate_per_m) > max(0.01 * self.cfg.rate_per_m, 1e-6) \
                    and not self._rate_mismatch_noted:
                self._rate_mismatch_noted = True
                self.notes.append(
                    f"显式单价 ${self.cfg.rate_per_m:.4f}/M 与 403 文案反推的 ${anchor_rate:.4f}/M"
                    f"（上限 {anchor_mt}）不符；按显式值计，B02/B04 的冻结额换算可能偏离")
            return
        self.cfg.rate_per_m = anchor_rate
        self.cfg.rate_source = "calibrated_from_403"

    async def _probe_reading(self, tag: str, max_tokens: int) -> Reading | None:
        """回退 oracle：故意超额下单，读错误文案里的剩余额度（被拒请求不扣款）。"""
        norm = await self.send(f"{tag}-balance", max_tokens, SHORT_PROMPT)
        if norm.status != 403:
            if self._transient_norm(norm):
                self.read_errors.append(f"超额探针（max_tokens={max_tokens}）HTTP {norm.status}"
                                        f"（网关/连接级故障，已重试 {self.cfg.max_retries} 次）")
            else:
                self.read_errors.append(f"超额探针（max_tokens={max_tokens}）HTTP {norm.status}"
                                        f"：未被拒，文案里没有 remain quota")
            return None
        balance = self._quota_amount(norm.error_text, "remain quota")
        if balance is None:
            self.read_errors.append("403 文案未含可解析的 remain quota")
            return None
        if self._spent_anchor is None:
            self._spent_anchor = balance
        return Reading(kind="403_remain", spent=self._spent_anchor - balance, balance=balance,
                       raw=norm.error_text[:220])

    async def _read(self, tag: str) -> Reading:
        self.read_errors = []
        if self.oracle_kind in (None, "dashboard"):
            found = await self._dashboard_reading(tag)
            if found is not None:
                self.oracle_kind = "dashboard"
                self.resolution = ORACLE_RESOLUTION["dashboard"]
                return found
        probe_mt = self.cfg.balance_probe_mt
        while True:
            found = await self._probe_reading(tag, probe_mt)
            if found is not None:
                self.oracle_kind = "403_remain"
                self.resolution = ORACLE_RESOLUTION["403_remain"]
                return found
            if not self.cfg.escalate_balance_probe or probe_mt >= self.cfg.balance_probe_ceiling:
                return Reading(kind=None, raw="；".join(self.read_errors) or "余额 oracle 不可用")
            probe_mt = min(probe_mt * 4, self.cfg.balance_probe_ceiling)

    def _delta(self, before: Reading, after: Reading) -> float | None:
        if before.kind is None or after.kind is None or before.kind != after.kind:
            return None
        if before.spent is not None and after.spent is not None:
            return after.spent - before.spent
        if before.balance is not None and after.balance is not None:
            return before.balance - after.balance
        return None

    async def _measure(self, tag: str, max_tokens: int, prompt: str,
                       extra: dict[str, Any] | None = None) -> tuple[Norm, float | None, Reading, Reading] | None:
        before = await self._read(f"{tag}-pre")
        if before.kind is None:
            return None
        norm = await self.send(tag, max_tokens, prompt, extra)
        after = await self._read(f"{tag}-post")
        return norm, self._delta(before, after), before, after

    # ---------------- 通用判定助手 ---------------- #
    def _blocked(self, cid: str, name: str, category: str, why: str, obs: dict[str, Any] | None = None) -> CaseResult:
        return CaseResult(cid, name, category, "blocked", False, why, obs or {})

    @staticmethod
    def _err(norm: Norm) -> str:
        return f"HTTP {norm.status}：{norm.error_text[:200]}"

    def _gate(self, norm: Norm, cid: str, name: str, category: str, what: str) -> CaseResult | None:
        """请求未成功时的统一收口：不可用/无权限 → blocked；其它 → failed。"""
        if norm.ok:
            return None
        blocked_markers = ("quota", "无权访问", "permission", "access denied", "forbidden", "not enabled",
                           "not available", "balance", "insufficient", "unauthorized", "authentication")
        text = self._err(norm).lower()
        if norm.status in (401, 403, 429) or any(m in text for m in blocked_markers):
            why = (f"{what}未取得有效响应，属平台级前提不满足：{self._err(norm)}；"
                   f"该用例的目标行为从未被触达，按 blocked 收口，不计入通道计费结论")
            return self._blocked(cid, name, category, why, {"blocked_response": norm.error_text[:300]})
        if not norm.ok and self._transient_norm(norm):
            # 瞬时故障（已重试 `max_retries` 次仍失败）**不得**记 failed：
            # 与项目纪律一致，否则网关抖动会污染"通道计费"结论。
            return self._blocked(cid, name, category,
                                 f"{what}未取得有效响应：{self._err(norm)}；已重试 {self.cfg.max_retries} 次仍为"
                                 f"网关/连接级故障（瞬时），按 blocked 收口，不计入通道计费结论",
                                 {"blocked_response": norm.error_text[:300],
                                  "transient_failures": self.transient_failures[-3:]})
        return CaseResult(cid, name, category, "failed", False, f"{what}失败：{self._err(norm)}",
                          {"response": norm.error_text[:300]})

    @staticmethod
    def _transient_norm(norm: Norm) -> bool:
        """Norm 侧的瞬时故障判定（结构化 `model_not_found` 例外）。"""
        if norm.status == 0:
            return True
        if norm.status not in TRANSIENT_STATUSES:
            return False
        return "model_not_found" not in norm.error_text.lower()

    # ---------------- 探针实现 ---------------- #
    async def case_channel(self, cid: str, name: str, category: str) -> CaseResult:
        tag = cid
        rec = await self._with_retry(tag, lambda: self.executor.execute_raw(
            case_id=tag, method="GET", url=f"{self.cfg.root}/models", headers=self._headers(tag),
        ))
        await self.log.write(rec)
        status = rec.http_status or 0
        body = rec.response_body if isinstance(rec.response_body, dict) else {}
        models = body.get("data") if isinstance(body.get("data"), list) else []
        ids = [str(m.get("id")) for m in models if isinstance(m, dict) and m.get("id")]
        obs = {"http_status": status, "model_count": len(ids), "model_present": self.cfg.model in ids,
               "models_sample": ids[:12], "oracle_preference": "dashboard_billing → 403_remain"}
        if status != 200:
            return self._blocked(cid, name, category,
                                 f"模型目录未取到（{self._failure_label(rec)}）：{str(rec.response_text)[:160]}"
                                 f"；通道不可用时不产生计费结论" + self._oracle_blind_suffix(), obs)
        if ids and self.cfg.model not in ids:
            return CaseResult(cid, name, category, "warning", True,
                              f"模型目录可达，但目标模型 {self.cfg.model!r} 未在 {len(ids)} 个可见模型内（可能是分组权限差异）", obs)
        return CaseResult(cid, name, category, "passed", True,
                          f"通道可达，模型目录 {len(ids)} 项，目标模型{'在列' if self.cfg.model in ids else '不在列（目录为空，跳过比对）'}", obs)

    async def case_usage_cap_factor(self, cid: str, name: str, category: str) -> CaseResult:
        ladder = (64, 256, 1024)
        rows: list[dict[str, Any]] = []
        for mt in ladder:
            outs, thinks, stops = [], [], []
            for _ in range(2):
                norm = await self.send(f"{cid}-mt{mt}", mt, SHORT_PROMPT)
                if not norm.ok:
                    gate = self._gate(norm, cid, name, category, "上限因子对照")
                    return gate if gate else CaseResult(cid, name, category, "failed", False, self._err(norm))
                outs.append(norm.out_tokens)
                thinks.append(norm.think_tokens)
                stops.append(norm.stop_reason)
            rows.append({"max_tokens": mt, "output_tokens": outs, "thinking_tokens": thinks, "stop_reason": stops})
        obs = {"ladder": rows, "rate_per_m": self.cfg.rate_per_m}
        hard, truncated, moved = [], [], []
        for row in rows:
            outs = [v for v in row["output_tokens"] if v is not None]
            if not outs:
                continue
            value = max(outs)
            if value == row["max_tokens"]:
                (truncated if "max_tokens" in row["stop_reason"] else hard).append(row["max_tokens"])
            moved.append(value)
        if len(hard) >= 2:
            return CaseResult(cid, name, category, "failed", False,
                              f"**计费诚实红线**：同一提示仅改上限，{len(hard)} 个档位的 output_tokens 精确等于 max_tokens"
                              f"（{hard}）且 stop_reason 自称自然结束 → `usage` 由 max_tokens 推导/夹取，不是实测值", obs)
        if len(hard) == 1:
            return CaseResult(cid, name, category, "warning", True,
                              f"有 1 个档位（{hard[0]}）output_tokens 贴上限但自称自然结束，其余档位正常：疑似夹取，证据不足", obs)
        if len(truncated) >= 2:
            return CaseResult(cid, name, category, "warning", True,
                              f"{len(truncated)} 个档位（{truncated}）被上限截断且如实回显 stop_reason=max_tokens："
                              f"用量未必不实，但小上限会被思考吃满，成本不可控", obs)
        if len(moved) >= 2 and min(moved) > 0 and max(moved) / min(moved) >= 3:
            return CaseResult(cid, name, category, "warning", True,
                              f"output_tokens 随上限同向变化（{moved}）但未贴上上限：用量口径可疑，尚不构成红线", obs)
        return CaseResult(cid, name, category, "passed", True,
                          f"同一提示改动上限 {list(ladder)}，output_tokens 未随上限变化（{moved}）→ 用量与上限解耦", obs)

    async def case_usage_content_factor(self, cid: str, name: str, category: str) -> CaseResult:
        mt = self.cfg.ladder[0]
        short = await self.send(f"{cid}-short", mt, SHORT_PROMPT)
        if not short.ok:
            gate = self._gate(short, cid, name, category, "内容因子对照（短答案）")
            return gate if gate else CaseResult(cid, name, category, "failed", False, self._err(short))
        long_norm = await self.send(f"{cid}-long", mt, LONG_PROMPT)
        if not long_norm.ok:
            gate = self._gate(long_norm, cid, name, category, "内容因子对照（长答案）")
            return gate if gate else CaseResult(cid, name, category, "failed", False, self._err(long_norm))
        items = len(re.findall(r"\d+", long_norm.text))
        obs = {"max_tokens": mt, "short": {"chars": len(short.text), "output_tokens": short.out_tokens,
                                           "stop_reason": short.stop_reason, "text": short.text[:60]},
               "long": {"chars": len(long_norm.text), "items": items, "output_tokens": long_norm.out_tokens,
                        "stop_reason": long_norm.stop_reason, "text": long_norm.text[:80]}}
        if short.out_tokens is None or long_norm.out_tokens is None:
            return self._blocked(cid, name, category, "响应未提供 output_tokens，无法比较用量口径", obs)
        if len(long_norm.text) < 4 * len(short.text):
            return CaseResult(cid, name, category, "warning", True,
                              f"长答案未被真正展开（{len(long_norm.text)} 字符 vs 短答案 {len(short.text)} 字符），本轮对照不成立", obs)
        if long_norm.out_tokens == short.out_tokens:
            return CaseResult(cid, name, category, "failed", False,
                              f"同一上限下，正文长度相差 {len(long_norm.text)}/{len(short.text)} 字符（{items} 个条目），"
                              f"output_tokens 却完全相同（{short.out_tokens}）→ 用量与真实输出长度无关，不是实测值", obs)
        # 长答案若被上限截断，条目数下界就不再成立（正文本身不完整），不能用它反证。
        if items and long_norm.out_tokens < items and long_norm.stop_reason != "max_tokens":
            return CaseResult(cid, name, category, "failed", False,
                              f"账目小于可见内容下界：正文含 {items} 个空格分隔条目（每条目至少 1 token），"
                              f"而 output_tokens 仅报 {long_norm.out_tokens} → 纯算术反证", obs)
        if long_norm.out_tokens == mt and long_norm.stop_reason != "max_tokens":
            return CaseResult(cid, name, category, "failed", False,
                              f"矛盾组合：output_tokens 贴上上限 {mt} 却自称 {long_norm.stop_reason}（非 max_tokens）", obs)
        return CaseResult(cid, name, category, "passed", True,
                          f"同上限下长答案 output_tokens（{long_norm.out_tokens}）显著高于短答案（{short.out_tokens}）"
                          f"→ 用量随真实输出长度变化", obs)

    async def case_input_ladder(self, cid: str, name: str, category: str) -> CaseResult:
        small = await self.send(f"{cid}-in1", 64, SHORT_PROMPT)
        if not small.ok:
            gate = self._gate(small, cid, name, category, "输入阶梯（1×）")
            return gate if gate else CaseResult(cid, name, category, "failed", False, self._err(small))
        padded = SHORT_PROMPT + "\n" + PAD_UNIT * 160
        big = await self.send(f"{cid}-in4", 64, padded)
        if not big.ok:
            gate = self._gate(big, cid, name, category, "输入阶梯（4×）")
            return gate if gate else CaseResult(cid, name, category, "failed", False, self._err(big))
        obs = {"baseline_input_tokens": small.in_tokens, "padded_input_tokens": big.in_tokens,
               "pad_chars": len(PAD_UNIT) * 160}
        if small.in_tokens is None or big.in_tokens is None:
            return self._blocked(cid, name, category, "响应未提供 input_tokens，无法验证其是否随输入变化", obs)
        if big.in_tokens <= small.in_tokens * 1.5:
            return CaseResult(cid, name, category, "failed", False,
                              f"输入侧账目与输入长度脱钩：基线 input_tokens={small.in_tokens}，"
                              f"追加 {len(PAD_UNIT) * 160} 字符后仍为 {big.in_tokens} → usage 整体不是实测值", obs)
        return CaseResult(cid, name, category, "passed", True,
                          f"input_tokens 随输入增长（{small.in_tokens} → {big.in_tokens}）", obs)

    async def case_usage_repeatability(self, cid: str, name: str, category: str) -> CaseResult:
        rows = []
        for index in range(3):
            norm = await self.send(f"{cid}-rep{index + 1}", 256, SHORT_PROMPT)
            if not norm.ok:
                gate = self._gate(norm, cid, name, category, "用量重复性")
                return gate if gate else CaseResult(cid, name, category, "failed", False, self._err(norm))
            rows.append({"output_tokens": norm.out_tokens, "thinking_tokens": norm.think_tokens,
                         "stop_reason": norm.stop_reason})
        outs = [r["output_tokens"] for r in rows if r["output_tokens"] is not None]
        think = [r["thinking_tokens"] for r in rows if r["thinking_tokens"] is not None]
        obs = {"repeats": rows}
        if not outs:
            return self._blocked(cid, name, category, "响应未提供 output_tokens，无法比较重复性", obs)
        spread = max(outs) - min(outs)
        obs["output_tokens_spread"] = spread
        if spread > max(8, 0.35 * max(outs)):
            return CaseResult(cid, name, category, "failed", False,
                              f"同一提示、同一上限重复 3 次，output_tokens 波动 {min(outs)}–{max(outs)}"
                              f"（思考 {think}）→ 用量报告不可预期", obs)
        if len(set(outs)) > 1:
            return CaseResult(cid, name, category, "warning", True,
                              f"重复 3 次 output_tokens 有轻微波动 {outs}，尚在容差内", obs)
        return CaseResult(cid, name, category, "passed", True, f"重复 3 次 output_tokens 完全一致（{outs[0]}）", obs)

    async def case_thinking_off(self, cid: str, name: str, category: str) -> CaseResult:
        if self.cfg.protocol != "anthropic":
            return CaseResult(cid, name, category, "warning", True,
                              "当前协议面无 thinking 开关参数，成本可控性不可测（留痕，不计入结论）", {"protocol": self.cfg.protocol})
        mt = 1024
        default = await self.send(f"{cid}-default", mt, SHORT_PROMPT)
        if not default.ok:
            gate = self._gate(default, cid, name, category, "默认思考请求")
            return gate if gate else CaseResult(cid, name, category, "failed", False, self._err(default))
        off = await self.send(f"{cid}-off", mt, SHORT_PROMPT, {"thinking": {"type": "disabled"}})
        obs = {"max_tokens": mt,
               "default": {"output_tokens": default.out_tokens, "thinking_tokens": default.think_tokens},
               "disabled": {"http_status": off.status, "output_tokens": off.out_tokens,
                            "thinking_tokens": off.think_tokens, "stop_reason": off.stop_reason},
               "note": "只有 thinking_tokens > 0 才算「真的在思考」；字段存在但值为 0 不等于带思考计费"}
        if not off.ok:
            return self._blocked(cid, name, category,
                                 f"显式关闭思考被拒（HTTP {off.status}）——无法评估关思考能否省钱：{off.error_text[:160]}", obs)
        # 依据 2026-10-09 sonnet-5 批次：`output_tokens_details.thinking_tokens` 的 **key 存在、值恒为 0**，
        # 原判据用 `"thinking" in usage` 会把它读成"仍带思考计费"。只认真实用量。
        think_default = bool(default.think_tokens)
        think_disabled = bool(off.think_tokens)
        if not think_default and not think_disabled:
            return CaseResult(cid, name, category, "warning", True,
                              f"两档 `thinking_tokens` 均为 0（默认未传 thinking 时思考就没产生用量，"
                              f"显式 disabled 后同为 0）→ 本项无法回答「关思考能否省钱」"
                              f"（思考在该通道上是 opt-in，留痕，不计入结论）", obs)
        if think_disabled and default.out_tokens is not None and off.out_tokens is not None \
                and off.out_tokens >= default.out_tokens * 0.95:
            return CaseResult(cid, name, category, "warning", True,
                              f"显式 thinking=disabled 后用量未下降（{default.out_tokens} → {off.out_tokens}）"
                              f"且响应仍带思考用量（thinking_tokens={off.think_tokens}）"
                              f"→ 思考不可关闭，成本不可控（计费不实之外的独立成本风险）", obs)
        if default.out_tokens is not None and off.out_tokens is not None \
                and off.out_tokens >= default.out_tokens * 0.95:
            return CaseResult(cid, name, category, "warning", True,
                              f"关闭思考后响应不再带思考用量，但总 output_tokens 未下降"
                              f"（{default.out_tokens} → {off.out_tokens}）→ 成本口径需向上游确认", obs)
        return CaseResult(cid, name, category, "passed", True,
                          f"关闭思考后用量下降（{default.out_tokens} → {off.out_tokens}）", obs)

    async def case_billing_factors(self, cid: str, name: str, category: str) -> CaseResult:
        """2 因子对照：上限因子（同内容不同上限）× 内容因子（同上限不同内容）。

        三个点必须**同内容比上限、同上限比内容**，否则两个因子会互相污染：
          A = (小上限, 长答案)  B = (大上限, 长答案)  C = (大上限, 短答案)
          A→B 只改上限，C→B 只改内容。
        """
        small_mt, big_mt = self.cfg.ladder
        first = await self._measure(f"{cid}-smallcap-long", small_mt, LONG_PROMPT)
        if first is None:
            return self._blocked(cid, name, category,
                                 "无可用余额读数（计费端点不可用且超额探针未被拒）→ 计费口径无法取证，"
                                 "该结论不应在无计量口径时给出"
                                 + self._oracle_blind_suffix(),
                                 {"oracle": self.oracle_kind})
        norm_s, delta_s, _, after_s = first
        if not norm_s.ok:
            gate = self._gate(norm_s, cid, name, category, "计费标定（小上限/长答案）")
            return gate if gate else CaseResult(cid, name, category, "failed", False, self._err(norm_s))
        second = await self._measure(f"{cid}-bigcap-long", big_mt, LONG_PROMPT)
        if second is None:
            return self._blocked(cid, name, category,
                                 "第二次余额读数不可用，上限因子差分中断" + self._oracle_blind_suffix(),
                                 {"oracle": self.oracle_kind})
        norm_b, delta_b, _, after_b = second
        if not norm_b.ok:
            gate = self._gate(norm_b, cid, name, category, "计费标定（大上限/长答案）")
            return gate if gate else CaseResult(cid, name, category, "failed", False, self._err(norm_b))
        third = await self._measure(f"{cid}-bigcap-short", big_mt, SHORT_PROMPT)
        if third is None:
            return self._blocked(cid, name, category,
                                 "第三次余额读数不可用，内容因子差分中断" + self._oracle_blind_suffix(),
                                 {"oracle": self.oracle_kind})
        norm_c, delta_c, _, after_c = third

        obs = {
            "oracle": self.oracle_kind,
            "rate_per_m": self.cfg.rate_per_m,
            "A_small_cap_long_answer": {"max_tokens": small_mt, "charged_usd": delta_s,
                                        "reserve_usd": self.cfg.reserve(small_mt),
                                        "output_tokens": norm_s.out_tokens, "chars": len(norm_s.text),
                                        "balance_after": after_s.balance},
            "B_big_cap_long_answer": {"max_tokens": big_mt, "charged_usd": delta_b,
                                      "reserve_usd": self.cfg.reserve(big_mt),
                                      "output_tokens": norm_b.out_tokens, "chars": len(norm_b.text),
                                      "balance_after": after_b.balance},
            "C_big_cap_short_answer": {"max_tokens": big_mt, "charged_usd": delta_c,
                                       "output_tokens": norm_c.out_tokens, "chars": len(norm_c.text),
                                       "balance_after": after_c.balance},
        }
        if delta_s is None or delta_b is None:
            return self._blocked(cid, name, category, "余额读数不足两笔可比差分，计费口径无法判定", obs)
        if delta_s <= self.resolution:
            return CaseResult(cid, name, category, "warning", True,
                              f"余额读数分辨率（约 ${self.resolution:.3f}）不足以分辨小上限档的扣款"
                              f"（观测 ${delta_s:.4f}）→ 请调大 --billing-ladder 或使用更细的余额口径；本轮不判红线", obs)
        cap_ratio = delta_b / delta_s
        expected_cap_ratio = big_mt / small_mt
        content_delta = None if delta_c is None else delta_b - delta_c
        content_scale = max(abs(delta_b), abs(delta_c) if delta_c is not None else 0.0)
        obs["cap_ratio_observed"] = cap_ratio
        obs["cap_ratio_if_charged_by_max_tokens"] = expected_cap_ratio
        obs["content_factor_delta_usd"] = content_delta

        cap_driven = abs(cap_ratio - expected_cap_ratio) <= 0.3 * expected_cap_ratio
        content_driven = content_delta is not None and content_delta > max(0.3 * content_scale, self.resolution)
        if cap_driven and not content_driven:
            return CaseResult(cid, name, category, "failed", False,
                              f"**计费红线**：扣款只随下单上限走、不随答案长度走——同内容把上限从 {small_mt} 提到 {big_mt}"
                              f"（{expected_cap_ratio}×）扣款 ${delta_s:.4f}→${delta_b:.4f}（{cap_ratio:.1f}×），"
                              f"而同一上限下把答案从 {len(norm_b.text)} 字符换成 {len(norm_c.text)} 字符扣款几乎不变"
                              f"（${delta_b:.4f}→${delta_c:.4f}）→ **按下单上限实收，预扣未按实际用量退回**", obs)
        if content_driven and not cap_driven:
            return CaseResult(cid, name, category, "passed", True,
                              f"扣款随真实输出长度变化、不随下单上限变化（同上限下长答案多扣 ${content_delta:.4f}，"
                              f"上限 {expected_cap_ratio}× 却只让扣款变为 {cap_ratio:.2f}×）→ 计费按生成量而非上限", obs)
        if cap_driven and content_driven:
            return CaseResult(cid, name, category, "warning", True,
                              f"两个因子都在推动扣款（上限比 {cap_ratio:.2f}×、内容差 ${content_delta:.4f}）→ "
                              f"疑似混合计费（上限＋用量），口径需向上游确认", obs)
        return CaseResult(cid, name, category, "warning", True,
                          f"计费口径不明：上限因子 {cap_ratio:.2f}（预期 {expected_cap_ratio}）与内容因子 "
                          f"${content_delta} 均无法解释三笔扣款（${delta_s:.4f}/${delta_b:.4f}/"
                          f"{'—' if delta_c is None else f'${delta_c:.4f}'}）→ 计费模型不可预期", obs)

    async def case_precharge_settlement(self, cid: str, name: str, category: str) -> CaseResult:
        big_mt = self.cfg.ladder[1]
        before = await self._read(f"{cid}-pre")
        if before.kind is None:
            return self._blocked(cid, name, category, "无可用余额读数，预扣结算口径无法取证"
                                 + self._oracle_blind_suffix(), {"oracle": self.oracle_kind})
        norm = await self.send(cid, big_mt, SHORT_PROMPT)
        if not norm.ok:
            gate = self._gate(norm, cid, name, category, "预扣结算探测")
            return gate if gate else CaseResult(cid, name, category, "failed", False, self._err(norm))
        readings: list[dict[str, Any]] = []
        previous = before
        refunded_at = None
        for delay in self.cfg.release_delays:
            if delay:
                await asyncio.sleep(delay)
            reading = await self._read(f"{cid}-t{delay}")
            delta = self._delta(before, reading)
            step = None if previous.balance is None or reading.balance is None else reading.balance - previous.balance
            readings.append({"delay_seconds": delay, "charged_usd": delta, "balance": reading.balance,
                             "balance_step_usd": step})
            if step is not None and step > self.resolution and refunded_at is None:
                refunded_at = delay
            if reading.kind is not None:
                previous = reading
        obs = {"max_tokens": big_mt, "reserve_usd": self.cfg.reserve(big_mt), "oracle": self.oracle_kind,
               "output_tokens": norm.out_tokens, "chars": len(norm.text), "readings": readings,
               "refunded_at_seconds": refunded_at}
        final = [r["charged_usd"] for r in readings if r["charged_usd"] is not None]
        if not final:
            return self._blocked(cid, name, category, "结算后余额读数全部不可用，无法判定是否退回", obs)
        settled = final[-1]
        if refunded_at is not None:
            return CaseResult(cid, name, category, "passed", True,
                              f"结算后 {refunded_at}s 余额回弹 → 预扣先占用、结算按实际用量退回差额", obs)
        if abs(settled - self.cfg.reserve(big_mt)) <= max(0.15 * self.cfg.reserve(big_mt), self.resolution):
            return CaseResult(cid, name, category, "failed", False,
                              f"下单上限 {big_mt} 的冻结额度 ${self.cfg.reserve(big_mt):.4f} 在结算后 {self.cfg.release_delays[-1]}s "
                              f"仍全额扣除（实际扣 ${settled:.4f}，而答案仅 {len(norm.text)} 字符、"
                              f"output_tokens={norm.out_tokens}）→ **预扣按上限实收，未退回**", obs)
        return CaseResult(cid, name, category, "passed", True,
                          f"结算扣款 ${settled:.4f} 与上限冻结 ${self.cfg.reserve(big_mt):.4f} 不符，"
                          f"更接近按实际生成量结算", obs)

    async def case_billing_repeatability(self, cid: str, name: str, category: str) -> CaseResult:
        mt = self.cfg.ladder[0]
        charges: list[float | None] = []
        repeats = []
        for index in range(3):
            measured = await self._measure(f"{cid}-r{index + 1}", mt, SHORT_PROMPT)
            if measured is None:
                return self._blocked(cid, name, category, "无可用余额读数，计费可重复性无法取证"
                                     + self._oracle_blind_suffix(), {"oracle": self.oracle_kind})
            norm, delta, _, _ = measured
            if not norm.ok:
                gate = self._gate(norm, cid, name, category, "计费重复性")
                return gate if gate else CaseResult(cid, name, category, "failed", False, self._err(norm))
            charges.append(delta)
            repeats.append({"charged_usd": delta, "output_tokens": norm.out_tokens})
        obs = {"max_tokens": mt, "repeats": repeats}
        known = [c for c in charges if c is not None]
        if len(known) < 2:
            return self._blocked(cid, name, category, "可比扣款次数不足，计费可重复性无法判定", obs)
        spread = max(known) - min(known)
        mean = sum(known) / len(known)
        obs["spread_usd"] = spread
        obs["mean_usd"] = mean
        if spread > max(0.35 * abs(mean), self.resolution):
            return CaseResult(cid, name, category, "failed", False,
                              f"完全相同请求重复 3 次的扣款为 {[f'${c:.4f}' for c in known]}（极差 ${spread:.4f}）"
                              f"→ 计费不可预期，无法做成本核算", obs)
        return CaseResult(cid, name, category, "passed", True,
                          f"完全相同请求重复 3 次扣款一致（{known[0]:.4f}）→ 计费可预期", obs)

    async def case_quota_floor(self, cid: str, name: str, category: str) -> CaseResult:
        reading = await self._read(f"{cid}-pre")
        if reading.kind is None or reading.balance is None:
            return CaseResult(cid, name, category, "warning", True,
                              "无法取得剩余额度，无法判断按上限预扣是否卡死小额请求（留痕，不计入结论）"
                              + self._oracle_blind_suffix(),
                              {"oracle": self.oracle_kind})
        balance = reading.balance
        reserve_at_probe = self.cfg.reserve(self.cfg.balance_probe_mt)
        tiny_mt = 16
        tiny = await self.send(f"{cid}-tiny", tiny_mt, SHORT_PROMPT)
        obs = {"oracle": self.oracle_kind, "balance_usd": balance, "probe_max_tokens": self.cfg.balance_probe_mt,
               "probe_reserve_usd": reserve_at_probe, "tiny_max_tokens": tiny_mt,
               "tiny_reserve_usd": self.cfg.reserve(tiny_mt), "tiny_status": tiny.status}
        if tiny.ok:
            if balance < reserve_at_probe:
                return CaseResult(cid, name, category, "warning", True,
                                  f"余额 ${balance:.6f} 低于上限 {self.cfg.balance_probe_mt} 的冻结额 "
                                  f"${reserve_at_probe:.4f}，但冻结额仅 ${self.cfg.reserve(tiny_mt):.6f} 的请求仍可执行"
                                  f"→ 预扣不长期占用资金；同时说明门槛按**下单上限**卡：余额连常规 max_tokens"
                                  f"（如 256，冻结额 ${self.cfg.reserve(256):.4f}）都覆盖不了，钱在账上却花不出去"
                                  f"（资金效率问题，非计费不实）", obs)
            return CaseResult(cid, name, category, "passed", True,
                              f"余额 ${balance:.6f} 足以覆盖 ${self.cfg.reserve(tiny_mt):.6f} 的小额请求且执行正常"
                              f"→ 预扣不长期占用资金", obs)
        if tiny.status in (401, 403, 429):
            if balance < self.cfg.reserve(tiny_mt):
                # 余额连本次的冻结额都覆盖不了 → 被拒是**正常的预扣检查**，不能据此
                # 说「冻结未释放」（否则会冤枉一个只是没钱的账号）。
                return self._blocked(cid, name, category,
                                     f"余额 ${balance:.6f} 连本次冻结额 ${self.cfg.reserve(tiny_mt):.6f} 都覆盖不了，"
                                     f"请求被拒属正常的预扣检查 → 「冻结是否长期占用」从未被触达；"
                                     f"同时可见门槛按**下单上限**卡：余额低于上限 {self.cfg.balance_probe_mt} 的冻结额 "
                                     f"${reserve_at_probe:.4f}，钱在账上却花不出去（资金效率问题，非计费不实）", obs)
            return CaseResult(cid, name, category, "failed", False,
                              f"余额 ${balance:.6f} 已覆盖该请求的冻结额 ${self.cfg.reserve(tiny_mt):.6f}，"
                              f"却仍被拒（HTTP {tiny.status}）→ 此前请求的冻结额度未释放，预扣被长期占用："
                              f"{tiny.error_text[:160]}", obs)
        return self._blocked(cid, name, category, f"小额请求未取得有效响应（HTTP {tiny.status}）：{tiny.error_text[:160]}", obs)

    async def case_thinking_residual(self, cid: str, name: str, category: str) -> CaseResult:
        observed = [n for n in self.pool if n.out_tokens is not None]
        # 只认**真实思考用量**（>0）。字段存在但值恒为 0 时（2026-10-09 sonnet-5 批次）本探针无判别力，
        # 若按 `is not None` 收样，会把 `output − 0 = output` 读成"余量自洽"——那是空断言。
        with_think = [n for n in observed if n.think_tokens]
        zero_think = [n for n in observed if n.think_tokens == 0]
        contradictions, saturated = [], []
        for norm in with_think:
            residual = norm.out_tokens - norm.think_tokens
            if residual <= 2 and len(norm.text) >= 10 and norm.stop_reason != "max_tokens":
                contradictions.append({"case": norm.tag, "max_tokens": norm.max_tokens,
                                       "output_tokens": norm.out_tokens, "thinking_tokens": norm.think_tokens,
                                       "residual_tokens": residual, "chars": len(norm.text),
                                       "stop_reason": norm.stop_reason, "text": norm.text[:50]})
            if norm.out_tokens == norm.max_tokens and norm.stop_reason not in (None, "max_tokens") and len(norm.text) >= 10:
                saturated.append({"case": norm.tag, "max_tokens": norm.max_tokens,
                                  "stop_reason": norm.stop_reason, "chars": len(norm.text)})
        obs = {"responses_analyzed": len(observed), "responses_with_thinking_tokens": len(with_think),
               "responses_thinking_zero": len(zero_think),
               "residual_le_2_count": len(contradictions), "saturated_end_turn_count": len(saturated),
               "samples": contradictions[:5] + saturated[:5]}
        if contradictions or saturated:
            return CaseResult(cid, name, category, "failed", False,
                              f"**账目自相矛盾**：{len(contradictions)} 条响应 `output_tokens − thinking_tokens ≤ 2`"
                              f"（账目只给可见正文留 ≤2 个 token），可见正文却 ≥10 字符且自称自然结束；"
                              f"另有 {len(saturated)} 条贴上上限却非 max_tokens 截断 → `usage` 不可能是实测值", obs)
        if not with_think and zero_think:
            return CaseResult(cid, name, category, "warning", True,
                              f"{len(zero_think)} 条响应的 `thinking_tokens` 均为 0（该模型在该通道上思考未产生用量）"
                              f"→ 余量指纹无判别力，本项不构成用量自洽的证据（留痕，不计入结论）", obs)
        if len(with_think) < 3:
            return CaseResult(cid, name, category, "warning", True,
                              f"仅 {len(with_think)} 条响应带思考用量明细，样本不足以检验余量指纹（留痕）", obs)
        return CaseResult(cid, name, category, "passed", True,
                          f"{len(with_think)} 条响应的 output−thinking 余量与可见正文长度自洽，未发现矛盾指纹", obs)


# --------------------------------------------------------------------------- #
# 汇总与报告
# --------------------------------------------------------------------------- #
def summarize(results: list[CaseResult], cfg: ProbeConfig,
              notes: Iterable[str] | None = None) -> dict[str, Any]:
    counts = {k: sum(1 for r in results if r.status == k) for k in ("passed", "failed", "warning", "blocked")}
    # blocked = 缺取证能力（无可用余额读数等），只记录不阻断准入（2026-10-10 口径）；
    # 但整批全为 blocked 时没有任何有效结论，仍不准入。
    admission = (bool(results) and counts["failed"] == 0
                 and any(r.status in {"passed", "warning"} for r in results))
    return {
        "schema_version": "volcperf.result.v2",
        "case": "RELAY-BILLING-PROBE",
        "test_kind": "billing-probe",
        "model": cfg.model,
        "base_url": cfg.base_url,
        "protocol": cfg.protocol,
        "rate_per_m": cfg.rate_per_m,
        "rate_per_m_source": cfg.rate_source,
        "rate_per_m_samples": [round(item, 6) for item in cfg.rate_samples],
        "rate_notes": list(notes or ()),
        "total": len(results),
        "selected_probes": list(cfg.probes),
        "probe_catalog": list(PROBE_ORDER),
        **counts,
        "not_evaluable": counts["blocked"],
        "pass_rate": counts["passed"] / len(results) if results else 0,
        "admission_passed": admission,
        "results": [asdict(r) for r in results],
        "evidence": "requests.jsonl",
        "note": "只判定中转计费可信性；官方文档合规类差异（承诺的 400 未执行等）不在范围内。"
                "无可用余额读数时计费类用例收口 blocked——缺取证能力不等于通道有问题，故 blocked 只记录不否决准入"
                "（整批全为 blocked 时仍不准入）。",
    }


def write_report(out: Path, summary: dict[str, Any], cfg: ProbeConfig) -> None:
    icon = {"passed": "通过", "failed": "不通过", "warning": "警告", "blocked": "不可评估"}
    lines = [
        "# 中转计费临时探针报告", "",
        "本报告只回答中转计费可信性：预扣是否退回、`usage` 是否真实、计费驱动因子、关思考能否省钱。",
        "官方文档合规类差异（承诺的 400 未执行、字段归一化等）**不在范围内**，仅留痕。", "",
        f"- 端址：`{cfg.base_url}`（协议 `{cfg.protocol}`，模型 `{cfg.model}`）",
        f"- 预扣单价：`${cfg.rate_per_m:.6f}/1M token`（{RATE_SOURCE_TEXT.get(cfg.rate_source, cfg.rate_source)}）",
        f"- 执行范围：{len(cfg.probes)}/{len(PROBE_ORDER)} 条探针"
        + ("" if len(cfg.probes) == len(PROBE_ORDER) else f"（`{'`, `'.join(cfg.probes)}`）"),
        f"- 准入结论：**{'通过' if summary['admission_passed'] else '不通过'}**",
        "", "| 用例 | 类别 | 状态 | 结论 |", "|---|---|---|---|",
    ]
    for item in summary["results"]:
        detail = str(item["details"]).replace("\n", " ").replace("|", "\\|")
        lines.append(f"| `{item['case_id']}` | {item['category']} | {icon.get(item['status'], item['status'])} | {detail} |")
    notes = summary.get("rate_notes") or []
    if notes:
        lines += ["", "## 单价标定与瞬时故障留痕", ""] + [f"- {note}" for note in notes]
    lines += [
        "", "## 证据", "",
        "- `requests.jsonl`：每一次请求（含余额探针、被拒的超额探针）的完整请求/响应/用量/校验",
        "- `summary.json` / `validation.json`：结构化结论与统一证据校验", "",
        "> 判定纪律：被上游以额度/权限为由拒绝的请求**不产生**计费结论（收口 `blocked`）；"
        "只有拿到可比余额差分或可比用量样本时才判 `failed`。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def config_from_env(args: argparse.Namespace) -> ProbeConfig:
    base = args.base_url or os.environ.get("BILLING_BASE_URL") or os.environ.get("VOLC_BASE_URL")
    key = args.api_key or os.environ.get("BILLING_API_KEY") or os.environ.get("VOLC_API_KEY")
    if not base or not key:
        raise ValueError("缺少配置：需要 --base-url/--api-key 或 BILLING_BASE_URL/BILLING_API_KEY")
    model = args.model or os.environ.get("BILLING_MODEL") or os.environ.get("VOLC_MODEL") or "claude-fable-5"
    protocol = args.protocol or os.environ.get("BILLING_PROTOCOL") or os.environ.get("VOLC_PROTOCOL") or "anthropic"
    ladder = tuple(int(x) for x in (args.billing_ladder or "1024,8192").split(",") if x.strip())
    raw_probes = args.probes if args.probes is not None else os.environ.get("BILLING_PROBES", "")
    selected_probes = [item.strip() for item in str(raw_probes or "").split(",")]
    raw_rate = args.rate_per_m if args.rate_per_m is not None else os.environ.get("BILLING_RATE_PER_M")
    rate_explicit = raw_rate is not None and str(raw_rate).strip() != ""
    return ProbeConfig(
        base_url=base.rstrip("/"), api_key=key, model=model, protocol=protocol,
        rate_per_m=float(raw_rate) if rate_explicit else ProbeConfig.rate_per_m,
        rate_explicit=rate_explicit,
        rate_source="configured" if rate_explicit else "default",
        timeout=float(args.timeout),
        verify_ssl=not args.insecure,
        trust_env=bool(args.trust_env),
        ladder=ladder if len(ladder) == 2 else (1024, 8192),
        balance_probe_mt=int(args.balance_probe_max_tokens),
        escalate_balance_probe=bool(args.escalate_balance_probe),
        balance_probe_ceiling=int(args.balance_probe_ceiling),
        max_retries=int(args.max_retries if args.max_retries is not None
                        else os.environ.get("BILLING_MAX_RETRIES", ProbeConfig.max_retries)),
        probes=expand_probe_selection(selected_probes),
    )


async def main_async(args: argparse.Namespace) -> int:
    cfg = config_from_env(args)
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else ROOT / "results"
    out = output_root / f"RELAY-BILLING-PROBE-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    log = ResultLog(out / "requests.jsonl")
    results: list[CaseResult] = []
    async with HttpExecutor(timeout=cfg.timeout, verify_ssl=cfg.verify_ssl, trust_env=cfg.trust_env,
                            max_connections=4) as executor:
        probe = BillingProbe(cfg, executor, log)
        for case_id, name, category, handler in PROBES:
            if case_id not in cfg.probes:
                continue
            try:
                outcome = await getattr(probe, handler)(case_id, name, category)
            except Exception as exc:  # 探针自身异常不能伪装成通道结论
                outcome = CaseResult(case_id, name, category, "blocked", False,
                                     f"探针执行异常，未取得任何通道结论：{type(exc).__name__}: {exc}")
            results.append(outcome)
            print(f"{outcome.case_id} {outcome.status} {outcome.details[:110]}", flush=True)
    log.close()
    if probe.transient_failures:
        uniq = list(dict.fromkeys(probe.transient_failures))
        probe.notes.append(
            f"本轮出现 {len(probe.transient_failures)} 次网关 5xx/连接错误（已各重试 "
            f"{cfg.max_retries} 次仍失败），示例：{'、'.join(uniq[:4])}"
            f"；受其影响的计费类用例收口 blocked/warning，属取证被打断，不代表通道计费有问题")
    summary = summarize(results, cfg, probe.notes)
    summary["transient_failures"] = list(probe.transient_failures)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finalized = finalize_result_dir(out)
    write_report(out, finalized, cfg)
    print(f"结果目录: {out}")
    return 0 if finalized.get("admission_passed") else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="中转计费临时探针")
    parser.add_argument("--base-url", help="中转端址，如 https://i-token.ai")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--model", help="模型名，默认 claude-fable-5")
    parser.add_argument("--protocol", choices=("anthropic", "openai"), help="调用面，默认 anthropic")
    parser.add_argument("--rate-per-m", default=None,
                        help="预扣单价（美元/1M token）。不填则从 403 预扣文案 `need quota` 自动标定"
                             "（跨模型通用）；取不到观测时退回默认 50。也可用 BILLING_RATE_PER_M")
    parser.add_argument("--billing-ladder", default="1024,8192",
                        help="计费标定的两个 max_tokens 档位（小,大），默认 1024,8192")
    parser.add_argument("--balance-probe-max-tokens", default=2048,
                        help="403 余额探针的下单上限，默认 2048（被拒不扣款，但若未被拒会按上限计费）")
    parser.add_argument("--escalate-balance-probe", action="store_true",
                        help="超额探针未被拒时继续放大（可能产生真实消耗，默认关闭）")
    parser.add_argument("--balance-probe-ceiling", default=524288, help="超额探针上限封顶")
    parser.add_argument("--max-retries", default=None, type=int,
                        help="5xx/连接错误的额外重试次数，默认 2（BILLING_MAX_RETRIES 亦可）；"
                             "瞬时故障先重试再收口，不会被记成 failed")
    parser.add_argument("--probes", default=os.environ.get("BILLING_PROBES", ""),
                        help="只执行指定探针（逗号分隔 case_id，如 BILLING-B02,BILLING-U03）；默认执行全部 11 条")
    parser.add_argument("--timeout", default=120, help="单请求超时（秒）")
    parser.add_argument("--insecure", action="store_true", help="跳过 TLS 校验")
    parser.add_argument("--trust-env", action="store_true", help="读取 HTTP(S)_PROXY 环境变量")
    parser.add_argument("--output-root", help="结果输出根目录，默认 <repo>/results")
    try:
        raise SystemExit(asyncio.run(main_async(parser.parse_args())))
    except ValueError as exc:
        raise SystemExit(f"配置错误: {exc}")


if __name__ == "__main__":
    main()
