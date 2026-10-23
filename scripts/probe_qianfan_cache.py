#!/usr/bin/env python3
"""qianfan-vl-v1 前缀缓存专项探测（**独立脚本，未接入测试平台登记**）。

本脚本不进 `volcperf/protocols.py`、`web/server.py` 的套件表，也不是任何
test_kind；只用与平台一致的请求形状（stream=true + include_usage）单独打
千帆 OpenAI 兼容面（`/v2/chat/completions`），用来量化前缀缓存行为：

  A 冷启动  —— 全新前缀 warm 后连续探测，观察缓存生效过程
  B 热缓存  —— 同前缀连续探测，测命中稳定性
  C 缺失率  —— 多组各自全新前缀，量化"响应里根本没有 cached_tokens"的概率
  D TTL     —— warm 后按 30/60/120/300s 间隔逐个探测，定位缓存存活边界

输出：`requests.jsonl`（逐条原始 usage / TTFT / 原始 SSE 尾部）
      `summary.json`（结构化汇总）
      `report.md`   （命中率报告，含逐条矩阵与 TTFT 对照）

用法（API Key 只走环境变量，不写盘、不进日志）：

  QIANFAN_API_KEY=bce-v3/... .venv/bin/python scripts/probe_qianfan_cache.py \
      --phases A,B,C,D --out /Users/liuyeyu/workspace/report/kimi/cache-probe

判定口径（本脚本自己算，不依赖平台 validators）：
  命中     = 响应 usage 里返回 `prompt_tokens_details.cached_tokens` 且
             cached_tokens / prompt_tokens >= --hit-ratio（默认 0.80）
  未命中   = 返回了该字段但比例低于阈值
  缺字段   = 整个 `prompt_tokens_details` 不出现（千帆未命中时不置 0 的情形）
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

DEFAULT_ENDPOINT = "https://qianfan.baidubce.com/v2/chat/completions"
DEFAULT_MODEL = "qianfan-vl-v1"

# 与平台 K3 缓存探测同源的前缀语料（24 字符左右的固定中文串，可复现）
BASE_SENTENCE = "K3 缓存稳定前缀。请保留并识别这段固定上下文。"
# 平台缓存探测用的经验换算：1 token ≈ 4 字符（英文为主）。中文按 1.7 更准，
# 2026-10-14 实测 deepseek-v4-pro-0813 1.70 / deepseek-v4.1-flash 1.73 字符/token，
# 故改为可配置：--chars-per-token，让 --target-tokens 直接表示"实际 prompt token"。
CHARS_PER_TOKEN = 4


def build_prefix(target_tokens: int, nonce: str, chars_per_token: float = CHARS_PER_TOKEN) -> str:
    """按 token 目标生成 byte-exact 可复现的稳定前缀（nonce 用于区分不同组）。"""
    chars = int(target_tokens * chars_per_token)
    unit = f"[{nonce}]" + BASE_SENTENCE
    repeats = chars // len(unit) + 1
    return (unit * repeats)[:chars]


def new_nonce() -> str:
    return uuid.uuid4().hex[:10]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def probe_once(
    client: httpx.Client,
    *,
    endpoint: str,
    model: str,
    api_key: str,
    text: str,
    request_id: str,
    max_tokens: int,
    hit_ratio: float,
) -> dict[str, Any]:
    """发一条流式请求，取 usage 与 TTFT；不改写任何参数。"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Request-ID": request_id,
    }
    payload = {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": text}],
    }
    started = time.perf_counter()
    record: dict[str, Any] = {
        "request_id": request_id,
        "started_at": now_iso(),
        "prompt_chars": len(text),
        "http_status": None,
        "usage": None,
        "prompt_tokens": None,
        "cached_tokens": None,
        "cache_ratio": None,
        "cache_state": None,  # hit / miss / missing_field
        "ttft_ms": None,
        "e2e_ms": None,
        "done_received": False,
        "invalid_sse_events": 0,
        "finish_reasons": [],
        "total_tokens": None,
        "transport_error": None,
    }
    try:
        with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
            record["http_status"] = resp.status_code
            if resp.status_code != 200:
                record["response_excerpt"] = resp.read().decode("utf-8", "replace")[:500]
            else:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        record["done_received"] = True
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        record["invalid_sse_events"] += 1
                        continue
                    choices = event.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        if record["ttft_ms"] is None and (
                            delta.get("content") or delta.get("reasoning_content")
                        ):
                            record["ttft_ms"] = (time.perf_counter() - started) * 1000
                        reason = choices[0].get("finish_reason")
                        if reason:
                            record["finish_reasons"].append(reason)
                    if event.get("usage"):
                        record["usage"] = event["usage"]
                    elif event.get("prompt_tokens_details") or event.get("prompt_tokens"):
                        # 少数网关把 usage 放在事件顶层
                        record["usage"] = {
                            k: event.get(k)
                            for k in (
                                "prompt_tokens",
                                "completion_tokens",
                                "total_tokens",
                                "prompt_tokens_details",
                                "completion_tokens_details",
                            )
                            if event.get(k) is not None
                        }
    except Exception as exc:  # noqa: BLE001 - 探测脚本，任何传输异常都留证不中断
        record["transport_error"] = f"{type(exc).__name__}: {exc}"
    record["e2e_ms"] = (time.perf_counter() - started) * 1000

    usage = record["usage"] or {}
    prompt = usage.get("prompt_tokens")
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens", usage.get("cached_tokens"))
    record["prompt_tokens"] = prompt
    record["cached_tokens"] = cached
    record["total_tokens"] = usage.get("total_tokens")
    if prompt and cached is not None:
        ratio = cached / prompt
        record["cache_ratio"] = ratio
        record["cache_state"] = "hit" if ratio >= hit_ratio else "miss"
    elif details or usage.get("cached_tokens") is not None:
        record["cache_state"] = "miss"
    else:
        record["cache_state"] = "missing_field"
    return record


def run_group(
    client: httpx.Client,
    cfg: dict[str, Any],
    *,
    phase: str,
    group: str,
    prefix: str,
    probe_count: int,
    warm_first: bool = True,
    prime_count: int = 0,
    ttl_marks: tuple[int, ...] = (),
    ttl_repeats: int = 1,
    emit,
) -> list[dict[str, Any]]:
    """一组探测：可选 1 次 warm + prime_count 次预热，再按 probe_count 或 TTL 间隔探测。

    warm/prime 只用于把缓存"焐热"，**不计入命中率**（与 warm 同待遇）；
    区分二者的意义在于：warm 是写缓存，prime 用来观察"第几次开始命中"。
    """
    out: list[dict[str, Any]] = []
    tag = f"{phase}-{group}"

    def budget_blocked() -> bool:
        stop = cfg.get("budget")
        if stop and stop.get("stopped"):
            print(f"  !! 累计 {stop['used']} token 已达预算上限 {stop['limit']}，跳过剩余探测", flush=True)
            return True
        return False

    if warm_first and not budget_blocked():
        rid = f"{tag}-WARM-{uuid.uuid4().hex[:6]}"
        record = probe_once(
            client,
            endpoint=cfg["endpoint"],
            model=cfg["model"],
            api_key=cfg["api_key"],
            text=prefix,
            request_id=rid,
            max_tokens=cfg["max_tokens"],
            hit_ratio=cfg["hit_ratio"],
        )
        record.update({"phase": phase, "group": group, "role": "warm"})
        emit(record)
        out.append(record)

    for index in range(1, prime_count + 1):
        if budget_blocked():
            break
        rid = f"{tag}-PRIME{index:02d}-{uuid.uuid4().hex[:6]}"
        record = probe_once(
            client,
            endpoint=cfg["endpoint"],
            model=cfg["model"],
            api_key=cfg["api_key"],
            text=prefix + f"\n预热尾部：prime-{index}",
            request_id=rid,
            max_tokens=cfg["max_tokens"],
            hit_ratio=cfg["hit_ratio"],
        )
        record.update({"phase": phase, "group": group, "role": f"prime_{index}"})
        emit(record)
        out.append(record)

    if ttl_marks:
        elapsed = 0
        for index, mark in enumerate(ttl_marks, 1):
            sleep_for = mark - elapsed
            if sleep_for > 0:
                print(f"    …等待 {sleep_for}s（warm 后第 {mark}s）", flush=True)
                time.sleep(sleep_for)
                elapsed = mark
            for repeat in range(1, ttl_repeats + 1):
                if budget_blocked():
                    return out
                rid = f"{tag}-TTL{mark:04d}-R{repeat}-{uuid.uuid4().hex[:6]}"
                record = probe_once(
                    client,
                    endpoint=cfg["endpoint"],
                    model=cfg["model"],
                    api_key=cfg["api_key"],
                    text=prefix + f"\n本次探测尾部：ttl-{mark}-r{repeat}",
                    request_id=rid,
                    max_tokens=cfg["max_tokens"],
                    hit_ratio=cfg["hit_ratio"],
                )
                record.update(
                    {
                        "phase": phase,
                        "group": group,
                        "role": f"ttl_{mark}s" if ttl_repeats == 1 else f"ttl_{mark}s_r{repeat}",
                        "ttl_mark_seconds": mark,
                        "ttl_repeat": repeat,
                    }
                )
                emit(record)
                out.append(record)
        return out

    for index in range(1, probe_count + 1):
        if budget_blocked():
            break
        rid = f"{tag}-P{index:02d}-{uuid.uuid4().hex[:6]}"
        record = probe_once(
            client,
            endpoint=cfg["endpoint"],
            model=cfg["model"],
            api_key=cfg["api_key"],
            text=prefix + f"\n本次探测尾部：{index}",
            request_id=rid,
            max_tokens=cfg["max_tokens"],
            hit_ratio=cfg["hit_ratio"],
        )
        record.update({"phase": phase, "group": group, "role": f"probe_{index}"})
        emit(record)
        out.append(record)
    return out


def summarize(records: list[dict[str, Any]], hit_ratio: float) -> dict[str, Any]:
    """只统计 probe 类请求（warm 是写缓存，不计入命中率）。"""

    def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        probes = [r for r in rows if str(r.get("role", "")).startswith(("probe_", "ttl_"))]
        hit = [r for r in probes if r["cache_state"] == "hit"]
        miss = [r for r in probes if r["cache_state"] == "miss"]
        missing = [r for r in probes if r["cache_state"] == "missing_field"]
        ratios = [r["cache_ratio"] for r in hit if r["cache_ratio"] is not None]
        ttft_hit = [r["ttft_ms"] for r in hit if r["ttft_ms"] is not None]
        ttft_missing = [r["ttft_ms"] for r in probes if r["cache_state"] != "hit" and r["ttft_ms"] is not None]
        return {
            "probes": len(probes),
            "hit": len(hit),
            "miss_low_ratio": len(miss),
            "missing_field": len(missing),
            "hit_rate": (len(hit) / len(probes)) if probes else None,
            "missing_field_rate": (len(missing) / len(probes)) if probes else None,
            "mean_cache_ratio_when_present": (statistics.mean(ratios) if ratios else None),
            "min_cache_ratio_when_present": (min(ratios) if ratios else None),
            "ttft_p50_hit_ms": (statistics.median(ttft_hit) if ttft_hit else None),
            "ttft_p50_not_hit_ms": (statistics.median(ttft_missing) if ttft_missing else None),
        }

    by_phase: dict[str, Any] = {}
    for phase in sorted({r.get("phase", "?") for r in records}):
        by_phase[phase] = stats([r for r in records if r.get("phase") == phase])
    probes_all = [r for r in records if str(r.get("role", "")).startswith(("probe_", "ttl_"))]
    return {
        "overall": stats(records),
        "by_phase": by_phase,
        "hit_definition": f"cached_tokens/prompt_tokens >= {hit_ratio}",
        "probe_total": len(probes_all),
        "transport_errors": sum(1 for r in records if r.get("transport_error")),
        "non_200": sum(1 for r in records if r.get("http_status") not in (200,)),
        "tokens_used_total": sum(r.get("total_tokens") or 0 for r in records),
        "prompt_tokens_total": sum(r.get("prompt_tokens") or 0 for r in records),
        "cached_tokens_total": sum(r.get("cached_tokens") or 0 for r in records),
    }


def write_report(
    out: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> None:
    order = ["WARM"] + [f"P{i:02d}" for i in range(1, cfg["probes"] + 1)]
    lines = [
        f"# {cfg['model']} 前缀缓存专项探测报告（独立脚本）",
        "",
        f"- 端点：`{cfg['endpoint']}`；模型：`{cfg['model']}`",
        f"- 稳定前缀目标：约 {cfg['target_tokens']:,} token（≈ {int(cfg['target_tokens'] * cfg['chars_per_token']):,} 字符）；输出上限 `max_tokens={cfg['max_tokens']}`",
        f"- 命中线：`{summary['hit_definition']}`；生成时间：{now_iso()}",
        "",
        "## 总体命中率（仅统计探测请求，不含 warm）",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    overall = summary["overall"]
    def pct(value):
        return "—" if value is None else f"{value * 100:.1f}%"
    def num(value, digits=1):
        return "—" if value is None else f"{value:.{digits}f}"
    lines += [
        f"| 探测请求数 | {overall['probes']} |",
        f"| 命中（≥{cfg['hit_ratio']:.0%}） | {overall['hit']} |",
        f"| 返回字段但低于命中线 | {overall['miss_low_ratio']} |",
        f"| **缺 `prompt_tokens_details` 字段** | **{overall['missing_field']}** |",
        f"| **命中率（按请求）** | **{pct(overall['hit_rate'])}** |",
        f"| 缺字段率 | {pct(overall['missing_field_rate'])} |",
        f"| 命中时平均 cached 占比 | {pct(overall['mean_cache_ratio_when_present'])} |",
        f"| 命中时最低 cached 占比 | {pct(overall['min_cache_ratio_when_present'])} |",
        f"| 命中请求 TTFT P50 | {num(overall['ttft_p50_hit_ms'], 0)} ms |",
        f"| 未命中(含缺字段)请求 TTFT P50 | {num(overall['ttft_p50_not_hit_ms'], 0)} ms |",
        f"| 非 200 响应 | {summary['non_200']} |",
        f"| 传输异常 | {summary['transport_errors']} |",
        f"| **本次消耗 token 合计** | **{summary['tokens_used_total']:,}**（prompt {summary['prompt_tokens_total']:,} / 其中命中 {summary['cached_tokens_total']:,}） |",
        "",
        "## 分阶段",
        "",
        "| 阶段 | 探测 | 命中 | 缺字段 | 命中率 | 命中时占比均值 | TTFT P50 命中 / 未命中 (ms) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    phase_name = {
        "A": "A 冷启动（全新前缀）",
        "B": "B 热缓存连续探测",
        "C": "C 缺失率（多组全新前缀）",
        "D": "D TTL（递增间隔）",
    }
    for phase, item in summary["by_phase"].items():
        lines.append(
            f"| {phase_name.get(phase, phase)} | {item['probes']} | {item['hit']} | {item['missing_field']} "
            f"| {pct(item['hit_rate'])} | {pct(item['mean_cache_ratio_when_present'])} "
            f"| {num(item['ttft_p50_hit_ms'], 0)} / {num(item['ttft_p50_not_hit_ms'], 0)} |"
        )
    lines += ["", "## 逐条明细", "", "| 阶段 | 组 | 角色 | HTTP | prompt_tokens | cached_tokens | 占比 | 状态 | TTFT(ms) |", "|---|---|---|---:|---:|---:|---:|---|---:|"]
    for r in records:
        ratio = "—" if r["cache_ratio"] is None else f"{r['cache_ratio'] * 100:.2f}%"
        lines.append(
            f"| {r.get('phase')} | {r.get('group')} | {r.get('role')} | {r.get('http_status')} "
            f"| {r.get('prompt_tokens') if r.get('prompt_tokens') is not None else '—'} "
            f"| {r.get('cached_tokens') if r.get('cached_tokens') is not None else '—'} "
            f"| {ratio} | {r.get('cache_state')} | {num(r.get('ttft_ms'), 0)} |"
        )
    lines += [
        "",
        "> 判定口径：命中 = 返回 `prompt_tokens_details.cached_tokens` 且 `cached_tokens / prompt_tokens` 达到命中线；",
        "> 缺字段 = 响应 usage 里根本没有 `prompt_tokens_details`（千帆未命中时不置 0）；warm 请求用于写缓存，不计入命中率。",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="qianfan-vl-v1 前缀缓存专项探测（独立脚本）")
    parser.add_argument("--endpoint", default=os.environ.get("QIANFAN_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--model", default=os.environ.get("QIANFAN_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.environ.get("QIANFAN_API_KEY") or os.environ.get("VOLC_API_KEY", ""))
    parser.add_argument("--phases", default="A,B,C", help="要跑的阶段，逗号分隔：A,B,C,D")
    parser.add_argument("--target-tokens", type=int, default=70000, help="稳定前缀目标 token 数（按 --chars-per-token 换算成字符）")
    parser.add_argument("--chars-per-token", type=float, default=CHARS_PER_TOKEN,
                        help="前缀语料的字符/token 换算比（中文实测约 1.7，默认 4 为英文经验值）")
    parser.add_argument("--budget-tokens", type=int, default=0,
                        help="本次探测允许消耗的 token 总量上限（0=不限）；达到即跳过剩余探测")
    parser.add_argument("--probes", type=int, default=7, help="A/B 阶段每组探测次数")
    parser.add_argument("--prime", type=int, default=0,
                        help="A/B 阶段探测前的预热次数（预热请求不计入命中率，用于跨过冷启动）")
    parser.add_argument("--groups", type=int, default=4, help="C 阶段全新前缀组数")
    parser.add_argument("--group-probes", type=int, default=3, help="C 阶段每组探测次数")
    parser.add_argument("--ttl-marks", default="30,60,120,300", help="D 阶段 warm 后的探测时点（秒，逗号分隔）")
    parser.add_argument("--ttl-repeats", type=int, default=1, help="D 阶段每个时点重复探测次数（用于区分 TTL 失效与随机未命中）")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--hit-ratio", type=float, default=0.80)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--no-proxy", action="store_true", help="忽略环境代理（默认 trust_env=true，走本机代理）")
    parser.add_argument("--out", default=None, help="输出目录；默认 ./cache-probe-out/<时间戳>")
    args = parser.parse_args()

    if not args.api_key:
        print("缺少 API Key：请设置 QIANFAN_API_KEY（或 VOLC_API_KEY），或用 --api-key 传入", file=sys.stderr)
        return 2

    phases = [p.strip().upper() for p in args.phases.split(",") if p.strip()]
    unknown = [p for p in phases if p not in {"A", "B", "C", "D"}]
    if unknown:
        print(f"未知阶段：{unknown}", file=sys.stderr)
        return 2

    out = Path(args.out).expanduser().resolve() if args.out else (Path.cwd() / "cache-probe-out" / time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)

    cfg = {
        "endpoint": args.endpoint,
        "model": args.model,
        "api_key": args.api_key,
        "target_tokens": args.target_tokens,
        "chars_per_token": args.chars_per_token,
        "max_tokens": args.max_tokens,
        "hit_ratio": args.hit_ratio,
        "probes": args.probes,
        "budget": {"limit": args.budget_tokens, "used": 0, "stopped": False},
    }
    ttl_marks = tuple(int(x) for x in args.ttl_marks.split(",") if x.strip())

    print(f"端点 {cfg['endpoint']} | 模型 {cfg['model']} | 阶段 {','.join(phases)} | 输出 {out}", flush=True)
    est_requests = 0
    if "A" in phases:
        est_requests += 1 + args.prime + args.probes
    if "B" in phases:
        est_requests += 1 + args.prime + args.probes
    if "C" in phases:
        est_requests += args.groups * (1 + args.prime + args.group_probes)
    if "D" in phases:
        est_requests += 1 + len(ttl_marks) * args.ttl_repeats
    est_tokens = est_requests * args.target_tokens
    cap = f"，预算上限 {args.budget_tokens:,}" if args.budget_tokens else ""
    print(f"预计请求 {est_requests} 次 × 约 {args.target_tokens:,} prompt token ≈ {est_tokens:,} token{cap}", flush=True)
    records: list[dict[str, Any]] = []
    jsonl = (out / "requests.jsonl").open("w", encoding="utf-8", buffering=1)

    def emit(record: dict[str, Any]) -> None:
        records.append(record)
        jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        budget = cfg["budget"]
        budget["used"] += record.get("total_tokens") or 0
        if budget["limit"] and budget["used"] >= budget["limit"]:
            budget["stopped"] = True
        state = record["cache_state"]
        ratio = "—" if record["cache_ratio"] is None else f"{record['cache_ratio'] * 100:.2f}%"
        ttft = "—" if record["ttft_ms"] is None else f"{record['ttft_ms'] / 1000:.2f}s"
        print(
            f"  {record.get('phase')}-{record.get('group'):<3} {record.get('role'):<12} "
            f"http={record['http_status']} cached={record['cached_tokens']} ratio={ratio} "
            f"state={state} ttft={ttft}",
            flush=True,
        )

    with httpx.Client(timeout=args.timeout, trust_env=not args.no_proxy, follow_redirects=True) as client:
        # A 冷启动：一组全新前缀，warm + probes
        if "A" in phases:
            print("\n[A] 冷启动（全新前缀，warm 后立即连续探测）", flush=True)
            run_group(client, cfg, phase="A", group="cold",
                      prefix=build_prefix(args.target_tokens, new_nonce(), args.chars_per_token),
                      probe_count=args.probes, prime_count=args.prime, emit=emit)
        # B 热缓存：复用 A 的前缀（若只跑 B，则用全新前缀 warm 一次再探测）
        if "B" in phases:
            print("\n[B] 热缓存连续探测", flush=True)
            run_group(client, cfg, phase="B", group="warm",
                      prefix=build_prefix(args.target_tokens, new_nonce(), args.chars_per_token),
                      probe_count=args.probes, prime_count=args.prime, emit=emit)
        # C 缺失率：多组各自全新前缀
        if "C" in phases:
            print(f"\n[C] 缺失率量化（{args.groups} 组全新前缀 × {args.group_probes} 次探测）", flush=True)
            for index in range(1, args.groups + 1):
                run_group(client, cfg, phase="C", group=f"g{index}",
                          prefix=build_prefix(args.target_tokens, new_nonce(), args.chars_per_token),
                          probe_count=args.group_probes, prime_count=args.prime, emit=emit)
        # D TTL
        if "D" in phases:
            print(f"\n[D] TTL 探测（warm 后 {ttl_marks} 秒）", flush=True)
            run_group(client, cfg, phase="D", group="ttl",
                      prefix=build_prefix(args.target_tokens, new_nonce(), args.chars_per_token),
                      probe_count=0, ttl_marks=ttl_marks, ttl_repeats=args.ttl_repeats, emit=emit)

    jsonl.close()

    summary = summarize(records, args.hit_ratio)
    summary.update(
        {
            "script": "scripts/probe_qianfan_cache.py (standalone, not registered in platform)",
            "generated_at": now_iso(),
            "endpoint": cfg["endpoint"],
            "model": cfg["model"],
            "phases": phases,
            "target_tokens": args.target_tokens,
            "max_tokens": args.max_tokens,
            "records": len(records),
        }
    )
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(out, summary, records, cfg)

    overall = summary["overall"]
    print("\n===== 命中率汇总（探测请求，不含 warm） =====")
    print(f"探测 {overall['probes']} 次：命中 {overall['hit']}，低比例 {overall['miss_low_ratio']}，缺字段 {overall['missing_field']}")
    hit_rate = overall["hit_rate"]
    mean_ratio = overall["mean_cache_ratio_when_present"]
    print(f"命中率 {'—' if hit_rate is None else f'{hit_rate * 100:.1f}%'}"
          f"（命中时 cached 占比均值 {'—' if mean_ratio is None else f'{mean_ratio * 100:.2f}%'}）")
    print(f"本次消耗 token：{summary['tokens_used_total']:,}（prompt {summary['prompt_tokens_total']:,}，其中命中 {summary['cached_tokens_total']:,}）")
    print(f"结果目录：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
