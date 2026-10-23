#!/usr/bin/env python3
"""qianfan-vl-v1 1M 长上下文专项探测（**独立脚本，未接入测试平台登记**）。

对应《Kimi-K3 准入标准》第 6 章（Long-context 1M）与第 5 章 L5 档：
  I5 输入 ~980,000–1,048,000 prompt token，固定 2 并发，max_tokens=400。

本脚本做四件事：
  1) 校准：先发一条小样本，实测"字符 → token"的换算比，据此生成目标长度的稳定前缀；
  2) 标记：在 1M 输入的**首 / 中 / 尾**三处各埋一个唯一编号，并以提问要求模型原样引用；
  3) 完整性判定（第 6 章逐条）：prompt_tokens ≥980,000 且与发送估算偏差 ≤2%、
     不被 400/413/context_length_exceeded 拒绝、三处标记全部命中、流式与 usage 正常、
     正常结束（finish_reason 非 length 截断导致的空正文另有记录）；
  4) 记录：requests.jsonl（逐条 usage / TTFT / 标记命中）+ summary.json + report.md。

用法（API Key 只走环境变量）：

  QIANFAN_API_KEY=bce-v3/... .venv/bin/python scripts/probe_qianfan_long_context.py \
      --target-tokens 1000000 --requests 3 --concurrency 2 \
      --out /Users/liuyeyu/workspace/report/kimi/long-context-1m
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

DEFAULT_ENDPOINT = "https://qianfan.baidubce.com/v2/chat/completions"
DEFAULT_MODEL = "qianfan-vl-v1"

# 填充语料：短句循环拼接，便于按字符数精确切分（不引入随机性，保证可复现）
FILLER_UNIT = "长上下文填充段落：本段用于构造稳定的测试输入，不承载业务语义，仅用于占位。"
CALIB_CHARS = 40_000

MARKERS = [
    ("HEAD", "【HEAD-MARKER-7F2A】机密编号 QX-7741"),
    ("MID", "【MID-MARKER-3B8E】机密编号 ZT-2298"),
    ("TAIL", "【TAIL-MARKER-C9D4】机密编号 LM-5136"),
]
# 标记位置（占正文比例）：最前 / 正中 / 最末。尾标记紧贴结尾，用于排除"只留前缀"式截断。
MARKER_POSITIONS = {"HEAD": 0.01, "MID": 0.50, "TAIL": 0.99}
QUESTION = (
    "\n\n问题：上文的首部、中部、尾部各埋有一个标记及其机密编号。"
    "请只输出这三个机密编号，按首部、中部、尾部的顺序，用逗号分隔，不要输出其他内容。"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def filler_text(total_chars: int) -> str:
    repeats = total_chars // len(FILLER_UNIT) + 1
    return (FILLER_UNIT * repeats)[:total_chars]


def build_prompt(total_chars: int) -> str:
    """把三处标记分别插在正文的 **最前(~1%) / 正中(~50%) / 最末(~99%)**。

    尾标记必须紧贴结尾，否则"只保留前缀、丢弃尾段"式的静默截断仍能通过
    （若尾标记放在 2/3 处，截掉最后 1/3 也不影响命中）。
    """
    q_len = len(QUESTION)
    m_total = sum(len(m) for _, m in MARKERS)
    body_chars = max(total_chars - m_total - q_len, 300)
    text = filler_text(body_chars)
    n = len(text)
    edge = max(int(n * 0.01), 1)
    mid_at = int(n * 0.50)
    head, mid, tail = MARKERS
    return (
        text[:edge]
        + f"\n{head[1]}\n"
        + text[edge:mid_at]
        + f"\n{mid[1]}\n"
        + text[mid_at : n - edge]
        + f"\n{tail[1]}\n"
        + text[n - edge :]
        + QUESTION
    )


def post_once(
    client: httpx.Client,
    *,
    endpoint: str,
    model: str,
    api_key: str,
    text: str,
    request_id: str,
    max_tokens: int,
    stream: bool = True,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
        "X-Request-ID": request_id,
    }
    payload: dict[str, Any] = {
        "model": model,
        "stream": stream,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": text}],
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    started = time.perf_counter()
    rec: dict[str, Any] = {
        "request_id": request_id,
        "started_at": now_iso(),
        "prompt_chars": len(text),
        "payload_bytes": len(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
        "http_status": None,
        "usage": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_tokens": None,
        "ttft_ms": None,
        "e2e_ms": None,
        "done_received": False,
        "invalid_sse_events": 0,
        "chunk_count": 0,
        "finish_reasons": [],
        "content": "",
        "reasoning_chars": 0,
        "markers_found": [],
        "markers_missing": [],
        "transport_error": None,
        "response_excerpt": None,
        # 空闲诊断：首块耗时、最后一个块耗时、两块之间最长静默间隔（判定代理空闲超时 vs 网关限制）
        "first_chunk_s": None,
        "last_chunk_s": None,
        "max_idle_gap_s": 0.0,
        "closed_after_chunks": None,
    }
    try:
        if not stream:
            resp = client.post(endpoint, json=payload, headers=headers)
            rec["http_status"] = resp.status_code
            if resp.status_code != 200:
                rec["response_excerpt"] = resp.text[:800]
            else:
                body = resp.json()
                rec["chunk_count"] = 1
                rec["usage"] = body.get("usage")
                choices = body.get("choices") or []
                if choices:
                    message = choices[0].get("message") or {}
                    rec["content"] = message.get("content") or ""
                    rec["reasoning_chars"] = len(message.get("reasoning_content") or "")
                    if choices[0].get("finish_reason"):
                        rec["finish_reasons"] = [choices[0]["finish_reason"]]
                    rec["done_received"] = True
            rec["e2e_ms"] = (time.perf_counter() - started) * 1000
            usage = rec["usage"] or {}
            rec["prompt_tokens"] = usage.get("prompt_tokens")
            rec["completion_tokens"] = usage.get("completion_tokens")
            rec["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            found, missing = [], []
            for name, marker in MARKERS:
                code = marker.split()[-1]
                (found if code in rec["content"] else missing).append(name)
            rec["markers_found"] = found
            rec["markers_missing"] = missing
            return rec
        with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
            rec["http_status"] = resp.status_code
            if resp.status_code != 200:
                rec["response_excerpt"] = resp.read().decode("utf-8", "replace")[:800]
            else:
                last_ts = None
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        rec["done_received"] = True
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        rec["invalid_sse_events"] += 1
                        continue
                    rec["chunk_count"] += 1
                    elapsed = time.perf_counter() - started
                    if rec["first_chunk_s"] is None:
                        rec["first_chunk_s"] = round(elapsed, 3)
                    if last_ts is not None:
                        gap = elapsed - last_ts
                        if gap > rec["max_idle_gap_s"]:
                            rec["max_idle_gap_s"] = round(gap, 3)
                    last_ts = elapsed
                    rec["last_chunk_s"] = round(elapsed, 3)
                    choices = event.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        if delta.get("content"):
                            rec["content"] += delta["content"]
                            if rec["ttft_ms"] is None:
                                rec["ttft_ms"] = (time.perf_counter() - started) * 1000
                        if delta.get("reasoning_content"):
                            rec["reasoning_chars"] += len(delta["reasoning_content"])
                        reason = choices[0].get("finish_reason")
                        if reason:
                            rec["finish_reasons"].append(reason)
                    if event.get("usage"):
                        rec["usage"] = event["usage"]
    except Exception as exc:  # noqa: BLE001
        rec["transport_error"] = f"{type(exc).__name__}: {exc}"
    rec["e2e_ms"] = (time.perf_counter() - started) * 1000
    if stream and not rec["done_received"] and rec["transport_error"] is None:
        # 流被对端关闭但没有 [DONE]：记录"收到几个块后被关"
        rec["closed_after_chunks"] = rec["chunk_count"]

    usage = rec["usage"] or {}
    rec["prompt_tokens"] = usage.get("prompt_tokens")
    rec["completion_tokens"] = usage.get("completion_tokens")
    rec["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")

    found, missing = [], []
    for name, marker in MARKERS:
        code = marker.split()[-1]
        (found if code in rec["content"] else missing).append(name)
    rec["markers_found"] = found
    rec["markers_missing"] = missing
    return rec


def calibrate(client: httpx.Client, cfg: dict[str, Any], attempts: int = 3) -> dict[str, Any]:
    """小样本校准字符→token 换算比。上游偶发抖动（超时/5xx）会让单次校准失败，
    因此重试；全部失败才回落到经验值。"""
    text = filler_text(CALIB_CHARS)
    last: dict[str, Any] = {}
    for i in range(1, attempts + 1):
        rec = post_once(
            client,
            endpoint=cfg["endpoint"],
            model=cfg["model"],
            api_key=cfg["api_key"],
            text=text,
            request_id=f"CALIB-{i}-{uuid.uuid4().hex[:8]}",
            max_tokens=8,
        )
        chars = len(text)
        tokens = rec["prompt_tokens"]
        ratio = (chars / tokens) if tokens else None
        if ratio:
            print(
                f"[校准] 第{i}次成功：{chars:,} 字符 → prompt_tokens={tokens} → 约 {ratio:.4f} 字符/token",
                flush=True,
            )
            return {"record": rec, "chars": chars, "tokens": tokens, "chars_per_token": ratio,
                    "attempts": i}
        last = {"record": rec, "chars": chars, "tokens": tokens, "chars_per_token": None,
                "attempts": i}
        print(
            f"[校准] 第{i}次失败：http={rec['http_status']} err={rec['transport_error']} "
            f"{(rec.get('response_excerpt') or '')[:200]}",
            flush=True,
        )
        if i < attempts:
            time.sleep(3)
    return last


def write_report(
    out: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    calib: dict[str, Any],
) -> None:
    s = summary
    dev_str = "—" if s["deviation_pct"] is None else f"{s['deviation_pct']:+.2f}%"
    lines = [
        "# qianfan-vl-v1 1M 长上下文专项探测（独立脚本）",
        "",
        f"- 端点：`{cfg['endpoint']}`；模型：`{cfg['model']}`",
        f"- 目标输入：{cfg['target_tokens']:,} prompt token；`max_tokens={cfg['max_tokens']}`；并发 {cfg['concurrency']}；请求 {cfg['requests']}",
        "- 标记位置（占正文）：HEAD {:.0%} / MID {:.0%} / TAIL {:.0%}（尾标记紧贴结尾，用于排除「只留前缀」式截断）".format(
            MARKER_POSITIONS["HEAD"], MARKER_POSITIONS["MID"], MARKER_POSITIONS["TAIL"]
        ),
        "- 每条请求正文加唯一 nonce（第 3.3 章：不同请求使用不同内容，避免前缀缓存失真）：{}".format(
            "是" if cfg.get("unique_per_request") else "否（--same-payload）"
        ),
        f"- 校准换算：{calib['chars']:,} 字符 → {calib['tokens']} token（约 {calib['chars_per_token']:.4f} 字符/token，第 {calib.get('attempts')} 次）" if calib.get("chars_per_token") and calib.get("chars") else (
            f"- 校准换算：指定 {calib['chars_per_token']:.4f} 字符/token（未重新校准）" if calib.get("chars_per_token") else "- 校准未成功"),
        f"- 生成时间：{now_iso()}",
        "",
        "## 判定（对照《Kimi-K3 准入标准》第 6 章）",
        "",
        "| # | 第 6 章要求 | 实测 | 判定 |",
        "|---|---|---|---|",
    ]
    ok = s["http_200"]
    lines += [
        f"| 1 | 完整接收 1M：`prompt_tokens` ≥980,000 且与发送估算偏差 ≤2%，不得 400/413/context_length_exceeded | "
        f"实测 prompt_tokens={s['median_prompt_tokens']}；偏差={dev_str}；"
        f"HTTP 2xx {s['http_200']}/{s['probes']}；4xx {s['http_4xx']}；错误码 {s['error_codes']} | "
        f"{'✅ 通过' if s['item1'] else '❌ 不通过'} |",
        f"| 2 | 不得静默截断：首 / 中 / 尾三处关键信息均被正确引用 | 三处标记全命中 {s['markers_all_ok']}/{s['probes']}；逐条命中 {s['markers_ok_list']} | "
        f"{'✅ 通过' if s['item2'] else '❌ 不通过'} |",
        f"| 3 | 行为一致：流式格式 / `usage` / `finish_reason` 与短输入一致 | SSE 非法事件合计 {s['invalid_sse']}；usage 完整 {s['usage_complete']}/{s['probes']}；finish_reason 分布 {s['finish_reasons']} | "
        f"{'✅ 通过' if s['item3'] else '⚠️ 见明细'} |",
        f"| 4 | 完整完成：1M 输入 + 输出正常结束，不中途超时/断流 | `[DONE]` {s['done_ok']}/{s['probes']}；传输异常 {s['transport_errors']}；E2E P50 {s['e2e_p50_s']}s | "
        f"{'✅ 通过' if s['item4'] else '❌ 不通过'} |",
        f"| 5 | 计量一致：token 计量与原厂口径一致 | 校准比 {cfg.get('chars_per_token_used')}；实测 prompt_tokens 与字符数换算偏差见第 1 项 | "
        f"{'✅ 见第 1 项' if s['item1'] else '⚠️ 见第 1 项'} |",
        "",
        "## 汇总",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 请求数 | {s['probes']} |",
        f"| HTTP 200 | {s['http_200']} |",
        f"| prompt_tokens 中位数 | {s['median_prompt_tokens']} |",
        f"| prompt_tokens 最小 / 最大 | {s['min_prompt_tokens']} / {s['max_prompt_tokens']} |",
        f"| 与发送估算偏差 | {s['deviation_pct']}% |",
        f"| 三处标记全命中 | {s['markers_all_ok']}/{s['probes']} |",
        f"| TTFT P50 | {s['ttft_p50_s']} s |",
        f"| E2E P50 / 最大 | {s['e2e_p50_s']} / {s['e2e_max_s']} s |",
        f"| 首块最慢 / 最大块间静默 | {s.get('first_chunk_s_max')} / {s.get('max_idle_gap_s')} s |",
        f"| 无 [DONE] 被关闭 | {s.get('closed_without_done')} |",
        f"| 传输异常 | {s['transport_errors']} |",
        "",
        "## 逐条明细",
        "",
        "| 请求 | HTTP | prompt_tokens | 偏差% | completion | 标记命中 | finish_reason | TTFT(s) | E2E(s) | 首块(s) | 末块(s) | 最大静默(s) | [DONE] | 异常 |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in records:
        est = r.get("estimated_prompt_tokens")
        dev = "—" if (not est or not r.get("prompt_tokens")) else f"{(r['prompt_tokens'] - est) / est * 100:+.2f}"
        lines.append(
            f"| {r['request_id']} | {r.get('http_status')} | {r.get('prompt_tokens')} | {dev} | {r.get('completion_tokens')} "
            f"| {','.join(r.get('markers_found') or []) or '—'} | {','.join(r.get('finish_reasons') or []) or '—'} "
            f"| {'—' if r.get('ttft_ms') is None else round(r['ttft_ms'] / 1000, 2)} "
            f"| {'—' if r.get('e2e_ms') is None else round(r['e2e_ms'] / 1000, 2)} "
            f"| {r.get('first_chunk_s', '—')} | {r.get('last_chunk_s', '—')} | {r.get('max_idle_gap_s', '—')} "
            f"| {r.get('done_received')} | {r.get('transport_error') or ('流被关闭于第 %s 块' % r['closed_after_chunks'] if r.get('closed_after_chunks') else '—')} |"
        )
    lines += ["", "## 证据", "", "- `requests.jsonl`：逐条请求、usage、TTFT、标记命中与原始片段", "- `summary.json`：结构化判定", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="qianfan-vl-v1 1M 长上下文专项探测（独立脚本）")
    ap.add_argument("--endpoint", default=os.environ.get("QIANFAN_ENDPOINT", DEFAULT_ENDPOINT))
    ap.add_argument("--model", default=os.environ.get("QIANFAN_MODEL", DEFAULT_MODEL))
    ap.add_argument("--api-key", default=os.environ.get("QIANFAN_API_KEY") or os.environ.get("VOLC_API_KEY", ""))
    ap.add_argument("--target-tokens", type=int, default=1_000_000)
    ap.add_argument("--requests", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--no-stream", action="store_true", help="用非流式请求做诊断（默认返回 JSON 而非 SSE）")
    ap.add_argument(
        "--chars-per-token", type=float, default=None,
        help="固定字符→token 换算比（跳过校准）。用于上游校准抖动时复现同一批输入。",
    )
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--no-proxy", action="store_true")
    ap.add_argument(
        "--same-payload", action="store_true",
        help="所有请求共用同一份正文（默认每条请求加唯一 nonce，避免前缀缓存失真）",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.api_key:
        print("缺少 API Key：设置 QIANFAN_API_KEY（或 VOLC_API_KEY）", file=sys.stderr)
        return 2

    out = Path(args.out).expanduser().resolve() if args.out else (Path.cwd() / "long-context-out" / time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "endpoint": args.endpoint,
        "model": args.model,
        "api_key": args.api_key,
        "target_tokens": args.target_tokens,
        "max_tokens": args.max_tokens,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "stream": not args.no_stream,
        "unique_per_request": not args.same_payload,
    }

    records: list[dict[str, Any]] = []
    lock = threading.Lock()
    jsonl = (out / "requests.jsonl").open("w", encoding="utf-8", buffering=1)

    def emit(rec: dict[str, Any]) -> None:
        with lock:
            records.append(rec)
            jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"  {rec['request_id']} http={rec.get('http_status')} prompt_tokens={rec.get('prompt_tokens')} "
            f"markers={','.join(rec.get('markers_found') or []) or '—'} e2e="
            f"{'—' if rec.get('e2e_ms') is None else round(rec['e2e_ms'] / 1000, 1)}s"
            f"{' err=' + rec['transport_error'] if rec.get('transport_error') else ''}",
            flush=True,
        )

    with httpx.Client(timeout=args.timeout, trust_env=not args.no_proxy, follow_redirects=True) as client:
        print(f"端点 {cfg['endpoint']} | 模型 {cfg['model']} | 目标 {args.target_tokens:,} token | 并发 {args.concurrency} | 输出 {out}", flush=True)
        print("\n[0] 校准字符/token 换算比", flush=True)
        if args.chars_per_token:
            calib = {"record": None, "chars": None, "tokens": None,
                     "chars_per_token": args.chars_per_token, "attempts": 0,
                     "source": "cli-override"}
            print(f"[校准] 跳过，使用指定换算比 {args.chars_per_token:.4f} 字符/token", flush=True)
        else:
            calib = calibrate(client, cfg)
        ratio = calib.get("chars_per_token")
        if not ratio:
            print("校准失败，按经验值 1.6 字符/token 继续", flush=True)
            ratio = 1.6
        target_chars = int(args.target_tokens * ratio)
        cfg["chars_per_token_used"] = round(ratio, 4)

        print(f"\n[1] 发起 {args.requests} 条 1M 上下文请求（{target_chars:,} 字符，三处标记）", flush=True)
        payload_text = build_prompt(target_chars)
        est_tokens = int(len(payload_text) / ratio)

        def job(index: int) -> dict[str, Any]:
            # 第 3.3 章：不同请求使用不同内容，避免提示缓存导致结果失真
            # 在正文最前面加一条唯一 nonce，破坏与前序请求的前缀共享。
            nonce = uuid.uuid4().hex
            text = f"[req-{index:02d}-{nonce}] " + payload_text if cfg["unique_per_request"] else payload_text
            rec = post_once(
                client,
                endpoint=cfg["endpoint"],
                model=cfg["model"],
                api_key=cfg["api_key"],
                text=text,
                request_id=f"CTX1M-{index:02d}-{nonce[:6]}",
                max_tokens=cfg["max_tokens"],
                stream=cfg["stream"],
            )
            rec["estimated_prompt_tokens"] = int(len(text) / ratio)
            rec["nonce"] = nonce
            emit(rec)
            return rec

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(job, range(1, args.requests + 1)))

    jsonl.close()

    probes = len(records)
    tokens = [r["prompt_tokens"] for r in records if r.get("prompt_tokens")]
    est = [r["estimated_prompt_tokens"] for r in records if r.get("estimated_prompt_tokens")]
    deviation = None
    if tokens and est:
        deviation = round((statistics.median(tokens) - statistics.median(est)) / statistics.median(est) * 100, 2)
    ttft = [r["ttft_ms"] for r in records if r.get("ttft_ms") is not None]
    e2e = [r["e2e_ms"] for r in records if r.get("e2e_ms") is not None]
    error_codes = sorted({str((r.get("usage") or {}).get("error")) for r in records if r.get("http_status") not in (200,)} - {"None"}) or []

    summary = {
        "script": "scripts/probe_qianfan_long_context.py (standalone, not registered in platform)",
        "generated_at": now_iso(),
        "endpoint": cfg["endpoint"],
        "model": cfg["model"],
        "target_tokens": args.target_tokens,
        "chars_per_token_used": cfg["chars_per_token_used"],
        "probes": probes,
        "http_200": sum(1 for r in records if r.get("http_status") == 200),
        "http_4xx": sum(1 for r in records if isinstance(r.get("http_status"), int) and 400 <= r["http_status"] < 500),
        "error_codes": error_codes,
        "median_prompt_tokens": int(statistics.median(tokens)) if tokens else None,
        "min_prompt_tokens": min(tokens) if tokens else None,
        "max_prompt_tokens": max(tokens) if tokens else None,
        "deviation_pct": deviation,
        "markers_all_ok": sum(1 for r in records if not r.get("markers_missing") and r.get("markers_found")),
        "markers_ok_list": ["".join(r.get("markers_found") or []) or "—" for r in records],
        "invalid_sse": sum(r.get("invalid_sse_events", 0) for r in records),
        "usage_complete": sum(1 for r in records if r.get("prompt_tokens")),
        "finish_reasons": sorted({x for r in records for x in (r.get("finish_reasons") or [])}),
        "done_ok": sum(1 for r in records if r.get("done_received")),
        "transport_errors": sum(1 for r in records if r.get("transport_error")),
        "ttft_p50_s": (round(statistics.median(ttft) / 1000, 2) if ttft else None),
        "e2e_p50_s": (round(statistics.median(e2e) / 1000, 1) if e2e else None),
        "e2e_max_s": (round(max(e2e) / 1000, 1) if e2e else None),
        "first_chunk_s_max": (max(round(r["first_chunk_s"], 2) for r in records if r.get("first_chunk_s") is not None) if any(r.get("first_chunk_s") is not None for r in records) else None),
        "max_idle_gap_s": (max((r.get("max_idle_gap_s") or 0.0) for r in records) if records else None),
        "closed_without_done": sum(1 for r in records if r.get("closed_after_chunks")),
        "calibration": {k: v for k, v in calib.items() if k != "record"},
        "records": records,
    }
    median_tokens = summary["median_prompt_tokens"] or 0
    summary["item1"] = bool(
        probes
        and summary["http_200"] == probes
        and median_tokens >= 980_000
        and deviation is not None
        and abs(deviation) <= 2.0
    )
    summary["item2"] = bool(probes and summary["markers_all_ok"] == probes)
    summary["item3"] = bool(probes and summary["usage_complete"] == probes and summary["invalid_sse"] == 0)
    summary["item4"] = bool(probes and summary["done_ok"] == probes and summary["transport_errors"] == 0)

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(out, summary, records, cfg, calib)

    print("\n===== 1M 长上下文判定（标准第 6 章）=====")
    print(f"请求 {probes}；HTTP200 {summary['http_200']}；prompt_tokens 中位 {summary['median_prompt_tokens']}（偏差 {deviation}%）")
    print(f"三处标记全命中 {summary['markers_all_ok']}/{probes}；[DONE] {summary['done_ok']}/{probes}；传输异常 {summary['transport_errors']}")
    print(f"TTFT P50 {summary['ttft_p50_s']}s；E2E P50/max {summary['e2e_p50_s']}/{summary['e2e_max_s']}s")
    print(f"第1项(完整接收1M)={'通过' if summary['item1'] else '不通过'}；第2项(不截断)={'通过' if summary['item2'] else '不通过'}；"
          f"第3项(行为一致)={'通过' if summary['item3'] else '见明细'}；第4项(完整完成)={'通过' if summary['item4'] else '不通过'}")
    print(f"结果目录：{out}")
    return 0 if (summary["item1"] and summary["item2"] and summary["item4"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
