#!/usr/bin/env python3
"""Run exactly 10 short-prompt requests to inspect content TTFT."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from k3perf.client import stream_chat
from k3perf.config import ApiConfig
from k3perf.metrics import percentile
from k3perf.reporting import write_artifacts
from k3perf.runner import ResultWriter


PROMPT = "这是一次短输入流式性能测试。请直接连续输出编号短句，格式为：item N verified stable。不要解释，不要调用工具。"


async def main() -> int:
    parser = argparse.ArgumentParser(description="固定发送10次短输入请求并统计正文TTFT")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0 or args.max_tokens <= 0:
        raise SystemExit("requests, concurrency and max-tokens must be positive")

    config = ApiConfig.from_env()
    out = args.output_root / f"SHORT-TTFT-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    metadata = {
        "case": "SHORT-TTFT",
        "purpose": "Short-input diagnostic only; not an admission replacement for I1/L1.",
        "started_at_epoch": time.time(),
        "endpoint": config.endpoint,
        "model": config.model,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "ttft_definition": "first non-empty delta.content - request start",
        "fixed_attempt_count_no_replenishment": True,
        "prompt": PROMPT,
        "environment": {"python": sys.version, "pid": os.getpid()},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    writer = ResultWriter(out / "requests.jsonl")
    semaphore = asyncio.Semaphore(args.concurrency)
    rows = []
    timeout = httpx.Timeout(config.request_timeout_seconds, connect=config.connect_timeout_seconds)
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)

    async with httpx.AsyncClient(timeout=timeout, limits=limits, verify=config.verify_ssl, trust_env=config.trust_env, headers={"Accept": "text/event-stream"}) as session:
        async def one(index: int) -> None:
            async with semaphore:
                row = await stream_chat(
                    session, config, f"SHORT-TTFT-{index:02d}-{uuid.uuid4().hex[:8]}",
                    PROMPT, args.max_tokens, "SHORT-TTFT", f"short-{index:02d}", "measured", True,
                )
            rows.append(row)
            writer.write(row)
            print(f"{index + 1}/{args.requests}: status={row.get('http_status')} success={row.get('success')} ttft={row.get('ttft_seconds')} error={row.get('error_type')}")

        await asyncio.gather(*(one(index) for index in range(args.requests)))
    writer.close()

    successful = [r for r in rows if r.get("success")]
    ttfts = [r["ttft_seconds"] for r in successful if r.get("ttft_seconds") is not None]
    e2es = [r["e2e_seconds"] for r in successful if r.get("e2e_seconds") is not None]
    prompts = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens") is not None]
    errors = {}
    for row in rows:
        if not row.get("success"):
            key = row.get("error_type") or f"http_{row.get('http_status')}"
            errors[key] = errors.get(key, 0) + 1
    summary = {
        "case": "SHORT-TTFT",
        "sent": len(rows),
        "successful": len(successful),
        "success_rate": len(successful) / len(rows),
        "errors": errors,
        "prompt_tokens_p50": percentile([float(v) for v in prompts], 0.5),
        "prompt_tokens_p90": percentile([float(v) for v in prompts], 0.9),
        "ttft_p50_seconds": percentile(ttfts, 0.5),
        "ttft_p90_seconds": percentile(ttfts, 0.9),
        "ttft_min_seconds": min(ttfts) if ttfts else None,
        "ttft_max_seconds": max(ttfts) if ttfts else None,
        "mean_tpot_ms_per_token": None,
        "e2e_p90_seconds": percentile(e2es, 0.9),
        "data_quality": {"usage_complete": all(r.get("prompt_tokens") is not None and r.get("completion_tokens") is not None for r in rows), "tpot_valid_samples": sum(r.get("tpot_ms_per_token") is not None for r in successful)},
        "checks": {},
        "capacity_windows": [],
        "admission_passed": None,
        "diagnostic_only": True,
        "note": "Exactly 10 short-input attempts; no failed-request replenishment.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_artifacts(out, rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {out.resolve()}")
    return 0 if len(successful) == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
