#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import httpx

from k3perf.cases import CASES
from k3perf.client import stream_chat
from k3perf.config import ApiConfig
from k3perf.data import load_prompts


async def main() -> int:
    parser = argparse.ArgumentParser(description="Send 10 calibration requests and verify API-reported prompt token range")
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=Path("results/calibration"))
    args = parser.parse_args()
    case = CASES[args.case]
    config = ApiConfig.from_env()
    prompts = load_prompts(case.data_file)[:args.requests]
    timeout = httpx.Timeout(config.request_timeout_seconds, connect=config.connect_timeout_seconds)
    limits = httpx.Limits(max_connections=min(20, config.max_connections), max_keepalive_connections=20)
    rows = []
    async with httpx.AsyncClient(timeout=timeout, limits=limits, verify=config.verify_ssl, headers={"Accept": "text/event-stream"}, trust_env=config.trust_env) as session:
        for index, record in enumerate(prompts, 1):
            row = await stream_chat(session, config, f"CAL-{case.name}-{uuid.uuid4().hex[:10]}", record.content, min(case.max_tokens, 16), f"CAL-{case.name}", record.id, "calibration", True)
            rows.append(row)
            print(f"{index}/{len(prompts)}: success={row['success']} prompt_tokens={row['prompt_tokens']}")
    values = [r["prompt_tokens"] for r in rows if r["success"] and r["prompt_tokens"] is not None]
    summary = {
        "case": case.name,
        "expected_range": [case.prompt_min, case.prompt_max],
        "requests": len(rows),
        "valid_usage": len(values),
        "all_in_range": len(values) == len(rows) and all(case.prompt_min <= v <= case.prompt_max for v in values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "max": max(values) if values else None,
        "rows": rows,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / f"{case.name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print(f"Calibration result: {path.resolve()}")
    return 0 if summary["all_in_range"] else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
