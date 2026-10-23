#!/usr/bin/env python3
"""用 50 个单并发真实请求校准所选数据档；失败也占固定样本，不补样。"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from volcperf.cases import CASES
from volcperf.config import ApiConfig
from volcperf.data import load_prompts
from volcperf.client import stream_chat
from volcperf.runner import session

async def run(case_name: str, count: int, output: Path) -> int:
    case, config = CASES[case_name], ApiConfig.from_env()
    prompts, rows = load_prompts(case.data_file), []
    async with session(config) as client:
        for i in range(count):
            record = prompts[i % len(prompts)]
            workload = record.workload(case.name == "O1")
            row = await stream_chat(client, config, request_id=f"CAL-{uuid.uuid4().hex[:12]}", messages=workload.messages,
                                    workload_type=workload.workload_type, target_visible_tokens=workload.target_visible_tokens,
                                    max_tokens=min(case.max_tokens, 32), case_name=f"CAL_{case.name}", dataset_id=record.id,
                                    phase="calibration", counted=True)
            rows.append(row); print(f"{i+1}/{count}: success={row['success']} prompt={row['prompt_tokens']} total={row['total_tokens']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    prompts_used = [r["prompt_tokens"] for r in rows if r.get("success") and r.get("prompt_tokens") is not None]
    totals = [r["total_tokens"] for r in rows if r.get("success") and r.get("total_tokens") is not None]
    result = {"case": case_name, "samples": count, "successful": sum(r["success"] for r in rows),
              "average_prompt_tokens": sum(prompts_used)/len(prompts_used) if prompts_used else None,
              "average_total_tokens": sum(totals)/len(totals) if totals else None,
              "c1_total_token_requirement_met": (sum(totals)/len(totals) >= 9524) if totals and case_name == "C1" else None,
              "rows": rows}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return 0 if len(prompts_used) == count and (case_name != "C1" or result["c1_total_token_requirement_met"]) else 2

def main():
    p=argparse.ArgumentParser(); p.add_argument("case", choices=sorted(CASES)); p.add_argument("--samples", type=int, default=50); p.add_argument("--output", type=Path)
    a=p.parse_args(); out=a.output or Path("results")/f"calibration-{a.case}.json"; raise SystemExit(asyncio.run(run(a.case,a.samples,out)))
if __name__ == "__main__": main()
