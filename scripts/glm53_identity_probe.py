#!/usr/bin/env python3
"""GLM-5.3 身份指纹补充探针（自研，2 问 / 模型）。

原批次 I01 判 warning 的依据是 code_signature 指纹"自述 claude"，但该子用例
max_tokens=4096 且强制思考，三条请求的可见正文全部为空（推理把预算吃满），
指纹实际取自不完整输出。本探针用足够预算重新取一次自述身份，只做证据补齐，
不改判模型能力。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

BASE_URL = "https://api.oinone.top"
OUT_ROOT = Path(__file__).resolve().parent.parent / "results"

PROMPTS = {
    "id-self": "你是什么模型？由哪家公司训练？只回一行，格式：模型名/公司名。",
    "id-kb": "你的知识截止到什么时间？只回「YYYY年M月」这一行。",
}

CLAUDE_HINTS = ("claude", "anthropic", "sonnet", "opus", "haiku")
GLM_HINTS = ("glm", "zhipu", "智谱", "chatglm")


def family_of(text: str) -> str:
    low = (text or "").lower()
    if any(h in low for h in GLM_HINTS):
        return "glm"
    if any(h in low for h in CLAUDE_HINTS):
        return "claude"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=os.environ.get("GLM_RETEST_KEY", ""))
    ap.add_argument("--models", default="glm-5.3,glm-5.3-flash,glm-5.3-flashx")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if not args.key:
        print("缺少 --key", file=sys.stderr)
        return 2

    stamp = args.tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = OUT_ROOT / f"GLM53_IDPROBE-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(
        base_url=BASE_URL, timeout=httpx.Timeout(180.0, connect=30.0), trust_env=True
    )
    all_rows = []
    with (out / "requests.jsonl").open("w", encoding="utf-8") as fh:
        for model in [m.strip() for m in args.models.split(",") if m.strip()]:
            rows = []
            for name, prompt in PROMPTS.items():
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1200,
                    "thinking": {"type": "enabled"},
                }
                started = time.time()
                resp = client.post(
                    "/v1/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {args.key}"},
                )
                try:
                    payload = resp.json()
                except Exception:
                    payload = {}
                choice = (payload.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                rec = {
                    "model": model,
                    "test": name,
                    "http_status": resp.status_code,
                    "elapsed_s": round(time.time() - started, 2),
                    "request": body,
                    "finish_reason": choice.get("finish_reason"),
                    "content": msg.get("content"),
                    "reasoning_len": len(msg.get("reasoning_content") or ""),
                    "usage": payload.get("usage"),
                    "error": payload.get("error"),
                }
                rec["family"] = family_of(rec["content"] or "")
                rows.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(
                    f"[{model}/{name}] http={rec['http_status']} "
                    f"finish={rec['finish_reason']} family={rec['family']} "
                    f"content={(rec['content'] or '')[:60]!r}",
                    flush=True,
                )
                time.sleep(1)
            votes = [r["family"] for r in rows if r["family"] != "unknown"]
            all_rows.append(
                {
                    "model": model,
                    "rows": rows,
                    "votes": votes,
                    "verdict": "consistent" if votes and set(votes) == {"glm"} else
                    ("mismatch" if votes else "indeterminate"),
                }
            )
    client.close()
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "models": all_rows,
        "tokens": {
            "prompt": sum(
                int((r["rows"][i].get("usage") or {}).get("prompt_tokens") or 0)
                for r in all_rows
                for i in range(len(r["rows"]))
            ),
            "completion": sum(
                int((r["rows"][i].get("usage") or {}).get("completion_tokens") or 0)
                for r in all_rows
                for i in range(len(r["rows"]))
            ),
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== 身份探针 ===")
    for r in all_rows:
        print(f"{r['model']}: votes={r['votes']} verdict={r['verdict']}")
    print(f"输出目录：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
