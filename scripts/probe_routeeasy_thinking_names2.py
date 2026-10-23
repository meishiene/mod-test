#!/usr/bin/env python3
"""第二轮：budget_tokens=1024 复测 4.x 系；验证「基名+thinking 参数」响应名是否一致。"""
from __future__ import annotations

import os

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fable5.model_profile import thinking_payload
from fable5.proto_probe import Client

BASE_URL = "https://api.routeeasy.ai/"
API_KEY = os.environ.get("ROUTEEASY_API_KEY", "")
PROMPT = "17*23 等于多少？只回答数字本身。"
MAX_TOKENS = 2048
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / (
    "routeeasy-thinking-name-probe2-" + time.strftime("%Y%m%d-%H%M%S"))

# (模型, 档位)：4.x 用 enabled+budget1024（含后缀模型与其基名）；5 系基名用 adaptive_effort
CASES = [
    ("claude-haiku-4-5-20251001-thinking", "enabled"),
    ("claude-opus-4-6-thinking", "enabled"),
    ("claude-sonnet-4-6-thinking", "enabled"),
    ("claude-haiku-4-5-20251001", "enabled"),
    ("claude-sonnet-4-6", "enabled"),
    ("claude-sonnet-5", "adaptive_effort"),
]


def probe(client: Client, model: str, mode: str) -> dict:
    thinking, output_config = thinking_payload(mode, budget=1024, effort="low")
    payload = {"model": model, "max_tokens": MAX_TOKENS, "thinking": thinking,
               "messages": [{"role": "user", "content": PROMPT}]}
    if output_config:
        payload["output_config"] = dict(output_config)
    resp = client.post(json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       accept="application/json", content_type="application/json")
    row = {"model": model, "mode": mode, "http_status": resp.status, "elapsed_s": round(resp.elapsed, 2)}
    body = resp.json()
    if body is None or resp.status != 200:
        row["error"] = ((body or {}).get("error") or {}).get("message") or resp.text_head
        return row
    content = body.get("content") or []
    think_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "thinking"]
    row.update({
        "response_model": body.get("model"),
        "model_matches_request": body.get("model") == model,
        "thinking_blocks": len(think_blocks),
        "has_signature": any(isinstance(b.get("signature"), str) and b["signature"] for b in think_blocks),
        "usage": body.get("usage"),
    })
    return row


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with Client(BASE_URL, API_KEY, model="(probe)", timeout=300, max_retries=1) as client:
        for model, mode in CASES:
            print(f"{model} ({mode}) ...", flush=True)
            try:
                row = probe(client, model, mode)
            except Exception as exc:  # noqa: BLE001
                row = {"model": model, "mode": mode, "http_status": 0, "error": str(exc)}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            time.sleep(1)
    (OUT_DIR / "probe_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(f"\n结果目录: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
