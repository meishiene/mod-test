#!/usr/bin/env python3
"""Routeeasy `-thinking` 模型名回显探针（省额度：每模型 1 条短请求）。

背景：签名两步校验对 `-thinking` 模型报「模型字段异常」，疑似网关把响应 model
写成不带 `-thinking` 后缀的基名。本探针不做两步回传，只对每个模型发**一条**
短请求（简单乘法题、小 max_tokens），记录：

- HTTP 状态 / 响应 model 字段 / 是否等于请求名
- thinking 块与签名是否存在（判断取签可行性）
- usage token 用量（对照额度消耗）

对照组：3 个不带后缀的基模型同样各发 1 条，确认基名模型响应 model 是否原样回显。
"""
from __future__ import annotations

import os

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fable5.model_profile import resolve_thinking_mode, same_model_id, thinking_payload
from fable5.proto_probe import Client

BASE_URL = "https://api.routeeasy.ai/"
API_KEY = os.environ.get("ROUTEEASY_API_KEY", "")

THINKING_MODELS = [
    "claude-haiku-4-5-20251001-thinking",
    "claude-haiku-5-5-thinking",
    "claude-opus-4-6-thinking",
    "claude-opus-4-7-thinking",
    "claude-opus-4-8-thinking",
    "claude-opus-5-thinking",
    "claude-opus-5-5-thinking",
    "claude-sonnet-4-6-thinking",
    "claude-sonnet-5-thinking",
    "claude-sonnet-5-5-thinking",
]
BASELINE_MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-haiku-5-5",
]
PROMPT = "17*23 等于多少？只回答数字本身。"
MAX_TOKENS = 1024          # enabled 档预算 512 + 正文上限；adaptive 档只作上限
BUDGET_TOKENS = 512        # enabled 档思考预算压到最小
EFFORT = "low"             # adaptive_effort 档压到 low 省额度

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / (
    "routeeasy-thinking-name-probe-" + time.strftime("%Y%m%d-%H%M%S"))


def probe_model(client: Client, model: str) -> dict:
    mode = resolve_thinking_mode(model)
    thinking, output_config = thinking_payload(mode, budget=BUDGET_TOKENS, effort=EFFORT)
    payload = {"model": model, "max_tokens": MAX_TOKENS, "thinking": thinking,
               "messages": [{"role": "user", "content": PROMPT}]}
    if output_config:
        payload["output_config"] = dict(output_config)
    resp = client.post(json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       accept="application/json", content_type="application/json")
    row: dict = {"model": model, "thinking_mode": mode, "http_status": resp.status,
                 "attempts": resp.attempts, "elapsed_s": round(resp.elapsed, 2)}
    body = resp.json()
    if body is None:
        row["error"] = "non-json"
        row["raw_head"] = resp.text_head
        return row
    if resp.status != 200:
        row["error_type"] = (body.get("error") or {}).get("type")
        row["error_message"] = (body.get("error") or {}).get("message")
        return row
    content = body.get("content") or []
    think_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "thinking"]
    row.update({
        "response_model": body.get("model"),
        "model_matches_request": same_model_id(body.get("model"), model),
        "thinking_blocks": len(think_blocks),
        "has_signature": any(isinstance(b.get("signature"), str) and b["signature"] for b in think_blocks),
        "stop_reason": body.get("stop_reason"),
        "usage": body.get("usage"),
    })
    return row


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with Client(BASE_URL, API_KEY, model="(probe)", timeout=300, max_retries=1) as client:
        for model in THINKING_MODELS + BASELINE_MODELS:
            group = "thinking" if model.endswith("-thinking") else "baseline"
            print(f"[{group}] {model} ...", flush=True)
            try:
                row = probe_model(client, model)
            except Exception as exc:  # noqa: BLE001
                row = {"model": model, "http_status": 0, "error": f"{type(exc).__name__}: {exc}"}
            row["group"] = group
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            time.sleep(1)
    (OUT_DIR / "probe_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")

    lines = ["# Routeeasy `-thinking` 模型名回显探针", "",
             f"- 端点：`{BASE_URL}`；每模型 1 条短请求（`{PROMPT}`），max_tokens={MAX_TOKENS}",
             f"- enabled 档 budget_tokens={BUDGET_TOKENS}；adaptive_effort 档 effort={EFFORT}",
             "", "| 组 | 模型 | 档位 | HTTP | 响应 model | 名字一致 | thinking块 | 签名 | 用量 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        usage = row.get("usage") or {}
        tokens = f"in{usage.get('input_tokens')}/out{usage.get('output_tokens')}" if usage else "-"
        lines.append("| {group} | `{model}` | {mode} | {status} | `{rm}` | {match} | {tb} | {sig} | {tk} |".format(
            group=row.get("group"), model=row.get("model"), mode=row.get("thinking_mode", "-"),
            status=row.get("http_status"), rm=row.get("response_model") or row.get("error_message") or row.get("error", "-"),
            match=("✅" if row.get("model_matches_request") else ("❌" if row.get("response_model") else "-")),
            tb=row.get("thinking_blocks", "-"), sig=("有" if row.get("has_signature") else "无" if row.get("thinking_blocks") else "-"),
            tk=tokens))
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\n结果目录: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
