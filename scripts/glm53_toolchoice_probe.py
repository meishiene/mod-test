#!/usr/bin/env python3
"""GLM-5.3 三档 · tool_choice 语义定钉探针。

主复测里出现两个互相矛盾的现象：
  - required + 不适配语境（问 2+2）→ 只有文本，没有 tool_calls
  - none + 天气提问 → flash / flashx 仍然发起 tool_calls
若两者同时成立，说明 tool_choice 并未被上游严格执行。本探针再加一发
**非法枚举值**（tool_choice="banana"）：若返回 200，则字段被链路吞掉/忽略，
这是「required 不强制 + none 不禁止」的共同原因。

每个模型 4 发：none×2（定复现率）、required、非法枚举。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

BASE_URL = "https://api.oinone.top"
OUT_ROOT = Path(__file__).resolve().parent.parent / "results"

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

CASES = [
    ("T1-none-weather-a", "北京今天天气怎么样？简单说说。", "none", "none + 明确会触发工具的问题（第 1 次）"),
    ("T2-none-weather-b", "上海今天天气怎么样？简单说说。", "none", "none + 同类问题（第 2 次，定复现率）"),
    ("T3-required-2plus2", "2 加 2 等于几？直接回答数字，不要调用任何工具。", "required", "required + 语义不适配"),
    ("T4-invalid-enum", "北京今天天气怎么样？", "banana", "非法枚举值，检验字段是否被校验/丢弃"),
]


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
    out = OUT_ROOT / f"GLM53_TOOLCHOICE-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    http = httpx.Client(
        base_url=BASE_URL, timeout=httpx.Timeout(180.0, connect=30.0), trust_env=True
    )
    summary = []
    with (out / "requests.jsonl").open("w", encoding="utf-8") as fh:
        for model in [m.strip() for m in args.models.split(",") if m.strip()]:
            rows = []
            for name, prompt, tc, note in CASES:
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 700,
                    "thinking": {"type": "enabled"},
                    "tools": [TOOL],
                    "tool_choice": tc,
                }
                resp = http.post(
                    "/v1/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {args.key}"},
                )
                try:
                    payload = resp.json()
                except Exception:
                    payload = {}
                ch = (payload.get("choices") or [{}])[0]
                msg = ch.get("message") or {}
                calls = msg.get("tool_calls") or []
                rec = {
                    "model": model,
                    "test": name,
                    "note": note,
                    "http_status": resp.status_code,
                    "request": body,
                    "finish_reason": ch.get("finish_reason"),
                    "tool_calls": [c["function"]["name"] for c in calls],
                    "content": msg.get("content"),
                    "error": payload.get("error"),
                    "usage": payload.get("usage"),
                }
                rows.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(
                    f"[{model}/{name}] http={rec['http_status']} "
                    f"finish={rec['finish_reason']} tool={rec['tool_calls']} "
                    f"err={(rec['error'] or {}).get('message') if rec['error'] else ''}",
                    flush=True,
                )
                time.sleep(1)
            none_calls = [r for r in rows[:2] if r["tool_calls"]]
            req_row = rows[2]
            inv_row = rows[3]
            summary.append(
                {
                    "model": model,
                    "none_tool_call_hits": f"{len(none_calls)}/2",
                    "required_forced": bool(req_row["tool_calls"]),
                    "invalid_enum_status": inv_row["http_status"],
                    "invalid_enum_error": (inv_row["error"] or {}).get("message"),
                    "rows": rows,
                }
            )
    http.close()
    (out / "summary.json").write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "models": summary},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n=== tool_choice 定钉 ===")
    for s in summary:
        print(
            f"{s['model']}: none 下仍调用工具 {s['none_tool_call_hits']} / "
            f"required 是否强制={s['required_forced']} / "
            f"非法枚举 HTTP {s['invalid_enum_status']}"
        )
    print(f"输出目录：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
