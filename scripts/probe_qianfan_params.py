#!/usr/bin/env python3
"""K3 官方契约的「取值域」闭环探测（**独立脚本，未接入测试平台登记**）。

补三条一直缺正向证据的探测（见 `report/kimi/qianfan-vl-v1-中转准入报告-20261013.md`
第 7.4 节）——过去只有网关错误文案自述「only 1 is allowed」，没有一条
`temperature=1` 成功返回的记录：

  1. `PARAM-TEMP1`        `temperature=1.0`（原厂固定值）     期望 200
  2. `PARAM-TEMP06`       `temperature=0.6`（区间内非固定值）  期望 400 ← 证明是「锁死」而非「越界拒绝」
  3. `PARAM-TOPP095`      `top_p=0.95`（原厂固定值）          期望 200
  4. `PARAM-TOPP09`       `top_p=0.9`（同区间非固定值）       200 ⇒ 未锁死；400 ⇒ 与 temperature 同锁
  5. `THINK-OFFICIAL`     `thinking={"type":"enabled","keep":"all"}` 期望 200
  6. `THINK-DISABLED`     `thinking={"type":"disabled"}`     400 拒绝 或 200 仍思考都算达标

用法（API Key 只走环境变量，不写盘、不进日志）：

  QIANFAN_API_KEY=bce-v3/... .venv/bin/python scripts/probe_qianfan_params.py \
      --out /Users/liuyeyu/workspace/report/kimi/params-probe

默认输出到 `<workspace>/report/kimi/params-probe/`（`requests.jsonl` + `summary.json` +
`report.md`）。判定在本脚本内自算，不依赖平台 validators；退出码 0 = 全部达标，
2 = 有 failed/warning（warning 表示「网关比原厂宽松」这类需要人工定性的情形）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT.parent / "report" / "kimi" / "params-probe"


def build_probes() -> list[dict[str, Any]]:
    """返回 (探测名, 请求参数, 期望判定) 三元组。"""
    return [
        {"id": "PARAM-TEMP1", "params": {"temperature": 1.0},
         "expect": "200", "why": "原厂固定值应被接受（缺的正向闭环证据）"},
        {"id": "PARAM-TEMP06", "params": {"temperature": 0.6},
         "expect": "400", "why": "区间内的非固定值也应被拒 ⇒ 取值域是单点而非越界校验"},
        {"id": "PARAM-TOPP095", "params": {"top_p": 0.95},
         "expect": "200", "why": "原厂固定值应被接受"},
        {"id": "PARAM-TOPP09", "params": {"top_p": 0.9},
         "expect": "200|400", "why": "同区间非固定值：200 ⇒ 未锁死；400 ⇒ 与 temperature 同锁"},
        {"id": "THINK-OFFICIAL", "params": {"thinking": {"type": "enabled", "keep": "all"}},
         "expect": "200", "why": "K3 唯一合法显式 thinking 形态"},
        {"id": "THINK-DISABLED", "params": {"thinking": {"type": "disabled"}},
         "expect": "400|200-thinking", "why": "K3 思考关不掉：400 拒绝或 200 仍思考都算达标"},
    ]


def read_reasoning(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    return str(message.get("reasoning_content") or message.get("reasoning") or "")


def judge(probe: dict[str, Any], status: int | None, body: Any, text: str) -> tuple[str, str]:
    """返回 (verdict, 说明)；verdict ∈ passed / warning / failed / blocked。"""
    expect = probe["expect"]
    if status is None:
        return "blocked", f"未取得 HTTP 状态（连接层失败）：{text[:200]}"
    if expect == "400|200-thinking":
        if status == 400:
            return "passed", "该非契约形态被 400 拒绝"
        if status == 200:
            reasoning = read_reasoning(body)
            if reasoning:
                return "passed", f"200 且思考照旧产生（reasoning {len(reasoning)} 字符）"
            return "failed", "200 且无 reasoning ⇒ 思考被真的关掉了，与原厂「思考关不掉」不符"
        return "blocked", f"既非 400 也非 200（{status}），无法判定"
    if expect == "200|400":
        if status == 200:
            return "passed", "非固定值被接受 ⇒ 该参数未被锁死"
        if status == 400:
            message = json.dumps(body, ensure_ascii=False) if body else text
            tag = "同锁" if probe["id"].endswith("TOPP09") else ""
            return "passed", f"400 拒绝（与 temperature {tag}）: {message[:200]}"
        return "blocked", f"既非 200 也非 400（{status}）"
    if expect == "200":
        if status == 200:
            return "passed", "200 接受"
        if 500 <= status:
            return "blocked", f"{status} 瞬时/上游故障，按平台口径记 blocked 不判失败"
        return "failed", f"期望 200，实得 {status}: {(json.dumps(body, ensure_ascii=False) if body else text)[:300]}"
    # expect == "400"
    if status == 400:
        return "passed", "400 拒绝"
    if status == 200:
        return "warning", "网关接受了非固定值（比原厂宽松）——与原厂行为不完全一致，需人工定性"
    if 500 <= status:
        return "blocked", f"{status} 瞬时/上游故障"
    return "failed", f"期望 400，实得 {status}"


def send(client: httpx.Client, *, url: str, key: str, model: str, extra: dict[str, Any],
         label: str) -> tuple[int | None, Any, str]:
    payload = {
        "model": model,
        "stream": False,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "只回复 OK"}],
    }
    payload.update(extra)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "X-Request-ID": f"{label}-{uuid.uuid4().hex[:10]}"}
    try:
        response = client.post(url, json=payload, headers=headers)
    except Exception as exc:  # noqa: BLE001 - 连接层失败按 blocked 记
        return None, None, f"{type(exc).__name__}: {exc}"
    try:
        body = response.json()
    except ValueError:
        body = None
    return response.status_code, body, response.text


def main() -> int:
    parser = argparse.ArgumentParser(description="K3 官方契约取值域闭环探测")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"输出目录（默认 {DEFAULT_OUT}）")
    parser.add_argument("--base-url", default=os.environ.get("QIANFAN_BASE_URL", "https://qianfan.baidubce.com"))
    parser.add_argument("--model", default=os.environ.get("QIANFAN_MODEL", "qianfan-vl-v1"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-proxy", action="store_true", help="忽略环境代理（默认 trust_env=true，走本机代理）")
    args = parser.parse_args()

    key = os.environ.get("QIANFAN_API_KEY", "").strip()
    if not key:
        print("缺少 QIANFAN_API_KEY（只从环境变量读，不写盘）", file=sys.stderr)
        return 1

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    url = args.base_url.rstrip("/") + "/v2/chat/completions"
    started_at = datetime.now(timezone.utc).isoformat()

    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=args.timeout, trust_env=not args.no_proxy, follow_redirects=True) as client:
        for probe in build_probes():
            began = time.perf_counter()
            status, body, text = send(client, url=url, key=key, model=args.model,
                                      extra=probe["params"], label=probe["id"])
            elapsed = time.perf_counter() - began
            verdict, note = judge(probe, status, body, text)
            usage = (body or {}).get("usage") if isinstance(body, dict) else None
            row = {
                "id": probe["id"], "params": probe["params"], "expect": probe["expect"],
                "why": probe["why"], "http_status": status, "verdict": verdict, "note": note,
                "elapsed_seconds": round(elapsed, 3), "usage": usage,
                "reasoning_chars": len(read_reasoning(body)),
                "error": (body or {}).get("error") if isinstance(body, dict) else None,
                "response_excerpt": (json.dumps(body, ensure_ascii=False) if body else text)[:800],
            }
            rows.append(row)
            print(f"{row['id']:16s} {str(status):>4s}  {verdict:8s} {note}", flush=True)

    with (out / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    failed = [row for row in rows if row["verdict"] == "failed"]
    warning = [row for row in rows if row["verdict"] == "warning"]
    blocked = [row for row in rows if row["verdict"] == "blocked"]
    summary = {
        "schema_version": "k3.param_probe.v1",
        "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url, "model": args.model, "endpoint": url,
        "total": len(rows), "passed": len(rows) - len(failed) - len(warning) - len(blocked),
        "failed": len(failed), "warning": len(warning), "blocked": len(blocked),
        "conclusion": "通过" if not failed and not warning and not blocked else "存在未达标/待定性项",
        "results": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# K3 官方契约取值域闭环探测", "",
             f"- 被测：`{args.model}` @ `{url}`",
             f"- 时间：{started_at}",
             f"- 结论：**{summary['conclusion']}**（通过 {summary['passed']} / 未达标 {summary['failed']} / 待定性 {summary['warning']} / 无法取证 {summary['blocked']}）",
             "", "| 探测 | 参数 | 期望 | HTTP | 判定 | 说明 |", "|---|---|---|---:|---|---|"]
    for row in rows:
        params = json.dumps(row["params"], ensure_ascii=False)
        lines.append(f"| `{row['id']}` | `{params}` | {row['expect']} | {row['http_status']} "
                     f"| {row['verdict']} | {row['note']} |")
    lines += ["", "> 判定在本脚本内自算（`expect` 列即口径）；证据见 `requests.jsonl`（逐条请求/响应/usage）。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n结果目录: {out}")
    return 0 if not failed and not warning and not blocked else 2


if __name__ == "__main__":
    raise SystemExit(main())
