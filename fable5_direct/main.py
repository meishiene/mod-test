#!/usr/bin/env python3
"""Fable 5 直连实测入口。

    python main.py --key sk-xxx [--groups 基础能力,增强能力] [--only B01,E01]

不经过平台的 runner / Web / 队列，直接对被测端点发 HTTP 请求。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import BASE_URL, MODEL, Probe, run_case, summarize  # noqa: E402
import cases as case_module  # noqa: E402

ICON = {"passed": "✅ 通过", "failed": "❌ 失败", "error": "❌ 错误", "warning": "⚠️ 警告", "blocked": "⏸ 未验证"}
GROUP_ORDER = ["基础能力", "增强能力", "专项能力", "契约健壮性", "满血指纹"]


def build_markdown(summary: dict) -> str:
    meta = summary["target"]
    lines = [
        "# Fable 5 直连实测报告",
        "",
        f"- 被测端点：`{meta['base_url']}`",
        f"- 被测模型：`{meta['model']}`",
        f"- 执行方式：独立直连（不经平台 runner / Web / 任务队列）",
        f"- 生成时间：{summary['generated_at']}",
        f"- 用例总数：{summary['total']}",
        "",
        "## 总览",
        "",
        "| 结果 | 数量 |",
        "|---|---:|",
        f"| ✅ 通过 | {summary['passed']} |",
        f"| ❌ 失败 | {summary['failed']} |",
        f"| ⚠️ 警告（留痕，不判失败） | {summary['warning']} |",
        f"| ⏸ 未验证（平台前提不满足） | {summary['blocked']} |",
        f"| 已评估用例通过率 | {summary['pass_rate']:.1%} |",
        f"| 准入判定 | {'通过' if summary['admission_passed'] else '不通过'} |",
        "",
        "说明：**警告**表示线上行为与文档承诺存在差异但不足以判定模型失败；"
        "**未验证**表示上游以平台级前提（端点缺失 / beta 未开通 / 网络抖动）阻断了目标特性的触达，"
        "不计入模型能力结论。",
        "",
        "## 分维度结果",
        "",
        "| 维度 | 总数 | 通过 | 失败 | 警告 | 未验证 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for g in GROUP_ORDER:
        s = summary["by_group"].get(g)
        if not s:
            continue
        lines.append(f"| {g} | {s['total']} | {s.get('passed', 0)} | "
                     f"{s.get('failed', 0) + s.get('error', 0)} | {s.get('warning', 0)} | {s.get('blocked', 0)} |")
    lines.append("")
    for g in GROUP_ORDER:
        rows = [r for r in summary["results"] if r["group"] == g]
        if not rows:
            continue
        lines += [f"## {g}", "", "| 用例 | 结果 | 耗时(s) | 判定说明 |", "|---|---|---:|---|"]
        for r in rows:
            detail = str(r["details"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{r['case_id']}` {r['title']} | {ICON.get(r['status'], r['status'])} | "
                         f"{r['elapsed']:.2f} | {detail} |")
        lines.append("")
    lines.append("## 关键观测明细")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({r["case_id"]: r["observations"] for r in summary["results"]},
                            ensure_ascii=False, indent=1)[:60000])
    lines.append("```")
    return "\n".join(lines) + "\n"


def build_html(summary: dict) -> str:
    meta = summary["target"]
    cards = []
    color = {"passed": "#16a34a", "failed": "#dc2626", "error": "#dc2626",
             "warning": "#d97706", "blocked": "#6b7280"}
    bg = {"passed": "#f0fdf4", "failed": "#fef2f2", "error": "#fef2f2",
          "warning": "#fffbeb", "blocked": "#f3f4f6"}
    for g in GROUP_ORDER:
        rows = [r for r in summary["results"] if r["group"] == g]
        if not rows:
            continue
        s = summary["by_group"][g]
        trs = []
        for r in rows:
            c = color.get(r["status"], "#111")
            trs.append(
                f'<tr><td class="cid">{r["case_id"]}</td><td>{r["title"]}</td>'
                f'<td><span class="pill" style="color:{c};background:{bg.get(r["status"], "#fff")}">'
                f'{ICON.get(r["status"], r["status"])}</span></td>'
                f'<td class="num">{r["elapsed"]:.1f}</td><td class="det">{r["details"]}</td></tr>')
        cards.append(
            f'<section><h2>{g}<span class="cnt">{s["total"]} 条 · 通过 {s.get("passed", 0)} · '
            f'失败 {s.get("failed", 0) + s.get("error", 0)} · 警告 {s.get("warning", 0)} · '
            f'未验证 {s.get("blocked", 0)}</span></h2>'
            f'<table><thead><tr><th>ID</th><th>用例</th><th>结果</th><th>耗时</th><th>判定说明</th></tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></section>')
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fable 5 直连实测报告</title>
<style>
 body{{margin:0;padding:32px 24px;background:#f8fafc;color:#0f172a;
   font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.6}}
 .wrap{{max-width:1180px;margin:0 auto}}
 h1{{font-size:26px;margin:0 0 4px}} .sub{{color:#64748b;font-size:13px;margin-bottom:20px}}
 .kpis{{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0 26px}}
 .kpi{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:14px 18px;min-width:116px}}
 .kpi .v{{font-size:26px;font-weight:700}} .kpi .k{{font-size:12px;color:#64748b}}
 section{{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:6px 18px 16px;margin-bottom:20px}}
 h2{{font-size:17px;display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}}
 .cnt{{font-size:12px;font-weight:400;color:#64748b}}
 table{{width:100%;border-collapse:collapse;font-size:13px}}
 th{{text-align:left;color:#64748b;font-weight:600;font-size:12px;padding:8px 8px;border-bottom:1px solid #e2e8f0}}
 td{{padding:9px 8px;border-bottom:1px solid #f1f5f9;vertical-align:top}}
 tr:last-child td{{border-bottom:none}}
 .cid{{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#475569;white-space:nowrap}}
 .num{{text-align:right;color:#64748b;white-space:nowrap}}
 .det{{color:#334155}} .pill{{padding:2px 9px;border-radius:999px;font-size:12px;white-space:nowrap}}
 code{{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}}
</style></head><body><div class="wrap">
<h1>Fable 5 直连实测报告</h1>
<div class="sub">端点 <code>{meta['base_url']}</code> · 模型 <code>{meta['model']}</code> ·
生成于 {summary['generated_at']} · 独立直连，未经平台队列</div>
<div class="kpis">
 <div class="kpi"><div class="v">{summary['total']}</div><div class="k">用例总数</div></div>
 <div class="kpi"><div class="v" style="color:#16a34a">{summary['passed']}</div><div class="k">通过</div></div>
 <div class="kpi"><div class="v" style="color:#dc2626">{summary['failed']}</div><div class="k">失败</div></div>
 <div class="kpi"><div class="v" style="color:#d97706">{summary['warning']}</div><div class="k">警告</div></div>
 <div class="kpi"><div class="v" style="color:#6b7280">{summary['blocked']}</div><div class="k">未验证</div></div>
 <div class="kpi"><div class="v">{summary['pass_rate']:.0%}</div><div class="k">已评估通过率</div></div>
</div>
{"".join(cards)}
</div></body></html>"""


async def main_async(args) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted_groups = set(args.groups.split(",")) if args.groups else None
    wanted_ids = set(x.strip() for x in args.only.split(",")) if args.only else None
    registry = [c for c in case_module.REGISTRY
                if (not wanted_groups or c[1] in wanted_groups) and (not wanted_ids or c[0] in wanted_ids)]
    print(f"[fable5-direct] 目标 {args.base_url} / {args.model}；待执行 {len(registry)} 条用例", flush=True)

    results = []
    async with Probe(args.key, args.base_url, timeout=args.timeout, out_dir=out_dir) as probe:
        for case_id, group, title, fn in registry:
            res = await run_case(probe, case_id, group, title, fn, args.model)
            results.append(res)
            print(f"  {ICON.get(res.status, res.status)} {case_id} [{res.elapsed:6.1f}s] {title} — {res.details[:110]}",
                  flush=True)

    summary = summarize(results, {"base_url": args.base_url, "model": args.model,
                                  "mode": "独立直连（不经平台 runner/Web/队列）",
                                  "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(build_markdown(summary), encoding="utf-8")
    (out_dir / "report.html").write_text(build_html(summary), encoding="utf-8")
    print(f"\n[fable5-direct] 完成：通过 {summary['passed']} / 失败 {summary['failed']} / "
          f"警告 {summary['warning']} / 未验证 {summary['blocked']}；报告 {out_dir}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=os.environ.get("FABLE5_KEY", ""))
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out"))
    ap.add_argument("--groups", default="")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    if not args.key:
        print("缺少 --key", file=sys.stderr)
        return 2
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
