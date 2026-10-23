#!/usr/bin/env python3
"""DeepSeek 中转接入口径准入报告生成器。

输入 = 基线批次 summary（今日 12:5x 全量三批）+ 定向复测 results/DS_V4_* 目录
（13:53 之后生成、带 case_filter 或为 FINGERPRINT 补跑），按「最新证据优先」合并
每个 case 的结论，再按 `dsv4/relay_profile` 的中转接入口径给出三款模型的准入判定。

用法：
    .venv/bin/python scripts/ds_relay_admission_report.py \
        --baseline results/BATCH-73c06912bca24347/summary.json \
                   results/BATCH-537a062c1dc24595/summary.json \
                   results/BATCH-2970101bd57f4b65/summary.json \
        --retest-since 2026-10-21T13:53 \
        --out-md  ~/DeepSeek中转接入口径准入报告-20261021.md \
        --out-html ~/DeepSeek中转接入口径准入报告-20261021.html
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dsv4.relay_profile import (  # noqa: E402
    PROFILE_LABEL,
    face_label,
    face_of,
    evaluate_relay_admission,
    merge_case_records,
)

BASELINE_BATCHES = {
    "deepseek-v4-flash-0731": "results/BATCH-73c06912bca24347/summary.json",
    "deepseek-v4-pro-0813": "results/BATCH-537a062c1dc24595/summary.json",
    "deepseek-v4.1-flash-ark": "results/BATCH-2970101bd57f4b65/summary.json",
}

#: 本轮对网关路径的直连探测（2026-10-21 14:11，domestic-ark 分组的同一密钥）
GATEWAY_SURFACE = [
    {"path": "/v1/models", "status": "200", "body": "`object=list`，4 个模型（三款 DS + glm-5.2）",
     "verdict": "✅ 可用；说明 DS-V4-43 失败源于用例先打根域 `/models` 被前端 SPA 兜底成 200，未回退到 `/v1/models`"},
    {"path": "/v1/files", "status": "400", "body": "`Model name not specified`",
     "verdict": "❌ 未实现 Files API（请求被转给 chat 处理器）"},
    {"path": "/v1/user/balance", "status": "404", "body": "`Invalid URL (GET /v1/user/balance)`",
     "verdict": "❌ 端点不存在"},
    {"path": "/v1/responses", "status": "404", "body": "`Invalid URL (GET /v1/responses)`",
     "verdict": "❌ Responses API 面未开放"},
    {"path": "/v1/messages", "status": "404", "body": "`Invalid URL (GET /v1/messages)`",
     "verdict": "❌ Anthropic Messages 面未开放"},
    {"path": "/beta/chat/completions", "status": "200", "body": "返回前端 SPA 的 `<!doctype html>`",
     "verdict": "❌ Beta/FIM 面缺失，被前端 catch-all 兜底成 200"},
]

def build_headline(models_payload: list[dict]) -> str:
    """结论句按判定结果现算，避免结论写死与判据改动脱节。"""
    passed = [m["model"] for m in models_payload if m["verdict"]["admission_passed"]]
    failed = [m["model"] for m in models_payload if not m["verdict"]["admission_passed"]]
    blocking_items = [
        f"`{m['model']}` 的 {item['case_id']}" for m in models_payload for item in m["verdict"]["blocking"]
    ]
    gaps = max((len(m["verdict"]["relay_gaps"]) for m in models_payload), default=0)
    undeclared = sum(
        1
        for m in models_payload
        for record in m["verdict"]["recorded"]
        if str(record.get("note") or "").startswith("该模型未声明视觉能力")
    )
    parts = [
        "**一句话结论**：换上 domestic-ark（火山 DS）分组的密钥后，"
        f"各类中转未承诺面缺口原样复现（单模型最多 {gaps} 项 relay_gap，不阻断）；"
    ]
    if blocking_items:
        parts.append("核心面只剩 " + "、".join(blocking_items) + " 构成阻断。")
    else:
        parts.append("核心面无阻断项。")
    if undeclared:
        parts.append(f"另有 {undeclared} 项视觉用例落在未声明视觉的模型版本上，按「该模型不提供该能力面」记 warning。")
    if passed and failed:
        parts.append("按中转接入口径：**" + "、".join(f"`{m}`" for m in passed) + " 准入通过**，"
                     + "、".join(f"`{m}`" for m in failed) + " 不通过。")
    elif passed:
        parts.append("按中转接入口径：**三款全部准入通过**。")
    else:
        parts.append("按中转接入口径：三款均不通过。")
    return "".join(parts)

NOTES = [
    "**视觉能力面更正（2026-10-21 14:32）**：`deepseek-v4-flash-0731` 先前被判「视觉不达标」属**判据错误**，已撤销。"
    "官方定价页「图像理解」只给 `deepseek-flash`（模型版本 V4.1-Flash-1017）打勾、`deepseek-v4-pro` 打叉；"
    "0731 是视觉首次开放（2026-08-21 vision-exp）之前的纯文本版本，官方从未声明其支持图片。"
    "实测（`scripts/probe_ds_vision_capability.py`，证据 `results/DS_VISION_PROBE-20261021-143231/`）：0731 与 pro 收到图片块"
    "都返回 200 且 `prompt_tokens` 都是 24 → 99（+75），但都答不出图（0731 双色带答「黑/白」、OCR 错成 4512）；"
    "真正看图的 `v4.1-flash-ark` 同请求是 24 → 208 且两题全对。⇒ **token 增量是网关记账产物，不能作为视觉证据**。"
    "按口径改为「该模型不提供该能力面」，记 warning 不阻断。",
    "**缓存归属更正（2026-10-21 14:45）**：`deepseek-v4.1-flash-ark` 的 DS-V4-16 原判「不同 user_id 也命中 ⇒ 数据隔离风险」"
    "**不成立，已撤销**。按三分法排查（`scripts/probe_ds_cache_ownership.py`，证据 `results/DS_CACHE_PROBE-20261021-144307/`）："
    "① 该链路**从不返回** DeepSeek 原生 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`，只有 OpenAI 形状 "
    "`prompt_tokens_details.cached_tokens`；② `cached_tokens` 是真记账——冷请求为 0、随前缀长度按 64 对齐增长"
    "（384 / 896 / 1920 / 4096）；③ 换 `user_id` 数值不变，但响应 `id` / `created` / 正文每次都新 ⇒ **不是**「网关整段响应回放」"
    "（那种会给回同一个 id）。即命中来自网关/上游服务层的前缀缓存，**本身没有 user 维度**：domestic-ark 分组实际服务方是"
    "火山方舟（响应 model `deepseek-v4-1-flash-261017`、请求 id 为火山 `0217…` 形态），DeepSeek 平台的 `user_id` KV 隔离语义"
    "在这条链路上不适用 ⇒ 记 blocked 不阻断。要 `user_id` 级 KV 隔离只能直连 `api.deepseek.com`（原生字段可验）。"
    "用例判据已同步收紧：原生字段跨 user 命中才判 failed，响应同 id 判「网关回放」blocked，只有 cached_tokens 判「不可归因」blocked。",
    "三款模型的全部未通过项在 domestic-ark 分组上**复现率 100%**（无一项改善），可确认失败面来自网关实现，与分组/密钥无关。",
    "`deepseek-v4-pro-0813` 的 FINGERPRINT 套件仍**未闭合**：AUTH-01~05 正常完成"
    "（证据见 `results/DS_V4_FINGERPRINT-20261021-141116|141321|141539/requests.jsonl`），"
    "AUTH-06「Thinking 静默吞参对照」的 `thinking=enabled` + `max_tokens=128` 请求连续 4 次在 180s 读超时"
    "（12:52、14:11、14:13、14:15），属上游时延/预算不收敛问题，按口径记未验证、不阻断；"
    "建议把该用例的预算抬到 512+ 后补跑。",
    "`deepseek-v4-flash-0731` 的 DS-V4-16 在两轮里表现为「未观测到同 user_id 命中」（缓存 best-effort），隔离性无法判定，记 blocked。",
    "DS-V4-43（模型列表）的失败是**用例路径 + 网关 SPA 兜底**共同造成：用例先按官方文档打根域 `/models`，"
    "网关用前端 index.html 兜了 200（不是 404），因此没走到 `/v1/models` 回退分支；"
    "实测 `/v1/models` 能正常返回 4 个模型。若把根域回退条件放宽到「非 JSON 响应」，本项可判通过。",
    "本轮复测新增 runner 能力：`DS_CASE_IDS` 环境变量可按用例 ID 定向复测；并发跑批时的结果目录撞名也做了序号兜底。",
]


def find_retest_dirs(since: str) -> list[Path]:
    threshold = _dt.datetime.fromisoformat(since).timestamp()
    results = REPO / "results"
    dirs: list[Path] = []
    for entry in results.iterdir():
        if not entry.name.startswith("DS_V4_") or not entry.is_dir():
            continue
        summary = entry / "summary.json"
        if not summary.exists():
            continue
        if summary.stat().st_mtime >= threshold:
            dirs.append(entry)
    return sorted(dirs, key=lambda p: p.stat().st_mtime)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_cases(batch_summary_path: Path) -> dict[str, list[dict]]:
    """从批次 summary 抽出各子套件的 case 级记录，全部摊平到一个时间戳下。"""
    batch = load(batch_summary_path)
    stamp = str(batch.get("generated_at") or "")
    cases: list[dict] = []
    extras: dict[str, dict] = {}
    for child in batch.get("results", []):
        nested = child.get("summary") or {}
        stamp = stamp or str(nested.get("generated_at") or "")
        cases.extend(nested.get("results") or [])
        suite = str(child.get("case") or "")
        extras[suite] = {
            "passed": child.get("passed", nested.get("all_passed", False)),
            "total": nested.get("total"),
            "summary": {k: nested.get(k) for k in ("passed", "failed", "warning", "blocked", "error")},
        }
    return {"stamp": stamp, "cases": cases, "suites": extras}


def render_markdown(payload: dict) -> str:
    lines: list[str] = []
    ap = lines.append
    ap("# DeepSeek 中转接入口径准入报告")
    ap("")
    ap(f"- **口径**：{PROFILE_LABEL}（中转网关只承诺 Chat Completions 主面；"
       "Strict Tool、Chat Prefix、FIM、Anthropic、Responses、Files、余额、模型列表"
       "等未承诺面的缺失记为 relay_gap，**不阻断准入**；核心面 failed 才阻断）")
    ap(f"- **测试日期**：{payload['date']}　**网关**：`{payload['endpoint']}`　**分组**：domestic-ark（火山 DS）")
    ap(f"- **方式**：上轮全量基线 + 本轮对未通过项的定向复测（新分组密钥），按最新证据合并")
    ap("")
    ap("## 一、准入结论")
    ap("")
    ap(payload["headline"])
    ap("")
    ap("| 模型 | 核心面通过 | 阻断 failed | 中转缺口(relay_gap) | 记录项 | 准入 |")
    ap("|---|---|---:|---:|---:|---|")
    for m in payload["models"]:
        mark = "✅ 通过" if m["verdict"]["admission_passed"] else "❌ 未通过"
        v = m["verdict"]
        s = v["stats"]
        ap(f"| `{m['model']}` | {s['core_passed']}/{s['core_total']} | "
           f"**{len(v['blocking'])}** | {len(v['relay_gaps'])}/{s['relay_optional_total']} | "
           f"{len(v['recorded'])} | {mark} |")
    ap("")
    for m in payload["models"]:
        v = m["verdict"]
        ap(f"### {m['model']}")
        ap("")
        if v["blocking"]:
            ap("**阻断项（核心面 failed，须整改后复测）**：")
            ap("")
            for c in v["blocking"]:
                ap(f"- `{c['case_id']}` {c['name']} —— {c['details'][:120]}")
        else:
            ap("**阻断项：无。**")
        ap("")
        if v["relay_gaps"]:
            ap("**中转未承诺面缺口（记录，不阻断）**：")
            ap("")
            for c in v["relay_gaps"]:
                ap(f"- `{c['case_id']}` {c['name']}（{c['face_label']}）—— {c['details'][:100]}")
        else:
            ap("**中转未承诺面缺口：本轮复测全部达标或不再出现。**")
        ap("")
        if v["recorded"]:
            ap("**其他记录（warning/blocked/未声明能力，不阻断）**：")
            ap("")
            for c in v["recorded"]:
                note = f"（{c['note']}）" if c.get("note") else ""
                status = c.get("effective_status") or c["status"]
                ap(f"- `{c['case_id']}` {c['name']} [{status}] {note} —— {c['details'][:90]}")
        ap("")
    ap("## 二、网关能力面实测（本轮直连探测）")
    ap("")
    ap("判定这些失败是「端点不存在」还是「返回了坏结构」，直接对网关做了路径探测（同一密钥）：")
    ap("")
    ap("| 路径 | 状态 | 响应特征 | 结论 |")
    ap("|---|---|---|---|")
    for row in payload.get("gateway_surface", []):
        ap(f"| `{row['path']}` | {row['status']} | {row['body']} | {row['verdict']} |")
    ap("")
    ap("## 三、定向复测明细（改判项）")
    ap("")
    if payload["changed"]:
        ap("| 模型 | 用例 | 上轮(官方口径) | 本轮复测 | 结论变化 |")
        ap("|---|---|---|---|---|")
        for r in payload["changed"]:
            ap(f"| `{r['model']}` | `{r['case_id']}` {r['name']} | {r['before']} | {r['after']} | {r['note']} |")
    else:
        ap("**证据层**无结论改判：所有未通过项在新分组上原样复现，说明失败面来自网关/模型本身，与分组无关。")
        revised = [
            (m["model"], record["case_id"], record.get("name") or "")
            for m in payload["models"]
            for record in m["verdict"]["recorded"]
            if record.get("bucket") == "vision_undeclared"
        ]
        if revised:
            ap("")
            ap("**判据层**有 1 类改判（能力面归属修正，非证据变化）：")
            ap("")
            ap("| 模型 | 用例 | 上轮结论 | 本轮结论 | 依据 |")
            ap("|---|---|---|---|---|")
            for model, case_id, name in revised:
                ap(f"| `{model}` | `{case_id}` {name} | failed（视觉不达标） | warning（该模型不提供视觉面） | "
                   "官方定价页：图像理解仅 `deepseek-flash`(V4.1-Flash-1017) 支持；0731 属视觉开放前纯文本版本 |")
            ap("")
            ap("⇒ `deepseek-v4-flash-0731` 阻断项因此清零，准入结论由「不通过」改为「通过」。")
    ap("")
    ap("## 四、口径说明")
    ap("")
    ap("- **核心能力面（必达）**：别名/思考模式/流式分离/工具闭环/JSON/Context Cache/"
       "user_id 隔离/参数边界/鉴权/视觉(模型声明支持时)。任一 failed ⇒ 准入不通过。")
    ap("- **中转未承诺面（记录）**：Strict Tool、Chat Prefix、FIM、Anthropic Messages、"
       "Responses API、Files API、余额、模型列表。failed ⇒ relay_gap 留痕，不阻断。")
    ap("- **平台缺失/瞬时故障**：404/5xx/限流 ⇒ blocked，不阻断（沿用 gateway_contract 口径）。")
    ap("- 上轮官方口径与本次中转口径的差异不影响负向用例判据本身，只影响「面缺失」的归档级别。")
    ap("")
    if payload.get("notes"):
        ap("## 五、复测执行情况与未闭合项")
        ap("")
        for note in payload["notes"]:
            ap(f"- {note}")
        ap("")
    ap("## 六、证据索引")
    ap("")
    for m in payload["models"]:
        ap(f"- `{m['model']}`：基线 `{m['baseline']}`")
        for d in m["retest_dirs"]:
            ap(f"  - 复测 `results/{d}/`")
    vision_probe = payload.get("vision_probe")
    if vision_probe:
        ap(f"- 视觉能力面探针（三款模型 × 双色带 + 点阵 OCR）：`{vision_probe}`")
    ap("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", default=[str(REPO / p) for p in BASELINE_BATCHES.values()])
    parser.add_argument("--retest-since", default="2026-10-21T13:53")
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-html", required=True)
    args = parser.parse_args()

    retest_dirs = find_retest_dirs(args.retest_since)
    retest_summaries = []
    for d in retest_dirs:
        s = load(d / "summary.json")
        s["_dir"] = d.name
        retest_summaries.append(s)

    models_payload = []
    changed: list[dict] = []
    for baseline_path in args.baseline:
        base = baseline_cases(Path(baseline_path))
        model = base["suites"].get("DS_V4_P0", {}).get("model") or None
        if model is None:
            # 从 batch summary 顶层取
            batch = load(Path(baseline_path))
            model = batch.get("model")
        own_retests = [s for s in retest_summaries if s.get("model") == model]
        merged = merge_case_records(
            [{"generated_at": base["stamp"], "results": base["cases"]}]
            + [{"generated_at": s.get("generated_at") or "", "results": s.get("results") or []} for s in own_retests]
        )
        verdict = evaluate_relay_admission(merged, model=model)
        # 判定变化表：上轮 failed → 本轮非 failed 的核心/中转面用例
        before_by_id = {c.get("case_id"): c for c in base["cases"]}
        for c in merged:
            cid = c.get("case_id")
            before = before_by_id.get(cid)
            if not before:
                continue
            b_status = str(before.get("case_status") or before.get("status") or "")
            a_status = str(c.get("case_status") or c.get("status") or "")
            if b_status == "failed" and a_status != "failed":
                changed.append({
                    "model": model, "case_id": cid, "name": c.get("name") or "",
                    "before": "failed", "after": a_status,
                    "note": "复测改善" if a_status == "passed" else "复测转非失败",
                })
            elif b_status != "failed" and a_status == "failed":
                changed.append({
                    "model": model, "case_id": cid, "name": c.get("name") or "",
                    "before": b_status, "after": "failed", "note": "复测转失败（口径内仍按最新证据）",
                })
        models_payload.append({
            "model": model,
            "baseline": str(Path(baseline_path).relative_to(REPO)),
            "retest_dirs": [s["_dir"] for s in own_retests],
            "verdict": verdict.as_dict(),
        })

    probe_dirs = sorted(
        (REPO / "results").glob("DS_VISION_PROBE-*"),
        key=lambda p: (p / "vision_probe.json").stat().st_mtime if (p / "vision_probe.json").exists() else 0,
    )
    payload = {
        "date": _dt.date.today().isoformat(),
        "endpoint": "https://api.oinone.top/v1",
        "models": models_payload,
        "changed": sorted(changed, key=lambda r: (r["model"], r["case_id"])),
        "gateway_surface": GATEWAY_SURFACE,
        "notes": NOTES,
        "headline": build_headline(models_payload),
        "vision_probe": (
            "results/" + probe_dirs[-1].name + "/vision_probe.json" if probe_dirs else None
        ),
    }

    md = render_markdown(payload)
    Path(args.out_md).write_text(md, encoding="utf-8")

    import markdown
    body = markdown.markdown(md, extensions=["tables"])
    html = (
        '<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">'
        "<title>DeepSeek 中转接入口径准入报告</title><style>"
        "body{font-family:-apple-system,'PingFang SC',sans-serif;max-width:960px;margin:32px auto;"
        "padding:0 24px;color:#1a1a1a;line-height:1.65;background:#fff}"
        "h1{font-size:26px;border-bottom:2px solid #e5e5e5;padding-bottom:10px}"
        "h2{font-size:20px;margin-top:36px;border-left:4px solid #2563eb;padding-left:10px}"
        "h3{font-size:17px;margin-top:24px}"
        "table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}"
        "th,td{border:1px solid #ddd;padding:7px 10px;text-align:left}"
        "th{background:#f5f5f5}code{background:#f2f2f2;padding:1px 5px;border-radius:4px;font-size:13px}"
        "li{margin:4px 0}</style></head><body>" + body + "</body></html>"
    )
    Path(args.out_html).write_text(html, encoding="utf-8")

    print(json.dumps({
        "models": {
            m["model"]: {
                "admission_passed": m["verdict"]["admission_passed"],
                "blocking": len(m["verdict"]["blocking"]),
                "relay_gaps": len(m["verdict"]["relay_gaps"]),
            } for m in models_payload
        },
        "retest_dirs": len(retest_dirs),
        "changed": len(changed),
        "out_md": args.out_md,
        "out_html": args.out_html,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
