from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.report_metrics import build_item_metrics, suite_request_metrics
from common.result_adapter import normalize_summary
from common.scoring import score_batch

SCHEMA_VERSION = "volcperf.result.v1"

# 指标口径已上移到 common/report_metrics.py，单项报告与聚合报告共用同一实现；
# 这里保留同名导入，历史 `from web.aggregation import build_item_metrics` 仍可用。
__all__ = ["SCHEMA_VERSION", "build_item_metrics", "suite_request_metrics"]


def ensure_batch_metrics(payload: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """给批量聚合 manifest 补齐每项 metrics（旧批次落盘时还没有该字段）。"""
    if not isinstance(payload, dict) or payload.get("record_type") != "batch":
        return payload
    base = root or Path.cwd()
    for item in payload.get("results") or []:
        if not isinstance(item, dict) or item.get("metrics"):
            continue
        result_dir = item.get("result_dir")
        if result_dir and not Path(result_dir).is_absolute():
            result_dir = base / result_dir
        item["metrics"] = build_item_metrics(item.get("summary") or {}, result_dir)
    return payload


def enrich_summary(summary_path: Path, *, record_id: str, batch_id: str | None, test_kind: str,
                   case_name: str, protocol: str, base_url: str, model: str) -> dict[str, Any]:
    summary = normalize_summary(json.loads(summary_path.read_text(encoding="utf-8")))
    summary.update({"schema_version": SCHEMA_VERSION, "record_id": record_id, "batch_id": batch_id,
                    "test_kind": test_kind, "case": case_name, "protocol": protocol,
                    "base_url": base_url, "model": model})
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary



def write_capability_analysis(out: Path, aggregate: dict[str, Any], items: list[dict[str, Any]]) -> None:
    """生成能力专项的性能与差异分析；不虚构缺失的 request-level latency。"""
    rows=[]
    for item in items:
        summary=item.get("summary") or {}
        results=summary.get("results") or []
        durations=[r.get("duration_seconds") for r in results if isinstance(r.get("duration_seconds"),(int,float))]
        passed=sum(1 for r in results if r.get("status")=="passed" or r.get("passed") is True)
        failed=sum(1 for r in results if r.get("status") in {"failed","error"})
        warnings=sum(1 for r in results if r.get("status") in {"warning","blocked"})
        rows.append({"case":item["case"],"record_id":item["record_id"],"total":len(results),"passed":passed,"failed":failed,"warnings":warnings,"duration":sum(durations),"p50":sorted(durations)[len(durations)//2] if durations else None})
    def f(v): return "—" if v is None else f"{v:.2f}"
    lines=["# 能力专项性能分析报告","",f"- 批次：`{aggregate['record_id']}`",f"- 模型：`{aggregate['model']}`",f"- 说明：这是测试用例执行耗时分析，不是吞吐/TTFT 压测报告；原始数据未提供请求级延迟时不推导虚构指标。","","## 套件执行概览","","| 套件 | 用例数 | 通过 | 失败 | 警告/阻断 | 总耗时(s) | 用例耗时P50(s) |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows: lines.append(f"| {r['case']} | {r['total']} | {r['passed']} | {r['failed']} | {r['warnings']} | {f(r['duration'])} | {f(r['p50'])} |")
    lines += ["","## 结论","",f"- 批次完成：{aggregate['completed']}/{aggregate['total']}；底层用例通过：{sum(r['passed'] for r in rows)}。", "- 失败主要应结合差异报告和子任务日志定位；不能将套件总耗时等同于模型响应延迟。", "- 若需要 TTFT、TPOT、E2E、吞吐等性能结论，应执行独立性能压测并保留请求级明细。"]
    (out/"performance-report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    lines=["# 能力专项差异分析报告","",f"- 批次：`{aggregate['record_id']}`",f"- 模型：`{aggregate['model']}`","","## 测试项差异","","| 测试项 | 子套件 | 状态 | 关键观察 | 初步判断 |","|---|---|---|---|---|"]
    for item in items:
        for r in (item.get("summary") or {}).get("results",[]):
            status=r.get("status","unknown"); obs=r.get("observations") or {}; detail=str(r.get("details") or "—").replace("|","\\|").replace("\n"," ")[:260]
            if status=="passed": judgment="符合当前断言"
            elif status in {"warning","blocked"}: judgment="接口/能力限制，需单独确认"
            else: judgment="未满足断言；需结合响应和日志定位"
            lines.append(f"| {r.get('case_id') or r.get('id','—')} {r.get('name','')} | {item['case']} | {status} | {detail} | {judgment} |")
    lines += ["","## 跨套件对比","","- GLM 与 DeepSeek 已取消完整聚合子项；多选后的批次报告只汇总用户实际选中的独立子项。", "- `GLM_CONTRACT` 是自研公共契约，不能与 GLM 官方/专项用例直接做一一对应的通过率比较。", "- 原批次生成于 GLM 关闭 Thinking 用例移除前；`GLM-53-05` 应标记为不适用，不作为模型失败依据。", "- 流式、工具和 JSON Schema 的失败若伴随 `finish_reason=length` 或空正文，应优先检查 Thinking 输出预算与 adapter，不宜直接判定模型能力不存在。", "","## 建议复测对照","","1. 所有 GLM 请求固定 `thinking.enabled`。", "2. 正常能力测试使用较大 `max_tokens`，并记录 reasoning token、finish_reason 和完整正文。", "3. 流式测试保存原始 SSE，分别核对正文 delta、reasoning delta 和结束事件。", "4. 将本报告与新的同模型、同参数、同套件结果进行对比，避免跨接口直接比较。"]
    (out/"difference-report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def aggregate_batch(out: Path, batch: dict[str, Any], children: list[dict[str, Any]]) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    items=[]
    for child in children:
        summary={}
        path=child.get("summary_path")
        if path and Path(path).is_file():
            summary=normalize_summary(json.loads(Path(path).read_text(encoding="utf-8")))
        def portable(value: str | None) -> str | None:
            if not value: return None
            path_value=Path(value)
            try: return str(path_value.relative_to(Path.cwd())) if path_value.is_absolute() else str(path_value)
            except ValueError: return str(path_value)
        items.append({"record_id":child["id"],"case":child["case_name"],
                      # 多模型批次（同 case × 不同 model）靠该字段在前端/报告里按模型分行。
                      "model":child.get("model") or batch.get("model"),"status":child["status"],
                      "passed":None if child.get("passed") is None else bool(child["passed"]),
                      "result_dir":portable(child.get("result_dir")),"report_path":portable(child.get("report_path")),
                      "summary_path":portable(child.get("summary_path")),"summary":summary,"error":child.get("error"),
                      "log_path":portable(child.get("log_path")),
                      "metrics":build_item_metrics(summary, child.get("result_dir"))})
    completed=[x for x in items if x["status"]=="completed"]
    passed=sum(x["passed"] is True for x in items)
    aggregate={"schema_version":SCHEMA_VERSION,"record_id":batch["id"],"record_type":"batch",
               "test_kind":batch["test_kind"],"protocol":batch["protocol"],"base_url":batch["base_url"],
               "model":batch["model"],"cases":[x["case"] for x in items],"child_record_ids":[x["record_id"] for x in items],
               "total":len(items),"completed":len(completed),"passed":passed,"failed":len(items)-passed,
               "admission_passed":bool(items) and passed==len(items),"generated_at":datetime.now(timezone.utc).isoformat(),
               "results":items}
    # 批量评分：同一型号跨卡片合并（复测取最新一次），每个型号一份分数。
    # 只覆盖「能力/契约」类套件；压测批次没有一致性用例，score_batch 会返回空列表。
    scores = score_batch(items)
    aggregate["scores"] = scores
    if len(scores) == 1:
        aggregate["score"] = scores[0]
        aggregate["rating_passed"] = scores[0].get("rating_passed")
    (out/"summary.json").write_text(json.dumps(aggregate,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    # Self-contained manifest is optimized for later Codex aggregation by one or many record IDs.
    (out/"aggregation_manifest.json").write_text(json.dumps(aggregate,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    # 能力/契约专项没有压测指标，聚合报告应展示每个子用例的真实断言和错误，而不是空的吞吐列。
    is_capability = batch.get("test_kind") in {"kvv", "deepseek", "glm", "qwen", "authenticity", "arxivmath", "deepswe", "swe-atlas-tw", "swe-atlas-rf", "legal-agent-benchmark"}
    if is_capability:
        # 多模型批次（同一用例 × 多个模型，如 glm-5.3 / glm-5.3-flash）：子用例表加「模型」列，
        # 否则同一条用例会出现多行却分不清属于哪个模型；单模型批次版式不变。
        multi_model = len({str(x.get("model") or "") for x in items}) > 1
        lines=["# 能力专项批量测试报告","",f"- 记录 ID：`{batch['id']}`",f"- 协议：`{batch['protocol']}`",f"- 模型：`{batch['model']}`",
               *([f"- 模型范围：{'、'.join(str(x.get('model') or '—') for x in items)}"] if multi_model else []),
               f"- 用例数：{len(items)}",f"- 完成：{len(completed)}/{len(items)}",f"- 通过：{passed}/{len(items)}",f"- 总结论：{'通过' if aggregate['admission_passed'] else '未通过'}",
               "",
               "## 子用例结果","",
               "| 用例 | 子记录 | 模型 | 状态 | 结论 | 测试结果/失败原因 |" if multi_model
               else "| 用例 | 子记录 | 状态 | 结论 | 测试结果/失败原因 |",
               "|---|---|---|---|---|---|" if multi_model else "|---|---|---|---|---|"]
        for item in items:
            summary=item["summary"] or {}
            model_cell=[str(item.get("model") or "—")] if multi_model else []
            rows=summary.get("results") or []
            if rows:
                for row in rows:
                    status=row.get("status", "unknown"); mark="✅" if status=="passed" or row.get("passed") is True else "⚠️" if status in {"warning","blocked"} else "❌"
                    detail=str(row.get("details") or row.get("error") or "—").replace("|","\\|").replace("\n"," ")[:500]
                    cells=[f"{row.get('case_id', item['case'])} {row.get('name','')}", f"`{item['record_id']}`", *model_cell, status, mark, detail]
                    lines.append("| " + " | ".join(cells) + " |")
            else:
                error = str(summary.get("error") or item.get("error") or item.get("status") or "未生成专项结果").replace("|","\\|")
                cells=[item["case"], f"`{item['record_id']}`", *model_cell, item["status"], "❌", error]
                lines.append("| " + " | ".join(cells) + " |")
        lines += ["","## 子报告与日志","","| 子记录 | 报告 | 原始日志 |","|---|---|---|"]
        for item in items:
            report=f"`{item['report_path']}`" if item.get('report_path') else "—"; log=f"`{item['log_path']}`" if item.get('log_path') else f"`results/web-{item['record_id']}.log`"
            lines.append(f"| `{item['record_id']}` | {report} | {log} |")
    else:
        # 多模型批次：同一用例 × 多个模型时，首列按模型分行（表头随之改为「模型」），
        # 其余批次维持按用例/轮次分行，历史报告版式不变。
        multi_model = len({str(x.get("model") or "") for x in items}) > 1
        row_head = "模型" if multi_model else "轮次"
        scope_line = [f"- 模型范围：{'、'.join(str(x.get('model') or '—') for x in items)}"] if multi_model else []
        lines=["# 批量测试聚合报告","",f"- 记录 ID：`{batch['id']}`",f"- 子记录 ID：{', '.join(f'`{x}' for x in aggregate['child_record_ids'])}",
           f"- 协议：{batch['protocol']}",f"- 模型：{batch['model']}",
           *scope_line,
           f"- 总结论：{'通过' if aggregate['admission_passed'] else '未通过'}",
           f"- 通过：{passed}/{len(items)}",
           "", "## 各轮结果","",
           "正式请求 / 成功率 / TTFT·TTFC·E2E P90 来自子记录执行证据（`requests.jsonl` 的请求级 `timing`）；"
           "带压测字段的套件（多轮压测）直接取 summary；能力专项的 TTFT/TTFC 仅统计流式请求样本，样本数为 0 时显示 `-`。","",
           f"| {row_head} | 子记录 ID | 状态 | 结论 | 正式请求 | 成功率 | 用例通过 | TTFT P90(s) | TTFC P90(s) | E2E P90(s) | TPOT P90(ms/token) |",
           "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        def metric_text(item: dict[str, Any]) -> list[str]:
            m=item.get("metrics") or {}
            rate=m.get("success_rate")
            rate_text=f"{rate:.2%}" if isinstance(rate,(int,float)) else "-"
            cases=m.get("case_total")
            case_text=f"{m.get('case_passed',0)}/{cases}" if isinstance(cases,int) and cases else "-"
            def f3(v): return f"{v:.3f}" if isinstance(v,(int,float)) else "-"
            return [str(m["requests"]) if isinstance(m.get("requests"),int) else "-", rate_text, case_text,
                    f3(m.get("ttft_p90_seconds")), f3(m.get("ttfc_p90_seconds")), f3(m.get("e2e_p90_seconds")),
                    f3(m.get("tpot_p90_ms_per_token"))]
        for item in items:
            row_label = (item.get("model") or item["case"]) if multi_model else item["case"]
            lines.append(f"| {row_label} | `{item['record_id']}` | {item['status']} | {'通过' if item['passed'] else '未通过'} | "
                         + " | ".join(metric_text(item)) + " |")
    lines += ["","## 聚合数据说明","",f"机器可读数据：`aggregation_manifest.json`（schema `{SCHEMA_VERSION}`）。"]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if is_capability:
        write_capability_analysis(out, aggregate, items)
    return aggregate
