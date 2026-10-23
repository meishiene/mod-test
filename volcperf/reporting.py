from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from common.report_style import stats_lines

REQUEST_COLUMNS = [
    "case", "protocol", "phase", "counted", "request_id", "dataset_id", "target_prompt_tokens", "max_tokens", "planned_epoch", "started_epoch", "completed_epoch",
    "send_delay_ms", "success", "http_status", "error_type", "error_message", "prompt_tokens", "completion_tokens",
    "reasoning_tokens", "visible_completion_tokens", "total_tokens", "finish_reason", "first_sse_seconds",
    "first_reasoning_seconds", "ttft_seconds", "ttfc_seconds", "e2e_seconds", "tpot_ms_per_visible_token", "tpot_token_basis",
    "content_chunk_count", "reasoning_chunk_count", "content_char_count", "sse_event_count", "invalid_sse_event_count",
    "saw_done", "response_request_id", "response_headers", "workload_type", "message_count", "target_visible_tokens",
]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            row = dict(source)
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            writer.writerow(row)


def write_artifacts(out: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    write_csv(out / "requests.csv", rows, REQUEST_COLUMNS)
    write_csv(out / "errors.csv", [r for r in rows if r.get("counted") and not r.get("success")], REQUEST_COLUMNS)
    windows = summary.get("capacity_windows") or []
    if windows:
        write_csv(out / "capacity_windows.csv", windows, list(windows[0]))
    stats = []
    for key, value in summary.items():
        if not isinstance(value, (dict, list)):
            stats.append({"metric": key, "value": value})
    for key, value in summary.get("data_quality", {}).items():
        stats.append({"metric": f"data_quality.{key}", "value": value})
    write_csv(out / "statistics.csv", stats, ["metric", "value"])
    f = lambda v: "-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
    sent = int(summary.get("sent") or 0)
    successful = int(summary.get("successful") or 0)
    lines = [f"# {summary['case']} 测试报告", "",
             *stats_lines(sent, successful, sent - successful, 0, 0, summary.get("success_rate"),
                          [f"本轮结论：{'通过' if summary['admission_passed'] else '未通过/数据不足'}",
                           f"请求上限：{summary.get('attempt_limit') or '按固定阶段/基线执行'}；成功目标：{summary.get('success_target') or '按固定阶段/基线执行'}",
                           f"前置连通性：{'通过' if (summary.get('connectivity') or {}).get('passed') else '失败'}",
                           f"负载画像：{summary.get('workload_profile', 'mixed-business-v1')}；Thinking：{summary.get('reasoning_effort') or '供应商默认'}",
                           "范围：合成业务混合负载的短时准入测试，不能替代真实流量回放和 30～60 分钟稳定性测试。"]),
             "", "## 指标", "",
             "| 指标 | 数值 |", "|---|---:|",
             f"| Prompt Token P50 | {f(summary.get('prompt_tokens_p50'))} |",
             f"| TTFT P50/P90/P99（秒） | {f(summary.get('ttft_p50_seconds'))} / {f(summary.get('ttft_p90_seconds'))} / {f(summary.get('ttft_p99_seconds'))} |",
             f"| TTFC P50/P90/P99（秒） | {f(summary.get('ttfc_p50_seconds'))} / {f(summary.get('ttfc_p90_seconds'))} / {f(summary.get('ttfc_p99_seconds'))} |",
             f"| 首 SSE P50（秒） | {f(summary.get('first_sse_p50_seconds'))} |",
             f"| 首 reasoning P50（秒） | {f(summary.get('first_reasoning_p50_seconds'))} |",
             f"| TPOT P90（毫秒/token） | {f(summary.get('tpot_p90_ms_per_token'))} |",
             f"| E2E P90（秒） | {f(summary.get('e2e_p90_seconds'))} |",
             f"| 发送延迟 P99（毫秒） | {f(summary.get('send_delay_p99_ms'))} |", "", "## 判定", "",
             "| 检查项 | 通过 | 详情 |", "|---|---|---|"]
    for name, item in summary.get("checks", {}).items():
        details = ", ".join(f"{k}={f(v)}" for k, v in item.items() if k != "passed")
        lines.append(f"| {name} | {'是' if item.get('passed') else '否'} | {details} |")
    if windows:
        lines += ["", "## 容量窗口", "", "| 阶段/窗口 | 成功 RPM | 成功 TPM | 失败 | 发送延迟P99(ms) | 在途增长/秒 | 通过 |", "|---|---:|---:|---:|---:|---:|---|"]
        for w in windows:
            lines.append(f"| {w['phase']}/{w['index']} | {w['successful_rpm']:.1f} | {w['successful_tpm']:.0f} | {w['failed']} | {f(w['send_delay_p99_ms'])} | {f(w.get('inflight_growth_per_second'))} | {'是' if w['passed'] else '否'} |")
    lines += ["", "> TTFT 为首个有效模型 Token：Thinking 模型通常是首个 reasoning Token；非 Thinking 模型回退为首个可见正文 Token。TTFC 始终是首个可见正文 Token。",
              "", "## 错误", "", json.dumps(summary.get("errors", {}), ensure_ascii=False), "",
              "> TPOT 优先使用可见输出 token；若接口只返回合并 completion usage，结果会标记 `completion_including_reasoning`。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
