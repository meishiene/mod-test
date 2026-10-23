from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from common.report_style import stats_lines

REQUEST_COLUMNS = [
    "case", "phase", "counted", "request_id", "dataset_id", "planned_epoch", "started_epoch",
    "completed_epoch", "queue_delay_ms", "success", "http_status", "error_type", "error_message",
    "prompt_tokens", "cached_tokens", "cache_ratio", "completion_tokens", "total_tokens", "ttft_seconds", "e2e_seconds",
    "tpot_ms_per_token", "content_chunk_count", "content_char_count", "sse_event_count",
    "invalid_sse_event_count", "response_request_id", "response_headers",
]


def _flat_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["total_tokens"] = ((row.get("prompt_tokens") or 0) + (row.get("completion_tokens") or 0)) if row.get("prompt_tokens") is not None and row.get("completion_tokens") is not None else None
    if isinstance(result.get("response_headers"), dict):
        result["response_headers"] = json.dumps(result["response_headers"], ensure_ascii=False, separators=(",", ":"))
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    columns = columns or REQUEST_COLUMNS
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_flat_row(row))


def write_artifacts(out: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    write_csv(out / "requests.csv", rows)
    errors = [row for row in rows if not row.get("success")]
    write_csv(out / "errors.csv", errors)

    windows = summary.get("capacity_windows") or []
    if windows:
        write_csv(out / "capacity_windows.csv", windows, list(windows[0].keys()))

    stats_rows: list[dict[str, Any]] = []
    for key in (
        "case", "sent", "successful", "success_rate", "prompt_tokens_p50", "prompt_tokens_p90",
        "ttft_p50_seconds", "ttft_p90_seconds", "mean_tpot_ms_per_token", "e2e_p90_seconds",
        "cached_tokens_total", "cache_ratio_mean", "cache_ratio_p50",
        "admission_passed",
    ):
        stats_rows.append({"metric": key, "value": summary.get(key)})
    for key, value in (summary.get("data_quality") or {}).items():
        stats_rows.append({"metric": f"data_quality.{key}", "value": value})
    for key, value in (summary.get("errors") or {}).items():
        stats_rows.append({"metric": f"errors.{key}", "value": value})
    write_csv(out / "statistics.csv", stats_rows, ["metric", "value"])

    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    sent = int(summary.get("sent") or 0)
    successful = int(summary.get("successful") or 0)
    success_rate = summary.get("success_rate")
    if success_rate is None and sent:
        success_rate = successful / sent
    lines = [
        f"# {summary.get('case')} 测试结果",
        "",
        *stats_lines(sent, successful, sent - successful, 0, 0, success_rate,
                     [f"准入结论：{'通过' if summary.get('admission_passed') else '未通过/数据不足'}"]),
        "",
        "## 核心统计",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| Prompt Token P50 | {fmt(summary.get('prompt_tokens_p50'))} |",
        f"| Prompt Token P90 | {fmt(summary.get('prompt_tokens_p90'))} |",
        f"| TTFT P50（秒） | {fmt(summary.get('ttft_p50_seconds'))} |",
        f"| TTFT P90（秒） | {fmt(summary.get('ttft_p90_seconds'))} |",
        f"| 平均 TPOT（毫秒/Token） | {fmt(summary.get('mean_tpot_ms_per_token'))} |",
        f"| E2E P90（秒） | {fmt(summary.get('e2e_p90_seconds'))} |",
        f"| 缓存 Token 合计 | {fmt(summary.get('cached_tokens_total'))} |",
        f"| 平均缓存率 | {fmt(summary.get('cache_ratio_mean'))} |",
        "",
        "## 错误统计",
        "",
    ]
    errors_summary = summary.get("errors") or {}
    if errors_summary:
        lines += ["| 类型 | 数量 |", "|---|---:|"] + [f"| {key} | {value} |" for key, value in errors_summary.items()]
    else:
        lines.append("无错误。")
    if windows:
        lines += ["", "## C1 分钟窗口", "", "| 窗口 | 成功 RPM | 成功 TPM | 完成数（诊断） | 通过 |", "|---:|---:|---:|---:|---|"]
        for item in windows:
            lines.append(f"| {item['index']} | {item['successful_arrivals_rpm']} | {item['successful_arrivals_tpm']} | {item['successful_completions_diagnostic']} | {'是' if item['passed'] else '否'} |")
    lines += [
        "", "## 文件说明", "",
        "- `requests.jsonl`：实时写入的请求级原始明细。",
        "- `requests.csv`：便于 Excel 查看和筛选的请求明细。",
        "- `errors.csv`：失败请求明细；无失败时仅包含表头。",
        "- `summary.json`：机器可读统计结果和准入判定。",
        "- `statistics.csv`：扁平化统计指标。",
        "- `metadata.json`：运行参数、模型、接口和环境信息。",
    ]
    if windows:
        lines.append("- `capacity_windows.csv`：C1 每个 60 秒窗口统计。")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
