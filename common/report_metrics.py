"""请求级与用例级指标的统一计算口径。

Web 聚合报告与各单项测试生成的 report.md 共用这里的实现，避免出现两套尺子：
- `suite_request_metrics`：从 requests.jsonl（执行证据）还原请求数、HTTP 成功率与延迟 P90。
- `build_item_metrics`：优先取 summary 自带的压测字段，缺口用执行证据补齐，并统计用例通过数。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * 0.9 + 0.999) - 1)]


def suite_request_metrics(result_dir: str | Path | None) -> dict[str, Any] | None:
    """从子记录的 requests.jsonl（执行证据）统计请求级指标。

    能力专项/专项套件（fable、kvv、deepseek 等）的 summary 只有用例级结论，没有
    压测字段；但每个请求都会落盘 `timing`（ttft_ms / ttfc_ms / e2e_ms），足以还原
    正式请求数、HTTP 成功率与延迟 P90，不再在报告里显示一排 "-"。
    """
    if not result_dir:
        return None
    path = Path(result_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    evidence = path / "requests.jsonl"
    if not evidence.is_file():
        return None
    total = http_ok = 0
    ttft: list[float] = []
    ttfc: list[float] = []
    e2e: list[float] = []
    for line in evidence.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        total += 1
        status = row.get("http_status")
        if isinstance(status, int) and 200 <= status < 300 and not row.get("transport_error"):
            http_ok += 1
        timing = row.get("timing") or {}
        for key, bucket in (("ttft_ms", ttft), ("ttfc_ms", ttfc), ("e2e_ms", e2e)):
            value = timing.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                bucket.append(float(value))
    if not total:
        return None
    return {
        "requests": total,
        "http_success": http_ok,
        "success_rate": http_ok / total,
        "ttft_p90_seconds": p90(ttft) / 1000 if ttft else None,
        "ttfc_p90_seconds": p90(ttfc) / 1000 if ttfc else None,
        "e2e_p90_seconds": p90(e2e) / 1000 if e2e else None,
        "ttft_samples": len(ttft),
        "ttfc_samples": len(ttfc),
        "source": "requests.jsonl",
    }


def _num(value: Any) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def build_item_metrics(summary: dict[str, Any], result_dir: str | Path | None) -> dict[str, Any]:
    """汇总一个记录（单项或批次子项）的指标：优先 summary 压测字段，缺口用执行证据补齐。"""
    summary = summary or {}
    metrics: dict[str, Any] = {
        "requests": _num(summary.get("sent")),
        "success_rate": _num(summary.get("success_rate")),
        "ttft_p90_seconds": _num(summary.get("ttft_p90_seconds")),
        "ttfc_p90_seconds": _num(summary.get("ttfc_p90_seconds")),
        "tpot_p90_ms_per_token": _num(summary.get("tpot_p90_ms_per_token")),
        "e2e_p90_seconds": _num(summary.get("e2e_p90_seconds")),
        "source": "summary",
    }
    computed = suite_request_metrics(result_dir)
    if computed:
        for key in ("requests", "success_rate", "ttft_p90_seconds", "ttfc_p90_seconds", "e2e_p90_seconds"):
            if metrics[key] is None and computed.get(key) is not None:
                metrics[key] = computed[key]
        if metrics["source"] == "summary" and computed.get("e2e_p90_seconds") is not None:
            metrics["source"] = "summary+requests.jsonl"
        elif metrics["source"] != "summary+requests.jsonl":
            metrics["source"] = computed["source"]
    results = summary.get("results") or []
    if results:
        statuses = [str(r.get("status") or ("passed" if r.get("passed") is True else "failed")) for r in results if isinstance(r, dict)]
        metrics["case_total"] = len(statuses)
        metrics["case_passed"] = sum(1 for s in statuses if s == "passed")
        metrics["case_warning"] = sum(1 for s in statuses if s == "warning")
        metrics["case_blocked"] = sum(1 for s in statuses if s == "blocked")
        metrics["case_failed"] = sum(1 for s in statuses if s in {"failed", "error"})
    return metrics
