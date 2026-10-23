from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def summarize(case: Any, rows: list[dict[str, Any]], steady_start_epoch: float | None = None) -> dict[str, Any]:
    measured = [r for r in rows if r.get("counted")]
    successful = [r for r in measured if r.get("success")]
    ttfts = [r["ttft_seconds"] for r in successful if r.get("ttft_seconds") is not None]
    e2es = [r["e2e_seconds"] for r in successful if r.get("e2e_seconds") is not None]
    tpots = [r["tpot_ms_per_token"] for r in successful if r.get("tpot_ms_per_token") is not None]
    prompts = [r["prompt_tokens"] for r in successful if r.get("prompt_tokens") is not None]
    cached = [r["cached_tokens"] for r in successful if r.get("cached_tokens") is not None]
    cache_ratios = [r["cache_ratio"] for r in successful if r.get("cache_ratio") is not None]
    errors = Counter(r.get("error_type") or f"http_{r.get('http_status')}" for r in measured if not r.get("success"))

    metric_values = {
        "ttft_p50_seconds": percentile(ttfts, 0.50),
        "ttft_p90_seconds": percentile(ttfts, 0.90),
        "mean_tpot_ms_per_token": sum(tpots) / len(tpots) if tpots else None,
        "e2e_p90_seconds": percentile(e2es, 0.90),
        "cache_ratio_mean": sum(cache_ratios) / len(cache_ratios) if cache_ratios else None,
        "cache_ratio_p50": percentile(cache_ratios, 0.50),
        "cached_tokens_total": sum(cached) if cached else None,
    }
    checks: dict[str, dict[str, Any]] = {}
    for key, limit in (
        ("ttft_p50_seconds", case.ttft_p50_limit),
        ("ttft_p90_seconds", case.ttft_p90_limit),
        ("mean_tpot_ms_per_token", case.mean_tpot_ms_limit),
        ("e2e_p90_seconds", case.e2e_p90_limit),
    ):
        if limit is not None:
            value = metric_values[key]
            checks[key] = {"value": value, "limit": limit, "passed": value is not None and value <= limit}

    if case.cache_ratio_min is not None:
        value = metric_values["cache_ratio_mean"]
        checks["cache_ratio_mean"] = {
            "value": value, "min": case.cache_ratio_min, "max": case.cache_ratio_max,
            "passed": value is not None and value >= case.cache_ratio_min and (case.cache_ratio_max is None or value <= case.cache_ratio_max),
        }

    windows: list[dict[str, Any]] = []
    if case.name == "C1" and steady_start_epoch is not None:
        for index in range(10):
            start = steady_start_epoch + index * 60
            end = start + 60
            # Primary attribution uses planned arrival time, then keeps only requests that eventually
            # completed successfully. Exactly 600 arrivals are scheduled per minute at 10 RPS; using
            # completion timestamps would make the first window structurally incomplete by one E2E delay.
            selected = [r for r in successful if r.get("planned_epoch") is not None and start <= r["planned_epoch"] < end]
            completed_selected = [r for r in successful if r.get("completed_epoch") is not None and start <= r["completed_epoch"] < end]
            rpm = len(selected)
            tpm = sum((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in selected)
            windows.append({"index": index + 1, "start_utc": _iso(start), "end_utc": _iso(end), "successful_arrivals_rpm": rpm, "successful_arrivals_tpm": tpm, "successful_completions_diagnostic": len(completed_selected), "passed": rpm >= 600 and tpm >= 6_000_000})
        checks["capacity_windows"] = {"passed": len(windows) == 10 and all(w["passed"] for w in windows), "required_rpm": 600, "required_tpm": 6_000_000}

    input_in_range = sum(1 for value in prompts if case.prompt_min <= value <= case.prompt_max)
    data_quality = {
        "usage_complete": len(prompts) == len(successful),
        "prompt_tokens_in_range": input_in_range,
        "prompt_tokens_out_of_range": len(prompts) - input_in_range,
        "tpot_valid_samples": len(tpots),
        "cache_usage_complete": len(cached) == len(successful) if case.cache_ratio_min is not None else None,
        "cache_ratio_valid_samples": len(cache_ratios),
    }
    overall = bool(checks) and all(item["passed"] for item in checks.values()) and data_quality["usage_complete"] and data_quality["prompt_tokens_out_of_range"] == 0
    if case.cache_ratio_min is not None:
        overall = overall and data_quality["cache_usage_complete"]
    if case.name == "T0":
        overall = len(successful) >= 10 and data_quality["usage_complete"]
    return {
        "case": case.name,
        "sent": len(measured),
        "successful": len(successful),
        "success_rate": len(successful) / len(measured) if measured else 0,
        "errors": dict(errors),
        "prompt_tokens_p50": percentile([float(v) for v in prompts], 0.50),
        "prompt_tokens_p90": percentile([float(v) for v in prompts], 0.90),
        **metric_values,
        "data_quality": data_quality,
        "checks": checks,
        "capacity_windows": windows,
        "admission_passed": overall,
        "note": "P1 is confirmed: 600 RPM / 6,000,000 TPM is an actual carrying-capacity target.",
    }
