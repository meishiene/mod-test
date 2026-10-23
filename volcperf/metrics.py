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
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _check(value: Any, *, maximum: float | None = None, minimum: float | None = None) -> dict[str, Any]:
    passed = value is not None
    if maximum is not None:
        passed = passed and value <= maximum
    if minimum is not None:
        passed = passed and value >= minimum
    return {"value": value, "maximum": maximum, "minimum": minimum, "passed": bool(passed)}


def summarize(case: Any, rows: list[dict[str, Any]], baseline_p90: float | None = None) -> dict[str, Any]:
    measured = [r for r in rows if r.get("counted")]
    success = [r for r in measured if r.get("success")]
    prompts = [float(r["prompt_tokens"]) for r in success if r.get("prompt_tokens") is not None]
    completions = [float(r["completion_tokens"]) for r in success if r.get("completion_tokens") is not None]
    visible = [float(r["visible_completion_tokens"]) for r in success if r.get("visible_completion_tokens") is not None]
    tpot_counts = [float(r["visible_completion_tokens"] if r.get("visible_completion_tokens") is not None else r["completion_tokens"]) for r in success if r.get("visible_completion_tokens") is not None or r.get("completion_tokens") is not None]
    ttfc = [float(r["ttfc_seconds"]) for r in success if r.get("ttfc_seconds") is not None]
    # TTFT 是首个有效模型 token：Thinking 模型取首 reasoning，非 Thinking 模型回退到首可见正文；不使用可能为空的首 SSE。
    ttft = [float(r.get("ttft_seconds") if r.get("ttft_seconds") is not None else
                  min(v for v in (r.get("first_reasoning_seconds"), r.get("ttfc_seconds")) if v is not None))
            for r in success if r.get("ttft_seconds") is not None or r.get("first_reasoning_seconds") is not None or r.get("ttfc_seconds") is not None]
    first_sse = [float(r["first_sse_seconds"]) for r in success if r.get("first_sse_seconds") is not None]
    first_reasoning = [float(r["first_reasoning_seconds"]) for r in success if r.get("first_reasoning_seconds") is not None]
    e2e = [float(r["e2e_seconds"]) for r in success if r.get("e2e_seconds") is not None]
    tpots = [float(r["tpot_ms_per_visible_token"]) for r in success if r.get("tpot_ms_per_visible_token") is not None and (r.get("visible_completion_tokens") if r.get("visible_completion_tokens") is not None else r.get("completion_tokens", 0)) >= 128]
    delays = [float(r["send_delay_ms"]) for r in measured if r.get("send_delay_ms") is not None]
    errors = Counter(r.get("error_type") or f"http_{r.get('http_status')}" for r in measured if not r.get("success"))
    success_rate = len(success) / len(measured) if measured else 0.0
    usage_rate = sum(r.get("prompt_tokens") is not None and r.get("completion_tokens") is not None for r in success) / len(success) if success else 0.0
    in_range = sum(case.prompt_min <= value <= case.prompt_max for value in prompts)
    input_rate = in_range / len(prompts) if prompts else 0.0
    prompt_p50 = percentile(prompts, .5)
    p50_in_target = prompt_p50 is not None and case.target_prompt_tokens * .95 <= prompt_p50 <= case.target_prompt_tokens * 1.05
    truncation_rate = sum(r.get("finish_reason") == "length" for r in success) / len(success) if success else 0.0
    metrics = {
        "prompt_tokens_p50": prompt_p50, "prompt_tokens_p90": percentile(prompts, .9),
        "completion_tokens_p50": percentile(completions, .5), "visible_tokens_p50": percentile(visible, .5),
        "first_sse_p50_seconds": percentile(first_sse, .5), "first_reasoning_p50_seconds": percentile(first_reasoning, .5),
        "ttft_p50_seconds": percentile(ttft, .5), "ttft_p90_seconds": percentile(ttft, .9), "ttft_p99_seconds": percentile(ttft, .99),
        "ttfc_p50_seconds": percentile(ttfc, .5), "ttfc_p90_seconds": percentile(ttfc, .9), "ttfc_p99_seconds": percentile(ttfc, .99),
        "tpot_p90_ms_per_token": percentile(tpots, .9), "e2e_p90_seconds": percentile(e2e, .9),
        "send_delay_p99_ms": percentile(delays, .99),
    }
    checks: dict[str, Any] = {}
    if case.name == "T0":
        checks["exactly_10_successes"] = _check(len(success), minimum=10, maximum=10)
    elif case.name.startswith("B1"):
        checks["error_rate"] = _check(1 - success_rate, maximum=0)
    elif case.name == "RANDOM_LOAD":
        checks["exact_request_count"] = _check(len(measured), minimum=case.samples, maximum=case.samples)
        checks["success_rate"] = _check(success_rate, minimum=.99)
    elif case.name.startswith("L"):
        checks["sample_count"] = _check(len(measured), minimum=case.samples)
        checks["success_rate"] = _check(success_rate, minimum=.99)
        checks["ttfc_p50"] = _check(metrics["ttfc_p50_seconds"], maximum=case.ttfc_p50_limit)
        checks["ttfc_p90"] = _check(metrics["ttfc_p90_seconds"], maximum=case.ttfc_p90_limit)
        checks["tpot_p90"] = _check(metrics["tpot_p90_ms_per_token"], maximum=case.tpot_p90_limit)
        if case.name in {"L1", "L2", "L3", "L4"}:
            checks["baseline_available"] = {"value": baseline_p90, "passed": baseline_p90 is not None}
            if baseline_p90 is not None:
                checks["relative_ttfc_degradation"] = _check(metrics["ttfc_p90_seconds"], maximum=baseline_p90 * 3)
    elif case.name == "O1":
        # 与 L1–L5 同口径：要求样本数即 case.samples（别写死 50，抬高档位要求时会被静默忽略）。
        checks["sample_count"] = _check(len(measured), minimum=case.samples)
        checks["success_rate"] = _check(success_rate, minimum=.99)
        checks["ttfc_p90"] = _check(metrics["ttfc_p90_seconds"], maximum=case.ttfc_p90_limit)
        checks["tpot_p90"] = _check(metrics["tpot_p90_ms_per_token"], maximum=case.tpot_p90_limit)
        checks["e2e_p90"] = _check(metrics["e2e_p90_seconds"], maximum=case.e2e_p90_limit)
        output_in_range = sum(case.min_visible_tokens <= value <= (case.max_visible_tokens or value) for value in visible) / len(visible) if visible else 0
        checks["visible_usage_complete_rate"] = _check(len(visible) / len(success) if success else 0, minimum=1)
        checks["visible_output_400_700_rate"] = _check(output_in_range, minimum=.95)
    workload_counts = dict(Counter(str(r.get("workload_type")) for r in measured if r.get("workload_type")))
    common_quality = {
        "usage_complete_rate": usage_rate, "input_tokens_in_range_rate": input_rate,
        "workload_distribution": workload_counts,
        "prompt_p50_in_target_plus_minus_5pct": p50_in_target, "truncation_rate": truncation_rate,
        "tpot_valid_samples": len(tpots), "tpot_basis": sorted(set(r.get("tpot_token_basis") for r in success if r.get("tpot_token_basis"))),
    }
    if case.name in {"L1", "L2", "L3", "L4", "L5", "O1", "RANDOM_LOAD"}:
        checks["usage_complete_rate"] = _check(usage_rate, minimum=1)
        checks["input_tokens_in_range_rate"] = _check(input_rate, minimum=.95)
        checks["truncation_rate"] = _check(truncation_rate, maximum=.05)
        if case.name != "RANDOM_LOAD":
            checks["prompt_p50_target"] = {"value": prompt_p50, "passed": p50_in_target}
    windows: list[dict[str, Any]] = []
    if case.name in {"C0", "C1"}:
        for phase, rps, duration, counted in case.rps_stages:
            phase_rows = [r for r in measured if r.get("phase") == phase]
            if not phase_rows:
                continue
            start = min(r["planned_epoch"] for r in phase_rows if r.get("planned_epoch") is not None)
            window_count = duration // 60 if case.name == "C1" else 1
            width = 60 if case.name == "C1" else duration
            for index in range(window_count):
                begin, end = start + index * width, start + (index + 1) * width
                selected = [r for r in phase_rows if r.get("planned_epoch") is not None and begin <= r["planned_epoch"] < end]
                good = [r for r in selected if r.get("success") and r.get("total_tokens") is not None]
                delays_w = [r["send_delay_ms"] for r in selected if r.get("send_delay_ms") is not None]
                rpm = len(good) * 60 / width
                tpm = sum(r["total_tokens"] for r in good) * 60 / width
                # 每秒重建在途量并计算后半段线性斜率；正斜率过大表示持续积压。
                samples = []
                for second in range(max(2, width // 2), width + 1):
                    at = begin + second
                    samples.append(sum((r.get("started_epoch") or 0) <= at < (r.get("completed_epoch") or float("inf")) for r in selected))
                n = len(samples)
                xmean = (n - 1) / 2 if n else 0
                ymean = sum(samples) / n if n else 0
                denom = sum((i - xmean) ** 2 for i in range(n))
                inflight_slope = sum((i - xmean) * (v - ymean) for i, v in enumerate(samples)) / denom if denom else 0
                scheduling_ok = (percentile(delays_w, .99) or float("inf")) <= 100
                inflight_ok = inflight_slope <= 0.10
                target_ok = rpm >= 600 and tpm >= 6_000_000 if case.name == "C1" else True
                passed = scheduling_ok and inflight_ok and target_ok
                windows.append({"phase": phase, "index": index + 1, "start_utc": datetime.fromtimestamp(begin, timezone.utc).isoformat(), "successful_rpm": rpm, "successful_tpm": tpm, "sent": len(selected), "failed": len(selected) - len(good), "send_delay_p99_ms": percentile(delays_w, .99), "inflight_growth_per_second": inflight_slope, "passed": passed})
        checks["capacity_windows"] = {"value": len(windows), "passed": bool(windows) and all(w["passed"] for w in windows)}
    baselines_by_dataset: dict[str, Any] = {}
    if case.name == "B1":
        for dataset in ("I1", "I2", "I3", "I4"):
            dataset_rows = [r for r in measured if str(r.get("dataset_id", "")).startswith(f"{dataset}-")]
            dataset_success = [r for r in dataset_rows if r.get("success")]
            dataset_ttfc = [float(r["ttfc_seconds"]) for r in dataset_success if r.get("ttfc_seconds") is not None]
            baselines_by_dataset[dataset] = {
                "sent": len(dataset_rows), "successful": len(dataset_success),
                "ttfc_p50_seconds": percentile(dataset_ttfc, .5),
                "ttfc_p90_seconds": percentile(dataset_ttfc, .9),
            }
        checks["four_requests_per_size"] = {
            "value": {name: item["sent"] for name, item in baselines_by_dataset.items()},
            "passed": all(item["sent"] == 4 for item in baselines_by_dataset.values()),
        }
    overall = bool(checks) and all(item["passed"] for item in checks.values())
    return {"case": case.name, "sent": len(measured), "successful": len(success), "success_rate": success_rate,
            "errors": dict(errors), **metrics, "data_quality": common_quality, "checks": checks,
            "capacity_windows": windows, "baselines_by_dataset": baselines_by_dataset, "admission_passed": overall,
            "scope_note": "短时准入测试，不能替代30-60分钟稳定性测试。"}
