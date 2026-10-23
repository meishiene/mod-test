from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from dataclasses import replace
from itertools import cycle
from pathlib import Path

from common.audit import AuditLog
from common.suite_runtime import finalize
from typing import Any

import httpx

from .cases import CASES, Case
from .client import stream_chat
from .config import ApiConfig
from .connectivity import check_connectivity
from .data import PromptRecord, generate_random_prompts, load_prompts
from .metrics import summarize
from .protocols import REASONING_EFFORT_PROTOCOLS
from .reporting import write_artifacts


B1_DATASETS = ("I1", "I2", "I3", "I4")


def effective_reasoning_effort(case: Case) -> str | None:
    """本次运行实际生效的思考档位；None 表示请求体不发该字段（走供应商默认）。

    优先级：环境变量 `VOLC_REASONING_EFFORT`（非空）> `Case.reasoning_effort`。
    协议不支持时一律 None —— 免得把"根本没发出去的值"写进 metadata/summary 和报告。
    """
    config = ApiConfig.from_env()
    if config.protocol not in REASONING_EFFORT_PROTOCOLS:
        return None
    return config.reasoning_effort or case.reasoning_effort


def request_config(case: Case) -> ApiConfig:
    """请求侧配置：在环境配置之上套用用例声明的思考档位。

    与前置连通性探针同口径（探针固定发 reasoning_effort=low）。不同口径会让
    "探针能过、正式请求大面积 length 截断"再次发生。
    """
    return replace(ApiConfig.from_env(), reasoning_effort=effective_reasoning_effort(case))


def load_case_prompts(
    case: Case,
    data_override: Path | None = None,
    *,
    quick: bool = False,
    samples: int | None = None,
) -> list[PromptRecord]:
    """B1 每档取固定数量并交错排列，确保并发 2 时各规格均匀执行。"""
    if case.name == "RANDOM_LOAD" and data_override is None:
        return generate_random_prompts(samples or case.samples)
    if case.name != "B1" or data_override is not None:
        return load_prompts(data_override or case.data_file)
    per_size = 1 if quick else 4
    groups = [load_prompts(Path(__file__).resolve().parent.parent / "data" / f"{dataset}.jsonl.gz")[:per_size]
              for dataset in B1_DATASETS]
    return [groups[size][index] for index in range(per_size) for size in range(len(groups))]


class ResultWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.audit = AuditLog(path)

    def write(self, row: dict[str, Any]) -> None:
        self.audit.write(row)

    def close(self) -> None:
        self.audit.close()


def session(config: ApiConfig, *, connection_limit: int | None = None) -> httpx.AsyncClient:
    # 连接数按实际并发设置上限，避免默认 1200 个连接在高并发时造成本机
    # socket/端口竞争和中转层突发建连；连接可复用，保持连接数略高于并发即可。
    limit = connection_limit or config.max_connections
    timeout = httpx.Timeout(config.request_timeout_seconds, connect=config.connect_timeout_seconds, pool=10.0)
    limits = httpx.Limits(max_connections=limit, max_keepalive_connections=limit, keepalive_expiry=60.0)
    return httpx.AsyncClient(timeout=timeout, limits=limits, verify=config.verify_ssl, trust_env=config.trust_env,
                             headers={"Accept": "text/event-stream"}, http2=False)


def _build_request_plan_item(
    case: Case,
    record: PromptRecord,
    sequence: int,
    *,
    phase: str | None,
    counted: bool | None,
    planned_offset_seconds: float | None,
) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "case": case.name,
        "sequence": sequence,
        "request_id": f"{case.name}-{sequence}-{uuid.uuid4().hex[:8]}",
        "dataset_id": record.id,
        "target_prompt_tokens": getattr(record, "target_prompt_tokens", case.target_prompt_tokens),
        "max_tokens": case.max_tokens,
        "phase": phase,
        "counted": counted,
        "planned_offset_seconds": planned_offset_seconds,
        "workload": None,
        "plan_error_type": None,
        "plan_error_message": None,
    }
    try:
        plan["workload"] = record.workload(case.name == "O1")
    except Exception as exc:
        # One malformed source record becomes one failed request plan. It must
        # not prevent the remaining plans from being generated or executed.
        plan["plan_error_type"] = f"plan_{type(exc).__name__}"
        plan["plan_error_message"] = str(exc)
    return plan


def build_closed_request_plan(case: Case, prompts: list[PromptRecord], samples: int) -> list[dict[str, Any]]:
    """Materialize every closed-loop request before opening the HTTP client."""
    prompt_iter = cycle(prompts)
    return [
        _build_request_plan_item(
            case,
            next(prompt_iter),
            sequence,
            phase=None,
            counted=None,
            planned_offset_seconds=None,
        )
        for sequence in range(1, samples + 1)
    ]


def build_open_request_plan(
    case: Case,
    prompts: list[PromptRecord],
    stages: tuple[tuple[str, float, int, bool], ...],
) -> list[dict[str, Any]]:
    """Materialize every open-loop request and its relative send schedule."""
    prompt_iter = cycle(prompts)
    plans: list[dict[str, Any]] = []
    stage_offset = 0.0
    sequence = 1
    for phase, rps, duration, counted in stages:
        interval = 1 / rps
        for index in range(round(rps * duration)):
            plans.append(
                _build_request_plan_item(
                    case,
                    next(prompt_iter),
                    sequence,
                    phase=phase,
                    counted=counted,
                    planned_offset_seconds=stage_offset + index * interval,
                )
            )
            sequence += 1
        stage_offset += duration
    return plans


def _failed_request_row(
    config: ApiConfig,
    plan: dict[str, Any],
    *,
    phase: str,
    counted: bool,
    planned_epoch: float | None,
    started_epoch: float,
    started_mono: float,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    workload = plan.get("workload")
    completed_epoch = time.time()
    return {
        "case": plan.get("case"),
        "sequence": plan.get("sequence"),
        "protocol": config.protocol,
        "phase": phase,
        "counted": counted,
        "request_id": plan.get("request_id"),
        "dataset_id": plan.get("dataset_id"),
        "target_prompt_tokens": plan.get("target_prompt_tokens"),
        "max_tokens": plan.get("max_tokens"),
        "planned_epoch": planned_epoch,
        "started_epoch": started_epoch,
        "completed_epoch": completed_epoch,
        "send_delay_ms": max(0.0, (started_epoch - planned_epoch) * 1000) if planned_epoch is not None else None,
        "success": False,
        "transport_status": "error",
        "http_status": None,
        "error_type": error_type,
        "error_message": error_message,
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "visible_completion_tokens": None,
        "total_tokens": None,
        "finish_reason": None,
        "first_sse_seconds": None,
        "first_reasoning_seconds": None,
        "ttft_seconds": None,
        "ttfc_seconds": None,
        "e2e_seconds": time.perf_counter() - started_mono,
        "response_headers_seconds": None,
        "tpot_ms_per_visible_token": None,
        "content_chunk_count": 0,
        "reasoning_chunk_count": 0,
        "content_char_count": 0,
        "sse_event_count": 0,
        "invalid_sse_event_count": 0,
        "saw_done": False,
        "response_request_id": None,
        "upstream_request_id": None,
        "response_object_id": None,
        "response_headers": {},
        "raw_response_body": "",
        "raw_sse_lines": [],
        "raw_sse_events": [],
        "workload_type": getattr(workload, "workload_type", None),
        "message_count": len(getattr(workload, "messages", []) or []),
        "target_visible_tokens": getattr(workload, "target_visible_tokens", None),
    }


async def execute_request_plan(
    client: httpx.AsyncClient,
    config: ApiConfig,
    plan: dict[str, Any],
    *,
    phase: str,
    counted: bool,
    planned_epoch: float | None,
) -> dict[str, Any]:
    """Execute one plan and convert every ordinary per-request exception to evidence."""
    started_epoch, started_mono = time.time(), time.perf_counter()
    error_type = plan.get("plan_error_type")
    if error_type:
        return _failed_request_row(
            config,
            plan,
            phase=phase,
            counted=counted,
            planned_epoch=planned_epoch,
            started_epoch=started_epoch,
            started_mono=started_mono,
            error_type=str(error_type),
            error_message=str(plan.get("plan_error_message") or "请求计划生成失败"),
        )
    try:
        workload = plan["workload"]
        # httpx 的 read timeout 是“无数据时长”，外层 timeout 才是请求总时长。
        async with asyncio.timeout(config.request_timeout_seconds):
            row = await stream_chat(
                client,
                config,
                request_id=str(plan["request_id"]),
                messages=workload.messages,
                workload_type=workload.workload_type,
                target_visible_tokens=workload.target_visible_tokens,
                max_tokens=int(plan["max_tokens"]),
                case_name=str(plan["case"]),
                dataset_id=str(plan["dataset_id"]),
                phase=phase,
                counted=counted,
                planned_epoch=planned_epoch,
            )
        if not isinstance(row, dict):
            raise TypeError(f"请求执行结果必须是 dict，实际为 {type(row).__name__}")
        row["sequence"] = plan["sequence"]
        row["target_prompt_tokens"] = plan.get("target_prompt_tokens")
        row["max_tokens"] = plan.get("max_tokens")
        return row
    except TimeoutError as exc:
        request_error_type = "request_timeout"
        request_error_message = str(exc) or f"请求总时长超过 {config.request_timeout_seconds:g} 秒"
    except Exception as exc:
        request_error_type = f"internal_{type(exc).__name__}"
        request_error_message = str(exc)
    return _failed_request_row(
        config,
        plan,
        phase=phase,
        counted=counted,
        planned_epoch=planned_epoch,
        started_epoch=started_epoch,
        started_mono=started_mono,
        error_type=request_error_type,
        error_message=request_error_message,
    )


async def store_request_row(
    writer: ResultWriter,
    rows: list[dict[str, Any]],
    lock: asyncio.Lock,
    row: dict[str, Any],
    counters: dict[str, int],
) -> tuple[int, int]:
    """Keep the worker alive even if incremental evidence persistence fails."""
    async with lock:
        try:
            writer.write(row)
        except Exception as exc:
            row["evidence_write_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        counters["completed"] += 1
        counters["measured"] += int(bool(row.get("counted")))
        return counters["completed"], counters["measured"]


async def run_closed(case: Case, prompts: list[PromptRecord], writer: ResultWriter, samples: int, warmup: int, duration: int) -> list[dict[str, Any]]:
    del duration  # 闭环压测严格按预生成的请求总数执行。
    request_plan = build_closed_request_plan(case, prompts, samples)
    config, rows = request_config(case), []
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for plan in request_plan:
        queue.put_nowait(plan)
    rows_lock = asyncio.Lock()
    counters = {"completed": 0, "measured": 0}
    warmup_end = time.monotonic() + warmup
    async with session(config, connection_limit=max(case.concurrency * 2, 20)) as client:
        async def worker(worker_id: int) -> None:
            del worker_id
            while True:
                try:
                    plan = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    counted = time.monotonic() >= warmup_end
                    phase = "measured" if counted else "warmup"
                    row = await execute_request_plan(
                        client,
                        config,
                        plan,
                        phase=phase,
                        counted=counted,
                        planned_epoch=None,
                    )
                    completed, measured = await store_request_row(writer, rows, rows_lock, row, counters)
                    if counted:
                        try:
                            print(
                                f"\r{case.name}: completed {measured}/{samples}, "
                                f"total {completed}/{samples}",
                                end="",
                                flush=True,
                            )
                        except Exception:
                            pass
                finally:
                    queue.task_done()
        await asyncio.gather(*(asyncio.create_task(worker(i)) for i in range(case.concurrency)))
    print()
    return sorted(rows, key=lambda row: int(row.get("sequence") or 0))


async def run_open(case: Case, prompts: list[PromptRecord], writer: ResultWriter, stages: tuple[tuple[str, float, int, bool], ...]) -> list[dict[str, Any]]:
    request_plan = build_open_request_plan(case, prompts, stages)
    config, rows = request_config(case), []
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for plan in request_plan:
        queue.put_nowait(plan)
    rows_lock = asyncio.Lock()
    counters = {"completed": 0, "measured": 0}
    for phase, rps, duration, _ in stages:
        print(f"{case.name}: {phase}, {rps:g} RPS, {duration}s")
    async with session(config, connection_limit=max(config.max_connections, 20)) as client:
        started_mono, started_epoch = time.monotonic(), time.time()

        async def worker() -> None:
            while True:
                try:
                    plan = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    planned_offset = float(plan["planned_offset_seconds"])
                    delay = started_mono + planned_offset - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    row = await execute_request_plan(
                        client,
                        config,
                        plan,
                        phase=str(plan["phase"]),
                        counted=bool(plan["counted"]),
                        planned_epoch=started_epoch + planned_offset,
                    )
                    completed, _ = await store_request_row(writer, rows, rows_lock, row, counters)
                    if completed % max(1, min(100, len(request_plan) // 20 or 1)) == 0:
                        try:
                            print(
                                f"\r{case.name}: completed {completed}/{len(request_plan)}",
                                end="",
                                flush=True,
                            )
                        except Exception:
                            pass
                finally:
                    queue.task_done()

        worker_count = min(config.max_connections, len(request_plan))
        if worker_count:
            await asyncio.gather(*(asyncio.create_task(worker()) for _ in range(worker_count)))
    print()
    return sorted(rows, key=lambda row: int(row.get("sequence") or 0))


def metadata(path: Path, case: Case, args: argparse.Namespace) -> None:
    cfg = request_config(case)
    obj = {"case": case.name, "started_epoch": time.time(), "endpoint": cfg.endpoint, "model": cfg.model,
           "max_tokens": case.max_tokens, "stream": True,
           "thinking": {"provider_default": cfg.reasoning_effort is None, "reasoning_effort": cfg.reasoning_effort},
           "workload_profile": "mixed-business-v1", "mode": case.mode,
           "arguments": vars(args), "python": os.sys.version, "pid": os.getpid()}
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="火山引擎 Kimi K3 短时准入性能测试")
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--samples", type=int)
    parser.add_argument("--warmup-seconds", type=int)
    parser.add_argument("--measured-seconds", type=int)
    parser.add_argument("--baseline-summary", type=Path, help="对应 B1 summary.json，用于相对退化检查")
    parser.add_argument("--quick", action="store_true", help="仅验证脚本，不能形成准入结论")
    parser.add_argument("--concurrency", type=int, help="覆盖默认并发数（仅闭环性能测试）")
    args = parser.parse_args(argv)
    case = CASES[args.case]
    if args.concurrency is not None and (args.concurrency < 1 or args.concurrency > 1000):
        parser.error("--concurrency 必须在 1～1000 之间")
    if args.samples is not None and (args.samples < 1 or args.samples > 100000):
        parser.error("--samples 必须在 1～100000 之间")
    planned_samples = min(args.samples or case.samples, 4) if args.quick else (args.samples or case.samples)
    prompts = load_case_prompts(case, args.data, quick=args.quick, samples=planned_samples)
    out = args.output_root / f"{case.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False); metadata(out / "metadata.json", case, args)
    connectivity = await asyncio.to_thread(check_connectivity, ApiConfig.from_env())
    (out / "connectivity.json").write_text(json.dumps(connectivity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not connectivity["passed"]:
        summary = {"case": case.name, "sent": 0, "successful": 0, "success_rate": 0,
                   "connectivity": connectivity, "checks": {"connectivity": connectivity},
                   "errors": {"connectivity_failed": 1}, "admission_passed": False,
                   "scope_note": "前置连通性测试失败，正式测试未启动。"}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_artifacts(out, [], summary)
        finalize(out)
        print(json.dumps(summary, ensure_ascii=False, indent=2)); print(f"结果目录: {out.resolve()}")
        return 2
    writer = ResultWriter(out / "requests.jsonl")
    try:
        if case.mode == "closed":
            samples = len(prompts) if case.name == "B1" and args.samples is None else (min(args.samples or case.samples, 4) if args.quick else (args.samples or case.samples))
            warmup = 0 if args.quick else (case.warmup_seconds if args.warmup_seconds is None else args.warmup_seconds)
            # 用户显式指定 --samples 时按“精确发送请求数”执行，不再叠加默认 measured_seconds。
            # 否则 L1/L2 等默认轮次会继续运行到时长结束，导致请求数超过界面填写值。
            duration = 0 if args.quick or args.samples is not None else (case.measured_seconds if args.measured_seconds is None else args.measured_seconds)
            case_overrides: dict[str, Any] = {}
            if args.concurrency is not None:
                case_overrides["concurrency"] = args.concurrency
            if case.name == "RANDOM_LOAD":
                case_overrides["samples"] = samples
            if case_overrides:
                case = case.__class__(**{**case.__dict__, **case_overrides})
            rows = await run_closed(case, prompts, writer, samples, warmup, duration)
        else:
            stages = (("quick", 1.0, 3, True),) if args.quick else case.rps_stages
            rows = await run_open(case, prompts, writer, stages)
    finally:
        writer.close()
    baseline = None
    if args.baseline_summary:
        baseline_summary = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
        dataset_baseline = (baseline_summary.get("baselines_by_dataset") or {}).get(case.dataset, {})
        baseline = dataset_baseline.get("ttfc_p90_seconds", baseline_summary.get("ttfc_p90_seconds"))
    summary = summarize(case, rows, baseline)
    summary["attempt_limit"] = samples if case.mode == "closed" else None
    summary["request_limit_scope"] = "预热与正式请求合计" if case.mode == "closed" else "按固定速率阶段"
    summary["success_target"] = samples if case.mode == "closed" else None
    env_config = ApiConfig.from_env()
    summary.update({"schema_version": "volcperf.result.v1", "record_id": os.getenv("VOLC_RECORD_ID"),
                    "batch_id": os.getenv("VOLC_BATCH_ID") or None, "test_kind": "load",
                    "protocol": env_config.protocol, "base_url": env_config.base_url,
                    "model": env_config.model,
                    "workload_profile": "mixed-business-v1",
                    "reasoning_effort": effective_reasoning_effort(case)})
    summary["connectivity"] = connectivity
    summary.setdefault("checks", {})["connectivity"] = connectivity
    summary["admission_passed"] = summary["admission_passed"] and connectivity["passed"]
    if args.quick:
        summary["quick_run"] = True; summary["admission_passed"] = False; summary["scope_note"] = "quick 模式仅验证脚本，不得作为准入结论。"
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_artifacts(out, rows, summary)
    finalize(out)
    print(json.dumps(summary, ensure_ascii=False, indent=2)); print(f"结果目录: {out.resolve()}")
    return 0 if summary["admission_passed"] else 2


def cli() -> None:
    try:
        raise SystemExit(asyncio.run(main_async()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except ValueError as exc:
        print(f"配置错误: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
