from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import time
import uuid
from itertools import cycle
from pathlib import Path

from common.audit import AuditLog
from common.suite_runtime import finalize
from typing import Any

import httpx

from .cases import CASES, Case
from .client import stream_chat
from .config import ApiConfig
from .data import PromptRecord, load_prompts
from .metrics import summarize
from .reporting import write_artifacts


class ResultWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.audit = AuditLog(path)

    def write(self, row: dict[str, Any]) -> None:
        self.audit.write(row)

    def close(self) -> None:
        self.audit.close()


def _session(config: ApiConfig) -> httpx.AsyncClient:
    timeout = httpx.Timeout(config.request_timeout_seconds, connect=config.connect_timeout_seconds)
    limits = httpx.Limits(max_connections=config.max_connections, max_keepalive_connections=config.max_connections)
    return httpx.AsyncClient(timeout=timeout, limits=limits, verify=config.verify_ssl, headers={"Accept": "text/event-stream"}, http2=False, trust_env=config.trust_env)


async def run_closed(case: Case, prompts: list[PromptRecord], writer: ResultWriter, samples: int, warmup: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    config = ApiConfig.from_env()
    prompt_iter = cycle(prompts)
    counter = 0
    attempts = 0
    max_attempts = max(samples * 2, samples + (case.concurrency or 1))
    lock = asyncio.Lock()
    stop = asyncio.Event()
    warmup_end = time.monotonic() + warmup

    async with _session(config) as session:
        async def worker(worker_id: int) -> None:
            nonlocal counter, attempts
            # 避免并发 worker 在同一时刻形成启动尖峰；这不是降低并发上限。
            if worker_id and config.worker_start_stagger_seconds > 0:
                await asyncio.sleep(worker_id * config.worker_start_stagger_seconds)
            consecutive_failures = 0
            while not stop.is_set():
                counted = time.monotonic() >= warmup_end
                async with lock:
                    if counted and (counter >= samples or attempts >= max_attempts):
                        stop.set()
                        return
                    record = next(prompt_iter)
                    if counted:
                        attempts += 1
                    seq = attempts
                if stop.is_set():
                    return
                row = await stream_chat(session, config, f"{case.name}-{worker_id}-{seq}-{uuid.uuid4().hex[:8]}", record.content, case.max_tokens, case.name, record.id, "measured" if counted else "warmup", counted)
                writer.write(row)
                rows.append(row)
                if not row["success"]:
                    # 失败请求不重试，避免无效请求风暴；仅让该 worker 退避后再发下一次。
                    # 连续失败时指数退避并加少量抖动，401/403/503 等限流或渠道错误会自然降速。
                    consecutive_failures += 1
                    base = config.failure_backoff_seconds
                    cap = max(base, config.failure_backoff_max_seconds)
                    delay = min(cap, base * (2 ** min(consecutive_failures - 1, 10)))
                    await asyncio.sleep(delay * (0.8 + random.random() * 0.4))
                else:
                    consecutive_failures = 0
                if counted and row["success"]:
                    async with lock:
                        counter += 1
                        if counter >= samples:
                            stop.set()
                        print(f"\r{case.name}: successful samples {min(counter, samples)}/{samples}", end="", flush=True)
        tasks = [asyncio.create_task(worker(i)) for i in range(case.concurrency or 1)]
        await asyncio.gather(*tasks)
    print()
    return rows


async def run_open(case: Case, prompts: list[PromptRecord], writer: ResultWriter, stages: tuple[tuple[str, float, int, bool], ...]) -> tuple[list[dict[str, Any]], float | None]:
    rows: list[dict[str, Any]] = []
    tasks: set[asyncio.Task] = set()
    config = ApiConfig.from_env()
    prompt_iter = cycle(prompts)
    steady_start_epoch: float | None = None
    semaphore = asyncio.Semaphore(config.max_connections)

    async with _session(config) as session:
        async def send_one(record: PromptRecord, phase: str, counted: bool, planned_epoch: float, sequence: int) -> None:
            async with semaphore:
                row = await stream_chat(session, config, f"{case.name}-{sequence}-{uuid.uuid4().hex[:8]}", record.content, case.max_tokens, case.name, record.id, phase, counted, planned_epoch)
            writer.write(row)
            rows.append(row)

        sequence = 0
        for phase, rps, duration, counted in stages:
            if counted and steady_start_epoch is None:
                steady_start_epoch = time.time()
            print(f"C1 phase={phase}, rate={rps:g} RPS, duration={duration}s")
            interval = 1.0 / rps
            stage_start_mono = time.monotonic()
            stage_start_epoch = time.time()
            total = round(rps * duration)
            for index in range(total):
                planned_mono = stage_start_mono + index * interval
                delay = planned_mono - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                planned_epoch = stage_start_epoch + index * interval
                task = asyncio.create_task(send_one(next(prompt_iter), phase, counted, planned_epoch, sequence))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                sequence += 1
            stage_deadline = stage_start_mono + duration
            if stage_deadline > time.monotonic():
                await asyncio.sleep(stage_deadline - time.monotonic())
        if tasks:
            print(f"Waiting for {len(tasks)} in-flight requests to finish...")
            await asyncio.gather(*tasks)
    return rows, steady_start_epoch


def _write_metadata(path: Path, case: Case, args: argparse.Namespace) -> None:
    config = ApiConfig.from_env()
    metadata = {
        "case": case.name,
        "started_at_epoch": time.time(),
        "endpoint": config.endpoint,
        "model": config.model,
        "max_tokens": case.max_tokens,
        "mode": case.mode,
        "max_attempts_policy": "成功请求目标的最多 2 倍尝试；达到上限后停止并判定数据不足",
        "p1_actual_capacity_target": True,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {"python": os.sys.version, "pid": os.getpid(),
                        "failure_backoff_seconds": config.failure_backoff_seconds,
                        "failure_backoff_max_seconds": config.failure_backoff_max_seconds,
                        "worker_start_stagger_seconds": config.worker_start_stagger_seconds},
    }
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


async def main_async(case_name: str, argv: list[str] | None = None) -> int:
    case = CASES[case_name]
    parser = argparse.ArgumentParser(description=f"Run Kimi K3 admission test {case_name}")
    parser.add_argument("--data", type=Path, default=case.data_file)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--samples", type=int, default=case.successful_samples)
    parser.add_argument("--warmup-seconds", type=int, default=case.warmup_seconds)
    parser.add_argument("--quick", action="store_true", help="Development-only short run; never use for admission conclusions")
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args(argv)
    prompts = load_prompts(args.data)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.output_root / f"{case.name}-{timestamp}"
    out.mkdir(parents=True, exist_ok=False)
    _write_metadata(out / "metadata.json", case, args)
    writer = ResultWriter(out / "requests.jsonl")
    try:
        if case.mode == "closed":
            env_samples = int(os.getenv("K3_CUSTOM_SAMPLES", "0") or 0)
            env_concurrency = int(os.getenv("K3_CUSTOM_CONCURRENCY", "0") or 0)
            samples = min(args.samples or env_samples or 1, 10) if args.quick else (args.samples or env_samples or case.successful_samples or 1)
            warmup = 0 if args.quick else args.warmup_seconds
            if args.concurrency is not None:
                case = case.__class__(**{**case.__dict__, "concurrency": args.concurrency})
            rows = await run_closed(case, prompts, writer, samples, warmup)
            steady_start = None
        else:
            stages = (("quick_steady", 1.0, 5, True),) if args.quick else case.rate_stages
            rows, steady_start = await run_open(case, prompts, writer, stages)
    finally:
        writer.close()
    summary = summarize(case, rows, steady_start)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_artifacts(out, rows, summary)
    finalize(out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {out.resolve()}")
    return 0 if summary["admission_passed"] else 2


def cli(case_name: str) -> None:
    try:
        raise SystemExit(asyncio.run(main_async(case_name)))
    except KeyboardInterrupt:
        print("Interrupted; partial request data remains in the result directory.")
        raise SystemExit(130)
