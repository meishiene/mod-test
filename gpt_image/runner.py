from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from common.audit import redact
from common.finalizer import finalize_result_dir
from common.image_store import (
    IMAGE_DIR_NAME,
    ImageBudget,
    env_flag,
    format_budget,
    persist_images,
    summarize_images,
)
from common.report_style import stats_lines
from common.image_prompt import normalize_image_prompt
from .naming import image_filename_stem
from common.image_fingerprint import build_image_fingerprints, fingerprint_lines
from common.image_c2pa import build_image_c2pa, c2pa_lines

from .cases import DEFAULT_REQUEST_TIMEOUT_SECONDS, GPT_IMAGE_ASPECT_RATIOS, LOAD_CASES, Case, model_quality_choices
from volcperf.protocols import IMAGE_COMPATIBLE_PROTOCOLS
from volcperf.protocols import api_root


PROMPTS = (
    "A clean editorial product photograph of a translucent glass teapot on a pale stone table, soft morning light, no text, no logo.",
    "A cinematic wide landscape of a red train crossing a high bridge above a misty forest valley at sunrise, detailed natural colors.",
    "A modern architectural poster showing a quiet concrete pavilion beside a reflecting pool, geometric composition, no words or logos.",
    "A friendly orange cat wearing a tiny blue raincoat walking through a neon-lit Tokyo side street at night, polished illustration.",
    "A top-down food photograph of a colorful seasonal vegetable platter on a dark ceramic plate, studio lighting, realistic texture.",
    "A hand-painted science-fiction observatory on an icy moon, distant planet in the sky, atmospheric perspective, highly detailed.",
    "A warm lifestyle photograph of a wooden desk with a notebook, dried flowers, and a brass lamp beside a sunny window.",
    "A whimsical miniature coastal town built inside a seashell, tiny boats and people, soft storybook lighting.",
)

# 默认单批保留图片总量上限，避免高请求数 4K 批次撑爆结果目录；超出部分显式跳过并记录。
DEFAULT_MAX_IMAGE_GIB = 20.0
KEEP_IMAGES_ENV = "GPT_IMAGE_KEEP_IMAGES"
MAX_IMAGE_GIB_ENV = "GPT_IMAGE_MAX_IMAGE_GIB"
# 压测持续时长（秒）：**默认不启用**。启用后在设定并发下持续压测该时长，时间到即停止
# 领取新请求，`--requests` 退化为上限保险丝（防止时长内打满预算）。
HOLD_SECONDS_ENV = "GPT_IMAGE_HOLD_SECONDS"


def image_endpoint(base_url: str, protocol: str = "openai") -> str:
    return f"{api_root(protocol, base_url)}/images/generations"


def request_headers(protocol: str, api_key: str, request_id: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Request-ID": request_id,
    }
    if protocol in IMAGE_COMPATIBLE_PROTOCOLS:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        raise ValueError(f"图片压测暂不支持 {protocol} 协议")
    return headers


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low, high = int(index), min(len(ordered) - 1, int(index) + 1)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _parse_positive_list(raw: str | None, allowed: tuple[str, ...], label: str) -> list[str]:
    values = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if not values:
        return list(allowed)
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(f"{label}包含不支持的值: {', '.join(invalid)}")
    return list(dict.fromkeys(values))


def _model_choices(case: Case, raw: str | None) -> list[str]:
    values = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if not values:
        return list(case.model_choices)
    invalid = [value for value in values if value not in case.model_choices]
    if invalid:
        raise ValueError(f"模型选择包含不支持的值: {', '.join(invalid)}")
    return list(dict.fromkeys(values))


def _safe_response_headers(response: httpx.Response) -> dict[str, str]:
    keep = {
        "content-type",
        "server",
        "via",
        "x-request-id",
        "request-id",
        "x-trace-id",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
    }
    return {
        key.lower(): value
        for key, value in response.headers.items()
        if key.lower() in keep
    }


def _response_summary(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {
            "image_count": 0,
            "result_types": [],
            "response_id": None,
            "created": None,
            "usage": None,
            "response_model": None,
        }
    items = body.get("data")
    if not isinstance(items, list):
        items = []
    result_types = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("b64_json"):
            result_types.append("b64_json")
        elif item.get("url"):
            result_types.append("url")
        elif item:
            result_types.append("object")
    return {
        "image_count": len(items),
        "result_types": result_types,
        "response_id": body.get("id"),
        "created": body.get("created"),
        "usage": body.get("usage"),
        "response_model": body.get("model"),
    }


def _image_sources(body: Any) -> list[dict[str, Any]]:
    """列出响应中的图片来源：b64_json（base64）或 url（需下载）。"""
    items = body.get("data") if isinstance(body, dict) else None
    sources: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        encoded = item.get("b64_json")
        if isinstance(encoded, str) and encoded:
            sources.append(
                {
                    "sequence": len(sources) + 1,
                    "b64": encoded,
                    "mime_type": item.get("mime_type"),
                    "source": "b64_json",
                }
            )
            continue
        url = item.get("url")
        if isinstance(url, str) and url:
            sources.append(
                {
                    "sequence": len(sources) + 1,
                    "url": url,
                    "mime_type": item.get("mime_type"),
                    "source": "url",
                }
            )
    return sources


def build_request_plan(
    *,
    case: Case,
    requests: int,
    qualities: list[str],
    sizes: list[str],
    models: list[str],
    seed: int,
    aspect_ratio: str | None = None,
    prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Generate every request definition before any HTTP request starts."""
    rng = random.Random(seed)
    custom_prompt = normalize_image_prompt(prompt)
    qualities_by_model = model_quality_choices(models, qualities)
    plan: list[dict[str, Any]] = []
    for index in range(requests):
        model = rng.choice(models)
        quality = rng.choice(qualities_by_model[model])
        size = rng.choice(sizes)
        prompt = custom_prompt if custom_prompt is not None else rng.choice(PROMPTS)
        plan.append(
            {
                "case": case.case_id,
                "index": index + 1,
                "model": model,
                "quality": quality,
                "size": size,
                "prompt": prompt,
                "payload": {
                    "model": model,
                    "prompt": prompt,
                    "size": size,
                    "quality": quality,
                    "n": 1,
                    **({"aspect_ratio": aspect_ratio} if aspect_ratio else {}),
                },
            }
        )
    return plan


async def _request_one(
    client: httpx.AsyncClient,
    *,
    request_plan: dict[str, Any],
    protocol: str,
    base_url: str,
    api_key: str,
    image_dir: Path | None = None,
    keep_images: bool = True,
    budget: ImageBudget | None = None,
) -> dict[str, Any]:
    case_name = request_plan["case"]
    index = int(request_plan["index"]) - 1
    model = request_plan["model"]
    quality = request_plan["quality"]
    size = request_plan["size"]
    prompt = request_plan["prompt"]
    payload = dict(request_plan["payload"])
    request_id = f"{case_name}-{index + 1}-{uuid.uuid4().hex[:10]}"
    started_epoch = time.time()
    started_mono = time.perf_counter()
    response: httpx.Response | None = None
    body: Any = None
    error_type: str | None = None
    error_message: str | None = None
    try:
        response = await client.post(
            image_endpoint(base_url, protocol),
            json=payload,
            headers=request_headers(protocol, api_key, request_id),
        )
        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError):
            body = None
        if not 200 <= response.status_code < 300:
            error_type = f"http_{response.status_code}"
            error_message = response.text[:2000]
        else:
            details = _response_summary(body)
            if details["image_count"] < 1:
                error_type = "invalid_image_response"
                error_message = "HTTP 成功但响应未包含 data 图片结果"
    except httpx.TimeoutException as exc:
        error_type, error_message = "timeout", str(exc)
    except httpx.HTTPError as exc:
        error_type, error_message = "client_error", str(exc)
    except Exception as exc:  # keep one failed request from aborting the batch
        error_type, error_message = type(exc).__name__, str(exc)

    elapsed = time.perf_counter() - started_mono
    details = _response_summary(body)
    # 先定格延迟：写盘是本地开销，不能计入被测服务耗时。
    images: list[dict[str, Any]] = []
    image_save_error: str | None = None
    if keep_images and image_dir is not None and body is not None:
        try:
            images = await persist_images(
                client,
                image_dir,
                request_key=f"req-{index + 1:04d}",
                filename_stem=image_filename_stem(model, quality),
                sources=_image_sources(body),
                budget=budget,
            )
        except Exception as exc:  # 落盘失败只记录，不改变请求判定
            image_save_error = f"{type(exc).__name__}: {exc}"
    return {
        "case": case_name,
        "index": index + 1,
        "request_id": request_id,
        "started_at": datetime.fromtimestamp(started_epoch, timezone.utc).isoformat(),
        "started_epoch": started_epoch,
        "completed_epoch": time.time(),
        "protocol": protocol,
        "endpoint": image_endpoint(base_url, protocol),
        "model": model,
        "quality": quality,
        "size": size,
        "prompt": prompt,
        "http_status": response.status_code if response is not None else None,
        "success": error_type is None,
        "error_type": error_type,
        "error_message": error_message,
        "elapsed_seconds": elapsed,
        "response_headers": _safe_response_headers(response) if response is not None else {},
        "image_count": details["image_count"],
        "result_types": details["result_types"],
        "response_id": details["response_id"],
        "response_model": details["response_model"],
        "created": details["created"],
        "usage": details["usage"],
        "transport_status": "completed" if response is not None else "error",
        # 图片原文件落在 images/；请求行只保留路径与校验信息，base64 永不进入证据。
        "images": images,
        "image_save_error": image_save_error,
        "response_body_shape": {
            "keys": sorted(body.keys()) if isinstance(body, dict) else [],
            "image_count": details["image_count"],
            "result_types": details["result_types"],
        },
    }


def _failed_request_row(
    request_plan: dict[str, Any],
    *,
    protocol: str,
    base_url: str,
    started_epoch: float,
    elapsed_seconds: float,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    """Build a complete request row even when transport/internal code fails."""
    return {
        "case": request_plan["case"],
        "index": int(request_plan["index"]),
        "request_id": f"{request_plan['case']}-{request_plan['index']}-{uuid.uuid4().hex[:10]}",
        "started_at": datetime.fromtimestamp(started_epoch, timezone.utc).isoformat(),
        "started_epoch": started_epoch,
        "completed_epoch": time.time(),
        "protocol": protocol,
        "endpoint": image_endpoint(base_url, protocol),
        "model": request_plan["model"],
        "quality": request_plan["quality"],
        "size": request_plan["size"],
        "prompt": request_plan["prompt"],
        "http_status": None,
        "success": False,
        "error_type": error_type,
        "error_message": error_message,
        "elapsed_seconds": elapsed_seconds,
        "response_headers": {},
        "image_count": 0,
        "result_types": [],
        "response_id": None,
        "response_model": None,
        "created": None,
        "usage": None,
        "transport_status": "error",
        "images": [],
        "image_save_error": None,
        "response_body_shape": {
            "keys": [],
            "image_count": 0,
            "result_types": [],
        },
    }


async def _run_requests(
    *,
    request_plan: list[dict[str, Any]],
    concurrency: int,
    protocol: str,
    base_url: str,
    api_key: str,
    request_timeout_seconds: float,
    requests_path: Path,
    image_dir: Path | None = None,
    keep_images: bool = True,
    budget: ImageBudget | None = None,
    hold_seconds: float | None = None,
) -> list[dict[str, Any]]:
    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in range(len(request_plan)):
        queue.put_nowait(index)
    rows: list[dict[str, Any]] = []
    rows_lock = asyncio.Lock()
    successful_count = 0
    timeout = httpx.Timeout(
        request_timeout_seconds,
        connect=min(30.0, request_timeout_seconds),
        write=min(30.0, request_timeout_seconds),
        pool=min(30.0, request_timeout_seconds),
    )
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
        keepalive_expiry=60.0,
    )
    # Incremental persistence is intentional: a SIGTERM or process crash must
    # not discard every request that already completed.
    with requests_path.open("w", encoding="utf-8", buffering=1) as request_stream:
        async with httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            verify=True,
            trust_env=False,
        ) as client:
            # 压测持续时长（默认不启用）：从第一批请求开跑前计时，到点后 worker 不再领取
            # 新请求；**已在途的请求仍在各自 worker 里跑完**，保证 requests.jsonl 不出现
            # 半截记录，也保证延迟统计包含被时长截断的那一批。
            hold_started = time.perf_counter()
            deadline = None if hold_seconds is None else hold_started + hold_seconds

            async def worker() -> None:
                nonlocal successful_count
                while True:
                    if deadline is not None and time.perf_counter() >= deadline:
                        return
                    try:
                        index = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    plan = request_plan[index]
                    started_epoch = time.time()
                    started_mono = time.perf_counter()
                    try:
                        # httpx read/write timeouts are inactivity timeouts. This
                        # outer deadline is the absolute wall-clock request limit.
                        async with asyncio.timeout(request_timeout_seconds):
                            row = await _request_one(
                                client,
                                request_plan=plan,
                                protocol=protocol,
                                base_url=base_url,
                                api_key=api_key,
                                image_dir=image_dir,
                                keep_images=keep_images,
                                budget=budget,
                            )
                    except TimeoutError as exc:
                        row = _failed_request_row(
                            plan,
                            protocol=protocol,
                            base_url=base_url,
                            started_epoch=started_epoch,
                            elapsed_seconds=time.perf_counter() - started_mono,
                            error_type="request_timeout",
                            error_message=str(exc) or f"请求总时长超过 {request_timeout_seconds:g} 秒",
                        )
                    except Exception as exc:  # isolate one bad request from other workers
                        row = _failed_request_row(
                            plan,
                            protocol=protocol,
                            base_url=base_url,
                            started_epoch=started_epoch,
                            elapsed_seconds=time.perf_counter() - started_mono,
                            error_type=f"internal_{type(exc).__name__}",
                            error_message=str(exc),
                        )
                    finally:
                        queue.task_done()

                    async with rows_lock:
                        rows.append(row)
                        request_stream.write(json.dumps(redact(row), ensure_ascii=False) + "\n")
                        request_stream.flush()
                        completed = len(rows)
                        successful_count += int(bool(row["success"]))
                        successful = successful_count
                    if (
                        completed % max(1, min(100, len(request_plan) // 20 or 1)) == 0
                        or completed == len(request_plan)
                    ):
                        print(
                            f"\r{request_plan[0]['case']}: 完成 {completed}/{len(request_plan)} "
                            f"(成功 {successful}, 失败 {completed - successful}, "
                            f"待完成 {len(request_plan) - completed})",
                            end="",
                            flush=True,
                        )

            await asyncio.gather(
                *(
                    asyncio.create_task(worker())
                    for _ in range(min(concurrency, len(request_plan)))
                )
            )
    print()
    return sorted(rows, key=lambda row: row["index"])


def summary_checks(
    *,
    rows: list[dict[str, Any]],
    successful: list[dict[str, Any]],
    requested: int,
    hold_seconds: float | None,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """领域校验（两类执行口径共用同一份 summary 契约）。

    - 未启用持续时长：沿用历史口径「请求数必须跑满」。
    - 启用持续时长：请求数退化为**上限保险丝**（时长到点即停，实际发出数由模型吞吐决定），
      因此只要求「发出过请求且不超过上限」，达标口径改为「本批实际发出的请求全部成功」。
      **不能**再用 exact_request_count，否则高并发时长压测永远判失败。
    """
    if hold_seconds is None:
        checks = {
            "exact_request_count": {
                "value": len(rows),
                "expected": requested,
                "passed": len(rows) == requested,
            },
            "all_successful": {
                "value": len(successful),
                "expected": requested,
                "passed": len(successful) == requested,
            },
        }
    else:
        checks = {
            "request_count_within_cap": {
                "value": len(rows),
                "expected": {"min": 1, "max": requested},
                "passed": 1 <= len(rows) <= requested,
            },
            "all_successful": {
                "value": len(successful),
                "expected": len(rows),
                "passed": len(rows) > 0 and len(successful) == len(rows),
            },
        }
    return checks, all(bool(item["passed"]) for item in checks.values())


def _write_report(out: Path, summary: dict[str, Any]) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    if summary.get("images_dir"):
        images_line = (
            f"图片保留：{summary['images_saved']} 张 / "
            f"{summary['image_bytes_saved'] / 1024 / 1024:.1f} MiB，保存在 `{summary['images_dir']}/`"
            f"（上限 {format_budget(summary['max_image_bytes'])}）"
        )
        if summary.get("images_skipped"):
            images_line += f"；已达上限跳过 {summary['images_skipped']} 张"
    else:
        images_line = "图片保留：本次未开启（--no-keep-images）"
    hold_seconds = summary.get("hold_seconds")
    if hold_seconds is None:
        total, pending = summary['requested'], summary['requested'] - summary['completed']
        hold_lines = [
            f"请求数：{summary['requested']}（固定档：必须跑满）",
            "压测持续时长：未启用",
            "执行方式：先生成完整 `generated_requests.jsonl`，再按预生成顺序以配置并发执行。",
        ]
    else:
        hold_elapsed = summary.get("hold_elapsed_seconds")
        total, pending = summary['sent'], 0
        hold_lines = [
            f"压测持续时长：{hold_seconds:g} 秒"
            + (
                f"（实测窗口 {hold_elapsed:g} 秒）"
                if isinstance(hold_elapsed, (int, float)) and not isinstance(hold_elapsed, bool)
                else ""
            ),
            f"总请求数上限：{summary['requested']}（保险丝：时长到点即停止发送新请求）",
            "执行方式：先生成完整 `generated_requests.jsonl`（按上限条数），再按预生成顺序以配置并发"
            "持续压测；时长到点后不再发送新请求，在途请求跑完才收尾。",
        ]
    lines = [
        f"# {summary['case']} 图片压测报告",
        "",
        *stats_lines(total, summary['successful'], summary['failed'], 0,
                     pending, summary['success_rate'],
                     [f"请求总超时：{summary['request_timeout_seconds']} 秒",
                      *hold_lines,
                      f"错误分布：{json.dumps(summary['error_distribution'], ensure_ascii=False)}",
                      f"并发数：{summary['concurrency']}",
                      f"实际模型分布：{json.dumps(summary['model_distribution'], ensure_ascii=False)}",
                      f"尺寸分布：{json.dumps(summary['size_distribution'], ensure_ascii=False)}",
                      f"质量分布：{json.dumps(summary['quality_distribution'], ensure_ascii=False)}"]),
        "",
        "## 延迟指标",
        "",
        "| 指标 | 数值（秒） |",
        "|---|---:|",
        f"| P50 | {fmt(summary['latency_p50_seconds'])} |",
        f"| P90 | {fmt(summary['latency_p90_seconds'])} |",
        f"| P99 | {fmt(summary['latency_p99_seconds'])} |",
        f"| 平均 | {fmt(summary['latency_avg_seconds'])} |",
        "",
        "> 请求明细只保存图片文件路径、字节数、像素尺寸与 SHA-256/dHash 指纹；base64 图片内容不进入 `requests.jsonl`。",
        f"> {images_line}",
        *fingerprint_lines(summary.get("image_fingerprints")),
        *c2pa_lines(summary.get("image_c2pa")),
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPT Image 并发图片压测")
    parser.add_argument("case", choices=sorted(LOAD_CASES))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--quality", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--sizes", default="")
    parser.add_argument("--aspect-ratio", choices=GPT_IMAGE_ASPECT_RATIOS, default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prompt", default=os.getenv("GPT_IMAGE_PROMPT"), help="自定义图片提示词；留空使用默认提示词池")
    parser.add_argument(
        "--keep-images",
        action=argparse.BooleanOptionalAction,
        default=env_flag(KEEP_IMAGES_ENV, True),
        help=f"是否把生成的图片原文件保存到结果目录 images/（默认开启；可用 {KEEP_IMAGES_ENV}=0 关闭）",
    )
    parser.add_argument(
        "--max-image-gib",
        type=float,
        default=None,
        help=f"单批保留图片总量上限（GiB，默认 {DEFAULT_MAX_IMAGE_GIB:g}；可用 {MAX_IMAGE_GIB_ENV} 覆盖）",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=None,
        help=(
            "压测持续时长（秒）。**默认不启用**；设置后在设定并发下持续压测该时长，"
            f"时间到即停止发送新请求，--requests 退化为上限（可用 {HOLD_SECONDS_ENV} 覆盖）"
        ),
    )
    args = parser.parse_args(argv)
    try:
        prompt = normalize_image_prompt(args.prompt)
    except ValueError as exc:
        parser.error(str(exc))

    if not 1 <= args.requests <= 100000:
        parser.error("--requests 必须在 1～100000 之间")
    if not 1 <= args.concurrency <= 1000:
        parser.error("--concurrency 必须在 1～1000 之间")

    keep_images = bool(args.keep_images)
    raw_max_gib = args.max_image_gib
    if raw_max_gib is None:
        raw_max_gib = os.getenv(MAX_IMAGE_GIB_ENV)
    try:
        max_image_gib = DEFAULT_MAX_IMAGE_GIB if raw_max_gib in (None, "") else float(raw_max_gib)
    except ValueError:
        parser.error(f"{MAX_IMAGE_GIB_ENV}/--max-image-gib 必须是数字")
    if max_image_gib < 0:
        parser.error(f"{MAX_IMAGE_GIB_ENV}/--max-image-gib 不能为负数")
    max_image_bytes = int(max_image_gib * 1024**3)

    case = LOAD_CASES[args.case]
    qualities = _parse_positive_list(
        args.quality or os.getenv("GPT_IMAGE_QUALITIES"),
        case.qualities,
        "quality",
    )
    sizes = _parse_positive_list(
        args.sizes or os.getenv("GPT_IMAGE_SIZES"),
        case.sizes,
        "size",
    )
    models = _model_choices(
        case,
        args.models or os.getenv("GPT_IMAGE_MODELS"),
    )
    aspect_ratio = args.aspect_ratio or os.getenv("GPT_IMAGE_ASPECT_RATIO") or None
    if aspect_ratio is not None and aspect_ratio not in GPT_IMAGE_ASPECT_RATIOS:
        parser.error("aspect_ratio 包含不支持的值")
    api_key = os.getenv("GPT_IMAGE_API_KEY") or os.getenv("VOLC_API_KEY") or ""
    base_url = os.getenv("GPT_IMAGE_BASE_URL") or os.getenv("VOLC_BASE_URL") or ""
    protocol = (os.getenv("GPT_IMAGE_PROTOCOL") or os.getenv("VOLC_PROTOCOL") or "openai").lower()
    if not api_key or not base_url:
        parser.error("缺少 GPT_IMAGE_API_KEY/VOLC_API_KEY 或 GPT_IMAGE_BASE_URL/VOLC_BASE_URL")
    if protocol not in IMAGE_COMPATIBLE_PROTOCOLS:
        parser.error("图片压测仅支持 OpenAI Images / 火山云 / 百度千帆协议")
    try:
        request_timeout_seconds = float(
            os.getenv(
                "GPT_IMAGE_REQUEST_TIMEOUT_SECONDS",
                str(DEFAULT_REQUEST_TIMEOUT_SECONDS),
            )
        )
    except ValueError:
        parser.error("GPT_IMAGE_REQUEST_TIMEOUT_SECONDS 必须是数字")
    if not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0:
        parser.error("GPT_IMAGE_REQUEST_TIMEOUT_SECONDS 必须是大于 0 的有限数字")

    # 压测持续时长：默认不启用（None）。CLI 优先于环境变量；空串视为不启用。
    raw_hold = args.hold_seconds if args.hold_seconds is not None else os.getenv(HOLD_SECONDS_ENV)
    if raw_hold in (None, ""):
        hold_seconds: float | None = None
    else:
        try:
            hold_seconds = float(raw_hold)
        except (TypeError, ValueError):
            parser.error(f"{HOLD_SECONDS_ENV}/--hold-seconds 必须是数字")
        if not math.isfinite(hold_seconds) or hold_seconds <= 0:
            parser.error("压测持续时长必须是大于 0 的有限秒数")

    run_id = os.getenv("VOLC_RECORD_ID") or uuid.uuid4().hex[:16]
    out = args.output_root / f"{case.case_id}-{time.strftime('%Y%m%d-%H%M%S')}-{run_id[:6]}"
    out.mkdir(parents=True, exist_ok=False)
    resolved_seed = (
        args.seed
        if args.seed is not None
        else random.SystemRandom().randrange(0, 2**63)
    )
    request_plan = build_request_plan(
        case=case,
        requests=args.requests,
        qualities=qualities,
        sizes=sizes,
        models=models,
        seed=resolved_seed,
        aspect_ratio=aspect_ratio,
        prompt=prompt,
    )
    metadata = {
        "schema_version": "mltest.gpt-image.v1",
        "suite": "gpt_image",
        "test_kind": "gpt-image",
        "case": case.case_id,
        "record_id": run_id,
        "protocol": protocol,
        "base_url": base_url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "request_timeout_seconds": request_timeout_seconds,
        "hold_seconds": hold_seconds,
        "aspect_ratio": aspect_ratio,
        "qualities": qualities,
        "sizes": sizes,
        "models": models,
        "keep_images": keep_images,
        "images_dir": IMAGE_DIR_NAME if keep_images else None,
        "max_image_bytes": max_image_bytes if keep_images else None,
        "seed": resolved_seed,
        "prompt_count": 1 if prompt is not None else len(PROMPTS),
        "prompt_mode": "custom" if prompt is not None else "random",
        "spec_source": "OpenAI official image generation model definitions",
        "generation_phase": "complete_before_execution",
    }
    (out / "metadata.json").write_text(
        json.dumps(redact(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "generated_requests.jsonl").write_text(
        "\n".join(
            json.dumps(redact(item), ensure_ascii=False) for item in request_plan
        )
        + "\n",
        encoding="utf-8",
    )
    image_dir = (out / IMAGE_DIR_NAME) if keep_images else None
    budget = ImageBudget(max_image_bytes) if keep_images else None

    rows = await _run_requests(
        request_plan=request_plan,
        concurrency=args.concurrency,
        protocol=protocol,
        base_url=base_url,
        api_key=api_key,
        request_timeout_seconds=request_timeout_seconds,
        requests_path=out / "requests.jsonl",
        image_dir=image_dir,
        keep_images=keep_images,
        budget=budget,
        hold_seconds=hold_seconds,
    )

    successful = [row for row in rows if row["success"]]
    latencies = [float(row["elapsed_seconds"]) for row in rows]
    successful_latencies = [float(row["elapsed_seconds"]) for row in successful]
    # 压测持续时长的实测窗口（首条开始 → 末条完成）；未启用时为 None。
    hold_elapsed_seconds = (
        round(
            max(float(row["completed_epoch"]) for row in rows)
            - min(float(row["started_epoch"]) for row in rows),
            3,
        )
        if hold_seconds is not None and rows
        else None
    )
    error_distribution = dict(
        Counter(row["error_type"] for row in rows if row["error_type"])
    )
    image_totals = summarize_images(rows)
    # 图片指纹（SHA-256 + dHash）：逐图数据与重复情况写进报告和 summary，
    # 但**不参与准入判定**——指纹只能提示「上游可能复用同一张图」，不能单独判假。
    fingerprints = build_image_fingerprints(rows)
    # C2PA 内容签名（官方指纹）：签名主体/签发者/生成器/水印，从落盘原图读取。
    c2pa = build_image_c2pa(rows, out)
    save_errors = sum(1 for row in rows if row.get("image_save_error"))
    checks, admission_passed = summary_checks(
        rows=rows, successful=successful, requested=args.requests, hold_seconds=hold_seconds
    )
    summary = {
        "schema_version": "mltest.gpt-image.v1",
        "record_id": run_id,
        "suite": "gpt_image",
        "test_kind": "gpt-image",
        "case": case.case_id,
        "protocol": protocol,
        "base_url": base_url,
        "models": models,
        "qualities": qualities,
        "sizes": sizes,
        "requested": args.requests,
        "sent": len(rows),
        "completed": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "success_rate": len(successful) / len(rows) if rows else 0.0,
        "concurrency": args.concurrency,
        "request_timeout_seconds": request_timeout_seconds,
        "hold_seconds": hold_seconds,
        "hold_elapsed_seconds": hold_elapsed_seconds,
        "aspect_ratio": aspect_ratio,
        "latency_p50_seconds": percentile(successful_latencies, 0.50),
        "latency_p90_seconds": percentile(successful_latencies, 0.90),
        "latency_p99_seconds": percentile(successful_latencies, 0.99),
        "latency_avg_seconds": sum(latencies) / len(latencies) if latencies else None,
        "model_distribution": dict(Counter(row["model"] for row in rows)),
        "quality_distribution": dict(Counter(row["quality"] for row in rows)),
        "size_distribution": dict(Counter(row["size"] for row in rows)),
        "images_dir": IMAGE_DIR_NAME if keep_images else None,
        "images_saved": image_totals["images_saved"],
        "image_bytes_saved": image_totals["image_bytes_saved"],
        "images_skipped": image_totals["images_skipped"],
        "image_save_errors": save_errors,
        "max_image_bytes": max_image_bytes if keep_images else None,
        "image_fingerprints": fingerprints,
        "image_c2pa": c2pa,
        "error_distribution": error_distribution,
        "checks": checks,
        "admission_passed": admission_passed,
        "scope_note": (
            "图片生成压测按请求完成时间统计；每次 n=1。返回图片的原文件保存在结果目录 images/，"
            "请求明细只记录路径、字节数、尺寸与 SHA-256/dHash 指纹。"
            "图片指纹（唯一性、重复与近似重复分组）只作证据，不参与准入判定。"
            + (
                "本批启用了压测持续时长：请求数上限仅作保险丝，实际发出数与成功率按 requests.jsonl 统计。"
                if hold_seconds is not None
                else ""
            )
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(out, summary)
    finalize_result_dir(out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果目录: {out.resolve()}")
    return 0 if summary["admission_passed"] else 2


def cli() -> None:
    try:
        raise SystemExit(asyncio.run(main_async()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except ValueError as exc:
        print(f"配置错误: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
