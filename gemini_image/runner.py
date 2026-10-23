"""Gemini 图像生成并发压测。

与 ``gpt_image/runner.py`` 平行，但走 Gemini 原生协议面：
``POST {base}/v1beta/models/{model}:generateContent``，请求体用 ``contents`` +
``generationConfig``。参数矩阵、证据落盘与收尾契约与其他套件保持一致。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

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
from common.image_fingerprint import build_image_fingerprints, fingerprint_lines
from common.image_c2pa import build_image_c2pa, c2pa_lines

from volcperf.protocols import api_root

from .cases import GEMINI_IMAGE_COUNTS, LOAD_CASES, MODEL_QUALITY_SUPPORT, Case


PROMPTS = (
    "A clean editorial product photograph of a translucent glass teapot on a pale stone table, soft morning light, no text, no logo.",
    "A cinematic wide landscape of a red train crossing a high bridge above a misty forest valley at sunrise, detailed natural colors.",
    "A modern architectural poster showing a quiet concrete pavilion beside a reflecting pool, geometric composition, no words or logos.",
    "A friendly orange cat wearing a tiny blue raincoat walking through a neon-lit Tokyo side street at night, polished illustration.",
    "A top-down food photograph of a colorful seasonal vegetable platter on a dark ceramic plate, studio lighting, realistic texture.",
    "A hand-painted science-fiction observatory on an icy moon, distant planet in the sky, atmospheric perspective, highly detailed.",
    "A warm lifestyle photograph of a wooden desk with a notebook, dried flowers, and a brass lamp beside a sunny window.",
    "A whimsical miniature coastal town built inside a seashell, tiny boats and people, soft storybook lighting.",
    "A minimalist travel poster of terraced rice fields in morning fog, layered depth, muted palette, no lettering.",
    "A studio macro photograph of a dew-covered dragonfly wing against a deep green bokeh background.",
)

DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
IMAGE_PROTOCOL = "gemini_images"
# 默认单批保留图片总量上限，避免高请求数 4K 批次撑爆结果目录；超出部分显式跳过并记录。
DEFAULT_MAX_IMAGE_GIB = 20.0
KEEP_IMAGES_ENV = "GEMINI_IMAGE_KEEP_IMAGES"
MAX_IMAGE_GIB_ENV = "GEMINI_IMAGE_MAX_IMAGE_GIB"


def image_endpoint(base_url: str, model: str) -> str:
    """Gemini 原生 generateContent 端点（非流式）。"""
    return f"{api_root(IMAGE_PROTOCOL, base_url)}/models/{quote(model, safe='')}:generateContent"


def request_headers(api_key: str, request_id: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Request-ID": request_id,
        "x-goog-api-key": api_key,
    }


def build_payload(prompt: str, quality: str, aspect_ratio: str, count: int) -> dict[str, Any]:
    """构造 Gemini generateContent 请求体。

    ``responseModalities`` 必须同时给出 TEXT 与 IMAGE，图像才会出现在
    candidates[].content.parts[] 的 inlineData 中。
    """
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": aspect_ratio, "imageSize": quality},
            "candidateCount": count,
        },
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low, high = int(index), min(len(ordered) - 1, int(index) + 1)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _parse_choice_list(raw: str | None, allowed: tuple[str, ...], label: str) -> list[str]:
    values = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if not values:
        return list(allowed)
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(f"{label}包含不支持的值: {', '.join(invalid)}")
    return list(dict.fromkeys(values))


def _parse_count_list(raw: str | None, allowed: tuple[int, ...] = GEMINI_IMAGE_COUNTS) -> list[int]:
    values = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if not values:
        return list(allowed)
    counts: list[int] = []
    for value in values:
        try:
            parsed = int(value)
        except ValueError:
            raise ValueError(f"生成数量必须是整数: {value}") from None
        if parsed not in allowed:
            raise ValueError(f"生成数量包含不支持的值: {value}")
        counts.append(parsed)
    return list(dict.fromkeys(counts))


def _model_choices(case: Case, raw: str | None) -> list[str]:
    values = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if not values:
        return list(case.models)
    invalid = [value for value in values if value not in case.models]
    if invalid:
        raise ValueError(f"模型选择包含不支持的值: {', '.join(invalid)}")
    return list(dict.fromkeys(values))


def _qualities_for_model(model: str, requested: list[str]) -> list[str]:
    supported = MODEL_QUALITY_SUPPORT.get(model)
    if not supported:
        return list(requested)
    allowed = [value for value in requested if value in supported]
    return allowed or list(supported)


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
    """只提取结构化元数据；inlineData 的 base64 图片内容由落盘模块单独处理。"""
    empty = {
        "image_count": 0,
        "result_types": [],
        "finish_reasons": [],
        "response_id": None,
        "response_model": None,
        "usage": None,
        "prompt_feedback": None,
    }
    if not isinstance(body, dict):
        return empty
    image_count = 0
    result_types: list[str] = []
    finish_reasons: list[str] = []
    candidates = body.get("candidates")
    for candidate in candidates if isinstance(candidates, list) else []:
        if not isinstance(candidate, dict):
            continue
        reason = candidate.get("finishReason")
        if reason:
            finish_reasons.append(str(reason))
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        for part in parts if isinstance(parts, list) else []:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict) and inline.get("data"):
                image_count += 1
                mime = inline.get("mimeType") or inline.get("mime_type") or "inlineData"
                result_types.append(str(mime))
            elif part.get("thought"):
                result_types.append("thought")
            elif "text" in part:
                result_types.append("text")
    return {
        "image_count": image_count,
        "result_types": result_types,
        "finish_reasons": finish_reasons,
        "response_id": body.get("responseId"),
        "response_model": body.get("modelVersion"),
        "usage": body.get("usageMetadata"),
        "prompt_feedback": body.get("promptFeedback"),
    }


def _image_sources(body: Any) -> list[dict[str, Any]]:
    """列出响应中的图片来源：inlineData（base64）或 fileData（URI 需下载）。"""
    sources: list[dict[str, Any]] = []
    candidates = body.get("candidates") if isinstance(body, dict) else None
    for candidate in candidates if isinstance(candidates, list) else []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        for part in parts if isinstance(parts, list) else []:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict) and inline.get("data"):
                sources.append(
                    {
                        "sequence": len(sources) + 1,
                        "b64": inline["data"],
                        "mime_type": inline.get("mimeType") or inline.get("mime_type"),
                        "source": "inlineData",
                    }
                )
                continue
            file_data = part.get("fileData") or part.get("file_data")
            if not isinstance(file_data, dict):
                continue
            uri = file_data.get("fileUri") or file_data.get("file_uri")
            if uri:
                sources.append(
                    {
                        "sequence": len(sources) + 1,
                        "url": uri,
                        "mime_type": file_data.get("mimeType") or file_data.get("mime_type"),
                        "source": "fileData",
                    }
                )
    return sources


def build_request_plan(
    *,
    case: Case,
    requests: int,
    qualities: list[str],
    aspect_ratios: list[str],
    models: list[str],
    counts: list[int],
    seed: int,
    prompt: str | None = None,
) -> list[dict[str, Any]]:
    """在执行任何 HTTP 请求之前，先生成全部请求定义（可复现）。"""
    rng = random.Random(seed)
    custom_prompt = normalize_image_prompt(prompt)
    plan: list[dict[str, Any]] = []
    for index in range(requests):
        model = rng.choice(models)
        # 分辨率档位按模型支持范围收敛，避免产出必然被上游拒绝的组合。
        quality = rng.choice(_qualities_for_model(model, qualities))
        aspect_ratio = rng.choice(aspect_ratios)
        count = rng.choice(counts)
        prompt = custom_prompt if custom_prompt is not None else rng.choice(PROMPTS)
        plan.append(
            {
                "case": case.case_id,
                "index": index + 1,
                "model": model,
                "quality": quality,
                "aspect_ratio": aspect_ratio,
                "count": count,
                "prompt": prompt,
                "payload": build_payload(prompt, quality, aspect_ratio, count),
            }
        )
    return plan


async def _request_one(
    client: httpx.AsyncClient,
    *,
    request_plan: dict[str, Any],
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
    aspect_ratio = request_plan["aspect_ratio"]
    count = int(request_plan["count"])
    prompt = request_plan["prompt"]
    payload = dict(request_plan["payload"])
    request_id = f"{case_name}-{index + 1}-{uuid.uuid4().hex[:10]}"
    endpoint = image_endpoint(base_url, model)
    started_epoch = time.time()
    started_mono = time.perf_counter()
    response: httpx.Response | None = None
    body: Any = None
    error_type: str | None = None
    error_message: str | None = None
    try:
        response = await client.post(
            endpoint,
            json=payload,
            headers=request_headers(api_key, request_id),
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
                error_message = "HTTP 成功但候选响应未包含 inlineData 图片结果"
    except httpx.TimeoutException as exc:
        error_type, error_message = "timeout", str(exc)
    except httpx.HTTPError as exc:
        error_type, error_message = "client_error", str(exc)
    except Exception as exc:  # 单个请求异常不得中断整批
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
        "protocol": IMAGE_PROTOCOL,
        "endpoint": endpoint,
        "model": model,
        "quality": quality,
        "aspect_ratio": aspect_ratio,
        "count": count,
        "prompt": prompt,
        "http_status": response.status_code if response is not None else None,
        "success": error_type is None,
        "error_type": error_type,
        "error_message": error_message,
        "elapsed_seconds": elapsed,
        "response_headers": _safe_response_headers(response) if response is not None else {},
        "image_count": details["image_count"],
        "count_honored": details["image_count"] == count,
        "result_types": details["result_types"],
        "finish_reasons": details["finish_reasons"],
        "response_id": details["response_id"],
        "response_model": details["response_model"],
        "usage": details["usage"],
        "prompt_feedback": details["prompt_feedback"],
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
    base_url: str,
    started_epoch: float,
    elapsed_seconds: float,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    """即使传输层或内部代码异常，也写出字段完整的请求行。"""
    return {
        "case": request_plan["case"],
        "index": int(request_plan["index"]),
        "request_id": f"{request_plan['case']}-{request_plan['index']}-{uuid.uuid4().hex[:10]}",
        "started_at": datetime.fromtimestamp(started_epoch, timezone.utc).isoformat(),
        "started_epoch": started_epoch,
        "completed_epoch": time.time(),
        "protocol": IMAGE_PROTOCOL,
        "endpoint": image_endpoint(base_url, request_plan["model"]),
        "model": request_plan["model"],
        "quality": request_plan["quality"],
        "aspect_ratio": request_plan["aspect_ratio"],
        "count": int(request_plan["count"]),
        "prompt": request_plan["prompt"],
        "http_status": None,
        "success": False,
        "error_type": error_type,
        "error_message": error_message,
        "elapsed_seconds": elapsed_seconds,
        "response_headers": {},
        "image_count": 0,
        "count_honored": False,
        "result_types": [],
        "finish_reasons": [],
        "response_id": None,
        "response_model": None,
        "usage": None,
        "prompt_feedback": None,
        "transport_status": "error",
        "images": [],
        "image_save_error": None,
        "response_body_shape": {"keys": [], "image_count": 0, "result_types": []},
    }


async def _run_requests(
    *,
    request_plan: list[dict[str, Any]],
    concurrency: int,
    base_url: str,
    api_key: str,
    request_timeout_seconds: float,
    requests_path: Path,
    image_dir: Path | None = None,
    keep_images: bool = True,
    budget: ImageBudget | None = None,
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
    # 增量落盘：SIGTERM 或进程崩溃不得丢掉已经完成的请求证据。
    with requests_path.open("w", encoding="utf-8", buffering=1) as request_stream:
        async with httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            verify=True,
            trust_env=False,
        ) as client:
            async def worker() -> None:
                nonlocal successful_count
                while True:
                    try:
                        index = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    plan = request_plan[index]
                    started_epoch = time.time()
                    started_mono = time.perf_counter()
                    try:
                        # httpx 的读写超时是"空闲超时"，这里再加绝对墙钟上限。
                        async with asyncio.timeout(request_timeout_seconds):
                            row = await _request_one(
                                client,
                                request_plan=plan,
                                base_url=base_url,
                                api_key=api_key,
                                image_dir=image_dir,
                                keep_images=keep_images,
                                budget=budget,
                            )
                    except TimeoutError as exc:
                        row = _failed_request_row(
                            plan,
                            base_url=base_url,
                            started_epoch=started_epoch,
                            elapsed_seconds=time.perf_counter() - started_mono,
                            error_type="request_timeout",
                            error_message=str(exc) or f"请求总时长超过 {request_timeout_seconds:g} 秒",
                        )
                    except Exception as exc:  # 隔离单个坏请求
                        row = _failed_request_row(
                            plan,
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


def _write_report(out: Path, summary: dict[str, Any]) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    honored = summary.get("count_honored_requests") or 0
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
    lines = [
        f"# {summary['case']} Gemini 图像生成压测报告",
        "",
        *stats_lines(
            summary["requested"], summary["successful"], summary["failed"], 0,
            summary["requested"] - summary["completed"], summary["success_rate"],
            [
                f"请求总超时：{summary['request_timeout_seconds']} 秒",
                f"错误分布：{json.dumps(summary['error_distribution'], ensure_ascii=False)}",
                f"并发数：{summary['concurrency']}",
                f"模型分布：{json.dumps(summary['model_distribution'], ensure_ascii=False)}",
                f"强度分布（imageSize）：{json.dumps(summary['quality_distribution'], ensure_ascii=False)}",
                f"尺寸分布（aspectRatio）：{json.dumps(summary['aspect_ratio_distribution'], ensure_ascii=False)}",
                f"每次生成张数分布：{json.dumps(summary['count_distribution'], ensure_ascii=False)}",
                f"实际返回张数分布：{json.dumps(summary['image_count_distribution'], ensure_ascii=False)}",
                f"张数兑现率（实际张数 = 请求张数）：{honored}/{summary['completed']}",
                "执行方式：先生成完整 `generated_requests.jsonl`，再按预生成顺序以配置并发执行。",
            ],
        ),
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
        "## Token 用量",
        "",
        "| 项目 | 合计 |",
        "|---|---:|",
        f"| 输入 Token（文本） | {fmt(summary['usage']['prompt_tokens'])} |",
        f"| 输出 Token（图像） | {fmt(summary['usage']['candidate_tokens'])} |",
        f"| 思考 Token | {fmt(summary['usage']['thoughts_tokens'])} |",
        f"| 合计 Token | {fmt(summary['usage']['total_tokens'])} |",
        "",
        "> 请求明细只保存图片文件路径、字节数、像素尺寸与 SHA-256/dHash 指纹；base64 图片内容不进入 `requests.jsonl`。",
        "> 张数兑现率用于观察上游是否真实兑现 `candidateCount`；它不参与准入判定，实际口径以请求明细为准。",
        f"> {images_line}",
        *fingerprint_lines(summary.get("image_fingerprints")),
        *c2pa_lines(summary.get("image_c2pa")),
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _usage_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "candidate_tokens": 0, "thoughts_tokens": 0, "total_tokens": 0}
    for row in rows:
        usage = row.get("usage")
        if not isinstance(usage, dict):
            continue
        totals["prompt_tokens"] += int(usage.get("promptTokenCount") or 0)
        totals["candidate_tokens"] += int(usage.get("candidatesTokenCount") or 0)
        totals["thoughts_tokens"] += int(usage.get("thoughtsTokenCount") or 0)
        totals["total_tokens"] += int(usage.get("totalTokenCount") or 0)
    return totals


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gemini 图像生成并发压测")
    parser.add_argument("case", choices=sorted(LOAD_CASES))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--quality", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--aspects", default="")
    parser.add_argument("--counts", default="")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prompt", default=os.getenv("GEMINI_IMAGE_PROMPT"), help="自定义图片提示词；留空使用默认提示词池")
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
    qualities = _parse_choice_list(
        args.quality or os.getenv("GEMINI_IMAGE_QUALITIES"),
        case.qualities,
        "强度(imageSize)",
    )
    aspect_ratios = _parse_choice_list(
        args.aspects or os.getenv("GEMINI_IMAGE_ASPECTS"),
        case.aspect_ratios,
        "尺寸(aspectRatio)",
    )
    models = _model_choices(case, args.models or os.getenv("GEMINI_IMAGE_MODELS"))
    counts = _parse_count_list(args.counts or os.getenv("GEMINI_IMAGE_COUNTS"), case.counts)
    api_key = os.getenv("GEMINI_IMAGE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    base_url = os.getenv("GEMINI_IMAGE_BASE_URL") or os.getenv("GEMINI_BASE_URL") or ""
    if not api_key or not base_url:
        parser.error("缺少 GEMINI_IMAGE_API_KEY/GEMINI_API_KEY 或 GEMINI_IMAGE_BASE_URL/GEMINI_BASE_URL")
    try:
        request_timeout_seconds = float(
            os.getenv("GEMINI_IMAGE_REQUEST_TIMEOUT_SECONDS", str(DEFAULT_REQUEST_TIMEOUT_SECONDS))
        )
    except ValueError:
        parser.error("GEMINI_IMAGE_REQUEST_TIMEOUT_SECONDS 必须是数字")
    if request_timeout_seconds <= 0:
        parser.error("GEMINI_IMAGE_REQUEST_TIMEOUT_SECONDS 必须大于 0")

    run_id = os.getenv("VOLC_RECORD_ID") or uuid.uuid4().hex[:16]
    out = args.output_root / f"{case.case_id}-{time.strftime('%Y%m%d-%H%M%S')}-{run_id[:6]}"
    out.mkdir(parents=True, exist_ok=False)
    resolved_seed = (
        args.seed if args.seed is not None else random.SystemRandom().randrange(0, 2**63)
    )
    request_plan = build_request_plan(
        case=case,
        requests=args.requests,
        qualities=qualities,
        aspect_ratios=aspect_ratios,
        models=models,
        counts=counts,
        seed=resolved_seed,
        prompt=prompt,
    )
    metadata = {
        "schema_version": "mltest.gemini-image.v1",
        "suite": "gemini_image",
        "test_kind": "gemini-image",
        "case": case.case_id,
        "record_id": run_id,
        "protocol": IMAGE_PROTOCOL,
        "base_url": base_url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "request_timeout_seconds": request_timeout_seconds,
        "qualities": qualities,
        "aspect_ratios": aspect_ratios,
        "models": models,
        "counts": counts,
        "keep_images": keep_images,
        "images_dir": IMAGE_DIR_NAME if keep_images else None,
        "max_image_bytes": max_image_bytes if keep_images else None,
        "seed": resolved_seed,
        "prompt_count": 1 if prompt is not None else len(PROMPTS),
        "prompt_mode": "custom" if prompt is not None else "random",
        "spec_source": "Google Gemini API image generation (generateContent + imageConfig)",
        "generation_phase": "complete_before_execution",
    }
    (out / "metadata.json").write_text(
        json.dumps(redact(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "generated_requests.jsonl").write_text(
        "\n".join(json.dumps(redact(item), ensure_ascii=False) for item in request_plan) + "\n",
        encoding="utf-8",
    )
    image_dir = (out / IMAGE_DIR_NAME) if keep_images else None
    budget = ImageBudget(max_image_bytes) if keep_images else None

    rows = await _run_requests(
        request_plan=request_plan,
        concurrency=args.concurrency,
        base_url=base_url,
        api_key=api_key,
        request_timeout_seconds=request_timeout_seconds,
        requests_path=out / "requests.jsonl",
        image_dir=image_dir,
        keep_images=keep_images,
        budget=budget,
    )

    successful = [row for row in rows if row["success"]]
    latencies = [float(row["elapsed_seconds"]) for row in rows]
    successful_latencies = [float(row["elapsed_seconds"]) for row in successful]
    honored = sum(1 for row in successful if row.get("count_honored"))
    error_distribution = dict(Counter(row["error_type"] for row in rows if row["error_type"]))
    image_totals = summarize_images(rows)
    # 图片指纹（SHA-256 + dHash）：逐图数据与重复情况写进报告和 summary，
    # 但**不参与准入判定**——指纹只能提示「上游可能复用同一张图」，不能单独判假。
    fingerprints = build_image_fingerprints(rows)
    # C2PA 内容签名（官方指纹）：签名主体/签发者/生成器/水印，从落盘原图读取。
    c2pa = build_image_c2pa(rows, out)
    save_errors = sum(1 for row in rows if row.get("image_save_error"))
    summary = {
        "schema_version": "mltest.gemini-image.v1",
        "record_id": run_id,
        "suite": "gemini_image",
        "test_kind": "gemini-image",
        "case": case.case_id,
        "protocol": IMAGE_PROTOCOL,
        "base_url": base_url,
        "models": models,
        "qualities": qualities,
        "aspect_ratios": aspect_ratios,
        "counts": counts,
        "requested": args.requests,
        "sent": len(rows),
        "completed": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "success_rate": len(successful) / len(rows) if rows else 0.0,
        "concurrency": args.concurrency,
        "request_timeout_seconds": request_timeout_seconds,
        "latency_p50_seconds": percentile(successful_latencies, 0.50),
        "latency_p90_seconds": percentile(successful_latencies, 0.90),
        "latency_p99_seconds": percentile(successful_latencies, 0.99),
        "latency_avg_seconds": sum(latencies) / len(latencies) if latencies else None,
        "model_distribution": dict(Counter(row["model"] for row in rows)),
        "quality_distribution": dict(Counter(row["quality"] for row in rows)),
        "aspect_ratio_distribution": dict(Counter(row["aspect_ratio"] for row in rows)),
        "count_distribution": dict(Counter(row["count"] for row in rows)),
        "image_count_distribution": dict(Counter(row["image_count"] for row in rows)),
        "expected_image_total": sum(int(row["count"]) for row in rows),
        "actual_image_total": sum(int(row["image_count"]) for row in rows),
        "count_honored_requests": honored,
        "count_honored_rate": honored / len(successful) if successful else 0.0,
        "images_dir": IMAGE_DIR_NAME if keep_images else None,
        "images_saved": image_totals["images_saved"],
        "image_bytes_saved": image_totals["image_bytes_saved"],
        "images_skipped": image_totals["images_skipped"],
        "image_save_errors": save_errors,
        "max_image_bytes": max_image_bytes if keep_images else None,
        "usage": _usage_totals(rows),
        "image_fingerprints": fingerprints,
        "image_c2pa": c2pa,
        "error_distribution": error_distribution,
        "checks": {
            "exact_request_count": {
                "value": len(rows),
                "expected": args.requests,
                "passed": len(rows) == args.requests,
            },
            "all_successful": {
                "value": len(successful),
                "expected": args.requests,
                "passed": len(successful) == args.requests,
            },
        },
        "admission_passed": len(rows) == args.requests and len(successful) == args.requests,
        "scope_note": (
            "图像生成压测按请求完成时间统计；每次请求的候选张数由 candidateCount 决定。"
            "返回图片的原文件保存在结果目录 images/，请求明细只记录路径、字节数、尺寸与 SHA-256/dHash 指纹。"
            "张数兑现率仅作观察指标，不参与准入判定；图片指纹同样只作证据，不参与准入判定。"
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
