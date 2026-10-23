#!/usr/bin/env python3
"""Send one minimal images/generations request to check a GPT Image model is callable."""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from gpt_image.runner import IMAGE_COMPATIBLE_PROTOCOLS, _response_summary, image_endpoint, request_headers

DEFAULT_MODEL = "gpt-image-2.5-sunburst"
DEFAULT_SIZE = "1024x1024"
DEFAULT_QUALITY = "low"
DEFAULT_PROMPT = "A single ripe red apple on a plain white background. No text, no logo."
DEFAULT_TIMEOUT_SECONDS = 300.0
SUPPORTED_PROTOCOLS = IMAGE_COMPATIBLE_PROTOCOLS


def _decode_b64(value: Any) -> bytes | None:
    if not isinstance(value, str) or not value:
        return None
    part = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        return base64.b64decode(part, validate=True)
    except Exception:  # noqa: BLE001 - probe keeps the raw response as evidence
        return None


async def run_probe(
    client: httpx.AsyncClient,
    *,
    model: str,
    protocol: str,
    base_url: str,
    api_key: str,
    size: str = DEFAULT_SIZE,
    quality: str = DEFAULT_QUALITY,
    prompt: str = DEFAULT_PROMPT,
    save_image: Path | None = None,
) -> dict[str, Any]:
    """Send exactly one minimal request and return an inspectable probe row."""
    request_id = f"PROBE-GPT-IMAGE-{uuid.uuid4().hex[:12]}"
    endpoint = image_endpoint(base_url, protocol)
    payload = {"model": model, "prompt": prompt, "size": size, "quality": quality, "n": 1}
    started_epoch = time.time()
    started_mono = time.perf_counter()
    row: dict[str, Any] = {
        "probe": "GPT_IMAGE_PROBE",
        "model": model,
        "endpoint": endpoint,
        "protocol": protocol,
        "size": size,
        "quality": quality,
        "n": 1,
        "request_id": request_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_epoch)),
    }
    image_bytes: bytes | None = None
    try:
        response = await client.post(
            endpoint, json=payload, headers=request_headers(protocol, api_key, request_id)
        )
        row["http_status"] = response.status_code
        body: Any = None
        try:
            body = response.json()
        except ValueError:
            row["body_preview"] = response.text[:2000]
        details = _response_summary(body)
        row["image_count"] = details["image_count"]
        row["result_types"] = details["result_types"]
        row["response_model"] = details["response_model"]
        row["usage"] = details["usage"]
        if isinstance(body, dict) and not details["image_count"]:
            row["error_body"] = json.dumps(body, ensure_ascii=False)[:2000]
        items = body.get("data") if isinstance(body, dict) else None
        first = items[0] if isinstance(items, list) and items and isinstance(items[0], dict) else {}
        image_bytes = _decode_b64(first.get("b64_json"))
        image_url = first.get("url")
        if image_bytes is None and save_image and isinstance(image_url, str) and image_url:
            fetched = await client.get(image_url)
            if fetched.is_success:
                image_bytes = fetched.content
        row["image_bytes"] = len(image_bytes) if image_bytes else None
    except httpx.TimeoutException as exc:
        row["error_type"] = "timeout"
        row["error_message"] = str(exc)
    except httpx.HTTPError as exc:
        row["error_type"] = "client_error"
        row["error_message"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - a probe reports any failure instead of crashing
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)
    finally:
        row["elapsed_seconds"] = round(time.perf_counter() - started_mono, 3)

    status = row.get("http_status")
    http_ok = isinstance(status, int) and 200 <= status < 300
    if "error_type" in row:
        row["callable"] = False
    elif http_ok and int(row.get("image_count") or 0) >= 1:
        row["callable"] = True
    else:
        row["callable"] = False
        if http_ok:
            row["error_type"] = "invalid_image_response"
            row["error_message"] = "HTTP 成功但响应未包含 data 图片结果"
        else:
            row["error_type"] = f"http_{status}"
            row["error_message"] = (
                row.get("error_body") or row.get("body_preview") or "上游返回非 2xx 状态"
            )

    if save_image and image_bytes:
        save_image.parent.mkdir(parents=True, exist_ok=True)
        save_image.write_bytes(image_bytes)
        row["saved_image"] = str(save_image)
    return row


def probe_exit_code(row: dict[str, Any]) -> int:
    """0 = callable, 2 = reachable but unusable response, 1 = call failed."""
    if row.get("callable"):
        return 0
    status = row.get("http_status")
    return 2 if isinstance(status, int) and 200 <= status < 300 else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="发送一次最小图片生成请求，检查指定 GPT Image 模型是否可调用"
    )
    parser.add_argument("--model", default=os.getenv("GPT_IMAGE_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--quality", default=DEFAULT_QUALITY)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--protocol", default="")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--output", type=Path, default=None, help="可选：保存完整探针结果")
    parser.add_argument(
        "--save-image",
        type=Path,
        default=None,
        help="可选：把返回的第一张图写到该路径，用于肉眼确认",
    )
    return parser.parse_args(argv)


async def main_async(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    api_key = args.api_key or os.getenv("GPT_IMAGE_API_KEY") or os.getenv("VOLC_API_KEY") or ""
    base_url = args.base_url or os.getenv("GPT_IMAGE_BASE_URL") or os.getenv("VOLC_BASE_URL") or ""
    protocol = (
        args.protocol or os.getenv("GPT_IMAGE_PROTOCOL") or os.getenv("VOLC_PROTOCOL") or "openai"
    ).lower()
    if not api_key or not base_url:
        print("配置错误：缺少 API Key 或 Base URL。", file=sys.stderr)
        print(
            "请设置 GPT_IMAGE_API_KEY / GPT_IMAGE_BASE_URL（或 VOLC_API_KEY / VOLC_BASE_URL），"
            "或显式传 --api-key / --base-url。",
            file=sys.stderr,
        )
        return 1
    if protocol not in SUPPORTED_PROTOCOLS:
        print(f"配置错误：图片探针仅支持 {', '.join(SUPPORTED_PROTOCOLS)}，收到 {protocol}。", file=sys.stderr)
        return 1

    print("开始 GPT Image 探针")
    print(f"接口：{image_endpoint(base_url, protocol)}")
    print(f"模型：{args.model}")
    print(f"请求：size={args.size} quality={args.quality} n=1")

    timeout = httpx.Timeout(args.timeout, connect=30.0, write=30.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout, verify=True, trust_env=False) as client:
        row = await run_probe(
            client,
            model=args.model,
            protocol=protocol,
            base_url=base_url,
            api_key=api_key,
            size=args.size,
            quality=args.quality,
            prompt=args.prompt,
            save_image=args.save_image,
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"完整结果：{args.output.resolve()}")

    print("\n探针结果")
    print(f"HTTP 状态：{row.get('http_status')}")
    print(f"返回图片数：{row.get('image_count', 0)}")
    print(f"结果类型：{', '.join(row.get('result_types') or []) or '—'}")
    print(f"耗时：{row['elapsed_seconds']} 秒")
    if row.get("saved_image"):
        print(f"已保存图片：{row['saved_image']}")

    code = probe_exit_code(row)
    if code == 0:
        print(f"\n结论：{args.model} 可调用，HTTP {row['http_status']} 且返回了有效图片。")
        return 0
    if code == 2:
        print(f"\n结论：{args.model} 接口可达但响应不可用：{row.get('error_type')}。")
        print("提示：上游未按 Images 协议返回 data 图片数组，请检查适配器实现。")
    else:
        print(f"\n结论：{args.model} 调用失败：{row.get('error_type')}。")
    message = row.get("error_message") or row.get("error_body") or row.get("body_preview")
    if message:
        print(f"错误：{str(message)[:800]}")
    return code


def cli() -> None:
    try:
        raise SystemExit(asyncio.run(main_async()))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    cli()
