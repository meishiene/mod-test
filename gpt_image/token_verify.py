from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from common.audit import AuditLog, redact
from common.finalizer import finalize_result_dir
from common.image_store import write_named_image
from .naming import image_filename_stem
from common.report_style import stats_lines
from common.image_fingerprint import (
    build_image_fingerprints,
    fingerprint_lines,
    image_signature,
)
from common.image_c2pa import build_image_c2pa, c2pa_lines

from .runner import IMAGE_COMPATIBLE_PROTOCOLS, image_endpoint, request_headers


CASE_ID = "GPT_IMAGE_4K_TOKEN_VERIFY"
MODEL_SPECS = (
    ("gpt-image-2", "high"),
    ("gpt-image-2.5-flare", "max"),
    ("gpt-image-2.5-sunburst", "max"),
)
MODELS = tuple(model for model, _quality in MODEL_SPECS)
SIZE = "3840x2160"
EXPECTED_DIMENSIONS = (3840, 2160)
OUTPUT_FORMAT = "png"
IMAGE_COUNT = 10
PROMPT = (
    "Create a highly detailed cinematic photograph of a futuristic coastal city "
    "at sunrise, with realistic architecture, water reflections, people, trees, "
    "and layered clouds; natural colors, no text, no logo."
)
TEXT_BYTES_PER_TOKEN = 4
IMAGE_BYTES_PER_TOKEN = 768
MIN_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_BYTES = 12 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 1800.0


def converted_tokens(byte_count: int, bytes_per_token: int) -> int:
    """Convert payload bytes with this verification case's ceiling rule."""
    return math.ceil(byte_count / bytes_per_token) if byte_count else 0


def expected_input_tokens(prompt: str) -> int:
    return converted_tokens(len(prompt.encode("utf-8")), TEXT_BYTES_PER_TOKEN)


def expected_output_tokens(image_sizes: list[int]) -> int:
    return sum(converted_tokens(size, IMAGE_BYTES_PER_TOKEN) for size in image_sizes)


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _check(actual: Any, expected: Any, detail: str) -> dict[str, Any]:
    return {
        "passed": actual == expected,
        "actual": actual,
        "expected": expected,
        "detail": detail,
    }


def verify_usage(
    usage: Any, prompt: str, images: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    usage = usage if isinstance(usage, dict) else {}
    input_details = usage.get("input_tokens_details")
    input_details = input_details if isinstance(input_details, dict) else {}
    output_details = usage.get("output_tokens_details")
    output_details = output_details if isinstance(output_details, dict) else {}
    input_tokens = expected_input_tokens(prompt)
    output_tokens = expected_output_tokens([int(item["bytes"]) for item in images])
    return {
        "input_tokens_match": _check(
            _integer(usage.get("input_tokens")),
            input_tokens,
            f"ceil(prompt UTF-8 bytes / {TEXT_BYTES_PER_TOKEN})",
        ),
        "input_text_tokens_match": _check(
            _integer(input_details.get("text_tokens")),
            input_tokens,
            "input_tokens_details.text_tokens 与提示词换算值一致",
        ),
        "output_tokens_match": _check(
            _integer(usage.get("output_tokens")),
            output_tokens,
            f"sum(ceil(each decoded image bytes / {IMAGE_BYTES_PER_TOKEN}))",
        ),
        "output_image_tokens_match": _check(
            _integer(output_details.get("image_tokens")),
            output_tokens,
            "output_tokens_details.image_tokens 与逐图换算值之和一致",
        ),
        "total_tokens_match": _check(
            _integer(usage.get("total_tokens")),
            input_tokens + output_tokens,
            "total_tokens = input_tokens + output_tokens",
        ),
    }


def _image_extension(image_format: str | None) -> str:
    return {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}.get(
        (image_format or "").upper(), "bin"
    )


def _decode_image(value: str) -> bytes:
    payload = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    return base64.b64decode(payload, validate=True)


def _inspect_and_save_image(
    encoded: str, *, model: str, quality: str, index: int, image_dir: Path
) -> dict[str, Any]:
    raw = _decode_image(encoded)
    with Image.open(io.BytesIO(raw)) as image:
        dimensions = tuple(image.size)
        image_format = image.format
        signature = image_signature(image)
    # verify() 必须在 open 之后立刻调用（加载过像素就会报 RuntimeError），
    # 所以指纹在另一个句柄上算完再单独校验完整性。
    with Image.open(io.BytesIO(raw)) as verifier:
        verifier.verify()
    extension = _image_extension(image_format)
    path = write_named_image(image_dir, image_filename_stem(model, quality), extension, raw)
    byte_count = len(raw)
    return {
        "index": index,
        "path": str(path.relative_to(image_dir.parent)),
        "format": image_format,
        "width": dimensions[0],
        "height": dimensions[1],
        "bytes": byte_count,
        "mib": round(byte_count / 1024 / 1024, 3),
        "converted_tokens": converted_tokens(byte_count, IMAGE_BYTES_PER_TOKEN),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dhash": signature["dhash"],
        "chash": signature["chash"],
        "dimensions_match": dimensions == EXPECTED_DIMENSIONS,
        "size_around_10m": MIN_IMAGE_BYTES <= byte_count <= MAX_IMAGE_BYTES,
    }


async def run_model(
    client: httpx.AsyncClient,
    *,
    model: str,
    quality: str,
    protocol: str,
    base_url: str,
    api_key: str,
    image_dir: Path,
) -> dict[str, Any]:
    request_id = f"{CASE_ID}-{model}-{uuid.uuid4().hex[:10]}"
    payload = {
        "model": model,
        "prompt": PROMPT,
        "size": SIZE,
        "quality": quality,
        "output_format": OUTPUT_FORMAT,
        "n": IMAGE_COUNT,
    }
    started_epoch = time.time()
    started_mono = time.perf_counter()
    row: dict[str, Any] = {
        "case": CASE_ID,
        "request_id": request_id,
        "model": model,
        "endpoint": image_endpoint(base_url, protocol),
        "prompt": PROMPT,
        "prompt_utf8_bytes": len(PROMPT.encode("utf-8")),
        "expected_input_tokens": expected_input_tokens(PROMPT),
        "size": SIZE,
        "quality": quality,
        "output_format": OUTPUT_FORMAT,
        "n": IMAGE_COUNT,
        "started_at": datetime.fromtimestamp(started_epoch, timezone.utc).isoformat(),
    }
    try:
        response = await client.post(
            image_endpoint(base_url, protocol),
            json=payload,
            headers=request_headers(protocol, api_key, request_id),
        )
        row["http_status"] = response.status_code
        if not 200 <= response.status_code < 300:
            row.update(
                success=False,
                case_passed=False,
                case_status="failed",
                error_type=f"http_{response.status_code}",
                error_message=response.text[:2000],
            )
            return row
        body = response.json()
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise ValueError("响应缺少 data 图片数组")
        images: list[dict[str, Any]] = []
        row["images"] = images
        for index, item in enumerate(data, start=1):
            encoded = item.get("b64_json") if isinstance(item, dict) else None
            if not isinstance(encoded, str) or not encoded:
                raise ValueError(f"第 {index} 张图片缺少 b64_json")
            images.append(
                _inspect_and_save_image(
                    encoded, model=model, quality=quality, index=index, image_dir=image_dir
                )
            )
            row["image_count"] = len(images)
        usage = body.get("usage") if isinstance(body, dict) else None
        checks = verify_usage(usage, PROMPT, images)
        checks["image_count_match"] = _check(
            len(images), IMAGE_COUNT, "响应图片数与 n 一致"
        )
        checks["all_dimensions_4k"] = _check(
            sum(bool(item["dimensions_match"]) for item in images),
            IMAGE_COUNT,
            f"每张图片均为 {SIZE}",
        )
        checks["all_images_around_10m"] = _check(
            sum(bool(item["size_around_10m"]) for item in images),
            IMAGE_COUNT,
            "每张图片均在 8–12 MiB",
        )
        passed = all(check["passed"] for check in checks.values())
        row.update(
            success=passed,
            case_passed=passed,
            case_status="passed" if passed else "failed",
            response_model=body.get("model"),
            response_quality=body.get("quality"),
            response_size=body.get("size"),
            usage=usage,
            image_count=len(images),
            image_total_bytes=sum(int(item["bytes"]) for item in images),
            expected_output_tokens=expected_output_tokens(
                [int(item["bytes"]) for item in images]
            ),
            images=images,
            checks=checks,
            error_type=None if passed else "verification_failed",
            error_message=None if passed else "一个或多个 4K、文件大小或 token 核验项未通过",
        )
        return row
    except httpx.TimeoutException as exc:
        row.update(success=False, case_passed=False, case_status="failed", error_type="timeout", error_message=str(exc))
        return row
    except Exception as exc:
        row.update(success=False, case_passed=False, case_status="failed", error_type=type(exc).__name__, error_message=str(exc))
        return row
    finally:
        row["elapsed_seconds"] = time.perf_counter() - started_mono
        row["completed_epoch"] = time.time()


def _write_report(out: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# GPT Image 4K / n=10 / Token 验真报告",
        "",
        *stats_lines(summary['requested'], summary['successful'], summary['failed'], 0, 0, None,
                     [f"结果：{'通过' if summary['admission_passed'] else '未通过'}",
                      f"固定参数：size={SIZE}，output_format={OUTPUT_FORMAT}，n={IMAGE_COUNT}",
                      f"模型/质量：{', '.join(f'{model}/{quality}' for model, quality in MODEL_SPECS)}",
                      f"完成请求：{summary['completed']}/{summary['requested']}；有效图片：{summary['generated_images']}/{len(MODELS) * IMAGE_COUNT}",
                      f"输入换算：ceil(提示词 UTF-8 字节数 / {TEXT_BYTES_PER_TOKEN})",
                      f"输出换算：sum(ceil(每张图片解码字节数 / {IMAGE_BYTES_PER_TOKEN}))",
                      f"单图大小门槛：{MIN_IMAGE_BYTES / 1024 / 1024:g}–{MAX_IMAGE_BYTES / 1024 / 1024:g} MiB"]),
        "",
        "| 模型 | 质量 | HTTP | 图片数 | 输入 Token | 输入换算 | 输出 Token | 输出换算 | 单图大小 MiB | 结论 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary["results"]:
        usage = row.get("usage") or {}
        sizes = ", ".join(str(image["mib"]) for image in row.get("images") or []) or "—"
        lines.append(
            f"| {row['model']} | {row['quality']} | {row.get('http_status', '—')} | {row.get('image_count', 0)} | "
            f"{usage.get('input_tokens', '—')} | {row.get('expected_input_tokens', '—')} | "
            f"{usage.get('output_tokens', '—')} | {row.get('expected_output_tokens', '—')} | "
            f"{sizes} | {'通过' if row.get('case_passed') else '未通过'} |"
        )
    lines.extend(
        [
            "",
            "图片原文件保存在 `images/`；逐图字节数、尺寸、SHA-256 与图片指纹（dHash / chash）"
            "和 token 换算值保存在 `requests.jsonl`。",
            *fingerprint_lines(summary.get("image_fingerprints")),
            *c2pa_lines(summary.get("image_c2pa")),
        ]
    )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPT Image 4K / n=10 / Token 一次性能力验证")
    parser.add_argument("case", nargs="?", default=CASE_ID, choices=(CASE_ID,))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    api_key = os.getenv("GPT_IMAGE_API_KEY") or os.getenv("VOLC_API_KEY") or ""
    base_url = os.getenv("GPT_IMAGE_BASE_URL") or os.getenv("VOLC_BASE_URL") or ""
    protocol = (os.getenv("GPT_IMAGE_PROTOCOL") or os.getenv("VOLC_PROTOCOL") or "openai").lower()
    if not api_key or not base_url:
        parser.error("缺少 GPT_IMAGE_API_KEY/VOLC_API_KEY 或 GPT_IMAGE_BASE_URL/VOLC_BASE_URL")
    if protocol not in IMAGE_COMPATIBLE_PROTOCOLS:
        parser.error("图片测试仅支持 OpenAI Images / 火山云 / 百度千帆协议")
    try:
        timeout_seconds = float(os.getenv("GPT_IMAGE_4K_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
    except ValueError:
        parser.error("GPT_IMAGE_4K_TIMEOUT_SECONDS 必须是数字")
    if timeout_seconds <= 0:
        parser.error("GPT_IMAGE_4K_TIMEOUT_SECONDS 必须大于 0")

    run_id = os.getenv("VOLC_RECORD_ID") or uuid.uuid4().hex[:16]
    out = args.output_root / f"{CASE_ID}-{time.strftime('%Y%m%d-%H%M%S')}-{run_id[:6]}"
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": "mltest.gpt-image-token-verify.v1",
        "suite": "gpt_image",
        "test_kind": "gpt-image",
        "case": CASE_ID,
        "record_id": run_id,
        "protocol": protocol,
        "base_url": base_url,
        "model_specs": [
            {"model": model, "quality": quality} for model, quality in MODEL_SPECS
        ],
        "size": SIZE,
        "n": IMAGE_COUNT,
        "prompt": PROMPT,
        "timeout_seconds": timeout_seconds,
        "token_conversion": {
            "input": f"ceil(prompt_utf8_bytes/{TEXT_BYTES_PER_TOKEN})",
            "output": f"sum(ceil(each_decoded_image_bytes/{IMAGE_BYTES_PER_TOKEN}))",
        },
        "image_size_range_bytes": [MIN_IMAGE_BYTES, MAX_IMAGE_BYTES],
        "official_reference": "https://platform.openai.com/docs/guides/image-generation",
    }
    (out / "metadata.json").write_text(
        json.dumps(redact(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    timeout = httpx.Timeout(timeout_seconds, connect=30.0, write=30.0, pool=30.0)
    rows: list[dict[str, Any]] = []
    audit = AuditLog(out / "requests.jsonl")
    try:
        async with httpx.AsyncClient(timeout=timeout, verify=True, trust_env=False) as client:
            for model, quality in MODEL_SPECS:
                print(f"{model}: 正在生成 {IMAGE_COUNT} 张 {SIZE} {quality} 图片…", flush=True)
                row = await run_model(
                    client,
                    model=model,
                    quality=quality,
                    protocol=protocol,
                    base_url=base_url,
                    api_key=api_key,
                    image_dir=image_dir,
                )
                rows.append(row)
                audit.write(row)
                print(f"{model}: {'通过' if row['case_passed'] else '未通过'}", flush=True)
    finally:
        audit.close()

    passed = all(row.get("case_passed") is True for row in rows) and len(rows) == len(MODELS)
    generated_images = sum(int(row.get("image_count") or 0) for row in rows)
    # 图片指纹：与压测套件同一套口径（行缺 request 序号，这里按执行顺序补上）。
    indexed_rows = [{"index": index, **row} for index, row in enumerate(rows, 1)]
    fingerprints = build_image_fingerprints(indexed_rows)
    # C2PA 内容签名（官方指纹）：签名主体/签发者/生成器/水印，从落盘原图读取。
    c2pa = build_image_c2pa(indexed_rows, out)
    summary = {
        "schema_version": "mltest.gpt-image-token-verify.v1",
        "record_id": run_id,
        "suite": "gpt_image",
        "test_kind": "gpt-image",
        "case": CASE_ID,
        "protocol": protocol,
        "base_url": base_url,
        "model": " + ".join(MODELS),
        "models": list(MODELS),
        "requested": len(MODELS),
        "sent": len(rows),
        "completed": len(rows),
        "successful": sum(row.get("case_passed") is True for row in rows),
        "failed": sum(row.get("case_passed") is not True for row in rows),
        "generated_images": generated_images,
        "checks": {
            "all_models_completed": _check(len(rows), len(MODELS), "每个模型各请求一次"),
            "all_model_verifications_passed": _check(
                sum(row.get("case_passed") is True for row in rows),
                len(MODELS),
                "三个模型的 4K、大小和 usage 核验全部通过",
            ),
            "exact_image_count": _check(
                generated_images,
                len(MODELS) * IMAGE_COUNT,
                "三个请求均返回 n=10",
            ),
        },
        "results": rows,
        "image_fingerprints": fingerprints,
        "image_c2pa": c2pa,
        "admission_passed": passed and generated_images == len(MODELS) * IMAGE_COUNT,
    }
    (out / "summary.json").write_text(
        json.dumps(redact(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(out, summary)
    finalized = finalize_result_dir(out)
    print(json.dumps(finalized, ensure_ascii=False, indent=2))
    print(f"结果目录: {out.resolve()}")
    return 0 if finalized["admission_passed"] else 2


def cli() -> None:
    try:
        raise SystemExit(asyncio.run(main_async()))
    except KeyboardInterrupt:
        raise SystemExit(130)
