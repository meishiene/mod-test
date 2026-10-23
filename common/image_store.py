"""把图片响应中的图片内容落盘到结果目录。

Gemini 把图片放在 ``candidates[].content.parts[].inlineData``（base64），OpenAI
Images 放在 ``data[].b64_json``（base64）或 ``data[].url``。两个图片套件都通过这里
保存原始文件：请求行只记录文件路径与校验信息，base64 内容永不进入证据文件。

设计约束：

* 落盘失败不得改变请求判定——被测模型的能力结论不能由本地磁盘决定；
* 单次请求内多张图片按 ``sequence`` 编号，文件名可预测（``req-0001-01.png``）；
* 结果目录体积可控：``ImageBudget`` 限制单批保留的总字节数，超出部分显式记录为
  ``saved=False``，绝不静默丢弃。
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import os
import threading
from pathlib import Path
from typing import Any, Iterable

import httpx
from PIL import Image

from .image_fingerprint import image_signature


IMAGE_DIR_NAME = "images"
FALLBACK_EXTENSION = "bin"
KEEP_IMAGES_ENV_DEFAULT = "1"

# PIL 识别出的真实格式优先（b64_json 不带 MIME），其次回落到响应声明的 MIME。
FORMAT_EXTENSIONS = {
    "PNG": "png",
    "JPEG": "jpg",
    "MPO": "jpg",
    "WEBP": "webp",
    "GIF": "gif",
    "BMP": "bmp",
    "TIFF": "tiff",
    "AVIF": "avif",
}
MIME_EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "image/x-ms-bmp": "bmp",
    "image/tiff": "tiff",
    "image/avif": "avif",
}


def env_flag(name: str, default: bool = True) -> bool:
    """读取布尔环境变量；未设置或空值用默认值。"""
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def decode_base64_image(value: str) -> bytes:
    """解开 ``b64_json`` / ``inlineData.data``，容忍 ``data:`` URL 前缀。"""
    payload = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    # validate=False：上游偶发换行或 URL-safe 变体不应导致整张图片丢失。
    return base64.b64decode(payload, validate=False)


def probe_image(raw: bytes) -> dict[str, Any]:
    """尽力解析图片元数据（含 dHash / 平均色指纹）；解析失败不影响落盘（只记 None）。"""
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image_format = image.format
            width, height = image.size
            signature = image_signature(image)
    except Exception:  # 非图片字节也要留档，不能因此丢掉证据
        return {
            "format": None,
            "width": None,
            "height": None,
            "dhash": None,
            "chash": None,
        }
    return {
        "format": image_format,
        "width": int(width),
        "height": int(height),
        "dhash": signature["dhash"],
        "chash": signature["chash"],
    }


def image_extension(mime_type: str | None, image_format: str | None) -> str:
    if image_format and str(image_format).upper() in FORMAT_EXTENSIONS:
        return FORMAT_EXTENSIONS[str(image_format).upper()]
    if mime_type and str(mime_type).lower() in MIME_EXTENSIONS:
        return MIME_EXTENSIONS[str(mime_type).lower()]
    return FALLBACK_EXTENSION


class ImageBudget:
    """限制单批结果目录保留的图片总量；``limit_bytes<=0`` 表示不限制。

    ``claim`` 在多协程/多线程下并发调用，因此内部加锁。
    """

    def __init__(self, limit_bytes: int = 0) -> None:
        self.limit_bytes = max(0, int(limit_bytes or 0))
        self._lock = threading.Lock()
        self.saved_files = 0
        self.saved_bytes = 0
        self.skipped_files = 0
        self.skipped_bytes = 0

    @property
    def unlimited(self) -> bool:
        return self.limit_bytes <= 0

    @property
    def remaining_bytes(self) -> int | None:
        if self.unlimited:
            return None
        return max(0, self.limit_bytes - self.saved_bytes)

    def claim(self, size: int) -> bool:
        """为一个待写文件预占额度；返回 False 表示已达上限。"""
        with self._lock:
            if self.unlimited:
                self.saved_files += 1
                self.saved_bytes += size
                return True
            if self.saved_bytes + size > self.limit_bytes:
                self.skipped_files += 1
                self.skipped_bytes += size
                return False
            self.saved_files += 1
            self.saved_bytes += size
            return True


def write_named_image(directory: Path, stem: str, extension: str, raw: bytes) -> Path:
    """Atomically reserve a readable name; concurrent images never overwrite each other."""
    if not stem or Path(stem).name != stem or stem in {".", ".."}:
        raise ValueError("invalid image filename")
    directory.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        name = f"{stem}{'' if suffix == 1 else f'-{suffix:02d}'}.{extension}"
        path = directory / name
        try:
            handle = path.open("xb")
        except FileExistsError:
            suffix += 1
            continue
        with handle:
            handle.write(raw)
        return path


def _write_images(
    directory: Path,
    request_key: str,
    items: list[tuple[dict[str, Any], bytes]],
    budget: ImageBudget | None,
    filename_stem: str | None = None,
) -> list[dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, Any]] = []
    for source, raw in items:
        sequence = int(source.get("sequence") or len(saved) + 1)
        if budget is not None and not budget.claim(len(raw)):
            saved.append(
                {
                    "sequence": sequence,
                    "source": source.get("source"),
                    "url": source.get("url"),
                    "saved": False,
                    "bytes": len(raw),
                    "error": (
                        f"达到图片保留上限（{budget.limit_bytes} 字节），已跳过；"
                        "可提高 --max-image-gib 或设置 GEMINI_IMAGE_MAX_IMAGE_GIB / GPT_IMAGE_MAX_IMAGE_GIB"
                    ),
                }
            )
            continue
        probe = probe_image(raw)
        extension = image_extension(source.get("mime_type"), probe["format"])
        if filename_stem:
            name = write_named_image(directory, filename_stem, extension, raw).name
        else:
            name = f"{request_key}-{sequence:02d}.{extension}"
            (directory / name).write_bytes(raw)
        saved.append(
            {
                "sequence": sequence,
                "file": f"{directory.name}/{name}",
                "source": source.get("source"),
                "url": source.get("url"),
                "saved": True,
                "mime_type": source.get("mime_type"),
                "format": probe["format"],
                "width": probe["width"],
                "height": probe["height"],
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                # 图片指纹：dHash 记明暗结构、chash 记配色（报告里据此判重复/近似重复）。
                "dhash": probe["dhash"],
                "chash": probe["chash"],
            }
        )
    return saved


async def persist_images(
    client: httpx.AsyncClient | None,
    directory: Path | None,
    *,
    request_key: str,
    sources: Iterable[dict[str, Any]],
    budget: ImageBudget | None = None,
    filename_stem: str | None = None,
) -> list[dict[str, Any]]:
    """保存一次请求返回的全部图片，返回逐图元数据。

    ``sources`` 每项支持两种形态：``{"b64": ...}``（直接解码）或
    ``{"url": ...}``（用 ``client`` 下载）。``directory`` 为 None 表示本次不保留图片。

    任何单张图片失败都只记录 ``saved=False`` + ``error``，不抛异常。
    """
    resolved = [source for source in sources if isinstance(source, dict)]
    if directory is None or not resolved:
        return []
    items: list[tuple[dict[str, Any], bytes]] = []
    failures: list[dict[str, Any]] = []
    for source in resolved:
        encoded = source.get("b64")
        if isinstance(encoded, str) and encoded:
            try:
                raw = await asyncio.to_thread(decode_base64_image, encoded)
            except Exception as exc:
                failures.append(
                    {
                        "sequence": source.get("sequence"),
                        "source": source.get("source"),
                        "saved": False,
                        "error": f"base64 解码失败: {type(exc).__name__}: {exc}",
                    }
                )
                continue
            items.append((source, raw))
            continue
        url = source.get("url")
        if isinstance(url, str) and url and client is not None:
            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                failures.append(
                    {
                        "sequence": source.get("sequence"),
                        "source": source.get("source"),
                        "url": url,
                        "saved": False,
                        "error": f"下载失败: {type(exc).__name__}: {exc}",
                    }
                )
                continue
            items.append((source, response.content))
    saved = (
        await asyncio.to_thread(_write_images, directory, request_key, items, budget, filename_stem)
        if items
        else []
    )
    return saved + failures


def format_budget(max_image_bytes: int | None) -> str:
    """把保留上限渲染成人类可读文本（报告用）。"""
    if not max_image_bytes or max_image_bytes <= 0:
        return "不限制"
    mib = max_image_bytes / 1024 / 1024
    return f"{mib / 1024:.1f} GiB" if mib >= 1024 else f"{mib:.0f} MiB"


def summarize_images(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """按落盘结果汇总图片数量与体积，供 summary/report 使用。"""
    saved = 0
    saved_bytes = 0
    skipped = 0
    for row in rows:
        for image in row.get("images") or []:
            if not isinstance(image, dict):
                continue
            if image.get("saved"):
                saved += 1
                saved_bytes += int(image.get("bytes") or 0)
            else:
                skipped += 1
    return {"images_saved": saved, "image_bytes_saved": saved_bytes, "images_skipped": skipped}
