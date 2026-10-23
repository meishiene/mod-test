import asyncio
import base64
import hashlib
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image

from common.image_store import (
    IMAGE_DIR_NAME,
    ImageBudget,
    decode_base64_image,
    env_flag,
    format_budget,
    image_extension,
    persist_images,
    probe_image,
    summarize_images,
)
from common.image_fingerprint import color_distance, hamming_distance


def _png(size: tuple[int, int] = (4, 4), color: tuple[int, int, int] = (10, 120, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _encoded(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def test_decode_accepts_raw_base64_and_data_uri():
    raw = _png()
    encoded = _encoded(raw)
    assert decode_base64_image(encoded) == raw
    assert decode_base64_image(f"data:image/png;base64,{encoded}") == raw


def test_extension_prefers_real_image_format_then_mime():
    assert image_extension("image/png", "JPEG") == "jpg"
    assert image_extension("image/webp", None) == "webp"
    assert image_extension("image/png", "not-a-real-format") == "png"
    assert image_extension(None, None) == "bin"


def test_probe_reports_dimensions_and_tolerates_garbage():
    probe = probe_image(_png((6, 3)))
    assert (probe["format"], probe["width"], probe["height"]) == ("PNG", 6, 3)
    assert isinstance(probe["dhash"], str) and len(probe["dhash"]) == 16
    assert isinstance(probe["chash"], str) and len(probe["chash"]) == 96
    assert probe_image(b"not an image at all") == {
        "format": None, "width": None, "height": None, "dhash": None, "chash": None,
    }


def test_probe_dhash_is_stable_for_same_pixels_and_differs_for_other_images():
    raw = _png((32, 32), (10, 120, 200))
    probe = probe_image(raw)
    assert probe["dhash"] == probe_image(raw)["dhash"]
    # 同图重编码（→JPEG）字节不同，但明暗差与配色差都应保持极小。
    buffer = io.BytesIO()
    Image.open(io.BytesIO(raw)).convert("RGB").save(buffer, format="JPEG", quality=95)
    reencoded = probe_image(buffer.getvalue())
    luma = hamming_distance(probe["dhash"], reencoded["dhash"])
    assert luma is not None and luma <= 6
    assert color_distance(probe["chash"], reencoded["chash"]) <= 4
    # 纯色图 dHash 会撞成全零，靠配色指纹区分。
    assert probe["dhash"] == probe_image(_png((32, 32), (0, 0, 0)))["dhash"]
    assert color_distance(probe["chash"], probe_image(_png((32, 32), (0, 0, 0)))["chash"]) > 12


def test_env_flag_parses_boolean_switches():
    assert env_flag("IMAGE_STORE_TEST_FLAG", True) is True
    for value in ("0", "false", "no", "off", "FALSE"):
        with patch.dict(os.environ, {"IMAGE_STORE_TEST_FLAG": value}):
            assert env_flag("IMAGE_STORE_TEST_FLAG", True) is False
    with patch.dict(os.environ, {"IMAGE_STORE_TEST_FLAG": "1"}):
        assert env_flag("IMAGE_STORE_TEST_FLAG", False) is True


def test_persist_images_writes_files_and_never_returns_base64(tmp_path):
    raw = _png((5, 5))
    encoded = _encoded(raw)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404))) as client:
            return await persist_images(
                client,
                tmp_path / IMAGE_DIR_NAME,
                request_key="req-0001",
                sources=[
                    {"sequence": 1, "b64": encoded, "mime_type": "image/png", "source": "inlineData"},
                ],
            )

    saved = asyncio.run(run())

    assert len(saved) == 1
    entry = saved[0]
    assert entry["file"] == "images/req-0001-01.png"
    assert entry["saved"] is True
    assert entry["format"] == "PNG"
    assert entry["width"] == 5 and entry["height"] == 5
    assert entry["bytes"] == len(raw)
    assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
    assert isinstance(entry["dhash"], str) and len(entry["dhash"]) == 16
    assert isinstance(entry["chash"], str) and len(entry["chash"]) == 96
    assert (tmp_path / entry["file"]).read_bytes() == raw
    assert encoded not in json.dumps(saved)


def test_persist_images_downloads_url_results(tmp_path):
    raw = _png((3, 9))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cdn.test":
            return httpx.Response(200, content=raw, headers={"content-type": "image/png"})
        return httpx.Response(404)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await persist_images(
                client,
                tmp_path / IMAGE_DIR_NAME,
                request_key="req-0002",
                sources=[{"sequence": 1, "url": "https://cdn.test/a.png", "source": "url"}],
            )

    saved = asyncio.run(run())

    assert saved[0]["saved"] is True
    assert saved[0]["file"] == "images/req-0002-01.png"
    assert saved[0]["width"] == 3 and saved[0]["height"] == 9
    assert saved[0]["url"] == "https://cdn.test/a.png"
    assert (tmp_path / saved[0]["file"]).read_bytes() == raw


def test_persist_images_records_failures_without_raising(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "expired"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await persist_images(
                client,
                tmp_path / IMAGE_DIR_NAME,
                request_key="req-0003",
                sources=[
                    {"sequence": 1, "url": "https://cdn.test/gone.png", "source": "url"},
                    {"sequence": 2, "b64": "!!!not-base64!!!", "source": "b64_json"},
                ],
            )

    saved = asyncio.run(run())

    assert [entry["saved"] for entry in saved] == [False, False]
    assert "下载失败" in saved[0]["error"]
    assert "base64 解码失败" in saved[1]["error"]
    assert not (tmp_path / IMAGE_DIR_NAME).exists()


def test_persist_images_is_a_noop_when_disabled_or_empty(tmp_path):
    async def run(sources):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404))) as client:
            return await persist_images(
                client, None, request_key="req-0004", sources=sources
            )

    assert asyncio.run(run([{"sequence": 1, "b64": _encoded(_png())}])) == []
    assert asyncio.run(run([])) == []
    assert not (tmp_path / IMAGE_DIR_NAME).exists()


def test_image_budget_caps_saved_bytes_and_flags_the_rest(tmp_path):
    raw = _png((16, 16))
    budget = ImageBudget(len(raw) + 1)  # 只够一张

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404))) as client:
            return await persist_images(
                client,
                tmp_path / IMAGE_DIR_NAME,
                request_key="req-0005",
                sources=[
                    {"sequence": index, "b64": _encoded(raw), "mime_type": "image/png", "source": "inlineData"}
                    for index in (1, 2)
                ],
                budget=budget,
            )

    saved = asyncio.run(run())

    assert [entry["saved"] for entry in saved] == [True, False]
    assert "达到图片保留上限" in saved[1]["error"]
    assert budget.saved_files == 1 and budget.skipped_files == 1
    assert len(list((tmp_path / IMAGE_DIR_NAME).glob("*.png"))) == 1
    assert ImageBudget(0).unlimited is True and ImageBudget(0).claim(10**9) is True


def test_format_budget_is_human_readable():
    assert format_budget(None) == "不限制"
    assert format_budget(0) == "不限制"
    assert format_budget(5 * 1024 * 1024) == "5 MiB"
    assert format_budget(20 * 1024**3) == "20.0 GiB"


def test_summarize_images_counts_saved_skipped_and_bytes():
    rows = [
        {"images": [{"saved": True, "bytes": 100}, {"saved": False, "error": "x"}]},
        {"images": []},
        {"images": [{"saved": True, "bytes": 50}]},
    ]
    assert summarize_images(rows) == {
        "images_saved": 2,
        "image_bytes_saved": 150,
        "images_skipped": 1,
    }
