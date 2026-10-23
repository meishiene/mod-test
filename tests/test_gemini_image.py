import asyncio
import base64
import io
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image

from gemini_image.cases import (
    GEMINI_IMAGE_ASPECT_RATIOS,
    GEMINI_IMAGE_COUNTS,
    GEMINI_IMAGE_QUALITIES,
    CASES,
)
from gemini_image.runner import (
    _failed_request_row,
    _parse_choice_list,
    _parse_count_list,
    _request_one,
    _run_requests,
    build_payload,
    build_request_plan,
    image_endpoint,
    main_async,
)


def test_case_matrix_matches_official_gemini_image_spec():
    case = CASES["GEMINI_IMAGE_GEN"]
    assert GEMINI_IMAGE_QUALITIES == ("1K", "2K", "4K")
    assert GEMINI_IMAGE_COUNTS == (1, 2, 3, 4)
    assert set(GEMINI_IMAGE_ASPECT_RATIOS) == {
        "1:1", "3:2", "2:3", "3:4", "4:3",
        "4:5", "5:4", "9:16", "16:9", "21:9",
    }
    assert case.models == ("gemini-3-pro-image", "gemini-3.1-flash-image")
    assert case.qualities == GEMINI_IMAGE_QUALITIES
    assert case.counts == GEMINI_IMAGE_COUNTS
    # 用例以协议 family 声明支持面，前端/后端据此与 gemini_images 协议匹配。
    assert case.protocols == ("gemini",)


def test_endpoint_targets_native_generate_content():
    assert (
        image_endpoint("https://mux.la", "gemini-3-pro-image")
        == "https://mux.la/v1beta/models/gemini-3-pro-image:generateContent"
    )
    # 已经带 /v1beta 的 base 不得重复拼接版本段。
    assert (
        image_endpoint("https://generativelanguage.googleapis.com/v1beta", "gemini-3.1-flash-image")
        == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image:generateContent"
    )


def test_payload_uses_native_field_names():
    payload = build_payload("a cat", "2K", "16:9", 3)
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "a cat"}]}]
    generation = payload["generationConfig"]
    assert generation["responseModalities"] == ["TEXT", "IMAGE"]
    assert generation["imageConfig"] == {"aspectRatio": "16:9", "imageSize": "2K"}
    assert generation["candidateCount"] == 3


def test_request_sends_image_config_and_never_stores_inline_bytes():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta/models/gemini-3-pro-image:generateContent"
        assert request.url.host == "mux.la"
        assert request.headers["x-goog-api-key"] == "test-secret"
        body = json.loads(request.read())
        assert body["generationConfig"]["imageConfig"] == {"aspectRatio": "16:9", "imageSize": "2K"}
        assert body["generationConfig"]["candidateCount"] == 2
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "here you go"},
                                {"inlineData": {"mimeType": "image/png", "data": "a-very-large-image-payload"}},
                                {"inlineData": {"mimeType": "image/png", "data": "another-large-payload"}},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 2240,
                    "thoughtsTokenCount": 264,
                    "totalTokenCount": 2516,
                },
                "modelVersion": "gemini-3-pro-image",
                "responseId": "resp-1",
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _request_one(
                client,
                request_plan={
                    "case": "GEMINI_IMAGE_GEN",
                    "index": 1,
                    "model": "gemini-3-pro-image",
                    "quality": "2K",
                    "aspect_ratio": "16:9",
                    "count": 2,
                    "prompt": "a cat",
                    "payload": build_payload("a cat", "2K", "16:9", 2),
                },
                base_url="https://mux.la",
                api_key="test-secret",
            )

    row = asyncio.run(run())
    assert row["success"] is True
    assert row["image_count"] == 2
    assert row["count_honored"] is True
    assert row["finish_reasons"] == ["STOP"]
    assert row["response_model"] == "gemini-3-pro-image"
    assert row["usage"]["thoughtsTokenCount"] == 264
    # 图片 base64 绝不能进入证据文件。
    assert "a-very-large-image-payload" not in str(row)
    assert "another-large-payload" not in str(row)


def _png_b64(size: tuple[int, int] = (7, 7)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 120, 40)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _image_plan(index: int = 1, count: int = 2) -> dict:
    return {
        "case": "GEMINI_IMAGE_GEN",
        "index": index,
        "model": "gemini-3-pro-image",
        "quality": "2K",
        "aspect_ratio": "16:9",
        "count": count,
        "prompt": "a cat",
        "payload": build_payload("a cat", "2K", "16:9", count),
    }


def _run_image_request(handler, image_dir=None, plan=None, **kwargs):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _request_one(
                client,
                request_plan=plan or _image_plan(),
                base_url="https://mux.la",
                api_key="secret",
                image_dir=image_dir,
                **kwargs,
            )

    return asyncio.run(run())


def test_inline_images_are_saved_and_evidence_keeps_paths_only(tmp_path):
    encoded = _png_b64()

    def handler(request: httpx.Request) -> httpx.Response:
        parts = [
            {"text": "here you go"},
            {"inlineData": {"mimeType": "image/png", "data": encoded}},
            {"inlineData": {"mimeType": "image/png", "data": encoded}},
        ]
        return httpx.Response(200, json={"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}]})

    row = _run_image_request(handler, image_dir=tmp_path / "images")

    assert row["success"] is True
    assert row["image_save_error"] is None
    assert [entry["file"] for entry in row["images"]] == [
        "images/req-0001-01.png",
        "images/req-0001-02.png",
    ]
    assert all(entry["saved"] is True for entry in row["images"])
    assert row["images"][0]["source"] == "inlineData"
    assert (row["images"][0]["width"], row["images"][0]["height"]) == (7, 7)
    assert (tmp_path / "images" / "req-0001-02.png").exists()
    # base64 内容绝不进入证据文件。
    assert encoded not in json.dumps(row)


def test_file_data_uri_result_is_downloaded_into_images_dir(tmp_path):
    encoded = _png_b64((5, 5))
    raw = base64.b64decode(encoded)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "files.test":
            return httpx.Response(200, content=raw, headers={"content-type": "image/png"})
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"fileData": {"mimeType": "image/png", "fileUri": "https://files.test/a.png"}}
                            ]
                        }
                    }
                ]
            },
        )

    row = _run_image_request(handler, image_dir=tmp_path / "images")

    entry = row["images"][0]
    assert entry["saved"] is True and entry["source"] == "fileData"
    assert entry["url"] == "https://files.test/a.png"
    assert (tmp_path / "images" / "req-0001-01.png").read_bytes() == raw


def test_http_success_without_inline_image_is_a_failure():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "text only"}]}}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _request_one(
                client,
                request_plan={
                    "case": "GEMINI_IMAGE_GEN",
                    "index": 1,
                    "model": "gemini-3-pro-image",
                    "quality": "1K",
                    "aspect_ratio": "1:1",
                    "count": 1,
                    "prompt": "t",
                    "payload": build_payload("t", "1K", "1:1", 1),
                },
                base_url="https://mux.la",
                api_key="secret",
            )

    row = asyncio.run(run())
    assert row["success"] is False
    assert row["error_type"] == "invalid_image_response"
    assert row["image_count"] == 0


def test_request_plan_is_complete_reproducible_and_respects_model_support():
    case = CASES["GEMINI_IMAGE_GEN"]
    kwargs = dict(
        case=case,
        requests=24,
        qualities=["1K", "2K", "4K"],
        aspect_ratios=["1:1", "16:9"],
        models=list(case.models),
        counts=[1, 2],
        seed=20261008,
    )
    first = build_request_plan(**kwargs)
    second = build_request_plan(**kwargs)
    assert first == second
    assert len(first) == 24
    assert [item["index"] for item in first] == list(range(1, 25))
    assert all(item["aspect_ratio"] in {"1:1", "16:9"} for item in first)
    assert all(item["count"] in {1, 2} for item in first)
    assert all(
        item["payload"]["generationConfig"]["imageConfig"]["imageSize"] == item["quality"]
        for item in first
    )


def test_cli_choice_parsing_rejects_unknown_values():
    assert _parse_choice_list("", ("1K", "2K"), "强度") == ["1K", "2K"]
    assert _parse_choice_list("1K, 2K ,1K", ("1K", "2K"), "强度") == ["1K", "2K"]
    assert _parse_count_list("", (1, 2, 3, 4)) == [1, 2, 3, 4]
    assert _parse_count_list("2,2,4", (1, 2, 3, 4)) == [2, 4]
    for bad in ("9", "x"):
        try:
            _parse_count_list(bad, (1, 2, 3, 4))
        except ValueError:
            continue
        raise AssertionError(f"{bad} 应被拒绝")
    try:
        _parse_choice_list("8K", ("1K", "2K"), "强度")
    except ValueError:
        pass
    else:
        raise AssertionError("8K 应被拒绝")


def test_absolute_timeout_does_not_block_following_requests_and_flushes_rows():
    plans = [
        {
            "case": "GEMINI_IMAGE_GEN",
            "index": index,
            "model": "gemini-3-pro-image",
            "quality": "1K",
            "aspect_ratio": "1:1",
            "count": 1,
            "prompt": "t",
            "payload": build_payload("t", "1K", "1:1", 1),
        }
        for index in range(1, 4)
    ]

    async def fake_request(
        client, *, request_plan, base_url, api_key,
        image_dir=None, keep_images=True, budget=None,
    ):
        if request_plan["index"] == 1:
            await asyncio.sleep(0.05)
        return _failed_request_row(
            request_plan,
            base_url=base_url,
            started_epoch=0.0,
            elapsed_seconds=0.001,
            error_type="http_503",
            error_message="test",
        )

    async def run(path: Path):
        with patch("gemini_image.runner._request_one", side_effect=fake_request):
            return await _run_requests(
                request_plan=plans,
                concurrency=1,
                base_url="https://mux.la",
                api_key="secret",
                request_timeout_seconds=0.01,
                requests_path=path,
            )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "requests.jsonl"
        rows = asyncio.run(run(path))
        persisted = [json.loads(line) for line in path.read_text().splitlines()]

    assert len(rows) == 3
    assert len(persisted) == 3
    assert rows[0]["error_type"] == "request_timeout"
    assert [row["index"] for row in rows] == [1, 2, 3]


def _patch_transport(handler):
    """把 runner 内部创建的 AsyncClient 换成 MockTransport。"""
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("verify", None)
        kwargs.pop("trust_env", None)
        return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    return patch("gemini_image.runner.httpx.AsyncClient", side_effect=factory)


def test_end_to_end_writes_evidence_and_finalized_summary():
    seen: list[dict] = []
    encoded = _png_b64((3, 3))

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        seen.append(body)
        count = body["generationConfig"]["candidateCount"]
        parts = [
            {"inlineData": {"mimeType": "image/png", "data": encoded}}
            for _ in range(count)
        ]
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 1120,
                    "thoughtsTokenCount": 100,
                    "totalTokenCount": 1230,
                },
                "modelVersion": "gemini-3-pro-image",
            },
        )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        env = {
            "GEMINI_IMAGE_API_KEY": "secret",
            "GEMINI_IMAGE_BASE_URL": "https://mux.la/v1beta",
            "GEMINI_IMAGE_QUALITIES": "1K,2K",
            "GEMINI_IMAGE_MODELS": "gemini-3-pro-image",
            "GEMINI_IMAGE_ASPECTS": "1:1,16:9",
            "GEMINI_IMAGE_COUNTS": "1,2",
        }
        with patch.dict(os.environ, env, clear=False), _patch_transport(handler):
            code = asyncio.run(
                main_async(
                    [
                        "GEMINI_IMAGE_GEN",
                        "--output-root", str(root),
                        "--requests", "4",
                        "--concurrency", "2",
                        "--seed", "7",
                    ]
                )
            )
        results = [p for p in root.iterdir() if p.is_dir()]
        assert len(results) == 1
        result = results[0]
        for name in ("metadata.json", "generated_requests.jsonl", "requests.jsonl", "summary.json", "report.md", "validation.json"):
            assert (result / name).is_file(), name
        assert (result / "requests.jsonl").stat().st_size > 0

        summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
        validation = json.loads((result / "validation.json").read_text(encoding="utf-8"))
        metadata = json.loads((result / "metadata.json").read_text(encoding="utf-8"))
        request_rows = [json.loads(line) for line in (result / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
        requests_text = (result / "requests.jsonl").read_text(encoding="utf-8")
        report = (result / "report.md").read_text(encoding="utf-8")
        image_files = sorted((result / "images").glob("*.png"))
        images_nonempty = all(image.stat().st_size > 0 for image in image_files)

    assert code == 0
    assert summary["requested"] == 4
    assert summary["successful"] == 4
    assert summary["admission_passed"] is True
    assert summary["expected_image_total"] == sum(row["count"] for row in request_rows)
    assert summary["actual_image_total"] == sum(row["count"] for row in request_rows)
    assert summary["count_honored_rate"] == 1.0
    assert summary["usage"]["thoughts_tokens"] == 400
    # 图片原文件落盘，请求行只留路径与校验信息。
    assert metadata["keep_images"] is True
    assert summary["images_dir"] == "images"
    assert summary["images_saved"] == summary["actual_image_total"]
    assert summary["images_skipped"] == 0
    assert summary["image_save_errors"] == 0
    assert len(image_files) == summary["images_saved"]
    assert images_nonempty is True
    assert all(row["images"] for row in request_rows)
    assert encoded not in requests_text
    # 图片指纹进报告：mock 每次都返回同一张图，所以唯一内容只有 1 个、其余全是重复。
    assert "## 图片指纹" in report
    assert "### 逐图指纹" in report
    assert "不参与准入判定" in report
    fingerprints = summary["image_fingerprints"]
    assert fingerprints["images"] == summary["images_saved"]
    assert fingerprints["unique_sha256"] == 1
    assert fingerprints["duplicate_images"] == summary["images_saved"] - 1
    assert all(row["images"][0]["dhash"] for row in request_rows)
    assert all(len(image["sha256"]) == 16 for image in fingerprints["detail"])
    # C2PA 内容签名小节：mock 图没有签名清单，但章节与 summary 字段必须在。
    assert "### 内容签名（C2PA）" in report
    assert summary["image_c2pa"]["available"] is True
    assert summary["image_c2pa"]["scanned"] == summary["images_saved"]
    assert summary["image_c2pa"]["signed"] == 0
    assert len(seen) == 4
    assert validation["passed"] is True
    assert all(check["passed"] for check in validation["checks"] if check["name"] == "evidence_present")


def test_no_keep_images_writes_no_image_files():
    def handler(request: httpx.Request) -> httpx.Response:
        count = json.loads(request.read())["generationConfig"]["candidateCount"]
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"inlineData": {"mimeType": "image/png", "data": _png_b64((2, 2))}}
                                for _ in range(count)
                            ]
                        }
                    }
                ]
            },
        )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        env = {
            "GEMINI_IMAGE_API_KEY": "secret",
            "GEMINI_IMAGE_BASE_URL": "https://mux.la/v1beta",
            "GEMINI_IMAGE_MODELS": "gemini-3-pro-image",
        }
        with patch.dict(os.environ, env, clear=False), _patch_transport(handler):
            code = asyncio.run(
                main_async(
                    [
                        "GEMINI_IMAGE_GEN",
                        "--output-root", str(root),
                        "--requests", "2",
                        "--concurrency", "1",
                        "--no-keep-images",
                    ]
                )
            )
        result = next(p for p in root.iterdir() if p.is_dir())
        summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
        images_dir_exists = (result / "images").exists()

    assert code == 0
    assert images_dir_exists is False
    assert summary["images_dir"] is None
    assert summary["images_saved"] == 0
    assert summary["max_image_bytes"] is None
