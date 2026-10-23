import asyncio
import base64
import hashlib
import io
import json
import os
import re
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from PIL import Image

from gpt_image.cases import (
    GPT_IMAGE_2_QUALITIES,
    GPT_IMAGE_25_QUALITIES,
    GPT_IMAGE_2_SIZES,
    CASES,
)
from gpt_image.runner import _failed_request_row, _request_one, _run_requests, image_endpoint
from gpt_image.runner import build_request_plan, summary_checks
from gpt_image.token_verify import (
    CASE_ID as TOKEN_VERIFY_CASE_ID,
    IMAGE_BYTES_PER_TOKEN,
    IMAGE_COUNT,
    PROMPT as TOKEN_VERIFY_PROMPT,
    SIZE as TOKEN_VERIFY_SIZE,
    expected_input_tokens,
    expected_output_tokens,
    run_model,
    verify_usage,
)


def test_official_quality_matrix_is_split_between_image_generations():
    assert GPT_IMAGE_2_QUALITIES == ("auto", "low", "medium", "high")
    assert GPT_IMAGE_25_QUALITIES == (
        "auto",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    )
    assert set(GPT_IMAGE_2_SIZES) == {
        "1K", "2K", "4K",
        "1024x1024",
        "2048x2048",
        "1536x1024",
        "1024x1536",
        "1536x1152",
        "2048x1536",
        "1152x1536",
        "1536x2048",
        "1536x864",
        "2048x1152",
        "2560x1440",
        "3840x2160",
        "864x1536",
        "1152x2048",
        "1440x2560",
        "2160x3840",
    }
    assert CASES["GPT_IMAGE_25"].model_choices == (
        "gpt-image-2.5-flare",
        "gpt-image-2.5-sunburst",
    )
    assert CASES[TOKEN_VERIFY_CASE_ID].model_choices == (
        "gpt-image-2",
        "gpt-image-2.5-flare",
        "gpt-image-2.5-sunburst",
    )


def test_image_request_uses_generation_endpoint_and_does_not_store_base64():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/images/generations"
        assert request.headers["authorization"] == "Bearer test-secret"
        body = request.read().decode()
        assert '"model":"gpt-image-2"' in body
        assert '"quality":"high"' in body
        assert '"size":"1024x1024"' in body
        return httpx.Response(
            200,
            json={
                "created": 1,
                "data": [{"b64_json": "a-very-large-image-payload"}],
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _request_one(
                client,
                request_plan={
                    "case": "GPT_IMAGE_2",
                    "index": 1,
                    "model": "gpt-image-2",
                    "quality": "high",
                    "size": "1024x1024",
                    "prompt": "test prompt",
                    "payload": {
                        "model": "gpt-image-2",
                        "prompt": "test prompt",
                        "size": "1024x1024",
                        "quality": "high",
                        "n": 1,
                    },
                },
                protocol="openai",
                base_url="https://example.test/v1",
                api_key="test-secret",
            )

    row = asyncio.run(run())
    assert row["success"] is True
    assert row["image_count"] == 1
    assert "a-very-large-image-payload" not in str(row)
    assert image_endpoint("https://api.openai.com/v1/") == "https://api.openai.com/v1/images/generations"
    assert image_endpoint("https://mux.la", "openai_images") == "https://mux.la/v1/images/generations"
    assert image_endpoint("https://ark.example", "volcengine") == "https://ark.example/api/v3/images/generations"


def _png_bytes(size: tuple[int, int] = (8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (30, 90, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


def _image_plan(index: int = 1) -> dict:
    return {
        "case": "GPT_IMAGE_2",
        "index": index,
        "model": "gpt-image-2",
        "quality": "high",
        "size": "1024x1024",
        "prompt": "test prompt",
        "payload": {
            "model": "gpt-image-2",
            "prompt": "test prompt",
            "size": "1024x1024",
            "quality": "high",
            "n": 1,
        },
    }


def _run_image_request(handler, image_dir=None, plan=None, **kwargs):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _request_one(
                client,
                request_plan=plan or _image_plan(),
                protocol="openai",
                base_url="https://example.test/v1",
                api_key="secret",
                image_dir=image_dir,
                **kwargs,
            )

    return asyncio.run(run())


def test_b64_image_result_is_saved_and_evidence_keeps_paths_only(tmp_path):
    raw = _png_bytes()
    encoded = base64.b64encode(raw).decode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"created": 1, "data": [{"b64_json": encoded}]})

    row = _run_image_request(handler, image_dir=tmp_path / "images")

    assert row["success"] is True
    assert row["image_save_error"] is None
    entry = row["images"][0]
    assert re.fullmatch(r"images/image2-high-\d{12}\.png", entry["file"])
    assert entry["saved"] is True
    assert entry["source"] == "b64_json"
    assert entry["format"] == "PNG" and (entry["width"], entry["height"]) == (8, 8)
    assert entry["bytes"] == len(raw)
    assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
    assert (tmp_path / row["images"][0]["file"]).read_bytes() == raw
    # base64 内容绝不进入证据文件。
    assert encoded not in json.dumps(row)


def test_url_image_result_is_downloaded_into_images_dir(tmp_path):
    raw = _png_bytes((4, 6))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cdn.test":
            return httpx.Response(200, content=raw, headers={"content-type": "image/png"})
        return httpx.Response(200, json={"created": 1, "data": [{"url": "https://cdn.test/a.png"}]})

    row = _run_image_request(handler, image_dir=tmp_path / "images")

    entry = row["images"][0]
    assert entry["saved"] is True and entry["source"] == "url"
    assert entry["url"] == "https://cdn.test/a.png"
    assert (entry["width"], entry["height"]) == (4, 6)
    assert (tmp_path / row["images"][0]["file"]).read_bytes() == raw


def test_image_write_failure_never_changes_the_request_verdict(tmp_path):
    blocker = tmp_path / "images"
    blocker.write_text("not a directory", encoding="utf-8")
    raw = _png_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"created": 1, "data": [{"b64_json": base64.b64encode(raw).decode()}]},
        )

    row = _run_image_request(handler, image_dir=blocker)

    assert row["success"] is True
    assert row["images"] == []
    assert "image_save_error" in row and row["image_save_error"]


def test_merged_image_plan_uses_supported_qualities_for_each_model():
    case = CASES["GPT_IMAGE_2"]
    plan = build_request_plan(
        case=case, requests=100, qualities=list(case.qualities),
        sizes=["1024x1024"], models=list(case.model_choices), seed=7,
    )
    assert len(plan) == 100
    assert {row["model"] for row in plan} == {
        "gpt-image-2", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst",
    }
    for row in plan:
        allowed = GPT_IMAGE_2_QUALITIES if row["model"] == "gpt-image-2" else GPT_IMAGE_25_QUALITIES
        assert row["payload"]["quality"] in allowed
    assert {"xhigh", "max"} <= {row["quality"] for row in plan}


def test_image_plan_rejects_models_without_compatible_quality():
    with pytest.raises(ValueError, match="gpt-image-2 不支持所选 quality"):
        build_request_plan(
            case=CASES["GPT_IMAGE_2"], requests=1, qualities=["max"],
            sizes=["1K"], models=["gpt-image-2", "gpt-image-2.5-flare"], seed=7,
        )


def test_request_plan_is_complete_and_reproducible_before_execution():
    case = CASES["GPT_IMAGE_25"]
    first = build_request_plan(
        case=case,
        requests=20,
        qualities=["low", "high"],
        sizes=["1024x1024", "1536x1024"],
        models=list(case.model_choices),
        seed=20260915,
    )
    second = build_request_plan(
        case=case,
        requests=20,
        qualities=["low", "high"],
        sizes=["1024x1024", "1536x1024"],
        models=list(case.model_choices),
        seed=20260915,
    )
    assert first == second
    assert len(first) == 20
    assert [item["index"] for item in first] == list(range(1, 21))
    assert all(item["payload"]["n"] == 1 for item in first)
    assert all("aspect_ratio" not in item["payload"] for item in first)


@pytest.mark.parametrize("case_id", ["GPT_IMAGE_2", "GPT_IMAGE_25"])
@pytest.mark.parametrize("size", ["1K", "2K", "4K"])
@pytest.mark.parametrize("ratio", [None, "1:1", "3:2", "2:3", "4:3", "3:4", "5:4", "4:5", "16:9", "9:16", "21:9"])
def test_image_load_options_reach_http_payload_unchanged(case_id, size, ratio):
    case = CASES[case_id]
    plan = build_request_plan(
        case=case, requests=1, qualities=["high"], sizes=[size],
        models=list(case.model_choices), seed=7, aspect_ratio=ratio,
    )[0]
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"b64_json": "stub"}]})

    _run_image_request(handler, plan=plan)
    assert seen[0]["size"] == size
    if ratio is None:
        assert "aspect_ratio" not in seen[0]
    else:
        assert seen[0]["aspect_ratio"] == ratio


def test_transport_timeout_returns_complete_failure_row():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("forced timeout", request=request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _request_one(
                client,
                request_plan={
                    "case": "GPT_IMAGE_2",
                    "index": 1,
                    "model": "gpt-image-2",
                    "quality": "low",
                    "size": "1024x1024",
                    "prompt": "test",
                    "payload": {"model": "gpt-image-2", "prompt": "test"},
                },
                protocol="openai",
                base_url="https://example.test/v1",
                api_key="secret",
            )

    row = asyncio.run(run())
    assert row["success"] is False
    assert row["error_type"] == "timeout"
    assert row["response_model"] is None


def test_absolute_timeout_does_not_block_following_requests_and_flushes_rows():
    plans = [
        {
            "case": "GPT_IMAGE_2",
            "index": index,
            "model": "gpt-image-2",
            "quality": "low",
            "size": "1024x1024",
            "prompt": "test",
            "payload": {"model": "gpt-image-2", "prompt": "test"},
        }
        for index in range(1, 4)
    ]

    async def fake_request(
        client, *, request_plan, protocol, base_url, api_key,
        image_dir=None, keep_images=True, budget=None,
    ):
        if request_plan["index"] == 1:
            await asyncio.sleep(0.05)
        row = _failed_request_row(
            request_plan,
            protocol=protocol,
            base_url=base_url,
            started_epoch=time.time(),
            elapsed_seconds=0.001,
            error_type="http_503",
            error_message="test",
        )
        return row

    async def run(path: Path):
        with patch("gpt_image.runner._request_one", side_effect=fake_request):
            return await _run_requests(
                request_plan=plans,
                concurrency=1,
                protocol="openai",
                base_url="https://example.test/v1",
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


def test_4k_token_verifier_matches_prompt_and_each_decoded_image():
    image_sizes = [10 * 1024 * 1024 + index for index in range(IMAGE_COUNT)]
    converted_output = expected_output_tokens(image_sizes)
    converted_input = expected_input_tokens(TOKEN_VERIFY_PROMPT)
    images = [{"bytes": size} for size in image_sizes]
    usage = {
        "generated_images": IMAGE_COUNT,
        "input_tokens": converted_input,
        "input_tokens_details": {"text_tokens": converted_input},
        "output_tokens": converted_output,
        "output_tokens_details": {"image_tokens": converted_output},
        "total_tokens": converted_input + converted_output,
    }

    checks = verify_usage(usage, TOKEN_VERIFY_PROMPT, images)

    assert all(check["passed"] for check in checks.values())
    assert expected_output_tokens([IMAGE_BYTES_PER_TOKEN + 1]) == 2


def test_4k_token_verifier_sends_one_n10_request_and_saves_images(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), "white").save(buffer, format="PNG")
    raw = buffer.getvalue()
    encoded = base64.b64encode(raw).decode()
    output_tokens = expected_output_tokens([len(raw)] * IMAGE_COUNT)
    input_tokens = expected_input_tokens(TOKEN_VERIFY_PROMPT)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        assert payload == {
            "model": "gpt-image-2",
            "prompt": TOKEN_VERIFY_PROMPT,
            "size": TOKEN_VERIFY_SIZE,
            "quality": "high",
            "output_format": "png",
            "n": IMAGE_COUNT,
        }
        return httpx.Response(
            200,
            json={
                "model": "gpt-image-2",
                "data": [{"b64_json": encoded} for _ in range(IMAGE_COUNT)],
                "usage": {
                    "generated_images": IMAGE_COUNT,
                    "input_tokens": input_tokens,
                    "input_tokens_details": {"text_tokens": input_tokens},
                    "output_tokens": output_tokens,
                    "output_tokens_details": {"image_tokens": output_tokens},
                    "total_tokens": input_tokens + output_tokens,
                },
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await run_model(
                client,
                model="gpt-image-2",
                quality="high",
                protocol="openai",
                base_url="https://example.test/v1",
                api_key="secret",
                image_dir=tmp_path,
            )

    row = asyncio.run(run())

    assert row["image_count"] == IMAGE_COUNT
    assert len(list(tmp_path.glob("*.png"))) == IMAGE_COUNT
    assert row["checks"]["input_tokens_match"]["passed"] is True
    assert row["checks"]["output_tokens_match"]["passed"] is True
    assert row["checks"]["all_dimensions_4k"]["passed"] is False
    assert row["checks"]["all_images_around_10m"]["passed"] is False
    assert row["case_passed"] is False


def test_4k_token_verify_report_includes_image_fingerprints():
    """4K token 验真与压测同口径：报告里也要有逐图指纹。"""
    from gpt_image.token_verify import main_async as token_main_async

    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), (200, 30, 30)).save(buffer, format="PNG")
    raw = buffer.getvalue()
    encoded = base64.b64encode(raw).decode()
    output_tokens = expected_output_tokens([len(raw)] * IMAGE_COUNT)
    input_tokens = expected_input_tokens(TOKEN_VERIFY_PROMPT)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-image-2",
                "data": [{"b64_json": encoded} for _ in range(IMAGE_COUNT)],
                "usage": {
                    "generated_images": IMAGE_COUNT,
                    "input_tokens": input_tokens,
                    "input_tokens_details": {"text_tokens": input_tokens},
                    "output_tokens": output_tokens,
                    "output_tokens_details": {"image_tokens": output_tokens},
                    "total_tokens": input_tokens + output_tokens,
                },
            },
        )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        env = {
            "GPT_IMAGE_API_KEY": "secret",
            "GPT_IMAGE_BASE_URL": "https://example.test/v1",
            "GPT_IMAGE_PROTOCOL": "openai",
        }
        with patch.dict(os.environ, env, clear=False), patch(
            "gpt_image.token_verify.httpx.AsyncClient", side_effect=_mock_client_factory(handler)
        ):
            asyncio.run(token_main_async(["--output-root", str(root)]))
        result = next(p for p in root.iterdir() if p.is_dir())
        summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
        report = (result / "report.md").read_text(encoding="utf-8")
        request_rows = [
            json.loads(line)
            for line in (result / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    fingerprints = summary["image_fingerprints"]
    assert "## 图片指纹" in report
    assert "### 逐图指纹" in report
    assert fingerprints["images"] == 3 * IMAGE_COUNT
    # 三个模型返回同一张图：唯一指纹 1 个，其余全部重复。
    assert fingerprints["unique_sha256"] == 1
    assert fingerprints["duplicate_images"] == 3 * IMAGE_COUNT - 1
    assert all(len(image["sha256"]) == 16 for image in fingerprints["detail"])
    assert all(row["images"][0]["dhash"] for row in request_rows)
    # C2PA 内容签名小节：假图没有签名清单，但章节与 summary 字段必须在。
    assert "### 内容签名（C2PA）" in report
    assert summary["image_c2pa"]["available"] is True
    assert summary["image_c2pa"]["scanned"] == 3 * IMAGE_COUNT
    assert summary["image_c2pa"]["signed"] == 0


def _mock_client_factory(handler):
    """把任意模块内创建的 AsyncClient 换成 MockTransport。"""
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("verify", None)
        kwargs.pop("trust_env", None)
        return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    return factory


def _patch_runner_transport(handler):
    """把 runner 内部创建的 AsyncClient 换成 MockTransport。"""
    return patch(
        "gpt_image.runner.httpx.AsyncClient", side_effect=_mock_client_factory(handler)
    )


def test_end_to_end_keeps_images_and_writes_evidence():
    from gpt_image.runner import main_async as gpt_main_async

    encoded = base64.b64encode(_png_bytes((6, 6))).decode()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        assert json.loads(request.content)["size"] == "4K"
        assert json.loads(request.content)["aspect_ratio"] == "21:9"
        return httpx.Response(
            200,
            json={"created": 1, "model": "gpt-image-2.5-flare", "data": [{"b64_json": encoded}]},
        )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        env = {
            "GPT_IMAGE_API_KEY": "secret",
            "GPT_IMAGE_BASE_URL": "https://example.test/v1",
            "GPT_IMAGE_QUALITIES": "low",
            "GPT_IMAGE_MODELS": "gpt-image-2.5-flare",
            "GPT_IMAGE_SIZES": "4K",
            "GPT_IMAGE_ASPECT_RATIO": "21:9",
            "GPT_IMAGE_REQUEST_TIMEOUT_SECONDS": "12.5",
            # 显式关闭压测持续时长，避免宿主环境残留把本批变成时长压测。
            "GPT_IMAGE_HOLD_SECONDS": "",
        }
        with patch.dict(os.environ, env, clear=False), _patch_runner_transport(handler):
            code = asyncio.run(
                gpt_main_async(
                    [
                        "GPT_IMAGE_25",
                        "--output-root", str(root),
                        "--requests", "3",
                        "--concurrency", "2",
                        "--seed", "11",
                    ]
                )
            )
        result = next(p for p in root.iterdir() if p.is_dir())
        summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
        report = (result / "report.md").read_text(encoding="utf-8")
        requests_text = (result / "requests.jsonl").read_text(encoding="utf-8")
        image_files = sorted((result / "images").glob("*.png"))

    assert code == 0
    assert seen == ["/v1/images/generations"] * 3
    assert summary["successful"] == 3
    assert summary["aspect_ratio"] == "21:9"
    assert summary["request_timeout_seconds"] == 12.5
    # 默认不启用压测持续时长：判定口径仍是「请求数必须跑满」。
    assert summary["hold_seconds"] is None
    assert set(summary["checks"]) == {"exact_request_count", "all_successful"}
    assert summary["images_dir"] == "images"
    assert summary["images_saved"] == 3
    assert summary["images_skipped"] == 0
    assert summary["image_save_errors"] == 0
    assert len(image_files) == 3
    assert encoded not in requests_text
    assert "图片保留：3 张" in report
    # 图片指纹进报告：mock 每次都返回同一张图，唯一内容只有 1 个、其余都是重复。
    assert "## 图片指纹" in report
    assert "### 逐图指纹" in report
    assert "不参与准入判定" in report
    fingerprints = summary["image_fingerprints"]
    assert fingerprints["images"] == 3
    assert fingerprints["unique_sha256"] == 1
    assert fingerprints["duplicate_images"] == 2
    assert fingerprints["duplicate_group_count"] == 1
    assert all(len(image["sha256"]) == 16 for image in fingerprints["detail"])


def _probe_png_b64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 30, 30)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


async def _run_probe(handler, save_image=None):
    from scripts.probe_gpt_image import run_probe

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await run_probe(
            client,
            model="gpt-image-2.5-sunburst",
            protocol="openai",
            base_url="https://example.test/v1",
            api_key="secret",
            save_image=save_image,
        )


def test_probe_reports_callable_for_valid_b64_response():
    from scripts.probe_gpt_image import DEFAULT_PROMPT, probe_exit_code

    seen: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            {
                "url": str(request.url),
                "auth": request.headers.get("authorization"),
                "payload": json.loads(request.content),
            }
        )
        return httpx.Response(
            200,
            json={
                "model": "gpt-image-2.5-sunburst",
                "data": [{"b64_json": _probe_png_b64()}],
                "usage": {"input_tokens": 21, "output_tokens": 6, "total_tokens": 27},
            },
        )

    row = asyncio.run(_run_probe(handler))

    assert seen[0]["url"] == "https://example.test/v1/images/generations"
    assert seen[0]["auth"] == "Bearer secret"
    assert seen[0]["payload"] == {
        "model": "gpt-image-2.5-sunburst",
        "prompt": DEFAULT_PROMPT,
        "size": "1024x1024",
        "quality": "low",
        "n": 1,
    }
    assert row["callable"] is True
    assert row["image_count"] == 1
    assert row["result_types"] == ["b64_json"]
    assert row["image_bytes"] > 0
    assert probe_exit_code(row) == 0


def test_probe_saves_url_result_and_flags_unusable_response():
    from scripts.probe_gpt_image import probe_exit_code

    async def url_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/images/generations"):
            return httpx.Response(
                200,
                json={"model": "gpt-image-2.5-sunburst", "data": [{"url": "https://cdn.test/a.png"}]},
            )
        return httpx.Response(200, content=base64.b64decode(_probe_png_b64()))

    async def empty_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "gpt-image-2.5-sunburst", "data": []})

    async def error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "model not found"}})

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "probe.png"
        url_row = asyncio.run(_run_probe(url_handler, target))
        assert url_row["callable"] is True
        assert url_row["result_types"] == ["url"]
        assert target.exists() and target.stat().st_size > 0
        assert url_row["saved_image"] == str(target)

    empty_row = asyncio.run(_run_probe(empty_handler))
    assert empty_row["callable"] is False
    assert empty_row["error_type"] == "invalid_image_response"
    assert empty_row["error_message"] == "HTTP 成功但响应未包含 data 图片结果"
    assert probe_exit_code(empty_row) == 2

    error_row = asyncio.run(_run_probe(error_handler))
    assert error_row["callable"] is False
    assert error_row["error_type"] == "http_404"
    assert "model not found" in error_row["error_message"]
    assert probe_exit_code(error_row) == 1


# —— 压测持续时长（默认不启用，可勾选） ——
def _evidence_row(index: int, success: bool = True) -> dict:
    return {
        "index": index,
        "success": success,
        "started_epoch": 1_000.0 + index,
        "completed_epoch": 1_010.0 + index,
    }


def test_summary_checks_keep_exact_count_when_hold_is_disabled():
    rows = [_evidence_row(1), _evidence_row(2)]
    checks, admission = summary_checks(rows=rows, successful=list(rows), requested=2, hold_seconds=None)
    assert set(checks) == {"exact_request_count", "all_successful"}
    assert checks["exact_request_count"]["passed"] is True
    assert admission is True

    # 少发一条即不准入（历史口径不变）。
    checks, admission = summary_checks(rows=rows[:1], successful=rows[:1], requested=2, hold_seconds=None)
    assert checks["exact_request_count"]["passed"] is False
    assert admission is False


def test_summary_checks_switch_to_cap_contract_when_hold_is_enabled():
    rows = [_evidence_row(1), _evidence_row(2), _evidence_row(3)]
    checks, admission = summary_checks(rows=rows, successful=list(rows), requested=100, hold_seconds=30.0)
    assert set(checks) == {"request_count_within_cap", "all_successful"}
    assert checks["request_count_within_cap"]["expected"] == {"min": 1, "max": 100}
    assert checks["request_count_within_cap"]["passed"] is True
    assert admission is True

    # 一条都没发出 / 超过上限 / 有失败请求：均不准入。
    assert summary_checks(rows=[], successful=[], requested=100, hold_seconds=30.0)[1] is False
    over = [_evidence_row(index) for index in range(1, 5)]
    assert summary_checks(rows=over, successful=over, requested=3, hold_seconds=30.0)[1] is False
    mixed = [_evidence_row(1), _evidence_row(2), _evidence_row(3, success=False)]
    checks, admission = summary_checks(rows=mixed, successful=mixed[:2], requested=100, hold_seconds=30.0)
    assert checks["all_successful"]["passed"] is False
    assert admission is False


def test_hold_seconds_stops_scheduling_new_requests(tmp_path):
    plans = [
        {
            "case": "GPT_IMAGE_2",
            "index": index,
            "model": "gpt-image-2",
            "quality": "low",
            "size": "1024x1024",
            "prompt": "test",
            "payload": {"model": "gpt-image-2", "prompt": "test"},
        }
        for index in range(1, 21)
    ]

    async def fake_request(
        client, *, request_plan, protocol, base_url, api_key,
        image_dir=None, keep_images=True, budget=None,
    ):
        await asyncio.sleep(0.02)
        return _failed_request_row(
            request_plan,
            protocol=protocol,
            base_url=base_url,
            started_epoch=time.time(),
            elapsed_seconds=0.02,
            error_type="http_503",
            error_message="test",
        )

    async def run(path: Path, hold):
        with patch("gpt_image.runner._request_one", side_effect=fake_request):
            return await _run_requests(
                request_plan=plans,
                concurrency=2,
                protocol="openai",
                base_url="https://example.test/v1",
                api_key="secret",
                request_timeout_seconds=5.0,
                requests_path=path,
                hold_seconds=hold,
            )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "requests.jsonl"
        rows = asyncio.run(run(path, 0.06))
        persisted = [json.loads(line) for line in path.read_text().splitlines()]

    # 时长到点即停止领取新请求：既没有跑满 20 条，也至少发出过一条。
    assert 1 <= len(rows) < len(plans)
    assert [row["index"] for row in rows] == sorted(row["index"] for row in rows)
    # 在途请求跑完才收尾：证据文件与实际发出的请求一一对应，不留半截记录。
    assert len(persisted) == len(rows)


def test_end_to_end_hold_mode_writes_evidence_and_reports_duration():
    from gpt_image.runner import main_async as gpt_main_async

    encoded = base64.b64encode(_png_bytes((6, 6))).decode()

    def handler(request: httpx.Request) -> httpx.Response:
        # MockTransport 的 handler 是同步的，会阻塞事件循环——正好让发送节奏可控。
        time.sleep(0.03)
        return httpx.Response(
            200,
            json={"created": 1, "model": "gpt-image-2", "data": [{"b64_json": encoded}]},
        )

    env = {
        "GPT_IMAGE_API_KEY": "secret",
        "GPT_IMAGE_BASE_URL": "https://example.test/v1",
        "GPT_IMAGE_QUALITIES": "low",
        "GPT_IMAGE_MODELS": "gpt-image-2",
        "GPT_IMAGE_SIZES": "1K",
        "GPT_IMAGE_REQUEST_TIMEOUT_SECONDS": "5",
        "GPT_IMAGE_HOLD_SECONDS": "0.05",
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with patch.dict(os.environ, env, clear=False), _patch_runner_transport(handler):
            code = asyncio.run(
                gpt_main_async(
                    [
                        "GPT_IMAGE_2",
                        "--output-root", str(root),
                        "--requests", "40",
                        "--concurrency", "1",
                        "--seed", "5",
                    ]
                )
            )
        result = next(p for p in root.iterdir() if p.is_dir())
        metadata = json.loads((result / "metadata.json").read_text(encoding="utf-8"))
        summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
        report = (result / "report.md").read_text(encoding="utf-8")
        persisted = [json.loads(line) for line in (result / "requests.jsonl").read_text().splitlines()]

    assert code == 0
    assert metadata["hold_seconds"] == 0.05
    assert summary["hold_seconds"] == 0.05
    # 请求数退化为上限保险丝：时长到点即停，实际发出数由吞吐决定。
    assert 1 <= summary["sent"] < 40
    assert summary["successful"] == summary["sent"]
    assert len(persisted) == summary["sent"]
    assert summary["checks"]["request_count_within_cap"]["passed"] is True
    assert summary["checks"]["request_count_within_cap"]["expected"] == {"min": 1, "max": 40}
    assert summary["admission_passed"] is True
    assert "压测持续时长：0.05 秒" in report
    assert "总请求数上限：40" in report
    assert "压测持续时长：未启用" not in report
