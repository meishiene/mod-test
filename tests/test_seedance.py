"""Seedance 视频生成套件的回归测试。

覆盖三块：
1. **协议面**：端点拼装（``/api/v3/contents/generations/tasks``）、请求体顶层字段、
   图片三种来源（本地文件 / data URI / URL）与证据脱敏。
2. **裁定矩阵**：提交与轮询的各种观测如何收口成 passed / failed / blocked
   （"未取得终态"必须是 blocked，不是 failed）。
3. **端到端**：MockTransport 回放真实响应体，验证证据文件齐备、报告里带视频链接、
   视频原文件落盘且 base64 不进入证据。
"""
import asyncio
import base64
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx

from seedance.cases import (
    SEEDANCE_CASE_ID,
    SEEDANCE_DEFAULT_DURATION,
    SEEDANCE_DEFAULT_IMAGE_ROLE,
    SEEDANCE_DEFAULT_MODE,
    SEEDANCE_DEFAULT_RATIO,
    SEEDANCE_DEFAULT_RESOLUTION,
    SEEDANCE_DEFAULT_VIDEO_ROLE,
    SEEDANCE_DURATION_MAX,
    SEEDANCE_DURATION_MIN,
    SEEDANCE_IMAGE_ROLES,
    SEEDANCE_MAX_CONCURRENCY,
    SEEDANCE_MAX_REQUESTS,
    SEEDANCE_MODE_IDS,
    SEEDANCE_MODES,
    SEEDANCE_RATIOS,
    SEEDANCE_RESOLUTIONS,
    CASES,
    duration_range_for,
    mode_allows_image,
    mode_allows_video,
    resolve_mode,
)
from seedance.core import (
    VideoConfig,
    build_payload,
    build_request_plan,
    extract_task,
    extract_task_id,
    poll_task,
    resolve_image,
    resolve_video,
    run_videos,
    sanitize_payload,
    submit_task,
    task_endpoint,
    task_status_endpoint,
    verdict_of,
    video_extension,
)
from seedance.runner import build_config, main_async

BASE = "https://token-hub.tangrankeji.com"
TASKS_PATH = "/api/v3/contents/generations/tasks"


def _config(**overrides) -> VideoConfig:
    values = dict(
        api_key="test-key",
        base_url=BASE,
        model="doubao-seedance-1-0-pro-250528",
        prompt="a cat walking on a rooftop at sunrise",
        poll_interval_seconds=0.01,
        task_timeout_seconds=0.3,
        request_timeout_seconds=5.0,
    )
    values.update(overrides)
    return VideoConfig(**values)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# 协议面
# ---------------------------------------------------------------------------
def test_case_declares_seedance_family_only():
    case = CASES[SEEDANCE_CASE_ID]
    assert case.protocols == ("seedance",)
    assert SEEDANCE_DEFAULT_RESOLUTION in SEEDANCE_RESOLUTIONS
    assert SEEDANCE_DEFAULT_RATIO in SEEDANCE_RATIOS
    assert SEEDANCE_DURATION_MIN <= SEEDANCE_DEFAULT_DURATION <= SEEDANCE_DURATION_MAX
    assert SEEDANCE_DEFAULT_IMAGE_ROLE in {item[0] for item in SEEDANCE_IMAGE_ROLES}
    assert SEEDANCE_MAX_REQUESTS >= 1 and SEEDANCE_MAX_CONCURRENCY >= 1


def test_endpoints_target_ark_task_paths():
    assert task_endpoint(BASE) == f"{BASE}{TASKS_PATH}"
    # 已经带 /api/v3 的 base 不得重复拼接版本段。
    assert task_endpoint(f"{BASE}/api/v3") == f"{BASE}{TASKS_PATH}"
    assert task_status_endpoint(BASE, "cgt-abc") == f"{BASE}{TASKS_PATH}/cgt-abc"
    # task id 来自响应体，属不可信输入：拼接前必须转义，避免越权路径。
    assert task_status_endpoint(BASE, "a/../../b") == f"{BASE}{TASKS_PATH}/a%2F..%2F..%2Fb"


def test_payload_uses_top_level_fields():
    payload = build_payload(_config(resolution="1080p", ratio="9:16", duration=8,
                                    generate_audio=True, watermark=True), None)
    assert payload["model"] == "doubao-seedance-1-0-pro-250528"
    assert payload["content"] == [{"type": "text", "text": "a cat walking on a rooftop at sunrise"}]
    assert payload["resolution"] == "1080p"
    assert payload["ratio"] == "9:16"
    assert payload["duration"] == 8
    assert payload["generate_audio"] is True
    assert payload["watermark"] is True


def test_payload_appends_image_with_role():
    payload = build_payload(_config(image_role="last_frame"), "https://cdn.test/a.png")
    assert payload["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "https://cdn.test/a.png"},
        "role": "last_frame",
    }


def test_image_sources_and_evidence_redaction(tmp_path):
    # 1) 本地文件 → data URI，证据里只有 sha256/字节数
    raw = b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes"
    image = tmp_path / "first.png"
    image.write_bytes(raw)
    value, evidence = resolve_image(None, str(image))
    assert value.startswith("data:image/png;base64,")
    assert base64.b64decode(value.split(",", 1)[1]) == raw
    assert evidence["kind"] == "file" and evidence["bytes"] == len(raw)
    assert evidence["mime"] == "image/png"
    assert len(evidence["sha256"]) == 64

    # 2) data URI 原样透传
    inline = "data:image/jpeg;base64," + base64.b64encode(b"jpeg-bytes").decode()
    value2, evidence2 = resolve_image(inline, None)
    assert value2 == inline and evidence2["kind"] == "data_uri"

    # 3) http(s) URL 原样透传
    value3, evidence3 = resolve_image("https://cdn.test/b.png?x=1", None)
    assert value3 == "https://cdn.test/b.png?x=1" and evidence3["kind"] == "url"

    # 4) 都没有 → 纯文生视频
    value4, evidence4 = resolve_image(None, None)
    assert value4 is None and evidence4 == {"provided": False}

    # 5) 证据脱敏：base64 只留摘要
    sanitized = sanitize_payload(build_payload(_config(), value))
    text = json.dumps(sanitized, ensure_ascii=False)
    assert "fake-image-bytes" not in text
    assert "base64" in text and "sha256=" in text


def test_request_plan_is_prebuilt_and_sanitized():
    payload = build_payload(_config(image_role="first_frame"), "data:image/png;base64,YWJj")
    plan = build_request_plan(_config(requests=3), payload)
    assert [item["index"] for item in plan] == [1, 2, 3]
    assert all(item["image_role"] == "first_frame" for item in plan)
    assert "YWJj" not in json.dumps(plan, ensure_ascii=False)
    assert len({item["request_id"] for item in plan}) == 3


def test_extractors_tolerate_wrapped_envelope():
    assert extract_task_id({"id": "cgt-1"}) == "cgt-1"
    assert extract_task_id({"code": 0, "data": {"task_id": "cgt-2"}}) == "cgt-2"
    assert extract_task_id({"code": 0, "data": {}}) is None
    task = extract_task({
        "code": 0,
        "data": {
            "task_id": "cgt-3",
            "status": "SUCCEEDED",
            "content": {"video_url": "https://cdn.test/v.mp4"},
            "usage": {"completion_tokens": 100, "total_tokens": 100},
        },
    })
    assert task["status"] == "succeeded"
    assert task["video_url"] == "https://cdn.test/v.mp4"
    assert task["usage"]["total_tokens"] == 100


def test_video_extension_from_content_type_or_url():
    assert video_extension("https://x/v.webm?t=1", None) == "webm"
    assert video_extension("https://x/v", "video/mp4") == "mp4"
    assert video_extension("https://x/v", "video/quicktime") == "mov"
    assert video_extension("https://x/v", None) == "mp4"


# ---------------------------------------------------------------------------
# 提交 / 轮询
# ---------------------------------------------------------------------------
def test_submit_returns_task_id():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == TASKS_PATH
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.headers["accept"] == "application/json"
        body = json.loads(request.read())
        assert body["resolution"] == "720p" and body["duration"] == 5
        return httpx.Response(200, json={"id": "cgt-ok"})

    async def run():
        async with _client(handler) as client:
            return await submit_task(client, _config(), build_payload(_config(), None), request_id="r1")

    result = asyncio.run(run())
    assert result["status"] == "ok" and result["task_id"] == "cgt-ok"
    assert result["http_status"] == 200 and result["attempts"] == 1


def test_submit_2xx_without_task_id_is_invalid_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 0, "message": "accepted"})

    async def run():
        async with _client(handler) as client:
            return await submit_task(client, _config(), build_payload(_config(), None), request_id="r")

    result = asyncio.run(run())
    assert result["status"] == "invalid_response"


def test_poll_until_succeeded():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.startswith(TASKS_PATH)
        seen.append(request.url.path)
        if len(seen) < 3:
            return httpx.Response(200, json={"id": "cgt-1", "status": "running"})
        return httpx.Response(200, json={
            "id": "cgt-1", "status": "succeeded",
            "content": {"video_url": "https://cdn.test/v.mp4"},
            "usage": {"completion_tokens": 42, "total_tokens": 42},
        })

    async def run():
        async with _client(handler) as client:
            return await poll_task(client, _config(), "cgt-1", request_id_prefix="r")

    result = asyncio.run(run())
    assert result["task_status"] == "succeeded"
    assert result["task"]["video_url"] == "https://cdn.test/v.mp4"
    assert result["attempts"] == 3
    assert [event["status"] for event in result["events"]] == ["running", "running", "succeeded"]
    assert all(path == f"{TASKS_PATH}/cgt-1" for path in seen)


def test_verdict_matrix():
    base = {
        "submit": {"status": "ok", "http_status": 200},
        "task_status": "succeeded",
        "video_url": "https://cdn.test/v.mp4",
    }
    assert verdict_of(dict(base))[0] == "passed"

    # 200 无产出 = 真缺陷，判 failed（不能因为 HTTP 成功就算通过）
    no_video = {**base, "video_url": None}
    assert verdict_of(no_video)[0] == "failed"

    # 任务失败 = failed
    assert verdict_of({**base, "task_status": "failed", "task_error": "content policy"})[0] == "failed"

    # 任务过期 / 平台清理 = blocked
    assert verdict_of({**base, "task_status": "expired", "video_url": None})[0] == "blocked"

    # 观察窗口内未达终态 = blocked（"没等到结果"不是模型结论）
    assert verdict_of({**base, "task_status": "running", "video_url": None, "timed_out": True})[0] == "blocked"
    assert verdict_of({**base, "task_status": None, "video_url": None, "timed_out": True})[0] == "blocked"

    # 提交侧：账号/额度/限流/5xx/传输错误 = blocked
    for status in (401, 402, 403, 404, 429, 503, 500):
        row = {"submit": {"status": "http_error", "http_status": status, "error_message": "x"},
               "task_status": None}
        assert verdict_of(row)[0] == "blocked", status
    row = {"submit": {"status": "transport_error", "http_status": None, "error_message": "timeout"},
           "task_status": None}
    assert verdict_of(row)[0] == "blocked"
    # 参数类 400 = 上游明确拒绝，属 failed
    row = {"submit": {"status": "http_error", "http_status": 400, "error_message": "bad ratio"},
           "task_status": None}
    assert verdict_of(row)[0] == "failed"
    # 查询端点 4xx 无法取证 = blocked
    row = {**base, "video_url": None, "task_status": None, "poll_error": "HTTP 404"}
    assert verdict_of(row)[0] == "blocked"


# ---------------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------------
def _patch_transport(handler):
    """把 runner 内部创建的 AsyncClient 换成 MockTransport。"""
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("verify", None)
        kwargs.pop("trust_env", None)
        return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    return patch("seedance.core.httpx.AsyncClient", side_effect=factory)


def test_end_to_end_writes_evidence_report_and_videos():
    poll_count: dict[str, int] = {}
    submitted: list[dict] = []
    payload_bytes = b"\x00\x00\x00\x18ftypmp42fake-mp4-bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cdn.test":
            return httpx.Response(200, content=payload_bytes,
                                  headers={"content-type": "video/mp4"})
        if request.method == "POST":
            submitted.append(json.loads(request.read()))
            return httpx.Response(200, json={"id": f"cgt-{len(submitted)}"})
        task_id = request.url.path.rsplit("/", 1)[-1]
        poll_count[task_id] = poll_count.get(task_id, 0) + 1
        if poll_count[task_id] < 2:
            return httpx.Response(200, json={"id": task_id, "status": "running"})
        return httpx.Response(200, json={
            "id": task_id,
            "status": "succeeded",
            "content": {"video_url": f"https://cdn.test/{task_id}.mp4"},
            "usage": {"completion_tokens": 24, "total_tokens": 24},
        })

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        env = {
            "SEEDANCE_API_KEY": "secret",
            "SEEDANCE_BASE_URL": BASE,
            "SEEDANCE_MODEL": "doubao-seedance-1-0-pro-250528",
            "SEEDANCE_PROMPT": "a cat walking on a rooftop at sunrise",
            "SEEDANCE_RESOLUTION": "1080p",
            "SEEDANCE_RATIO": "16:9",
            "SEEDANCE_DURATION": "6",
            "SEEDANCE_GENERATE_AUDIO": "1",
        }
        with patch.dict(os.environ, env, clear=False), _patch_transport(handler):
            code = asyncio.run(main_async([
                SEEDANCE_CASE_ID,
                "--output-root", str(root),
                "--requests", "2",
                "--concurrency", "2",
                "--poll-interval", "0.01",
                "--task-timeout", "5",
            ]))
        results = [path for path in root.iterdir() if path.is_dir()]
        assert len(results) == 1
        result = results[0]
        for name in ("metadata.json", "generated_requests.jsonl", "requests.jsonl",
                     "summary.json", "report.md", "validation.json"):
            assert (result / name).is_file(), name
        requests_text = (result / "requests.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(line) for line in requests_text.splitlines()]
        summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
        metadata = json.loads((result / "metadata.json").read_text(encoding="utf-8"))
        report = (result / "report.md").read_text(encoding="utf-8")
        videos = sorted((result / "videos").glob("*.mp4"))
        video_payloads = [path.read_bytes() for path in videos]

    assert code == 0
    assert summary["admission_passed"] is True
    assert summary["requested"] == 2 and summary["successful"] == 2
    assert summary["resolution"] == "1080p" and summary["duration"] == 6
    assert summary["generate_audio"] is True
    assert summary["videos_saved"] == 2
    assert summary["total_polls"] == 4
    assert metadata["keep_videos"] is True
    assert len(rows) == 2
    assert all(row["success"] is True and row["verdict"] == "passed" for row in rows)
    assert all(row["video_url"].startswith("https://cdn.test/") for row in rows)
    assert all(row["task_status"] == "succeeded" for row in rows)
    # 视频原文件落盘，字节一致
    assert video_payloads == [payload_bytes, payload_bytes]
    assert all(row["video"]["saved"] is True for row in rows)
    # 报告必须给出可复制的视频链接
    assert "## 视频链接" in report
    assert "https://cdn.test/cgt-1.mp4" in report
    assert "https://cdn.test/cgt-2.mp4" in report
    assert all(body["generate_audio"] is True for body in submitted)
    assert all(body["resolution"] == "1080p" and body["duration"] == 6 for body in submitted)


def test_non_playable_video_format_is_archived_without_a_dead_artifact_link(tmp_path):
    """上游给 .avi 之类的格式时：文件照旧归档，但**不给**在线预览链接。

    Web 端的产物白名单（`SEEDANCE_VIDEO_PATTERN`）只放行浏览器能直接播放的后缀；
    如果这里无条件写 `artifact_url`，报告里就会出现一个点了 404 的地址。
    """
    payload_bytes = b"RIFF....AVI fake"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cdn.test":
            return httpx.Response(200, content=payload_bytes,
                                  headers={"content-type": "video/x-msvideo"})
        if request.method == "POST":
            return httpx.Response(200, json={"id": "cgt-avi"})
        return httpx.Response(200, json={
            "id": "cgt-avi",
            "status": "succeeded",
            "content": {"video_url": "https://cdn.test/clip.avi"},
        })

    # record_id 必须给：artifact_url 只在「有 Web 记录」时才谈得上（CLI 直跑没有记录页）。
    config = _config(requests=1, concurrency=1, request_timeout_seconds=5.0,
                     record_id="run-avi-0001")
    with _patch_transport(handler):
        outcome = asyncio.run(run_videos(config, tmp_path, result_dir=tmp_path / "run"))

    row = outcome.rows[0]
    # 任务本身是成功的：格式不参与判定，只影响"能不能在线看"。
    assert row["verdict"] == "passed" and row["success"] is True
    assert row["video"]["file"] == "videos/video-0001.avi"
    assert (tmp_path / "run" / "videos" / "video-0001.avi").read_bytes() == payload_bytes
    assert "artifact_url" not in row
    assert "非浏览器可播放" in row["artifact_note"]
    link = outcome.summary["video_links"][0]
    assert link["local_file"] == "videos/video-0001.avi"
    assert link["local_viewable"] is False
    assert link["url"] == "https://cdn.test/clip.avi"
    report = (tmp_path / "run" / "report.md").read_text(encoding="utf-8")
    assert "`videos/video-0001.avi`（非浏览器可播放格式）" in report
    assert "/artifact/videos/video-0001.avi" not in report


def test_playable_video_format_exposes_a_working_artifact_link(tmp_path):
    """与上一条互为反向：能播的格式必须给出在线预览链接（白名单与落盘后缀同源）。"""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cdn.test":
            return httpx.Response(200, content=b"fake-mp4", headers={"content-type": "video/mp4"})
        if request.method == "POST":
            return httpx.Response(200, json={"id": "cgt-mp4"})
        return httpx.Response(200, json={
            "id": "cgt-mp4", "status": "succeeded",
            "content": {"video_url": "https://cdn.test/clip.mp4"},
        })

    config = _config(requests=1, concurrency=1, request_timeout_seconds=5.0,
                     record_id="run-mp4-0001")
    with _patch_transport(handler):
        outcome = asyncio.run(run_videos(config, tmp_path, result_dir=tmp_path / "run"))

    row = outcome.rows[0]
    assert row["artifact_url"] == "/api/runs/run-mp4-0001/artifact/videos/video-0001.mp4"
    assert "artifact_note" not in row
    link = outcome.summary["video_links"][0]
    assert link["local_file"] == "videos/video-0001.mp4" and link["local_viewable"] is True
    report = (tmp_path / "run" / "report.md").read_text(encoding="utf-8")
    assert "[本地文件](/api/runs/run-mp4-0001/artifact/videos/video-0001.mp4)" in report


def test_blocked_is_recorded_not_flattened_to_failed(tmp_path):
    """账号/额度类观测必须是 blocked，且不把整批判成硬失败（但整批全 blocked 不准入）。"""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(429, json={"error": {"code": "rate_limited", "message": "too many"}})
        raise AssertionError("被限流时不应发起轮询")

    config = _config(requests=1, concurrency=1, keep_videos=False)
    with _patch_transport(handler):
        outcome = asyncio.run(run_videos(config, tmp_path))

    row = outcome.rows[0]
    assert row["success"] is False
    assert row["verdict"] == "blocked"
    assert outcome.summary["blocked"] == 1
    assert outcome.summary["failed"] == 0
    # 整批只有 blocked：没有有效结论 ⇒ 不准入（但也不是 failed 结论）
    assert outcome.summary["admission_passed"] is False
    assert outcome.summary["checks"]["no_hard_failure"]["passed"] is True
    assert outcome.summary["checks"]["effective_verdict_present"]["passed"] is False


def test_task_timeout_is_blocked(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "cgt-slow"})
        return httpx.Response(200, json={"id": "cgt-slow", "status": "running"})

    config = _config(requests=1, concurrency=1, keep_videos=False,
                     poll_interval_seconds=0.01, task_timeout_seconds=0.05)
    with _patch_transport(handler):
        outcome = asyncio.run(run_videos(config, tmp_path))

    row = outcome.rows[0]
    assert row["task_status"] == "running"
    assert row["timed_out"] is True
    assert row["verdict"] == "blocked"
    assert row["error_type"] == "blocked"
    assert outcome.summary["failed"] == 0


def test_partial_blocked_still_admits_when_a_video_was_produced(tmp_path):
    """口径钉死：blocked 只记录不阻断；有 passed 且无 failed 时准入为真。"""
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            counter["n"] += 1
            if counter["n"] == 1:
                return httpx.Response(200, json={"id": "cgt-a"})
            return httpx.Response(403, json={"error": {"code": "quota", "message": "no quota"}})
        return httpx.Response(200, json={
            "id": "cgt-a", "status": "succeeded",
            "content": {"video_url": "https://cdn.test/a.mp4"},
        })

    config = _config(requests=2, concurrency=1, keep_videos=False)
    with _patch_transport(handler):
        outcome = asyncio.run(run_videos(config, tmp_path))

    assert outcome.summary["successful"] == 1
    assert outcome.summary["blocked"] == 1
    assert outcome.summary["failed"] == 0
    assert outcome.summary["admission_passed"] is True
    statuses = [row["verdict"] for row in outcome.rows]
    assert "passed" in statuses and "blocked" in statuses


def test_failed_task_produces_hard_failure(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "cgt-bad"})
        return httpx.Response(200, json={
            "id": "cgt-bad", "status": "failed",
            "error": {"code": "ContentPolicyViolation", "message": "prompt rejected"},
        })

    config = _config(requests=1, concurrency=1, keep_videos=False)
    with _patch_transport(handler):
        outcome = asyncio.run(run_videos(config, tmp_path))

    row = outcome.rows[0]
    assert row["verdict"] == "failed"
    assert "prompt rejected" in row["verdict_detail"]
    assert outcome.summary["failed"] == 1
    assert outcome.summary["admission_passed"] is False
    # 报告里要能看到失败原因，不能只写 "HTTP 200"
    report = (outcome.result_dir / "report.md").read_text(encoding="utf-8")
    assert "prompt rejected" in report

    # 失败取证：笼统文案（中转的「请求参数校验失败」就是这种）必须配上机器可读的 code、
    # 上游回显的顶层键名和末次响应体原文，否则报告只说"失败了"、答不出"哪个参数被拒"。
    assert "ContentPolicyViolation" in row["verdict_detail"]
    assert row["task_error_code"] == "ContentPolicyViolation"
    assert row["task_body_keys"] == ["error", "id", "status"]
    assert "prompt rejected" in row["task_body_sample"]
    assert "ContentPolicyViolation" in report
    assert "未通过用例的上游原文" in report
    assert "['error', 'id', 'status']" in report


def test_extract_task_keeps_error_code_and_top_level_keys():
    """message 与 code 都要留：中转常把真正的原因放在 error.code。"""
    task = extract_task({
        "id": "cgt-x", "status": "failed",
        "error": {"code": "InvalidParameter", "type": "invalid_request_error",
                  "message": "请求参数校验失败"},
        "model": "Seedance-2.0",
    })
    assert task["error"] == "请求参数校验失败"
    assert task["error_code"] == "InvalidParameter"
    # 顶层键清单是"上游到底回显了哪些参数"的唯一线索。
    assert task["body_keys"] == ["error", "id", "model", "status"]

    # 没有 message 时 code 顶上，不让错误文案变空。
    assert extract_task({"status": "failed", "error": {"code": "E"}})["error"] == "E"
    # 完全没有 error 时不能凭空造出 code。
    assert extract_task({"status": "running"})["error_code"] is None


def test_poll_task_keeps_the_last_response_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "cgt-p", "status": "failed",
            "error": {"code": "InvalidParameter", "message": "请求参数校验失败"},
            "resolution": "720p", "duration": 5,
        })

    config = _config(poll_interval_seconds=0.01, task_timeout_seconds=0.2)

    async def run():
        async with _client(handler) as client:
            return await poll_task(client, config, "cgt-p", request_id_prefix="r")

    poll = asyncio.run(run())
    assert poll["task_status"] == "failed"
    assert poll["task_body"]["resolution"] == "720p"
    assert poll["task"]["error_code"] == "InvalidParameter"
    assert poll["task"]["echo"] == {"resolution": "720p", "duration": 5}


def test_successful_run_has_no_diagnostics_section(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "cgt-ok"})
        return httpx.Response(200, json={
            "id": "cgt-ok", "status": "succeeded",
            "content": {"video_url": "https://cdn.test/a.mp4"},
        })

    config = _config(requests=1, concurrency=1, keep_videos=False)
    with _patch_transport(handler):
        outcome = asyncio.run(run_videos(config, tmp_path))
    report = (outcome.result_dir / "report.md").read_text(encoding="utf-8")
    assert outcome.rows[0]["verdict"] == "passed"
    assert "未通过用例的上游原文" not in report


def test_request_rows_expose_poll_timeline(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "cgt-t"})
        return httpx.Response(200, json={
            "id": "cgt-t", "status": "succeeded",
            "content": {"video_url": "https://cdn.test/t.mp4"},
        })

    config = _config(requests=1, concurrency=1, keep_videos=False)
    with _patch_transport(handler):
        outcome = asyncio.run(run_videos(config, tmp_path))

    persisted = json.loads((outcome.result_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert persisted["poll_count"] >= 1
    assert persisted["polls"][0]["status"] == "succeeded"
    assert isinstance(persisted["success"], bool)
    assert isinstance(persisted["verdict"], str)
    assert persisted["submit"]["task_id"] == "cgt-t"


# ---------------------------------------------------------------------------
# CLI / 配置校验
# ---------------------------------------------------------------------------
def _cfg(argv, env=None):
    base_env = {
        "SEEDANCE_API_KEY": "k",
        "SEEDANCE_BASE_URL": BASE,
        "SEEDANCE_MODEL": "doubao-seedance-1-0-pro-250528",
        "SEEDANCE_PROMPT": "a cat",
    }
    base_env.update(env or {})
    with patch.dict(os.environ, base_env, clear=False):
        return build_config(_parse(argv))


def _parse(argv):
    from seedance.runner import _parse_args
    return _parse_args(argv)


def test_cli_config_defaults_and_overrides():
    config = _cfg([])
    assert config.resolution == SEEDANCE_DEFAULT_RESOLUTION
    assert config.ratio == SEEDANCE_DEFAULT_RATIO
    assert config.duration == SEEDANCE_DEFAULT_DURATION
    assert config.requests == 1 and config.concurrency == 1
    # 官方默认：generate_audio 开（原生音频默认生成）、watermark 关。
    assert config.generate_audio is True and config.watermark is False

    config = _cfg(["--resolution", "1080p", "--ratio", "adaptive", "--duration", "12",
                   "--requests", "3", "--concurrency", "2", "--generate-audio", "--watermark"],
                  {"SEEDANCE_IMAGE_ROLE": "reference_image"})
    assert (config.resolution, config.ratio, config.duration) == ("1080p", "adaptive", 12)
    assert (config.requests, config.concurrency) == (3, 2)
    assert config.generate_audio is True and config.watermark is True
    assert config.image_role == "reference_image"

    # 显式关掉原生音频（BooleanOptionalAction 的反向开关）必须生效。
    assert _cfg(["--no-generate-audio"]).generate_audio is False


def test_cli_config_rejects_bad_values():
    for argv in (
        ["--resolution", "2k"],
        # 480p 是 1.0 系列的档位，2.0 / 2.5 的可选集里没有它 —— 首跑就是拿它打 2.0
        # 才烧掉一次完整任务（见 cases.SEEDANCE_RESOLUTIONS 注释）。
        ["--resolution", "480p"],
        ["--ratio", "5:4"],
        ["--duration", "3"],
        ["--duration", "31"],
        ["--requests", "0"],
        ["--requests", "1000"],
        ["--concurrency", "99"],
        ["--image-url", "ftp://x/a.png"],
        ["--image-path", "/no/such/file.png"],
    ):
        try:
            _cfg(argv)
        except ValueError:
            continue
        raise AssertionError(f"{argv} 应被拒绝")


def test_duration_range_follows_the_model_tier():
    """时长区间按**被测模型档位**解析：2.5 → 4–30，2.0 → 4–15，识别不出取并集。"""
    assert duration_range_for("Seedance-2.5") == (4, 30)
    assert duration_range_for("doubao-seedance-2-5-260128") == (4, 30)
    assert duration_range_for("Seedance-2.0") == (4, 15)
    # 档位识别不出时不做本地收紧（非法值交上游拒绝，那本身是可观测结论）。
    assert duration_range_for("doubao-seedance-1-0-pro-250528") == (4, 30)

    # 2.0 档：30 秒出界，本地就拒，免得白烧一次完整生成。
    try:
        _cfg(["--duration", "30"], {"SEEDANCE_MODEL": "Seedance-2.0"})
    except ValueError as exc:
        assert "15" in str(exc)
    else:
        raise AssertionError("2.0 档的 30 秒应被本地拒绝")
    # 2.5 档：30 秒是合法上界。
    assert _cfg(["--duration", "30"], {"SEEDANCE_MODEL": "Seedance-2.5"}).duration == 30


def test_cli_requires_prompt():
    with patch.dict(os.environ, {
        "SEEDANCE_API_KEY": "k",
        "SEEDANCE_BASE_URL": BASE,
        "SEEDANCE_MODEL": "m",
    }, clear=False):
        os.environ.pop("SEEDANCE_PROMPT", None)
        try:
            build_config(_parse([]))
        except ValueError as exc:
            assert "提示词" in str(exc)
        else:
            raise AssertionError("缺少提示词应被拒绝")


# ---------------------------------------------------------------------------
# 参考视频（唯一一个"只有 URL 形态能生效"的输入面）
# ---------------------------------------------------------------------------
def test_payload_appends_video_with_role():
    payload = build_payload(_config(video_role="reference_video"), None, "https://cdn.test/a.mp4")
    assert payload["content"][1] == {
        "type": "video_url",
        "video_url": {"url": "https://cdn.test/a.mp4"},
        "role": "reference_video",
    }


def test_empty_video_role_omits_the_role_field():
    """界面选「不指定角色」时整条 role 字段都不出现。

    这样才能把"上游不认这个 role"和"上游不认视频块"两种拒绝区分开。
    """
    payload = build_payload(_config(video_role=""), None, "https://cdn.test/a.mp4")
    assert payload["content"][1] == {"type": "video_url", "video_url": {"url": "https://cdn.test/a.mp4"}}
    assert "role" not in payload["content"][1]


def test_content_orders_text_then_image_then_video():
    payload = build_payload(
        _config(image_role="first_frame", video_role="reference_video"),
        "https://cdn.test/a.png", "https://cdn.test/a.mp4",
    )
    assert [item["type"] for item in payload["content"]] == ["text", "image_url", "video_url"]


# ---------------------------------------------------------------------------
# 生成方式（text / image / video / both）：请求体形状的开关
# ---------------------------------------------------------------------------
def test_case_exposes_the_generation_modes():
    assert SEEDANCE_MODE_IDS == ("text", "image", "video", "both")
    assert SEEDANCE_DEFAULT_MODE in SEEDANCE_MODE_IDS
    assert dict(SEEDANCE_MODES)["text"].startswith("仅文生视频")


def test_generation_mode_decides_which_blocks_enter_the_payload():
    """声明不带某类输入时，即便调用方多传了值也不会出现在请求体里。

    这是"仅文生视频不提交图片与视频"的最后一道闸——首跑就是拿 480p 打 2.0、
    上游跑到执行阶段才以「请求参数校验失败」拒绝（见 cases.SEEDANCE_RESOLUTIONS 注释），
    参数错配的代价是一次完整生成，因此不能只靠上层记得清空输入。
    """
    image, video = "https://cdn.test/a.png", "https://cdn.test/a.mp4"
    assert [item["type"] for item in build_payload(_config(mode="text"), image, video)["content"]] == ["text"]
    assert [item["type"] for item in build_payload(_config(mode="image"), image, video)["content"]] == [
        "text", "image_url",
    ]
    assert [item["type"] for item in build_payload(_config(mode="video"), image, video)["content"]] == [
        "text", "video_url",
    ]
    both = build_payload(_config(mode="both", image_role="reference_image"), image, video)
    assert [item["type"] for item in both["content"]] == ["text", "image_url", "video_url"]
    assert both["content"][1]["role"] == "reference_image"


def test_unspecified_mode_is_inferred_from_the_inputs():
    """空值 = 按素材推导：既有调用方式（直接构造 VideoConfig）行为不变。"""
    assert resolve_mode(None, has_image=False, has_video=False) == "text"
    assert resolve_mode("", has_image=True, has_video=False) == "image"
    assert resolve_mode("", has_image=False, has_video=True) == "video"
    assert resolve_mode("", has_image=True, has_video=True) == "both"
    # 显式声明优先于输入：不符由校验层报错，不在推导里悄悄纠偏。
    assert resolve_mode("text", has_image=True, has_video=True) == "text"
    assert resolve_mode("IMAGE", has_image=False, has_video=False) == "image"
    assert mode_allows_image("both") and mode_allows_video("both")
    assert not mode_allows_image("video") and not mode_allows_video("image")


def test_local_video_is_archived_but_not_sent(tmp_path):
    raw = b"\x00\x00\x00\x18ftypmp42local-clip"
    clip = tmp_path / "ref.mp4"
    clip.write_bytes(raw)
    value, evidence = resolve_video(None, str(clip))
    # 关键判据：本地视频**不产生**发给上游的值（几十 MB base64 会撑爆请求体）
    assert value is None
    assert evidence["provided"] is True
    assert evidence["sent_to_upstream"] is False
    assert evidence["kind"] == "file"
    assert evidence["local_asset"]["mime"] == "video/mp4"
    assert evidence["local_asset"]["bytes"] == len(raw)
    assert len(evidence["local_asset"]["sha256"]) == 64
    assert "只接受 URL" in evidence["note"]


def test_video_url_wins_over_local_file(tmp_path):
    clip = tmp_path / "ref.mov"
    clip.write_bytes(b"\x00\x00\x00\x14ftypqt  local")
    value, evidence = resolve_video("https://cdn.test/ref.mp4", str(clip))
    assert value == "https://cdn.test/ref.mp4"
    assert evidence["sent_to_upstream"] is True
    assert evidence["kind"] == "url"
    # 本地素材仍留痕（事后可核对"这条视频是拿哪个素材跑的"）
    assert evidence["local_asset"]["name"] == "ref.mov"


def test_video_evidence_defaults_to_not_provided():
    assert resolve_video(None, None) == (None, {"provided": False})


def test_video_data_uri_is_summarised_in_evidence():
    value, evidence = resolve_video("data:video/mp4;base64,AAAA", None)
    assert value.startswith("data:video/mp4;base64,")
    assert evidence["kind"] == "data_uri" and evidence["sent_to_upstream"] is True
    assert evidence["bytes"] == 3
    assert sanitize_payload({"content": [{"type": "video_url", "video_url": {"url": value}}]})[
        "content"
    ][0]["video_url"]["url"].startswith("data:video/mp4;base64,<剥离 3 字节")


def test_rejected_video_face_is_blocked_only_when_the_error_names_video():
    """带参考视频的 4xx 归因：错误文案点到视频 ⇒ 平台能力缺失（blocked）。

    文案没点到视频（例如"提示词不合法"）时仍旧是上游明确拒绝本次请求 ⇒ failed，
    免得把任意 400 都放成"平台没开放"。
    """
    base = {
        "submit": {"status": "http_error", "http_status": 400, "error_message": ""},
        "task_status": None,
        "video_input": {"provided": True, "sent_to_upstream": True},
    }
    named = {**base, "submit": {**base["submit"], "error_message": 'invalid content block type "video_url"'}}
    assert verdict_of(named)[0] == "blocked"
    unnamed = {**base, "submit": {**base["submit"], "error_message": "prompt is too long"}}
    assert verdict_of(unnamed)[0] == "failed"

    # 没带视频块时同一句话仍是 failed：归因条件必须两条同时成立。
    no_video_request = {**named, "video_input": {"provided": False}}
    assert verdict_of(no_video_request)[0] == "failed"


def test_end_to_end_sends_video_url_and_archives_the_local_asset():
    clip_bytes = b"\x00\x00\x00\x18ftypmp42local-reference"
    submitted: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            submitted.append(json.loads(request.read()))
            return httpx.Response(200, json={"id": "cgt-video"})
        return httpx.Response(200, json={
            "id": "cgt-video", "status": "succeeded",
            "content": {"video_url": "https://cdn.test/cgt-video.mp4"},
        })

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        clip = root / "reference.mp4"
        clip.write_bytes(clip_bytes)
        target = root / SEEDANCE_CASE_ID
        config = _config(
            requests=1, concurrency=1, record_id="run-video-0001",
            video_url="https://cdn.test/reference.mp4", video_path=str(clip),
        )
        with _patch_transport(handler):
            outcome = asyncio.run(run_videos(config, root, result_dir=target))

        body = submitted[0]
        video_blocks = [item for item in body["content"] if item["type"] == "video_url"]
        assert video_blocks == [{
            "type": "video_url",
            "video_url": {"url": "https://cdn.test/reference.mp4"},
            "role": "reference_video",
        }]
        # 证据文件里不得出现 base64/视频字节，只有摘要与归档路径
        generated = (target / "generated_requests.jsonl").read_text(encoding="utf-8")
        assert "local-reference" not in generated
        archive = target / "inputs" / "input-video-reference.mp4"
        assert archive.is_file() and archive.read_bytes() == clip_bytes

        summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
        assert summary["video_role"] == "reference_video"
        assert summary["video_input"]["sent_to_upstream"] is True
        assert summary["video_input"]["local_asset"]["saved_as"] == "inputs/input-video-reference.mp4"
        assert summary["video_input_note"] is None
        assert outcome.rows[0]["video_input"]["sent_to_upstream"] is True
        report = (target / "report.md").read_text(encoding="utf-8")
        assert "输入参考视频：URL（https://cdn.test/reference.mp4）" in report


def test_end_to_end_local_only_video_is_archived_and_flagged_in_the_report(tmp_path):
    clip_bytes = b"\x00\x00\x00\x18ftypmp42local-only"
    submitted: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            submitted.append(json.loads(request.read()))
            return httpx.Response(200, json={"id": "cgt-only"})
        return httpx.Response(200, json={
            "id": "cgt-only", "status": "succeeded",
            "content": {"video_url": "https://cdn.test/cgt-only.mp4"},
        })

    clip = tmp_path / "only.mp4"
    clip.write_bytes(clip_bytes)
    target = tmp_path / SEEDANCE_CASE_ID
    config = _config(requests=1, concurrency=1, video_path=str(clip))
    with _patch_transport(handler):
        outcome = asyncio.run(run_videos(config, tmp_path, result_dir=target))

    # 只上传了本地文件：请求体里没有视频块，但素材照样归档、报告里明确写清
    assert [item["type"] for item in submitted[0]["content"]] == ["text"]
    assert (target / "inputs" / "input-video-only.mp4").read_bytes() == clip_bytes
    summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    assert summary["video_input"]["sent_to_upstream"] is False
    assert summary["video_role"] is None
    assert "只接受 URL" in summary["video_input_note"]
    report = (target / "report.md").read_text(encoding="utf-8")
    assert "输入参考视频：本地素材未提交给上游" in report
    assert f"> ⚠️ {summary['video_input_note']}" in report
    # 归档本地素材不影响判定：任务本身成功就是 passed
    assert outcome.rows[0]["verdict"] == "passed"


def test_generation_mode_is_recorded_in_the_evidence(tmp_path):
    """生成方式进 metadata/summary/报告；被模式排除的素材留痕但**不归档**。"""
    submitted: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            submitted.append(json.loads(request.read()))
            return httpx.Response(200, json={"id": "cgt-mode"})
        return httpx.Response(200, json={
            "id": "cgt-mode", "status": "succeeded",
            "content": {"video_url": "https://cdn.test/cgt-mode.mp4"},
        })

    image = tmp_path / "first.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    clip = tmp_path / "ref.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42x")
    target = tmp_path / SEEDANCE_CASE_ID
    config = _config(requests=1, mode="text", image_path=str(image), video_path=str(clip))
    with _patch_transport(handler):
        outcome = asyncio.run(run_videos(config, tmp_path, result_dir=target))

    assert [item["type"] for item in submitted[0]["content"]] == ["text"]
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "text" and metadata["mode_declared"] == "text"
    summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "text" and summary["mode_label"].startswith("仅文生视频")
    # 素材既不提交也不归档：生成方式没启用它，归档只会让"这批到底带没带素材"变含糊。
    assert summary["image"]["skipped_by_mode"] == "text"
    assert summary["video_input"]["skipped_by_mode"] == "text"
    assert summary["image_role"] is None and summary["video_role"] is None
    assert not (target / "inputs").exists()
    report = (target / "report.md").read_text(encoding="utf-8")
    assert "生成方式：仅文生视频" in report
    assert "已按生成方式忽略" in report
    assert outcome.rows[0]["mode"] == "text"


def test_cli_generation_mode_is_inferred_and_enforced(tmp_path):
    clip = tmp_path / "ref.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42x")
    # 不声明：按素材推导（老调用方式不受影响）。
    assert _cfg([]).mode == "text"
    assert _cfg(["--image-url", "https://cdn.test/a.png"]).mode == "image"
    assert _cfg(["--video-url", "https://cdn.test/a.mp4"]).mode == "video"
    assert _cfg(["--image-url", "https://cdn.test/a.png",
                 "--video-url", "https://cdn.test/a.mp4"]).mode == "both"
    # 只有本地视频（没地址）：沿用"归档留痕 + 报告 ⚠️"的旧行为，不拦——推导出的档位不做校验。
    assert _cfg(["--video-path", str(clip)]).mode == "video"
    # 环境变量同样受支持（Web 派发走的就是 env 通道）。
    assert _cfg([], {"SEEDANCE_MODE": "image", "SEEDANCE_IMAGE_URL": "https://cdn.test/a.png"}).mode == "image"

    # 显式声明：与输入严格对齐，多给/少给都拒。
    assert _cfg(["--mode", "both", "--image-url", "https://a/b.png",
                 "--video-url", "https://a/b.mp4"]).mode == "both"
    for argv, marker in (
        (["--mode", "image"], "没有图片输入"),
        (["--mode", "video"], "没有参考视频地址"),
        (["--mode", "both", "--image-url", "https://a/b.png"], "没有参考视频地址"),
        (["--mode", "video", "--video-path", str(clip)], "只给了本地视频文件不算"),
        (["--mode", "text", "--image-url", "https://a/b.png"], "不会提交图片"),
        (["--mode", "image", "--image-url", "https://a/b.png",
          "--video-url", "https://a/b.mp4"], "不会提交参考视频"),
    ):
        try:
            _cfg(argv)
        except ValueError as exc:
            assert marker in str(exc), f"{argv} 的报错文案应含「{marker}」，实际：{exc}"
        else:
            raise AssertionError(f"{argv} 应被拒绝")


def test_cli_mode_read_from_the_environment_must_be_known():
    try:
        _cfg([], {"SEEDANCE_MODE": "video2video"})
    except ValueError as exc:
        assert "生成方式" in str(exc)
    else:
        raise AssertionError("未知的 SEEDANCE_MODE 应被拒绝")


def test_cli_reads_video_inputs_and_rejects_bad_ones(tmp_path):
    clip = tmp_path / "ref.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42x")
    # 什么都不给 → 用默认角色（不是"不带 role"）——`_env()` 会把未设置与空串混为一谈，
    # 这条断言钉住这个坑不再出现。
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SEEDANCE_VIDEO_ROLE", None)
        assert _cfg([]).video_role == SEEDANCE_DEFAULT_VIDEO_ROLE
    config = _cfg([
        "--video-url", "https://cdn.test/ref.mp4",
        "--video-path", str(clip),
        "--video-role", "reference_video",
    ])
    assert config.video_url == "https://cdn.test/ref.mp4"
    assert config.video_path == str(clip)
    assert config.video_role == "reference_video"
    # 空串 = 不带 role 字段
    assert _cfg(["--video-role", ""]).video_role == ""

    bad_suffix = tmp_path / "ref.txt"
    bad_suffix.write_bytes(b"not a video")
    for argv in (
        ["--video-url", "ftp://x/a.mp4"],
        ["--video-role", "first_frame"],
        ["--video-path", "/no/such/clip.mp4"],
        ["--video-path", str(bad_suffix)],
    ):
        try:
            _cfg(argv)
        except ValueError:
            continue
        raise AssertionError(f"{argv} 应被拒绝")
