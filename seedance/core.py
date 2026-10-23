"""Seedance 视频生成执行核心。

协议面是火山方舟线格式的异步任务接口（``POST`` 提交 / ``GET`` 轮询）：

* 提交：``POST {root}/contents/generations/tasks``
  → ``{"id": "<任务 ID>"}``，任务 ID 仅保留 7 天；
* 查询：``GET {root}/contents/generations/tasks/{task_id}``
  → ``status`` ∈ ``queued`` / ``running`` / ``succeeded`` / ``failed`` / ``expired``，
  成功后 ``content.video_url`` 给出视频下载链接。

与平台其它套件一致的三条硬约束：

1. **证据链完整**：结果目录必须有非空 ``requests.jsonl``；每个任务一行，同时带
   布尔 ``success`` 与字符串 ``verdict``（warning/blocked 不得压平成 failed）。
2. **裁定只区分"是不是模型/平台问题"**：账号、权限、额度、限流、瞬时故障、任务
   未在观察窗口内结束 ⇒ ``blocked``（平台前提不满足）；任务成功但 200 无产出、
   上游明确拒绝 ⇒ ``failed``。
3. **base64 与视频二进制不进入证据**：上传图只记录 sha256/字节数，视频落盘后只记录
   路径、字节数与 sha256。
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote, urlparse

import httpx

from common.audit import redact
from common.image_store import ImageBudget, env_flag, format_budget
from common.report_style import stats_lines
from volcperf.protocols import api_root

from .cases import (
    SEEDANCE_CASE_ID,
    SEEDANCE_DEFAULT_DURATION,
    SEEDANCE_DEFAULT_GENERATE_AUDIO,
    SEEDANCE_DEFAULT_IMAGE_ROLE,
    SEEDANCE_DEFAULT_RATIO,
    SEEDANCE_DEFAULT_RESOLUTION,
    SEEDANCE_DEFAULT_VIDEO_ROLE,
    SEEDANCE_DEFAULT_WATERMARK,
    SEEDANCE_IMAGE_ROLES,
    SEEDANCE_INPUT_VIDEO_MIME_EXTENSIONS,
    SEEDANCE_MODE_AUTO,
    SEEDANCE_MODES,
    SEEDANCE_VIDEO_PATTERN,
    mode_allows_image,
    mode_allows_video,
    resolve_mode,
)

VIDEO_PROTOCOL = "seedance"
SUITE_NAME = "seedance"
TEST_KIND = "seedance"
SCHEMA_VERSION = "mltest.seedance.v1"

_MODE_LABELS = {mode: label for mode, label in SEEDANCE_MODES}


def mode_label(mode: str) -> str:
    """生成方式的中文标签（报告与前端提示共用同一份文案）。"""
    return _MODE_LABELS.get(str(mode or "").strip().lower(), str(mode or ""))

VIDEO_DIR_NAME = "videos"
INPUT_DIR_NAME = "inputs"

DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
# 单任务从提交到终态的墙钟上限。720p/5s 通常几十秒；1080p/12s 或排队高峰会更久，
# 因此给到 15 分钟——超时只记 blocked（"没等到结果"不是模型结论）。
DEFAULT_TASK_TIMEOUT_SECONDS = 900.0
DEFAULT_MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.0
# 单批保留视频总量上限。视频原文件动辄几十 MB，默认上限 4 GiB，超出部分显式跳过并记录。
DEFAULT_MAX_VIDEO_GIB = 4.0
KEEP_VIDEOS_ENV = "SEEDANCE_KEEP_VIDEOS"
MAX_VIDEO_GIB_ENV = "SEEDANCE_MAX_VIDEO_GIB"
# 轮询事件留证上限：防止超长任务把 requests.jsonl 撑成几十万行；截断会显式记录。
MAX_POLL_EVENTS = 400

# 账号/权限/额度/端点不可用/限流：按 blocked 记录（平台前提不满足，不是模型结论）。
_BLOCKED_HTTP_STATUSES = {401, 402, 403, 404, 429}
# 任务终态。
_TERMINAL_STATUSES = {"succeeded", "success", "completed", "failed", "expired", "cancelled", "canceled"}
_OK_STATUSES = {"succeeded", "success", "completed"}
_FAIL_STATUSES = {"failed", "error"}
_EXPIRED_STATUSES = {"expired", "cancelled", "canceled"}

_VIDEO_EXTENSIONS = {"mp4", "webm", "mov", "mkv", "m4v", "avi"}
_MIME_EXTENSIONS = {
    "video/mp4": "mp4",
    "video/webm": "webm",
    "video/quicktime": "mov",
    "video/x-matroska": "mkv",
}
_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;,]+)?(?P<b64>;base64)?,(?P<payload>.*)$", re.DOTALL)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VideoConfig:
    api_key: str
    base_url: str
    model: str
    prompt: str
    resolution: str = SEEDANCE_DEFAULT_RESOLUTION
    ratio: str = SEEDANCE_DEFAULT_RATIO
    duration: int = SEEDANCE_DEFAULT_DURATION
    generate_audio: bool = SEEDANCE_DEFAULT_GENERATE_AUDIO
    watermark: bool = SEEDANCE_DEFAULT_WATERMARK
    # 生成方式（四档 text/image/video/both；空串 = 按输入推导，见 cases.resolve_mode）。
    # 它是**请求体形状的唯一开关**：声明 text 时图片与参考视频一律不进 content。
    mode: str = SEEDANCE_MODE_AUTO
    image_url: str = ""
    image_path: str = ""
    image_role: str = SEEDANCE_DEFAULT_IMAGE_ROLE
    # 参考视频：只有 video_url 会真的发给上游；video_path 是本地素材（归档留痕用）。
    video_url: str = ""
    video_path: str = ""
    video_role: str = SEEDANCE_DEFAULT_VIDEO_ROLE
    requests: int = 1
    concurrency: int = 1
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    task_timeout_seconds: float = DEFAULT_TASK_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    keep_videos: bool = True
    max_video_bytes: int = int(DEFAULT_MAX_VIDEO_GIB * 1024**3)
    verify_ssl: bool = True
    trust_env: bool = False
    record_id: str = ""

    @property
    def endpoint(self) -> str:
        return task_endpoint(self.base_url)


def task_endpoint(base_url: str) -> str:
    """任务集合端点（提交用）。"""
    return f"{api_root(VIDEO_PROTOCOL, base_url)}/contents/generations/tasks"


def task_status_endpoint(base_url: str, task_id: str) -> str:
    """单个任务查询端点。task_id 来自响应体，属于不可信输入，必须转义后再拼路径。"""
    return f"{task_endpoint(base_url)}/{quote(str(task_id), safe='')}"


def request_headers(api_key: str, request_id: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Request-ID": request_id,
        "Authorization": f"Bearer {api_key}",
    }


# ---------------------------------------------------------------------------
# 请求体与图片
# ---------------------------------------------------------------------------
def build_content(
    prompt: str,
    image_value: str | None = None,
    image_role: str | None = None,
    video_value: str | None = None,
    video_role: str | None = None,
    *,
    mode: str = SEEDANCE_MODE_AUTO,
) -> list[dict[str, Any]]:
    """``content`` 数组：文本提示词在前，可选图片、参考视频在后（各带角色）。

    图片与视频可以同时出现（首帧图 + 参考视频），顺序固定为 文本 → 图片 → 视频，
    保证同一组输入生成完全一致的请求体（可复现、可核对）。

    **生成方式在这里兜底**：``mode`` 显式声明时不带该模式允许的输入就一律不出现
    （声明 text 时即便调用方多传了 image_value / video_value 也会被丢掉）。这是"仅文生
    视频不提交图片与视频"的最后一道闸——比"指望上层记得清空输入"可靠得多。
    """
    resolved = resolve_mode(mode, has_image=bool(image_value), has_video=bool(video_value))
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_value and mode_allows_image(resolved):
        content.append({
            "type": "image_url",
            "image_url": {"url": image_value},
            "role": image_role or SEEDANCE_DEFAULT_IMAGE_ROLE,
        })
    if video_value and mode_allows_video(resolved):
        block: dict[str, Any] = {"type": "video_url", "video_url": {"url": video_value}}
        # 空串 = 界面选了"不指定角色"：整条 role 字段都不出现，便于区分
        # "上游不接受这个 role" 与 "上游不接受视频块本身"。
        if video_role:
            block["role"] = video_role
        content.append(block)
    return content


def payload_mode(
    mode: str | None, *, image_url: str = "", image_path: str = "", video_url: str = "", video_path: str = ""
) -> str:
    """把配置里的生成方式收口成四档之一（空值按**素材**推导，含只有本地文件的情形）。"""
    return resolve_mode(
        mode,
        has_image=bool(image_url or image_path),
        has_video=bool(video_url or video_path),
    )


def build_payload(
    config: VideoConfig,
    image_value: str | None,
    video_value: str | None = None,
) -> dict[str, Any]:
    """构造视频生成任务请求体。

    参数一律走**顶层字段**（官方推荐写法）：把 ``--resolution`` 之类塞进提示词文本里
    既容易被模型当成正文，也让证据无法结构化核对。图片/视频块由 ``config.mode`` 决定，
    未启用该输入时即便传了值也不会出现在请求体里。
    """
    return {
        "model": config.model,
        "content": build_content(config.prompt, image_value, config.image_role,
                                 video_value, config.video_role, mode=config.mode),
        "resolution": config.resolution,
        "ratio": config.ratio,
        "duration": int(config.duration),
        "watermark": bool(config.watermark),
        "generate_audio": bool(config.generate_audio),
    }


def _split_data_uri(value: str) -> tuple[str, bytes]:
    match = _DATA_URI_RE.match(value)
    if not match:
        raise ValueError("不是合法的 data URI")
    mime = (match.group("mime") or "").strip() or "application/octet-stream"
    payload = match.group("payload")
    raw = base64.b64decode(payload, validate=False) if match.group("b64") else payload.encode("utf-8")
    return mime, raw


def _guess_image_mime(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
        "bmp": "image/bmp",
    }.get(suffix, "application/octet-stream")


def resolve_image(image_url: str | None, image_path: str | None) -> tuple[str | None, dict[str, Any]]:
    """把界面输入的图片收口成 (发给上游的 url 值, 可入库的证据描述)。

    三种来源：本地上传文件（上传接口落盘的路径）、``data:`` URI、``http(s)`` URL。
    前两者会被转成/保留为 data URI —— 原始 base64 只发给上游，**不写进证据文件**。
    """
    if image_path:
        path = Path(image_path)
        raw = path.read_bytes()
        mime = _guess_image_mime(path)
        value = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
        return value, {
            "provided": True,
            "kind": "file",
            "name": path.name,
            "mime": mime,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if image_url:
        value = image_url.strip()
        if value.startswith("data:"):
            mime, raw = _split_data_uri(value)
            return value, {
                "provided": True,
                "kind": "data_uri",
                "mime": mime,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        return value, {"provided": True, "kind": "url", "url": value}
    return None, {"provided": False}


def _guess_video_mime(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    for mime, extension in SEEDANCE_INPUT_VIDEO_MIME_EXTENSIONS.items():
        if extension == suffix:
            return mime
    return "application/octet-stream"


def resolve_video(video_url: str | None, video_path: str | None) -> tuple[str | None, dict[str, Any]]:
    """把界面输入的参考视频收口成 (发给上游的 url 值, 可入库的证据描述)。

    与图片的关键差异：**本地文件不能提交给上游**。异步任务面只接受 URL，图片那套
    "读成 data URI 内联发送"的写法对视频不可行——64 MiB 视频 base64 后约 85 MiB，
    既撑爆 JSON 请求体，也会把证据文件变成不可读的大对象。因此：

    * ``video_url`` 是**唯一**能真正生效的输入（原样透传，``data:`` URI 只留摘要）；
    * ``video_path``（本地上传）只做**归档留痕**：记录名字 / 字节 / sha256，由
      ``run_videos`` 复制进结果目录 ``inputs/``，并在报告里注明"未随任务提交"。

    两者都填时**以 video_url 为准**（与图片"文件优先"相反，理由同上），
    本地文件仍会归档，便于日后核对"这条视频是拿哪个素材跑的"。
    """
    local: dict[str, Any] = {}
    if video_path:
        path = Path(video_path)
        raw = path.read_bytes()
        local = {
            "kind": "file",
            "name": path.name,
            "mime": _guess_video_mime(path),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if video_url:
        value = video_url.strip()
        if value.startswith("data:"):
            mime, raw = _split_data_uri(value)
            evidence: dict[str, Any] = {
                "provided": True, "kind": "data_uri", "sent_to_upstream": True,
                "mime": mime, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            }
            # 本地素材只作为附带证据记录；"未提交"的提示位（note）只属于"只有本地文件"那一支，
            # 否则一个正常的"URL 提交 + 本地留档"也会被渲染成 ⚠️。
            if local:
                evidence["local_asset"] = local
            return value, evidence
        evidence = {"provided": True, "kind": "url", "url": value, "sent_to_upstream": True}
        if local:
            evidence["local_asset"] = local
        return value, evidence
    if local:
        return None, {
            "provided": True,
            "kind": "file",
            "sent_to_upstream": False,
            "local_asset": local,
            "note": ("本地视频只归档留痕，未随任务提交（异步任务面只接受 URL，没有文件直传通道）；"
                     "要让上游使用参考视频，请填写视频地址"),
        }
    return None, {"provided": False}


def sanitize_payload(value: Any) -> Any:
    """把请求体里的大体量内联数据换成摘要标记，保证证据文件可读且不夹带 base64。"""
    if isinstance(value, dict):
        return {key: sanitize_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
        mime, raw = _split_data_uri(value)
        return (f"data:{mime};base64,<剥离 {len(raw)} 字节 "
                f"sha256={hashlib.sha256(raw).hexdigest()[:16]}>")
    return value


# ---------------------------------------------------------------------------
# 响应解析（对"信封再包一层"的中转保持容忍）
# ---------------------------------------------------------------------------
def _json_or_none(response: httpx.Response | None) -> Any:
    if response is None:
        return None
    try:
        return response.json()
    except (ValueError, json.JSONDecodeError):
        return None


def _unwrap(body: Any) -> dict[str, Any]:
    """兼容两种信封：方舟原生（顶层 id/status/content）与 ``{"code":0,"data":{...}}``。"""
    if not isinstance(body, dict):
        return {}
    inner = body.get("data")
    if isinstance(inner, dict) and any(
        key in inner for key in ("id", "task_id", "status", "content", "video_url")
    ):
        return {**body, **inner}
    return body


def _first_str(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    return None


EVIDENCE_BODY_LIMIT = 2000


def _evidence_json(value: Any, limit: int = EVIDENCE_BODY_LIMIT) -> str | None:
    """把原始响应体降成一条**有上限的**证据字符串；空值一律返回 None。

    留原文是为了定位「上游到底拒了哪个参数」——笼统的失败文案 + 白名单式回显不足以
    定位（首跑实测：任务回 `请求参数校验失败`，但 model/resolution/ratio/duration 一个
    都没回显）。截断只为控制证据体积，超长时显式标出被截断。
    """
    if value is None or value == "" or value == {} or value == []:
        return None
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > limit:
        return f"{text[:limit]}…（共 {len(text)} 字符，已截断）"
    return text


def extract_task_id(body: Any) -> str | None:
    unwrapped = _unwrap(body)
    return _first_str(unwrapped.get("id"), unwrapped.get("task_id"), unwrapped.get("taskId"))


def extract_task(body: Any) -> dict[str, Any]:
    """从任务查询响应里提取状态、视频链接、用量与回显参数。"""
    unwrapped = _unwrap(body)
    content = unwrapped.get("content") if isinstance(unwrapped.get("content"), dict) else {}
    error = unwrapped.get("error")
    error_text = None
    error_code = None
    if isinstance(error, dict):
        # message 与 code 都要留：中转（New API / One API 一类）常把真正的机器可读原因
        # 放在 `error.code`，只取 message 会让「请求参数校验失败」这类笼统文案彻底失去线索。
        error_text = _first_str(error.get("message"), error.get("code"), error.get("type"))
        error_code = _first_str(error.get("code"), error.get("type"))
    elif isinstance(error, str):
        error_text = error
    status = _first_str(unwrapped.get("status"), unwrapped.get("state"))
    return {
        "task_id": extract_task_id(body),
        "status": status.lower() if isinstance(status, str) else None,
        "video_url": _first_str(content.get("video_url"), content.get("videoUrl"),
                                unwrapped.get("video_url"), unwrapped.get("videoUrl")),
        "last_frame_url": _first_str(content.get("last_frame_url"), content.get("lastFrameUrl"),
                                     unwrapped.get("last_frame_url")),
        "usage": unwrapped.get("usage") if isinstance(unwrapped.get("usage"), dict) else None,
        "error": error_text,
        "error_code": error_code,
        # 顶层键名本身就是证据：任务响应「到底回显了哪些参数」只能从这里看出来，
        # 而 echo 的固定白名单一旦与上游字段名对不上，就会得到一个空 dict 而无从判断。
        "body_keys": sorted(str(key) for key in unwrapped.keys()),
        "echo": {
            key: unwrapped.get(key)
            for key in ("model", "resolution", "ratio", "duration", "watermark",
                        "generate_audio", "framespersecond", "seed", "created_at", "updated_at")
            if key in unwrapped
        },
    }


# ---------------------------------------------------------------------------
# 提交 / 轮询
# ---------------------------------------------------------------------------
async def submit_task(
    client: httpx.AsyncClient,
    config: VideoConfig,
    payload: dict[str, Any],
    *,
    request_id: str,
) -> dict[str, Any]:
    """提交任务；瞬时故障（5xx / 传输错误）按配置重试，其余不做无谓重试。"""
    endpoint = task_endpoint(config.base_url)
    started_mono = time.perf_counter()
    attempts = 0
    result: dict[str, Any] = {
        "status": "transport_error",
        "http_status": None,
        "task_id": None,
        "attempts": 0,
        "elapsed_seconds": None,
        "response_headers": {},
        "error_message": None,
        "response_body_shape": {"keys": [], "task_id": None},
    }
    while attempts <= max(0, config.max_retries):
        attempts += 1
        try:
            response = await client.post(
                endpoint, json=payload, headers=request_headers(config.api_key, request_id)
            )
        except httpx.TimeoutException as exc:
            result.update(status="transport_error", error_message=f"timeout: {exc}")
            if attempts <= config.max_retries:
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempts)
                continue
            break
        except httpx.HTTPError as exc:
            result.update(status="transport_error", error_message=f"{type(exc).__name__}: {exc}")
            if attempts <= config.max_retries:
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempts)
                continue
            break
        body = _json_or_none(response)
        result["http_status"] = response.status_code
        result["response_headers"] = _safe_response_headers(response)
        result["response_body_shape"] = {
            "keys": sorted(body.keys()) if isinstance(body, dict) else [],
            "task_id": extract_task_id(body),
        }
        if 200 <= response.status_code < 300:
            task_id = extract_task_id(body)
            if task_id:
                result.update(status="ok", task_id=task_id, error_message=None)
            else:
                result.update(
                    status="invalid_response",
                    error_message=f"HTTP {response.status_code} 但响应未包含任务 ID：{response.text[:400]}",
                )
            break
        result["error_message"] = response.text[:2000]
        if response.status_code >= 500 and attempts <= config.max_retries:
            await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempts)
            continue
        result["status"] = "http_error"
        break
    result["attempts"] = attempts
    result["elapsed_seconds"] = round(time.perf_counter() - started_mono, 3)
    return result


async def poll_task(
    client: httpx.AsyncClient,
    config: VideoConfig,
    task_id: str,
    *,
    request_id_prefix: str,
) -> dict[str, Any]:
    """轮询任务状态直到终态或墙钟上限。

    "未取得终态"本身不是模型结论：任务可能只是排队/生成得慢，因此这一分支在裁定里
    落 ``blocked``，绝不落 ``failed``。
    """
    endpoint = task_status_endpoint(config.base_url, task_id)
    deadline = time.perf_counter() + max(1.0, config.task_timeout_seconds)
    interval = max(0.1, config.poll_interval_seconds)
    events: list[dict[str, Any]] = []
    attempt = 0
    last_status: str | None = None
    task: dict[str, Any] = {}
    last_body: Any = None
    timed_out = False
    poll_error: str | None = None
    while True:
        attempt += 1
        http_status: int | None = None
        body: Any = None
        error: str | None = None
        try:
            response = await client.get(
                endpoint,
                headers=request_headers(config.api_key, f"{request_id_prefix}-poll-{attempt}"),
            )
            http_status = response.status_code
            body = _json_or_none(response)
            if not 200 <= http_status < 300:
                error = response.text[:400]
        except httpx.HTTPError as exc:
            error = f"{type(exc).__name__}: {exc}"
        if body is not None:
            task = extract_task(body)
            last_body = body
            if task.get("status"):
                last_status = task["status"]
        if len(events) < MAX_POLL_EVENTS:
            events.append({
                "attempt": attempt,
                "http_status": http_status,
                "status": (task.get("status") if body is not None else None) or last_status,
                "elapsed_seconds": round(max(0.0, config.task_timeout_seconds - (deadline - time.perf_counter())), 3),
                "error": error,
            })
        if last_status in _TERMINAL_STATUSES:
            break
        # 明确不会自愈的错误（401/403/404 等）：继续轮询也没有意义。
        if isinstance(http_status, int) and 400 <= http_status < 500 and http_status != 429:
            poll_error = f"HTTP {http_status}：{error or ''}"[:600]
            break
        if time.perf_counter() >= deadline:
            timed_out = True
            break
        await asyncio.sleep(min(interval, max(0.0, deadline - time.perf_counter())))
    return {
        "attempts": attempt,
        "events": events,
        "events_truncated": attempt > MAX_POLL_EVENTS,
        "task": task,
        "task_body": last_body,
        "task_status": last_status,
        "timed_out": timed_out,
        "poll_error": poll_error,
    }


def _safe_response_headers(response: httpx.Response) -> dict[str, str]:
    keep = {
        "content-type", "server", "via", "x-request-id", "request-id", "x-trace-id",
        "x-oneapi-request-id", "cf-ray", "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",
    }
    return {key.lower(): value for key, value in response.headers.items() if key.lower() in keep}


# ---------------------------------------------------------------------------
# 裁定
# ---------------------------------------------------------------------------
def _mentions_video_input(text: str) -> bool:
    """错误文案是否指向参考视频字段。

    窄口径：只有原文提到 ``video`` / ``视频`` 才算"上游不认这个字段"。这样
    "提示词不合法""分辨率不支持"之类的业务拒绝仍落 failed，不会被误放成 blocked。
    """
    lowered = (text or "").lower()
    return "video" in lowered or "视频" in lowered


def verdict_of(row: dict[str, Any]) -> tuple[str, str]:
    """把一次任务的原始观测收口成裁定（passed / failed / warning / blocked）。"""
    submit = row.get("submit") or {}
    state = submit.get("status")
    submit_status = submit.get("http_status")
    if state == "transport_error":
        return "blocked", f"提交请求未能完成（{submit.get('error_message')}）"
    if state == "invalid_response":
        return "failed", str(submit.get("error_message") or "提交返回 2xx 但没有任务 ID，无法查询状态")
    if state == "http_error":
        message = str(submit.get("error_message") or "")
        if submit_status in _BLOCKED_HTTP_STATUSES:
            return "blocked", f"HTTP {submit_status}：账号/权限/额度或端点不可用（{message[:200]}）"
        if isinstance(submit_status, int) and submit_status >= 500:
            return "blocked", f"HTTP {submit_status}：上游/链路瞬时故障（{message[:200]}）"
        # 参考视频是**未经实测的新增输入面**：上游对 video_url 块（或它的 role）不支持时
        # 属于平台能力缺失，按 blocked 记录（不是模型缺陷），条件有两条——本请求确实带了
        # 视频块，且错误文案点到了视频。两者缺一仍旧按"上游拒绝本次请求"记 failed。
        if (row.get("video_input") or {}).get("sent_to_upstream") and _mentions_video_input(message):
            return "blocked", (
                "HTTP "
                f"{submit_status}：上游拒绝带参考视频的请求（该模型/网关可能未开放 video_url 面）："
                f"{message[:240]}"
            )
        return "failed", f"HTTP {submit_status}：上游拒绝了本次生成请求（{message[:300]}）"

    status = row.get("task_status")
    if status in _OK_STATUSES:
        if row.get("video_url"):
            return "passed", "任务成功并返回视频链接"
        return "failed", "任务状态为成功，但响应未给出视频链接（200 无产出）"
    if status in _FAIL_STATUSES:
        message = str(row.get("task_error") or "")
        detail = f"任务失败：{message or '上游未给出错误详情'}"
        code = row.get("task_error_code")
        if code and code not in detail:
            detail += f"（code={code}）"
        return "failed", detail[:300]
    if status in _EXPIRED_STATUSES:
        return "blocked", f"任务状态 {status}：平台侧任务被清理/取消，未产出视频"
    if row.get("poll_error"):
        return "blocked", f"轮询任务状态失败：{row['poll_error']}"
    if row.get("timed_out"):
        return "blocked", (
            f"任务在 {row.get('task_timeout_seconds')} 秒观察窗口内未达终态"
            f"（最后状态：{status or '未知'}）"
        )
    return "blocked", f"未取得任务终态（最后状态：{status or '未知'}）"


# ---------------------------------------------------------------------------
# 视频落盘
# ---------------------------------------------------------------------------
def video_extension(url: str, content_type: str | None) -> str:
    if content_type:
        mime = str(content_type).split(";")[0].strip().lower()
        if mime in _MIME_EXTENSIONS:
            return _MIME_EXTENSIONS[mime]
    suffix = Path(urlparse(url).path).suffix.lower().lstrip(".")
    return suffix if suffix in _VIDEO_EXTENSIONS else "mp4"


def _write_video(directory: Path, name: str, raw: bytes, budget: ImageBudget | None) -> dict[str, Any]:
    if budget is not None and not budget.claim(len(raw)):
        return {
            "saved": False,
            "bytes": len(raw),
            "error": (f"达到视频保留上限（{budget.limit_bytes} 字节），已跳过；"
                      f"可提高 --max-video-gib 或设置 {MAX_VIDEO_GIB_ENV}"),
        }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(raw)
    return {
        "saved": True,
        "file": f"{VIDEO_DIR_NAME}/{name}",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "content_type": None,
    }


async def download_video(
    client: httpx.AsyncClient,
    url: str,
    directory: Path | None,
    *,
    name: str,
    budget: ImageBudget | None = None,
) -> dict[str, Any] | None:
    """把生成的视频下载到结果目录；失败只记录，不改变任务裁定。"""
    if directory is None or not url:
        return None
    try:
        response = await client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return {"saved": False, "url": url, "error": f"下载失败: {type(exc).__name__}: {exc}"}
    raw = response.content
    content_type = response.headers.get("content-type")
    try:
        entry = await asyncio.to_thread(_write_video, directory, name, raw, budget)
    except OSError as exc:
        return {"saved": False, "url": url, "error": f"写入失败: {type(exc).__name__}: {exc}"}
    entry["url"] = url
    entry["content_type"] = content_type
    return entry


# ---------------------------------------------------------------------------
# 请求计划与执行
# ---------------------------------------------------------------------------
@dataclass
class VideoOutcome:
    result_dir: Path
    rows: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def build_request_plan(config: VideoConfig, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """执行任何 HTTP 请求之前先生成全部请求定义（可复现、可核对）。"""
    sanitized = sanitize_payload(payload)
    mode = payload_mode(
        config.mode,
        image_url=config.image_url, image_path=config.image_path,
        video_url=config.video_url, video_path=config.video_path,
    )
    return [
        {
            "case": SEEDANCE_CASE_ID,
            "index": index + 1,
            "request_id": f"{SEEDANCE_CASE_ID}-{index + 1}-{uuid.uuid4().hex[:10]}",
            "model": config.model,
            "mode": mode,
            "resolution": config.resolution,
            "ratio": config.ratio,
            "duration": int(config.duration),
            "generate_audio": bool(config.generate_audio),
            "watermark": bool(config.watermark),
            "image_role": config.image_role if payload_has_image(sanitized) else None,
            "video_role": config.video_role if payload_has_video(sanitized) else None,
            "payload": sanitized,
        }
        for index in range(max(1, int(config.requests)))
    ]


def payload_has_image(payload: dict[str, Any]) -> bool:
    content = payload.get("content")
    return any(isinstance(item, dict) and item.get("type") == "image_url" for item in content or [])


def payload_has_video(payload: dict[str, Any]) -> bool:
    content = payload.get("content")
    return any(isinstance(item, dict) and item.get("type") == "video_url" for item in content or [])


async def _run_one(
    client: httpx.AsyncClient,
    config: VideoConfig,
    plan: dict[str, Any],
    payload: dict[str, Any],
    *,
    video_dir: Path | None,
    budget: ImageBudget | None,
    record_id: str,
) -> dict[str, Any]:
    index = int(plan["index"])
    started_epoch = time.time()
    started_mono = time.perf_counter()
    submit = await submit_task(client, config, payload, request_id=plan["request_id"])
    poll: dict[str, Any] = {"attempts": 0, "events": [], "task_status": None, "task": {}, "timed_out": False, "poll_error": None}
    if submit["status"] == "ok":
        poll = await poll_task(client, config, submit["task_id"], request_id_prefix=plan["request_id"])
    task = poll.get("task") or {}
    row: dict[str, Any] = {
        "case": SEEDANCE_CASE_ID,
        "index": index,
        "request_id": plan["request_id"],
        "protocol": VIDEO_PROTOCOL,
        "endpoint": task_endpoint(config.base_url),
        "model": config.model,
        "mode": plan.get("mode"),
        "prompt": config.prompt,
        "resolution": config.resolution,
        "ratio": config.ratio,
        "duration": int(config.duration),
        "generate_audio": bool(config.generate_audio),
        "watermark": bool(config.watermark),
        "image": plan.get("image_evidence") or {"provided": False},
        "video_input": plan.get("video_input_evidence") or {"provided": False},
        "started_at": datetime.fromtimestamp(started_epoch, timezone.utc).isoformat(),
        "started_epoch": started_epoch,
        "completed_epoch": time.time(),
        "submit": submit,
        "polls": poll.get("events") or [],
        "poll_count": poll.get("attempts") or 0,
        "polls_truncated": bool(poll.get("events_truncated")),
        "task_status": poll.get("task_status"),
        "task_id": submit.get("task_id") or task.get("task_id"),
        "task_error": task.get("error"),
        "task_error_code": task.get("error_code"),
        "task_body_keys": task.get("body_keys") or [],
        "task_body_sample": _evidence_json(poll.get("task_body")),
        "task_echo": task.get("echo") or {},
        "video_url": task.get("video_url"),
        "last_frame_url": task.get("last_frame_url"),
        "usage": task.get("usage"),
        "http_status": submit.get("http_status"),
        "transport_error": submit.get("status") == "transport_error",
        "task_timeout_seconds": config.task_timeout_seconds,
        "timed_out": bool(poll.get("timed_out")),
        "poll_error": poll.get("poll_error"),
        "error_type": None,
        "error_message": None,
        "video": None,
        "video_save_error": None,
    }
    elapsed = time.perf_counter() - started_mono
    row["elapsed_seconds"] = round(elapsed, 3)
    row["timing"] = {"e2e_ms": round(elapsed * 1000, 3)}
    if submit.get("status") == "ok" and poll.get("task_status"):
        row["generation_seconds"] = round(
            max(0.0, elapsed - float(submit.get("elapsed_seconds") or 0.0)), 3
        )
    verdict, detail = verdict_of(row)
    row["verdict"], row["verdict_detail"] = verdict, detail
    row["success"] = verdict == "passed"
    if verdict != "passed":
        row["error_type"] = {
            "failed": f"http_{submit.get('http_status')}" if submit.get("status") == "http_error" else "task_failed",
            "blocked": "blocked",
            "warning": "warning",
        }.get(verdict, verdict)
        row["error_message"] = detail

    if row["success"] and config.keep_videos and video_dir is not None:
        name = f"video-{index:04d}.{video_extension(row['video_url'] or '', None)}"
        entry = await download_video(
            client, row["video_url"], video_dir, name=name, budget=budget
        )
        row["video"] = entry
        if entry and not entry.get("saved"):
            row["video_save_error"] = entry.get("error")
    if row.get("video") and row["video"].get("saved") and record_id:
        # artifact_url 只在**浏览器能直接播放**的格式上给出：Web 端的产物白名单是
        # `SEEDANCE_VIDEO_PATTERN`，写上白名单外的名字只会得到一个 404 链接。
        # 非白名单格式（如上游给 .avi）仍会落盘并记录路径，只是不给可点击链接。
        if SEEDANCE_VIDEO_PATTERN.fullmatch(str(row["video"].get("file") or "")):
            row["artifact_url"] = f"/api/runs/{record_id}/artifact/{row['video']['file']}"
        else:
            row["artifact_note"] = "非浏览器可播放格式，未提供在线预览链接（文件已在结果目录内归档）"
    return row


async def _run_requests(
    *,
    config: VideoConfig,
    request_plan: list[dict[str, Any]],
    payload: dict[str, Any],
    requests_path: Path,
    video_dir: Path | None,
    budget: ImageBudget | None,
    record_id: str,
    progress: Callable[[int, int, int], None] | None = None,
) -> list[dict[str, Any]]:
    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in range(len(request_plan)):
        queue.put_nowait(index)
    rows: list[dict[str, Any]] = []
    rows_lock = asyncio.Lock()
    timeout = httpx.Timeout(
        config.request_timeout_seconds,
        connect=min(30.0, config.request_timeout_seconds),
        write=min(30.0, config.request_timeout_seconds),
        pool=min(30.0, config.request_timeout_seconds),
    )
    limits = httpx.Limits(
        max_connections=max(1, config.concurrency),
        max_keepalive_connections=max(1, config.concurrency),
        keepalive_expiry=60.0,
    )
    # 增量落盘：SIGTERM 或进程崩溃不得丢掉已经完成的任务证据。
    with requests_path.open("w", encoding="utf-8", buffering=1) as stream:
        async with httpx.AsyncClient(
            timeout=timeout, limits=limits, verify=config.verify_ssl, trust_env=config.trust_env
        ) as client:
            async def worker() -> None:
                while True:
                    try:
                        index = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    plan = request_plan[index]
                    try:
                        row = await _run_one(
                            client, config, plan, payload,
                            video_dir=video_dir, budget=budget, record_id=record_id,
                        )
                    except Exception as exc:  # 单个任务异常不得中断整批
                        row = _failed_request_row(
                            plan, config, error_type=f"internal_{type(exc).__name__}", error_message=str(exc)
                        )
                    finally:
                        queue.task_done()
                    async with rows_lock:
                        rows.append(row)
                        stream.write(json.dumps(redact(row), ensure_ascii=False, default=str) + "\n")
                        stream.flush()
                        if progress:
                            progress(len(rows), len(request_plan),
                                     sum(1 for item in rows if item.get("success")))
            await asyncio.gather(
                *(asyncio.create_task(worker())
                  for _ in range(min(max(1, config.concurrency), len(request_plan))))
            )
    return sorted(rows, key=lambda item: item["index"])


def _failed_request_row(
    plan: dict[str, Any],
    config: VideoConfig,
    *,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    """即使内部代码异常，也写出字段完整的请求行（证据契约要求 success 为布尔值）。"""
    return {
        "case": SEEDANCE_CASE_ID,
        "index": int(plan["index"]),
        "request_id": plan["request_id"],
        "protocol": VIDEO_PROTOCOL,
        "endpoint": task_endpoint(config.base_url),
        "model": config.model,
        "mode": plan.get("mode"),
        "prompt": config.prompt,
        "resolution": config.resolution,
        "ratio": config.ratio,
        "duration": int(config.duration),
        "generate_audio": bool(config.generate_audio),
        "watermark": bool(config.watermark),
        "image": plan.get("image_evidence") or {"provided": False},
        "video_input": plan.get("video_input_evidence") or {"provided": False},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "started_epoch": time.time(),
        "completed_epoch": time.time(),
        "submit": {"status": "transport_error", "http_status": None, "task_id": None,
                   "attempts": 0, "elapsed_seconds": None, "response_headers": {},
                   "error_message": error_message, "response_body_shape": {"keys": [], "task_id": None}},
        "polls": [],
        "poll_count": 0,
        "polls_truncated": False,
        "task_status": None,
        "task_id": None,
        "task_error": None,
        "task_error_code": None,
        "task_body_keys": [],
        "task_body_sample": None,
        "task_echo": {},
        "video_url": None,
        "last_frame_url": None,
        "usage": None,
        "http_status": None,
        "transport_error": True,
        "task_timeout_seconds": config.task_timeout_seconds,
        "timed_out": False,
        "poll_error": None,
        "error_type": error_type,
        "error_message": error_message,
        "video": None,
        "video_save_error": None,
        "elapsed_seconds": 0.0,
        "timing": {"e2e_ms": 0.0},
        "success": False,
        "verdict": "blocked",
        "verdict_detail": f"{error_type}: {error_message}",
    }


# ---------------------------------------------------------------------------
# 汇总与报告
# ---------------------------------------------------------------------------
def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low, high = int(index), min(len(ordered) - 1, int(index) + 1)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize_videos(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    saved = saved_bytes = skipped = 0
    for row in rows:
        entry = row.get("video")
        if not isinstance(entry, dict):
            continue
        if entry.get("saved"):
            saved += 1
            saved_bytes += int(entry.get("bytes") or 0)
        else:
            skipped += 1
    return {"videos_saved": saved, "video_bytes_saved": saved_bytes, "videos_skipped": skipped}


def build_summary(
    config: VideoConfig,
    rows: list[dict[str, Any]],
    *,
    endpoint: str,
    image_evidence: dict[str, Any],
    requested: int,
    video_input: dict[str, Any] | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    verdicts = [str(row.get("verdict") or "failed") for row in rows]
    passed = sum(1 for item in verdicts if item == "passed")
    blocked = sum(1 for item in verdicts if item == "blocked")
    warning = sum(1 for item in verdicts if item == "warning")
    hard_failed = sum(1 for item in verdicts if item in {"failed", "error"})
    video_totals = summarize_videos(rows)
    durations = [float(row["elapsed_seconds"]) for row in rows if isinstance(row.get("elapsed_seconds"), (int, float))]
    resolved_mode = mode or payload_mode(
        config.mode,
        image_url=config.image_url, image_path=config.image_path,
        video_url=config.video_url, video_path=config.video_path,
    )
    # 准入口径（与平台 2026-10-10 口径一致）：blocked 只记录不阻断；
    # 硬阻断 = failed/error，或整批没有任何有效结论（passed / warning）。
    effective = passed + warning
    admission = (
        len(rows) == requested
        and hard_failed == 0
        and effective >= 1
    )
    results = [
        {
            "case_id": f"{SEEDANCE_CASE_ID}#{row['index']}",
            "name": f"任务 {row['index']}",
            "status": row.get("verdict"),
            "duration_seconds": row.get("elapsed_seconds"),
            "details": f"{row.get('verdict_detail')}"
                       + (f"；视频：{row['video_url']}" if row.get("video_url") else ""),
        }
        for row in rows
    ]
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE_NAME,
        "test_kind": TEST_KIND,
        "case": SEEDANCE_CASE_ID,
        "record_id": config.record_id or None,
        "protocol": VIDEO_PROTOCOL,
        "base_url": config.base_url,
        "model": config.model,
        "endpoint": endpoint,
        "prompt": config.prompt,
        "mode": resolved_mode,
        "mode_label": mode_label(resolved_mode),
        "resolution": config.resolution,
        "ratio": config.ratio,
        "duration": int(config.duration),
        "generate_audio": bool(config.generate_audio),
        "watermark": bool(config.watermark),
        "image": image_evidence,
        "image_role": config.image_role if image_evidence.get("provided") else None,
        "video_input": video_input or {"provided": False},
        "video_role": config.video_role if (video_input or {}).get("sent_to_upstream") else None,
        # 参考视频没随任务提交时的唯一提示位（本地素材上传后放弃填 URL 的情形）。
        # 它不是判定项：视频素材缺失不影响"这次生成是否成功"的结论。
        "video_input_note": (video_input or {}).get("note"),
        "requested": requested,
        "sent": len(rows),
        "completed": len(rows),
        "successful": passed,
        "failed": hard_failed,
        "warning": warning,
        "blocked": blocked,
        "success_rate": passed / len(rows) if rows else 0.0,
        "concurrency": config.concurrency,
        "poll_interval_seconds": config.poll_interval_seconds,
        "task_timeout_seconds": config.task_timeout_seconds,
        "total_polls": sum(int(row.get("poll_count") or 0) for row in rows),
        "elapsed_p90_seconds": percentile(durations, 0.90),
        "elapsed_avg_seconds": (sum(durations) / len(durations)) if durations else None,
        "e2e_p90_seconds": percentile(durations, 0.90),
        "videos_dir": VIDEO_DIR_NAME if config.keep_videos else None,
        "videos_saved": video_totals["videos_saved"],
        "video_bytes_saved": video_totals["video_bytes_saved"],
        "videos_skipped": video_totals["videos_skipped"],
        "video_save_errors": sum(1 for row in rows if row.get("video_save_error")),
        "max_video_bytes": config.max_video_bytes if config.keep_videos else None,
        "video_links": [
            {
                "index": row["index"],
                "task_id": row.get("task_id"),
                "status": row.get("verdict"),
                "detail": row.get("verdict_detail"),
                "url": row.get("video_url"),
                "local_file": (row.get("video") or {}).get("file"),
                "local_viewable": bool(row.get("artifact_url")),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "submit_http_status": row.get("http_status"),
                "task_status": row.get("task_status"),
                "poll_count": row.get("poll_count"),
                # 失败取证：笼统的失败文案（如「请求参数校验失败」）必须配上机器可读的
                # code、上游回显的顶层键名和末次响应体原文，否则报告只说"失败了"，
                # 无法回答"到底哪个参数被拒"。
                "error_code": row.get("task_error_code"),
                "task_body_keys": row.get("task_body_keys") or [],
                "task_body_sample": row.get("task_body_sample"),
            }
            for row in rows
        ],
        "usage": _usage_totals(rows),
        "results": results,
        "checks": {
            "exact_request_count": {
                "value": len(rows), "expected": requested, "passed": len(rows) == requested,
            },
            "no_hard_failure": {
                "value": hard_failed, "expected": 0, "passed": hard_failed == 0,
                "detail": "failed/error 表示上游明确拒绝或 200 无产出，属硬阻断",
            },
            "effective_verdict_present": {
                "value": effective, "expected": ">= 1", "passed": effective >= 1,
                "detail": "整批只有 blocked 时没有有效结论，准入无依据",
            },
            "all_videos_produced": {
                "value": passed, "expected": requested, "passed": passed == requested,
                "detail": "全部任务都返回了视频链接才是完整达标；部分 blocked 仅记录",
            },
        },
        "admission_passed": admission,
        "scope_note": (
            "视频生成是异步任务：提交拿 task id 后轮询状态直到 succeeded/failed/expired。"
            "账号、权限、额度、限流、瞬时故障与「观察窗口内未达终态」记 blocked（平台前提不满足）；"
            "任务成功但无视频链接、上游明确拒绝记 failed。blocked 只记录不阻断准入，"
            "但整批只有 blocked 时视为没有有效结论、不准入。"
        ),
    }
    return summary


def _usage_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"completion_tokens": 0, "total_tokens": 0}
    for row in rows:
        usage = row.get("usage")
        if not isinstance(usage, dict):
            continue
        totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        totals["total_tokens"] += int(usage.get("total_tokens") or 0)
    return totals


def _artifact_link(record_id: str | None, name: str) -> str:
    return f"[本地文件](/api/runs/{record_id}/artifact/{name})" if record_id else f"`{name}`"


_STATUS_TEXT = {
    "passed": "✅ 成功",
    "failed": "❌ 失败",
    "error": "❌ 错误",
    "warning": "⚠️ 警告",
    "blocked": "⏸ 未验证",
}


def _video_input_line(summary: dict[str, Any]) -> str:
    """报告里的"输入参考视频"一行。

    本地素材（未提交给上游）与 URL（真的进了请求体）必须能一眼区分开——
    否则"我上传了视频"和"上游用到了这个视频"会被混为一谈。
    """
    video = summary.get("video_input") or {}
    if not video.get("provided"):
        if video.get("skipped_by_mode"):
            return "输入参考视频：无（提供了视频，但生成方式未启用视频，已忽略）"
        return "输入参考视频：无"
    local = video.get("local_asset") or {}
    archive = ""
    if local:
        archive = (f"，本地素材 {local.get('name')}（{local.get('bytes')} 字节，"
                   f"sha256 {str(local.get('sha256'))[:16]}…）")
    if video.get("sent_to_upstream"):
        if video.get("kind") == "url":
            return (f"输入参考视频：URL（{video.get('url')}），位置 {summary.get('video_role') or '未指定'}"
                    f"{archive}")
        return (f"输入参考视频：data URI（{video.get('bytes')} 字节，"
                f"sha256 {str(video.get('sha256'))[:16]}…），位置 {summary.get('video_role') or '未指定'}"
                f"{archive}")
    return (f"输入参考视频：本地素材未提交给上游（只归档留痕{archive}）"
            "——异步任务面只接受 URL，需要上游使用参考视频时请填写视频地址")


def write_report(result_dir: Path, summary: dict[str, Any]) -> None:
    """写出报告。

    报告的主产物是**生成视频的链接**：先给一张逐任务的结果表（含可点击链接），
    再单独列一节纯 URL 列表（便于直接复制/粘贴到别处），最后是判定口径与证据说明。
    报告只依赖 summary，保证与 summary.json 内容一致。
    """
    record_id = summary.get("record_id")
    links = sorted(summary.get("video_links") or [], key=lambda item: item.get("index") or 0)
    if summary.get("videos_dir"):
        videos_line = (
            f"视频保留：{summary['videos_saved']} 个 / "
            f"{summary['video_bytes_saved'] / 1024 / 1024:.1f} MiB，保存在 `{summary['videos_dir']}/`"
            f"（上限 {format_budget(summary['max_video_bytes'])}）"
        )
        if summary.get("videos_skipped"):
            videos_line += f"；已达上限跳过 {summary['videos_skipped']} 个"
    else:
        videos_line = "视频保留：本次未开启（--no-keep-videos）"
    image = summary.get("image") or {}
    if image.get("provided"):
        if image.get("kind") == "url":
            image_line = f"输入图片：URL（{image.get('url')}），位置 {summary.get('image_role')}"
        else:
            image_line = (
                f"输入图片：{image.get('kind')}（{image.get('bytes')} 字节，"
                f"sha256 {str(image.get('sha256'))[:16]}…），位置 {summary.get('image_role')}"
            )
    elif image.get("skipped_by_mode"):
        image_line = "输入图片：无（提供了图片，但生成方式未启用图片，已忽略）"
    else:
        image_line = "输入图片：无（纯文生视频）"
    video_line = _video_input_line(summary)
    mode = summary.get("mode")
    mode_line = f"生成方式：{summary.get('mode_label') or mode}（`{mode}`）"

    lines = [
        f"# {SEEDANCE_CASE_ID} Seedance 视频生成报告",
        "",
        *stats_lines(
            summary["requested"], summary["successful"], summary["failed"],
            summary.get("warning", 0), summary.get("blocked", 0), summary["success_rate"],
            [
                f"模型：`{summary['model']}`",
                f"协议面：`seedance`（POST/GET `{summary['endpoint']}`）",
                mode_line,
                f"分辨率：{summary['resolution']}；画面比例：{summary['ratio']}；"
                f"时长：{summary['duration']} 秒",
                f"原生音频：{'开' if summary['generate_audio'] else '关'}；"
                f"水印：{'开' if summary['watermark'] else '关'}",
                image_line,
                video_line,
                f"任务轮询：间隔 {summary['poll_interval_seconds']:g} 秒，"
                f"单任务观察窗口 {summary['task_timeout_seconds']:g} 秒；"
                f"本次累计轮询 {summary['total_polls']} 次",
                f"请求（视频任务）数：{summary['requested']}；并发：{summary['concurrency']}",
                "生成方式：先 `POST …/contents/generations/tasks` 拿 task id，"
                "再 `GET …/tasks/{task_id}` 轮询至终态；任务 id 官方仅保留 7 天。",
            ],
        ),
        "",
        "## 生成结果",
        "",
        "| # | 任务 ID | 状态 | 分辨率 | 比例 | 时长(s) | 耗时(s) | 轮询 | 提交 HTTP | 视频链接 | 本地文件 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in links:
        elapsed = row.get("elapsed_seconds")
        elapsed_text = f"{float(elapsed):.1f}" if isinstance(elapsed, (int, float)) else "—"
        link = f"[打开视频]({row['url']})" if row.get("url") else "—"
        # 本地归档列同样只在浏览器可播放时给链接，避免报告里出现点了 404 的地址。
        if not row.get("local_file"):
            local = "—"
        elif row.get("local_viewable"):
            local = _artifact_link(record_id, row["local_file"])
        else:
            local = f"`{row['local_file']}`（非浏览器可播放格式）"
        lines.append(
            f"| {row['index']} | {row.get('task_id') or '—'} | "
            f"{_STATUS_TEXT.get(str(row.get('status')), str(row.get('status')))} | "
            f"{summary['resolution']} | {summary['ratio']} | {summary['duration']} | "
            f"{elapsed_text} | {row.get('poll_count') or 0} | "
            f"{row.get('submit_http_status') or '—'} | {link} | {local} |"
        )

    produced = [row for row in links if row.get("url")]
    lines += ["", "## 视频链接", ""]
    if produced:
        for row in produced:
            lines.append(f"- 任务 {row['index']}（{_STATUS_TEXT.get(str(row.get('status')), row.get('status'))}）：{row['url']}")
    else:
        lines.append("- 本次没有生成出视频链接（详见下方明细与 `requests.jsonl`）。")

    lines += [
        "",
        "## 任务明细",
        "",
        "| # | 裁定 | 提交 HTTP | 任务状态 | 轮询次数 | 说明 |",
        "|---|---|---:|---|---:|---|",
    ]
    for row in links:
        lines.append(
            f"| {row['index']} | {_STATUS_TEXT.get(str(row.get('status')), row.get('status'))} | "
            f"{row.get('submit_http_status') or '—'} | {row.get('task_status') or '—'} | "
            f"{row.get('poll_count') or 0} | "
            f"{str(row.get('detail') or '—').replace('|', chr(92) + '|')[:240]} |"
        )
    lines += [
        "",
        f"> {videos_line}",
        "> 证据：`requests.jsonl`（每任务一行：提交结果 + 逐次轮询状态 + 最终裁定，含逐轮 status 时间线）、"
        "`generated_requests.jsonl`（执行前生成的请求体；内联图片已剥离为 sha256 摘要，base64 不进入证据）。",
        f"> {summary.get('scope_note')}",
    ]
    if summary.get("video_input_note"):
        lines.append(f"> ⚠️ {summary['video_input_note']}")
    skipped = [
        name for name, key in (("图片", "image"), ("参考视频", "video_input"))
        if (summary.get(key) or {}).get("skipped_by_mode")
    ]
    if skipped:
        lines.append(
            f"> ⚠️ 生成方式为「{summary.get('mode_label') or mode}」，"
            f"本次提供的 {'/'.join(skipped)} 已按生成方式忽略，未随任务提交给上游"
            "（入队时应当已拦截，出现此处说明是直接调用执行器）。"
        )
    # 上游原文取证：只在**未通过**的用例上出现（成功用例的响应体没有诊断价值），
    # 且必须真的留到了东西，避免正常批次里塞一堆空章节。
    diagnostics = [
        row for row in links
        if str(row.get("status")) != "passed" and (row.get("task_body_sample") or row.get("error_code"))
    ]
    if diagnostics:
        lines += [
            "",
            "## 未通过用例的上游原文",
            "",
            "> 异步任务面把参数校验放在执行阶段，失败时常见文案很笼统（如「请求参数校验失败」）。"
            "下表给出机器可读的 `error.code`、上游响应回显的顶层键名与末次响应体原文——"
            "键名清单可以直接回答「上游是否回显了 model / resolution / ratio / duration」。",
            "",
        ]
        for row in diagnostics:
            keys = row.get("task_body_keys") or []
            code = row.get("error_code") or "—"
            lines.append(f"- 任务 {row['index']}：error.code=`{code}`；"
                         f"响应顶层键={keys if keys else '（未取到）'}")
            if row.get("task_body_sample"):
                lines.append(f"  - 末次响应体：`{str(row['task_body_sample'])[:1200]}`")
    (result_dir / "report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
async def run_videos(
    config: VideoConfig,
    output_root: Path,
    *,
    result_dir: Path | None = None,
    progress: Callable[[int, int, int], None] | None = None,
) -> VideoOutcome:
    """执行一次视频生成批次（不含收尾；收尾由 ``common.finalizer`` 统一执行）。"""
    requested = max(1, int(config.requests))
    # 生成方式决定"这次带不带图片 / 参考视频"：声明 text 时即使配置里塞了素材也不解析、
    # 不提交（但留一条 skipped_by_mode 证据，免得"给了却被丢掉"变成无声行为）。
    mode = payload_mode(
        config.mode,
        image_url=config.image_url, image_path=config.image_path,
        video_url=config.video_url, video_path=config.video_path,
    )
    if mode_allows_image(mode):
        image_value, image_evidence = resolve_image(config.image_url, config.image_path)
    else:
        image_value = None
        image_evidence = {"provided": False}
        if config.image_url or config.image_path:
            image_evidence["skipped_by_mode"] = mode
    if mode_allows_video(mode):
        video_value, video_input = resolve_video(config.video_url, config.video_path)
    else:
        video_value = None
        video_input = {"provided": False}
        if config.video_url or config.video_path:
            video_input["skipped_by_mode"] = mode
    payload = build_payload(config, image_value, video_value)
    plan = build_request_plan(config, payload)
    for item in plan:
        item["image_evidence"] = image_evidence
        item["video_input_evidence"] = video_input
    directory = result_dir or output_root / (
        f"{SEEDANCE_CASE_ID}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    input_dir = directory / INPUT_DIR_NAME
    if image_evidence.get("provided") and image_evidence.get("kind") == "file":
        try:
            source = Path(config.image_path)
            input_dir.mkdir(parents=True, exist_ok=True)
            (input_dir / f"input-{source.name}").write_bytes(source.read_bytes())
            image_evidence["saved_as"] = f"{INPUT_DIR_NAME}/input-{source.name}"
        except OSError as exc:
            image_evidence["input_save_error"] = f"{type(exc).__name__}: {exc}"
    # 本地参考视频：**只归档**（不转 base64、不进请求体），报告里据此说明出处。
    # 文件前缀 input-video- 与图片区分，避免同名素材互相覆盖。
    # 归档条件是"给了本地文件"（local_asset 只在 video_path 非空时存在）——
    # 同时填了 URL 时 URL 负责提交、本地文件仍归档，两者互不排斥。
    local_asset = video_input.get("local_asset") or {}
    if local_asset:
        try:
            source = Path(config.video_path)
            input_dir.mkdir(parents=True, exist_ok=True)
            (input_dir / f"input-video-{source.name}").write_bytes(source.read_bytes())
            local_asset["saved_as"] = f"{INPUT_DIR_NAME}/input-video-{source.name}"
        except OSError as exc:
            local_asset["input_save_error"] = f"{type(exc).__name__}: {exc}"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE_NAME,
        "test_kind": TEST_KIND,
        "case": SEEDANCE_CASE_ID,
        "record_id": config.record_id or None,
        "protocol": VIDEO_PROTOCOL,
        "base_url": config.base_url,
        "endpoint": task_endpoint(config.base_url),
        "model": config.model,
        "mode": mode,
        "mode_label": mode_label(mode),
        # 界面/CLI 声明的档位；为 null 表示"未声明、按素材推导"。
        "mode_declared": config.mode or None,
        "prompt": config.prompt,
        "resolution": config.resolution,
        "ratio": config.ratio,
        "duration": int(config.duration),
        "generate_audio": bool(config.generate_audio),
        "watermark": bool(config.watermark),
        "image": image_evidence,
        "video_input": video_input,
        "requests": requested,
        "concurrency": config.concurrency,
        "poll_interval_seconds": config.poll_interval_seconds,
        "task_timeout_seconds": config.task_timeout_seconds,
        "request_timeout_seconds": config.request_timeout_seconds,
        "keep_videos": config.keep_videos,
        "videos_dir": VIDEO_DIR_NAME if config.keep_videos else None,
        "max_video_bytes": config.max_video_bytes if config.keep_videos else None,
        "spec_source": "火山方舟 / Seedance 视频生成任务接口（contents/generations/tasks）",
        "generation_phase": "complete_before_execution",
    }
    (directory / "metadata.json").write_text(
        json.dumps(redact(metadata), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (directory / "generated_requests.jsonl").write_text(
        "\n".join(json.dumps(redact(item), ensure_ascii=False, default=str) for item in plan) + "\n",
        encoding="utf-8",
    )
    video_dir = (directory / VIDEO_DIR_NAME) if config.keep_videos else None
    budget = ImageBudget(config.max_video_bytes) if config.keep_videos else None
    rows = await _run_requests(
        config=config,
        request_plan=plan,
        payload=payload,
        requests_path=directory / "requests.jsonl",
        video_dir=video_dir,
        budget=budget,
        record_id=config.record_id,
        progress=progress,
    )
    summary = build_summary(
        config, rows,
        endpoint=task_endpoint(config.base_url),
        image_evidence=image_evidence,
        requested=requested,
        video_input=video_input,
        mode=mode,
    )
    (directory / "summary.json").write_text(
        json.dumps(redact(summary), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    write_report(directory, summary)
    return VideoOutcome(result_dir=directory, rows=rows, summary=summary)


def keep_videos_default() -> bool:
    return env_flag(KEEP_VIDEOS_ENV, True)
