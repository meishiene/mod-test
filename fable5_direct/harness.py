"""Fable 5 直连实测的执行骨架（独立脚本，不依赖平台 runner / Web / test_framework）。

只用 httpx 直接打被测中转的 Anthropic Messages 端点，逐请求落盘证据。
判定口径沿用项目既有约定：
  passed / failed / warning / blocked 四档；负向断言要求 4xx 且非 404，且错误文案必须
  指向被验证的字段（防"凭状态码自证"）；瞬时故障（5xx / 连接错误）重试耗尽记 blocked。
"""
from __future__ import annotations

import asyncio
import base64
import json
import struct
import time
import uuid
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
BASE_URL = "https://i-token.ai/v1"
MODEL = "claude-fable-5"

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
STOP_REASONS = {"end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal", "pause_turn"}
MODEL_TOKENS = ("claude-fable-5", "fable-5", "fable5")

BETA_TASK_BUDGET = "task-budgets-2026-03-13"
BETA_CONTEXT_MANAGEMENT = "context-management-2025-06-27"
BETA_SERVER_FALLBACK = "server-side-fallback-2026-06-01"

MIN_CACHEABLE_PREFIX_TOKENS = 2048
LONG_CONTEXT_CHARS = 64000
LONG_CONTEXT_MIN_INPUT_TOKENS = 4000
INPUT_TOKEN_INFLATION_THRESHOLD = 2000

TRANSIENT_MIN_STATUS = 500
MAX_RETRIES = 2

_MESSAGE_SCHEMA_FIELDS = frozenset({
    "id", "type", "role", "model", "content", "stop_reason", "stop_sequence", "stop_details",
    "usage", "container", "context_management", "diagnostics",
})
_RELAY_FIELD_MARKERS = (
    "billing_usage", "access_programs", "claude_cache_creation", "new_api_error",
    "system_fingerprint", "prompt_tokens_details", "completion_tokens_details",
    "usage_semantic", "usage_source",
)
_MODEL_NOT_FOUND_MARKERS = (
    "model_not_found", "no available channel for model", "does not exist", "unknown model",
    "invalid model", "model not found", "not_found_error", "无可用渠道",
)
_ACCOUNT_LIMITATION_MARKERS = (
    "not available for your account", "your account is not enabled",
    "not enabled for this organization", "organization does not have access",
    "requires an api-key account", "zero data retention", "zdr agreement", "covered model",
    "data retention policy", "not available on your plan", "usage credits", "无权限", "未开通",
)


def rng(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8].upper()}"


# ---------------------------------------------------------------------------
# 纯标准库 PNG 夹具：固定"上红下蓝"，识别用例可复现
# ---------------------------------------------------------------------------
def png_bands(top=(220, 20, 20), bottom=(20, 20, 220), width: int = 512, height: int = 512) -> bytes:
    raw = b""
    for y in range(height):
        rgb = top if y < height // 2 else bottom
        raw += b"\x00" + bytes(rgb) * width
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def image_block(width: int = 512, height: int = 512) -> dict[str, Any]:
    data = base64.b64encode(png_bands(width=width, height=height)).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def filler(chars: int) -> str:
    sentence = "The quick brown fox jumps over the lazy dog. "
    return (sentence * (chars // len(sentence) + 1))[:chars]


# ---------------------------------------------------------------------------
# 响应解析
# ---------------------------------------------------------------------------
def blocks(body: dict | None) -> list[dict]:
    content = (body or {}).get("content")
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def text_of(body: dict | None) -> str:
    return "".join(str(b.get("text", "")) for b in blocks(body) if b.get("type") == "text")


def thinking_blocks(body: dict | None) -> list[dict]:
    return [b for b in blocks(body) if b.get("type") in {"thinking", "redacted_thinking"}]


def thinking_text(body: dict | None) -> str:
    return "".join(str(b.get("thinking", "")) for b in blocks(body) if b.get("type") == "thinking")


def tool_uses(body: dict | None) -> list[dict]:
    return [b for b in blocks(body) if b.get("type") == "tool_use"]


def usage_of(body: dict | None) -> dict:
    u = (body or {}).get("usage")
    return u if isinstance(u, dict) else {}


def num(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def stop_reason_of(body: dict | None) -> str | None:
    v = (body or {}).get("stop_reason")
    return str(v) if v is not None else None


def error_body(body: dict | None) -> dict | None:
    e = (body or {}).get("error")
    return e if isinstance(e, dict) else None


def error_text(body: dict | None) -> str:
    e = error_body(body)
    if e:
        return str(e.get("message") or e)
    if isinstance(body, dict):
        return json.dumps(body, ensure_ascii=False)[:800]
    return ""


def error_type(body: dict | None) -> str | None:
    e = error_body(body)
    v = e.get("type") if e else None
    return str(v) if v is not None else None


def error_signal(body: dict | None) -> str:
    e = error_body(body) or {}
    parts = [error_type(body) or "", str(e.get("code") or ""), error_text(body)]
    return " ".join(p for p in parts if p).lower()


def is_rejection(status: int) -> bool:
    """文档承诺"应被拒绝"：4xx 且非 404（404 是链路/端点不存在，不能冒充参数校验）。"""
    return 400 <= status < 500 and status != 404


def mentions(body: dict | None, *terms: str) -> bool:
    low = error_text(body).lower()
    return any(t.lower() in low for t in terms)


def rejected_because(status: int, body: dict | None, *terms: str) -> tuple[bool, str]:
    if not is_rejection(status):
        return False, ""
    if mentions(body, *terms):
        return True, ""
    return False, (f"HTTP {status} 被拒但文案未指向目标字段（期望含 {list(terms)} 之一）："
                   f"{error_text(body)[:200] or '(空响应体)'}")


def account_limitation(status: int, body: dict | None, feature: str) -> str | None:
    if not (400 <= status < 500):
        return None
    if mentions(body, *_ACCOUNT_LIMITATION_MARKERS):
        return (f"{feature} 被上游以账号/通道权限为由拒绝（HTTP {status}）：{error_text(body)[:200]}；"
                f"目标特性未触及，按平台前提不满足记 blocked")
    return None


_QUOTA_MARKERS = ("quota is not enough", "insufficient quota", "余额不足", "欠费", "额度不足",
                  "token remain quota", "insufficient_user_quota")


def quota_exhausted_reason(status: int, body: dict | None, feature: str) -> str | None:
    """中转账号预扣额度不足 → 请求在进入模型前就被拒，属性属平台前提不满足。

    该站按 ``max_tokens × 输出单价`` 预扣：``need quota`` 即预留额，与实际生成量无关。
    这类 403 不对应任何模型能力结论，必须记 blocked，不能记 failed。
    """
    if status != 403:
        return None
    if mentions(body, *_QUOTA_MARKERS):
        return (f"{feature} 因中转账号可用额度不足被拒（HTTP 403）：{error_text(body)[:200]}；"
                f"请求未进入模型，属平台前提不满足，判 blocked（充值后重跑方可取证）")
    return None


def model_not_found_evidence(requested: str, status: int, body: dict | None) -> str | None:
    low = error_signal(body)
    hits = [m for m in _MODEL_NOT_FOUND_MARKERS if m in low]
    if not hits:
        return None
    if ("model_not_found" not in low and "no available channel for model" in low
            and requested.lower() not in low):
        return None
    return f"结构化证据 {hits}（HTTP {status}）"


def relay_fields(body: dict | None) -> list[str]:
    if not isinstance(body, dict):
        return []
    return sorted({k for k in body if k in _RELAY_FIELD_MARKERS or any(m in k for m in _RELAY_FIELD_MARKERS)})


def foreign_protocol_evidence(body: dict | None) -> str | None:
    if not isinstance(body, dict):
        return None
    kind = str(body.get("object") or "")
    if "choices" in body:
        return "响应体出现 `choices`（Chat Completions 结构混入 Messages 响应）"
    if kind in {"chat.completion", "chat.completion.chunk", "response"}:
        return f"响应体 object={kind!r}（OpenAI 结构混入 Messages 响应）"
    if body.get("type") == "response.created":
        return "响应体出现 Responses API 事件结构"
    return None


def injected_system_reason(body: dict | None) -> str | None:
    system = (body or {}).get("system")
    if isinstance(system, str) and system.strip():
        return f"响应回显了非空 system（未在请求中提供）：{system[:200]!r}"
    if isinstance(system, list) and system:
        return f"响应回显了非空 system（未在请求中提供）：{json.dumps(system, ensure_ascii=False)[:200]}"
    return None


# ---------------------------------------------------------------------------
# 请求层
# ---------------------------------------------------------------------------
@dataclass
class Resp:
    status: int
    body: dict | None
    raw: str
    headers: dict[str, str]
    elapsed: float
    stream: dict[str, Any] | None = None
    transient: dict[str, Any] | None = None


@dataclass
class CaseResult:
    case_id: str
    group: str
    title: str
    passed: bool
    status: str
    details: str
    elapsed: float = 0.0
    observations: dict[str, Any] = field(default_factory=dict)


class Probe:
    def __init__(self, api_key: str, base_url: str = BASE_URL, timeout: float = 300.0, out_dir: Path | None = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.out_dir = out_dir
        self.client: httpx.AsyncClient | None = None
        self.log = None
        self.case_id = "FABLE5-DIRECT"
        self.transient_events: list[dict] = []
        self.quota_events: list[dict] = []

    def headers(self, extra: dict[str, str] | None = None, beta: str | None = None) -> dict[str, str]:
        h = {
            "x-api-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "X-Request-ID": f"FABLE5-DIRECT-{uuid.uuid4().hex[:16]}",
        }
        if beta:
            h["anthropic-beta"] = beta
        if extra:
            h.update(extra)
        return h

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=45.0), trust_env=True)
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.log = (self.out_dir / "requests.jsonl").open("a", encoding="utf-8")
        return self

    async def __aexit__(self, *exc):
        if self.log:
            self.log.close()
        if self.client:
            await self.client.aclose()

    def _record(self, entry: dict) -> None:
        if self.log:
            self.log.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.log.flush()

    async def call(self, path: str, payload: dict[str, Any] | None, *, method: str = "POST",
                   stream: bool = False, extra_headers: dict[str, str] | None = None,
                   beta: str | None = None, raw_body: str | None = None) -> Resp:
        assert self.client is not None
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = self.headers(extra_headers, beta)
        attempts = 0
        started = time.perf_counter()
        status, body, raw, hdrs, stream_info, transient = 0, None, "", {}, None, None
        while True:
            attempts += 1
            started = time.perf_counter()
            try:
                if stream:
                    status, body, raw, hdrs, stream_info = await self._stream(url, headers, payload)
                else:
                    resp = await self.client.request(
                        method, url, headers=headers,
                        content=raw_body if raw_body is not None else json.dumps(payload, ensure_ascii=False).encode(),
                    )
                    status = resp.status_code
                    hdrs = {k.lower(): v for k, v in resp.headers.items()}
                    raw = resp.text
                    try:
                        body = resp.json()
                    except Exception:
                        body = None
            except Exception as exc:  # 传输层错误
                status, body, raw, hdrs = 0, None, f"{type(exc).__name__}: {exc}", {}
            elapsed = time.perf_counter() - started
            definitive = bool(model_not_found_evidence("", status, body))
            transient = None
            if (status == 0 or status >= TRANSIENT_MIN_STATUS) and not definitive:
                transient = {"http_status": status, "message": (error_text(body) or raw)[:300], "attempts": attempts}
            else:
                break
            if attempts > MAX_RETRIES:
                break
            await asyncio.sleep(1.0 * (2 ** (attempts - 1)))
        if transient:
            self.transient_events.append({**transient, "path": path})
        if quota_exhausted_reason(status, body, path):
            self.quota_events.append({"path": path, "http_status": status, "message": error_text(body)[:300]})
        self._record({
            "case_id": self.case_id, "path": path, "method": method, "attempts": attempts,
            "request": payload, "request_raw": raw_body,
            "status": status, "elapsed": round(elapsed, 3), "response_body": body,
            "response_text": None if body is not None else raw[:2000],
            "response_headers": {k: hdrs.get(k) for k in
                                 ("x-request-id", "request-id", "cf-ray", "server", "content-type") if hdrs.get(k)},
            "stream_summary": (stream_info or {}).get("summary") if stream_info else None,
            "transient": transient,
        })
        return Resp(status, body, raw, hdrs, elapsed, stream_info, transient)

    async def _stream(self, url: str, headers: dict[str, str], payload: dict | None):
        assert self.client is not None
        lines: list[str] = []
        ttft = None
        status = 0
        hdrs: dict[str, str] = {}
        async with self.client.stream("POST", url, headers=headers,
                                      content=json.dumps(payload, ensure_ascii=False).encode()) as resp:
            status = resp.status_code
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            if status != 200:
                body = await resp.aread()
                raw = body.decode("utf-8", "replace")
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = None
                return status, parsed, raw, hdrs, None
            t0 = time.perf_counter()
            async for line in resp.aiter_lines():
                lines.append(line)
                if ttft is None and line.strip().startswith("data:"):
                    ttft = time.perf_counter() - t0
        events: list[dict] = []
        invalid = 0
        done = False
        for line in lines:
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                done = True
                events.append({"_done": True})
                continue
            try:
                events.append(json.loads(data))
            except Exception:
                invalid += 1
        text_parts, reasoning_parts = [], []
        event_types, delta_types, stop_reasons = [], [], []
        merged_usage: dict = {}
        delta_usage = None
        for ev in events:
            if not isinstance(ev, dict):
                continue
            if ev.get("type"):
                event_types.append(str(ev["type"]))
            d = ev.get("delta")
            if isinstance(d, dict) and d.get("type"):
                delta_types.append(str(d["type"]))
                if d["type"] == "text_delta":
                    text_parts.append(str(d.get("text") or ""))
                if d["type"] == "thinking_delta":
                    reasoning_parts.append(str(d.get("thinking") or ""))
            if isinstance(d, dict) and d.get("stop_reason"):
                stop_reasons.append(str(d["stop_reason"]))
            u = ev.get("usage")
            if not isinstance(u, dict):
                msg = ev.get("message")
                u = msg.get("usage") if isinstance(msg, dict) else None
            if isinstance(u, dict) and u:
                merged_usage.update(u)
                if ev.get("type") == "message_delta":
                    delta_usage = u
            if ev.get("type") == "message_start":
                m = ev.get("message")
                if isinstance(m, dict) and m.get("stop_reason"):
                    stop_reasons.append(str(m["stop_reason"]))
        info = {
            "text": "".join(text_parts), "reasoning": "".join(reasoning_parts),
            "event_types": event_types, "delta_types": delta_types, "usage": merged_usage,
            "delta_usage": delta_usage, "stop_reasons": stop_reasons, "invalid_events": invalid,
            "done": done, "ttft": ttft, "raw_line_count": len(lines),
            "summary": {"events": len(events), "delta_types": sorted(set(delta_types)),
                        "done_marker": done, "invalid": invalid,
                        "usage": {k: merged_usage.get(k) for k in ("input_tokens", "output_tokens")}},
        }
        return status, None, "\n".join(lines), hdrs, info

    async def messages(self, model: str, messages: list[dict], **kwargs: Any) -> Resp:
        max_tokens = int(kwargs.pop("max_tokens", 512))
        beta = kwargs.pop("_beta", None)
        extra = kwargs.pop("extra_headers", None)
        path = kwargs.pop("_path", "messages")
        payload = {"model": model, "max_tokens": max_tokens, "messages": messages, **kwargs}
        return await self.call(path, payload, stream=bool(payload.get("stream")), extra_headers=extra, beta=beta)

    # ------------------------------------------------------------------
    @staticmethod
    def tool(*, strict: bool = False, name: str = "get_order_status") -> dict[str, Any]:
        t: dict[str, Any] = {
            "name": name, "description": "查询订单状态",
            "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}},
                             "required": ["order_id"], "additionalProperties": False},
        }
        if strict:
            t["strict"] = True
        return t


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------
CaseFn = Callable[[Probe, str], Awaitable[tuple[bool, str, dict, str] | tuple[bool, str, dict]]]


async def run_case(probe: Probe, case_id: str, group: str, title: str, fn: CaseFn, model: str) -> CaseResult:
    probe.case_id = case_id
    probe.transient_events = []
    probe.quota_events = []
    started = time.perf_counter()
    try:
        out = await fn(probe, model)
        passed, details, obs = out[0], out[1], out[2]
        status = out[3] if len(out) > 3 else ("passed" if passed else "failed")
    except Exception as exc:
        passed, details, obs, status = False, f"{type(exc).__name__}: {exc}", {}, "error"
    if probe.quota_events and status in ("failed", "error", "warning"):
        ev = probe.quota_events[0]
        status = "blocked"
        passed = False
        details = (f"中转账号预扣额度不足，请求在进入模型前即被拒（HTTP 403）：{ev.get('message')}；"
                   f"平台前提不满足，判 blocked（充值后重跑方可取证）")
        obs = {**obs, "quota_failures": list(probe.quota_events)}
    if status in ("failed", "error") and probe.transient_events:
        ev = probe.transient_events[0]
        status = "blocked"
        passed = False
        details = (f"上游瞬时故障（HTTP {ev.get('http_status') or '连接错误'}，已重试 {ev['attempts'] - 1} 次仍失败）："
                   f"{ev.get('message') or '无响应体'}；不由被测模型决定，按 blocked 收口。")
        obs = {**obs, "transient_failures": list(probe.transient_events)}
    return CaseResult(case_id, group, title, passed, status, details,
                      round(time.perf_counter() - started, 2), obs)


def summarize(results: list[CaseResult], meta: dict) -> dict:
    counts = {k: sum(r.status == k for r in results) for k in ("passed", "failed", "warning", "blocked", "error")}
    hard = counts["failed"] + counts["error"]
    evaluated = counts["passed"] + hard
    by_group: dict[str, dict] = {}
    for r in results:
        g = by_group.setdefault(r.group, {"total": 0, "passed": 0, "failed": 0, "warning": 0, "blocked": 0, "error": 0})
        g["total"] += 1
        g[r.status] = g.get(r.status, 0) + 1
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target": meta,
        "total": len(results), "passed": counts["passed"], "failed": hard,
        "warning": counts["warning"], "blocked": counts["blocked"], "error": counts["error"],
        "pass_rate": counts["passed"] / evaluated if evaluated else 0.0,
        # blocked（平台前提不满足）只记录、不阻断准入（2026-10-10 口径，与 common/finalizer 一致）；
        # 但整批无有效结论（无 passed/warning）时仍不准入。
        "admission_passed": hard == 0 and any(r.status in {"passed", "warning"} for r in results),
        "by_group": by_group,
        "results": [asdict(r) for r in results],
    }
