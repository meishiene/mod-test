"""Shared, auditable test execution framework.

Transport execution and case assertions are deliberately separate.  A transport
success (for example HTTP 200) is never treated as a case pass by itself.
"""
from __future__ import annotations
import asyncio, json, time, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Awaitable, Callable
import httpx
from .audit import redact

@dataclass
class Assertion:
    name: str
    passed: bool | None
    expected: Any = None
    actual: Any = None
    detail: str = ""
    severity: str = "error"

@dataclass
class ValidationResult:
    status: str  # passed, failed, not_evaluable, error
    passed: bool = False
    checks: list[Assertion] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

@dataclass
class ExecutionRecord:
    case_id: str
    request_id: str
    request: dict[str, Any]
    request_headers: dict[str, str]
    started_at: str
    started_epoch: float
    http_status: int | None = None
    response_headers: dict[str, str] = field(default_factory=dict)
    response_body: Any = None
    response_text: str = ""
    raw_response_body: str = ""
    raw_sse_lines: list[str] = field(default_factory=list)
    raw_sse_events: list[str] = field(default_factory=list)
    parsed_sse_events: list[dict[str, Any]] = field(default_factory=list)
    done_received: bool = False
    invalid_sse_events: int = 0
    finish_reasons: list[str] = field(default_factory=list)
    content: str = ""
    reasoning: str = ""
    usage: dict[str, Any] | None = None
    provider_request_id: str | None = None
    timing: dict[str, Any] = field(default_factory=dict)
    transport_status: str = "unknown"
    transport_error: dict[str, str] | None = None
    validation: ValidationResult | None = None

    def jsonable(self) -> dict[str, Any]:
        value = asdict(self)
        return value

class ResultLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = asyncio.Lock()
        self.fp = path.open("a", encoding="utf-8", buffering=1)
    async def write(self, record: ExecutionRecord) -> None:
        async with self._lock:
            self.fp.write(json.dumps(redact(record.jsonable()), ensure_ascii=False, default=str) + "\n")
            self.fp.flush()
    def close(self): self.fp.close()

class HttpExecutor:
    def __init__(self, *, timeout: float = 300, verify_ssl: bool = True, trust_env: bool = False, max_connections: int = 100, client: httpx.AsyncClient | None = None):
        self.timeout = httpx.Timeout(timeout, connect=min(timeout, 30))
        self.limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
        self.verify_ssl = verify_ssl
        self.trust_env = trust_env
        self.client: httpx.AsyncClient | None = client
        self._external_client = client is not None
    async def __aenter__(self):
        if self.client is not None: return self
        self.client = httpx.AsyncClient(timeout=self.timeout, limits=self.limits, verify=self.verify_ssl, trust_env=self.trust_env, headers={"Accept":"text/event-stream"})
        return self
    async def __aexit__(self, *_):
        if self.client and not self._external_client: await self.client.aclose()
    async def execute(self, *, case_id: str, url: str, payload: dict[str, Any], headers: dict[str, str], stream: bool) -> ExecutionRecord:
        assert self.client is not None
        rid = headers.get("X-Request-ID") or headers.get("x-request-id") or f"{case_id}-{uuid.uuid4().hex[:12]}"
        started_epoch, started = time.time(), time.perf_counter()
        rec = ExecutionRecord(case_id, rid, payload, {k:v for k,v in headers.items() if k.lower() != "authorization"}, time.strftime("%Y-%m-%dT%H:%M:%S%z"), started_epoch)
        first_sse = first_reasoning = first_content = None
        try:
            if not stream:
                response = await self.client.post(url, json=payload, headers=headers)
                rec.http_status, rec.response_headers = response.status_code, dict(response.headers)
                rec.timing["content_type"] = response.headers.get("content-type")
                rec.response_text = response.text
                rec.raw_response_body = response.content.decode("utf-8", "replace")
                try: rec.response_body = response.json()
                except (ValueError, json.JSONDecodeError): rec.response_body = None
                rec.response_body = rec.response_body
                rec.usage = (rec.response_body or {}).get("usage") if isinstance(rec.response_body, dict) else None
                rec.provider_request_id = self._provider_id(response.headers, rec.response_body)
            else:
                async with self.client.stream("POST", url, json=payload, headers=headers) as response:
                    rec.http_status, rec.response_headers = response.status_code, dict(response.headers)
                    rec.timing["content_type"] = response.headers.get("content-type")
                    rec.provider_request_id = self._provider_id(response.headers, None)
                    async for line in response.aiter_lines():
                        now = time.perf_counter()
                        rec.raw_sse_lines.append(line)
                        rec.raw_response_body += line + "\n"
                        if not line.startswith("data:"): continue
                        first_sse = first_sse or now
                        raw = line[5:].strip(); rec.raw_sse_events.append(raw)
                        if raw == "[DONE]": rec.done_received = True; continue
                        try: obj = json.loads(raw)
                        except json.JSONDecodeError: rec.invalid_sse_events += 1; continue
                        obj["_received_ms"] = (now - started) * 1000
                        rec.parsed_sse_events.append(obj)
                        if rec.provider_request_id is None: rec.provider_request_id = obj.get("id")
                        if obj.get("type") in {"message_stop", "response.completed"}: rec.done_received = True
                        choices = obj.get("choices") or []
                        if choices:
                            choice = choices[0] or {}; delta = choice.get("delta") or choice.get("message") or {}
                            content = delta.get("content") or ""; reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                            if isinstance(content, list): content = "".join(str(x.get("text", x)) if isinstance(x,dict) else str(x) for x in content)
                            if isinstance(reasoning, list): reasoning = "".join(str(x.get("text", x)) if isinstance(x,dict) else str(x) for x in reasoning)
                            if content: first_content = first_content or now; rec.content += str(content)
                            if reasoning: first_reasoning = first_reasoning or now; rec.reasoning += str(reasoning)
                            if choice.get("finish_reason"): rec.finish_reasons.append(choice["finish_reason"])
                        delta = obj.get("delta") or {}
                        if obj.get("type") == "content_block_delta":
                            content = delta.get("text") or ""
                            reasoning = delta.get("thinking") or delta.get("reasoning_content") or ""
                            if content: first_content = first_content or now; rec.content += str(content)
                            if reasoning: first_reasoning = first_reasoning or now; rec.reasoning += str(reasoning)
                        if obj.get("type") == "message_delta" and delta.get("stop_reason"): rec.finish_reasons.append(str(delta["stop_reason"]))
                        if obj.get("usage"): rec.usage = obj["usage"]
                        if isinstance(obj.get("message"), dict) and obj["message"].get("usage"): rec.usage = obj["message"]["usage"]
                    if rec.http_status and rec.http_status >= 300:
                        rec.response_text = "".join(rec.raw_sse_lines)
                        rec.raw_response_body = rec.response_text
            rec.transport_status = "completed"
        except Exception as exc:
            rec.transport_status = "error"
            rec.transport_error = {"type": type(exc).__name__, "message": str(exc)}
        ended = time.perf_counter()
        rec.timing = {"first_sse_ms": (first_sse-started)*1000 if first_sse else None,
                      "first_reasoning_ms": (first_reasoning-started)*1000 if first_reasoning else None,
                      "ttft_ms": ((min(x for x in (first_reasoning, first_content) if x is not None)-started)*1000) if first_reasoning or first_content else None,
                      "ttfc_ms": (first_content-started)*1000 if first_content else None,
                      "e2e_ms": (ended-started)*1000}
        return rec
    async def execute_raw(self, *, case_id: str, method: str, url: str, headers: dict[str, str],
                          params: dict[str, Any] | None = None, json_body: dict[str, Any] | None = None,
                          files: dict[str, Any] | None = None, data: dict[str, Any] | None = None,
                          raw_body: str | bytes | None = None) -> ExecutionRecord:
        """非 JSON-POST 通道（GET / DELETE / multipart 上传 / 原始请求体）。

        与 execute 共享同一套 ExecutionRecord 与证据写入逻辑，便于统一审计：
        GET（模型列表、余额）与 multipart（文件上传）无法走 execute 的 json=payload 路径。
        ``raw_body`` 逐字节发送原始请求体，用于构造**故意解析失败**的请求（如畸形 JSON），
        这类用例无法用 ``json=`` 表达（json.dumps 永远产出合法 JSON）。
        """
        assert self.client is not None
        rid = headers.get("X-Request-ID") or headers.get("x-request-id") or f"{case_id}-{uuid.uuid4().hex[:12]}"
        started_epoch, started = time.time(), time.perf_counter()
        payload: dict[str, Any] = json_body if json_body is not None else {"_form": data or {}, "_files": sorted((files or {}).keys())}
        send_headers = dict(headers)
        if files is not None:
            # httpx 会从 Content-Type 头里**回读 boundary** 来编码 multipart body。
            # 若这里沿用客户端默认头（如 runner 建 client 时设的 application/json），
            # 结果就是 body 用了随机 boundary、而头仍写 application/json（无 boundary），
            # 上游（DeepSeek Files API）直接 400 `Invalid boundary for multipart/form-data request`。
            # 因此显式剔除内容类型并写入自生成的 boundary，保证头与 body 一致。
            send_headers = {k: v for k, v in send_headers.items() if k.lower() != "content-type"}
            send_headers["Content-Type"] = f"multipart/form-data; boundary={uuid.uuid4().hex}"
        rec = ExecutionRecord(case_id, rid, payload, {k:v for k,v in send_headers.items() if k.lower() != "authorization"}, time.strftime("%Y-%m-%dT%H:%M:%S%z"), started_epoch)
        rec.request["_method"] = method.upper()
        rec.request["_url"] = url
        if raw_body is not None:
            rec.request["_raw_body"] = raw_body if isinstance(raw_body, str) else raw_body.decode("utf-8", "replace")
        if params:
            rec.request["_params"] = dict(params)
        try:
            send_kwargs: dict[str, Any] = {"params": params, "files": files, "headers": send_headers}
            if raw_body is not None:
                send_kwargs["content"] = raw_body
            else:
                send_kwargs["json"] = json_body
                send_kwargs["data"] = data
            response = await self.client.request(method, url, **send_kwargs)
            rec.http_status, rec.response_headers = response.status_code, dict(response.headers)
            rec.timing["content_type"] = response.headers.get("content-type")
            rec.response_text = response.text
            rec.raw_response_body = response.content.decode("utf-8", "replace")
            try: rec.response_body = response.json()
            except (ValueError, json.JSONDecodeError): rec.response_body = None
            rec.usage = (rec.response_body or {}).get("usage") if isinstance(rec.response_body, dict) else None
            rec.provider_request_id = self._provider_id(response.headers, rec.response_body)
            rec.transport_status = "completed"
        except Exception as exc:
            rec.transport_status = "error"
            rec.transport_error = {"type": type(exc).__name__, "message": str(exc)}
        rec.timing["e2e_ms"] = (time.perf_counter() - started) * 1000
        return rec

    @staticmethod
    def _provider_id(headers, body):
        return headers.get("x-request-id") or headers.get("request-id") or headers.get("x-trace-id") or headers.get("x-oneapi-request-id") or ((body or {}).get("id") if isinstance(body,dict) else None)

Validator = Callable[[ExecutionRecord], Assertion | list[Assertion]]

def check(name, passed, expected=None, actual=None, detail="", severity="error"):
    return Assertion(name, None if passed is None else bool(passed), expected, actual, detail, severity)

def validate_case(record: ExecutionRecord, assertions: list[Validator]) -> ValidationResult:
    checks=[]
    try:
        for validator in assertions:
            value=validator(record); checks.extend(value if isinstance(value,list) else [value])
    except Exception as exc:
        return ValidationResult("error", False, checks, [f"validator_error: {type(exc).__name__}: {exc}"])
    unknown=[x for x in checks if x.passed is None]
    failed=[x for x in checks if x.passed is False and x.severity == "error"]
    if failed: return ValidationResult("failed", False, checks, [x.detail or x.name for x in failed])
    if unknown: return ValidationResult("not_evaluable", False, checks, [x.detail or x.name for x in unknown])
    return ValidationResult("passed", True, checks, [])

def expect_status(*statuses: int) -> Validator:
    return lambda r: check("http_status", r.http_status in statuses, list(statuses), r.http_status, f"expected HTTP {statuses}, got {r.http_status}")
def expect_success_status() -> Validator: return expect_status(200)
def expect_error(*, types=(), message_contains=()) -> Validator:
    def v(r):
        body=r.response_body if isinstance(r.response_body,dict) else {}
        err=body.get("error") if isinstance(body,dict) else {}
        actual_type=err.get("type") if isinstance(err,dict) else None; msg=str(err.get("message", "") if isinstance(err,dict) else r.response_text)
        checks=[check("error_type", not types or actual_type in types, list(types), actual_type, f"error type mismatch: {actual_type}")]
        checks.append(check("error_message", not message_contains or all(x.lower() in msg.lower() for x in message_contains), list(message_contains), msg[:500], "error message semantic check failed"))
        return checks
    return v
def expect_content(*terms: str, exact: str | None = None) -> Validator:
    def v(r):
        text=r.content or (((r.response_body or {}).get("choices") or [{}])[0].get("message") or {}).get("content", "") if isinstance(r.response_body,dict) else r.content
        ok=(exact is None or text == exact) and all(t in text for t in terms)
        return check("content", ok, exact or list(terms), text[:500], "content assertion failed")
    return v
def expect_usage(*fields: str) -> Validator:
    return lambda r: check("usage_fields", r.usage is not None and all(_nested_get(r.usage,f) is not None for f in fields), list(fields), r.usage, "usage incomplete")
def expect_sse() -> Validator:
    return lambda r: [check("sse_done", r.done_received, True, r.done_received, "missing [DONE]"), check("sse_json", r.invalid_sse_events == 0, 0, r.invalid_sse_events, "invalid SSE events"), check("sse_content", bool(r.content), True, bool(r.content), "no visible content")]
def expect_json_content() -> Validator:
    def v(r):
        text=(((r.response_body or {}).get("choices") or [{}])[0].get("message") or {}).get("content", "") if isinstance(r.response_body,dict) else r.content
        try: value=json.loads(text); ok=isinstance(value,dict)
        except (ValueError,TypeError): ok=False; value=None
        return check("json_content",ok,"JSON object",text[:500],"content is not valid JSON object")
    return v

def _nested_get(obj, path):
    for p in path.split('.'):
        if not isinstance(obj,dict): return None
        obj=obj.get(p)
    return obj
