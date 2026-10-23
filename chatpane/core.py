"""对话窗口执行核心：发一次自由对话，把完整响应原样落盘。

设计要点（与平台既有约束对齐）：

* **证据链完整**：这是平台的核心约束。因此对话不直连、不旁路，而是走与其它套件
  同一条流水线——结果目录里必须有非空 ``requests.jsonl`` 与 ``summary.json``，
  最后交给 ``common.finalizer.finalize_result_dir`` 收尾。
* **不重复实现 SSE 解析**：四类线格式（OpenAI Chat Completions / OpenAI Responses /
  Anthropic Messages / Gemini GenerateContent）的逐事件解析统一复用
  ``volcperf.client.parse_event``，避免出现第二套尺子。
* **判定只看"这次对话是否拿到正文"**：账号/权限/额度/瞬时故障按 ``blocked``
  记录（平台前提不满足，不等于模型有问题），其余 4xx 与空正文按 ``failed``/``warning``
  记录；能力结论由各专项套件负责，这里不做能力判定。
"""
from __future__ import annotations

import contextlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from common.audit import AuditLog, redact
from volcperf.client import parse_event
from volcperf.config import trust_env_enabled
from volcperf.protocols import endpoint as protocol_endpoint
from volcperf.protocols import headers as protocol_headers
from volcperf.protocols import protocol_spec

from .cases import CASES, CHAT_PROMPT, CHAT_PROTOCOLS, DEFAULT_MAX_TOKENS

# 这些状态码表示"平台前提不满足/账号侧拒绝"，按 blocked 记录而不是判定成模型问题。
_BLOCKED_STATUSES = {401, 402, 403, 404, 429}
_RESPONSE_FILE = "response.md"
_DEFAULT_TIMEOUT_SECONDS = 300.0
_DEFAULT_MAX_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 1.0


@dataclass(frozen=True)
class ChatConfig:
    api_key: str
    base_url: str
    model: str
    protocol: str
    prompt: str
    system: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float | None = None
    request_timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    verify_ssl: bool = True
    trust_env: bool = False
    include_usage: bool = True
    max_retries: int = _DEFAULT_MAX_RETRIES
    retry_backoff_seconds: float = _RETRY_BACKOFF_SECONDS
    record_id: str = ""

    @property
    def endpoint(self) -> str:
        return protocol_endpoint(self.protocol, self.base_url, self.model)

    @property
    def messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.system.strip():
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.prompt})
        return messages


def build_payload(config: ChatConfig) -> dict[str, Any]:
    """按协议家族构造单轮对话请求体。

    ``volcperf.protocols.payload`` 会强插一句压测用系统提示词，对话窗口不能用它，
    否则用户输入的系统提示词会被顶掉。
    """
    protocol = config.protocol
    if protocol not in CHAT_PROTOCOLS:
        raise ValueError(f"对话窗口不支持协议: {protocol}")
    family = protocol_spec(protocol).family
    messages = config.messages
    if protocol == "openai_responses":
        system = [m["content"] for m in messages if m["role"] == "system"]
        body: dict[str, Any] = {
            "model": config.model,
            "stream": True,
            "max_output_tokens": config.max_tokens,
            "input": [m for m in messages if m["role"] != "system"] or [{"role": "user", "content": ""}],
        }
        if system:
            body["instructions"] = "\n\n".join(system)
        if config.temperature is not None:
            body["temperature"] = config.temperature
        return body
    if family == "openai":
        body = {"model": config.model, "stream": True, "max_tokens": config.max_tokens, "messages": messages}
        if config.temperature is not None:
            body["temperature"] = config.temperature
        if config.include_usage:
            body["stream_options"] = {"include_usage": True}
        return body
    if family == "anthropic":
        system = [m["content"] for m in messages if m["role"] == "system"]
        body = {
            "model": config.model,
            "stream": True,
            "max_tokens": config.max_tokens,
            "messages": [m for m in messages if m["role"] != "system"] or [{"role": "user", "content": ""}],
        }
        if system:
            body["system"] = "\n\n".join(system)
        if config.temperature is not None:
            body["temperature"] = config.temperature
        return body
    if family == "gemini":
        system = [m["content"] for m in messages if m["role"] == "system"]
        role_map = {"assistant": "model", "user": "user"}
        contents = [
            {"role": role_map.get(m["role"], "user"), "parts": [{"text": m["content"]}]}
            for m in messages
            if m["role"] != "system"
        ] or [{"role": "user", "parts": [{"text": ""}]}]
        generation_config: dict[str, Any] = {"maxOutputTokens": config.max_tokens}
        if config.temperature is not None:
            generation_config["temperature"] = config.temperature
        body = {"contents": contents, "generationConfig": generation_config}
        if system:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
        return body
    raise ValueError(f"对话窗口不支持协议: {protocol}")


def _safe_response_headers(response: httpx.Response) -> dict[str, str]:
    keep = {
        "content-type", "server", "via", "x-request-id", "request-id", "x-trace-id",
        "x-oneapi-request-id", "x-tt-logid", "anthropic-organization-id", "cf-ray",
        "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",
    }
    return {key.lower(): value for key, value in response.headers.items() if key.lower() in keep}


class ChatStreamError(RuntimeError):
    """上游在 SSE 流内返回了 error 事件。"""


def _consume_stream(response: httpx.Response, protocol: str, request_id: str) -> dict[str, Any]:
    """把 SSE 流累积成完整响应。返回文本、思考、usage 与时间点。"""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    prompt_tokens = completion_tokens = reasoning_tokens = total_tokens = None
    finish_reason: str | None = None
    first_reasoning = first_content = None
    event_count = 0
    invalid_events = 0
    saw_done = False
    object_id: str | None = None
    for line in response.iter_lines():
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        event_count += 1
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            invalid_events += 1
            continue
        if not isinstance(obj, dict):
            invalid_events += 1
            continue
        try:
            event = parse_event(protocol, obj)
        except ValueError as exc:
            raise ChatStreamError(str(exc)) from exc
        response_id = obj.get("id") or (obj.get("response") or {}).get("id")
        if object_id is None and response_id:
            object_id = str(response_id)
        now = time.perf_counter()
        if event["reasoning"]:
            first_reasoning = first_reasoning or now
            reasoning_parts.append(str(event["reasoning"]))
        if event["content"]:
            first_content = first_content or now
            text_parts.append(str(event["content"]))
        if event["finish_reason"]:
            finish_reason = str(event["finish_reason"])
        if event["done"]:
            saw_done = True
        candidate = event.get("prompt_tokens")
        if candidate is not None:
            prompt_tokens = int(candidate)
        candidate = event.get("completion_tokens")
        if candidate is not None:
            completion_tokens = int(candidate)
        candidate = event.get("reasoning_tokens")
        if candidate is not None:
            reasoning_tokens = int(candidate)
    # Gemini 的 SSE 以 EOF 收尾，没有显式终止标记；拿到 finishReason 即视为收尾。
    if protocol_spec(protocol).family == "gemini" and finish_reason is not None:
        saw_done = True
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "response_text": "".join(text_parts),
        "reasoning_text": "".join(reasoning_parts),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "finish_reason": finish_reason,
        "sse_event_count": event_count,
        "invalid_sse_event_count": invalid_events,
        "saw_done": saw_done,
        "response_object_id": object_id,
        "first_content_mono": first_content,
        "first_reasoning_mono": first_reasoning,
        "request_id": request_id,
    }


def chat_once(config: ChatConfig, *, request_id: str | None = None,
              client: httpx.Client | None = None) -> dict[str, Any]:
    """执行一次对话请求，返回一条字段完整的证据行（永不抛异常）。

    ``client`` 可注入（回归测试用 ``httpx.MockTransport`` 回放真实批次响应体）；
    不传则按 config 自建。
    """
    request_id = request_id or f"{CHAT_PROMPT}-{uuid.uuid4().hex[:10]}"
    url = config.endpoint
    headers = protocol_headers(config.protocol, config.api_key, request_id)
    payload = build_payload(config)
    row: dict[str, Any] = {
        "case": CHAT_PROMPT,
        "index": 1,
        "request_id": request_id,
        "protocol": config.protocol,
        "endpoint": url,
        "model": config.model,
        "system_prompt": config.system,
        "prompt": config.prompt,
        "max_tokens": config.max_tokens,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "started_epoch": time.time(),
        "completed_epoch": None,
        "http_status": None,
        "success": False,
        "error_type": None,
        "error_message": None,
        "attempts": 0,
        "elapsed_seconds": None,
        "response_text": "",
        "reasoning_text": "",
        "content_chars": 0,
        "reasoning_chars": 0,
        "finish_reason": None,
        "usage": None,
        "sse_event_count": 0,
        "invalid_sse_event_count": 0,
        "saw_done": False,
        "response_object_id": None,
        "response_headers": {},
        "verdict": "failed",
        "verdict_detail": "",
        "timing": {"ttft_ms": None, "ttfc_ms": None, "e2e_ms": None, "response_headers_ms": None},
    }
    started_mono = time.perf_counter()
    timeout = httpx.Timeout(
        config.request_timeout_seconds,
        connect=min(30.0, config.request_timeout_seconds),
        write=min(30.0, config.request_timeout_seconds),
        pool=min(30.0, config.request_timeout_seconds),
    )
    attempts = 0
    last_error_type: str | None = None
    last_error_message: str | None = None
    # 注入的 client（回归测试用 MockTransport 回放真实响应体）由调用方负责关闭。
    client_context = (
        httpx.Client(timeout=timeout, verify=config.verify_ssl, trust_env=config.trust_env)
        if client is None else contextlib.nullcontext(client)
    )
    with client_context as client:
        while attempts <= max(0, config.max_retries):
            attempts += 1
            last_error_type = last_error_message = None
            try:
                with client.stream("POST", url, json=payload, headers=headers) as response:
                    headers_mono = time.perf_counter()
                    row["http_status"] = response.status_code
                    row["response_headers"] = _safe_response_headers(response)
                    row["timing"]["response_headers_ms"] = round((headers_mono - started_mono) * 1000, 3)
                    if not 200 <= response.status_code < 300:
                        body = response.read().decode("utf-8", "replace")
                        last_error_type = f"http_{response.status_code}"
                        last_error_message = body[:2000]
                        if response.status_code >= 500 and attempts <= config.max_retries:
                            time.sleep(config.retry_backoff_seconds * attempts)
                            continue
                        break
                    stream = _consume_stream(response, config.protocol, request_id)
                    ended_mono = time.perf_counter()
                    row["response_text"] = stream["response_text"]
                    row["reasoning_text"] = stream["reasoning_text"]
                    row["content_chars"] = len(stream["response_text"])
                    row["reasoning_chars"] = len(stream["reasoning_text"])
                    row["finish_reason"] = stream["finish_reason"]
                    row["sse_event_count"] = stream["sse_event_count"]
                    row["invalid_sse_event_count"] = stream["invalid_sse_event_count"]
                    row["saw_done"] = stream["saw_done"]
                    row["response_object_id"] = stream["response_object_id"]
                    row["usage"] = {
                        "prompt_tokens": stream["prompt_tokens"],
                        "completion_tokens": stream["completion_tokens"],
                        "reasoning_tokens": stream["reasoning_tokens"],
                        "total_tokens": stream["total_tokens"],
                    }
                    row["timing"]["e2e_ms"] = round((ended_mono - started_mono) * 1000, 3)
                    if stream["first_content_mono"] is not None:
                        row["timing"]["ttfc_ms"] = round((stream["first_content_mono"] - started_mono) * 1000, 3)
                    if stream["first_reasoning_mono"] is not None:
                        row["timing"]["ttft_ms"] = round((stream["first_reasoning_mono"] - started_mono) * 1000, 3)
                    elif stream["first_content_mono"] is not None:
                        row["timing"]["ttft_ms"] = row["timing"]["ttfc_ms"]
                    break
            except ChatStreamError as exc:
                # 流内错误事件是上游明确报错，重试没有意义。
                last_error_type, last_error_message = "stream_error", str(exc)[:2000]
                break
            except httpx.TimeoutException as exc:
                last_error_type, last_error_message = "timeout", str(exc)
            except httpx.HTTPError as exc:
                last_error_type, last_error_message = "client_error", str(exc)
            except OSError as exc:
                last_error_type, last_error_message = "connection_error", str(exc)
            if attempts <= config.max_retries:
                time.sleep(config.retry_backoff_seconds * attempts)
            else:
                break
    row["attempts"] = attempts
    row["completed_epoch"] = time.time()
    row["elapsed_seconds"] = round(time.perf_counter() - started_mono, 3)
    status = row["http_status"]
    transport_failed = last_error_type in {"timeout", "client_error", "connection_error"}
    if transport_failed and row["content_chars"] == 0:
        row["error_type"], row["error_message"] = last_error_type, last_error_message
    elif status is not None and not 200 <= status < 300:
        row["error_type"], row["error_message"] = last_error_type or f"http_{status}", last_error_message
    elif last_error_type == "stream_error":
        row["error_type"], row["error_message"] = last_error_type, last_error_message
    if row["error_type"] is None and row["content_chars"] == 0 and row["reasoning_chars"] == 0:
        row["error_type"] = "empty_response"
        row["error_message"] = "HTTP 成功但响应不含任何正文字符"
    verdict, detail = _verdict(row)
    row["verdict"] = verdict
    row["verdict_detail"] = detail
    row["success"] = verdict == "passed"
    return row


def _verdict(row: dict[str, Any]) -> tuple[str, str]:
    """把一次对话的原始观测收口成裁定。"""
    status = row.get("http_status")
    error_type = row.get("error_type")
    if error_type is None:
        return "passed", f"HTTP {status}，正文 {row['content_chars']} 字符"
    if error_type in {"empty_response", "stream_error"}:
        return "warning", str(row.get("error_message") or error_type)
    if status in _BLOCKED_STATUSES:
        return "blocked", f"HTTP {status}：账号/权限/额度或端点不可用（{str(row.get('error_message') or '')[:200]}）"
    if isinstance(status, int) and status >= 500:
        return "blocked", f"HTTP {status}：上游/链路瞬时故障（{str(row.get('error_message') or '')[:200]}）"
    if error_type in {"timeout", "client_error", "connection_error"}:
        return "blocked", f"{error_type}：请求未能完成（{str(row.get('error_message') or '')[:200]}）"
    return "failed", f"HTTP {status}：上游拒绝了本次请求（{str(row.get('error_message') or '')[:400]}）"


def _fmt_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    usage = usage or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def write_response_file(result_dir: Path, row: dict[str, Any]) -> None:
    """纯模型输出，供界面「原文视图 / 复制原文」直接取用。"""
    (result_dir / _RESPONSE_FILE).write_text(row.get("response_text") or "", encoding="utf-8")


def write_report(result_dir: Path, row: dict[str, Any]) -> None:
    """报告保留完整输入与完整响应，便于脱离界面复盘。"""
    usage = _fmt_usage(row.get("usage"))
    lines = [
        f"# {CHAT_PROMPT} 对话窗口响应",
        "",
        f"- 输出上限：{row.get('max_tokens')} tokens",
        f"- HTTP 状态：{row.get('http_status') or '—'}",
        f"- 本次裁定：{row.get('verdict')}（{row.get('verdict_detail')}）",
        f"- 首字时延：{row['timing'].get('ttft_ms')} ms；端到端：{row['timing'].get('e2e_ms')} ms",
        f"- 正文长度：{row.get('content_chars')} 字符；思考长度：{row.get('reasoning_chars')} 字符",
        f"- SSE 事件：{row.get('sse_event_count')}（非法 {row.get('invalid_sse_event_count')}）",
        "",
        "## 输入",
        "",
    ]
    if row.get("system_prompt"):
        lines += ["### 系统提示词", "", "```text", str(row["system_prompt"]), "```", ""]
    lines += ["### 用户消息", "", "```text", str(row.get("prompt") or ""), "```", ""]
    lines += ["## 响应内容", "", str(row.get("response_text") or "（空响应）"), ""]
    if row.get("reasoning_text"):
        lines += ["## 思考过程", "", str(row["reasoning_text"]), ""]
    if row.get("error_message"):
        lines += ["## 错误原文", "", "```text", str(row["error_message"]), "```", ""]
    lines += [
        "## Token 用量",
        "",
        "| 项目 | 数值 |",
        "|---|---:|",
        f"| 输入 Token | {usage['prompt_tokens'] if usage['prompt_tokens'] is not None else '—'} |",
        f"| 输出 Token（含思考） | {usage['completion_tokens'] if usage['completion_tokens'] is not None else '—'} |",
        f"| 思考 Token | {usage['reasoning_tokens'] if usage['reasoning_tokens'] is not None else '—'} |",
        f"| 合计 Token | {usage['total_tokens'] if usage['total_tokens'] is not None else '—'} |",
        "",
        f"> 请求地址：`{row.get('endpoint')}`",
        "> 对话窗口只判定「本次对话是否真的拿到了正文」：账号/权限/额度/瞬时故障记 `blocked`，"
        "上游拒绝记 `failed`，HTTP 成功但无正文记 `warning`；模型能力结论由各专项套件负责。",
    ]
    (result_dir / "report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_summary(config: ChatConfig, row: dict[str, Any]) -> dict[str, Any]:
    usage = _fmt_usage(row.get("usage"))
    timing = row.get("timing") or {}
    e2e_seconds = (timing.get("e2e_ms") or 0) / 1000 if timing.get("e2e_ms") is not None else None
    return {
        "schema_version": "mltest.chat.v1",
        "suite": "chatpane",
        "test_kind": "chat",
        "case": CHAT_PROMPT,
        "record_id": config.record_id or None,
        "protocol": config.protocol,
        "base_url": config.base_url,
        "model": config.model,
        "endpoint": row.get("endpoint"),
        "requested": 1,
        "sent": 1,
        "completed": 1,
        "successful": 1 if row.get("success") else 0,
        "failed": 0 if row.get("success") else 1,
        "success_rate": 1.0 if row.get("success") else 0.0,
        "prompt": config.prompt,
        "system_prompt": config.system,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "response_text": row.get("response_text") or "",
        "response_chars": row.get("content_chars") or 0,
        "reasoning_text": row.get("reasoning_text") or "",
        "reasoning_chars": row.get("reasoning_chars") or 0,
        # 单请求的 P90 就是它自己；写进 summary 让统一指标表有真实数值而不是一排 “—”。
        "ttft_p90_seconds": timing.get("ttft_ms") / 1000 if timing.get("ttft_ms") is not None else None,
        "ttfc_p90_seconds": timing.get("ttfc_ms") / 1000 if timing.get("ttfc_ms") is not None else None,
        "e2e_p90_seconds": e2e_seconds,
        "elapsed_seconds": row.get("elapsed_seconds"),
        "http_status": row.get("http_status"),
        "finish_reason": row.get("finish_reason"),
        "usage": usage,
        "sse_event_count": row.get("sse_event_count"),
        "invalid_sse_event_count": row.get("invalid_sse_event_count"),
        "error_type": row.get("error_type"),
        "error_message": row.get("error_message"),
        "results": [{
            "case_id": CHAT_PROMPT,
            "name": CASES[CHAT_PROMPT].title,
            "status": row.get("verdict"),
            "duration_seconds": row.get("elapsed_seconds"),
            "details": row.get("verdict_detail"),
        }],
        "checks": {
            "http_success": {
                "value": row.get("http_status"),
                "expected": "2xx",
                "passed": isinstance(row.get("http_status"), int) and 200 <= row["http_status"] < 300,
                "detail": "上游是否接受了这次请求",
            },
            "response_nonempty": {
                "value": row.get("content_chars"),
                "expected": "> 0",
                "passed": bool(row.get("content_chars")),
                "detail": "响应是否包含正文",
            },
        },
        "admission_passed": row.get("verdict") == "passed",
        "scope_note": (
            "对话窗口是自由探针：只判定本次对话是否拿到正文，不做模型能力判定；"
            "账号/权限/额度/瞬时故障记 blocked，上游拒绝记 failed，HTTP 成功但无正文记 warning。"
        ),
    }


@dataclass
class ChatOutcome:
    result_dir: Path
    row: dict[str, Any]
    summary: dict[str, Any] = field(default_factory=dict)


def run_chat(config: ChatConfig, output_root: Path,
             client: httpx.Client | None = None) -> ChatOutcome:
    """跑一次对话并落盘（不含收尾；收尾由 ``common.finalizer`` 统一执行）。"""
    result_dir = output_root / f"{CHAT_PROMPT}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    result_dir.mkdir(parents=True, exist_ok=False)
    (result_dir / "chat_request.json").write_text(
        json.dumps(redact({
            "case": CHAT_PROMPT,
            "protocol": config.protocol,
            "base_url": config.base_url,
            "model": config.model,
            "endpoint": config.endpoint,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "system_prompt": config.system,
            "prompt": config.prompt,
        }), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    audit = AuditLog(result_dir / "requests.jsonl")
    try:
        row = chat_once(config, request_id=f"{CHAT_PROMPT}-1-{uuid.uuid4().hex[:10]}", client=client)
        audit.write(row)
    finally:
        audit.close()
    write_response_file(result_dir, row)
    summary = build_summary(config, row)
    (result_dir / "summary.json").write_text(
        json.dumps(redact(summary), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    write_report(result_dir, row)
    return ChatOutcome(result_dir=result_dir, row=row, summary=summary)
