from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .config import ApiConfig


def _usage_from(obj: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    usage = obj.get("usage") or {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens", usage.get("cached_tokens"))
    return (int(prompt) if prompt is not None else None, int(completion) if completion is not None else None, int(cached) if cached is not None else None)


async def stream_chat(
    session: httpx.AsyncClient,
    config: ApiConfig,
    request_id: str,
    content: str,
    max_tokens: int,
    case_name: str,
    dataset_id: str,
    phase: str,
    counted: bool,
    planned_epoch: float | None = None,
) -> dict[str, Any]:
    started_epoch = time.time()
    started_mono = time.perf_counter()
    result: dict[str, Any] = {
        "case": case_name,
        "phase": phase,
        "counted": counted,
        "request_id": request_id,
        "dataset_id": dataset_id,
        "planned_epoch": planned_epoch,
        "started_epoch": started_epoch,
        "queue_delay_ms": max(0.0, (started_epoch - planned_epoch) * 1000) if planned_epoch else None,
        "success": False,
        "transport_status": "unknown",
        "http_status": None,
        "error_type": None,
        "error_message": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_tokens": None,
        "cache_ratio": None,
        "ttft_seconds": None,
        "e2e_seconds": None,
        "tpot_ms_per_token": None,
        "content_chunk_count": 0,
        "content_char_count": 0,
        "sse_event_count": 0,
        "invalid_sse_event_count": 0,
        "response_request_id": None,
        "response_headers": {},
        "raw_response_body": "", "raw_sse_lines": [], "raw_sse_events": [],
    }
    payload = {
        "model": config.model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 1,
        "reasoning_effort": "low",
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": "你是性能测试助手。严格按要求持续输出，不要调用工具，不要提前结束。"},
            {"role": "user", "content": content},
        ],
    }
    first_content_mono: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    saw_done = False
    try:
        async with session.stream(
            "POST", config.endpoint, json=payload,
            headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json", "X-Request-ID": request_id},
        ) as response:
            result["http_status"] = response.status_code
            # Keep only non-sensitive diagnostic headers useful for NewAPI/load-balancer correlation.
            diagnostic_names = {"x-request-id", "x-trace-id", "traceparent", "server", "via", "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests"}
            result["response_headers"] = {key.lower(): value for key, value in response.headers.items() if key.lower() in diagnostic_names}
            result["response_request_id"] = response.headers.get("x-request-id") or response.headers.get("x-trace-id")
            if response.status_code < 200 or response.status_code >= 300:
                result["error_type"] = "http_error"
                result["raw_response_body"] = (await response.aread()).decode("utf-8", errors="replace")
                result["error_message"] = result["raw_response_body"][:2000]
                return result
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                result["raw_sse_lines"].append(line)
                data = line[5:].strip(); result["raw_sse_events"].append(data)
                if data == "[DONE]":
                    saw_done = True
                    continue
                result["sse_event_count"] += 1
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    result["invalid_sse_event_count"] += 1
                    continue
                p, c, cached = _usage_from(obj)
                prompt_tokens = p if p is not None else prompt_tokens
                completion_tokens = c if c is not None else completion_tokens
                cached_tokens = cached if cached is not None else cached_tokens
                choices = obj.get("choices") or []
                for choice in choices:
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if isinstance(text, str) and text:
                        result["content_chunk_count"] += 1
                        result["content_char_count"] += len(text)
                        if first_content_mono is None:
                            first_content_mono = time.perf_counter()
            result["transport_status"] = "completed"
            ended_mono = time.perf_counter()
            result["completed_epoch"] = time.time()
            result["prompt_tokens"] = prompt_tokens
            result["completion_tokens"] = completion_tokens
            result["cached_tokens"] = cached_tokens
            result["cache_ratio"] = cached_tokens / prompt_tokens if cached_tokens is not None and prompt_tokens else None
            result["ttft_seconds"] = first_content_mono - started_mono if first_content_mono else None
            result["e2e_seconds"] = ended_mono - started_mono
            if first_content_mono is not None and completion_tokens is not None and completion_tokens >= 2:
                result["tpot_ms_per_token"] = (ended_mono - first_content_mono) * 1000 / (completion_tokens - 1)
            if first_content_mono is None:
                result["error_type"] = "no_nonempty_chunk"
            elif prompt_tokens is None or completion_tokens is None:
                result["error_type"] = "missing_usage"
            elif not saw_done:
                result["error_type"] = "missing_done_marker"
            else:
                result["success"] = True
            return result
    except TimeoutError as exc:
        result["transport_status"] = "error"
        result["error_type"] = "timeout"
        result["error_message"] = str(exc)
    except httpx.HTTPError as exc:
        result["transport_status"] = "error"
        result["error_type"] = "client_error"
        result["error_message"] = str(exc)
    except Exception as exc:
        result["transport_status"] = "error"  # preserve unexpected errors in raw output
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
    result["completed_epoch"] = time.time()
    return result
