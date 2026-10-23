from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .config import ApiConfig
from .protocols import OPENAI_COMPATIBLE_PROTOCOLS
from .protocols import headers as protocol_headers
from .protocols import payload as protocol_payload


def _int(value: Any) -> int | None:
    return int(value) if value is not None else None


def parse_event(protocol: str, obj: dict[str, Any]) -> dict[str, Any]:
    """Normalize one provider SSE JSON object into the common metric event shape."""
    result: dict[str, Any] = {"content": "", "reasoning": "", "finish_reason": None,
                              "prompt_tokens": None, "completion_tokens": None,
                              "reasoning_tokens": None, "visible_tokens": None, "done": False}
    if protocol == "openai_responses":
        event_type = obj.get("type")
        response = obj.get("response") or {}
        usage = response.get("usage") or obj.get("usage") or {}
        result["prompt_tokens"] = _int(usage.get("input_tokens"))
        result["completion_tokens"] = _int(usage.get("output_tokens"))
        details = usage.get("output_tokens_details") or {}
        result["reasoning_tokens"] = _int(details.get("reasoning_tokens"))
        if event_type == "response.output_text.delta":
            result["content"] = str(obj.get("delta") or "")
        elif event_type == "response.reasoning_summary_text.delta":
            result["reasoning"] = str(obj.get("delta") or "")
        elif event_type in {"response.completed", "response.incomplete", "response.failed"}:
            result["done"] = True
            if event_type == "response.completed":
                result["finish_reason"] = "stop"
            elif event_type == "response.incomplete":
                reason = (response.get("incomplete_details") or {}).get("reason")
                result["finish_reason"] = "length" if reason == "max_output_tokens" else (reason or "incomplete")
            else:
                result["finish_reason"] = "failed"
        elif event_type == "error":
            raise ValueError(json.dumps(obj.get("error") or obj, ensure_ascii=False))
        return result
    if protocol in OPENAI_COMPATIBLE_PROTOCOLS:
        usage = obj.get("usage") or {}
        result["prompt_tokens"] = _int(usage.get("prompt_tokens", usage.get("input_tokens")))
        result["completion_tokens"] = _int(usage.get("completion_tokens", usage.get("output_tokens")))
        details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
        result["reasoning_tokens"] = _int(details.get("reasoning_tokens", usage.get("reasoning_tokens")))
        result["visible_tokens"] = _int(usage.get("visible_completion_tokens"))
        choices = obj.get("choices") or []
        if choices:
            choice = choices[0] or {}; delta = choice.get("delta") or choice.get("message") or {}
            content = delta.get("content") or ""; reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if isinstance(content, list): content = "".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content)
            if isinstance(reasoning, list): reasoning = "".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in reasoning)
            result.update(content=str(content), reasoning=str(reasoning), finish_reason=choice.get("finish_reason"))
        return result
    if protocol == "anthropic":
        event_type = obj.get("type")
        if event_type == "message_start":
            usage = (obj.get("message") or {}).get("usage") or {}
            result["prompt_tokens"] = _int(usage.get("input_tokens"))
        elif event_type == "content_block_delta":
            delta = obj.get("delta") or {}; delta_type = delta.get("type")
            if delta_type == "text_delta": result["content"] = str(delta.get("text") or "")
            elif delta_type in {"thinking_delta", "signature_delta"}: result["reasoning"] = str(delta.get("thinking") or "")
        elif event_type == "message_delta":
            usage = obj.get("usage") or {}; result["completion_tokens"] = _int(usage.get("output_tokens"))
            stop_reason = (obj.get("delta") or {}).get("stop_reason")
            result["finish_reason"] = "length" if stop_reason == "max_tokens" else stop_reason
        elif event_type == "message_stop": result["done"] = True
        elif event_type == "error": raise ValueError(json.dumps(obj.get("error") or obj, ensure_ascii=False))
        return result
    if protocol == "gemini":
        usage = obj.get("usageMetadata") or {}
        result["prompt_tokens"] = _int(usage.get("promptTokenCount"))
        result["completion_tokens"] = _int(usage.get("candidatesTokenCount"))
        result["reasoning_tokens"] = _int(usage.get("thoughtsTokenCount"))
        candidates = obj.get("candidates") or []
        if candidates:
            candidate = candidates[0] or {}; finish = candidate.get("finishReason")
            if finish: result["finish_reason"] = "length" if finish in {"MAX_TOKENS", "MAX_TOKENS_REACHED"} else str(finish).lower()
            for part in ((candidate.get("content") or {}).get("parts") or []):
                text = part.get("text") or ""
                if part.get("thought"): result["reasoning"] += str(text)
                else: result["content"] += str(text)
        return result
    raise ValueError(f"不支持的协议: {protocol}")


async def stream_chat(session: httpx.AsyncClient, config: ApiConfig, *, request_id: str, content: str | None = None,
                      messages: list[dict[str, str]] | None = None, workload_type: str = "synthetic",
                      target_visible_tokens: int | None = None, max_tokens: int, case_name: str, dataset_id: str,
                      phase: str, counted: bool, planned_epoch: float | None = None) -> dict[str, Any]:
    started_epoch, started_mono = time.time(), time.perf_counter()
    row: dict[str, Any] = {"case": case_name, "protocol": config.protocol, "phase": phase, "counted": counted,
        "request_id": request_id, "dataset_id": dataset_id, "planned_epoch": planned_epoch, "started_epoch": started_epoch,
        "completed_epoch": None, "send_delay_ms": max(0.0, (started_epoch-planned_epoch)*1000) if planned_epoch is not None else None,
        "success": False, "transport_status": "unknown", "http_status": None, "error_type": None, "error_message": None, "prompt_tokens": None,
        "completion_tokens": None, "reasoning_tokens": None, "visible_completion_tokens": None, "total_tokens": None,
        "finish_reason": None, "first_sse_seconds": None, "first_reasoning_seconds": None, "ttft_seconds": None, "ttfc_seconds": None,
        "e2e_seconds": None, "response_headers_seconds": None, "tpot_ms_per_visible_token": None, "content_chunk_count": 0, "reasoning_chunk_count": 0,
        "content_char_count": 0, "sse_event_count": 0, "invalid_sse_event_count": 0, "saw_done": False,
        "response_request_id": None, "upstream_request_id": None, "response_object_id": None, "response_headers": {}, "raw_response_body": "", "raw_sse_lines": [], "raw_sse_events": [], "workload_type": workload_type,
        "message_count": len(messages or []), "target_visible_tokens": target_visible_tokens}
    first_sse = first_reasoning = first_content = None
    response_headers_time = None
    prompt_tokens = completion_tokens = reasoning_tokens = visible_tokens = None
    try:
        async with session.stream("POST", config.endpoint,
                json=protocol_payload(config.protocol, config.model, messages if messages is not None else (content or ""), max_tokens, config.include_usage, config.reasoning_effort),
                headers=protocol_headers(config.protocol, config.api_key, request_id)) as response:
            response_headers_time = time.perf_counter()
            row["http_status"] = response.status_code
            row["response_headers_seconds"] = response_headers_time - started_mono
            names = {"x-request-id", "request-id", "x-trace-id", "trace-id", "traceparent", "x-oneapi-request-id", "x-tt-logid", "server", "via", "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests"}
            row["response_headers"] = {k.lower():v for k,v in response.headers.items() if k.lower() in names}
            row["response_request_id"] = response.headers.get("x-request-id") or response.headers.get("request-id") or response.headers.get("x-trace-id") or response.headers.get("x-oneapi-request-id") or response.headers.get("x-tt-logid")
            row["upstream_request_id"] = row["response_request_id"]
            if not 200 <= response.status_code < 300:
                raw_body = (await response.aread()).decode("utf-8","replace")
                row["raw_response_body"] = raw_body
                row["error_type"] = f"http_{response.status_code}"; row["error_message"] = raw_body[:2000]
                return _finish(row, started_mono)
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"): continue
                now = time.perf_counter(); first_sse = first_sse or now; row["sse_event_count"] += 1
                row["raw_sse_lines"].append(line)
                data = line[5:].strip(); row["raw_sse_events"].append(data)
                if data == "[DONE]": row["saw_done"] = True; continue
                try: obj = json.loads(data); event = parse_event(config.protocol, obj)
                except json.JSONDecodeError: row["invalid_sse_event_count"] += 1; continue
                p,c,r,v = (event[k] for k in ("prompt_tokens","completion_tokens","reasoning_tokens","visible_tokens"))
                prompt_tokens = p if p is not None else prompt_tokens; completion_tokens = c if c is not None else completion_tokens
                reasoning_tokens = r if r is not None else reasoning_tokens; visible_tokens = v if v is not None else visible_tokens
                response_id = obj.get("id") or (obj.get("response") or {}).get("id")
                if row.get("response_object_id") is None and response_id: row["response_object_id"] = response_id
                if event["finish_reason"] is not None: row["finish_reason"] = event["finish_reason"]
                if event["reasoning"]: first_reasoning = first_reasoning or now; row["reasoning_chunk_count"] += 1
                if event["content"]: first_content = first_content or now; row["content_chunk_count"] += 1; row["content_char_count"] += len(event["content"])
                if event["done"]: row["saw_done"] = True
        # Gemini SSE normally ends by EOF and does not emit an explicit terminal marker.
        if config.protocol == "gemini" and row["finish_reason"] is not None: row["saw_done"] = True
        if visible_tokens is None and completion_tokens is not None and reasoning_tokens is not None:
            visible_tokens = max(0, completion_tokens-reasoning_tokens)
        row["transport_status"] = "completed"
        row.update(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, reasoning_tokens=reasoning_tokens,
                   visible_completion_tokens=visible_tokens, total_tokens=prompt_tokens+completion_tokens if prompt_tokens is not None and completion_tokens is not None else None,
                   first_sse_seconds=first_sse-started_mono if first_sse else None, first_reasoning_seconds=first_reasoning-started_mono if first_reasoning else None,
                   ttft_seconds=min(value for value in (first_reasoning, first_content) if value is not None)-started_mono if first_reasoning is not None or first_content is not None else None,
                   ttfc_seconds=first_content-started_mono if first_content else None)
        ended=time.perf_counter(); row["e2e_seconds"]=ended-started_mono
        basis_tokens=visible_tokens if visible_tokens is not None else completion_tokens
        row["tpot_token_basis"]="visible" if visible_tokens is not None else "completion_including_reasoning"
        if first_content is not None and basis_tokens is not None and basis_tokens >= 2: row["tpot_ms_per_visible_token"]=(ended-first_content)*1000/(basis_tokens-1)
        if first_content is None: row["error_type"]="no_visible_content"
        elif prompt_tokens is None or completion_tokens is None: row["error_type"]="missing_usage"
        elif not row["saw_done"]: row["error_type"]="missing_done_marker"
        else: row["success"]=True
        return _finish(row, started_mono, True)
    except TimeoutError as exc: row["transport_status"],row["error_type"],row["error_message"]="error","timeout",str(exc)
    except httpx.HTTPError as exc: row["transport_status"],row["error_type"],row["error_message"]="error","client_error",str(exc)
    except Exception as exc: row["transport_status"],row["error_type"],row["error_message"]="error",type(exc).__name__,str(exc)
    return _finish(row, started_mono)


def _finish(row: dict[str, Any], started_mono: float, preserve_e2e: bool=False) -> dict[str, Any]:
    row["completed_epoch"]=time.time()
    if not preserve_e2e or row.get("e2e_seconds") is None: row["e2e_seconds"]=time.perf_counter()-started_mono
    return row
