"""Composable validators shared by every suite.

A validator returns a check result; it never infers pass from transport success.
"""
from __future__ import annotations
import json
from typing import Any, Callable
from .test_framework import Assertion, ExecutionRecord, check

Validator = Callable[[ExecutionRecord], Assertion | list[Assertion]]

def status(*expected: int) -> Validator:
    return lambda r: check("http_status", r.http_status in expected, list(expected), r.http_status, f"HTTP 状态码不符合预期")

def status_2xx() -> Validator: return status(200)

def status_error(*expected: int, error_types: tuple[str, ...] = (), message_contains: tuple[str, ...] = ()) -> Validator:
    def validate(r):
        body = r.response_body if isinstance(r.response_body, dict) else {}
        err = body.get("error") if isinstance(body.get("error"), dict) else {}
        msg = str(err.get("message") or r.response_text or "")
        out = [status(*expected)(r)]
        if error_types:
            out.append(check("error_type", err.get("type") in error_types, list(error_types), err.get("type"), "错误类型不符合预期"))
        if message_contains:
            out.append(check("error_message", any(x.lower() in msg.lower() for x in message_contains), list(message_contains), msg[:500], "错误消息未包含预期语义"))
        out.append(check("error_body", bool(body.get("error")) or bool(r.response_text), True, bool(body.get("error")) or bool(r.response_text), "错误响应不可解析"))
        return out
    return validate

def content(*, exact: str | None = None, contains: tuple[str, ...] = (), nonempty: bool = False) -> Validator:
    def validate(r):
        value = r.content
        if not value and isinstance(r.response_body, dict):
            choices = r.response_body.get("choices") or []
            if choices: value = ((choices[0].get("message") or {}).get("content") or "")
        ok = (exact is None or value == exact) and all(x in value for x in contains) and (not nonempty or bool(value))
        return check("content", ok, exact or list(contains), value[:1000], "正文断言失败")
    return validate

def usage(*fields: str) -> Validator:
    def validate(r):
        u = r.usage or {}
        missing = [f for f in fields if _get(u, f) is None]
        return check("usage", not missing, fields, u, f"usage 缺少字段: {missing}")
    return validate

def sse(*, require_done=True, require_content=True) -> Validator:
    def validate(r):
        return [check("sse_done", not require_done or r.done_received, True, r.done_received, "缺少 [DONE]"),
                check("sse_json", r.invalid_sse_events == 0, 0, r.invalid_sse_events, "存在非法 SSE JSON"),
                check("sse_content", not require_content or bool(r.content), True, bool(r.content), "没有可见正文")]
    return validate

def json_content(*, required: dict[str, Any] | None = None) -> Validator:
    def validate(r):
        raw = r.content
        if not raw and isinstance(r.response_body, dict):
            choices = r.response_body.get("choices") or []
            raw = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
        try: value = json.loads(raw); valid = isinstance(value, dict)
        except (TypeError, ValueError): value, valid = None, False
        if required and valid: valid = all(value.get(k) == v for k, v in required.items())
        return check("json_content", valid, required or "JSON object", raw[:1000], "正文不是符合要求的 JSON")
    return validate

def tool_calls(*, count: int | None = None, names: tuple[str, ...] = (), args: dict[str, Any] | None = None) -> Validator:
    def validate(r):
        body = r.response_body if isinstance(r.response_body, dict) else {}
        choices = body.get("choices") or []
        msg = (choices[0].get("message") or {}) if choices else {}
        calls = msg.get("tool_calls") or []
        ok = (count is None or len(calls) == count) and all((c.get("function") or {}).get("name") in names for c in calls) if names else (count is None or len(calls) == count)
        if args and calls:
            try: actual = json.loads((calls[0].get("function") or {}).get("arguments") or "{}")
            except ValueError: actual = {}
            ok = ok and all(actual.get(k) == v for k, v in args.items())
        return check("tool_calls", ok, {"count": count, "names": names, "args": args}, calls, "工具调用断言失败")
    return validate

def metric(name: str, *, minimum: float | None = None, maximum: float | None = None, field: str | None = None) -> Validator:
    def validate(r):
        value = r.timing.get(field or name)
        ok = value is not None and (minimum is None or value >= minimum) and (maximum is None or value <= maximum)
        return check(name, ok, {"minimum": minimum, "maximum": maximum}, value, f"性能指标 {name} 不达标或缺失")
    return validate


def content_type(*expected: str) -> Validator:
    """Validate response Content-Type; missing headers are not evaluable."""
    def validate(r):
        actual = next((v for k, v in r.response_headers.items() if k.lower() == "content-type"), None)
        ok = actual is not None and any(str(actual).lower().split(";", 1)[0].strip() == x.lower() or x.lower() in str(actual).lower() for x in expected)
        return check("content_type", ok, list(expected), actual, "Content-Type 缺失或不符合预期")
    return validate

def model_identity(*expected: str, allow_missing: bool = False) -> Validator:
    def validate(r):
        actual = r.response_body.get("model") if isinstance(r.response_body, dict) else None
        ok = (allow_missing and actual is None) or (actual in expected)
        return check("model_identity", ok, list(expected), actual, "响应模型身份不符合预期")
    return validate

def reasoning_separated(*, required: bool = True) -> Validator:
    def validate(r):
        body = r.response_body if isinstance(r.response_body, dict) else {}
        choices = body.get("choices") or []
        msg = (choices[0].get("message") or {}) if choices else {}
        reasoning = r.reasoning or msg.get("reasoning_content") or msg.get("reasoning") or ""
        visible = r.content or msg.get("content") or ""
        ok = bool(reasoning) and bool(visible) if required else not (reasoning and reasoning == visible)
        return check("reasoning_separation", ok, {"reasoning": required, "visible_content": required}, {"reasoning_chars": len(str(reasoning)), "content_chars": len(str(visible))}, "reasoning 与正文未按预期分离")
    return validate

def tool_replay(*, require_tool_call_id: bool = True, content_contains: str | None = None) -> Validator:
    def validate(r):
        body = r.response_body if isinstance(r.response_body, dict) else {}
        choices = body.get("choices") or []; msg = (choices[0].get("message") or {}) if choices else {}
        calls = msg.get("tool_calls") or []
        ids_ok = all(c.get("id") for c in calls) if require_tool_call_id else True
        ok = bool(calls) and ids_ok
        if content_contains: ok = ok and content_contains in (r.content or "")
        return check("tool_replay", ok, {"tool_call_id": require_tool_call_id, "content_contains": content_contains}, {"calls": len(calls), "ids_ok": ids_ok}, "工具调用/回传链路断言失败")
    return validate

def percentile(values: list[float], p: float) -> float | None:
    if not values: return None
    xs = sorted(float(x) for x in values); rank = (len(xs) - 1) * p
    lo, hi = int(rank), min(int(rank) + 1, len(xs) - 1); frac = rank - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac

def aggregate_percentile(name: str, values: list[float], *, p: float, maximum: float, minimum_samples: int = 1) -> Assertion:
    actual = percentile(values, p) if len(values) >= minimum_samples else None
    ok = actual is not None and actual <= maximum
    return check(name, ok, {"percentile": p, "maximum": maximum, "minimum_samples": minimum_samples}, actual, "性能分位数缺失、样本不足或未达标")

def _get(obj: dict[str, Any], path: str):
    for part in path.split("."):
        if not isinstance(obj, dict): return None
        obj = obj.get(part)
    return obj

def aggregate_metric(name: str, values: list[float], *, percentile_value: float | None = None, maximum: float | None = None, minimum_samples: int = 1) -> Assertion:
    """Validate a suite-level metric while treating missing/insufficient samples as failure."""
    if len(values) < minimum_samples:
        return check(name, False, {"minimum_samples": minimum_samples}, {"samples": len(values)}, f"指标样本不足: {len(values)}/{minimum_samples}")
    actual = percentile_value if percentile_value is not None else sum(values) / len(values)
    return check(name, (maximum is None or actual <= maximum), {"maximum": maximum}, actual, f"指标未达标: {actual}")


def response_field(path: str, *, expected: Any = None, present: bool = True) -> Validator:
    def validate(r):
        value = _get(r.response_body or {}, path) if isinstance(r.response_body, dict) else None
        ok = (value is not None) == present and (expected is None or value == expected)
        return check(f"response.{path}", ok, expected if expected is not None else present, value, "响应字段断言失败")
    return validate

def finish_reason(*expected: str) -> Validator:
    return lambda r: check("finish_reason", bool(set(r.finish_reasons) & set(expected)), list(expected), r.finish_reasons, "finish_reason 不符合预期")

def request_field(path: str, *, expected: Any = None, present: bool = True) -> Validator:
    def validate(r):
        value = _get(r.request, path)
        ok = (value is not None) == present and (expected is None or value == expected)
        return check(f"request.{path}", ok, expected if expected is not None else present, value, "实际请求与案例定义不一致")
    return validate

def cache_ratio(*, minimum: float, prompt_path: str = "prompt_tokens", cached_path: str = "prompt_tokens_details.cached_tokens") -> Validator:
    def validate(r):
        prompt = _get(r.usage or {}, prompt_path); cached = _get(r.usage or {}, cached_path)
        ratio = (float(cached) / float(prompt)) if prompt and cached is not None else None
        return check("cache_ratio", ratio is not None and ratio >= minimum, minimum, ratio, "缓存命中率缺失或未达标")
    return validate
