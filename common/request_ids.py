"""统一提取上游链路标识，供所有测试执行器写入原始明细。"""
from __future__ import annotations
from typing import Any, Mapping

UPSTREAM_ID_HEADERS = (
    "x-request-id", "request-id", "x-trace-id", "x-correlation-id",
    "x-tt-logid", "x-b3-traceid", "trace-id", "traceparent",
)

def response_request_id(headers: Mapping[str, Any] | None, body: Any = None) -> str | None:
    h = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    for name in UPSTREAM_ID_HEADERS:
        if h.get(name):
            return h[name]
    # 仅作兜底：响应对象 id 通常是中转层 completion id，不保证等同于上游后台 request id。
    if isinstance(body, dict) and body.get("id"):
        return str(body["id"])
    return None

def id_fields(headers: Mapping[str, Any] | None, body: Any = None) -> dict[str, Any]:
    h = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    fields = {"response_headers": dict(h), "upstream_request_id": response_request_id(h, None)}
    if isinstance(body, dict) and body.get("id"):
        fields["response_object_id"] = str(body["id"])
    return fields
