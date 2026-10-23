"""Manifest-driven suite support shared by all test families.

Cases describe requests and expected outcomes; the executor only transports and
records evidence; validators produce the verdict.  Unsupported scenario-only
cases are explicitly marked ``not_evaluable`` rather than replaced by a fake
successful request.
"""
from __future__ import annotations
import copy, json, time, uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable
from .test_framework import ExecutionRecord, HttpExecutor, ResultLog, validate_case
from .validators import status, status_2xx, status_error, content, usage, sse, json_content, tool_calls

class ManifestCase:
    # 只描述测试场景、不构成可发送请求体的字段。它们既不是请求参数，也不能凭
    # 它们伪造一个"看起来像请求"的东西（架构约定：不把 scenario 降级为普通请求）。
    SCENARIO_ONLY_KEYS = frozenset({
        "headers", "scenario", "input_tokens_target", "concurrency", "repetition",
        "batch", "phase", "wait_seconds", "warm_input_tokens_target",
        "same_prefix_byte_exact", "append_tail",
    })

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw; self.id = raw["id"]; self.request = raw.get("request") or {}
        self.expected = raw.get("expected") or {}
        self.assertions = raw.get("assertions") or []

    def blocking_keys(self) -> list[str]:
        """用例里被执行器剥离的场景字段，用于生成可读的 not_evaluable 原因。"""
        return sorted(k for k in self.request if k in self.SCENARIO_ONLY_KEYS)

    def build_payload(self, *, model: str) -> dict[str, Any] | None:
        payload = copy.deepcopy({k:v for k,v in self.request.items() if k not in self.SCENARIO_ONLY_KEYS})
        payload.setdefault("model", model)
        if "messages" not in payload and "input_tokens_target" in self.request:
            target = int(self.request["input_tokens_target"])
            payload["messages"] = [{"role":"user", "content":("性能测试固定前缀。" * max(1, target * 4 // 8))[:target * 4]}]
        # 没有 messages 就不是一个可发送的 chat 请求。发出去必然被上游 400 拒绝，
        # 从而把一个"用例没写完"的问题误报成"被测模型不达标"——这是最坏的假阴性。
        if "messages" not in payload:
            return None
        return payload

    def validators(self) -> list[Callable]:
        e = self.expected
        validators=[]
        expected_status=e.get("http_status") or e.get("status")
        if expected_status is None:
            for text in self.assertions:
                if text.startswith("HTTP "):
                    try: expected_status=int(text.split()[1]); break
                    except (ValueError, IndexError): pass
        if expected_status is not None: validators.append(status(*expected_status) if isinstance(expected_status, list) else status(int(expected_status)))
        error=e.get("error") or {}
        if error or any("错误" in str(x) for x in self.assertions):
            validators.append(status_error(*(expected_status if isinstance(expected_status,list) else [int(expected_status)] if expected_status else [400]), error_types=tuple(error.get("types", ()) or ((error.get("type"),) if error.get("type") else ())), message_contains=tuple(error.get("message_contains", ()))))
        if e.get("content_exact"): validators.append(content(exact=e["content_exact"]))
        if e.get("content_contains"): validators.append(content(contains=tuple(e["content_contains"])))
        if any("正文" in str(x) for x in self.assertions) and not e.get("content_exact") and not e.get("content_contains"): validators.append(content(nonempty=True))
        if e.get("usage_fields"): validators.append(usage(*e["usage_fields"]))
        if self.request.get("stream") and any("SSE" in str(x) or "DONE" in str(x) for x in self.assertions): validators.append(sse())
        if self.request.get("response_format", {}).get("type") == "json_object" or self.id.endswith("JSON-001"): validators.append(json_content())
        if e.get("tool_calls"): validators.append(tool_calls(**e["tool_calls"]))
        return validators

def not_evaluable(case_id: str, reason: str) -> dict[str, Any]:
    return {"case_id": case_id, "status": "not_evaluable", "passed": False, "checks": [], "failure_reasons": [reason]}

async def execute_manifest_case(executor: HttpExecutor, log: ResultLog, case: ManifestCase, *, model: str, url: str, base_headers: dict[str,str]) -> dict[str,Any]:
    payload=case.build_payload(model=model)
    if payload is None:
        keys="、".join(f"`{k}`" for k in case.blocking_keys()) or "无"
        return not_evaluable(case.id, f"用例未提供可发送的请求内容：请求里只有 {keys} 这类场景/传输字段，没有 `messages`；未伪造请求，需要专用执行器或在清单中补齐")
    headers=dict(base_headers); headers.update(case.request.get("headers") or {})
    headers.setdefault("X-Request-ID", f"{case.id}-{uuid.uuid4().hex[:10]}")
    stream=bool(payload.get("stream"))
    record=await executor.execute(case_id=case.id,url=url,payload=payload,headers=headers,stream=stream)
    verdict=validate_case(record,case.validators()); record.validation=verdict
    await log.write(record)
    return {"case_id":case.id,"status":verdict.status,"passed":verdict.passed,"checks":[asdict(x) for x in verdict.checks],"failure_reasons":verdict.reasons,"request_id":record.request_id}
