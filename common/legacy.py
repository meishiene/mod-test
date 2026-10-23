"""Bridge helpers for migrating existing suite-specific runners."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from typing import Any
from .verdict import Check, Verdict, verdict

def checks_from_legacy(*, http_status: int | None, expected_status: int | tuple[int, ...] = 200,
                       assertions: list[tuple[str, bool | None, Any, Any, str]] = ()) -> Verdict:
    expected = (expected_status,) if isinstance(expected_status, int) else tuple(expected_status)
    checks=[Check("http_status", http_status in expected, expected, http_status, "HTTP 状态码不符合预期")]
    checks.extend(Check(name, passed, expected, actual, detail) for name, passed, expected, actual, detail in assertions)
    return verdict(checks)

def attach_case_verdict(record: dict[str, Any], result: Verdict) -> dict[str, Any]:
    value=dict(record); value["validation"] = result.jsonable(); value["case_status"] = result.status; value["case_passed"] = result.passed; return value

def legacy_result(result: Any, *, record: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize dataclass/dict results without losing suite-specific observations."""
    value=asdict(result) if is_dataclass(result) else dict(result)
    passed=value.pop("passed", None); status=value.pop("status", None)
    value["case_passed"]=passed; value["case_status"]=status or ("passed" if passed is True else "failed" if passed is False else "not_evaluable")
    if record is not None: value["execution"]=record
    return value
