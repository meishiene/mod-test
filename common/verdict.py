"""统一测试结果状态，不把 HTTP 成功等同于案例通过。"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

STATUSES = {"passed", "failed", "not_evaluable", "error", "skipped", "warning"}
@dataclass
class Check:
    name: str; passed: bool | None; expected: Any = None; actual: Any = None; detail: str = ""; severity: str = "error"
@dataclass
class Verdict:
    status: str; passed: bool; checks: list[Check]; reasons: list[str]
    def jsonable(self): return asdict(self)

def verdict(checks: list[Check]) -> Verdict:
    hard = [x for x in checks if x.passed is False and x.severity == "error"]
    unknown = [x for x in checks if x.passed is None]
    if hard: return Verdict("failed", False, checks, [x.detail or x.name for x in hard])
    if unknown: return Verdict("not_evaluable", False, checks, [x.detail or x.name for x in unknown])
    return Verdict("passed", True, checks, [])

def check(name: str, passed: bool | None, expected: Any = None, actual: Any = None, detail: str = "", severity: str = "error") -> Check:
    return Check(name, passed, expected, actual, detail, severity)

def transport_verdict(record: dict[str, Any], *, expected_status: int | tuple[int, ...] = 200) -> Verdict:
    expected = (expected_status,) if isinstance(expected_status, int) else tuple(expected_status)
    return verdict([check("http_status", record.get("http_status") in expected, expected, record.get("http_status"), "HTTP 状态码不符合案例预期"),
                    check("transport", record.get("transport_error") is None and record.get("error_type") is None, True, record.get("error_type"), "传输层异常")])

def attach_verdict(record: dict[str, Any], result: Verdict) -> dict[str, Any]:
    out = dict(record); out["validation"] = result.jsonable(); out["case_status"] = result.status; out["case_passed"] = result.passed; return out
