"""Normalize every suite's summary to one verdict-bearing schema."""
from __future__ import annotations
from typing import Any
from .artifacts import SCHEMA_VERSION

# 领域套件可以在行内 status/case_status 给出显式裁定。只承认裁定词表内的取值：
# 传输层的 status（如 ok / completed）不能被当成案例结论，否则会把 warning、blocked
# 这类非阻断结论压平成 failed，或把传输成功误升级为案例通过。
VERDICT_STATUSES = {"passed", "failed", "error", "blocked", "warning", "skipped", "not_evaluable", "interrupted"}

def normalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    out=dict(summary)
    # 结果可能被收尾两次（套件自身一次、Web 聚合再一次），必须保留首次记录的来源版本，
    # 否则第二次会把上一轮写入的 mltest schema 当成来源版本，元数据被逐步污染。
    out["source_schema_version"] = summary.get("source_schema_version") or summary.get("schema_version")
    out["schema_version"] = SCHEMA_VERSION
    results = out.get("results") or out.get("records") or []
    normalized=[]
    for item in results:
        row=dict(item) if isinstance(item,dict) else {"value":item}
        legacy_passed=row.get("case_passed", row.get("passed"))
        legacy_status=row.get("case_status")
        if legacy_status is None and row.get("status") in VERDICT_STATUSES:
            legacy_status = row["status"]
        # Adapters for common external verifier result shapes.  A raw HTTP/status
        # row is intentionally not upgraded: it still needs a domain verdict.
        if legacy_passed is None and "correct" in row and row.get("status") == "ok":
            legacy_passed = bool(row.get("correct")); legacy_status = "passed" if legacy_passed else "failed"
        if legacy_passed is None and "reward" in row and row.get("evaluated", True):
            legacy_passed = float(row.get("reward") or 0) >= 1; legacy_status = "passed" if legacy_passed else "failed"
        if legacy_passed is None and "score" in row and row.get("evaluated", True):
            legacy_passed = float(row.get("score") or 0) >= 1; legacy_status = "passed" if legacy_passed else "failed"
        legacy_status = legacy_status or ("passed" if legacy_passed is True else "failed" if legacy_passed is False else "not_evaluable")
        row["case_status"] = legacy_status
        row["case_passed"] = bool(legacy_passed is True and legacy_status == "passed")
        normalized.append(row)
    if results: out["results"] = normalized
    # Missing evidence or unknown verdict is never upgraded to pass.
    # ``blocked`` 属"平台前提不满足/无法观测"，只记录不阻断准入（与 common/finalizer 口径一致）；
    # 只有 failed/error/not_evaluable 才是硬阻断。
    # 套件自带判据契约（如千问的 relay：高级面缺口只记 relay_gap）时，
    # 结论以套件权威口径为准，通用「有 failed 即不准入」会把它推翻。
    relay = out.get("relay_verdict") if isinstance(out.get("relay_verdict"), dict) else None
    if relay and relay.get("contract") == "relay":
        out["admission_passed"] = bool(relay.get("admission_passed"))
        out["verdict_contract"] = "relay; core/trust faces block, advanced faces are recorded gaps"
        return out
    # 面口径：模型能力/验真面失败才阻断，通道契约面失败只记 channel_gap。
    # 必须在下面「有 failed 即不准入」之前返回，否则通道面的 404/400 会把结论推翻。
    faces = out.get("face_verdict") if isinstance(out.get("face_verdict"), dict) else None
    if faces and faces.get("contract") == "faces":
        out["admission_passed"] = bool(faces.get("admission_passed"))
        out["verdict_contract"] = "faces; model/verify faces block, channel gaps are recorded"
        return out
    if any(r.get("case_status") in {"failed","error","not_evaluable"} for r in normalized): out["admission_passed"] = False
    out.setdefault("admission_passed", False)
    out["verdict_contract"] = "validator-derived; transport success is not a case verdict"
    return out
