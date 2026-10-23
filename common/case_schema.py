"""Case schema and validation helpers used by all suite manifests."""
from __future__ import annotations
from typing import Any
REQUIRED_CASE_KEYS={"id","name","request","expected"}

def validate_case_definition(case: dict[str,Any]) -> list[str]:
    errors=[f"missing:{key}" for key in REQUIRED_CASE_KEYS if key not in case]
    req=case.get("request") or {}; expected=case.get("expected") or {}
    if "scenario" in req and not any(k in req for k in ("messages","tools","response_format")):
        errors.append("scenario_without_executable_request")
    statuses=expected.get("http_status",expected.get("status"))
    if statuses is not None and not (isinstance(statuses,int) or isinstance(statuses,list) and all(isinstance(x,int) for x in statuses)):
        errors.append("expected.http_status_must_be_int_or_list")
    return errors

def validate_manifest(manifest: dict[str,Any]) -> dict[str,Any]:
    cases=manifest.get("cases") or []; problems=[]
    seen=set()
    for case in cases:
        cid=case.get("id")
        if cid in seen: problems.append(f"duplicate:{cid}")
        seen.add(cid); problems.extend(f"{cid}:{e}" for e in validate_case_definition(case))
    return {"valid":not problems,"total":len(cases),"problems":problems}
