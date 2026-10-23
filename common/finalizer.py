"""Central finalization for every test suite.

The web orchestrator calls this for load, strict, KVV, DeepSeek, GLM,
moderation and external benchmarks alike.  It preserves domain-specific
metrics while adding one common verdict/evidence contract.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .artifacts import SCHEMA_VERSION, write_json
from .report_style import is_image_summary, unify_report
from .result_adapter import normalize_summary
from .scoring import score_summary

# 硬阻断集合：这些裁定表示"用例真的跑出了坏结论"（失败/错误/无法评估/中断）。
BLOCKING = {"failed", "error", "not_evaluable", "interrupted"}
# 非阻断裁定：``blocked``/``skipped`` 表示"平台前提不满足、无法观测"，**只记录不阻断准入**
# （2026-10-10 口径：脱离官方理想契约、按被测通道实际行为判定；中转下 blocked 必然存在，
#  例如账号层拒绝 previous_response_id、端点未代理、取证前提落空——这些都不是模型能力结论）。
# 但"整批只有非阻断裁定"= 没有任何有效结论（passed/warning），准入无依据，仍不准入，
# 否则"网关完全不可达"也会被误判为通过。
EFFECTIVE_VERDICTS = {"passed", "warning"}


def _unify_report_document(result_dir: Path, summary: dict[str, Any]) -> None:
    """把单项测试报告改写成聚合报告同款版式（图片类套件保留自己的报告）。

    报告正文（用例表、判定、核心统计等章节）原样保留，只重写报告头并补统一指标表；
    重复 finalize 时通过 `unify_report` 的幂等逻辑覆盖上次生成的段落。
    """
    if is_image_summary(summary):
        return
    report_path = result_dir / "report.md"
    try:
        existing = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
        report_path.write_text(unify_report(existing, summary, result_dir), encoding="utf-8")
    except OSError:
        # 报告版式只是展示层，写失败不能影响准入判定与证据落盘。
        return

def _check(name: str, passed: bool | None, expected: Any, actual: Any, detail: str = "") -> dict[str, Any]:
    return {"name":name,"passed":passed,"expected":expected,"actual":actual,"detail":detail}

def _suite_checks(summary: dict[str, Any]) -> list[dict[str, Any]]:
    checks=[]
    for name,item in (summary.get("checks") or {}).items():
        if isinstance(item,dict): checks.append(_check(name,item.get("passed"),{k:v for k,v in item.items() if k not in {"passed","value"}},item.get("value"),"领域指标校验"))
    results=summary.get("results") or []
    if results:
        statuses=[r.get("case_status",r.get("status")) for r in results if isinstance(r,dict)]
        relay=summary.get("relay_verdict") if isinstance(summary.get("relay_verdict"),dict) else None
        faces=summary.get("face_verdict") if isinstance(summary.get("face_verdict"),dict) else None
        if relay and relay.get("contract")=="relay":
            # 中转口径：只有阻断面（核心/信任）的失败算硬阻断，高级面缺口（relay_gap）只记录；
            # 「矩阵声明未开放的能力」整批被拒也不算没有有效结论。
            hard_blocked=bool(relay.get("blocking"))
            has_effective=bool(relay.get("evaluated")) or bool(relay.get("vision_undeclared"))
            detail=("核心/信任面存在失败" if hard_blocked
                    else relay.get("summary_line") or "按中转接入口径无阻断项")
        elif faces and faces.get("contract")=="faces":
            # 面口径：模型能力面 / 验真面的失败才阻断；通道契约面（网关端点、第二协议、
            # beta、参数校验）的失败只记 channel_gap——中转必然存在，不是模型不行。
            hard_blocked=bool(faces.get("blocking"))
            has_effective=bool(faces.get("evaluated"))
            detail=(faces.get("summary_line") or "按测试面口径无阻断项")
        else:
            hard_blocked=any(x in BLOCKING for x in statuses)
            has_effective=any(x in EFFECTIVE_VERDICTS for x in statuses)
            detail="存在硬失败或不可评估案例" if hard_blocked else "整批均未取得有效结论（全为 blocked/skipped），准入无依据"
        checks.append(_check("case_verdicts_complete", not hard_blocked and has_effective,
                             "无 failed/error/not_evaluable/interrupted，且至少一条有效结论（passed/warning）；blocked/skipped 仅记录",
                             statuses, detail))
    suites=summary.get("suites") or []
    if suites:
        checks.append(_check("official_suite_exit",all(x.get("exit_code")==0 and not x.get("failed") and not x.get("errors") for x in suites),"所有官方套件 exit=0 且无失败/错误",[{k:x.get(k) for k in ("suite","exit_code","failed","errors")} for x in suites],"官方测试套件失败"))
    # External verifier suites must have actual scored/evaluated evidence.
    count_keys=("scored_tasks","evaluated_tasks","evaluated","total_problems")
    present=[(k,summary.get(k)) for k in count_keys if k in summary]
    for key,value in present:
        if key in {"scored_tasks","evaluated_tasks","evaluated"}: checks.append(_check(f"{key}_nonzero",isinstance(value,(int,float)) and value>0,"> 0",value,"没有可评分证据"))
    if "sent" in summary: checks.append(_check("formal_samples_nonzero",int(summary.get("sent") or 0)>0,"> 0",summary.get("sent"),"正式样本为 0；warmup 不能替代正式指标"))
    return checks

def finalize_result_dir(result_dir: Path) -> dict[str, Any]:
    path=result_dir/"summary.json"
    summary=normalize_summary(json.loads(path.read_text(encoding="utf-8")))
    checks=_suite_checks(summary)
    evidence=[]
    for name in ("execution.jsonl","requests.jsonl","generated_requests.jsonl","requests.csv","pytest.log","run.log","connectivity.json","metadata.json"):
        p=result_dir/name
        if p.is_file(): evidence.append({"path":name,"bytes":p.stat().st_size,"nonempty":p.stat().st_size>0})
    evidence_ok=bool(evidence) and any(x["nonempty"] for x in evidence)
    checks.append(_check("evidence_present",evidence_ok,"至少一份非空请求/执行证据",evidence,"缺少执行证据"))
    blocking=[x for x in checks if x.get("passed") is not True]
    original=bool(summary.get("admission_passed"))
    admission=original and not blocking
    validation={"schema_version":SCHEMA_VERSION,"status":"passed" if admission else "failed" if any(x.get("passed") is False for x in checks) else "not_evaluable","passed":admission,"original_admission_passed":original,"checks":checks,"evidence":evidence,"rule":"只有领域校验与统一证据校验全部通过才允许准入；HTTP/进程退出成功不能单独构成案例通过"}
    summary["schema_version"]=SCHEMA_VERSION; summary["validation"]=validation; summary["admission_passed"]=admission
    summary.setdefault("record_id", result_dir.name)
    # Versioned 120-point rating; original admission and validation remain raw evidence.
    summary["score"] = score_summary(summary)
    summary["rating_passed"] = (summary["score"] or {}).get("rating_passed")
    write_json(path,summary); write_json(result_dir/"validation.json",validation)
    if summary.get("score"):
        write_json(result_dir/"score.json", summary["score"])
    _unify_report_document(result_dir, summary)
    return summary
