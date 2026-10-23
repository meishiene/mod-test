"""评分 v2：模型侧 100 分、通道侧 20 分，总分 120。

只计算评分，不改写原始用例状态或准入证据。未测项不赚分，保留覆盖度与区间；
信任面失败仍不形成有效评分。通道失分只影响通道侧，等级由总分决定。
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping
from math import isfinite
from kvv.cases import OPTIONAL_UNSCORED_CASES
from .capabilities import build_capabilities

from .faces import ALL_FACES, FACE_LABELS as SCOPE_LABELS
from .faces import suite_case_faces
from .score_registry import (
    EVIDENCE_FACE,
    FACE_LABELS,
    FACE_ORDER,
    FACE_WEIGHTS,
    PERFORMANCE_FACE,
    SCORING_VERSION,
    UNKNOWN_FACE,
    VETO_FACE,
    ZERO_WEIGHT_FACES,
    CaseMeta,
    face_for,
    inventory,
    not_applicable,
    relay_optional_cases,
)

# passed / warning / failed / error 是**可入分母**的裁定；其余一律算「无证据」。
CREDITS: dict[str, float] = {"passed": 1.0, "warning": 0.5, "failed": 0.0, "error": 0.0}
MISSING_STATUSES = frozenset({"blocked", "not_evaluable", "interrupted", "skipped"})
BAD_STATUSES = frozenset({"failed", "error"})

SIDE_MAXIMA = {"model": 100.0, "channel": 20.0}
MIN_COVERAGE_FOR_GRADE = 0.80
TOP_DEDUCTIONS = 8


def rating_grade(score: float | None) -> str | None:
    if score is None:
        return None
    if score > 110:
        return "官方"
    for threshold, label in ((100, "十分优秀"), (90, "优秀"), (80, "良好"),
                             (70, "一般"), (60, "可用")):
        if score >= threshold:
            return label
    return "不可用"


def _status(row: Mapping[str, Any]) -> str:
    """取该行的最终裁定；认不出就按「无证据」处理，绝不默认 passed。"""
    for key in ("case_status", "status"):
        value = row.get(key)
        if value in CREDITS or value in MISSING_STATUSES:
            return str(value)
    passed = row.get("case_passed", row.get("passed"))
    if passed is True:
        return "passed"
    if passed is False:
        return "failed"
    return "not_evaluable"


def _credit(row: Mapping[str, Any], status: str) -> float | None:
    """状态 → 得分系数；无证据返回 None（不进分母）。"""
    if status in MISSING_STATUSES:
        return None
    # 外部基准自带连续分（reward / score）时直接用比例，不被二值化抹平。
    for key in ("reward", "score"):
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value) and row.get("evaluated", True):
            return max(0.0, min(1.0, float(value)))
    return CREDITS.get(status, 0.0)


def _attribution(status: str, case_id: str, relay: Mapping[str, Any] | None,
                 optional_cases: set[str]) -> str:
    """失分归因：复用既有 relay 口径，不新增判断。"""
    if status in BAD_STATUSES and isinstance(relay, Mapping) and relay.get("contract") == "relay":
        blocking = {str(item.get("case_id")) for item in relay.get("blocking") or [] if isinstance(item, Mapping)}
        gaps = {str(item.get("case_id")) for item in relay.get("relay_gap") or [] if isinstance(item, Mapping)}
        if case_id in blocking:
            return "model"
        if case_id in gaps:
            return "channel"
    if status in BAD_STATUSES and case_id in optional_cases:
        # 套件没有把 relay 结论写进 summary（如 DeepSeek 由报告脚本侧算）时，
        # 退回到「这条用例是否属中转可选能力面」这一既有权威表。
        return "channel"
    if status == "warning":
        return "deviation"
    return "model"


def score_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    test_kind: str | None,
    model: str | None = None,
    inventory_override: Mapping[str, CaseMeta] | None = None,
    not_applicable_ids: Iterable[str] | None = None,
    relay: Mapping[str, Any] | None = None,
    scope_note: str = "",
    executed_faces: Iterable[str] | None = None,
    case_faces: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """模型侧 100 + 通道侧 20；各侧权重独立，未观测用例不赚取分数。

    分母来自完整适用清单，不因只执行部分用例而缩小。沿用面权重及优先级，
    在每侧分别归一；verify 的可计分能力属于模型侧，身份否决门保持有效。
    """
    inv = dict(inventory_override) if inventory_override is not None else inventory(test_kind, model)
    observed = {str(row.get("case_id") or row.get("id")): row for row in rows
                if isinstance(row, Mapping) and (row.get("case_id") or row.get("id"))}
    if not inv or not observed:
        return None
    excluded = set(not_applicable_ids) if not_applicable_ids is not None else not_applicable(test_kind, model)
    table = dict(suite_case_faces(test_kind))
    table.update(case_faces or {})
    extra_cases, unknown_cases = [], []
    for cid, row in observed.items():
        if cid not in inv:
            face = face_for(test_kind, cid)
            inv[cid] = CaseMeta(cid, str(row.get("name") or ""), "P2", face)
            if face == UNKNOWN_FACE:
                unknown_cases.append(cid)
            elif face not in ZERO_WEIGHT_FACES:
                extra_cases.append(cid)
    applicable = {cid: meta for cid, meta in inv.items() if cid not in excluded}
    # Explicit saved mappings win; fall back to the suite's authoritative face table.
    sides = {cid: "channel" if table.get(cid, "model") == "channel" else "model" for cid in applicable}
    active = set(executed_faces or ())
    skipped = {cid for cid in applicable if active and table.get(cid, "model") not in active}
    groups: dict[tuple[str, str], list[str]] = {}
    for cid, meta in applicable.items():
        if FACE_WEIGHTS.get(meta.face, 0):
            groups.setdefault((sides[cid], meta.face), []).append(cid)
    if not groups:
        return None
    weights: dict[str, float] = {}
    for side, maximum in SIDE_MAXIMA.items():
        pool = sum(FACE_WEIGHTS[face] for group_side, face in groups if group_side == side)
        for (group_side, face), ids in groups.items():
            if group_side != side:
                continue
            priority = sum(applicable[cid].priority_weight for cid in ids)
            for cid in ids:
                weights[cid] = maximum * FACE_WEIGHTS[face] / pool * applicable[cid].priority_weight / priority
    counts: dict[str, int] = {}
    faces: dict[tuple[str, str], dict[str, Any]] = {}
    components = {side: {"max_score": maximum, "earned": 0.0, "measured_weight": 0.0,
                         "scope_weight": sum(w for cid, w in weights.items() if sides[cid] == side),
                         "skipped_weight": 0.0, "counts": {}}
                  for side, maximum in SIDE_MAXIMA.items()}
    deductions, veto_failed = [], []
    case_scores = {}
    capability_evidence = {}
    for cid, meta in applicable.items():
        side = sides[cid]
        component = components[side]
        weight = weights.get(cid, 0.0)
        stats = faces.setdefault((side, meta.face), {
            "side": side, "face": meta.face, "label": FACE_LABELS.get(meta.face, meta.face),
            "weight": 0.0, "measured_weight": 0.0, "earned": 0.0, "cases": 0, "counts": {}})
        stats["weight"] += weight
        stats["cases"] += 1
        row = observed.get(cid)
        status = "not_selected" if cid in skipped else "unobserved" if row is None else _status(row)
        contribution = {"case_id": cid, "side": side, "status": status,
                        "weight": round(weight, 8), "earned": None, "lost": None,
                        "veto_failed": meta.face == VETO_FACE and status in BAD_STATUSES}
        case_scores[cid] = contribution
        capability_evidence[cid] = {"status": status, "side": side,
                                    "name": meta.name, "total_weight": round(weight, 8),
                                    "credit": None}
        for counter in (counts, component["counts"], stats["counts"]):
            counter[status] = counter.get(status, 0) + 1
        if status == "not_selected":
            component["skipped_weight"] += weight
        if status in {"not_selected", "unobserved"}:
            continue
        credit = _credit(row, status)
        capability_evidence[cid]["credit"] = credit
        if credit is None:
            continue
        contribution.update(earned=round(weight * credit, 8), lost=round(weight * (1-credit), 8))
        if meta.face == VETO_FACE and status in BAD_STATUSES:
            veto_failed.append(cid)
        component["measured_weight"] += weight
        component["earned"] += weight * credit
        stats["measured_weight"] += weight
        stats["earned"] += weight * credit
        if weight and credit < 1:
            deductions.append({"case_id": cid, "name": meta.name or str(row.get("name") or ""),
                               "side": side, "face": meta.face, "status": status,
                               "points": round(weight * (1-credit), 4),
                               "attribution": _attribution(status, cid, relay, relay_optional_cases(test_kind)),
                               "detail": str(row.get("details") or row.get("error") or "")[:240]})
    if all(c["scope_weight"] - c["skipped_weight"] < 1e-9 for c in components.values()):
        return None
    notes = []
    for side, component in components.items():
        maximum = component["max_score"]
        measured = component["measured_weight"]
        scope = component["scope_weight"]
        selected = scope - component["skipped_weight"]
        component["score"] = round(min(maximum, component["earned"]), 4) if measured > 0 else None
        component["coverage"] = round(measured / scope, 4) if scope else 0.0
        component["status"] = ("not_applicable" if not scope else "not_selected" if selected < 1e-9
                               else "unmeasured" if not measured else "partial" if measured < scope - 1e-7 else "measured")
        component["range"] = [round(component["earned"], 4), round(min(maximum, component["earned"] + scope - measured), 4)]
        component["partial"] = bool(selected > 1e-9 and component["coverage"] < MIN_COVERAGE_FOR_GRADE)
        label = "模型侧" if side == "model" else "通道侧"
        if component["status"] != "measured":
            notes.append(f"{label}：{'本次未选' if component['status']=='not_selected' else '无适用计分项' if not scope else '尚未完整覆盖'}；未测项不计得分，保留覆盖度与可能区间")
        for key in ("earned", "measured_weight", "scope_weight", "skipped_weight"):
            component[key] = round(component[key], 4)
    face_rows = []
    for (side, face), stats in faces.items():
        measured, weight = stats["measured_weight"], stats["weight"]
        stats["ratio"] = round(stats["earned"] / weight, 4) if weight and measured else None
        if face in ZERO_WEIGHT_FACES:
            stats["status"] = ("veto_failed" if veto_failed else "veto_passed") if face == VETO_FACE else "performance" if face == PERFORMANCE_FACE else "evidence"
        elif not measured:
            stats["status"] = "not_selected" if stats["counts"].get("not_selected") == stats["cases"] else "untested" if stats["counts"].get("unobserved") == stats["cases"] else "unevaluated"
        else:
            stats["status"] = "full" if stats["earned"] >= weight - 1e-7 else "gap"
        for key in ("weight", "earned", "measured_weight"):
            stats[key] = round(stats[key], 4)
        face_rows.append(stats)
    for face in face_rows:
        if face["weight"] and face["measured_weight"] < face["weight"] - 1e-7:
            notes.append(f"{'模型侧' if face['side']=='model' else '通道侧'} · {face['label']}：部分用例尚无有效证据")
    face_rows.sort(key=lambda f: (f["side"] != "model", [*FACE_ORDER, *sorted(ZERO_WEIGHT_FACES)].index(f["face"])))
    model_component = components["model"]
    valid = not veto_failed
    raw_score = sum(c["earned"] for c in components.values()) if model_component["measured_weight"] else None
    score = round(min(120.0, raw_score), 4) if raw_score is not None and valid else None
    partial = any(c["partial"] for c in components.values()) or not model_component["measured_weight"]
    grade = rating_grade(score) if score is not None and not partial else None
    if extra_cases:
        notes.append("跨套件探针已按所属面计入评分：" + "、".join(sorted(extra_cases)))
    if unknown_cases:
        notes.append("未登记用例不参与评分：" + "、".join(sorted(unknown_cases)))
    if scope_note:
        notes.append(scope_note)
    counts.update(not_applicable=len(set(inv) & excluded), extra_cases=len(extra_cases), unknown=len(unknown_cases))
    deductions.sort(key=lambda d: d["points"], reverse=True)
    scope_weight = sum(c["scope_weight"] for c in components.values())
    measured_weight = sum(c["measured_weight"] for c in components.values())
    skipped_weight = sum(c["skipped_weight"] for c in components.values())
    return {"scoring_version": SCORING_VERSION, "test_kind": test_kind, "model": model,
            "valid": valid, "invalid_reason": "模型一致性（信任面）存在失败项：" + "、".join(sorted(veto_failed)) if veto_failed else None,
            "score": score, "raw_score": raw_score, "max_score": 120,
            "model_score": model_component["score"], "channel_score": components["channel"]["score"],
            "components": components, "grade": grade,
            "rating_passed": score >= 60 if grade is not None else None,
            "gap_to_official": round(120-score, 4) if score is not None else None,
            "coverage": round(measured_weight / scope_weight, 4) if scope_weight else 0,
            "scope_weight": round(scope_weight, 4), "executed_weight": round(scope_weight-skipped_weight, 4),
            "skipped_weight": round(skipped_weight, 4), "skipped_cases": len(skipped), "executed_faces": sorted(active),
            "uncovered_weight": round(120-scope_weight, 4), "untested_faces": [], "partial": partial,
            "range": [round(sum(c["range"][0] for c in components.values()), 4), round(sum(c["range"][1] for c in components.values()), 4)],
            "caps": [], "counts": counts, "faces": face_rows, "deductions": deductions[:TOP_DEDUCTIONS],
            "deduction_total": round(sum(d["points"] for d in deductions), 4), "notes": notes,
            "weights": dict(SIDE_MAXIMA), "case_scores": case_scores,
            "capabilities": build_capabilities(test_kind, inv, capability_evidence, excluded)}


def score_summary(summary: Mapping[str, Any]) -> dict[str, Any] | None:
    """从单次运行的 summary 取裁定并打分；缺少套件/结果时返回 None（不出分）。

    批量聚合的 summary（`record_type=batch`）里 `results` 是子记录而不是用例行，
    必须走 `score_batch`；多型号批次没有「一个分数」，返回 None，由调用方决定怎么展示。
    """
    if not isinstance(summary, Mapping):
        return None
    # A bounded evidence workload is not a complete legacy 120-point test scope.
    if summary.get("economy_validation") or (summary.get("case") or summary.get("case_name")) in OPTIONAL_UNSCORED_CASES:
        return None
    rows = summary.get("results") or summary.get("records") or []
    if not isinstance(rows, list) or not rows:
        return None
    if str(summary.get("record_type") or "") == "batch" or summary.get("is_batch"):
        reports = score_batch([item for item in rows if isinstance(item, Mapping)])
        return reports[0] if len(reports) == 1 else None
    return score_rows(rows, test_kind=summary.get("test_kind"), model=summary.get("model"),
                      relay=summary.get("relay_verdict") if isinstance(summary.get("relay_verdict"), Mapping) else None,
                      executed_faces=_faces_of(summary),
                      case_faces=summary.get("case_faces") if isinstance(summary.get("case_faces"), Mapping) else None)


def _faces_of(summary: Mapping[str, Any]) -> list[str] | None:
    """summary 里记录的「本次执行了哪些测试面」；没有记录返回 None（按全量口径打分）。"""
    faces = summary.get("executed_faces")
    if isinstance(faces, str):
        faces = [part for part in faces.split(",") if part.strip()]
    if not isinstance(faces, (list, tuple)) or not faces:
        return None
    return [str(face) for face in faces]


def score_batch(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """批量（多卡片 / 多型号）评分：同一型号跨卡片合并，复测取后出现的那次。"""
    grouped: dict[str, dict[str, Any]] = {}
    for item in items:
        summary = item.get("summary") or {}
        if not isinstance(summary, Mapping):
            continue
        if (summary.get("economy_validation")
                or (summary.get("case") or summary.get("case_name") or item.get("case_name")) in OPTIONAL_UNSCORED_CASES):
            continue
        model = str(item.get("model") or summary.get("model") or "")
        test_kind = str(summary.get("test_kind") or item.get("test_kind") or "")
        bucket = grouped.setdefault(f"{test_kind}|{model}", {"test_kind": test_kind, "model": model,
                                                            "rows": {}, "relay": None,
                                                            "faces": set(), "case_faces": {}, "unknown_scope": False})
        if bucket["relay"] is None and isinstance(summary.get("relay_verdict"), Mapping):
            # 归因（模型侧 / 通道侧）尽量沿用子任务自己的 relay 结论。
            bucket["relay"] = summary["relay_verdict"]
        # 批次内只要有一张卡跑了某个面，就把该面算进执行范围（保守：宁可少划「未执行」）。
        faces = _faces_of(summary)
        if faces:
            bucket["faces"].update(faces)
        else:
            bucket["unknown_scope"] = True
        if isinstance(summary.get("case_faces"), Mapping):
            bucket["case_faces"].update(summary["case_faces"])
        for row in summary.get("results") or []:
            if isinstance(row, Mapping) and (row.get("case_id") or row.get("id")):
                bucket["rows"][str(row.get("case_id") or row.get("id"))] = row
    reports: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        report = score_rows(bucket["rows"].values(), test_kind=bucket["test_kind"], model=bucket["model"] or None,
                            relay=bucket["relay"],
                            executed_faces=None if bucket["unknown_scope"] else sorted(bucket["faces"]) or None,
                            case_faces=bucket["case_faces"] or None)
        if report:
            reports.append(report)
    return reports


# ---- 报告渲染 --------------------------------------------------------------
FACE_STATE_LABELS = {
    "full": "满分", "gap": "有失分", "unevaluated": "无证据", "untested": "未覆盖",
    "not_selected": "本次未选（按测试面）",
    "veto_passed": "已通过", "veto_failed": "未通过 ⇒ 评分作废",
    "evidence": "仅证据", "performance": "不计分",
}
ATTRIBUTION_LABELS = {"model": "模型侧", "channel": "通道侧", "deviation": "偏差未定责"}


def headline(report: Mapping[str, Any]) -> str:
    if not report:
        return "未评分"
    if not report.get("valid"):
        return f"评分不成立：{report.get('invalid_reason') or '模型一致性存在失败项'}"
    score = report.get("score")
    if score is None:
        return "无有效模型侧证据，不出总分"
    model = report.get("model_score")
    channel = report.get("channel_score")
    sides = f"模型侧 {model if model is not None else '未测'} / 100，通道侧 {channel if channel is not None else '未测'} / 20"
    if report.get("partial"):
        low, high = report.get("range") or [None, None]
        interval = f"，可能区间 {low:g}–{high:g}" if isinstance(low, (int, float)) and isinstance(high, (int, float)) else ""
        return f"{score:g} / 120（仅已测面得分；{sides}；证据覆盖 {report.get('coverage', 0)*100:.0f}%{interval}，不出等级）"
    return f"{score:g} / 120（{report.get('grade') or '待评分'}；{sides}；距满分 {report.get('gap_to_official')}）"


def render_markdown(report: Mapping[str, Any] | None, *, max_deductions: int = 5) -> str:
    """把评分写成报告章节（单项报告与聚合报告共用同一实现）。"""
    if not report:
        return ""
    lines = ["## 评分", "", f"- 结论：{headline(report)}"]
    lines.append(f"- 口径：`{report.get('scoring_version')}`；证据覆盖度 "
                 f"{report.get('coverage', 0) * 100:.0f}%；可测面 {report.get('scope_weight')} / 120 分权重")
    if report.get("skipped_cases"):
        lines.append(f"- 评分范围：本次执行 {report.get('executed_weight')} / {report.get('scope_weight')} 分权重；"
                     f"{report.get('skipped_cases')} 条用例所属面未勾选"
                     f"（{report.get('skipped_weight')} 分未计得分），未测项保留为证据缺口")
    low, high = report.get("range") or [None, None]
    if isinstance(low, (int, float)) and isinstance(high, (int, float)) and high > low:
        lines.append(f"- 真实区间：{low:g} – {high:g}（未观测部分按全错/全对两端夹逼）")
    for cap in report.get("caps") or []:
        lines.append(f"- 封顶：{cap}")
    for note in report.get("notes") or []:
        lines.append(f"- 说明：{note}")
    lines += ["", "| 侧别 | 计分面 | 权重 | 得分率 | 得分 | 状态 |", "|---|---|---:|---:|---:|---|"]
    for face in report.get("faces") or []:
        if not face.get("weight") and face.get("status") not in {"evidence", "performance", "veto_passed", "veto_failed"}:
            continue
        ratio = face.get("ratio")
        lines.append("| {side} | {label} | {weight:g} | {ratio} | {earned:g} | {state} |".format(
            side="通道侧" if face.get("side")=="channel" else "模型侧", label=face.get("label"), weight=float(face.get("weight") or 0),
            ratio="—" if ratio is None else f"{ratio * 100:.1f}%",
            earned=float(face.get("earned") or 0),
            state=FACE_STATE_LABELS.get(str(face.get("status")), str(face.get("status")))))
    deductions = report.get("deductions") or []
    if deductions:
        lines += ["", f"### 失分项（Top {min(max_deductions, len(deductions))}）", "",
                  "| 用例 | 计分面 | 状态 | 失分 | 归因 | 证据 |", "|---|---|---|---:|---|---|"]
        for item in deductions[:max_deductions]:
            detail = str(item.get("detail") or "—").replace("|", "\\|").replace("\n", " ")[:120]
            lines.append(f"| {item.get('case_id')} {item.get('name') or ''} | {FACE_LABELS.get(str(item.get('face')), item.get('face'))} "
                         f"| {item.get('status')} | {float(item.get('points') or 0):g} | "
                         f"{ATTRIBUTION_LABELS.get(str(item.get('attribution')), item.get('attribution'))} | {detail} |")
    return "\n".join(lines) + "\n"
