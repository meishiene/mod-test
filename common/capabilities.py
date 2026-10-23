"""Six independent 100-point capability axes, projected from saved test verdicts.

This projection does not reweight the model100/channel20 total. Every objective
case belongs to one axis; identity assertions and evidence summaries earn nothing.
"""
from typing import Any, Mapping

from .score_registry import CaseMeta, normalize_kind

CAPABILITY_VERSION = "capabilities-1.1"
AXES = (
    ("protocol", "协议合规", "请求响应、流式协议、参数与错误契约"),
    ("tools", "工具调用", "工具选择、调用参数与多轮工具闭环"),
    ("thinking", "思考与推理", "思考模式、推理输出与控制；以具体用例断言为准"),
    ("context", "上下文记忆", "多轮记忆、长文本与缓存"),
    ("vision", "视觉理解", "图像内容识别、看图答题与视觉输入"),
    ("quality", "输出质量", "结构化输出、指令遵循与专项题目；不代表通用智力基准"),
)
FACE_AXIS = {"core": "protocol", "params": "protocol", "envelope": "protocol",
             "tools": "tools", "thinking": "thinking", "context": "context",
             "vision": "vision", "quality": "quality"}
# Explicit semantic corrections only: never classify by names supplied in a report.
OVERRIDES = {
    "glm": {"GLM-53-06": "quality", "GLM-53-F04": "quality",
            "GLM-53-07": "context", "GLM-53-V01": "vision"},
    "deepseek": {"DS-V4-11": "quality", "AUTH-02": "quality"},
    "qwen": {"QWEN-04": "quality", "QWEN-05": "context", "QWEN-06": "context"},
    "kvv": {"K3-JSON-001": "quality"},
}
# GLM's existing cross-model visual probe is not in its core CASE_SPECS.
# Show legacy evidence only when actually present; removed probes are not a
# permanent denominator for current GLM suites.
EXTRA = {"glm": {"GLM-53-V01": CaseMeta("GLM-53-V01", "图像内容理解（五级看图答题）", "P2", "vision")}}


def build_capabilities(test_kind: str | None, inventory: Mapping[str, CaseMeta],
                       evidence: Mapping[str, Mapping[str, Any]], excluded: set[str]) -> dict[str, Any]:
    kind = normalize_kind(test_kind)
    inv = {**{cid: meta for cid, meta in EXTRA.get(kind, {}).items() if cid in evidence}, **inventory}
    groups: dict[str, list[dict[str, Any]]] = {key: [] for key, _, _ in AXES}
    excluded_axes = set()
    for cid, meta in inv.items():
        axis = OVERRIDES.get(kind, {}).get(cid, FACE_AXIS.get(meta.face))
        if axis is None:
            continue
        if cid in excluded:
            excluded_axes.add(axis)
            continue
        item = evidence.get(cid, {})
        groups[axis].append({"case_id": cid, "name": meta.name or item.get("name") or cid,
                            "side": item.get("side", "model"),
                            "status": item.get("status", "unobserved"),
                            "credit": item.get("credit"), "priority_weight": meta.priority_weight,
                            "total_weight": item.get("total_weight", 0)})
    axes = []
    for key, label, description in AXES:
        cases = groups[key]
        pool = sum(c["priority_weight"] for c in cases)
        earned = measured = 0.0
        evaluated = 0
        for case in cases:
            weight = 100 * case.pop("priority_weight") / pool
            credit = case.pop("credit")
            case["weight"] = round(weight, 6)
            case["earned"] = None if credit is None else round(weight * credit, 6)
            case["lost"] = None if credit is None else round(weight * (1-credit), 6)
            if credit is not None:
                evaluated += 1
                measured += weight
                earned += weight * credit
        coverage = min(1.0, measured / 100)
        score = round(earned, 4) if evaluated and coverage >= .8 - 1e-6 else None
        state = ("已评估" if score is not None else "证据不足" if evaluated else
                 "本次未选" if cases and all(c["status"] == "not_selected" for c in cases) else
                 "未测试" if cases else "不适用" if key in excluded_axes else "暂无用例")
        axes.append({"key": key, "label": label, "description": description,
                     "weight": 100 if cases else 0, "earned": round(earned, 4),
                     "measured": round(measured, 4), "coverage": round(coverage, 6),
                     "score": score, "state": state, "evaluated": evaluated,
                     "case_count": len(cases), "cases": cases})
    return {"version": CAPABILITY_VERSION, "axes": axes}
