"""千问（Qwen）**中转接入口径**——本套件对外的唯一准入权威。

为什么需要它
------------
官方口径逐条对齐百炼文档：任何契约点不达标（`tool_choice=required` 不强制、
`top_p` 越界不报错、思考模式 `max_tokens` 上限不生效……）都算不合格。
但实测链路是**中转网关 + 分组 domestic**（上游百炼），网关会主动收窄能力面、
也会在厂商静默升级后改变行为：把「网关没实现」读成「模型不行」，会把可用型号误判掉。

分层原则（与 `dsv4/relay_profile.py` 同一套思想）
------------------------------------------------
- **core 面**（基础对话/协议/Usage/流式/长上下文）：failed ⇒ **阻断**。这是接进来就要能用的底线。
- **trust 面**（响应 model 回显为异族模型）：failed ⇒ **阻断**。这已不是能力问题，是「测的不是这个模型」。
- **高级面**（工具、思考参数、档位指纹、错误信封、视觉）：failed / blocked ⇒ 记 `relay_gap`
  只留痕不阻断——中转丢弃 `tool_choice`、不认思考预算、不给视觉输入，都不代表纯文本能力不可用。
- **视觉**额外按**矩阵声明**分流：`QwenModel.image_input=False` 的型号（如 `qwen3.7-max`）
  视觉被拒属 `vision_undeclared`，不阻断；声明支持却整批被拒，同样只记 gap（网关侧收窄）。

准入 = 无 blocking 用例，且整批至少有 1 条 passed/warning（全 blocked 视为没有有效结论）。
`official_*` 字段保留官方严格结论，便于对照，不参与准入。
"""
from __future__ import annotations

from typing import Any, Iterable

from .cases import model_spec

BLOCKING_FACES = ("core", "trust")

# 用例 → 判据面的映射。新增用例必须在这里登记，否则测试会失败（见 tests/test_qwen.py）。
CASE_FACES: dict[str, str] = {
    # 核心面：接进来就要能用的文字/协议能力
    "QWEN-01": "core", "QWEN-02": "core", "QWEN-03": "core", "QWEN-04": "core",
    "QWEN-05": "core", "QWEN-06": "core", "QWEN-07": "core", "QWEN-08": "core",
    # 信任面：响应模型被换成别族/别款
    "QWEN-F01": "trust",
    # 高级面：中转常丢弃或收窄的能力
    "QWEN-T01": "thinking", "QWEN-T02": "thinking", "QWEN-T03": "thinking",
    "QWEN-T04": "thinking", "QWEN-T05": "thinking", "QWEN-T06": "thinking",
    "QWEN-A01": "tools", "QWEN-A02": "strict_tool", "QWEN-A03": "tools",
    "QWEN-A04": "tools", "QWEN-A05": "tools",
    "QWEN-F02": "fingerprint", "QWEN-F03": "fingerprint", "QWEN-F04": "params",
    "QWEN-F05": "envelope", "QWEN-F06": "fingerprint",
    "QWEN-V01": "vision",
}

FACE_LABELS = {
    "core": "核心对话面", "trust": "模型一致性（信任级）", "thinking": "思考参数面",
    "tools": "工具调用面", "strict_tool": "强制工具语义（Strict Tool）", "fingerprint": "档位/身份指纹",
    "params": "采样参数取值域", "envelope": "错误信封", "vision": "视觉输入",
}

# 高级面不阻断，但要写清「网关侧缺口」的人话说明，报告直接引用。
FACE_NOTES = {
    "thinking": "思考参数（enable_thinking / thinking_budget / 上限契约）在中转侧可能被丢弃或改写，属能力收窄",
    "tools": "工具调用面由网关转发策略决定，缺失不代表纯文本通道不可用",
    "strict_tool": "OpenAI 严格语义（tool_choice=required 必须强制调用）在中转侧普遍被降级为 auto",
    "fingerprint": "档位/窗口自述长期不可靠，只作留痕，不参与准入",
    "params": "采样参数取值域由网关自校验，越界是否报错不构成模型能力结论",
    "envelope": "错误信封格式与错误码映射由网关决定，不构成模型能力结论",
    "vision": "视觉输入是否透传由网关 / 上游版本决定，按矩阵声明能力分流",
}


from common.model_validation import PROBE_DEFINITIONS
CASE_FACES.update({f"QWEN-{suffix}": "trust" if suffix.startswith("R") else "core"
                   for suffix, *_ in PROBE_DEFINITIONS})

def face_of(case_id: str) -> str:
    return CASE_FACES.get(case_id, "unknown")


def relay_verdict(results: Iterable[Any], model: str | None = None) -> dict[str, Any]:
    """按中转接入口径给出结论。

    `results` 可以是 `CaseResult` 对象列表，也可以是 summary.json 里的 dict 列表。
    """
    rows: list[dict[str, Any]] = []
    for item in results:
        get = item.get if isinstance(item, dict) else lambda key, default=None, _i=item: getattr(_i, key, default)
        case_id = get("case_id") or ""
        status = get("status") or ""
        rows.append({"case_id": case_id, "name": get("name") or "", "status": status,
                     "face": face_of(case_id), "details": get("details") or ""})

    def bad(status: str) -> bool:
        return status in {"failed", "error", "not_evaluable", "interrupted"}

    blocking = [row for row in rows if bad(row["status"]) and row["face"] in BLOCKING_FACES]
    gaps = [row for row in rows if row["status"] in {"failed", "error", "blocked",
                                                    "not_evaluable", "interrupted"}
            and not (bad(row["status"]) and row["face"] in BLOCKING_FACES)]
    evaluated = [row for row in rows if row["status"] in {"passed", "warning"} and row["case_id"] != "QWEN-F06"]

    declared_vision = True
    spec = model_spec(model or "") if model else None
    if spec is not None:
        declared_vision = bool(spec.image_input)
    vision_undeclared = [row for row in rows
                         if row["face"] == "vision" and not declared_vision
                         and row["status"] in {"blocked", "failed", "error"}]
    # 「整批都是矩阵声明未开放的能力」→ 没有适用用例，属**不适用**，不是不准入。
    # （例如单跑 QWEN_VISION 验 qwen3.7-max：矩阵已声明 image_input=False。）
    only_undeclared = bool(rows) and len(vision_undeclared) == len(rows)

    passed = not blocking and (bool(evaluated) or only_undeclared)
    if blocking:
        head = "核心面存在失败：" + "、".join(f"{row['case_id']}（{row['face']}）" for row in blocking)
    elif only_undeclared:
        head = (f"视觉未开放（矩阵声明 {spec.model_id if spec else '该型号'} 无图像输入），"
                "本批无适用用例，不构成阻断")
    elif not evaluated:
        head = "整批无 passed/warning，没有有效结论"
    else:
        gap_faces = sorted({row["face"] for row in gaps})
        warnings = [row for row in rows if row["status"] == "warning"]
        if gaps:
            head = "核心面与模型一致性面无失败；高级面记录 " + str(len(gaps)) + " 项缺口（" + \
                   "、".join(FACE_LABELS.get(face, face) for face in gap_faces) + "），不阻断准入"
        elif warnings:
            head = f"核心面与模型一致性面无失败；{len(warnings)} 项 warning 仅留痕，不阻断准入"
        else:
            head = "核心面与模型一致性面无失败；全部用例通过"
        if vision_undeclared:
            head += f"；视觉按矩阵声明记为未开放（{spec.model_id if spec else '该型号'} 声明无图像输入）"

    return {
        "contract": "relay",
        "admission_passed": passed,
        "blocking": blocking,
        "relay_gap": gaps,
        "vision_undeclared": [row["case_id"] for row in vision_undeclared],
        "evaluated": len(evaluated),
        "summary_line": head,
        "face_breakdown": _face_breakdown(rows),
    }


def _face_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = ["core", "trust", "thinking", "tools", "strict_tool", "vision",
             "fingerprint", "params", "envelope", "unknown"]
    out: list[dict[str, Any]] = []
    for face in order:
        bucket = [row for row in rows if row["face"] == face]
        if not bucket:
            continue
        out.append({
            "face": face,
            "label": FACE_LABELS.get(face, face),
            "blocking_face": face in BLOCKING_FACES,
            "counts": {name: sum(row["status"] == name for row in bucket)
                       for name in ("passed", "warning", "blocked", "failed", "error")},
            "verdict": "阻断" if (face in BLOCKING_FACES
                                  and any(_is_bad(row["status"]) for row in bucket)) else "记录",
            "notes": FACE_NOTES.get(face, ""),
        })
    return out


def _is_bad(status: str) -> bool:
    return status in {"failed", "error", "not_evaluable", "interrupted"}
