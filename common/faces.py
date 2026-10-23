"""用例「面」分层：模型能力 / 模型验真 / 通道契约。

为什么要分面
------------
同一批用例里混了两类命题：

- **模型面**（``model``）：这个模型会不会思考、能不能读图、代码写不写得对、
  长上下文扛不扛得住 —— 换一家中转也不该变的东西。
- **验真面**（``verify``）：响应对不对得上这个模型 —— 身份回显、指纹挑战、
  思考真实性、注入识别。测的是「有没有被偷换」，不是「会不会做题」。
- **通道面**（``channel``）：网关有没有实现某个端点或严格语义 —— Anthropic/Responses
  第二条协议、Files、余额、/models、countTokens、beta 特性、参数越界是否报错、
  错误信封长什么样。

中转接入时通道面几乎必然大面积不通过（网关只承诺 Chat Completions 主面），
把「网关少实现几个端点」记成「模型不行」会直接毁掉准入结论。因此：

1. **默认只执行 model + verify**（``DEFAULT_FACES``），通道面要显式勾选才跑
   （界面「执行范围」/ CLI ``--faces`` / 环境变量 ``VOLC_FACES``）；
2. 即使勾选执行，**通道面的 failed/error/blocked 也只记 ``channel_gap``，永不阻断准入**
   （``face_verdict``）。原始 status 原样留证，不压平成 warning。

登记约定
--------
每个套件在 ``<suite>/faces.py`` 里给出 ``CASE_FACES: {case_id: face}``，覆盖全部
``CASE_SPECS``。**未登记的用例按 ``model`` 处理**——宁可多跑一条，也不能因为漏登记
把模型能力用例静默跳过（``tests/test_faces.py`` 会钉死登记完整性）。
"""
from __future__ import annotations

import importlib
import os
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

FACE_MODEL = "model"
FACE_VERIFY = "verify"
FACE_CHANNEL = "channel"

FACE_LABELS: dict[str, str] = {
    FACE_MODEL: "模型能力",
    FACE_VERIFY: "模型验真",
    FACE_CHANNEL: "通道契约",
}

FACE_NOTES: dict[str, str] = {
    FACE_MODEL: "模型自身能力（推理/代码/长上下文/工具闭环/视觉/缓存命中），与接入通道无关",
    FACE_VERIFY: "身份与指纹取证：响应模型是否被替换、思考是否真实、系统指令是否被注入",
    FACE_CHANNEL: "网关端点实现与严格语义（第二协议/beta/端点/参数校验/错误信封）；"
                  "中转常不承诺，失败只记通道缺口，不构成模型能力结论",
}

#: 默认执行面：模型能力 + 模型验真。通道面默认不跑，避免大量 404/400 噪声淹没结论。
DEFAULT_FACES: tuple[str, ...] = (FACE_MODEL, FACE_VERIFY)
ALL_FACES: tuple[str, ...] = (FACE_MODEL, FACE_VERIFY, FACE_CHANNEL)
BLOCKING_FACES: frozenset[str] = frozenset({FACE_MODEL, FACE_VERIFY})

FACES_ENV = "VOLC_FACES"

#: 通道面里这些裁定算「缺口」（记录）；其余裁定（passed/warning）正常统计。
_GAP_STATUSES = frozenset({"failed", "error", "blocked", "not_evaluable", "interrupted"})
_HARD_STATUSES = frozenset({"failed", "error", "not_evaluable", "interrupted"})
_EFFECTIVE_STATUSES = frozenset({"passed", "warning"})


class FaceSelectionError(ValueError):
    """面选择非法（拼错的面名会让整批静默空跑，必须拦回）。"""


def parse_faces(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    """把 ``"model,verify"`` / ``["model"]`` / ``None`` 解析成面元组。

    - ``None`` 或空串 → ``DEFAULT_FACES``（默认不跑通道面）
    - ``all`` / ``*`` → ``ALL_FACES``
    - 未知面名 → 抛 ``FaceSelectionError``（不静默丢弃）
    """
    if raw is None:
        return DEFAULT_FACES
    if isinstance(raw, str):
        text = raw.strip().lower()
        if not text:
            return DEFAULT_FACES
        if text in {"all", "*"}:
            return ALL_FACES
        items = [part.strip().lower() for part in text.split(",") if part.strip()]
    else:
        items = [str(part).strip().lower() for part in raw if str(part).strip()]
    if not items:
        return DEFAULT_FACES
    unknown = [item for item in items if item not in ALL_FACES]
    if unknown:
        raise FaceSelectionError(
            f"未知的测试面：{', '.join(unknown)}；可选 {', '.join(ALL_FACES)} 或 all"
        )
    # 固定顺序输出，避免同一组选择在不同入口产生不同的 summary 文本。
    return tuple(face for face in ALL_FACES if face in items)


def selected_faces(raw: str | Sequence[str] | None = None) -> tuple[str, ...]:
    """当前生效的面：显式入参 > 环境变量 ``VOLC_FACES`` > 默认集。"""
    if raw is not None:
        return parse_faces(raw)
    return parse_faces(os.environ.get(FACES_ENV))


def face_of(table: Mapping[str, str], case_id: str) -> str:
    """查用例所属面；未登记按 ``model`` 处理（宁可多跑，不可漏跑）。"""
    return table.get(case_id) or FACE_MODEL


def face_enabled(face: str, faces: Sequence[str] | None = None) -> bool:
    return face in (faces if faces is not None else DEFAULT_FACES)


#: 套件名（``summary.test_kind``）→ 面登记模块。放这里而不是各套件自述，
#: 是为了让评分/聚合能在「只有 summary、没有套件进程」时也查出面归属。
_SUITE_FACE_MODULES: dict[str, str] = {
    "deepseek": "dsv4.faces",
    "qwen": "qwen.faces",
    "glm": "glm53.faces",
    "gpt": "gpt56.faces",
    "fable": "fable5.faces",
    "gemini": "gemini3.faces",
    "kvv": "kvv.faces",
}

#: ``test_kind`` 的历史别名（与 ``common.score_registry._KIND_ALIASES`` 同口径）。
_SUITE_ALIASES: dict[str, str] = {
    "deepseek": "deepseek", "ds": "deepseek",
    "glm_stream_retest": "glm",
    "fable": "fable", "awsb": "fable",
}


@lru_cache(maxsize=None)
def suite_case_faces(test_kind: str | None) -> Mapping[str, str]:
    """取某套件的 ``{case_id: face}`` 表；套件未登记面信息时返回空表。

    导入是**惰性**的：各套件的 ``faces.py`` 自己也会 import 本模块，顶层导入会成环。
    失败一律吞掉返回空表——查不到面归属只影响「哪条用例算未执行」，不能因此让
    评分或收尾整体炸掉。
    """
    key = str(test_kind or "").strip().lower()
    key = _SUITE_ALIASES.get(key, key)
    module = _SUITE_FACE_MODULES.get(key)
    if not module:
        return {}
    try:
        table = getattr(importlib.import_module(module), "CASE_FACES", {}) or {}
    except Exception:
        return {}
    return dict(table) if isinstance(table, Mapping) else {}


def skipped_case_ids(case_ids: Iterable[str], test_kind: str | None,
                     faces: Sequence[str] | None) -> set[str]:
    """这些用例里「所属面本次没被选中」的那些（即：没跑不是因为失败，是因为没勾选）。"""
    active = tuple(faces) if faces is not None else ()
    if not active:
        return set()
    table = suite_case_faces(test_kind)
    if not table:
        return set()
    return {case_id for case_id in case_ids if face_of(table, case_id) not in active}


def filter_specs(specs: Iterable[Any], table: Mapping[str, str],
                 faces: Sequence[str] | None = None,
                 id_attr: str = "case_id") -> list[Any]:
    """按面过滤用例规格，保持原顺序。"""
    active = tuple(faces) if faces is not None else DEFAULT_FACES
    return [spec for spec in specs
            if face_of(table, str(getattr(spec, id_attr))) in active]


def has_model_face(faces: Sequence[str]) -> bool:
    return any(face in BLOCKING_FACES for face in faces)


def face_verdict(results: Iterable[Any], table: Mapping[str, str],
                 faces: Sequence[str] | None = None) -> dict[str, Any]:
    """按面给出结论：只有 model/verify 面的硬失败才阻断准入。

    ``results`` 接受 ``CaseResult`` 对象或 summary 里的 dict；原 status 不改写，
    通道面的失败单独归入 ``channel_gap`` 供报告「网关缺口」小节引用。
    """
    active = tuple(faces) if faces is not None else DEFAULT_FACES
    rows: list[dict[str, Any]] = []
    for item in results:
        def get(key: str, default: Any = None, _item: Any = item) -> Any:
            return _item.get(key, default) if isinstance(_item, dict) else getattr(_item, key, default)

        case_id = str(get("case_id") or "")
        status = str(get("status") or get("case_status") or "").lower()
        rows.append({
            "case_id": case_id,
            "name": get("name") or "",
            "status": status,
            "face": face_of(table, case_id),
            "details": (get("details") or ""),
        })

    blocking = [row for row in rows
                if row["face"] in BLOCKING_FACES and row["status"] in _HARD_STATUSES]
    channel_gap = [row for row in rows
                   if row["face"] == FACE_CHANNEL and row["status"] in _GAP_STATUSES]
    evaluated = [row for row in rows if row["status"] in _EFFECTIVE_STATUSES
                 and row["case_id"] not in {"QWEN-F06", "GLM-53-F08"}]
    model_rows = [row for row in rows if row["face"] in BLOCKING_FACES]

    admission = not blocking and bool(evaluated)
    if blocking:
        head = "模型面存在失败：" + "、".join(f"{row['case_id']}（{row['face']}）" for row in blocking)
    elif not evaluated:
        head = "整批无 passed/warning，没有有效结论"
    elif not model_rows:
        head = (f"本次只执行了通道面（{len(rows)} 条），通过与否不构成模型能力结论；"
                f"通道缺口 {len(channel_gap)} 项仅记录")
    elif channel_gap:
        head = (f"模型能力与验真面无失败；通道面记录 {len(channel_gap)} 项缺口"
                f"（网关端点/严格语义未实现），不阻断准入")
    else:
        warnings = [row for row in rows if row["status"] == "warning"]
        head = ("模型能力与验真面无失败；全部用例通过" if not warnings
                else f"模型能力与验真面无失败；{len(warnings)} 项 warning 仅留痕")

    return {
        "contract": "faces",
        "executed_faces": list(active),
        "admission_passed": admission,
        "blocking": blocking,
        "channel_gap": channel_gap,
        "evaluated": len(evaluated),
        "summary_line": head,
        "face_breakdown": face_breakdown(rows),
    }


def face_breakdown(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(rows)
    out: list[dict[str, Any]] = []
    for face in ALL_FACES:
        bucket = [row for row in rows if row.get("face") == face]
        if not bucket:
            continue
        out.append({
            "face": face,
            "label": FACE_LABELS[face],
            "blocking_face": face in BLOCKING_FACES,
            "counts": {name: sum(row.get("status") == name for row in bucket)
                       for name in ("passed", "warning", "blocked", "failed", "error")},
            "verdict": "阻断" if (face in BLOCKING_FACES
                                  and any(row.get("status") in _HARD_STATUSES for row in bucket))
            else "记录",
            "notes": FACE_NOTES[face],
        })
    return out


def annotate_summary(summary: dict[str, Any], results: Iterable[Any],
                     table: Mapping[str, str], faces: Sequence[str],
                     *, with_verdict: bool = True) -> dict[str, Any]:
    """把「本次执行了哪些面、每条用例属于哪个面」写进 summary，供报告/聚合使用。

    ``with_verdict=False`` 用于已有权威判据契约的套件（DeepSeek / 千问的
    ``relay_verdict``）——只补执行范围与面映射，不叠加第二套准入结论。
    """
    results = list(results)
    case_faces = {str(getattr(item, "case_id", "") or ""): face_of(table, str(getattr(item, "case_id", "") or ""))
                  for item in results}
    summary["executed_faces"] = list(faces)
    summary["case_faces"] = case_faces
    if with_verdict:
        summary["face_verdict"] = face_verdict(results, table, faces)
    return summary
