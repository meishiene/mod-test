"""中转接入口径（relay admission profile）：中转/托管网关接入时的准入门槛。

与「官方口径」的区别
--------------------
`dsv4/` 的 61 条用例是按 **DeepSeek 官方文档** 设计的，官方口径要求文档承诺的每个
能力面都达标。但通过中转网关（如本仓库常打的 `api.oinone.top`、千帆）接入时，网关
往往只承诺 **Chat Completions 主面**：

- 部分官方面（Beta FIM、Chat Prefix、Anthropic Messages、Responses、Files、余额、
  模型列表）压根没有实现，或实现了但返回「HTTP 200 + 空结构/坏结构」；
- Strict Tool 的严格 schema 约束、prefix 续写语义常被静默忽略。

把这些「中转未承诺面」的缺失当成准入失败，会把「网关少实现几个端点」误判成「模型不行」。
中转接入口径的做法是 **按面分层**：

======================  =====================================================
 面                      判法
======================  =====================================================
 核心能力面（必达）      对话/思考/工具/JSON/流式/缓存/鉴权/参数边界/视觉(声明支持时)
                        ⇒ failed/error 即 **阻断准入**
 中转未承诺面（记录）    Strict Tool、Chat Prefix、FIM、Anthropic、Responses、
                        Files、余额、模型列表 ⇒ failed 记为 ``relay_gap``，不阻断
 文档-线上差异（记录）   宽松放行、比文档更严的强校验 ⇒ warning，不阻断
 平台缺失/瞬时故障        端点 404 / 5xx / 限流 ⇒ blocked，不阻断（沿用 gateway_contract）
======================  =====================================================

视觉按模型声明分流：仅 `flash` 系声明支持图片输入；未声明视觉的模型上，视觉用例
（DS-V4-36/42/53）记为「未声明能力」，既不阻断也不判失败。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .gateway_contract import CAPABILITY_VISION, model_supports_capability

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_ERROR = "error"
STATUS_WARNING = "warning"
STATUS_BLOCKED = "blocked"

#: 面定义：face_key -> (中文名, 该面下的用例 ID)
RELAY_OPTIONAL_FACES: dict[str, tuple[str, tuple[str, ...]]] = {
    "strict_tool": ("Strict Tool 严格 Schema 约束", ("DS-V4-19", "DS-V4-20")),
    "chat_prefix": ("Chat Prefix 续写", ("DS-V4-21", "DS-V4-22", "DS-V4-59")),
    "fim": ("Beta FIM 补全", ("DS-V4-23", "DS-V4-47", "DS-V4-58")),
    "anthropic": (
        "Anthropic Messages 兼容面",
        ("DS-V4-24", "DS-V4-35", "DS-V4-48", "DS-V4-49", "DS-V4-60"),
    ),
    "responses": (
        "Responses API 面",
        ("DS-V4-25", "DS-V4-34", "DS-V4-50", "DS-V4-51", "DS-V4-61"),
    ),
    "files": ("Files API 文件面", ("DS-V4-41", "DS-V4-54")),
    "account": ("账户余额端点", ("DS-V4-55",)),
    "model_list": ("模型列表端点", ("DS-V4-43",)),
}

#: 视觉用例（是否判失败取决于被测模型是否声明视觉）
VISION_CASE_IDS = ("DS-V4-36", "DS-V4-42", "DS-V4-53")

CASE_TO_FACE: dict[str, str] = {
    case_id: face
    for face, (_label, case_ids) in RELAY_OPTIONAL_FACES.items()
    for case_id in case_ids
}

PROFILE_NAME = "relay"
PROFILE_LABEL = "中转接入口径"


@dataclass
class RelayVerdict:
    """单个模型在中转接入口径下的准入结论。"""

    model: str | None
    admission_passed: bool
    blocking: list[dict[str, Any]] = field(default_factory=list)
    relay_gaps: list[dict[str, Any]] = field(default_factory=list)
    recorded: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": PROFILE_NAME,
            "profile_label": PROFILE_LABEL,
            "model": self.model,
            "admission_passed": self.admission_passed,
            "blocking": self.blocking,
            "relay_gaps": self.relay_gaps,
            "recorded": self.recorded,
            "stats": self.stats,
        }


def face_of(case_id: str) -> str | None:
    return CASE_TO_FACE.get(case_id)


def face_label(face: str) -> str:
    return RELAY_OPTIONAL_FACES[face][0]


def case_bucket(case_id: str, *, model: str | None = None) -> str:
    """用例归属：``core`` / ``relay_optional`` / ``vision_undeclared``。"""
    if case_id in VISION_CASE_IDS and not model_supports_capability(model, CAPABILITY_VISION):
        return "vision_undeclared"
    if case_id in CASE_TO_FACE:
        return "relay_optional"
    return "core"


def classify_case(
    case_id: str,
    status: str,
    *,
    model: str | None = None,
) -> str:
    """把一条用例在中转口径下归类。

    返回 ``blocking`` / ``relay_gap`` / ``vision_undeclared`` / ``informational``。
    """
    normalized = (status or "").strip().lower()
    if normalized not in {STATUS_FAILED, STATUS_ERROR}:
        return "informational"
    bucket = case_bucket(case_id, model=model)
    if bucket == "vision_undeclared":
        return "vision_undeclared"
    if bucket == "relay_optional":
        return "relay_gap"
    return "blocking"


def evaluate_relay_admission(
    cases: Iterable[dict[str, Any]],
    *,
    model: str | None = None,
) -> RelayVerdict:
    """按中转接入口径给出准入结论。

    ``cases`` 每项至少含 ``case_id`` 与 ``status``，可选 ``name`` / ``details``。
    """
    verdict = RelayVerdict(model=model, admission_passed=True)
    stats = {
        "total": 0, STATUS_PASSED: 0, STATUS_FAILED: 0, STATUS_WARNING: 0, STATUS_BLOCKED: 0, STATUS_ERROR: 0,
        "core_total": 0, "core_passed": 0, "relay_optional_total": 0, "relay_optional_passed": 0,
    }
    for case in cases:
        case_id = str(case.get("case_id") or "")
        status = str(case.get("status") or case.get("case_status") or "").lower()
        stats["total"] += 1
        if status in stats:
            stats[status] += 1
        bucket_kind = case_bucket(case_id, model=model)
        if bucket_kind == "core":
            stats["core_total"] += 1
            stats["core_passed"] += status == STATUS_PASSED
        elif bucket_kind == "relay_optional":
            stats["relay_optional_total"] += 1
            stats["relay_optional_passed"] += status == STATUS_PASSED
        record = {
            "case_id": case_id,
            "name": case.get("name") or "",
            "status": status,
            "details": (case.get("details") or "").strip(),
        }
        bucket = classify_case(case_id, status, model=model)
        if bucket == "blocking":
            verdict.blocking.append(record)
            verdict.admission_passed = False
        elif bucket == "relay_gap":
            record["face"] = face_of(case_id)
            record["face_label"] = face_label(face_of(case_id) or "")
            verdict.relay_gaps.append(record)
        elif bucket == "vision_undeclared":
            record["note"] = "该模型未声明视觉能力，不判失败"
            # 证据里的 failed 是「按旧能力面假设跑出来的」，口径修正后等效为 warning；
            # 渲染时用 effective_status，避免报告里出现「不阻断却标 failed」的自相矛盾。
            record["bucket"] = "vision_undeclared"
            record["effective_status"] = STATUS_WARNING
            verdict.recorded.append(record)
        elif status in {STATUS_WARNING, STATUS_BLOCKED}:
            verdict.recorded.append(record)
    verdict.relay_gaps.sort(key=lambda r: (r.get("face") or "", r["case_id"]))
    verdict.stats = stats
    return verdict


def merge_case_records(summaries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """把多份 summary 的用例结果按「最新证据优先」合并。

    用于「上轮全量 + 本轮定向复测」的场景：同一 case_id 以 ``generated_at`` 较晚者
    为准；缺失的用例保留上一轮结论。
    """
    latest: dict[str, tuple[str, dict[str, Any]]] = {}
    for summary in summaries:
        stamp = str(summary.get("generated_at") or "")
        for case in summary.get("results") or []:
            # 公共契约探针没有 case_id（只有中文名），用名字合成稳定键，否则会被当成重复项丢弃。
            key = str(case.get("case_id") or "").strip() or f"name:{case.get('name') or ''}"
            previous = latest.get(key)
            if previous is None or stamp >= previous[0]:
                latest[key] = (stamp, case)
    return [case for _stamp, case in latest.values()]
