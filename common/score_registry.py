"""评分注册表：把每个原子用例登记到「计分面」，并提供套件全量清单。

为什么单独一个文件
------------------
评分必须**不可挑选**：分母来自这里的全量清单，而不是结果文件里出现了哪些 id。
否则「少跑几条就自动满分」。计分面相对权重在本文件登记，分侧换算与等级在 `common/scoring.py`，
本文件只负责「这条用例属于哪个面、优先级多少」。

登记原则（权威口径见 docs/模型测试评分体系设计.md）
--------------------------------------------------
- ``trust``（否决门，权重 0）：只放**身份 / 模型一致性 / 协议改写 / 注入 / 签名真实**
  这类用例。它 failed 说明「测的不是这个模型」，此时分数没有意义 ⇒ 评分作废。
- ``fingerprint``（权重 0，仅证据）：只放**自述、汇总、结构取证**这类不可客观判定或
  纯证据的用例（档位自述、满血证据汇总、Protobuf 结构解码、计费探针、图片指纹）。
- 指纹卡里**断言客观**的用例（流式完整性、参数静默忽略、上下文边界、缓存命中）
  按语义归到 core / params / context —— 不因为写在指纹卡里就被当成「只是证据」。
- ``performance``（权重 0）：TTFT / TPOT / 压测项，不并入一致性分。
- ``unknown``：未登记且无前缀兜底的 id，权重 0 并在报告里告警（防改 id 绕过）。
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

SCORING_VERSION = "conformance-2.1"

# ---- 计分面相对权重（各侧独立归一为模型 100 / 通道 20）------------------------------------------------
FACE_ORDER: tuple[str, ...] = (
    "core", "quality", "tools", "thinking", "context", "vision", "params", "envelope",
)
FACE_WEIGHTS: dict[str, float] = {
    "core": 28.0,      # 核心对话与协议契约
    "quality": 20.0,   # 能力质量（满血度）
    "tools": 14.0,     # 工具与 Agent（含 strict_tool）
    "thinking": 10.0,  # 思考与推理控制
    "context": 10.0,   # 上下文与缓存
    "vision": 8.0,     # 多模态输入
    "params": 6.0,     # 参数与采样契约
    "envelope": 4.0,   # 错误契约与鉴权
}
FACE_LABELS: dict[str, str] = {
    "core": "核心对话与协议",
    "quality": "能力质量（满血度）",
    "tools": "工具与 Agent",
    "thinking": "思考与推理控制",
    "context": "上下文与缓存",
    "vision": "多模态视觉",
    "params": "参数与采样契约",
    "envelope": "错误契约与鉴权",
    "trust": "模型一致性（否决门）",
    "fingerprint": "身份与取证（仅证据）",
    "performance": "性能项（不计分）",
    "unknown": "未登记用例",
}
# 0 权重的特殊面：不参与打分，只进报告。
VETO_FACE = "trust"
EVIDENCE_FACE = "fingerprint"
PERFORMANCE_FACE = "performance"
UNKNOWN_FACE = "unknown"
ZERO_WEIGHT_FACES = frozenset({VETO_FACE, EVIDENCE_FACE, PERFORMANCE_FACE, UNKNOWN_FACE})

# ---- 面内权重：按卡片优先级 ------------------------------------------------
PRIORITY_WEIGHTS: dict[str, float] = {
    "P0": 3.0, "BASIC": 3.0,
    "P1": 2.0,
    "P2": 1.0,
}
# 专项卡（PTHINK / PTIER / PVISION / PENVELOPE / PIDENT / PCACHE / AUTH / PELICAN…）
# 通常独占自己的面，归一后自然等于该面权重，统一按 1 计。
DEFAULT_PRIORITY_WEIGHT = 1.0


@dataclass(frozen=True)
class CaseMeta:
    case_id: str
    name: str
    priority: str
    face: str

    @property
    def priority_weight(self) -> float:
        return PRIORITY_WEIGHTS.get(self.priority, DEFAULT_PRIORITY_WEIGHT)


# ---- 各套件：用例 → 面 ------------------------------------------------------
# 说明：DeepSeek / GLM / Claude / Gemini / GPT 的用例编号本身没有语义，只能逐条登记；
# 千问复用 qwen/relay_profile.CASE_FACES（那里已是准入权威），把 strict_tool 并入 tools。

_DSV4_FACE: dict[str, str] = {
    # 核心对话与协议（含 completion 面：Prefix / FIM 属端点级能力）
    "DS-V4-01": "core", "DS-V4-11": "core", "DS-V4-12": "core", "DS-V4-13": "core",
    # Prefix / FIM 两个补全面属端点级能力契约（不是参数、也不是工具语义）
    "DS-V4-21": "core", "DS-V4-22": "core", "DS-V4-23": "core",
    "DS-V4-25": "core", "DS-V4-26": "core", "DS-V4-27": "core", "DS-V4-34": "core",    "DS-V4-35": "core", "DS-V4-37": "core", "DS-V4-44": "core", "DS-V4-47": "core",    "DS-V4-48": "core", "DS-V4-50": "core", "DS-V4-58": "core", "DS-V4-59": "core",
    # 思考与推理控制（含 reasoning 回传、工具链里的 reasoning 约束）
    "DS-V4-02": "thinking", "DS-V4-03": "thinking", "DS-V4-04": "thinking",
    "DS-V4-05": "thinking", "DS-V4-06": "thinking", "DS-V4-07": "thinking",
    "DS-V4-08": "thinking", "DS-V4-09": "thinking", "DS-V4-10": "thinking",
    "DS-V4-38": "thinking", "DS-V4-49": "thinking",
    # 上下文与缓存
    "DS-V4-14": "context", "DS-V4-15": "context", "DS-V4-16": "context",
    "DS-V4-17": "context", "DS-V4-18": "context",
    # 工具与 Agent
    "DS-V4-19": "tools", "DS-V4-20": "tools", "DS-V4-32": "tools",
    "DS-V4-33": "tools", "DS-V4-52": "tools", "DS-V4-60": "tools", "DS-V4-61": "tools",
    # 参数与采样契约
    "DS-V4-28": "params", "DS-V4-29": "params", "DS-V4-30": "params",
    "DS-V4-39": "params", "DS-V4-45": "params", "DS-V4-46": "params",
    "DS-V4-51": "params", "DS-V4-56": "params", "DS-V4-57": "params",
    # 错误契约与鉴权
    "DS-V4-24": "envelope", "DS-V4-31": "envelope", "DS-V4-40": "envelope",
    "DS-V4-41": "envelope", "DS-V4-43": "envelope", "DS-V4-54": "envelope",
    "DS-V4-55": "envelope",
    # 视觉
    "DS-V4-36": "vision", "DS-V4-42": "vision", "DS-V4-53": "vision",
    # 满血真实性探针（authenticity.py）：按断言语义归面
    "AUTH-01": "tools", "AUTH-02": "core", "AUTH-03": "context",
    "AUTH-04": "thinking", "AUTH-05": "core", "AUTH-06": "params",
    # Protobuf 签名两步校验：响应侧真实性 ⇒ 信任级
    "SIG-01": "trust", "SIG-02": "trust",
}

# GLM：指纹卡（F01–F08）里只有「身份」和「汇总」是不可客观判定的，
# 其余（thinking 字段、tool_choice、JSON Schema、流式完整性、静默吞参、上下文边界）
# 都是可判定契约，按语义归面。
_GLM_FACE: dict[str, str] = {
    "GLM-53-01": "core", "GLM-53-02": "core", "GLM-53-03": "core",
    "GLM-53-04": "thinking", "GLM-53-06": "core", "GLM-53-07": "core",
    "GLM-53-08": "context",
    "GLM-53-09": "tools", "GLM-53-10": "tools", "GLM-53-11": "tools", "GLM-53-12": "tools",
    "GLM-53-C01": "context", "GLM-53-C02": "context",
    "GLM-53-F01": "trust",
    "GLM-53-F02": "thinking", "GLM-53-F03": "tools", "GLM-53-F04": "core",
    "GLM-53-F05": "core", "GLM-53-F06": "params", "GLM-53-F07": "context",
    "GLM-53-F08": "fingerprint",
}

from .model_validation import PROBE_DEFINITIONS
_GLM_FACE.update({f"GLM-53-{suffix}": face for suffix, _, _, _, face in PROBE_DEFINITIONS})

# Claude（fable5）：BASIC 15 条里 B01 是身份，其余是能力质量。
_FABLE_FACE: dict[str, str] = {
    "FABLE5-B01": "trust",
    **{f"FABLE5-B{n:02d}": "quality" for n in range(2, 16)},
    "FABLE5-E01": "thinking", "FABLE5-E02": "thinking", "FABLE5-E03": "thinking",
    "FABLE5-E04": "thinking", "FABLE5-E05": "thinking", "FABLE5-E06": "thinking",
    "FABLE5-E07": "tools", "FABLE5-E08": "tools", "FABLE5-E09": "tools",
    "FABLE5-E10": "tools",
    "FABLE5-E11": "core", "FABLE5-E12": "core", "FABLE5-E13": "core",
    "FABLE5-E14": "vision",
    "FABLE5-E15": "context", "FABLE5-E16": "context",
    "FABLE5-E17": "params", "FABLE5-E18": "core", "FABLE5-E19": "params",
    "FABLE5-E20": "params",
    "FABLE5-S01": "params", "FABLE5-S02": "context", "FABLE5-S03": "context",
    "FABLE5-S04": "tools", "FABLE5-S05": "tools", "FABLE5-S06": "tools",
    "FABLE5-S07": "tools", "FABLE5-S08": "context", "FABLE5-S09": "params",
    "FABLE5-S10": "quality", "FABLE5-S11": "core",
    "FABLE5-C01": "envelope", "FABLE5-C02": "envelope", "FABLE5-C03": "params",
    "FABLE5-C04": "params", "FABLE5-C05": "params", "FABLE5-C06": "params",
    "FABLE5-C07": "params", "FABLE5-C08": "envelope", "FABLE5-C09": "envelope",
    "FABLE5-C10": "envelope", "FABLE5-C11": "envelope", "FABLE5-C12": "envelope",
    "FABLE5-C13": "core", "FABLE5-C14": "core", "FABLE5-C15": "core",
    "FABLE5-C16": "envelope", "FABLE5-C17": "params", "FABLE5-C18": "envelope",
    "FABLE5-C19": "params", "FABLE5-C20": "envelope", "FABLE5-C21": "core",
    "AUTH-F01": "quality", "AUTH-F02": "quality", "AUTH-F03": "quality",
    "AUTH-F04": "thinking", "AUTH-F05": "thinking",
    "AUTH-F06": "trust", "AUTH-F07": "trust", "AUTH-F08": "trust",
    "AUTH-F09": "trust", "AUTH-F10": "trust", "AUTH-F11": "core",
    "AUTH-F12": "trust", "AUTH-F13": "trust", "AUTH-F14": "trust",
}

# Gemini 3.x：编号规则与 Claude 平行（B 基础 / E 核心 / S 专项 / C 契约 / AUTH 指纹）。
_GEMINI_FACE: dict[str, str] = {
    "G3-B01": "trust",
    "G3-B02": "quality", "G3-B03": "quality", "G3-B04": "quality",
    "G3-B05": "quality", "G3-B06": "quality",
    "G3-B07": "core", "G3-B08": "core",
    "G3-B09": "quality", "G3-B10": "quality",
    "G3-E01": "thinking", "G3-E02": "thinking", "G3-E03": "thinking",
    "G3-E04": "thinking", "G3-E05": "thinking", "G3-E06": "thinking",
    "G3-E07": "tools", "G3-E08": "tools", "G3-E09": "tools",
    "G3-E10": "tools", "G3-E11": "tools", "G3-E12": "tools",
    "G3-E13": "core", "G3-E14": "vision", "G3-E15": "vision",
    "G3-E16": "context", "G3-E17": "context",
    "G3-E18": "params", "G3-E19": "params", "G3-E20": "core",
    "G3-S01": "thinking", "G3-S02": "vision", "G3-S03": "tools",
    "G3-S04": "params", "G3-S05": "params", "G3-S06": "core",
    "G3-S07": "context", "G3-S08": "vision", "G3-S09": "thinking",
    "G3-S10": "core",
    "G3-C01": "envelope", "G3-C02": "envelope", "G3-C03": "params",
    "G3-C04": "params", "G3-C05": "params", "G3-C06": "envelope",
    "G3-C07": "envelope", "G3-C08": "envelope", "G3-C09": "envelope",
    "G3-C10": "envelope", "G3-C11": "core", "G3-C12": "core",
    "G3-C13": "core", "G3-C14": "envelope", "G3-C15": "params",
    "G3-C16": "params",
    "AUTH-G3-01": "quality", "AUTH-G3-02": "quality", "AUTH-G3-03": "quality",
    "AUTH-G3-04": "thinking", "AUTH-G3-05": "thinking", "AUTH-G3-06": "thinking",
    "AUTH-G3-07": "trust", "AUTH-G3-08": "trust", "AUTH-G3-09": "trust",
    "AUTH-G3-10": "trust", "AUTH-G3-11": "core", "AUTH-G3-12": "thinking",
}

# GPT 家族：GPT-6 与 GPT-6.1 编号平行（01–37），GPT-5.6 编号不连续，单独登记。
_GPT6_NUM_FACE: dict[str, str] = {
    "01": "trust", "02": "thinking", "03": "thinking", "04": "thinking",
    "05": "thinking", "06": "core", "07": "tools", "08": "params",
    "09": "tools", "10": "core", "11": "context", "12": "context",
    "13": "context", "14": "context", "15": "params", "16": "core",
    "17": "core", "18": "vision", "19": "core", "20": "tools",
    "21": "tools", "22": "context", "23": "params", "24": "envelope",
    "25": "envelope", "26": "core", "27": "core", "28": "core",
    "29": "envelope", "30": "envelope", "31": "params", "32": "envelope",
    "33": "envelope", "34": "envelope", "35": "envelope", "36": "core",
    "37": "quality",
}
_GPT56_NUM_FACE: dict[str, str] = {
    "01": "core", "02": "thinking", "03": "thinking", "04": "thinking",
    "05": "thinking", "06": "thinking", "07": "tools", "08": "core",
    "09": "core", "10": "context", "11": "context", "12": "context",
    "13": "params", "14": "core", "15": "core", "16": "core",
    "17": "thinking", "18": "thinking", "19": "tools", "20": "context",
    "21": "context", "22": "envelope", "23": "envelope", "24": "core",
    "25": "core", "26": "core", "27": "params", "28": "envelope",
    "29": "tools", "30": "tools", "31": "params", "32": "params",
    "33": "envelope", "34": "envelope", "35": "core", "36": "vision",
    "37": "vision", "40": "core", "41": "core", "49": "context",
    "53": "tools", "54": "envelope", "55": "envelope", "56": "envelope",
    "57": "envelope", "61": "core", "62": "quality",
}
# 满血真实性指纹（AUTH-*）：随机挑战类归 quality（测的是「还满不满血」），
# 身份/档位下限/协议改写/注入逐条归 trust。
_GPT_AUTH_NUM_FACE: dict[str, str] = {
    "01": "quality", "02": "quality", "03": "quality", "04": "thinking",
    "05": "core", "06": "thinking", "07": "trust", "08": "trust",
    "09": "quality", "10": "trust", "11": "trust", "12": "trust",
}
_GPT_FACE: dict[str, str] = {
    **{f"GPT6-{n}": face for n, face in _GPT6_NUM_FACE.items()},
    **{f"GPT61-{n}": face for n, face in _GPT6_NUM_FACE.items()},
    **{f"GPT56-{n}": face for n, face in _GPT56_NUM_FACE.items()},
    **{f"AUTH-G{n}": face for n, face in _GPT_AUTH_NUM_FACE.items()},
    **{f"AUTH-G6-{n}": face for n, face in _GPT_AUTH_NUM_FACE.items()},
    **{f"AUTH-G61-{n}": face for n, face in _GPT_AUTH_NUM_FACE.items()},
}

# Kimi K3 临时案例集：按清单里的 category 归面（TTFT/TPOT 是性能项 ⇒ 不计分）。
_K3_CATEGORY_FACE: dict[str, str] = {
    "基础连通": "core",
    "认证与错误": "envelope",
    "Thinking 模式": "thinking",
    "采样参数": "params",
    "流式与计量": "core",
    "结构化输出": "core",
    "工具调用": "tools",
    "可观测性": "envelope",
    "缓存": "context",
    "TTFT": PERFORMANCE_FACE,
    "TPOT": PERFORMANCE_FACE,
}

# 套件名归一（前端 tab 名、历史别名都指同一套用例）
_KIND_ALIASES: dict[str, str] = {
    "deepseek": "deepseek", "ds": "deepseek",
    "qwen": "qwen",
    "glm": "glm", "glm_stream_retest": "glm",
    "gpt": "gpt",
    "fable": "fable", "awsb": "fable",
    "gemini": "gemini",
    "kvv": "kvv",
}


def normalize_kind(test_kind: str | None) -> str:
    return _KIND_ALIASES.get(str(test_kind or "").strip().lower(), "")


# 千问：**评分面与准入面不是一回事**——准入仍以 qwen/relay_profile.CASE_FACES 为唯一权威，
# 这里只按「用例断言的语义」重挂计分面。F03 除窗口自述外还带真实长输入探测，
# 属上下文证据（准入侧仍记 fingerprint，不影响 relay 结论）。
_QWEN_FACE_OVERRIDE: dict[str, str] = {
    "QWEN-F03": "context",
}


def _qwen_faces() -> dict[str, str]:
    """千问复用准入权威的 CASE_FACES，只做面名翻译（strict_tool ⇒ tools）+ 语义覆盖。"""
    from qwen.relay_profile import CASE_FACES

    faces = {case_id: ("tools" if face == "strict_tool" else face)
             for case_id, face in CASE_FACES.items()}
    faces.update(_QWEN_FACE_OVERRIDE)
    from .model_validation import PROBE_DEFINITIONS
    faces.update({f"QWEN-{suffix}": face for suffix, _, _, _, face in PROBE_DEFINITIONS})
    return faces


_SUITE_FACE: dict[str, dict[str, str]] = {
    "deepseek": _DSV4_FACE,
    "glm": _GLM_FACE,
    "gpt": _GPT_FACE,
    "fable": _FABLE_FACE,
    "gemini": _GEMINI_FACE,
}


_K3_FACE_CACHE: dict[str, str] | None = None


def _k3_faces() -> dict[str, str]:
    """K3 临时案例集没有语义化编号，只能按清单里的 category 归面（读一次后缓存）。"""
    global _K3_FACE_CACHE
    if _K3_FACE_CACHE is not None:
        return _K3_FACE_CACHE
    table: dict[str, str] = {}
    try:
        from kvv.temp_suite import load_manifest, manifest_cases

        for item in manifest_cases(load_manifest()):
            if isinstance(item, dict):
                case_id, category = str(item.get("id") or ""), str(item.get("category") or "")
            else:
                case_id, category = str(getattr(item, "id", "")), str(getattr(item, "category", ""))
            if case_id:
                table[case_id] = _K3_CATEGORY_FACE.get(category, UNKNOWN_FACE)
    except Exception:
        table = {}
    _K3_FACE_CACHE = table
    return table


def _face_table(test_kind: str | None) -> dict[str, str]:
    kind = normalize_kind(test_kind)
    if kind == "qwen":
        return _qwen_faces()
    if kind == "kvv":
        return _k3_faces()
    return _SUITE_FACE.get(kind, {})


# 未登记 id 的前缀兜底：这些卡片本身就是取证/性能/非文本面，权重 0。
_PREFIX_FACE: tuple[tuple[str, str], ...] = (
    ("ECO-", EVIDENCE_FACE),    # 独立经济验证证据，不兑换旧制120分。
    ("K3-TTFT", PERFORMANCE_FACE),
    ("K3-TPOT", PERFORMANCE_FACE),
    ("PB", EVIDENCE_FACE),        # Protobuf 结构取证
    ("PBR", EVIDENCE_FACE),
    ("VAL", EVIDENCE_FACE),
    ("BILLING", EVIDENCE_FACE),   # 计费探针
    ("MN", "trust"),              # 模型名一致性
    ("ZI", "trust"),              # 零注入侧信道
    ("SIG", "trust"),             # 签名两步回传
    ("SEEDANCE", EVIDENCE_FACE),  # 视频生成：v1 不计分，只作证据
    ("CHAT", EVIDENCE_FACE),
)


def _prefix_face(case_id: str) -> str:
    for prefix, face in _PREFIX_FACE:
        if case_id.startswith(prefix):
            return face
    return UNKNOWN_FACE


# 跨套件通用的契约探针（common/model_contracts.py，任何 openai 家族套件都可能出现）。
_GLOBAL_FACE: dict[str, str] = {
    "CONTRACT-01": "trust",     # 身份与路由
    "CONTRACT-02": "envelope",  # 参数错误契约
    "CONTRACT-03": "core",      # 流式协议完整性
    "CONTRACT-04": "tools",     # 工具调用多轮闭环
    "CONTRACT-05": "thinking",  # Thinking 与 reasoning_effort
}


def face_for(test_kind: str | None, case_id: str) -> str:
    """用例 → 计分面。未登记时按全局表/前缀兜底，仍未知则 ``unknown``（权重 0 + 告警）。"""
    table = _face_table(test_kind)
    return table.get(case_id) or _GLOBAL_FACE.get(case_id) or _prefix_face(case_id)


def relay_optional_cases(test_kind: str | None) -> set[str]:
    """该套件里属「中转可选能力面」的用例（失败只算通道缺口，不算模型缺陷）。

    数据源是既有准入权威：千问取 `relay_profile.BLOCKING_FACES` 之外的面，
    DeepSeek 取 `relay_profile.RELAY_OPTIONAL_FACES` 的用例表。评分只用它做**归因**，
    不改变准入，也不改变扣分多少——差距是真实的，只是责任方不同。
    """
    kind = normalize_kind(test_kind)
    try:
        if kind == "qwen":
            from qwen.relay_profile import BLOCKING_FACES, CASE_FACES

            return {case_id for case_id, face in CASE_FACES.items() if face not in BLOCKING_FACES}
        if kind == "deepseek":
            from dsv4.relay_profile import RELAY_OPTIONAL_FACES

            return {case_id for _label, case_ids in RELAY_OPTIONAL_FACES.values() for case_id in case_ids}
    except Exception:
        return set()
    return set()


# ---- 套件全量清单（评分分母的权威来源）------------------------------------
def _spec_rows(module: str) -> list[tuple[str, str, str]]:
    specs = importlib.import_module(module).CASE_SPECS
    return [(spec.case_id, getattr(spec, "name", ""), getattr(spec, "priority", "")) for spec in specs]


def _gpt_variant(model: str | None) -> str | None:
    """被测型号 → 用例档位。GPT-5.6 不在 MODEL_TIERS 里（只有一条 MODELS 条目），单独判。"""
    key = str(model or "").strip().lower()
    if not key:
        return None
    if key.startswith(("gpt-5.6", "gpt5.6", "gpt56")):
        return "gpt5.6"
    try:
        from gpt56.core import TIER_BY_MODEL
    except Exception:
        return None
    tier = TIER_BY_MODEL.get(key)
    return getattr(tier, "variant", None) if tier else None


def _kvv_rows() -> list[tuple[str, str, str]]:
    from kvv.temp_suite import load_manifest, manifest_cases

    rows: list[tuple[str, str, str]] = []
    for item in manifest_cases(load_manifest()):
        if isinstance(item, dict):
            rows.append((str(item.get("id") or ""), str(item.get("name") or ""),
                         str(item.get("priority") or "")))
        else:  # 兼容对象形态
            rows.append((str(getattr(item, "id", "")), str(getattr(item, "name", "")),
                         str(getattr(item, "priority", ""))))
    return [row for row in rows if row[0]]


# 套件里不走 CASE_SPECS 的卡片（探针 / 签名 / 视频等），显式登记，避免「未登记」告警。
_SUITE_EXTRA: dict[str, list[tuple[str, str, str]]] = {
    "deepseek": [
        ("SIG-01", "原签名回传", "P1"),
        ("SIG-02", "篡改签名拒绝", "P1"),
        ("AUTH-01", "随机动态工具诱导调用", "AUTH"),
        ("AUTH-02", "随机 JSON 对象输出", "AUTH"),
        ("AUTH-03", "随机多轮记忆", "AUTH"),
        ("AUTH-04", "Thinking 开启与 reasoning 字段", "AUTH"),
        ("AUTH-05", "流式结束与事件完整性", "AUTH"),
        ("AUTH-06", "Thinking 静默吞参对照", "AUTH"),
    ],
}

_LOADERS: dict[str, Any] = {
    "deepseek": lambda model=None: _spec_rows("dsv4.core"),
    "qwen": lambda model=None: _spec_rows("qwen.core"),
    "glm": lambda model=None: _spec_rows("glm53.core"),
    "fable": lambda model=None: _spec_rows("fable5.core"),
    "gemini": lambda model=None: _spec_rows("gemini3.core"),
    "kvv": lambda model=None: _kvv_rows(),
    "gpt": lambda model=None: [
        row for row in _spec_rows("gpt56.core")
        if _gpt_variant(model) is None
        or (importlib.import_module("gpt56.core")._variant_of(row[0]) == _gpt_variant(model))
    ],
}


def inventory(test_kind: str | None, model: str | None = None) -> dict[str, CaseMeta]:
    """该套件对该型号的**全量**用例清单（含 0 权重的证据/性能项）。"""
    kind = normalize_kind(test_kind)
    loader = _LOADERS.get(kind)
    if loader is None:
        return {}
    try:
        rows = list(loader(model))
    except Exception:
        return {}
    rows.extend(_SUITE_EXTRA.get(kind, []))
    out: dict[str, CaseMeta] = {}
    for case_id, name, priority in rows:
        if not case_id or case_id in out:
            continue
        out[case_id] = CaseMeta(case_id=case_id, name=name, priority=priority,
                                face=face_for(kind, case_id))
    return out


# ---- 能力声明：矩阵说不支持的面，整条移出权重池 --------------------------
def not_applicable(test_kind: str | None, model: str | None) -> set[str]:
    """按**能力矩阵声明**（不是被测端点自述）判定不适用的用例。"""
    kind = normalize_kind(test_kind)
    if kind == "qwen":
        try:
            from qwen.cases import model_spec

            spec = model_spec(model or "")
            if spec is not None and not spec.image_input:
                return {cid for cid, meta in inventory(kind, model).items() if meta.face == "vision"}
        except Exception:
            return set()
    if kind == "deepseek":
        try:
            from dsv4.gateway_contract import CAPABILITY_VISION, model_supports_capability

            if not model_supports_capability(model, CAPABILITY_VISION):
                return {cid for cid, meta in inventory(kind, model).items() if meta.face == "vision"}
        except Exception:
            return set()
    return set()


def face_weights() -> dict[str, float]:
    """权重快照（写进 score.json，保证历史分数可回溯口径）。"""
    return dict(FACE_WEIGHTS)
