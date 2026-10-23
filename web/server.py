#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import queue
import re
import signal
import shutil
import sqlite3
import subprocess
import sys
import zipfile
import httpx
import threading
import time
import uuid
from html import escape
from collections.abc import Iterator
from collections import Counter
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from common.report_style import strip_report_scores
from web.image_details import image_fingerprint_fields
from web.report_details import ensure_report_details
from volcperf.protocols import (
    OPENAI_COMPATIBLE_PROTOCOLS,
    PROTOCOLS,
    api_root,
    models_endpoint,
    models_headers,
    normalize_base_url,
    parse_models,
    split_path_override,
    supports_test_kind,
)
from volcperf.config import ApiConfig, trust_env_enabled
from volcperf.cases import CASES as LOAD_CASES
from volcperf.connectivity import check_connectivity
from kvv.cases import CASES as KVV_CASES, TEMP_CASE_ID as KVV_TEMP_CASE_ID, TEMP_PROTOCOLS as KVV_TEMP_PROTOCOLS, OPTIONAL_UNSCORED_CASES
from dsv4.cases import CASES as DEEPSEEK_CASES
from glm53.cases import CASES as GLM_CASES, GLM_MODELS
from qwen.cases import CASES as QWEN_CASES, QWEN_DEFAULT_MODEL, QWEN_MODELS
from gpt56.cases import CASES as GPT56_CASES, PELICAN_CASE_IDS
from gpt56.core import (MODEL_TIERS, PELICAN_ARTIFACT_TITLES, PELICAN_MAX_CONCURRENCY,
                        PELICAN_MAX_REQUESTS, PELICAN_ROUND_SVG_PATTERN)
from fable5.cases import CASES as FABLE5_CASES, Fable5Case
from fable5.model_profile import resolve_thinking_mode
from authenticity import CASES as AUTH_CASES
from forensics.cases import CASES as FORENSICS_CASES
from gpt_image import CASES as GPT_IMAGE_CASES
from gpt_image.cases import DEFAULT_REQUEST_TIMEOUT_SECONDS, GPT_IMAGE_ASPECT_RATIOS, model_quality_choices
from gemini_image import CASES as GEMINI_IMAGE_CASES
from seedance import CASES as SEEDANCE_CASES
from seedance import cases as seedance_cases
from gemini3.cases import CASES as GEMINI3_CASES
from gemini3.core import MODEL_TIERS as GEMINI3_MODEL_TIERS
from chatpane.cases import (
    CASES as CHAT_CASES,
    CHAT_CASE_ID,
    DEFAULT_MAX_TOKENS as CHAT_DEFAULT_MAX_TOKENS,
    MAX_MAX_TOKENS as CHAT_MAX_MAX_TOKENS,
    MAX_PROMPT_CHARS as CHAT_MAX_PROMPT_CHARS,
    MAX_SYSTEM_CHARS as CHAT_MAX_SYSTEM_CHARS,
)
from arxivmath_cases import CASE as ARXIVMATH_CASE
from deepswe_cases import CASE as DEEPSWE_CASE
from sweatlas_cases import CASE as SWE_ATLAS_TW_CASE
from sweatlas_rf_cases import CASE as SWE_ATLAS_RF_CASE
from legal_agent_benchmark_cases import CASE as LEGAL_AGENT_BENCHMARK_CASE

from common.result_adapter import normalize_summary
from common.faces import (ALL_FACES, DEFAULT_FACES, FACE_CHANNEL, FACE_LABELS,
                          FaceSelectionError, face_of, parse_faces, selected_faces)
from common.image_prompt import MAX_IMAGE_PROMPT_CHARS, normalize_image_prompt
from web.aggregation import SCHEMA_VERSION, aggregate_batch, build_item_metrics, enrich_summary, ensure_batch_metrics
from common.finalizer import finalize_result_dir
from common.scoring import score_batch, score_summary
from common.economy_profiles import ECONOMY_CASES, economy_case_payload, economy_profile
from web.score_reports import current_score
from kvv.temp_suite import MANIFEST_PATH as KVV_TEMP_CASES_PATH, TempCase, find_case, load_manifest, manifest_cases, merge_cases

ROOT = Path(__file__).resolve().parent.parent
# K3 供应商临时案例集已并入「Kimi 测试」（test_kind=kvv）模块：清单与解析器都在
# kvv 包内（kvv/temp_suite.py + kvv/kimi_k3_temp_cases.json），执行器是
# kvv/temp_runner.py（入口 scripts/run_kvv_temp.py）。它与同模块的官方 KVV pytest
# 套件分属两条执行链，由 case id（KVV_TEMP_CASE_ID）在派发层分流。
# 中转计费临时探针：与 K3 清单同构（无 request.batch → 每条各自成一个可选项），
# 复用 kvv.temp_suite 的通用清单解析。它现在是「Claude 测试」（test_kind=fable）
# 模块下的一个**常驻**案例组，与各模型变体（Fable 5 / Opus 5 / Sonnet 5 /
# Haiku 4.5 / Sonnet 4.6）的五套件并列、不随模型变体切换；执行器走 Anthropic
# Messages 面。
BILLING_CASES_PATH = ROOT / "test_cases" / "relay_billing_cases.json"
STATIC = ROOT / "web" / "static"
DB_PATH = ROOT / "web" / "data" / "runs.db"
RESULTS = ROOT / "results"
RUN_SCRIPT = ROOT / "scripts" / "run_test.py"
KVV_SCRIPT = ROOT / "scripts" / "run_kvv.py"
# 「Kimi 测试」模块的第二条执行链：K3 供应商临时案例集（清单驱动，环境变量派发）。
KVV_TEMP_SCRIPT = ROOT / "scripts" / "run_kvv_temp.py"
# DS_V4_P0/P1/P2 的权威实现是 dsv4 包（dsv4/cases.py 元数据 + dsv4/core.py 用例 + dsv4/runner.py 装配），
# 入口脚本 scripts/run_dsv4.py 接受位置参数（用例名）并转发到 dsv4.runner。
# 注意：init 时代的旧副本 k3perf/deepseek_v4.py 与 scripts/run_deepseek_v4.py 已删除，不要再引入。
DEEPSEEK_SCRIPT = ROOT / "scripts" / "run_dsv4.py"
GLM_SCRIPT = ROOT / "scripts" / "run_glm53.py"
# 千问（Qwen 3.6 / 3.7 / 3.8）专项：权威实现是 qwen 包，入口脚本只做参数转发。
QWEN_SCRIPT = ROOT / "scripts" / "run_qwen.py"
GPT56_SCRIPT = ROOT / "scripts" / "run_gpt56.py"
# Fable 5（Anthropic Mythos 级）专项：权威实现是 fable5 包，入口脚本只做参数转发。
FABLE5_SCRIPT = ROOT / "scripts" / "run_fable5.py"
CONTRACT_SCRIPT = ROOT / "scripts" / "run_contracts.py"
ECONOMY_SCRIPT = ROOT / "scripts" / "run_economy_validation.py"
AUTH_SCRIPT = ROOT / "scripts" / "run_authenticity.py"
# 中转验真（跨模型取证：隐式身份 / 图像理解 / 错误信封泄露）：权威实现在 forensics 包，
# 入口脚本只做参数转发。三个用例都是固定小探针，参数全部走环境变量。
FORENSICS_SCRIPT = ROOT / "scripts" / "run_forensics.py"
GPT_IMAGE_SCRIPT = ROOT / "scripts" / "run_gpt_image.py"
GEMINI_IMAGE_SCRIPT = ROOT / "scripts" / "run_gemini_image.py"
# Seedance 视频生成（异步任务面）：权威实现是 seedance 包，入口脚本只做参数转发。
# 上传的首帧/参考图由前端 POST 到 /api/uploads（原始二进制），Web 只把落盘路径交给
# 子进程——图片 base64 走环境变量会撞 execve 的 ARG_MAX 上限，且会让 job 落进数据库。
SEEDANCE_SCRIPT = ROOT / "scripts" / "run_seedance.py"
SEEDANCE_UPLOAD_DIR = ROOT / "web" / "data" / "uploads"
# 上传的图片只在「提交任务 → 子进程读走 → 结果目录留档」这条链上短暂有用。
# 正常路径由 _release_seedance_upload() 在子进程退出后立刻删除；用户上传后中途放弃
# 则没有回收时机，因此每次上传前顺手清一次陈旧文件，避免目录无界增长。
SEEDANCE_UPLOAD_TTL_SECONDS = 24 * 3600


def sweep_stale_uploads(max_age_seconds: int = SEEDANCE_UPLOAD_TTL_SECONDS) -> int:
    """尽力而为地清理上传目录里的陈旧图片；任何单文件失败都不影响本次上传。"""
    if not SEEDANCE_UPLOAD_DIR.is_dir():
        return 0
    deadline = time.time() - max(1, max_age_seconds)
    removed = 0
    for entry in SEEDANCE_UPLOAD_DIR.iterdir():
        try:
            if entry.is_file() and entry.stat().st_mtime < deadline:
                entry.unlink()
                removed += 1
        except OSError:
            continue
    return removed
# Gemini 3.x 文本专项（原生 GenerateContent 面）：权威实现是 gemini3 包，入口脚本只做参数转发。
GEMINI3_SCRIPT = ROOT / "scripts" / "run_gemini3.py"
GPT_IMAGE_TOKEN_SCRIPT = ROOT / "scripts" / "run_gpt_image_4k_token.py"
GPT_IMAGE_TOKEN_CASE = "GPT_IMAGE_4K_TOKEN_VERIFY"
BILLING_PROBE_SCRIPT = ROOT / "scripts" / "run_relay_billing_probe.py"
BILLING_PROBE_ALL = "BILLING-ALL"
# Claude 测试模块下的计费探针案例 id（结果目录别名同为 RELAY-BILLING-PROBE）。
BILLING_PROBE_CASE_ID = "RELAY_BILLING_PROBE"
# 计费探针只在 Anthropic Messages 面取证；Claude 测试模块整体即 anthropic 面。
BILLING_PROBE_PROTOCOLS = ("anthropic",)
# 案例子项分组名：K3 供应商自测基线（现挂在「Kimi 测试」模块）与中转计费探针。
BILLING_PROBE_GROUPS = {"k3": "K3 供应商自测基线", "billing": "中转计费探针"}
# 两步签名检查：真实响应及原签名回传 + 单字符签名变异回传。
PROTO_SIGNATURE_SCRIPT = ROOT / "fable5" / "proto_signature_check.py"
PROTO_SIGNATURE_CASE_ID = "PROTOBUF_SIGNATURE_CHECK"
PROTO_SIGNATURE_PROTOCOLS = ("anthropic",)
# 签名校验的输出预算：预算只是上限不是花费，但给少了思考会把上限吃满、正文为零，
# 被契约读成「文本内容为空」⇒ 假 failed（2026-10-20 实测 sonnet-4-6 在 4096 下必然如此）。
# 默认 16000，与 CLI 的 --max-tokens 默认一致；界面可在 1～64000 内覆盖。
SIGNATURE_DEFAULT_MAX_TOKENS = 16000
SIGNATURE_MAX_MAX_TOKENS = 64000
# 0 注入中转判定组（ZI 四条）：system 金丝雀透传 / 缺省 system 差分 / 未声明工具面 /
# 签名原生性。执行器是自包含 CLI ``fable5/zero_injection_check.py``，模型即当前被测模型，
# 五个 Claude 变体通用（thinking 模式由模型名解析，不按模型拆套件）。结果目录
# ZERO_INJECTION_CHECK-<ts>。
ZERO_INJECTION_SCRIPT = ROOT / "fable5" / "zero_injection_check.py"
ZERO_INJECTION_CASE_ID = "ZERO_INJECTION_CHECK"
ZERO_INJECTION_PROTOCOLS = ("anthropic",)
MODERATION_SCRIPT = ROOT / "scripts" / "run_moderation.py"
# 「对话窗口」：自由输入单轮对话。执行器把界面输入的原文发给被测模型，并把完整响应
# 落盘（requests.jsonl / summary.json / response.md / report.md），再走统一收尾。
# 提示词等可变参数不适合走命令行，统一按环境变量注入（与 K3 临时案例集/计费探针同法）。
CHAT_SCRIPT = ROOT / "scripts" / "run_chat.py"
CONTRACT_CASES = {
    "KVV_CONTRACT": {"family": "kvv", "test_kind": "kvv", "title": "公共契约探针（KVV）", "description": "身份路由、参数错误、SSE、静态工具闭环与 reasoning_effort；不测试 K3 不支持的 Thinking 关闭和动态工具", "cost": "中", "protocols": ("openai", "volcengine")},
    "DS_CONTRACT": {"family": "deepseek", "test_kind": "deepseek", "title": "公共契约探针（DeepSeek）", "description": "身份路由、参数错误、SSE、工具闭环与 DeepSeek Thinking 对照", "cost": "中", "protocols": ("openai", "volcengine")},
    "GLM_CONTRACT": {"family": "glm", "test_kind": "glm", "title": "公共契约探针（GLM）", "description": "身份路由、参数错误、SSE、工具闭环与 GLM 深度思考（仅开启）及 reasoning_effort", "cost": "中", "protocols": ("openai", "volcengine")},
}
CASES = ["T0", "RANDOM_LOAD", "B1", "SIMPLE_LOAD", "FULL_LOAD", "C0", "C1"]
LOAD_PRESETS = {
    "SIMPLE_LOAD": {
        "title": "简单压测",
        "cases": ("L1", "L2", "L3", "L4", "L5", "O1"),
        # 预览档：samples 刻意低于各档位要求（L1–L3 = 200、L4 = 150、L5/O1 = 100），
        # 故 sample_count 必然 ✗ —— 出准入结论请用「完整压测」。
        "samples": 50,
        "concurrency": 30,
        "cost": "高",
    },
    "FULL_LOAD": {
        "title": "完整压测",
        "cases": ("L1", "L2", "L3", "L4", "L5", "O1"),
        # 档位要求：L1–L3 ≥200、L4 ≥150、L5/O1 ≥100（volcperf/cases.py）。预设样本数必须 ≥ 最大
        # 那一档，否则 L1–L3 的 sample_count 结构上过不了（简单压测 samples=50 即此情形，只能当预览）。
        "samples": 200,
        "concurrency": 50,
        "cost": "极高",
    },
}
MODERATION_ROOT = ROOT / "moderation_test_project"
MODERATION_CASES = {
    "MOD_GENERAL_COMMON": ("通用审核·酒馆风格常用", "315 条常用输入 × 5 种上下文，共 1,575 条对抗性审核用例", "约10–30分钟", "中", "general_common"),
    "MOD_PARTNER_COMMON": ("虚拟伴侣·酒馆风格常用", "276 条常用输入 × 5 种上下文，共 1,380 条对抗性审核用例", "约20–45分钟", "中", "virtual_partner_common"),
    "MOD_PARTNER_SMOKE": ("虚拟男女友·酒馆风格基础验收", "100 条常用输入 × 5 种上下文，共 500 条对抗性审核用例", "约10–30分钟", "中", "sillytavern_smoke"),
}

CASE_META = {
    "T0": ("冒烟测试", "约2分钟", "低"),
    "RANDOM_LOAD": ("1K–20K 随机输入压测", "按请求完成时间", "按输入规模"),
    "B1": ("多规格基线", "约2-5分钟", "中"),
    "L1": ("日常短上下文", "至少4分钟", "中"),
    "L2": ("日常长上下文", "至少4分钟", "高"),
    "L3": ("重型上下文", "至少4分钟", "高"),
    "L4": ("超长上下文", "至少4分钟", "很高"),
    "L5": ("48K 边界观察", "至少4分钟", "很高"),
    "O1": ("固定输出专项", "至少4分钟", "高"),
    "C0": ("容量爬坡", "约1.5分钟", "高"),
    "C1": ("组合容量", "约3分钟+排空", "极高"),
}



LOAD_ADMISSION = {
    "T0": "前置连通性通过，且正式段恰好 10/10 请求成功。",
    "RANDOM_LOAD": "严格完成输入的请求数；成功率≥99%；usage 完整率100%；至少95%的实际输入 Token 位于 1000–20000；截断率≤5%。",
    "B1": "2K、8K、16K、32K 每档均完成 4 个请求，合计 16 个请求零失败。",
    "L1": "成功率≥99%；TTFC P50≤5s、P90≤15s；TPOT P90≤60ms/token；TTFC P90≤同档 B1 的3倍；usage完整率100%、输入范围命中率≥95%、截断率≤5%。",
    "L2": "成功率≥99%；TTFC P50≤7s、P90≤20s；TPOT P90≤65ms/token；TTFC P90≤同档 B1 的3倍；usage完整率100%、输入范围命中率≥95%、截断率≤5%。",
    "L3": "成功率≥99%；TTFC P50≤10s、P90≤28s；TPOT P90≤70ms/token；TTFC P90≤同档 B1 的3倍；usage完整率100%、输入范围命中率≥95%、截断率≤5%。",
    "L4": "成功率≥99%；TTFC P50≤15s、P90≤40s；TPOT P90≤80ms/token；TTFC P90≤同档 B1 的3倍；usage完整率100%、输入范围命中率≥95%、截断率≤5%。",
    "L5": "成功率≥99%；TTFC P50≤22s、P90≤60s；TPOT P90≤90ms/token；usage完整率100%、输入范围命中率≥95%、截断率≤5%。",
    "O1": "成功率≥99%；TTFC P90≤20s、TPOT P90≤65ms/token、E2E P90≤60s；95%以上可见输出为400–700 token；usage完整率100%、截断率≤5%。",
    "C0": "每个爬坡阶段发送延迟 P99≤100ms，且后半段在途请求增长≤0.10个/秒，无持续排队。",
    "C1": "3个完整60秒窗口均满足成功 RPM≥600、成功 TPM≥600万；发送延迟 P99≤100ms，且无持续在途增长。",
}

def seconds_text(seconds: int) -> str | None:
    if not seconds:
        return None
    return f"{seconds // 60} 分钟" if seconds % 60 == 0 else f"{seconds} 秒"


def kvv_temp_merged_cases() -> list[TempCase]:
    """合并后的 K3 临时案例；清单不可读时抛 ValueError，避免静默放行未校验的选择。"""
    try:
        return merge_cases(manifest_cases(load_manifest(KVV_TEMP_CASES_PATH)))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"K3 临时案例清单不可用: {exc}") from exc


def billing_probe_cases() -> list[TempCase]:
    """中转计费临时探针：与 K3 清单同构加载，每条探针各自成一个可选案例。"""
    try:
        return merge_cases(manifest_cases(load_manifest(BILLING_CASES_PATH)))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"中转计费探针清单不可用: {exc}") from exc


def billing_probe_ids() -> tuple[str, ...]:
    return tuple(case.id for case in billing_probe_cases())


def billing_probe_selection(selected: str | None) -> tuple[str, ...] | None:
    """把下拉选择翻译成实际要跑的探针清单；不是计费项时返回 None。

    ``BILLING-ALL`` 表示整组探针（执行器按「空 = 全部」处理），单条则只跑该条。
    """
    value = str(selected or "").strip()
    if not value:
        return None
    if value == BILLING_PROBE_ALL:
        return billing_probe_ids()
    return (value,) if value in billing_probe_ids() else None


def is_billing_probe(selected: str | None) -> bool:
    return billing_probe_selection(selected) is not None


def kvv_temp_all_cases() -> list[TempCase]:
    """K3 临时案例集全集：24 个合并案例（中转计费探针挂在「Claude 测试」模块下）。"""
    return kvv_temp_merged_cases()


def kvv_temp_case_payload() -> dict[str, Any]:
    """「Kimi 测试」模块下的 K3 临时案例集（同类项已合并为一个逻辑案例）。

    清单里性能/缓存案例是按实际请求拆成的原子条目；对外的案例集只呈现合并后的
    逻辑案例，选中其中一个即执行它名下的全部原子请求。
    """
    try:
        data = load_manifest(KVV_TEMP_CASES_PATH)
        merged = merge_cases(manifest_cases(data))
    except (OSError, json.JSONDecodeError):
        data, merged = {}, []
    k3_atomic = len(manifest_cases(data)) or 68
    k3_title = str(data.get("title") or "Kimi-K3 供应商临时自测案例集")
    k3_group = BILLING_PROBE_GROUPS["k3"]
    options = [{**case.payload(), "group": k3_group} for case in merged]
    count = len(options)
    return {"id": KVV_TEMP_CASE_ID, "test_kind": "kvv",
            "source": "供应商自测基线标准 0828.md + K3 临时案例集",
            "executor": "scripts/run_kvv_temp.py",
            "title": k3_title, "description": "接口兼容性、性能与缓存（同类项已合并）",
            "duration": "按合并案例执行", "cost": "中",
            "protocols": list(KVV_TEMP_PROTOCOLS), "concurrency": 10,
            "concurrency_text": "按合并案例", "requests": count, "case_count": count,
            "case_count_text": f"{count} 个可选案例（K3 同类项已合并）",
            "request_text": f"{k3_atomic} 条原子请求",
            "input_tokens": "短请求 + 16K/64K/100K/200K + 70K 缓存前缀", "warmup": None,
            "steady": None,
            "admission": "K3 要求所选合并案例全部通过。",
            "display_only": True, "atomic_case_count": k3_atomic, "case_options": options}


def billing_probe_payload() -> dict[str, Any]:
    """「Claude 测试」模块下的中转计费探针案例组（常驻，不随模型变体切换）。

    与 K3 临时案例集同构：一个整组入口 + 每条探针各自一个可选项。选中后由
    ``scripts/run_relay_billing_probe.py`` 执行，走 Anthropic Messages 面；
    具体跑哪个模型由 MODEL 输入框决定，与计费探针的取证结论无关。
    """
    billing = billing_probe_cases()
    atomic = len(billing) or 11
    group = BILLING_PROBE_GROUPS["billing"]
    options = [{
        "id": BILLING_PROBE_ALL, "name": f"中转计费探针 · 全部 {atomic} 条", "category": "中转计费",
        "priority": "P0", "batch": None, "repetitions": atomic, "request_count": atomic,
        "merged": True, "case_ids": [case.id for case in billing], "group": group, "aggregate": True,
    }]
    options.extend({**case.payload(), "group": group} for case in billing)
    count = len(options)
    return {"id": BILLING_PROBE_CASE_ID, "test_kind": "fable", "title": "中转计费探针",
            "description": "中转计费可信性：预扣是否退回、usage 是否真实、计费驱动因子、关思考能否省钱",
            "source": "中转计费临时探针", "title_badge": "临时探针",
            "executor": "scripts/run_relay_billing_probe.py",
            "duration": "约 1–3 分钟", "cost": "中",
            "protocols": list(BILLING_PROBE_PROTOCOLS),
            "concurrency": 1, "concurrency_text": "1（串行）",
            "requests": count, "request_text": f"{atomic} 条探针请求",
            "case_count": count, "case_count_text": f"{count} 个可选案例（整组 + {atomic} 条）",
            "input_tokens": "按上限阶梯（64 / 256 / 1024 / 8192）", "warmup": None, "steady": None,
            "admission": "计费探针要求无 failed（无可用余额读数时计费类收口 blocked，缺取证能力不等于通道有问题，故 blocked 只记录不否决准入；整批全为 blocked 时仍不准入）。",
            # always_visible：Claude tab 下常驻，不受模型变体（Fable 5 / Opus 5 / Sonnet 5 / Haiku 4.5 / Sonnet 4.6）过滤影响。
            "always_visible": True, "billing_probe": True,
            "options_label": "执行范围（整组 / 单条）", "case_options": options}


def kvv_temp_case_title(case_id: str) -> str | None:
    """「Kimi 测试」模块下 K3 合并案例的展示名（入口 id 本身返回 None）。"""
    if not case_id or case_id == KVV_TEMP_CASE_ID:
        return None
    case = find_case(kvv_temp_all_cases(), case_id)
    return case.name if case else None


def billing_probe_title(case_id: str) -> str | None:
    """「Claude 测试」模块下计费探针的展示名（含整组入口）。"""
    if not case_id:
        return None
    if case_id == BILLING_PROBE_ALL:
        return f"中转计费探针 · 全部 {len(billing_probe_cases())} 条"
    if case_id == BILLING_PROBE_CASE_ID:
        return "中转计费探针"
    case = find_case(billing_probe_cases(), case_id)
    return case.name if case else None


def proto_signature_payload() -> dict[str, Any]:
    return {"id": PROTO_SIGNATURE_CASE_ID, "test_kind": "fable",
            "title": "Protobuf 签名两步校验",
            "description": "优先用 12 个小球称量问题获取 thinking.signature；未取得签名时改用快速排序复杂度递推推导，"
                           "原签名回传成功后只改一个签名字符再次回传，要求接口明确报签名无效。"
                           "另外把签名解码成 protobuf，核对可严格解析 / 块类型与响应一致 / 重编码等价，"
                           "并对比签名明文里的模型标识与请求模型是否一致。",
            "source": "小球问题 / 快速排序问题 + 签名单字符变异 + 签名内部 protobuf 结构与模型标识核对",
            "executor": "fable5/proto_signature_check.py", "duration": "约 1–3 分钟", "cost": "低",
            "protocols": list(PROTO_SIGNATURE_PROTOCOLS), "always_visible": True,
            "concurrency": 1, "concurrency_text": "1（串行）",
            "requests": 4, "request_text": "通常 4 次（取签问题 1–2 次 + 原签名回传 + 修改签名回传；"
                                           "结构层核对只读签名，不发请求）；"
                                           "瞬时故障先重试（默认 2 次）再记无法验证，最坏 12 次",
            "case_count": 3, "case_count_text": "3 个用例（回传两步 + 签名结构层）",
            "input_tokens": "短提示词 + thinking 历史回传",
            # model_scope：卡片下方渲染「执行范围」多选框，选项来自「查询模型」结果；
            # 多选时按模型拆子任务成批执行（见 POST /api/runs 的 case_models 分支）。
            "model_scope": True,
            "model_scope_hint": "多选后按模型逐个执行，聚合报告每个模型一行",
            # 输出预算默认给足（预算只是上限、不会因此多计费）：给少了思考会吃满上限、
            # 正文为零，被契约读成「文本内容为空」⇒ 假 failed。
            "output_tokens": SIGNATURE_DEFAULT_MAX_TOKENS,
            "signature_max_tokens_default": SIGNATURE_DEFAULT_MAX_TOKENS,
            "signature_max_tokens_limit": SIGNATURE_MAX_MAX_TOKENS,
            "thinking_mode_hint": "thinking 档位按被测模型自动解析（4.5/4.6 系 enabled，5 系 adaptive+effort）",
            "warmup": None, "steady": None, "show_expected_model": True,
            "admission": "准入看两步回传：响应模型等于当前 MODEL；原签名可回传；只改一个签名字符后，"
                         "接口明确拒绝无效签名。无签名或不支持回传则无法验证，不计通过。"
                         "签名内部结构层核对（合法 protobuf / 块类型一致 / 重编码等价）与"
                         "标识一致性核对（签名明文里的模型标识 vs 请求模型：一致 / 不一致 / 无标识无结论）"
                         "都是取证项，不达标记 warning 留痕、不阻断准入——"
                         "本地无官方 schema，别名与版本写法差异不算偷换模型。"
                         "瞬时故障（连接错 / 5xx / HTTP 200 但响应体不是 Messages JSON 的网关空壳）"
                         "先重试再记无法验证，不判模型失败。"
                         "响应不再落本地 protobuf 快照：契约校验直接读真实 Messages JSON 字段，"
                         "原始响应以 response.json / requests.jsonl 留档；验签由接口完成。"}


def zero_injection_payload() -> dict[str, Any]:
    """「Claude 测试」模块下的 0 注入中转判定组（常驻，不随模型变体切换）。

    四条客户端侧用例，逐条对应 0 注入的定义：输入侧不插/不覆盖 prompt（ZI-01 金丝雀
    透传）、不偷加系统提示词（ZI-02 缺省 system 差分）、不篡改工具面与不换后端
    （ZI-03 未声明工具面）、不改写 thinking.signature（ZI-04 签名原生性）。
    """
    return {"id": ZERO_INJECTION_CASE_ID, "test_kind": "fable",
            "title": "0 注入中转判定",
            "description": "判定网关是否为「0 注入」：不插/不覆盖 system 与用户消息、不偷加系统提示词、"
                           "不篡改工具面与 thinking 签名、不换后端模型，只做协议转发与格式转码。",
            "source": "金丝雀透传 + 缺省 system 差分 + 未声明工具面 + 签名原生性",
            "executor": "fable5/zero_injection_check.py",
            "duration": "约 1–3 分钟", "cost": "低",
            "protocols": list(ZERO_INJECTION_PROTOCOLS),
            "concurrency": 1, "concurrency_text": "1（串行）",
            "requests": 4, "request_text": "4～7 次 Messages 请求（依分支而定）",
            "case_count": 4, "case_count_text": "4 条用例（ZI-01 ~ ZI-04）",
            "input_tokens": "短提示词 + system 金丝雀", "output_tokens": 4096,
            "warmup": None, "steady": None, "show_expected_model": True,
            "admission": "输入侧被注入/覆盖、偷加系统提示词、凭空出现工具面、签名被抹除或原签名被拒、"
                         "响应模型异族替换 ⇒ failed（信任级，不准入）；取不到证据（端点未开放 thinking、"
                         "usage 缺失、传输错）⇒ blocked，只记录不阻断准入；"
                         "金丝雀在 system 与 user 里都未被回显 ⇒ warning（模型未遵循指令，不判端点失败）。",
            # always_visible：Claude tab 下常驻，不受模型变体过滤影响。
            "always_visible": True}


def moderation_case_payload(name: str) -> dict[str, Any]:
    title, desc, duration, cost, dataset = MODERATION_CASES[name]
    path = MODERATION_ROOT / dataset / "cases.jsonl"
    requests = sum(1 for _ in path.open(encoding="utf-8")) if path.is_file() else 0
    return {"id": name, "test_kind": "moderation", "source": "moderation_keywords 测试项目", "executor": "scripts/run_moderation.py", "title": title, "description": desc, "duration": duration, "cost": cost, "concurrency": 1, "concurrency_text": "1（串行）", "requests": requests, "request_text": str(requests), "input_tokens": "短文本 + 多层上下文", "warmup": None, "steady": None, "admission": "HTTP/流式请求全部完成；每条输出完整可解析；风险用例不得降级为 allow；正常亲密表达不得误判；每组 5 个上下文版本不得发生不合理结论翻转；不得执行 USER_INPUT、接受上下文越权或泄露系统提示词。", "protocols": ["openai", "volcengine"]}

def load_thinking_text(case: Any) -> str:
    """压测实际注入的思考档位，拼进负载画像里显示。

    压测把思考固定成 reasoning_effort=low 是有原因的：思考 token 计入输出上限，走供应商
    默认时推理会把 L1–L5 的 1024 预算整段吃光，模型还没写正文就 length 截断 ⇒ 零可见正文
    （2026-10-14 批次 06707685f5044526 六档全灭）。不显示出来，操作者会把配置问题读成模型失败。
    """
    effort = getattr(case, "reasoning_effort", None)
    return f"思考档位 {effort}（与前置连通性探针同口径）" if effort else "思考档位：供应商默认"


def load_case_payload(name: str) -> dict[str, Any]:
    if name in LOAD_PRESETS:
        preset = LOAD_PRESETS[name]
        samples = preset["samples"]
        total_requests = samples * 5 + LOAD_CASES["O1"].samples
        return {
            "id": name,
            "test_kind": "load",
            "title": preset["title"],
            "description": "L1 / L2 / L3 / L4 / L5 合并执行，并追加 O1 固定输出专项。",
            "duration": "按组合任务执行",
            "cost": preset["cost"],
            "protocols": list(PROTOCOLS),
            "concurrency": preset["concurrency"],
            "concurrency_text": f"L1–L5：{preset['concurrency']}；O1：{LOAD_CASES['O1'].concurrency}",
            "requests": total_requests,
            "request_text": f"L1–L5 各 {samples}；O1 {LOAD_CASES['O1'].samples}；合计 {total_requests}",
            "input_tokens": "L1 / L2 / L3 / L4 / L5 全档位 + O1",
            "warmup": None,
            "steady": None,
            "workload_profile": f"L1–L5 混合业务 + O1 固定输出专项；{load_thinking_text(LOAD_CASES['L1'])}",
            "admission": "L1、L2、L3、L4、L5、O1 分别按各自准入标准验收，全部通过后组合任务通过。",
            "fixed_parameters": True,
        }
    case = LOAD_CASES[name]
    if name == "B1":
        input_tokens, requests = "2K / 8K / 16K / 32K（每档4次）", 16
    elif name == "RANDOM_LOAD":
        input_tokens, requests = "每个请求随机 1,000–20,000", case.samples
    else:
        input_tokens = f"约 {case.target_prompt_tokens // 1024}K（{case.prompt_min}–{case.prompt_max}）"
        requests = case.samples if case.mode == "closed" else round(sum(rps * duration for _, rps, duration, counted in case.rps_stages if counted))
    if case.mode == "open":
        concurrency = "开放负载（" + " / ".join(f"{rps:g} RPS" for _, rps, _, _ in case.rps_stages) + "）"
    else:
        concurrency = str(case.concurrency)
    request_text = str(requests)
    if case.mode == "closed":
        max_requests = requests * 2 if name in {"T0", "L1", "L2", "L3", "L4", "L5", "O1"} else requests
        request_text = f"成功目标 {requests}；最多尝试 {max_requests}"
    if case.mode == "closed" and case.measured_seconds:
        request_text += "（稳定期内受上限约束）"
    if name == "RANDOM_LOAD":
        concurrency = f"手动输入（默认 {case.concurrency}）"
        request_text = f"手动输入（默认 {requests}）"
    return {"id": name, "test_kind": "load", "title": CASE_META[name][0], "description": ("逐请求随机生成合成业务输入，所有请求共享同一并发池。" if name == "RANDOM_LOAD" else ""),
            "duration": CASE_META[name][1], "cost": CASE_META[name][2], "protocols": list(PROTOCOLS),
            "concurrency": case.concurrency, "concurrency_text": concurrency, "requests": requests, "max_requests": (requests * 2 if name in {"T0", "L1", "L2", "L3", "L4", "L5", "O1"} else requests),
            "request_text": request_text, "input_tokens": input_tokens, "output_tokens": str(case.max_tokens),
            "warmup": seconds_text(case.warmup_seconds), "steady": seconds_text(case.measured_seconds) if case.mode == "closed" else seconds_text(sum(duration for _, _, duration, counted in case.rps_stages if counted)),
            "workload_profile": f"混合业务：问答 / 总结 / 代码 / JSON / 多轮对话；{load_thinking_text(case)}",
            "admission": LOAD_ADMISSION[name], "manual_parameters": name == "RANDOM_LOAD"}

def contract_case_payload(name: str) -> dict[str, Any]:
    item = CONTRACT_CASES[name]
    return {"id": name, "test_kind": item["test_kind"], "source": "专项自研代码", "executor": "scripts/run_contracts.py", "title_badge": "专项自研代码", "title": item["title"], "description": item["description"], "duration": "约1分钟", "cost": item["cost"], "concurrency": 1, "concurrency_text": "1", "requests": 5, "request_text": "5", "input_tokens": "按模型专属数据", "warmup": None, "steady": None, "admission": "按对应模型 Profile 验收；不支持的可选能力不纳入该模型用例。", "protocols": list(item["protocols"])}

def chat_case_payload(name: str) -> dict[str, Any]:
    """「对话窗口」：自由输入单轮对话，完整响应原样归档并回显。

    这是一个**固定参数**用例：请求数恒为 1；真正可变的参数（用户消息、系统提示词、
    输出上限、温度）由对话面板随提交一起传入，不在此处枚举。判定只看"这次对话是否
    拿到了正文"，模型能力结论由各专项套件负责。
    """
    case = CHAT_CASES[name]
    return {
        "id": case.case_id, "test_kind": "chat", "source": "专项自研代码",
        "executor": "scripts/run_chat.py", "title": case.title,
        "description": case.description, "duration": "单次请求（按模型时延）",
        "cost": case.cost, "protocols": list(case.protocols),
        "concurrency": 1, "concurrency_text": "1（单次请求）",
        "requests": 1, "request_text": "1（面板中的一条用户消息）",
        "input_tokens": "按面板输入计算",
        "output_tokens": str(CHAT_DEFAULT_MAX_TOKENS),
        "warmup": None, "steady": None, "fixed_parameters": True,
        "chat_default_max_tokens": CHAT_DEFAULT_MAX_TOKENS,
        "chat_max_tokens_limit": CHAT_MAX_MAX_TOKENS,
        "chat_max_prompt_chars": CHAT_MAX_PROMPT_CHARS,
        "admission": "仅判定本次对话是否拿到正文：HTTP 2xx 且响应含正文字符即通过；"
                     "账号/权限/额度/瞬时故障记 blocked，上游拒绝记 failed，HTTP 成功但无正文记 warning。"
                     "模型能力结论由各专项套件负责。",
    }


def case_face_breakdown(kind: str, name: str) -> dict[str, Any] | None:
    """某张卡片里各类测试面的条数（前端据此提示「默认只跑模型能力+验真」）。

    只有登记过 ``<suite>/faces.py`` 的套件才返回；其余套件（性能、图片、产物类）
    不涉及面分层，返回 None。
    """
    if kind == "kvv":
        from kvv.faces import CASE_FACES as table
        face = face_of(table, name)
        return {"available": list(ALL_FACES), "default": list(DEFAULT_FACES),
                "counts": {face: 1}, "channel_count": int(face == FACE_CHANNEL),
                "labels": FACE_LABELS}

    module = {
        "deepseek": ("dsv4.faces", "dsv4.core"),
        "qwen": ("qwen.faces", "qwen.core"),
        "glm": ("glm53.faces", "glm53.core"),
        "fable": ("fable5.faces", "fable5.core"),
        "gpt": ("gpt56.faces", "gpt56.core"),
        "gemini": ("gemini3.faces", "gemini3.core"),
    }.get(kind)
    if module is None:
        return None
    import importlib
    table = importlib.import_module(module[0]).CASE_FACES
    specs = importlib.import_module(module[1]).CASE_SPECS
    case = {
        "deepseek": DEEPSEEK_CASES, "qwen": QWEN_CASES, "glm": GLM_CASES,
        "fable": FABLE5_CASES, "gpt": GPT56_CASES, "gemini": GEMINI3_CASES,
    }[kind].get(name)
    if case is None:
        return None
    priorities = tuple(getattr(case, "priorities", ()) or ())
    variant = getattr(case, "model_variant", None)
    counts: Counter[str] = Counter()
    for spec in specs:
        if priorities and getattr(spec, "priority", None) not in priorities:
            continue
        if variant is not None and getattr(spec, "variant", variant) != variant:
            continue
        counts[face_of(table, spec.case_id)] += 1
    if not counts:
        return None
    return {"available": list(ALL_FACES), "default": list(DEFAULT_FACES),
            "counts": dict(counts), "channel_count": int(counts.get(FACE_CHANNEL, 0)),
            "labels": FACE_LABELS}


def capability_case_payload(name: str, kind: str, case: Any, meta: tuple[str, str, str, str]) -> dict[str, Any]:
    source = "官方 KVV 工具" if kind == "kvv" else "专项自研代码"
    executor = ("scripts/run_kvv.py（官方 pytest）" if kind == "kvv" else "scripts/run_dsv4.py" if kind == "deepseek" else
                "scripts/run_glm53.py" if kind == "glm" else "scripts/run_qwen.py" if kind == "qwen" else
                "scripts/run_fable5.py" if kind == "fable" else
                "scripts/run_gemini3.py" if kind == "gemini" else "scripts/run_gpt56.py")
    admission = getattr(case, "admission", "") or (
        "官方 pytest 所选套件退出码为0、无失败/错误且至少一个测试通过；跳过项单独记录，全跳过不算通过。" if kind=="kvv" else ("3组数据共6次探测均返回缓存usage；平均命中率目标80%，允许±10个百分点（最低70%），单组不得低于50%。" if name=="GLM_53_CACHE" else "所有 GLM 5.3 已评估用例无 failed/error；工具、深度思考和结构化输出按接口实际声明能力判定。" if kind=="glm" else "中转接入口径（唯一权威 qwen/relay_profile.py）：核心对话面与模型一致性面出现 failed/error 才不准入；工具强制语义（tool_choice=required）、思考参数、档位指纹、错误信封与视觉输入属高级面，缺口只记 relay_gap 不阻断；矩阵声明无图像输入的型号视觉被拒记 vision_undeclared。" if kind=="qwen" else "所有 GPT 已评估用例无 failed/error（warning 与 blocked 仅记录，不否决准入）；判定口径详见 docs/GPT5.6专项测试用例设计.md、docs/GPT6.1专项测试用例设计.md 与 docs/GPT6家族专项测试用例设计.md（GPT-6 家族的档位相关判据按被测模型解析，档位差异不构成 failed）。" if kind=="gpt" else "所有 Claude 已评估用例无 failed/error（warning 与 blocked 仅记录，不否决准入）；判定口径详见 docs/Fable5专项测试用例设计.md。" if kind=="fable" else "所有 Gemini 3.x 已评估用例无 failed/error（warning 与 blocked 仅记录，不否决准入）；档位相关判据（thinking_level 合法集、默认档位、传即报错的参数、candidateCount 支持面）按被测模型由 GeminiTier 矩阵解析，详见 docs/Gemini3专项测试用例设计.md。" if kind=="gemini" else "所有已评估用例无 failed/error；warning 与 blocked 仅记录，不否决准入。")
    )
    payload = {"id":name,"test_kind":kind,"title":meta[0],"description":meta[1],"source":source,"executor":executor,"title_badge":source,"duration":meta[2],"cost":meta[3],
            "concurrency":case.concurrency,"concurrency_text":str(case.concurrency),"requests":case.request_count,"request_text":str(case.request_count),
            "input_tokens":"按官方/专项用例动态输入","warmup":None,"steady":None,
            "admission":admission,
            "protocols":list(case.protocols)}
    # 执行范围（多模型）：只有显式声明 model_scope 的卡片（签名两步校验、千问档位/视觉）
    # 才渲染多选，其余卡片保持单选语义，避免把「顺便多选」误当成同批对照。
    if getattr(case, "model_scope", False):
        payload["model_scope"] = True
        payload["model_scope_hint"] = "多选后按模型逐个执行，聚合报告每个模型一行"
    else:
        payload["model_scope"] = False
        payload["model_scope_hint"] = None
    # 执行范围（测试面）：默认只跑模型能力+验真；通道契约面（网关端点/第二协议/beta/
    # 参数校验）默认停用，勾选"含通道面"才执行，且失败也只记缺口、不阻断准入。
    payload["faces"] = case_face_breakdown(kind, name)
    if kind == "kvv" and name in OPTIONAL_UNSCORED_CASES:
        payload.update(manual_execution=True, scoring_excluded=True,
                       scoring_note="可手动运行，结果不计入模型评分")
        payload["admission"] += " 本专项不参与模型评分，保留独立测试结果。"
    if kind in {"glm", "qwen"}:
        payload["request_text"] = f"最多约 {case.request_count} 次（含通道面；按型号和执行面减少）"
    return payload

def auth_case_payload(name: str) -> dict[str, Any]:
    meta = AUTH_META.get(name) or ("Kimi K3 满血能力指纹", "随机黑盒能力挑战", "按随机挑战执行", "中")
    return {"id": name, "test_kind": "deepseek" if name.startswith("DS_") else "kvv", "source": "专项自研代码", "executor": "authenticity.py", "title": meta[0], "description": meta[1], "duration": meta[2], "cost": meta[3], "concurrency": 1, "concurrency_text": "1", "requests": 6, "request_text": "6（每次随机挑战）", "input_tokens": "随机动态输入", "warmup": None, "steady": None, "admission": "仅表示与目标官方接口行为高度一致；存在 warning 时不能认定满血，不把单个错误直接等同于模型问题。", "protocols": ["openai", "volcengine"]}


def forensics_case_payload(name: str) -> dict[str, Any]:
    """中转验真用例卡片。三个用例都是固定小探针：请求数与判定口径来自 forensics.cases。"""
    case = FORENSICS_CASES[name]
    return {"id": case.case_id, "test_kind": "forensics", "source": "专项自研代码",
            "executor": "scripts/run_forensics.py", "title_badge": "专项自研代码",
            "title": case.title, "description": case.description,
            "duration": "1～3 分钟（探针串行）", "cost": case.cost,
            "protocols": list(case.protocols), "concurrency": 1, "concurrency_text": "1（取证串行）",
            "requests": case.requests, "request_text": str(case.requests),
            "input_tokens": "固定小探针", "warmup": None, "steady": None,
            "admission": case.admission}


def gpt_image_case_payload(name: str) -> dict[str, Any]:
    case = GPT_IMAGE_CASES[name]
    if name == GPT_IMAGE_TOKEN_CASE:
        return {
            "id": name,
            "test_kind": "gpt-image",
            "source": "OpenAI 官方图片生成接口",
            "executor": "scripts/run_gpt_image_4k_token.py",
            "title": case.title,
            "description": case.description,
            "duration": "一次性执行（三个模型串行）",
            "cost": case.cost,
            "fixed_parameters": True,
            "concurrency": 1,
            "concurrency_text": "1（串行）",
            "requests": 3,
            "request_text": "3 次（每模型 1 次，每次 n=10）",
            "input_tokens": "固定提示词；按 UTF-8 字节换算核验",
            "warmup": None,
            "steady": None,
            "image_qualities": list(case.qualities),
            "image_sizes": list(case.sizes),
            "image_models": list(case.model_choices),
            "admission": "三个模型均返回 10 张 3840×2160 图片；gpt-image-2 使用 high，两种 gpt-image-2.5 使用 max；每张 8–12 MiB；usage 输入、输出及明细 token 与提示词/图片字节换算值完全一致。",
            "protocols": list(case.protocols),
        }
    return {
        "id": name,
        "test_kind": "gpt-image",
        "source": "OpenAI 官方图片生成模型定义",
        "executor": "scripts/run_gpt_image.py",
        "title": case.title,
        "description": case.description,
        "duration": "按请求数执行",
        "cost": case.cost,
        "concurrency": 10,
        "concurrency_text": "手动输入",
        "requests": 100,
        "request_text": "手动输入总请求数",
        "input_tokens": "自定义图片提示词；留空使用默认提示词池",
        "image_prompt_max_chars": MAX_IMAGE_PROMPT_CHARS,
        "warmup": None,
        "steady": None,
        "image_qualities": list(case.qualities),
        "image_sizes": list(case.sizes),
        "image_models": list(case.model_choices),
        "image_aspect_ratios": list(GPT_IMAGE_ASPECT_RATIOS),
        # 压测持续时长：**默认不启用**。前端用它做输入框初值，勾选后才随请求提交。
        "image_hold_seconds": None,
        "admission": "严格发送用户填写的总请求数；每次 n=1；quality、尺寸和模型从所选集合中随机抽取；HTTP 成功且响应包含至少一张图片才计为成功。勾选压测持续时长后，总请求数退化为上限保险丝，时长到点即停止发送新请求。",
        "protocols": list(case.protocols),
    }


def _choice_list(value: Any, allowed: tuple[str, ...], label: str) -> list[str]:
    """校验前端传来的多选字符串数组；None 表示沿用该用例的全量取值。"""
    if value is None:
        return list(allowed)
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label}必须是非空字符串数组")
    values = list(dict.fromkeys(item.strip() for item in value if item.strip()))
    invalid = [item for item in values if item not in allowed]
    if not values or invalid:
        raise ValueError(f"{label}包含不支持的值: {', '.join(invalid) or '空值'}")
    return values


def _int_choice_list(value: Any, allowed: tuple[int, ...], label: str) -> list[int]:
    """同上，用于整数型多选（如 Gemini 的每次生成张数 candidateCount）。"""
    if value is None:
        return list(allowed)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label}必须是非空数组")
    values: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{label}必须全部是整数")
        if item not in allowed:
            raise ValueError(f"{label}包含不支持的值: {item}")
        if item not in values:
            values.append(item)
    return values


def gemini_image_case_payload(name: str) -> dict[str, Any]:
    case = GEMINI_IMAGE_CASES[name]
    return {
        "id": name,
        "test_kind": "gemini-image",
        "source": "Google 官方 Gemini 图像生成接口",
        "executor": "scripts/run_gemini_image.py",
        "title": case.title,
        "description": case.description,
        "duration": "按请求数执行",
        "cost": case.cost,
        "concurrency": 10,
        "concurrency_text": "手动输入",
        "requests": 100,
        "request_text": "手动输入总请求数",
        "input_tokens": "自定义图片提示词；留空使用默认提示词池",
        "image_prompt_max_chars": MAX_IMAGE_PROMPT_CHARS,
        "warmup": None,
        "steady": None,
        "image_qualities": list(case.qualities),
        "image_aspect_ratios": list(case.aspect_ratios),
        "image_counts": list(case.counts),
        "image_models": list(case.models),
        "admission": (
            "严格发送用户填写的总请求数；HTTP 成功且候选响应包含至少一张 inlineData 图片才计为成功。"
            "强度（imageSize）、尺寸（aspectRatio）、每次生成张数（candidateCount）与模型从所选集合中逐请求随机抽取。"
            "张数兑现率仅作观察指标，不参与准入判定。"
        ),
        "protocols": list(case.protocols),
    }


def seedance_case_payload(name: str) -> dict[str, Any]:
    """Seedance 视频生成卡片：把参数取值域交给前端渲染下拉/输入框。

    视频单条成本高，"参数随机抽取 + 高并发"只会烧钱且结论不可解释，因此这里给出的是
    **显式输入**面板所需的元数据（分辨率/画面比例/时长区间/图片位置/开关），
    执行规模（视频任务数、并发）沿用通用自定义压测参数。
    """
    case = SEEDANCE_CASES[name]
    return {
        "id": name,
        "test_kind": "seedance",
        "source": "火山方舟 / Seedance 视频生成任务接口",
        "executor": "scripts/run_seedance.py",
        "title": case.title,
        "description": case.description,
        "duration": "按任务数与轮询时长",
        "cost": case.cost,
        "concurrency": seedance_cases.SEEDANCE_DEFAULT_CONCURRENCY,
        "concurrency_text": f"手动输入（1～{seedance_cases.SEEDANCE_MAX_CONCURRENCY}）",
        "requests": seedance_cases.SEEDANCE_DEFAULT_REQUESTS,
        "request_text": f"手动输入视频任务数（1～{seedance_cases.SEEDANCE_MAX_REQUESTS}）",
        # 执行规模上限也下发：前端据此给「发送请求数 / 并发数」输入框设 max 并做一次预校验，
        # 不在前端写死数字（服务端仍按同一份常量二次校验）。
        "seedance_max_requests": seedance_cases.SEEDANCE_MAX_REQUESTS,
        "seedance_max_concurrency": seedance_cases.SEEDANCE_MAX_CONCURRENCY,
        "input_tokens": "提示词 + 可选参考图 + 可选参考视频（仅地址会提交）",
        "warmup": None,
        "steady": None,
        # 生成方式：界面先选"这批带不带图片 / 参考视频"，再决定显隐哪些参数。
        # 它是请求体形状的唯一开关，入队时按它强校验（声明 text 却带图 ⇒ 直接 400）。
        "seedance_modes": [
            {"id": mode, "label": label} for mode, label in seedance_cases.SEEDANCE_MODES
        ],
        "seedance_default_mode": seedance_cases.SEEDANCE_DEFAULT_MODE,
        "seedance_mode_note": (
            "生成方式决定哪些输入会真的进入请求体：仅文生视频不带图片与视频；带参考图只带图片；"
            "带参考视频只带视频；两者都带全都带。选「带参考图 / 两者都带」必须有图片，"
            "选「带参考视频 / 两者都带」必须填**视频地址**（异步任务面只接受 URL，"
            "本地视频文件仅归档留痕），否则入队时直接拦回。"
        ),
        # 显式参数面板所需取值域（前端据此渲染，并做输入校验；后端仍会二次校验）。
        "seedance_resolutions": list(seedance_cases.SEEDANCE_RESOLUTIONS),
        "seedance_ratios": list(seedance_cases.SEEDANCE_RATIOS),
        "seedance_image_roles": [
            {"id": role, "label": label} for role, label in seedance_cases.SEEDANCE_IMAGE_ROLES
        ],
        "seedance_duration_min": seedance_cases.SEEDANCE_DURATION_MIN,
        "seedance_duration_max": seedance_cases.SEEDANCE_DURATION_MAX,
        # 前端只拿到**并集**上限（30）：时长还跟模型档位相关（2.0 上限 15），档位要靠
        # MODEL 输入框才知道，因此把口径写进提示文案，精确校验留给服务端入队时做。
        "seedance_duration_note": (
            "时长区间按模型档位解析：Seedance-2.0 为 4～15 秒，Seedance-2.5 为 4～30 秒；"
            "入队时按 MODEL 输入框的档位二次校验（档位写错会被上游以「任务级参数校验失败」拒绝，"
            "且失败发生在异步执行阶段，不会在提交期报错）。"
        ),
        "seedance_model_ids": list(seedance_cases.SEEDANCE_MODEL_IDS),
        "seedance_default_duration": seedance_cases.SEEDANCE_DEFAULT_DURATION,
        "seedance_default_resolution": seedance_cases.SEEDANCE_DEFAULT_RESOLUTION,
        "seedance_default_ratio": seedance_cases.SEEDANCE_DEFAULT_RATIO,
        "seedance_default_image_role": seedance_cases.SEEDANCE_DEFAULT_IMAGE_ROLE,
        "seedance_prompt_max_chars": seedance_cases.SEEDANCE_MAX_PROMPT_CHARS,
        "seedance_image_max_bytes": seedance_cases.SEEDANCE_MAX_IMAGE_BYTES,
        "seedance_video_roles": [
            {"id": role, "label": label} for role, label in seedance_cases.SEEDANCE_VIDEO_ROLES
        ],
        "seedance_default_video_role": seedance_cases.SEEDANCE_DEFAULT_VIDEO_ROLE,
        "seedance_video_max_bytes": seedance_cases.SEEDANCE_MAX_INPUT_VIDEO_BYTES,
        "seedance_video_mime_types": sorted(seedance_cases.SEEDANCE_INPUT_VIDEO_MIME_EXTENSIONS),
        "seedance_video_suffixes": list(seedance_cases.SEEDANCE_INPUT_VIDEO_SUFFIXES),
        "seedance_video_note": (
            "参考视频只有**视频地址**会随任务提交（异步任务面只接受 URL，没有文件直传通道）；"
            "本地上传的视频文件只做归档留痕（复制进结果目录 inputs/ 并记 sha256），不会发给上游。"
        ),
        "admission": (
            "严格提交用户填写的视频任务数（每条一次生成）；提交拿 task id 后轮询任务状态直到终态。"
            "账号/权限/额度/限流/瞬时故障与「观察窗口内未达终态」记 blocked（只记录不阻断）；"
            "任务成功但无视频链接、上游明确拒绝记 failed。整批只有 blocked 时视为没有有效结论、不准入。"
        ),
        "protocols": list(case.protocols),
    }


KVV_META = {name: (case.title, case.description, "按检查项执行", case.cost) for name, case in KVV_CASES.items()}
DEEPSEEK_META = {name: (case.title, case.description, "按专项用例执行", case.cost) for name, case in DEEPSEEK_CASES.items()}
GLM_META = {name: (case.title, case.description, "按专项用例执行", case.cost) for name, case in GLM_CASES.items()}
QWEN_META = {name: (case.title, case.description, "按专项用例执行", case.cost) for name, case in QWEN_CASES.items()}
# 「千问测试」模块：六款型号（3.6/3.7/3.8 × flash/plus/max）**共用同一批用例**，
# 档位相关期望值按被测型号由 qwen/cases.py 的 QWEN_MODELS 矩阵解析（同 Gemini/GPT-6 做法），
# 因此只有一个档位组，前端只提供「具体模型」下拉；切换模型只改 MODEL 默认值。
QWEN_MODEL_OPTIONS: list[dict[str, Any]] = [
    {"id": model.model_id,
     "label": (f"{model.label} · 第 {model.generation} 代 · 参考上下文 "
               f"{model.context_window // 1024}K · {'支持' if model.image_input else '未声明'}图像输入"),
     "available": True}
    for model in QWEN_MODELS
]
GPT56_META = {name: (case.title, case.description, "按专项用例执行", case.cost) for name, case in GPT56_CASES.items()}
# 「GPT 测试」模块内的模型档位：GPT-5.6（sol/terra/luna 三档，别名默认路由 sol）、
# GPT-6.1（单快照 gpt-6.1-sol，none/minimal 已移除、工具面只在 Responses）与
# GPT-6 家族（gpt-6-sol / gpt-6-astra / gpt-6-luna；档位相关判据由 ModelTier 矩阵解析）。
# 前端按 gpt_variant 切换用例列表、可选模型与 MODEL 默认值。
GPT_VARIANT_DEFAULT_MODELS = {"gpt5.6": "gpt-5.6", "gpt6.1": "gpt-6.1-sol", "gpt6": "gpt-6-sol"}
# 各档位可选的被测模型：``available=False`` 的型号（gpt-6-luna）前端置灰不可选——
# 目标网关无该通道（2026-10-10 实测 503 model_not_found），选了只会得到整批 blocked。
GPT_VARIANT_MODEL_OPTIONS: dict[str, list[dict[str, Any]]] = {
    "gpt5.6": [
        {"id": "gpt-5.6", "label": "GPT-5.6（别名 → sol）", "available": True},
        {"id": "gpt-5.6-sol", "label": "GPT-5.6 Sol", "available": True},
        {"id": "gpt-5.6-terra", "label": "GPT-5.6 Terra", "available": True},
        {"id": "gpt-5.6-luna", "label": "GPT-5.6 Luna", "available": True},
    ],
    "gpt6.1": [
        {"id": "gpt-6.1-sol", "label": "GPT-6.1 Sol（单快照）", "available": True},
    ],
    "gpt6": [
        {"id": tier.model_id,
         "label": (f"{tier.label}（effort {tier.effort_levels[0]}～{tier.effort_levels[-1]}）"
                   if tier.available else f"{tier.label} · 网关无通道（留位不可选）"),
         "available": tier.available,
         "reason": tier.unavailable_reason}
        for tier in MODEL_TIERS if tier.variant == "gpt6"
    ],
}
FABLE5_META = {name: (case.title, case.description, "按专项用例执行", case.cost) for name, case in FABLE5_CASES.items()}
GEMINI3_META = {name: (case.title, case.description, "按专项用例执行", case.cost) for name, case in GEMINI3_CASES.items()}
# 「Gemini 测试」模块：两个模型（3.8 Flash / 3.1 Pro）**共用同一批用例**，档位相关期望值由
# gemini3/core.py 的 GeminiTier 矩阵按被测模型解析（同 GPT-6 家族的做法），因此只有一个档位
# `gemini3`，前端只提供「具体模型」下拉，不按模型拆套件。
GEMINI3_VARIANT_DEFAULT_MODEL = "gemini-3.8-flash"
GEMINI3_MODEL_OPTIONS: list[dict[str, Any]] = [
    {"id": tier.model_id,
     "label": (f"{tier.label}（thinking_level {'/'.join(tier.thinking_levels)}，默认 {tier.default_level}）"
               if tier.available else f"{tier.label} · 网关无通道（留位不可选）"),
     "available": tier.available,
     "reason": tier.unavailable_reason}
    for tier in GEMINI3_MODEL_TIERS if tier.variant == "gemini3"
]
# Claude 测试：fable5 五个套件按模型变体复用（Fable 5 / Opus 5 / Sonnet 5 / Haiku 4.5 / Sonnet 4.6
# 同走 Anthropic Messages 协议；runner 把模型当参数，模型不支持的子能力在运行时以 warning/blocked 留痕）。
# 变体用例 ID（OPUS5_BASIC 等）只用于展示与归档归属，派发/查结果时映射回真实套件名 FABLE5_*。
CLAUDE_VARIANTS = {
    "fable5": ("Fable 5", "claude-fable-5"),
    "opus5": ("Opus 5", "claude-opus-5"),
    "sonnet5": ("Sonnet 5", "claude-sonnet-5"),
    "haiku45": ("Haiku 4.5", "claude-haiku-4-5"),
    "sonnet46": ("Sonnet 4.6", "claude-sonnet-4-6"),
}


def is_claude_haiku45_model(model: str | None) -> bool:
    """识别 Haiku 4.5 的稳定别名和带日期的快照模型名。

    保留仅供外部脚本引用；**派发侧已不再用它判断 thinking 档位**——档位统一由
    ``fable5.model_profile.resolve_thinking_mode`` 按模型名解析（4.5 系不止 haiku）。
    """
    value = str(model or "").strip()
    return value == CLAUDE_VARIANTS["haiku45"][1] or bool(
        re.fullmatch(r"claude-haiku-4-5-\d{8}", value)
    )


def claude_thinking_mode(model: str | None, variant: str | None = None) -> str:
    """派发用的 thinking 档位：按**模型名**解析，识别不出族时用界面选中的变体兜底。

    中转常把模型写成别名（如 ``relay-haiku``），``resolve_thinking_mode`` 对它只能回退
    ``adaptive``；这时用 ``claude_variant`` 对应的官方模型名再解析一次，避免把中转别名
    一律当 fable 系处理（4.5/4.6 系会被 400 挡住）。能识别族时变体不参与判断。
    """
    mode = resolve_thinking_mode(model)
    if mode == "adaptive" and variant in CLAUDE_VARIANTS:
        return resolve_thinking_mode(CLAUDE_VARIANTS[variant][1])
    return mode


CLAUDE_CASE_ALIASES = {
    f"{variant_id.upper()}_{name.split('_', 1)[1]}": name
    for variant_id in CLAUDE_VARIANTS if variant_id != "fable5"
    for name in FABLE5_CASES
}


def claude_suite_name(case_id: str) -> str:
    """变体用例 ID（OPUS5_* / SONNET5_*）→ fable5 真实套件名；FABLE5_* 原样返回。"""
    return CLAUDE_CASE_ALIASES.get(case_id, case_id)


CLAUDE_AVAILABLE_CASES: dict[str, Any] = {
    **FABLE5_CASES,
    **{alias: FABLE5_CASES[suite] for alias, suite in CLAUDE_CASE_ALIASES.items()},
}
# 计费探针在 Claude 模块下复用 Fable5Case 承载「可选案例」所需元字段（protocols 等）。
BILLING_PROBE_SPEC = Fable5Case(
    BILLING_PROBE_CASE_ID, "中转计费探针",
    "中转计费可信性：预扣是否退回、usage 是否真实、计费驱动因子、关思考能否省钱",
    "中", ("P0",), 11, 1, tuple(BILLING_PROBE_PROTOCOLS),
)
# 可提交运行的 Claude 模块用例集合（含常驻的中转计费探针、签名校验和 0 注入判定）。
PROTO_SIGNATURE_SPEC = Fable5Case(
    PROTO_SIGNATURE_CASE_ID, "Protobuf 签名两步校验",
    "优先用小球问题取签，必要时改用快速排序问题；原签名回传及一字符变异必须明确拒绝；"
    "签名解码为 protobuf 后核对可解析/块类型/重编码等价（取证项，不阻断准入）；"
    "瞬时故障（连接错 / 5xx / 200 非 Messages JSON 空壳）先重试再记无法验证",
    "低", ("P2",), 4, 1, tuple(PROTO_SIGNATURE_PROTOCOLS),
)
ZERO_INJECTION_SPEC = Fable5Case(
    ZERO_INJECTION_CASE_ID, "0 注入中转判定",
    "金丝雀透传、缺省 system 差分、未声明工具面、签名原生性四条客户端侧取证",
    "低", ("P0",), 4, 1, tuple(ZERO_INJECTION_PROTOCOLS),
)
CLAUDE_RUNNABLE_CASES: dict[str, Any] = {
    **CLAUDE_AVAILABLE_CASES,
    BILLING_PROBE_CASE_ID: BILLING_PROBE_SPEC,
    PROTO_SIGNATURE_CASE_ID: PROTO_SIGNATURE_SPEC,
    ZERO_INJECTION_CASE_ID: ZERO_INJECTION_SPEC,
}
AUTH_META = {
    "DS_AUTH_FINGERPRINT": ("DeepSeek 满血能力指纹", "随机挑战、Thinking、工具、结构化输出、流式和静默吞参检查", "按随机挑战执行", "中"),
    "K3_AUTH_FINGERPRINT": ("Kimi K3 满血能力指纹", "随机挑战、工具、结构化输出、上下文、流式和参数行为检查", "按随机挑战执行", "中"),
}
JOB_QUEUE: queue.Queue[dict[str, Any]] = queue.Queue()
WORKER_STARTED = False
WORKER_LOCK = threading.Lock()
WORKER_THREAD: threading.Thread | None = None
ACTIVE_PROCESSES: dict[str, subprocess.Popen] = {}
ACTIVE_LOCK = threading.Lock()
CANCELLED_RUNS: set[str] = set()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def terminate_process(process: subprocess.Popen) -> None:
    """终止测试子进程：POSIX 杀进程组，Windows 用 taskkill /T 连带子进程。"""
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            process.terminate()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def apply_path_override(protocol: str, base_url: str, path_override: str | None) -> str:
    """把前端的自定义请求路径拼进已归一化的 host-only base_url。

    split_path_override 负责剥掉已知操作后缀只留根前缀；根前缀非空时 base_url
    变成带路径的地址，api_root 对这类地址原样返回，执行器追加的操作后缀
    因此落在自定义前缀之后。
    """
    root = split_path_override(path_override or "")
    return f"{base_url.rstrip('/')}{root}" if root else base_url


def db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def initialize() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with db() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, test_kind TEXT NOT NULL DEFAULT 'load', case_name TEXT NOT NULL, protocol TEXT NOT NULL DEFAULT 'volcengine', base_url TEXT NOT NULL, model TEXT NOT NULL,
                status TEXT NOT NULL, quick INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
                batch_id TEXT, is_batch INTEGER NOT NULL DEFAULT 0, case_ids TEXT, manifest_path TEXT,
                started_at TEXT, finished_at TEXT, exit_code INTEGER, result_dir TEXT,
                report_path TEXT, summary_path TEXT, passed INTEGER, error TEXT, log_path TEXT
            )
        """)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)").fetchall()}
        if "protocol" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN protocol TEXT NOT NULL DEFAULT 'volcengine'")
        if "test_kind" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN test_kind TEXT NOT NULL DEFAULT 'load'")
        for name, definition in (("batch_id", "TEXT"), ("is_batch", "INTEGER NOT NULL DEFAULT 0"),
                                 ("case_ids", "TEXT"), ("manifest_path", "TEXT"),
                                 # Versioned rating cache. Historical rows are recalculated read-only
                                 # from archived evidence by current_score; passed stays raw evidence.
                                 ("score", "REAL"), ("score_grade", "TEXT"), ("score_coverage", "REAL"),
                                 ("path_override", "TEXT NOT NULL DEFAULT ''"),
                                 # 执行面快照（model,verify / all）：记录页据此说明"这批跑了哪些面"，
                                 # 重跑也能复现同一范围。NULL = 旧记录，按默认集处理。
                                 ("faces", "TEXT")):
            if name not in columns:
                connection.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")
        # 进程意外退出后，不把旧 running 状态伪装成仍在执行。
        connection.execute("UPDATE runs SET status='interrupted', finished_at=?, error=COALESCE(error, 'Web 服务重启，任务状态已中断') WHERE status IN ('queued','running')", (now_iso(),))


def update(run_id: str, **values: Any) -> None:
    if not values:
        return
    assignments = ", ".join(f"{key}=?" for key in values)
    with db() as connection:
        connection.execute(f"UPDATE runs SET {assignments} WHERE id=?", (*values.values(), run_id))


def _mark_worker_failure(job: dict[str, Any], exc: BaseException) -> None:
    """Make an executor crash visible instead of leaving jobs queued forever."""
    detail = f"后台测试 worker 异常：{type(exc).__name__}: {exc}"
    run_ids = [job["id"]]
    if job.get("is_batch"):
        run_ids.extend(child["id"] for child in job.get("children", []))
    for run_id in run_ids:
        try:
            update(run_id, status="failed", finished_at=now_iso(), exit_code=-1, error=detail)
        except Exception:
            # A database failure must not be allowed to kill the worker too.
            pass


def latest_result(case_name: str, since: float, *, aliases: tuple[str, ...] = ()) -> Path | None:
    # K3 临时案例集脚本按套件统一输出 K3_TEMP_SUITE-*，即使只执行了其中
    # 一个合并案例（例如 CACHE-HIT-7）。收尾阶段仍按选中的案例名查找
    # 会找不到 summary/report，导致单案例记录只有日志而没有报告按钮。
    names = [case_name, *aliases]
    # 内容审核执行脚本会将 MOD_* 展示用例名映射为 moderation_test_project 下的实际目录名。
    moderation_names = {
        "MOD_GENERAL_COMMON": "general_common",
        "MOD_PARTNER_COMMON": "virtual_partner_common",
        "MOD_PARTNER_SMOKE": "sillytavern_smoke",
    }
    if case_name in moderation_names:
        names.append(f"MODERATION-{moderation_names[case_name]}")
    candidates = [p for name in names for p in RESULTS.glob(f"{name}-*") if p.is_dir() and p.stat().st_mtime >= since - 2]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def interrupted_request_summary(
    job: dict[str, Any], run_id: str, result: Path | None
) -> dict[str, Any]:
    """Summarize request rows already flushed before a child was interrupted."""
    rows: list[dict[str, Any]] = []
    requests_path = result / "requests.jsonl" if result else None
    if requests_path and requests_path.is_file():
        with requests_path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    requested = int(job.get("requests") or len(rows))
    completed = len(rows)
    successful = sum(bool(row.get("success")) for row in rows)
    image_rows = [
        image
        for row in rows
        for image in (row.get("images") or [])
        if isinstance(image, dict) and image.get("saved")
    ]
    # 视频生成套件的产物字段不同（row["video"] 单个字典），单独汇总，避免被写成"0 张图片"。
    video_rows = [
        row["video"]
        for row in rows
        if isinstance(row.get("video"), dict) and row["video"].get("saved")
    ]
    error_distribution = dict(
        Counter(
            str(row.get("error_type") or "unknown_error")
            for row in rows
            if not row.get("success")
        )
    )
    schema_versions = {
        "gemini-image": "mltest.gemini-image.v1",
        "gpt-image": "mltest.gpt-image.v1",
        "seedance": "mltest.seedance.v1",
    }
    return {
        "schema_version": schema_versions.get(job["test_kind"], "mltest.gpt-image.v1"),
        "record_id": run_id,
        "batch_id": job.get("batch_id"),
        "test_kind": job["test_kind"],
        "case": job["case"],
        "protocol": job["protocol"],
        "base_url": job["base_url"],
        "model": job["model"],
        "status": "interrupted",
        "requested": requested,
        # Only completed rows are provable after SIGTERM. Unfinished requests
        # may already have reached the gateway, so do not present this as an
        # exact upstream-received count.
        "sent": completed,
        "completed": completed,
        "successful": successful,
        "failed": completed - successful,
        "unfinished": max(0, requested - completed),
        "images_dir": "images" if image_rows else None,
        "images_saved": len(image_rows),
        "image_bytes_saved": sum(int(image.get("bytes") or 0) for image in image_rows),
        "videos_dir": "videos" if video_rows else None,
        "videos_saved": len(video_rows),
        "video_bytes_saved": sum(int(video.get("bytes") or 0) for video in video_rows),
        "video_links": [
            {"index": row.get("index"), "task_id": row.get("task_id"),
             "status": row.get("task_status"), "url": row.get("video_url"), "local_file": None}
            for row in rows
        ],
        "error_distribution": error_distribution,
        "admission_passed": False,
        "error": "用户主动停止测试",
        "results": [],
    }


def _release_seedance_upload(job: dict[str, Any]) -> None:
    """删除该次任务上传的临时素材（图片与参考视频）。

    执行器已把原图/原视频复制进结果目录（`inputs/`），临时文件没有再留的必要——
    否则 web/data/uploads 会随着每次视频测试无上限增长。
    """
    for key in ("seedance_image_path", "seedance_video_path"):
        path = job.get(key)
        if not path:
            continue
        try:
            (ROOT / path).unlink(missing_ok=True)
        except OSError:
            pass


def _move_log_into_result(run_id: str, log_path: Path, result_dir: Path | None) -> Path:
    """执行日志应随报告归档：结果目录确定后把日志移进去并同步 DB 的 log_path。

    结果目录由子进程结束后才可知（latest_result），因此日志先写在 results/ 顶层，
    收尾时迁移；迁移失败时保留原路径，不影响准入判定。
    """
    if result_dir is None:
        return log_path
    final_log = result_dir / log_path.name
    if final_log == log_path:
        return log_path
    try:
        if log_path.exists():
            log_path.replace(final_log)
        update(run_id, log_path=str(final_log.relative_to(ROOT)))
    except (OSError, ValueError):
        return log_path
    return final_log


def single_run_manifest(row: sqlite3.Row) -> dict[str, Any] | None:
    """给单轮记录合成批量版式 manifest，让冒烟/专项报告复用聚合报告的套件表格。

    批次落盘有 aggregation_manifest.json；单轮没有，这里用 summary.json 现场合成，
    指标列（请求/成功率/TTFT P90 等）复用 build_item_metrics 的同一套口径。
    """
    summary: dict[str, Any] = {}
    if row["summary_path"]:
        path = (ROOT / row["summary_path"]).resolve()
        if path.is_file():
            try:
                summary = normalize_summary(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                return None
    else:
        return None
    item = {
        "record_id": row["id"], "case": row["case_name"], "status": row["status"],
        "passed": None if row["passed"] is None else bool(row["passed"]),
        "result_dir": row["result_dir"], "report_path": row["report_path"],
        "summary_path": row["summary_path"], "summary": summary, "error": row["error"],
        "log_path": dict(row).get("log_path"),
        "metrics": build_item_metrics(summary, row["result_dir"]),
    }
    manifest = {"schema_version": SCHEMA_VERSION, "record_type": "batch", "record_id": row["id"],
            "test_kind": row["test_kind"], "protocol": row["protocol"], "base_url": row["base_url"],
            "model": row["model"], "results": [item]}
    ensure_image_fingerprints(manifest)
    ensure_report_details(manifest, ROOT, RESULTS)
    return manifest


def _execute_child(job: dict[str, Any]) -> None:
    run_id = job["id"]
    started_epoch = time.time()
    log_path = RESULTS / f"web-{run_id}.log"
    update(run_id, status="running", started_at=now_iso(), log_path=str(log_path.relative_to(ROOT)))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # 自定义请求路径：根前缀拼进 base_url 后地址带路径，api_root 会原样返回，
    # 执行器追加的操作后缀（/chat/completions 等）落在自定义前缀之后。
    override_root = split_path_override(job.get("path_override") or "")
    effective_base_url = f"{job['base_url'].rstrip('/')}{override_root}" if override_root else job["base_url"]
    resolved_base_url = api_root(job["protocol"], effective_base_url)
    env.update({"VOLC_RECORD_ID": run_id, "VOLC_BATCH_ID": job.get("batch_id") or "", "VOLC_API_KEY": job["api_key"],
                "VOLC_BASE_URL": resolved_base_url, "VOLC_MODEL": job["model"], "VOLC_PROTOCOL": job["protocol"]})
    # 执行面：默认 model,verify（通道契约面停用）。子进程按此过滤 CASE_SPECS，
    # 通道面即使勾选执行，其失败也只是 channel_gap，不阻断准入。
    if job.get("faces"):
        env["VOLC_FACES"] = job["faces"]
    # 「Kimi 测试」模块下有两套执行链，按 case 分流：K3 临时案例集走清单流水线
    # （环境变量驱动，不接受位置参数），其余 kvv 用例走官方 KVV pytest。
    # 单独跑某个合并案例时 job["case"] 是该合并案例名（如 CACHE-HIT-7），
    # 此时靠 temporary_case 认出来；整组跑时 case 就是入口 id。
    kvv_temp = job["test_kind"] == "kvv" and (job["case"] == KVV_TEMP_CASE_ID or bool(job.get("temporary_case")))
    if kvv_temp:
        env["VOLC_TEMP_CASE"] = job.get("temporary_case") or ""
    # 中转计费探针（现挂在「Claude 测试」模块下）：选中计费项时换执行器与协议面。
    # 探针脚本自行按 --protocol 拼 /v1 前缀，故这里给原始域名而不是 api_root 结果。
    billing_probes = billing_probe_selection(job.get("billing_probe"))
    if billing_probes:
        env.update({
            "BILLING_BASE_URL": job["base_url"],
            "BILLING_API_KEY": job["api_key"],
            "BILLING_MODEL": job["model"],
            "BILLING_PROTOCOL": "anthropic" if job["protocol"] == "anthropic" else "openai",
            # 空 = 全部探针；单条/子集时按 id 列出（执行器会自动补依赖前置）。
            "BILLING_PROBES": "" if job.get("billing_probe") == BILLING_PROBE_ALL else ",".join(billing_probes),
        })
    if job["test_kind"] == "deepseek":
        env.update({"DS_API_KEY": job["api_key"], "DS_BASE_URL": resolved_base_url, "DS_MODEL": job["model"]})
    if job["test_kind"] in {"glm", "glm_stream_retest"}:
        env.update({"GLM_API_KEY": job["api_key"], "GLM_BASE_URL": resolved_base_url, "GLM_MODEL": job["model"]})
    if job["test_kind"] == "qwen":
        env.update({"QWEN_API_KEY": job["api_key"], "QWEN_BASE_URL": resolved_base_url,
                    "QWEN_MODEL": job["model"]})
    if job["case"] in CONTRACT_CASES:
        family = CONTRACT_CASES[job["case"]]["family"]
        env.update({"CONTRACT_CASE": job["case"], "DS_API_KEY": job["api_key"], "DS_BASE_URL": resolved_base_url, "DS_MODEL": job["model"], "GLM_API_KEY": job["api_key"], "GLM_BASE_URL": resolved_base_url, "GLM_MODEL": job["model"], "VOLC_API_KEY": job["api_key"], "VOLC_BASE_URL": resolved_base_url, "VOLC_MODEL": job["model"]})
    if job["test_kind"] == "authenticity" or job["case"] in {"DS_V4_FINGERPRINT"}:
        env.update({"VOLC_API_KEY": job["api_key"], "VOLC_BASE_URL": resolved_base_url, "VOLC_MODEL": job["model"]})
    if job["test_kind"] == "gpt":
        env.update({"GPT_API_KEY": job["api_key"], "GPT_BASE_URL": resolved_base_url, "GPT_MODEL": job["model"], "GPT_PROTOCOL": job["protocol"]})
    if job["test_kind"] == "fable":
        env.update({"FABLE_API_KEY": job["api_key"], "FABLE_BASE_URL": resolved_base_url, "FABLE_MODEL": job["model"], "FABLE_PROTOCOL": job["protocol"]})
    if job["test_kind"] == "gpt-image":
        env.update({
            "GPT_IMAGE_API_KEY": job["api_key"],
            "GPT_IMAGE_BASE_URL": resolved_base_url,
            "GPT_IMAGE_PROTOCOL": job["protocol"],
            "GPT_IMAGE_QUALITIES": ",".join(job.get("image_quality") or []),
            "GPT_IMAGE_MODELS": ",".join(job.get("image_models") or []),
            "GPT_IMAGE_SIZES": ",".join(job.get("image_sizes") or []),
            "GPT_IMAGE_PROMPT": (job.get("image_prompt") or "") if job["case"] != GPT_IMAGE_TOKEN_CASE else "",
            "GPT_IMAGE_ASPECT_RATIO": job.get("image_aspect_ratio") or "",
        })
        if job.get("image_request_timeout_seconds") is not None:
            env["GPT_IMAGE_REQUEST_TIMEOUT_SECONDS"] = str(job["image_request_timeout_seconds"])
        # 压测持续时长：**默认不启用**。必须显式写空串——父进程若残留
        # GPT_IMAGE_HOLD_SECONDS，子进程会继承并把单批被动变成时长压测。
        env["GPT_IMAGE_HOLD_SECONDS"] = (
            "" if job.get("image_hold_seconds") is None else str(job["image_hold_seconds"])
        )
    if job["test_kind"] == "gemini-image":
        env.update({
            "GEMINI_IMAGE_API_KEY": job["api_key"],
            "GEMINI_IMAGE_BASE_URL": resolved_base_url,
            "GEMINI_IMAGE_QUALITIES": ",".join(job.get("gemini_quality") or []),
            "GEMINI_IMAGE_MODELS": ",".join(job.get("gemini_models") or []),
            "GEMINI_IMAGE_ASPECTS": ",".join(job.get("gemini_aspects") or []),
            "GEMINI_IMAGE_COUNTS": ",".join(str(item) for item in (job.get("gemini_counts") or [])),
            "GEMINI_IMAGE_PROMPT": job.get("image_prompt") or "",
        })
    if job["test_kind"] == "seedance":
        # 内容型参数走环境变量（提示词含换行/引号；图片给的是本机路径，由执行器读成
        # data URI 再发给上游）。requests/concurrency 走命令行，见下方参数装配。
        env.update({
            "SEEDANCE_API_KEY": job["api_key"],
            "SEEDANCE_BASE_URL": resolved_base_url,
            "SEEDANCE_MODEL": job["model"],
            "SEEDANCE_PROMPT": job.get("seedance_prompt") or "",
            "SEEDANCE_RESOLUTION": job.get("seedance_resolution") or "",
            "SEEDANCE_RATIO": job.get("seedance_ratio") or "",
            "SEEDANCE_DURATION": str(job.get("seedance_duration") or ""),
            "SEEDANCE_MODE": job.get("seedance_mode") or "",
            "SEEDANCE_IMAGE_ROLE": job.get("seedance_image_role") or "",
            "SEEDANCE_IMAGE_PATH": job.get("seedance_image_path") or "",
            "SEEDANCE_IMAGE_URL": job.get("seedance_image_url") or "",
            "SEEDANCE_VIDEO_ROLE": job.get("seedance_video_role") or "",
            "SEEDANCE_VIDEO_PATH": job.get("seedance_video_path") or "",
            "SEEDANCE_VIDEO_URL": job.get("seedance_video_url") or "",
            "SEEDANCE_GENERATE_AUDIO": "1" if job.get("seedance_audio") else "0",
            "SEEDANCE_WATERMARK": "1" if job.get("seedance_watermark") else "0",
        })
    if job["test_kind"] == "gemini":
        env.update({"GEMINI_API_KEY": job["api_key"], "GEMINI_BASE_URL": resolved_base_url,
                    "GEMINI_MODEL": job["model"], "GEMINI_PROTOCOL": job["protocol"]})
    if job["test_kind"] == "chat":
        # 对话窗口的内容型参数走环境变量：提示词含换行/引号/极长文本，命令行不适合承载。
        env.update({
            "CHAT_API_KEY": job["api_key"],
            "CHAT_BASE_URL": resolved_base_url,
            "CHAT_MODEL": job["model"],
            "CHAT_PROTOCOL": job["protocol"],
            "CHAT_PROMPT": job.get("chat_prompt") or "",
            "CHAT_SYSTEM": job.get("chat_system") or "",
            "CHAT_MAX_TOKENS": str(job.get("chat_max_tokens") or CHAT_DEFAULT_MAX_TOKENS),
        })
        if job.get("chat_temperature") is not None:
            env["CHAT_TEMPERATURE"] = str(job["chat_temperature"])
    if job["test_kind"] == "forensics":
        # 中转验真：配置全部走环境变量（与对话窗口同法）；探针内部自按协议拼操作后缀。
        env.update({"FORENSICS_API_KEY": job["api_key"], "FORENSICS_BASE_URL": resolved_base_url,
                    "FORENSICS_MODEL": job["model"], "FORENSICS_PROTOCOL": job["protocol"]})
    script = CHAT_SCRIPT if job["test_kind"] == "chat" else FORENSICS_SCRIPT if job["test_kind"] == "forensics" else BILLING_PROBE_SCRIPT if billing_probes else PROTO_SIGNATURE_SCRIPT if job["test_kind"] == "fable" and job["case"] == PROTO_SIGNATURE_CASE_ID else ZERO_INJECTION_SCRIPT if job["test_kind"] == "fable" and job["case"] == ZERO_INJECTION_CASE_ID else KVV_TEMP_SCRIPT if kvv_temp else GPT_IMAGE_TOKEN_SCRIPT if job["case"] == GPT_IMAGE_TOKEN_CASE else GPT_IMAGE_SCRIPT if job["test_kind"] == "gpt-image" else GEMINI_IMAGE_SCRIPT if job["test_kind"] == "gemini-image" else SEEDANCE_SCRIPT if job["test_kind"] == "seedance" else GEMINI3_SCRIPT if job["test_kind"] == "gemini" else LEGAL_AGENT_BENCHMARK_SCRIPT if job["test_kind"] == "legal-agent-benchmark" else SWE_ATLAS_RF_SCRIPT if job["test_kind"] == "swe-atlas-rf" else SWE_ATLAS_TW_SCRIPT if job["test_kind"] == "swe-atlas-tw" else DEEPSWE_SCRIPT if job["test_kind"] == "deepswe" else ARXIVMATH_SCRIPT if job["test_kind"] == "arxivmath" else MODERATION_SCRIPT if job["test_kind"] == "moderation" else AUTH_SCRIPT if job["case"] in {"DS_V4_FINGERPRINT"} else CONTRACT_SCRIPT if job["case"] in CONTRACT_CASES else KVV_SCRIPT if job["test_kind"] == "kvv" else DEEPSEEK_SCRIPT if job["test_kind"] == "deepseek" else QWEN_SCRIPT if job["test_kind"] == "qwen" else GLM_SCRIPT if job["test_kind"] == "glm" else GLM_STREAM_RETEST_SCRIPT if job["test_kind"] == "glm_stream_retest" else FABLE5_SCRIPT if job["test_kind"] == "fable" else GPT56_SCRIPT if job["test_kind"] == "gpt" else RUN_SCRIPT
    if job["case"] in ECONOMY_CASES:
        script = ECONOMY_SCRIPT
        command = [sys.executable, str(script), job["case"]]
    elif kvv_temp or billing_probes:
        # K3 临时案例集与计费探针都由环境变量驱动，不接受位置参数。
        command = [sys.executable, str(script)]
    elif job["test_kind"] == "fable" and job["case"] in {PROTO_SIGNATURE_CASE_ID, ZERO_INJECTION_CASE_ID}:
        # Protobuf 签名两步校验 / 0 注入判定：自包含 CLI，凭据与端点走命令行参数
        # （不给原始域名以外的前缀——脚本内部自调 api_root 拼 /v1），模型直传 --model。
        command = [sys.executable, str(script), "--model", job["model"],
                   "--base-url", job["base_url"], "--api-key", job["api_key"]]
        if job["case"] in {PROTO_SIGNATURE_CASE_ID, ZERO_INJECTION_CASE_ID}:
            # 两步签名校验与 0 注入判定都要取签名：thinking 档位按被测模型解析
            # （唯一权威 fable5/model_profile.resolve_thinking_mode：4.5/4.6 系 enabled、
            # 5 系 adaptive+output_config.effort、其余 adaptive），不按模型拆套件。
            # 模型名是中转别名（识别不出族）时用界面选中的变体兜底。
            command += ["--thinking-mode", claude_thinking_mode(job["model"], job.get("claude_variant"))]
        if job["case"] == PROTO_SIGNATURE_CASE_ID:
            # 输出预算给足：给少了思考会吃满上限、正文为零，被契约读成「文本内容为空」。
            budget = job.get("signature_max_tokens") or SIGNATURE_DEFAULT_MAX_TOKENS
            command += ["--max-tokens", str(budget)]
    elif job["test_kind"] in {"arxivmath", "deepswe", "swe-atlas-tw", "swe-atlas-rf", "legal-agent-benchmark"}:
        command = [sys.executable, str(script), "--case", job["case"]]
    else:
        # Claude 变体用例（OPUS5_* 等）派发前映射回 fable5 真实套件名。
        case_arg = claude_suite_name(job["case"]) if job["test_kind"] == "fable" else job["case"]
        command = [sys.executable, str(script), CONTRACT_CASES[case_arg]["family"] if case_arg in CONTRACT_CASES else case_arg]
    command += ["--output-root", str(RESULTS)]
    if job["quick"] and job["test_kind"] == "load": command.append("--quick")
    if job["test_kind"] == "moderation":
        if job.get("concurrency") is not None: command.extend(["--concurrency", str(job["concurrency"])])
    if job["test_kind"] == "load":
        if job.get("requests") is not None: command.extend(["--samples", str(job["requests"])])
        if job.get("concurrency") is not None: command.extend(["--concurrency", str(job["concurrency"])])
        if job.get("warmup_seconds") is not None: command.extend(["--warmup-seconds", str(job["warmup_seconds"])])
    if job["test_kind"] == "gpt-image" and job["case"] != GPT_IMAGE_TOKEN_CASE:
        command.extend(["--requests", str(job["requests"]), "--concurrency", str(job["concurrency"])])
    if job["test_kind"] == "gpt" and job["case"] in PELICAN_CASE_IDS:
        # 产物类用例的多轮参数（默认 1×1，单轮行为与引入前一致）。
        command.extend(pelican_round_args(job))
    if job["test_kind"] == "gemini-image":
        command.extend(["--requests", str(job["requests"]), "--concurrency", str(job["concurrency"])])
    if job["test_kind"] == "seedance":
        command.extend(["--requests", str(job["requests"]), "--concurrency", str(job["concurrency"])])
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as stream:
            stream.write(f"[{now_iso()}] 启动 {job['case']} 测试；记录 ID={run_id}; 批次 ID={job.get('batch_id') or '-'}\n")
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            with ACTIVE_LOCK:
                ACTIVE_PROCESSES[run_id] = process
            returncode = process.wait()
            with ACTIVE_LOCK:
                ACTIVE_PROCESSES.pop(run_id, None)
        _release_seedance_upload(job)
        result_aliases = (KVV_TEMP_CASE_ID,) if kvv_temp else ("RELAY-BILLING-PROBE",) if billing_probes else ((claude_suite_name(job["case"]),) if job["test_kind"] == "fable" else ())
        result = latest_result(job["case"], started_epoch, aliases=result_aliases)
        log_path = _move_log_into_result(run_id, log_path, result)
        summary_path = result / "summary.json" if result else None
        report_path = result / "report.md" if result else None
        summary = enrich_summary(summary_path, record_id=run_id, batch_id=job.get("batch_id"), test_kind=job["test_kind"],
                                 case_name=job["case"], protocol=job["protocol"], base_url=job["base_url"], model=job["model"]) if summary_path and summary_path.exists() else {}
        if summary and result:
            summary = finalize_result_dir(result)
        cancelled = run_id in CANCELLED_RUNS
        passed = bool(summary.get("admission_passed")); status = "interrupted" if cancelled else ("completed" if summary else "failed")
        score_report = summary.get("score") if isinstance(summary.get("score"), dict) else None
        failure_detail = None
        if not summary:
            try:
                failure_detail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            except OSError:
                failure_detail = f"测试未生成 summary.json，退出码 {returncode}"
        if cancelled and not summary:
            if job["test_kind"] in {"gpt-image", "gemini-image", "seedance"}:
                summary = interrupted_request_summary(job, run_id, result)
            else:
                summary = {
                    "schema_version": "volcperf.result.v1",
                    "record_id": run_id,
                    "batch_id": job.get("batch_id"),
                    "test_kind": job["test_kind"],
                    "case": job["case"],
                    "protocol": job["protocol"],
                    "base_url": job["base_url"],
                    "model": job["model"],
                    "status": "interrupted",
                    "admission_passed": False,
                    "error": "用户主动停止测试",
                    "results": [],
                }
            result_dir = result or RESULTS / f"{job['case']}-{time.strftime('%Y%m%d-%H%M%S')}"
            result_dir.mkdir(parents=True, exist_ok=True)
            summary_path = result_dir / "summary.json"; report_path = result_dir / "report.md"
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            request_detail = result_dir / "requests.jsonl" if (result_dir / "requests.jsonl").exists() else None
            detail_line = f"- 请求级明细：`{request_detail.relative_to(ROOT)}`\n" if request_detail else f"- 请求明细未找到；执行日志：`{log_path.relative_to(ROOT)}`\n"
            count_lines = ""
            if job["test_kind"] in {"gpt-image", "gemini-image", "seedance"}:
                count_lines = (
                    f"- 计划请求：{summary['requested']}\n"
                    f"- 已完成：{summary['completed']}\n"
                    f"- 成功：{summary['successful']}\n"
                    f"- 失败：{summary['failed']}\n"
                    f"- 未完成：{summary['unfinished']}\n"
                    f"- 错误分布：`{json.dumps(summary['error_distribution'], ensure_ascii=False)}`\n"
                )
                if summary.get("images_saved"):
                    count_lines += f"- 已保存图片：{summary['images_saved']} 张（`images/`）\n"
                if summary.get("videos_saved"):
                    count_lines += f"- 已保存视频：{summary['videos_saved']} 个（`videos/`）\n"
                links = [item.get("url") for item in (summary.get("video_links") or []) if item.get("url")]
                if links:
                    count_lines += "- 已生成的视频链接：\n" + "".join(f"  - {url}\n" for url in links)
            report_path.write_text(
                f"# 测试中断报告\n\n"
                f"- 用例：`{job['case']}`\n"
                f"- 记录 ID：`{run_id}`\n"
                f"- 模型：`{job['model']}`\n"
                f"- 状态：用户主动停止\n"
                f"- 子进程退出码：`{returncode}`\n"
                + count_lines
                + detail_line,
                encoding="utf-8",
            )
            result = result_dir
            log_path = _move_log_into_result(run_id, log_path, result)
        update(run_id, status=status, finished_at=now_iso(), exit_code=returncode,
               result_dir=str(result.relative_to(ROOT)) if result else None,
               report_path=str(report_path.relative_to(ROOT)) if report_path and report_path.exists() else None,
               summary_path=str(summary_path.relative_to(ROOT)) if summary_path and summary_path.exists() else None,
               score=(score_report or {}).get("score"), score_grade=(score_report or {}).get("grade"),
               score_coverage=(score_report or {}).get("coverage"),
               passed=int(passed) if summary else None, error=None if summary else failure_detail)
    except Exception as exc:
        update(run_id, status="failed", finished_at=now_iso(), exit_code=-1, error=str(exc))


def _finalize_batch(job: dict[str, Any]) -> None:
    batch_id=job["id"]
    with db() as connection:
        children=[dict(row) for row in connection.execute("SELECT * FROM runs WHERE batch_id=? ORDER BY created_at",(batch_id,)).fetchall()]
        batch=dict(connection.execute("SELECT * FROM runs WHERE id=?",(batch_id,)).fetchone())
    for child in children:
        if child.get("summary_path"): child["summary_path"] = str((ROOT / child["summary_path"]).resolve())
    out=RESULTS/f"BATCH-{batch_id}"
    aggregate=aggregate_batch(out,batch,children)
    child_statuses = {child.get("status") for child in children}
    batch_status = "interrupted" if "interrupted" in child_statuses else ("failed" if "failed" in child_statuses else "completed")
    batch_score = aggregate.get("score") if isinstance(aggregate.get("score"), dict) else {}
    update(batch_id,status=batch_status,finished_at=now_iso(),exit_code=0 if aggregate["admission_passed"] else 2,
           result_dir=str(out.relative_to(ROOT)),report_path=str((out/"report.md").relative_to(ROOT)),
           summary_path=str((out/"summary.json").relative_to(ROOT)),manifest_path=str((out/"aggregation_manifest.json").relative_to(ROOT)),
           score=batch_score.get("score"), score_grade=batch_score.get("grade"),
           score_coverage=batch_score.get("coverage"),
           passed=int(aggregate["admission_passed"]))


def worker() -> None:
    while True:
        job = JOB_QUEUE.get()
        try:
            try:
                if job.get("is_batch"):
                    update(job["id"],status="running",started_at=now_iso())
                    for child in job["children"]: _execute_child(child)
                    _finalize_batch(job)
                else:
                    _execute_child(job)
            except BaseException as exc:
                _mark_worker_failure(job, exc)
        finally:
            job["api_key"] = ""
            for child in job.get("children",[]): child["api_key"]=""
            JOB_QUEUE.task_done()


def start_worker() -> None:
    global WORKER_STARTED, WORKER_THREAD
    with WORKER_LOCK:
        if WORKER_THREAD is not None and WORKER_THREAD.is_alive():
            WORKER_STARTED = True
            return
        WORKER_THREAD = threading.Thread(target=worker, name="test-worker", daemon=True)
        WORKER_THREAD.start()
        WORKER_STARTED = True


# 结果目录内可直接在浏览器查看的产物（记录页给「查看」入口，报告「产物」小节给同一链接）。
# 与 gpt56/core.py:PELICAN_ARTIFACT_TITLES 同源：套件负责落盘，Web 只按这份白名单暴露——
# 白名单同时是取值域，杜绝把结果目录变成任意文件下载通道。
VIEWABLE_ARTIFACTS: dict[str, str] = dict(PELICAN_ARTIFACT_TITLES)
# 多轮批次里各轮的原始 SVG（`pelican-<轮次>.svg`）不在固定名单里：它的名字随轮数变化，
# 但**取值域仍必须受限**，所以只放行 gpt56 声明的这一条文件名正则（不是"目录里任意文件"）。
# Seedance 生成的视频同理：`videos/video-<序号>.mp4` 由执行器声明正则，Web 只按它放行。
VIEWABLE_ARTIFACT_PATTERNS = (PELICAN_ROUND_SVG_PATTERN, seedance_cases.SEEDANCE_VIDEO_PATTERN)
# 受控正则类产物的枚举上限（记录列表每 3 秒轮询一次，枚举必须封顶）。
VIEWABLE_ARTIFACT_GLOB_LIMIT = 24
VIEWABLE_ARTIFACT_GLOBS: tuple[tuple[str, str], ...] = (("videos/video-*.mp4", "生成的视频"),)
ARTIFACT_CONTENT_TYPES = {
    "html": "text/html; charset=utf-8",
    "svg": "image/svg+xml",
    "md": "text/plain; charset=utf-8",
    "mp4": "video/mp4",
    "webm": "video/webm",
    "mov": "video/quicktime",
    "mkv": "video/x-matroska",
    "m4v": "video/x-m4v",
}

# 只读取图片套件保存到 images/ 的位图；不开放 SVG/HTML 等可执行内容。
GENERATED_IMAGE_CONTENT_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
    ".tif": "image/tiff", ".tiff": "image/tiff", ".avif": "image/avif",
}


def generated_image_directory(result_dir: str | None) -> Path | None:
    if not result_dir:
        return None
    directory = (ROOT / result_dir).resolve()
    if RESULTS.resolve() not in directory.parents:
        return None
    images = directory / "images"
    if images.is_symlink() or not images.is_dir():
        return None
    return images


def generated_image_paths(result_dir: str | None) -> Iterator[Path]:
    """惰性探测落盘图片，记录列表找到第一张即停止；仅图库页完整枚举。"""
    directory = generated_image_directory(result_dir)
    if directory is None:
        return
    for path in directory.iterdir():
        if path.suffix.lower() in GENERATED_IMAGE_CONTENT_TYPES and not path.is_symlink() and path.is_file():
            yield path


def generated_image_details(run_id: str, result_dir: str | None) -> list[dict[str, Any]]:
    """按原图一一读取签名，不使用带截断的历史汇总分组推断单张签名。"""
    return [{"name": path.name,
             "url": f"/api/runs/{quote(run_id, safe='')}/images/{quote(path.name, safe='')}",
             "fields": image_fingerprint_fields(path)}
            for path in sorted(generated_image_paths(result_dir), key=lambda path: path.name)]


def ensure_image_fingerprints(payload: dict[str, Any]) -> None:
    """为单项/批量报告添加逐图信息；包括只有历史图片文件的记录。"""
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        summary = item.get("summary") or {}
        kind = summary.get("test_kind") or payload.get("test_kind")
        case = str(item.get("case") or summary.get("case") or "")
        if kind in {"gpt-image", "gemini-image"} or case.startswith(("GPT_IMAGE", "GEMINI_IMAGE")):
            item["image_fingerprints"] = generated_image_details(str(item.get("record_id") or ""), item.get("result_dir"))


def render_image_gallery(row: sqlite3.Row) -> str:
    images = generated_image_details(row["id"], row["result_dir"])
    title = f"{row['case_name']} · {row['model']}"
    figures = []
    for index, image in enumerate(images, 1):
        url = image["url"]
        label = escape(image["name"])
        fingerprint = ''.join(f'<div><dt>{escape(field["label"])}</dt><dd>{escape(field["value"])}</dd></div>'
                              for field in image["fields"])
        loading = "eager" if index == 1 else "lazy"
        figures.append(
            f'<figure class="image-card" id="image-{index}">'
            f'<figcaption><div><span class="image-number">{index} / {len(images)}</span>'
            f'<span class="image-name">{label}</span></div>'
            f'<a class="image-original" href="{url}" target="_blank" rel="noopener noreferrer">查看原图</a>'
            f'</figcaption><div class="image-preview">'
            f'<img src="{url}" alt="第 {index} 张图片：{label}" loading="{loading}" decoding="async"'
            ' onerror="this.hidden=true;this.nextElementSibling.hidden=false">'
            '<p class="image-error" hidden>图片加载失败或浏览器不支持此格式，请点击“查看原图”。</p>'
            f'</div><section class="image-fingerprint"><h2>图片指纹</h2><dl>{fingerprint}</dl></section></figure>'
        )
    content = "\n".join(figures) or '<p class="gallery-empty">暂无已保存的图片</p>'
    return (STATIC / "image-gallery.html").read_text(encoding="utf-8").format(
        title=escape(title), record_id=escape(row["id"]), created_at=escape(row["created_at"]),
        count=len(images), content=content,
    )


def is_viewable_artifact(name: str) -> bool:
    """产物名是否在白名单内（固定名单 ∪ 受控文件名正则）。"""
    return name in VIEWABLE_ARTIFACTS or any(pattern.fullmatch(name) for pattern in VIEWABLE_ARTIFACT_PATTERNS)


def pelican_round_args(job: dict[str, Any]) -> list[str]:
    """产物类用例（鹈鹕骑车 SVG）的 CLI 参数：--requests/--concurrency。

    只有这几个用例的请求数可变；其余用例（含其它 GPT 用例）一律不追加参数——
    run_gpt56.py 收到这两个参数时也会二次拦回，避免"以为调了其实没生效"。
    """
    if job.get("test_kind") != "gpt" or job.get("case") not in PELICAN_CASE_IDS:
        return []
    return ["--requests", str(job.get("requests") or 1), "--concurrency", str(job.get("concurrency") or 1)]


def viewable_artifacts(run_id: str, result_dir: str | None) -> list[dict[str, str]]:
    """本记录结果目录里存在、且在白名单内的可直接查看产物。

    只做 stat 探测（记录列表每 3 秒轮询一次，不读文件内容）；目录必须落在 RESULTS 下，
    与 artifact 路由的取值域保持一致，避免暴露结果目录之外的路径。
    """
    if not result_dir:
        return []
    directory = (ROOT / result_dir).resolve()
    if RESULTS.resolve() not in directory.parents:
        return []
    found = [
        {"name": name, "title": title, "url": f"/api/runs/{run_id}/artifact/{name}"}
        for name, title in VIEWABLE_ARTIFACTS.items()
        if (directory / name).is_file()
    ]
    # 名字随序号变化的产物（视频）按受控 glob 有限枚举，再逐条过白名单校验。
    for pattern, title in VIEWABLE_ARTIFACT_GLOBS:
        for path in sorted(directory.glob(pattern))[:VIEWABLE_ARTIFACT_GLOB_LIMIT]:
            name = str(path.relative_to(directory))
            if not is_viewable_artifact(name) or path.suffix.lower().lstrip(".") not in ARTIFACT_CONTENT_TYPES:
                continue
            found.append({
                "name": name,
                "title": f"{title} {path.stem}",
                "url": f"/api/runs/{run_id}/artifact/{name}",
            })
    return found


def public_run(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["quick"] = bool(data["quick"])
    data["is_batch"] = bool(data.get("is_batch"))
    # 执行面快照：旧记录（NULL）按默认集显示，前端据此标注"本批只跑了模型面"。
    data["faces"] = data.get("faces") or ",".join(DEFAULT_FACES)
    data["case_ids"] = json.loads(data["case_ids"]) if data.get("case_ids") else [data["case_name"]]
    if data["case_name"] in ECONOMY_CASES:
        data["case_title"] = economy_case_payload(data["case_name"])["title"]
    elif data.get("test_kind") == "kvv":
        # 只有 K3 合并案例有子项标题；官方 KVV 套件返回 None。
        data["case_title"] = kvv_temp_case_title(data["case_name"])
    elif data.get("test_kind") == "fable":
        # 计费探针有子项标题；Protobuf 校验与普通 Claude 套件返回 None（直接用案例名）。
        data["case_title"] = billing_probe_title(data["case_name"])
    else:
        data["case_title"] = None
    data["passed"] = None if data["passed"] is None else bool(data["passed"])
    data["report_url"] = f"/api/runs/{data['id']}/report" if data.get("report_path") else None
    data["summary_url"] = f"/api/runs/{data['id']}/summary" if data.get("summary_path") else None
    data["log_url"] = f"/api/runs/{data['id']}/log" if data.get("log_path") else None
    data["manifest_url"] = f"/api/runs/{data['id']}/manifest" if data.get("manifest_path") else None
    # Raw passed remains evidence; rating_passed is the score-based acceptance result.
    report = current_score(data, ROOT, RESULTS)
    primary = report if report and not report.get("scores") else {}
    for key, report_key in (("score", "score"), ("score_grade", "grade"), ("score_coverage", "coverage"),
                            ("model_score", "model_score"), ("channel_score", "channel_score"),
                            ("rating_passed", "rating_passed"), ("scoring_version", "scoring_version")):
        data[key] = primary.get(report_key)
    score_reports = report.get("scores", []) if report and report.get("scores") else [primary]
    data["has_failure_items"] = any(
        any((item.get("counts") or {}).get(status, 0) for status in ("failed", "error"))
        for item in score_reports) or data["passed"] is False
    data["score_max"] = 120
    data["score_partial"] = primary.get("partial")
    data["score_valid"] = primary.get("valid")
    data["score_components"] = primary.get("components", {})
    data["score_models"] = len(report.get("scores", [])) if report else 0
    data["score_url"] = f"/api/runs/{data['id']}/score" if data.get("result_dir") else None
    data["download_url"] = f"/api/runs/{data['id']}/download" if data.get("result_dir") else None
    # 请求级审计文件不写入数据库路径，按结果目录动态暴露。
    if data.get("result_dir") and (ROOT / data["result_dir"] / "requests.jsonl").is_file():
        data["requests_url"] = f"/api/runs/{data['id']}/requests"
    else:
        data["requests_url"] = None
    # 「对话窗口」的模型原文（response.md）：只有该套件会生成，按结果目录动态暴露，
    # 让对话面板与记录页都能直接取到未经包装的完整响应。
    if data.get("result_dir") and (ROOT / data["result_dir"] / "response.md").is_file():
        data["response_url"] = f"/api/runs/{data['id']}/response"
    else:
        data["response_url"] = None
    # 套件落盘的可直接查看产物（如鹈鹕骑车用例的 pelican.html / pelican.svg）。
    data["artifacts"] = viewable_artifacts(data["id"], data.get("result_dir"))
    data["images_url"] = (
        f"/api/runs/{data['id']}/images"
        if next(generated_image_paths(data.get("result_dir")), None) is not None else None
    )
    return data


class Handler(BaseHTTPRequestHandler):
    server_version = "VolcPerfWeb/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def json_response(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def text_response(self, text: str, content_type: str = "text/plain; charset=utf-8", status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 64 * 1024:
            raise ValueError("请求体过大")
        return json.loads(self.rfile.read(length) or b"{}")

    # 允许上传的图片类型（与 seedance 执行器的 `_guess_image_mime` 保持同一集合）。
    _UPLOAD_IMAGE_MIME_EXTENSIONS = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/bmp": "bmp",
    }
    # 允许上传的**输入**参考视频类型：与 seedance.cases 的取值域是同一份定义。
    _UPLOAD_VIDEO_MIME_EXTENSIONS = dict(seedance_cases.SEEDANCE_INPUT_VIDEO_MIME_EXTENSIONS)
    _UPLOAD_MIME_EXTENSIONS = {**_UPLOAD_IMAGE_MIME_EXTENSIONS, **_UPLOAD_VIDEO_MIME_EXTENSIONS}

    def handle_upload(self) -> None:
        """接收 Seedance 的首帧/参考图与参考视频（原始二进制 body）。

        JSON 通道有 64 KiB 上限，而图片 base64 必然超过；因此单开一条原始 body 通道，
        只接受受控 MIME + 受控大小，落盘名用随机串（不采信客户端文件名），
        返回相对 ROOT 的路径交给子进程读取——素材内容既不进数据库也不进环境变量。

        视频走同一条通道（上限放宽到 64 MiB）：它**不随任务提交给上游**（异步任务面
        只接受 URL），落盘只为归档留痕，由执行器复制进结果目录 `inputs/`。
        """
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        extension = self._UPLOAD_MIME_EXTENSIONS.get(content_type)
        if not extension:
            if content_type.startswith("video/"):
                allowed = "/".join(sorted(self._UPLOAD_VIDEO_MIME_EXTENSIONS))
            elif content_type.startswith("image/"):
                allowed = "/".join(sorted(self._UPLOAD_IMAGE_MIME_EXTENSIONS))
            else:
                allowed = "/".join(sorted(self._UPLOAD_MIME_EXTENSIONS))
            raise ValueError(f"只支持上传 {allowed}：{content_type or '未知类型'}")
        is_video = content_type in self._UPLOAD_VIDEO_MIME_EXTENSIONS
        limit = (
            seedance_cases.SEEDANCE_MAX_INPUT_VIDEO_BYTES if is_video
            else seedance_cases.SEEDANCE_MAX_IMAGE_BYTES
        )
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("上传内容为空")
        if length > limit:
            raise ValueError(f"上传文件过大（{length} 字节，上限 {limit}）")
        raw = self.rfile.read(length)
        SEEDANCE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        swept = sweep_stale_uploads()
        name = f"{uuid.uuid4().hex}.{extension}"
        path = SEEDANCE_UPLOAD_DIR / name
        path.write_bytes(raw)
        source_name = str(self.headers.get("X-File-Name") or name)[:120]
        return self.json_response({
            "path": str(path.relative_to(ROOT)),
            "name": source_name,
            "bytes": len(raw),
            "mime": content_type,
            "kind": "video" if is_video else "image",
            "swept": swept,
        })

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/protocols":
            return self.json_response([{
                "id": item.id, "provider": item.provider, "name": item.name,
                "default_base_url": item.default_base_url, "path_template": item.path_template,
                "description": item.description, "model_hint": item.model_hint,
                "family": item.family, "test_kinds": list(item.test_kinds),
            } for item in PROTOCOLS.values()])
        if path == "/api/cases":
            load_cases = [load_case_payload(name) for name in CASES]
            kvv_cases = [capability_case_payload(name, "kvv", KVV_CASES[name], KVV_META[name]) for name in KVV_CASES]
            deepseek_cases = [auth_case_payload(name) if name == "DS_V4_FINGERPRINT" else capability_case_payload(name, "deepseek", DEEPSEEK_CASES[name], DEEPSEEK_META[name]) for name in DEEPSEEK_CASES]
            glm_cases = [capability_case_payload(name, "glm", GLM_CASES[name], GLM_META[name]) for name in GLM_CASES]
            for item in glm_cases:
                item["model_profiles"] = [{"id": p.model_id, "tier": p.tier,
                                           "context_probe_chars": p.context_probe_chars} for p in GLM_MODELS]
            # 「千问测试」：单一档位 + 六款具体型号，用例列表不随型号变（档位相关判据由后端解析）。
            qwen_cases = []
            for name, case in QWEN_CASES.items():
                item = capability_case_payload(name, "qwen", case, QWEN_META[name])
                item["qwen_variant"] = "qwen"
                item["default_model"] = QWEN_DEFAULT_MODEL
                item["qwen_models"] = QWEN_MODEL_OPTIONS
                qwen_cases.append(item)
            gpt56_cases = []
            for name, case in GPT56_CASES.items():
                item = capability_case_payload(name, "gpt", case, GPT56_META[name])
                variant = getattr(case, "model_variant", "gpt5.6")
                item["gpt_variant"] = variant
                item["default_model"] = GPT_VARIANT_DEFAULT_MODELS.get(variant, "gpt-5.6")
                # 该档位可选的具体型号（含可用性标记），前端据此渲染模型下拉并置灰无通道型号。
                item["gpt_models"] = GPT_VARIANT_MODEL_OPTIONS.get(variant, [])
                if name in PELICAN_CASE_IDS:
                    # 产物类用例支持一次跑多轮：前端据此显示「请求轮数 / 并发数」输入框，
                    # 并把上限交给界面做输入校验（后端仍会二次校验，不信任前端）。
                    item["multi_round"] = {
                        "requests": 1,
                        "concurrency": 1,
                        "max_requests": PELICAN_MAX_REQUESTS,
                        "max_concurrency": PELICAN_MAX_CONCURRENCY,
                        "requests_label": "请求轮数",
                        "hint": (f"同一句提示词重复执行（1～{PELICAN_MAX_REQUESTS} 轮，并发 1～"
                                 f"{PELICAN_MAX_CONCURRENCY}）：逐轮产物落盘到同一结果目录，"
                                 "pelican.html 为各轮渲染视图的总览页"),
                    }
                    item["request_text"] = f"1（默认；可调至 {PELICAN_MAX_REQUESTS} 轮）"
                gpt56_cases.append(item)
            fable5_cases = []
            for variant_id, (variant_label, variant_model) in CLAUDE_VARIANTS.items():
                for name, case in FABLE5_CASES.items():
                    item = capability_case_payload(name, "fable", case, FABLE5_META[name])
                    if variant_id != "fable5":
                        item["id"] = f"{variant_id.upper()}_{name.split('_', 1)[1]}"
                        item["title"] = case.title.replace("Fable 5", variant_label)
                    item["claude_variant"] = variant_id
                    item["default_model"] = variant_model
                    fable5_cases.append(item)
            # 计费探针、签名和 0 注入校验常驻 Claude 模块（不随模型变体切换，故不带 claude_variant）。
            fable5_cases.append(billing_probe_payload())
            fable5_cases.append(proto_signature_payload())
            fable5_cases.append(zero_injection_payload())
            contract_cases = [contract_case_payload(name) for name in CONTRACT_CASES]
            kvv_cases += [x for x in contract_cases if x["test_kind"] == "kvv"]
            # K3 供应商临时案例集与官方 KVV pytest 套件同属「Kimi 测试」模块。
            kvv_cases.append(kvv_temp_case_payload())
            deepseek_cases += [x for x in contract_cases if x["test_kind"] == "deepseek"]
            glm_cases += [x for x in contract_cases if x["test_kind"] == "glm"]
            economy_lists = {"kvv": kvv_cases, "deepseek": deepseek_cases, "glm": glm_cases,
                             "qwen": qwen_cases, "fable": fable5_cases, "gpt": gpt56_cases}
            for case_id, kind in ECONOMY_CASES.items():
                economy_lists[kind].append(economy_case_payload(case_id))
            moderation_cases = [moderation_case_payload(name) for name in MODERATION_CASES]
            chat_cases = [chat_case_payload(name) for name in CHAT_CASES]
            # 2 / 2.5 共用一个入口；旧 2.5 case 仅保留 API / CLI 兼容。
            gpt_image_cases = [gpt_image_case_payload(name) for name in GPT_IMAGE_CASES if name != "GPT_IMAGE_25"]
            gemini_image_cases = [gemini_image_case_payload(name) for name in GEMINI_IMAGE_CASES]
            seedance_cases_payload = [seedance_case_payload(name) for name in SEEDANCE_CASES]
            forensics_cases = [forensics_case_payload(name) for name in FORENSICS_CASES]
            gemini3_cases = []
            for name, case in GEMINI3_CASES.items():
                item = capability_case_payload(name, "gemini", case, GEMINI3_META[name])
                # 单档位（gemini3）+ 两个具体模型：前端按模型下拉切换 MODEL 默认值，
                # 用例列表不随模型变（档位相关期望值由后端矩阵按被测模型解析）。
                item["gemini_variant"] = getattr(case, "model_variant", "gemini3")
                item["default_model"] = GEMINI3_VARIANT_DEFAULT_MODEL
                item["gemini_models"] = GEMINI3_MODEL_OPTIONS
                gemini3_cases.append(item)
            arxivmath_case = {"id": "ARXIVMATH", "test_kind": "arxivmath", "source": "MathArena 官方开源数据集", "executor": "scripts/run_arxivmath.py（Hugging Face）", "title": ARXIVMATH_CASE.title, "description": ARXIVMATH_CASE.description, "duration": "按题目数量执行", "cost": ARXIVMATH_CASE.cost, "protocols": list(ARXIVMATH_CASE.protocols), "concurrency": 1, "concurrency_text": "1（按题目串行）", "requests": None, "request_text": "数据集全量题目", "case_count": None, "case_count_text": "全量题目（运行时确定）", "input_tokens": "研究级数学问题", "warmup": None, "steady": None, "admission": "使用官方 answer 字段评分；题目和答案不在本项目中改写。"}
            deepswe_case = {"id": "DEEPSWE", "test_kind": "deepswe", "source": "DataCurve 官方开源数据集", "executor": "scripts/run_deepswe.py（Pier）", "title": DEEPSWE_CASE.title, "description": DEEPSWE_CASE.description, "duration": "单任务约数分钟至数十分钟", "cost": DEEPSWE_CASE.cost, "protocols": list(DEEPSWE_CASE.protocols), "concurrency": 1, "concurrency_text": "由 Pier 管理隔离任务", "requests": None, "request_text": "官方任务集（默认全量）", "case_count": 113, "case_count_text": "113 个官方任务", "input_tokens": "代码仓库 + Agent 轨迹", "warmup": None, "steady": None, "admission": "官方 verifier 通过；使用官方任务、环境和测试，不在本项目重写。"}
            sweatlas_case = {"id": "SWE_ATLAS_TW", "test_kind": "swe-atlas-tw", "source": "Scale AI 官方开源数据集", "executor": "scripts/run_sweatlas_tw.py（Harbor）", "title": SWE_ATLAS_TW_CASE.title, "description": SWE_ATLAS_TW_CASE.description, "duration": "单任务约数分钟至数十分钟", "cost": SWE_ATLAS_TW_CASE.cost, "protocols": list(SWE_ATLAS_TW_CASE.protocols), "concurrency": 1, "concurrency_text": "由 Harbor 管理隔离任务", "requests": None, "request_text": "官方 TW 任务集", "case_count": 66, "case_count_text": "66 个官方任务（默认）", "input_tokens": "代码仓库 + Agent 轨迹", "warmup": None, "steady": None, "admission": "官方 verifier 通过；使用官方任务、环境和测试，不在本项目重写。"}
            sweatlas_rf_case = {"id": "SWE_ATLAS_RF", "test_kind": "swe-atlas-rf", "source": "Scale AI 官方开源数据集", "executor": "scripts/run_sweatlas_rf.py（Harbor）", "title": SWE_ATLAS_RF_CASE.title, "description": SWE_ATLAS_RF_CASE.description, "duration": "单任务约数分钟至数十分钟", "cost": SWE_ATLAS_RF_CASE.cost, "protocols": list(SWE_ATLAS_RF_CASE.protocols), "concurrency": 1, "concurrency_text": "由 Harbor 管理隔离任务", "requests": None, "request_text": "官方 RF 任务集", "case_count": 66, "case_count_text": "66 个官方任务（默认）", "input_tokens": "代码仓库 + Agent 轨迹", "warmup": None, "steady": None, "admission": "官方 verifier 通过；使用官方任务、环境和测试，不在本项目重写。"}
            legal_case = {"id": "LEGAL_AGENT_BENCHMARK", "test_kind": "legal-agent-benchmark", "source": "Harvey Labs v1.0 官方开源数据集", "executor": "scripts/run_legal_agent_benchmark.py（官方 harness + evaluator）", "title": LEGAL_AGENT_BENCHMARK_CASE.title, "description": LEGAL_AGENT_BENCHMARK_CASE.description, "duration": "单任务约数分钟至数十分钟", "cost": LEGAL_AGENT_BENCHMARK_CASE.cost, "protocols": list(LEGAL_AGENT_BENCHMARK_CASE.protocols), "concurrency": 1, "concurrency_text": "由官方 harness 管理 Agent 工具循环", "requests": None, "request_text": "官方 v1.0 Legal Agent 任务集", "case_count": 1, "case_count_text": "1 个任务（默认抽样）", "input_tokens": "法律材料 + Agent 轨迹", "warmup": None, "steady": None, "admission": "官方 evaluator 完成评分；使用官方任务、材料、rubric 和 harness，不在本项目重写。"}
            return self.json_response(load_cases + kvv_cases + deepseek_cases + glm_cases + qwen_cases + gpt56_cases + fable5_cases + gemini3_cases + gpt_image_cases + gemini_image_cases + seedance_cases_payload + moderation_cases + chat_cases + forensics_cases + [arxivmath_case, deepswe_case, sweatlas_case, sweatlas_rf_case, legal_case])
        if path == "/api/runs":
            query = parse_qs(urlparse(self.path).query)
            try:
                page = max(1, int(query.get("page", ["1"])[0]))
                page_size = min(100, max(1, int(query.get("page_size", ["20"])[0])))
            except (TypeError, ValueError):
                return self.json_response({"error": "分页参数无效"}, 400)
            status_filter = query.get("status", ["all"])[0]
            conclusion_filter = query.get("conclusion", ["all"])[0]
            search = query.get("q", [""])[0].strip()[:120]
            where = ["batch_id IS NULL"]
            params: list[Any] = []
            if status_filter == "running":
                where.append("status IN ('queued','running')")
            elif status_filter == "failed":
                where.append("status IN ('failed','interrupted')")
            elif status_filter in {"queued", "completed", "interrupted"}:
                where.append("status=?")
                params.append(status_filter)
            if conclusion_filter == "pass":
                where.append("passed=1")
            elif conclusion_filter == "fail":
                where.append("passed=0")
            elif conclusion_filter == "pending":
                where.append("passed IS NULL")
            if search:
                pattern = f"%{search}%"
                where.append("(id LIKE ? OR case_name LIKE ? OR case_ids LIKE ? OR test_kind LIKE ? OR model LIKE ? OR base_url LIKE ? OR protocol LIKE ? OR created_at LIKE ?)")
                params.extend([pattern] * 8)
            clause = " AND ".join(where)
            if conclusion_filter in {"score_pass", "score_fail", "score_pending"}:
                with db() as connection:
                    candidates = connection.execute(
                        f"SELECT * FROM runs WHERE {clause} ORDER BY created_at DESC", params).fetchall()
                expected = {"score_pass": True, "score_fail": False, "score_pending": None}[conclusion_filter]
                items = [item for row in candidates if (item := public_run(row)).get("rating_passed") is expected]
                offset = (page - 1) * page_size
                return self.json_response({"items": items[offset:offset+page_size], "total": len(items),
                                           "page": page, "page_size": page_size})
            with db() as connection:
                total = connection.execute(f"SELECT COUNT(*) FROM runs WHERE {clause}", params).fetchone()[0]
                rows = connection.execute(
                    f"SELECT * FROM runs WHERE {clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    [*params, page_size, (page - 1) * page_size],
                ).fetchall()
            return self.json_response({"items": [public_run(row) for row in rows], "total": total,
                                       "page": page, "page_size": page_size})
        parts = path.strip("/").split("/")
        if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
            run_id = parts[2]
            with db() as connection:
                row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if not row:
                return self.json_response({"error": "记录不存在"}, 404)
            if len(parts) == 3:
                data=public_run(row)
                if data["is_batch"]:
                    with db() as connection:
                        children=connection.execute("SELECT * FROM runs WHERE batch_id=? ORDER BY created_at",(run_id,)).fetchall()
                    data["children"]=[public_run(child) for child in children]
                return self.json_response(data)
            kind = parts[3]
            if kind == "images":
                if len(parts) == 4:
                    return self.text_response(render_image_gallery(row), "text/html; charset=utf-8")
                directory = generated_image_directory(row["result_dir"])
                if len(parts) != 5 or directory is None:
                    return self.json_response({"error": "图片不存在"}, 404)
                target = directory / parts[4]
                content_type = GENERATED_IMAGE_CONTENT_TYPES.get(target.suffix.lower())
                if (not content_type or target.is_symlink() or target.resolve().parent != directory
                        or not target.is_file()):
                    return self.json_response({"error": "图片不存在"}, 404)
                return self.serve_file(target, content_type)
            if kind == "download":
                with db() as connection:
                    rows_to_pack = [row]
                    if row["is_batch"]:
                        rows_to_pack += connection.execute("SELECT * FROM runs WHERE batch_id=? ORDER BY created_at", (run_id,)).fetchall()
                return self.send_result_archive(
                    {
                        "run_id": run_id,
                        "is_batch": bool(row["is_batch"]),
                        "test_kind": row["test_kind"],
                        "model": row["model"],
                        "created_at": row["created_at"],
                        "included_records": [r["id"] for r in rows_to_pack],
                    },
                    rows_to_pack,
                    f"test-data-{run_id}.zip",
                )
            if kind == "requests":
                if not row["result_dir"]:
                    return self.json_response({"error": "文件尚未生成"}, 404)
                target = (ROOT / row["result_dir"] / "requests.jsonl").resolve()
                if RESULTS.resolve() not in target.parents or not target.is_file():
                    return self.json_response({"error": "文件不存在"}, 404)
                return self.text_response(target.read_text(encoding="utf-8", errors="replace"), "application/x-ndjson; charset=utf-8")
            if kind == "response":
                # 「对话窗口」的模型原文：纯响应文本，不含报告包装，供面板回显与复制。
                if not row["result_dir"]:
                    return self.json_response({"error": "文件尚未生成"}, 404)
                target = (ROOT / row["result_dir"] / "response.md").resolve()
                if RESULTS.resolve() not in target.parents or not target.is_file():
                    return self.json_response({"error": "文件不存在"}, 404)
                return self.text_response(target.read_text(encoding="utf-8", errors="replace"), "text/markdown; charset=utf-8")
            if kind == "artifact":
                # 套件在结果目录里落盘的可直接查看产物（如鹈鹕骑车用例的 pelican.html / pelican.svg、
                # Seedance 生成的 videos/video-0001.mp4）。文件名必须命中白名单（固定名单 ∪ 受控正则），
                # 且路径落在该记录的结果目录内——白名单同时是取值域，杜绝任意文件下载。
                name = "/".join(parts[4:]) if len(parts) > 4 else ""
                if not is_viewable_artifact(name):
                    return self.json_response({"error": "未知产物"}, 404)
                if not row["result_dir"]:
                    return self.json_response({"error": "文件尚未生成"}, 404)
                target = (ROOT / row["result_dir"] / name).resolve()
                if RESULTS.resolve() not in target.parents or not target.is_file():
                    return self.json_response({"error": "文件不存在"}, 404)
                return self.serve_file(target, ARTIFACT_CONTENT_TYPES.get(name.rsplit(".", 1)[-1].lower()))
            if kind == "score":
                report = current_score(dict(row), ROOT, RESULTS)
                if report:
                    return self.json_response(report)
                return self.json_response({"error": "评分尚未生成"}, 404)
            key = {"report": "report_path", "summary": "summary_path", "log": "log_path", "manifest": "manifest_path"}.get(kind)
            if not key or not row[key]:
                if kind == "manifest" and row["summary_path"]:
                    payload = single_run_manifest(row)
                    if payload:
                        return self.json_response(payload)
                return self.json_response({"error": "文件尚未生成"}, 404)
            target = (ROOT / row[key]).resolve()
            allowed = RESULTS.resolve()
            if allowed not in target.parents or not target.is_file():
                return self.json_response({"error": "文件不存在"}, 404)
            if kind == "summary":
                return self.text_response(target.read_text(encoding="utf-8", errors="replace"), "application/json; charset=utf-8")
            if kind == "manifest":
                # 批量聚合 manifest 附带每项 metrics；旧批次落盘时还没有该字段，读取时即时补齐，
                # 保证前端聚合报告的指标列对历史记录同样有数据。
                try:
                    payload = json.loads(target.read_text(encoding="utf-8"))
                except ValueError:
                    return self.text_response(target.read_text(encoding="utf-8", errors="replace"), "text/plain; charset=utf-8")
                if isinstance(payload, dict):
                    ensure_batch_metrics(payload, ROOT)
                    ensure_image_fingerprints(payload)
                    ensure_report_details(payload, ROOT, RESULTS)
                    return self.json_response(payload)
            text = target.read_text(encoding="utf-8", errors="replace")
            if kind == "report":
                text = strip_report_scores(text)
            return self.text_response(text, "text/plain; charset=utf-8")
        if path in {"/", "/index.html"}:
            return self.serve_index()
        target = (STATIC / path.lstrip("/")).resolve()
        if STATIC.resolve() not in target.parents or not target.is_file():
            return self.text_response("Not Found", status=404)
        return self.serve_file(target)

    def do_DELETE(self) -> None:
        request_path = urlparse(self.path).path
        if not request_path.startswith("/api/runs/"):
            return self.json_response({"error": "不支持的删除目标"}, 404)
        run_id = unquote(request_path.removeprefix("/api/runs/")).strip("/")
        if not run_id or "/" in run_id:
            return self.json_response({"error": "记录 ID 无效"}, 400)
        with db() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if not row:
                return self.json_response({"error": "记录不存在"}, 404)
            if row["batch_id"]:
                return self.json_response({"error": "批次子记录不能单独删除，请删除整个批次"}, 409)
            records = [row]
            if row["is_batch"]:
                records.extend(connection.execute("SELECT * FROM runs WHERE batch_id=?", (run_id,)).fetchall())
            if any(record["status"] in {"queued", "running"} for record in records):
                return self.json_response({"error": "运行中的记录不能删除，请先等待结束或停止任务"}, 409)

            directories: set[Path] = set()
            allowed = RESULTS.resolve()
            for record in records:
                if not record["result_dir"]:
                    continue
                directory = (ROOT / record["result_dir"]).resolve()
                if allowed not in directory.parents or not directory.is_dir():
                    continue
                directories.add(directory)
            for directory in directories:
                shutil.rmtree(directory)

            ids = [record["id"] for record in records]
            connection.executemany("DELETE FROM runs WHERE id=?", [(record_id,) for record_id in ids])
        return self.json_response({"id": run_id, "deleted_records": len(ids)})

    def do_POST(self) -> None:
        request_path = urlparse(self.path).path
        if request_path == "/api/uploads":
            try:
                return self.handle_upload()
            except ValueError as exc:
                return self.json_response({"error": str(exc)}, 400)
        if request_path == "/api/runs/download":
            try:
                payload = self.body_json()
                requested = payload.get("run_ids") or []
                if not isinstance(requested, list) or not requested or len(requested) > 200:
                    raise ValueError("run_ids 必须是 1～200 个记录 ID")
                ids = list(dict.fromkeys(str(x).strip() for x in requested if str(x).strip()))
                with db() as connection:
                    placeholders = ",".join("?" * len(ids))
                    selected = connection.execute(f"SELECT * FROM runs WHERE id IN ({placeholders})", ids).fetchall()
                    rows_to_pack = list(selected)
                    batch_ids = [r["id"] for r in selected if r["is_batch"]]
                    if batch_ids:
                        placeholders = ",".join("?" * len(batch_ids))
                        rows_to_pack += connection.execute(f"SELECT * FROM runs WHERE batch_id IN ({placeholders}) ORDER BY created_at", batch_ids).fetchall()
                if len(selected) != len(ids):
                    raise ValueError("部分测试记录不存在")
                return self.send_result_archive(
                    {
                        "type": "selected-runs",
                        "selected_run_ids": ids,
                        "included_records": [r["id"] for r in rows_to_pack],
                    },
                    rows_to_pack,
                    "test-data-selected.zip",
                )
            except (ValueError, json.JSONDecodeError) as exc:
                return self.json_response({"error": str(exc)}, 400)
        if request_path.startswith("/api/runs/") and request_path.endswith("/cancel"):
            run_id = request_path.split("/")[3]
            with db() as connection:
                row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
                children = connection.execute("SELECT id FROM runs WHERE batch_id=?", (run_id,)).fetchall() if row and row["is_batch"] else []
            if not row:
                return self.json_response({"error": "记录不存在"}, 404)
            ids = [run_id] + [x["id"] for x in children]
            with ACTIVE_LOCK:
                CANCELLED_RUNS.update(ids)
                procs = [ACTIVE_PROCESSES.get(x) for x in ids]
            for proc in procs:
                if proc:
                    terminate_process(proc)
            with db() as connection:
                connection.execute("UPDATE runs SET status='interrupted', finished_at=COALESCE(finished_at,?), error=COALESCE(error,'用户主动停止测试') WHERE id IN (%s) AND status IN ('queued','running')" % ",".join("?"*len(ids)), [now_iso(), *ids])
            return self.json_response({"id": run_id, "status": "interrupted", "message": "已发送停止信号；执行过的子任务将生成中断报告"})
        if request_path == "/api/connectivity/check":
            try:
                payload = self.body_json()
                protocol = str(payload.get("protocol", "")).strip().lower()
                base_url = str(payload.get("base_url", "")).strip()
                api_key = str(payload.get("api_key", "")).strip()
                model = str(payload.get("model", "")).strip()
                path_override = str(payload.get("path_override") or "").strip()
                if protocol not in PROTOCOLS:
                    raise ValueError("未知模型协议")
                base_url = normalize_base_url(base_url)
                base_url = apply_path_override(protocol, base_url, path_override)
                if not api_key:
                    raise ValueError("api_key 不能为空")
                if not model:
                    raise ValueError("model 不能为空")
                config = ApiConfig(api_key=api_key, base_url=base_url, model=model, protocol=protocol,
                                   trust_env=trust_env_enabled())
                result = check_connectivity(config)
                return self.json_response(result, 200 if result["passed"] else 502)
            except (ValueError, json.JSONDecodeError) as exc:
                return self.json_response({"error": str(exc)}, 400)
        if request_path == "/api/models/query":
            try:
                payload = self.body_json()
                protocol = str(payload.get("protocol", "")).strip().lower()
                base_url = str(payload.get("base_url", "")).strip()
                api_key = str(payload.get("api_key", "")).strip()
                path_override = str(payload.get("path_override") or "").strip()
                if protocol not in PROTOCOLS:
                    raise ValueError("未知模型协议")
                base_url = normalize_base_url(base_url)
                base_url = apply_path_override(protocol, base_url, path_override)
                if not api_key:
                    raise ValueError("api_key 不能为空")
                try:
                    response = httpx.get(models_endpoint(protocol, base_url), headers=models_headers(protocol, api_key), timeout=30, follow_redirects=False, trust_env=trust_env_enabled())
                except httpx.HTTPError as exc:
                    return self.json_response({"error": f"模型接口连接失败: {exc}"}, 502)
                if not 200 <= response.status_code < 300:
                    detail = response.text[:1000]
                    return self.json_response({"error": f"模型接口返回 HTTP {response.status_code}", "detail": detail}, 502)
                try:
                    models = parse_models(protocol, response.json())
                except (ValueError, json.JSONDecodeError):
                    return self.json_response({"error": "模型接口未返回有效 JSON"}, 502)
                return self.json_response({"protocol": protocol, "models": models, "count": len(models)})
            except (ValueError, json.JSONDecodeError) as exc:
                return self.json_response({"error": str(exc)}, 400)
        if request_path != "/api/runs":
            return self.json_response({"error": "Not Found"}, 404)
        try:
            payload = self.body_json()
            base_url = str(payload.get("base_url", "")).strip()
            api_key = str(payload.get("api_key", "")).strip()
            model = str(payload.get("model", "")).strip()
            requested_cases = payload.get("cases") or [payload.get("case")]
            if not isinstance(requested_cases, list): raise ValueError("cases 必须是数组")
            case_names = list(dict.fromkeys(str(item or "").strip() for item in requested_cases if str(item or "").strip()))
            case_name = case_names[0] if case_names else ""
            test_kind = str(payload.get("test_kind", "load")).strip().lower()
            protocol = str(payload.get("protocol", "volcengine")).strip().lower()
            # 执行面（测试面）：前端默认只提交 model,verify；勾选「含通道面」才带 channel。
            # 未提交 = 走默认集（通道面停用），与 CLI 的 VOLC_FACES 同一套解析。
            try:
                faces_value: tuple[str, ...] = parse_faces(payload.get("faces"))
            except FaceSelectionError as exc:
                raise ValueError(str(exc)) from exc
            path_override = str(payload.get("path_override") or "").strip()
            claude_variant = str(payload.get("claude_variant") or "").strip() or None
            if test_kind == "fable" and claude_variant is not None and claude_variant not in CLAUDE_VARIANTS:
                raise ValueError("未知 Claude 模型选项")
            temporary_case = str(payload.get("temporary_case") or "").strip() or None
            billing_probe = str(payload.get("billing_probe") or "").strip() or None
            quick = bool(payload.get("quick", False))
            custom_requests = payload.get("requests")
            custom_concurrency = payload.get("concurrency")
            image_quality = payload.get("image_quality")
            image_models = payload.get("image_models")
            image_sizes = payload.get("image_sizes")
            image_aspect_ratio = None
            image_request_timeout_seconds = None
            image_hold_seconds = None
            image_prompt = None
            gemini_quality = payload.get("gemini_quality")
            gemini_models = payload.get("gemini_models")
            gemini_aspects = payload.get("gemini_aspects")
            gemini_counts = payload.get("gemini_counts")
            chat_prompt = payload.get("chat_prompt")
            chat_system = payload.get("chat_system")
            chat_max_tokens = payload.get("chat_max_tokens")
            chat_temperature = payload.get("chat_temperature")
            signature_max_tokens = payload.get("signature_max_tokens")
            seedance_prompt = None
            seedance_resolution = None
            seedance_ratio = None
            seedance_duration = None
            seedance_audio = False
            seedance_watermark = False
            seedance_image_role = None
            seedance_image_url = None
            seedance_image_path = None
            seedance_video_role = None
            seedance_video_url = None
            seedance_video_path = None
            seedance_mode = None
            if custom_requests is not None and (not isinstance(custom_requests, int) or not 1 <= custom_requests <= 100000): raise ValueError("发送请求数必须是 1～100000 的整数")
            if custom_concurrency is not None and (not isinstance(custom_concurrency, int) or not 1 <= custom_concurrency <= 1000): raise ValueError("并发数必须是 1～1000 的整数")
            if test_kind == "moderation" and custom_concurrency is None: custom_concurrency = 4
            if test_kind == "chat":
                # 对话窗口：请求数恒为 1，参数全部来自对话面板，不接受压测式的自定义参数。
                if len(case_names) != 1:
                    raise ValueError("对话窗口一次只能发送一条消息")
                chat_prompt = str(chat_prompt if chat_prompt is not None else "")
                chat_system = str(chat_system if chat_system is not None else "")
                if not chat_prompt.strip():
                    raise ValueError("请先填写要发送的用户消息")
                if len(chat_prompt) > CHAT_MAX_PROMPT_CHARS:
                    raise ValueError(f"用户消息过长（{len(chat_prompt)} 字符，上限 {CHAT_MAX_PROMPT_CHARS}）")
                if len(chat_system) > CHAT_MAX_SYSTEM_CHARS:
                    raise ValueError(f"系统提示词过长（{len(chat_system)} 字符，上限 {CHAT_MAX_SYSTEM_CHARS}）")
                if chat_max_tokens is None:
                    chat_max_tokens = CHAT_DEFAULT_MAX_TOKENS
                if isinstance(chat_max_tokens, bool) or not isinstance(chat_max_tokens, int) or not 1 <= chat_max_tokens <= CHAT_MAX_MAX_TOKENS:
                    raise ValueError(f"最大输出 Token 必须是 1～{CHAT_MAX_MAX_TOKENS} 的整数")
                if chat_temperature is not None:
                    if isinstance(chat_temperature, bool) or not isinstance(chat_temperature, (int, float)) or not 0 <= chat_temperature <= 2:
                        raise ValueError("温度必须是 0～2 的数字")
                    chat_temperature = float(chat_temperature)
                custom_requests, custom_concurrency = 1, 1
                quick = False
            else:
                chat_prompt = chat_system = chat_max_tokens = chat_temperature = None
            pelican_rounds = pelican_concurrency = None
            if test_kind == "gpt":
                # 请求轮数/并发数只对产物类用例（鹈鹕骑车 SVG）有意义：其余 GPT 用例请求数固定，
                # 界面上的残留值必须被丢弃——否则会被静默带进别的用例（"以为调了其实没生效"）。
                # 注意判定按**本批是否包含**产物类用例，而不是只看第一个用例：同批提交
                # 「P0 + PELICAN」时轮数仍要落到 PELICAN 那条子任务上。
                if any(name in PELICAN_CASE_IDS for name in case_names):
                    pelican_rounds = 1 if custom_requests is None else custom_requests
                    pelican_concurrency = 1 if custom_concurrency is None else custom_concurrency
                    if not 1 <= pelican_rounds <= PELICAN_MAX_REQUESTS:
                        raise ValueError(f"请求轮数必须是 1～{PELICAN_MAX_REQUESTS} 的整数")
                    if not 1 <= pelican_concurrency <= PELICAN_MAX_CONCURRENCY:
                        raise ValueError(f"并发数必须是 1～{PELICAN_MAX_CONCURRENCY} 的整数")
                custom_requests, custom_concurrency = None, None
            if test_kind == "gpt-image":
                if len(case_names) != 1:
                    raise ValueError("图片压测一次只能选择一个子项目")
                if case_name == GPT_IMAGE_TOKEN_CASE:
                    custom_requests, custom_concurrency = 3, 1
                    image_quality, image_models, image_sizes = None, None, None
                elif custom_requests is None or custom_concurrency is None:
                    raise ValueError("图片压测必须填写总请求数和并发数")
            if test_kind == "gemini-image":
                if len(case_names) != 1:
                    raise ValueError("Gemini 图片测试一次只能选择一个子项目")
                if custom_requests is None or custom_concurrency is None:
                    raise ValueError("Gemini 图片测试必须填写总请求数和并发数")
            if test_kind == "seedance":
                # 视频生成：参数是**显式输入**（提示词/分辨率/比例/时长/图片/开关），
                # 执行规模走通用自定义压测参数。所有取值域在这里二次校验，不信任前端。
                if len(case_names) != 1:
                    raise ValueError("Seedance 视频生成一次只能选择一个子项目")
                if custom_requests is None or custom_concurrency is None:
                    raise ValueError("Seedance 视频生成必须填写视频任务数和并发数")
                if not 1 <= custom_requests <= seedance_cases.SEEDANCE_MAX_REQUESTS:
                    raise ValueError(f"视频任务数必须是 1～{seedance_cases.SEEDANCE_MAX_REQUESTS} 的整数")
                if not 1 <= custom_concurrency <= seedance_cases.SEEDANCE_MAX_CONCURRENCY:
                    raise ValueError(f"并发数必须是 1～{seedance_cases.SEEDANCE_MAX_CONCURRENCY} 的整数")
                seedance_prompt = str(payload.get("seedance_prompt") or "")
                if not seedance_prompt.strip():
                    raise ValueError("请先填写视频提示词")
                if len(seedance_prompt) > seedance_cases.SEEDANCE_MAX_PROMPT_CHARS:
                    raise ValueError(
                        f"提示词过长（{len(seedance_prompt)} 字符，"
                        f"上限 {seedance_cases.SEEDANCE_MAX_PROMPT_CHARS}）"
                    )
                seedance_resolution = str(
                    payload.get("seedance_resolution") or seedance_cases.SEEDANCE_DEFAULT_RESOLUTION
                )
                if seedance_resolution not in seedance_cases.SEEDANCE_RESOLUTIONS:
                    raise ValueError(
                        f"分辨率必须是 {'/'.join(seedance_cases.SEEDANCE_RESOLUTIONS)} 之一"
                    )
                seedance_ratio = str(payload.get("seedance_ratio") or seedance_cases.SEEDANCE_DEFAULT_RATIO)
                if seedance_ratio not in seedance_cases.SEEDANCE_RATIOS:
                    raise ValueError(f"画面比例必须是 {'/'.join(seedance_cases.SEEDANCE_RATIOS)} 之一")
                raw_duration = payload.get("seedance_duration")
                if raw_duration is None or raw_duration == "":
                    seedance_duration = seedance_cases.SEEDANCE_DEFAULT_DURATION
                elif isinstance(raw_duration, bool) or not isinstance(raw_duration, int):
                    raise ValueError("时长必须是整数秒")
                else:
                    seedance_duration = raw_duration
                # 时长区间按**被测模型档位**收紧（2.0 → 4–15，2.5 → 4–30）：该网关把参数
                # 校验放在异步执行阶段，档位越界不会在提交期报错，而是烧掉一次完整生成才
                # 以「任务级参数校验失败」收场，所以这里必须提前拦（前端只拿到并集上限）。
                # 档位识别不出来时时长区间退化成并集 4–30。
                duration_min, duration_max = seedance_cases.duration_range_for(model)
                if not duration_min <= seedance_duration <= duration_max:
                    raise ValueError(
                        f"时长必须是 {duration_min}～{duration_max} 的整数秒"
                        f"（按模型 {model} 的档位解析；2.0 上限 15，2.5 上限 30）"
                    )
                role_ids = {role for role, _ in seedance_cases.SEEDANCE_IMAGE_ROLES}
                seedance_image_role = str(
                    payload.get("seedance_image_role") or seedance_cases.SEEDANCE_DEFAULT_IMAGE_ROLE
                )
                if seedance_image_role not in role_ids:
                    raise ValueError(f"图片位置必须是 {'/'.join(sorted(role_ids))} 之一")
                # 两个开关缺省（键不存在或为 null）时按官方默认补齐（generate_audio 开、
                # watermark 关）；显式传布尔的照样以传入值为准——前端一定会传，这里兜住的是
                # 直接打 API 的调用方。
                raw_audio = payload.get("seedance_audio")
                seedance_audio = bool(
                    seedance_cases.SEEDANCE_DEFAULT_GENERATE_AUDIO if raw_audio is None else raw_audio
                )
                raw_watermark = payload.get("seedance_watermark")
                seedance_watermark = bool(
                    seedance_cases.SEEDANCE_DEFAULT_WATERMARK if raw_watermark is None else raw_watermark
                )
                # 上传图片：只认本服务上传目录里的文件（取值域受限，杜绝任意文件读取）。
                raw_image_path = str(payload.get("seedance_image_path") or "").strip()
                if raw_image_path:
                    resolved = (ROOT / raw_image_path).resolve()
                    if SEEDANCE_UPLOAD_DIR.resolve() not in resolved.parents or not resolved.is_file():
                        raise ValueError("上传的图片无效或已过期，请重新上传")
                    seedance_image_path = str(resolved.relative_to(ROOT))
                seedance_image_url = str(payload.get("seedance_image_url") or "").strip()
                if seedance_image_url and not seedance_image_url.startswith(("http://", "https://", "data:")):
                    raise ValueError("图片地址必须是 http(s):// 或 data: URI")
                # 参考视频：角色、URL（唯一会提交给上游的形态）、本地素材（只归档）。
                video_role_ids = {role for role, _ in seedance_cases.SEEDANCE_VIDEO_ROLES}
                raw_video_role = payload.get("seedance_video_role")
                seedance_video_role = (
                    seedance_cases.SEEDANCE_DEFAULT_VIDEO_ROLE
                    if raw_video_role is None
                    else str(raw_video_role).strip()
                )
                if seedance_video_role not in video_role_ids:
                    raise ValueError(
                        "参考视频位置必须是 "
                        + "/".join(item or "(空=不带 role)" for item in sorted(video_role_ids))
                        + " 之一"
                    )
                raw_video_path = str(payload.get("seedance_video_path") or "").strip()
                if raw_video_path:
                    resolved = (ROOT / raw_video_path).resolve()
                    if SEEDANCE_UPLOAD_DIR.resolve() not in resolved.parents or not resolved.is_file():
                        raise ValueError("上传的参考视频无效或已过期，请重新上传")
                    suffix = resolved.suffix.lower().lstrip(".")
                    if suffix not in seedance_cases.SEEDANCE_INPUT_VIDEO_SUFFIXES:
                        raise ValueError(
                            "上传的参考视频格式不支持：只接受 "
                            + "/".join("." + item for item in seedance_cases.SEEDANCE_INPUT_VIDEO_SUFFIXES)
                        )
                    if resolved.stat().st_size > seedance_cases.SEEDANCE_MAX_INPUT_VIDEO_BYTES:
                        raise ValueError(
                            f"上传的参考视频过大（上限 {seedance_cases.SEEDANCE_MAX_INPUT_VIDEO_BYTES} 字节）"
                        )
                    seedance_video_path = str(resolved.relative_to(ROOT))
                seedance_video_url = str(payload.get("seedance_video_url") or "").strip()
                if seedance_video_url and not seedance_video_url.startswith(("http://", "https://", "data:")):
                    raise ValueError("视频地址必须是 http(s):// 或 data: URI")
                # 生成方式：界面一定显式传（前端会按模式把不该带的输入置空），此时与输入
                # 严格对齐——少给（声明带图却没有图）或多给（声明文生视频却带着图/视频）都
                # 直接拦回。**未传**（直接打 API 的调用方）则按素材推导、不做冲突校验，
                # 与执行器 runner.py 的规则保持一致。
                raw_mode = str(payload.get("seedance_mode") or "").strip().lower()
                if raw_mode and raw_mode not in seedance_cases.SEEDANCE_MODE_IDS:
                    raise ValueError(
                        "生成方式必须是 "
                        + "/".join(seedance_cases.SEEDANCE_MODE_IDS)
                        + f" 之一：{raw_mode}"
                    )
                seedance_mode = seedance_cases.resolve_mode(
                    raw_mode,
                    has_image=bool(seedance_image_path or seedance_image_url),
                    has_video=bool(seedance_video_path or seedance_video_url),
                )
                if raw_mode:
                    if seedance_cases.mode_allows_image(seedance_mode):
                        if not (seedance_image_path or seedance_image_url):
                            raise ValueError(
                                f"生成方式为「{seedance_mode}」但没有图片：请上传参考图或填写图片地址，"
                                "或改用「仅文生视频」/「带参考视频」"
                            )
                    elif seedance_image_path or seedance_image_url:
                        raise ValueError(
                            f"生成方式为「{seedance_mode}」，不会提交图片："
                            "请清除图片，或改用「带参考图」/「两者都带」"
                        )
                    if seedance_cases.mode_allows_video(seedance_mode):
                        if not seedance_video_url:
                            hint = ("（只上传本地视频不算：异步任务面只接受 URL，本地素材仅归档留痕）"
                                    if seedance_video_path else "")
                            raise ValueError(
                                f"生成方式为「{seedance_mode}」但没有视频地址{hint}："
                                "请填写参考视频地址，或改用「仅文生视频」/「带参考图」"
                            )
                    elif seedance_video_path or seedance_video_url:
                        raise ValueError(
                            f"生成方式为「{seedance_mode}」，不会提交参考视频："
                            "请清除视频输入，或改用「带参考视频」/「两者都带」"
                        )
            if test_kind == "load" and "RANDOM_LOAD" in case_names:
                if len(case_names) != 1:
                    raise ValueError("随机输入压测一次只能单独运行")
                if custom_requests is None or custom_concurrency is None:
                    raise ValueError("随机输入压测必须填写请求数和并发数")
                quick = False
            base_url = normalize_base_url(base_url)
            # 自定义请求路径：这里只做校验，入库仍存 host-only 域名，
            # 执行阶段（_execute_child）再把根前缀拼回 resolved_base_url。
            split_path_override(path_override)
            if not api_key:
                raise ValueError("api_key 不能为空")
            if not model:
                raise ValueError("model 不能为空")
            if test_kind == "kvv" and case_names == [KVV_TEMP_CASE_ID] and temporary_case is not None:
                # 选择项是合并案例 id（batch 名）；原子案例 id 继续兼容。
                if find_case(kvv_temp_all_cases(), temporary_case) is None:
                    raise ValueError("未知 K3 临时测试案例")
            if test_kind == "kvv" and KVV_TEMP_CASE_ID not in case_names:
                # 官方 KVV 套件不认子项选择：一律置空，避免把上一条 K3 选择的残留
                # 带进官方套件（派发层据此判定执行链）。
                temporary_case = None
            if test_kind == "kvv":
                # 手动不计分专项不受通道开关拦截；其余卡片按执行面过滤。
                from kvv.faces import CASE_FACES as KVV_FACES
                blocked = [n for n in case_names
                           if n not in OPTIONAL_UNSCORED_CASES
                           and face_of(KVV_FACES, n) == FACE_CHANNEL and FACE_CHANNEL not in faces_value]
                if blocked:
                    raise ValueError(
                        f"{'、'.join(blocked)} 属通道契约面，"
                        "已默认停用；如需执行请勾选「含通道面」"
                    )
            # 中转计费探针：只要本次提交包含计费探针，子选项须是 BILLING-ALL 或某条探针 id。
            # 不能要求 case_names 恰好只有探针一条——探针在 Claude tab 下常与各套件**同批**
            # 提交（批量场景），旧的等值判断会把整批的 billing_probe 置空，探针子任务于是
            # 落到 run_fable5.py 报 "invalid choice"（历史批次因此整批不准入）。
            # 未选中探针时一律置空，避免把上一条选择的残留带进普通 Claude 用例。
            billing_probe_requested = test_kind == "fable" and BILLING_PROBE_CASE_ID in case_names
            if billing_probe_requested:
                billing_probe = billing_probe or BILLING_PROBE_ALL
                if not is_billing_probe(billing_probe):
                    raise ValueError("未知的中转计费探针选择")
            else:
                billing_probe = None
            available = {"LEGAL_AGENT_BENCHMARK": LEGAL_AGENT_BENCHMARK_CASE} if test_kind == "legal-agent-benchmark" else {"SWE_ATLAS_RF": SWE_ATLAS_RF_CASE} if test_kind == "swe-atlas-rf" else {"SWE_ATLAS_TW": SWE_ATLAS_TW_CASE} if test_kind == "swe-atlas-tw" else {"DEEPSWE": DEEPSWE_CASE} if test_kind == "deepswe" else {"ARXIVMATH": ARXIVMATH_CASE} if test_kind == "arxivmath" else CASES if test_kind == "load" else MODERATION_CASES if test_kind == "moderation" else CHAT_CASES if test_kind == "chat" else KVV_CASES if test_kind == "kvv" else DEEPSEEK_CASES if test_kind == "deepseek" else GLM_CASES if test_kind == "glm" else QWEN_CASES if test_kind == "qwen" else GPT56_CASES if test_kind == "gpt" else CLAUDE_RUNNABLE_CASES if test_kind == "fable" else GPT_IMAGE_CASES if test_kind == "gpt-image" else GEMINI_IMAGE_CASES if test_kind == "gemini-image" else SEEDANCE_CASES if test_kind == "seedance" else GEMINI3_CASES if test_kind == "gemini" else AUTH_CASES if test_kind == "authenticity" else FORENSICS_CASES if test_kind == "forensics" else {}
            if test_kind in {"kvv", "deepseek", "glm"}:
                available = dict(available)
                available.update({name: item for name, item in CONTRACT_CASES.items() if item["test_kind"] == test_kind})
            if test_kind == "kvv":
                # K3 临时案例集是「Kimi 测试」模块里的第二条执行链入口。
                available[KVV_TEMP_CASE_ID] = kvv_temp_case_payload()
            available = dict.fromkeys(available) if isinstance(available, list) else dict(available)
            available.update({name: economy_case_payload(name) for name, kind in ECONOMY_CASES.items() if kind == test_kind})
            for name in case_names:
                if name in ECONOMY_CASES:
                    economy_profile(test_kind, model)
                    if not {"model", "verify"} <= set(faces_value):
                        raise ValueError("经济验证需要同时选择模型能力与验真面")
            if not case_names or any(name not in available for name in case_names):
                raise ValueError("未知测试类型")
            if test_kind in {"gpt-image", "gemini-image"} and GPT_IMAGE_TOKEN_CASE not in case_names:
                image_prompt = normalize_image_prompt(payload.get("image_prompt"))
            if test_kind == "kvv" and KVV_TEMP_CASE_ID in case_names and case_names != [KVV_TEMP_CASE_ID]:
                raise ValueError("K3 临时案例集只能单独运行")
            if test_kind == "gpt-image":
                image_case = GPT_IMAGE_CASES[case_names[0]]
                image_quality = _choice_list(image_quality, image_case.qualities, "quality")
                image_models = _choice_list(image_models, image_case.model_choices, "模型")
                image_sizes = _choice_list(image_sizes, image_case.sizes, "尺寸")
                if case_names[0] != GPT_IMAGE_TOKEN_CASE:
                    model_quality_choices(image_models, image_quality)
                    image_aspect_ratio = payload.get("image_aspect_ratio")
                    if image_aspect_ratio is not None and image_aspect_ratio not in GPT_IMAGE_ASPECT_RATIOS:
                        raise ValueError("aspect_ratio 包含不支持的值")
                    raw_timeout = payload.get("image_request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS)
                    try:
                        image_request_timeout_seconds = float(raw_timeout)
                    except (TypeError, ValueError):
                        raise ValueError("图片请求时间必须是大于 0 的有限秒数")
                    if isinstance(raw_timeout, bool) or not math.isfinite(image_request_timeout_seconds) or image_request_timeout_seconds <= 0:
                        raise ValueError("图片请求时间必须是大于 0 的有限秒数")
                    # 压测持续时长：**默认不启用**（缺省 / null / 空串一律视为关闭）。
                    # 启用后请求数退化为上限保险丝，判定口径同步切换到 request_count_within_cap。
                    raw_hold = payload.get("image_hold_seconds")
                    if raw_hold not in (None, ""):
                        try:
                            image_hold_seconds = float(raw_hold)
                        except (TypeError, ValueError):
                            raise ValueError("压测持续时长必须是大于 0 的有限秒数")
                        if isinstance(raw_hold, bool) or not math.isfinite(image_hold_seconds) or image_hold_seconds <= 0:
                            raise ValueError("压测持续时长必须是大于 0 的有限秒数")
            if test_kind == "gemini-image":
                gemini_case = GEMINI_IMAGE_CASES[case_names[0]]
                gemini_quality = _choice_list(gemini_quality, gemini_case.qualities, "强度(imageSize)")
                gemini_models = _choice_list(gemini_models, gemini_case.models, "模型")
                gemini_aspects = _choice_list(gemini_aspects, gemini_case.aspect_ratios, "尺寸(aspectRatio)")
                gemini_counts = _int_choice_list(gemini_counts, gemini_case.counts, "生成数量")
            if protocol not in PROTOCOLS:
                raise ValueError("未知模型协议")
            def protocols_for(name: str) -> tuple[str, ...]:
                if name in ECONOMY_CASES:
                    return tuple(economy_case_payload(name)["protocols"])
                if test_kind == "kvv" and name == KVV_TEMP_CASE_ID:
                    # K3 临时案例集走 Chat Completions 面。
                    return KVV_TEMP_PROTOCOLS
                if test_kind == "fable" and name == BILLING_PROBE_CASE_ID:
                    # 计费探针只在 Anthropic Messages 面取证。
                    return BILLING_PROBE_PROTOCOLS
                if test_kind == "fable" and name == PROTO_SIGNATURE_CASE_ID:
                    return PROTO_SIGNATURE_PROTOCOLS
                if test_kind == "fable" and name == ZERO_INJECTION_CASE_ID:
                    # 0 注入判定走 Anthropic Messages 面（金丝雀/签名都只在 Messages 面取证）。
                    return ZERO_INJECTION_PROTOCOLS
                if test_kind == "chat":
                    # 对话窗口只走文字面（openai / volcengine / anthropic / gemini）；
                    # 图像协议家族虽同，但 test_kinds 白名单不含 chat，由 supports_test_kind 拦回。
                    return CHAT_CASES[name].protocols
                if test_kind == "forensics":
                    # 中转验真只走文字面；图像/视频协议靠家族与 test_kinds 白名单拦回。
                    return FORENSICS_CASES[name].protocols
                if test_kind == "legal-agent-benchmark":
                    return LEGAL_AGENT_BENCHMARK_CASE.protocols
                if test_kind == "swe-atlas-rf":
                    return SWE_ATLAS_RF_CASE.protocols
                if test_kind == "swe-atlas-tw":
                    return SWE_ATLAS_TW_CASE.protocols
                if test_kind == "deepswe":
                    return DEEPSWE_CASE.protocols
                if test_kind == "arxivmath":
                    return ARXIVMATH_CASE.protocols
                if test_kind == "gpt-image":
                    return GPT_IMAGE_CASES[name].protocols
                if test_kind == "gemini-image":
                    return GEMINI_IMAGE_CASES[name].protocols
                if test_kind == "seedance":
                    # Seedance 只走 seedance 家族（异步任务面）；文字/图像协议一律拦回。
                    return SEEDANCE_CASES[name].protocols
                if test_kind == "gemini":
                    return GEMINI3_CASES[name].protocols
                if name in CONTRACT_CASES:
                    return CONTRACT_CASES[name]["protocols"]
                return KVV_CASES[name].protocols if test_kind == "kvv" else DEEPSEEK_CASES[name].protocols if test_kind == "deepseek" else GLM_CASES[name].protocols if test_kind == "glm" else QWEN_CASES[name].protocols if test_kind == "qwen" else FABLE5_CASES[claude_suite_name(name)].protocols if test_kind == "fable" else GPT56_CASES[name].protocols if test_kind == "gpt" else ("openai", "volcengine") if test_kind == "authenticity" else ("openai", "volcengine") if test_kind == "moderation" else tuple(PROTOCOLS)
            if test_kind == "kvv" and KVV_TEMP_CASE_ID in case_names and protocol not in OPENAI_COMPATIBLE_PROTOCOLS:
                raise ValueError("K3 临时案例集只支持 openai / volcengine 协议")
            if billing_probe_requested and protocol != "anthropic":
                raise ValueError("中转计费探针只支持 Anthropic Messages 协议面")
            unsupported = [name for name in case_names if not supports_test_kind(protocol, test_kind, protocols_for(name))]
            if unsupported:
                raise ValueError(f"{', '.join(unsupported)} 不支持 {protocol} 协议")
            preset_names = [name for name in case_names if name in LOAD_PRESETS]
            if preset_names and (len(case_names) != 1 or custom_requests is not None or custom_concurrency is not None):
                raise ValueError("简单压测和完整压测使用固定组合参数，不能与其他测试或自定义参数同时运行")
            if preset_names:
                preset = LOAD_PRESETS[preset_names[0]]
                quick = False
                execution_cases = [
                    {
                        # L1–L5 用预设样本数；O1 是固定输出专项，样本数取自己的档位要求
                        # （volcperf/cases.py 的 O1.samples），不随预设缩放。
                        "name": name,
                        "requests": LOAD_CASES[name].samples if name == "O1" else preset["samples"],
                        "concurrency": LOAD_CASES[name].concurrency if name == "O1" else preset["concurrency"],
                        "warmup_seconds": 0,
                    }
                    for name in preset["cases"]
                ]
            elif test_kind == "kvv" and case_name == KVV_TEMP_CASE_ID:
                execution_cases = [{"name": temporary_case or KVV_TEMP_CASE_ID, "requests": custom_requests, "concurrency": custom_concurrency, "warmup_seconds": None}]
            else:
                execution_cases = [
                    {"name": name, "requests": custom_requests, "concurrency": custom_concurrency, "warmup_seconds": None}
                    for name in case_names
                ]
            if test_kind == "gpt" and pelican_rounds is not None:
                # 同批只有产物类用例带轮数/并发；其余 GPT 用例保持 requests=None，
                # 派发时（pelican_round_args）也不会给它们追加 --requests/--concurrency。
                for execution in execution_cases:
                    if execution["name"] in PELICAN_CASE_IDS:
                        execution["requests"], execution["concurrency"] = pelican_rounds, pelican_concurrency
            # 执行范围（多模型）：前端把「查询模型」结果里的多选模型随 case_models 提交。
            # 只认 ≥2 个模型（单选/未选与直接填 MODEL 同口径），多选时按模型拆子任务成批
            # 串行执行，每个模型一份子记录与报告行。当前只在**声明了 model_scope 的卡片**上生效：
            # 签名两步校验（Claude 面）与千问档位/视觉（同族多档对照）。
            scope_models: list[str] = []
            # 输出预算只对签名两步校验有意义；其他用例的界面残留值必须丢弃，
            # 否则会被静默带进别的套件（"以为调了其实没生效"）。
            signature_max_tokens_value: int | None = None
            scope_enabled = (
                (bool(case_names) and all(name in ECONOMY_CASES for name in case_names))
                or (test_kind == "fable" and bool(case_names)
                    and all(name in ECONOMY_CASES or name == PROTO_SIGNATURE_CASE_ID for name in case_names))
                # 千问六款同族不同档：档位指纹与视觉卡片开放多模型对照。
                or (test_kind == "glm" and bool(case_names)
                    and all(name in ECONOMY_CASES or getattr(GLM_CASES.get(name), "model_scope", False) for name in case_names))
                or (test_kind == "qwen" and bool(case_names)
                    and all(name in ECONOMY_CASES or getattr(QWEN_CASES.get(name), "model_scope", False) for name in case_names))
            )
            if scope_enabled:
                raw_scope = payload.get("case_models")
                if isinstance(raw_scope, str):
                    raw_scope = [raw_scope]
                if isinstance(raw_scope, list):
                    for entry in raw_scope:
                        text = str(entry or "").strip()
                        if text and text not in scope_models:
                            scope_models.append(text)
            if test_kind == "fable" and PROTO_SIGNATURE_CASE_ID in case_names:
                if signature_max_tokens is None or signature_max_tokens == "":
                    signature_max_tokens_value = SIGNATURE_DEFAULT_MAX_TOKENS
                elif (isinstance(signature_max_tokens, bool)
                      or not isinstance(signature_max_tokens, int)
                      or not 1 <= signature_max_tokens <= SIGNATURE_MAX_MAX_TOKENS):
                    raise ValueError(f"输出预算必须是 1～{SIGNATURE_MAX_MAX_TOKENS} 的整数")
                else:
                    signature_max_tokens_value = signature_max_tokens
            if any(name in ECONOMY_CASES for name in case_names):
                for scoped_model in scope_models:
                    economy_profile(test_kind, scoped_model)
            # Seedance 视频生成的内容型参数（提示词/图片路径等）；非 seedance 任务全为 None。
            seedance_fields = {
                "seedance_prompt": seedance_prompt,
                "seedance_resolution": seedance_resolution,
                "seedance_ratio": seedance_ratio,
                "seedance_duration": seedance_duration,
                "seedance_mode": seedance_mode,
                "seedance_audio": seedance_audio,
                "seedance_watermark": seedance_watermark,
                "seedance_image_role": seedance_image_role,
                "seedance_image_url": seedance_image_url,
                "seedance_image_path": seedance_image_path,
                "seedance_video_role": seedance_video_role,
                "seedance_video_url": seedance_video_url,
                "seedance_video_path": seedance_video_path,
            }
            run_id = uuid.uuid4().hex[:16]; created=now_iso()
            faces_text = ",".join(faces_value)
            multi_model = len(scope_models) > 1
            if len(execution_cases) > 1 or multi_model:
                children=[]
                # 多模型批次父记录的 model 列存摘要文案；逐模型明细在子记录与
                # aggregation_manifest.json 的 results[].model 里。
                parent_model = f"{len(scope_models)} 个模型" if multi_model else model
                with db() as connection:
                    connection.execute("INSERT INTO runs(id,test_kind,case_name,protocol,base_url,path_override,model,status,quick,created_at,is_batch,case_ids,faces) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                       (run_id,test_kind,case_names[0] if len(case_names) == 1 else "MULTI",protocol,base_url,path_override,parent_model,"queued",int(quick),created,1,json.dumps(case_names),faces_text))
                    for index,execution in enumerate(execution_cases):
                        name = execution["name"]
                        # 计费探针是同批里的**一条**子任务：只有它携带 billing_probe
                        # （执行器/协议面/结果别名都据此切换），同批其它套件必须保持空值。
                        child_billing_probe = billing_probe if name == BILLING_PROBE_CASE_ID else None
                        # 探针子记录按所选子项归档（BILLING-ALL / BILLING-xx），与单条运行口径一致。
                        stored_name = child_billing_probe or name
                        for child_model in (scope_models if multi_model else [model]):
                            child_id=uuid.uuid4().hex[:16]; child_created=f"{created}.{len(children):03d}"
                            connection.execute("INSERT INTO runs(id,test_kind,case_name,protocol,base_url,path_override,model,status,quick,created_at,batch_id,faces) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                                               (child_id,test_kind,stored_name,protocol,base_url,path_override,child_model,"queued",int(quick),child_created,run_id,faces_text))
                            children.append({"id":child_id,"batch_id":run_id,"test_kind":test_kind,"case":name,"protocol":protocol,"base_url":base_url,"path_override":path_override,"api_key":api_key,"model":child_model,"quick":quick,"requests":execution["requests"],"concurrency":execution["concurrency"],"warmup_seconds":execution["warmup_seconds"],"image_prompt":image_prompt,"image_quality":image_quality,"image_models":image_models,"image_sizes":image_sizes,"image_aspect_ratio":image_aspect_ratio,"image_request_timeout_seconds":image_request_timeout_seconds,"image_hold_seconds":image_hold_seconds,"gemini_quality":gemini_quality,"gemini_models":gemini_models,"gemini_aspects":gemini_aspects,"gemini_counts":gemini_counts,"temporary_case":temporary_case,"billing_probe":child_billing_probe,"chat_prompt":chat_prompt,"chat_system":chat_system,"chat_max_tokens":chat_max_tokens,"chat_temperature":chat_temperature,"signature_max_tokens":signature_max_tokens_value,"claude_variant":claude_variant,"faces":faces_text, **seedance_fields})
                start_worker()
                JOB_QUEUE.put({"id":run_id,"is_batch":True,"test_kind":test_kind,"api_key":api_key,"children":children})
            else:
                execution = execution_cases[0]
                if test_kind == "kvv" and case_name == KVV_TEMP_CASE_ID and temporary_case:
                    stored_case_name = temporary_case
                elif test_kind == "fable" and billing_probe:
                    # 计费探针按所选子项归档（BILLING-ALL / BILLING-xx），便于记录页显示。
                    stored_case_name = billing_probe
                else:
                    stored_case_name = case_name
                with db() as connection:
                    connection.execute("INSERT INTO runs(id,test_kind,case_name,protocol,base_url,path_override,model,status,quick,created_at,case_ids,faces) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                                       (run_id,test_kind,stored_case_name,protocol,base_url,path_override,model,"queued",int(quick),created,json.dumps(case_names),faces_text))
                start_worker()
                JOB_QUEUE.put({"id":run_id,"test_kind":test_kind,"case":execution["name"],"protocol":protocol,"base_url":base_url,"path_override":path_override,"api_key":api_key,"model":model,"quick":quick,"requests":execution["requests"],"concurrency":execution["concurrency"],"warmup_seconds":execution["warmup_seconds"],"image_prompt":image_prompt,"image_quality":image_quality,"image_models":image_models,"image_sizes":image_sizes,"image_aspect_ratio":image_aspect_ratio,"image_request_timeout_seconds":image_request_timeout_seconds,"image_hold_seconds":image_hold_seconds,"gemini_quality":gemini_quality,"gemini_models":gemini_models,"gemini_aspects":gemini_aspects,"gemini_counts":gemini_counts,"temporary_case":temporary_case,"billing_probe":billing_probe,"chat_prompt":chat_prompt,"chat_system":chat_system,"chat_max_tokens":chat_max_tokens,"chat_temperature":chat_temperature,"signature_max_tokens":signature_max_tokens_value,"claude_variant":claude_variant,"faces":faces_text, **seedance_fields})
            with db() as connection:
                row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            return self.json_response(public_run(row), HTTPStatus.ACCEPTED)
        except (ValueError, json.JSONDecodeError) as exc:
            return self.json_response({"error": str(exc)}, 400)

    def serve_file(self, path: Path, content_type: str | None = None) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def serve_index(self) -> None:
        """服务 index.html，并给静态资源引用附加基于文件 mtime 的版本号。

        旧标签页/激进缓存可能长期持有旧版 app.js/app.css（旧 JS 配新接口会静默失败，
        表现为列表恒为空、搜索框样式错乱）；版本号随文件更新变化，保证刷新后必取新资源。
        """
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        assets = {"href=\"/app.css\"": "app.css", "src=\"/app.js\"": "app.js",
                  "src=\"/capability-radar.js\"": "capability-radar.js",
                  "src=\"/case-presets.js\"": "case-presets.js"}
        for attribute, filename in assets.items():
            asset_path = STATIC / filename
            if asset_path.is_file():
                html = html.replace(attribute, f'{attribute[:-1]}?v={int(asset_path.stat().st_mtime)}"')
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_result_archive(
        self, metadata: dict[str, Any], rows_to_pack: list[Any], filename: str
    ) -> None:
        """把结果目录打包成 zip 发给客户端。

        图片套件会把生成的图片原文件保存在结果目录的 images/ 下，整包体积可能达到
        数 GB，因此先归档到结果目录里的临时文件，再分块写出，避免把整个包读进内存。
        """
        allowed_root = RESULTS.resolve()
        archive_path = RESULTS / f".download-{uuid.uuid4().hex}.zip"
        headers_sent = False
        try:
            with archive_path.open("w+b") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("package-metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
                seen_dirs: set[str] = set()
                for packed_row in rows_to_pack:
                    result_dir = packed_row["result_dir"]
                    if not result_dir:
                        continue
                    target_dir = (ROOT / result_dir).resolve()
                    if allowed_root not in target_dir.parents or not target_dir.is_dir():
                        continue
                    folder = target_dir.name
                    if folder in seen_dirs:
                        continue
                    seen_dirs.add(folder)
                    for source in sorted(target_dir.rglob("*")):
                        if source.is_file():
                            archive_name = f"{folder}/{source.relative_to(target_dir)}"
                            if source.name == "report.md":
                                bundle.writestr(archive_name, strip_report_scores(source.read_text(encoding="utf-8", errors="replace")))
                            else:
                                bundle.write(source, archive_name)
            size = archive_path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            headers_sent = True
            with archive_path.open("rb") as payload:
                shutil.copyfileobj(payload, self.wfile, length=1024 * 1024)
        except OSError as exc:
            # 断连（BrokenPipe）与磁盘错误都不能让服务端无声无息；响应头已发出时只能记日志。
            print(f"打包下载失败: {exc}", file=sys.stderr)
            if not headers_sent:
                self.json_response({"error": f"打包失败: {exc}"}, 500)
        finally:
            archive_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Volcengine K3 测试管理页面")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8085)
    args = parser.parse_args()
    initialize(); start_worker()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"测试管理页面：http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
