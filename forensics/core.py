"""中转验真执行核心：跨协议传输层 + 三条取证探针。

设计要点（与平台既有约束对齐）：

* **证据链完整**：每个 HTTP 往返都落 ``requests.jsonl``（含请求体、响应头、
  响应体、耗时），最后交 ``common.finalizer.finalize_result_dir`` 统一收尾。
* **非流式请求**：取证探针只关心内容与错误信封，不需要流式；Gemini 用
  ``:generateContent``（非 SSE 端点），Accept 一律 ``application/json``。
* **判定口径**：信任级矛盾才 ``failed``（两条以上独立指纹冲突 / 泄露高危信息）；
  启发式信号一律 ``warning`` 并下调准入但不判模型失败；账号/瞬时故障/不支持
  视觉输入记 ``blocked`` 只记录不阻断。
"""
from __future__ import annotations

import base64
import json
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from common.audit import AuditLog, redact
from common.png_fixture import (
    BLUE,
    GREEN,
    RED,
    WHITE,
    YELLOW,
    data_url,
    png_chessboard,
    png_halves,
    png_solid,
    png_text,
)
from common.report_style import case_table_report
from volcperf.protocols import OPENAI_COMPATIBLE_PROTOCOLS, api_root, protocol_spec

from .cases import (
    CASES,
    ERROR_ENVELOPE_LEAKAGE,
    IMPLICIT_IDENTITY,
    IMAGE_UNDERSTANDING,
    ForensicsCase,
)

# 账号/权限/额度/端点不存在 = 平台前提不满足，记 blocked（不判模型问题）。
_BLOCKED_STATUSES = {401, 402, 403, 404, 415, 422, 429}
_DEFAULT_TIMEOUT_SECONDS = 120.0
_MAX_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 1.0
# 上游错误信封里常见的是"参数/能力类"措辞；带 image/not support 的 4xx 视为不支持视觉输入。
_VISION_UNSUPPORTED_MARKERS = ("image", "multimodal", "vision", "图片", "图像", "not support", "unsupported")

_IDENTITY_MAX_TOKENS = 2048
_VISION_MAX_TOKENS = 1024


@dataclass(frozen=True)
class ForensicsConfig:
    api_key: str
    base_url: str
    model: str
    protocol: str
    request_timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    verify_ssl: bool = True
    trust_env: bool = False
    max_retries: int = _MAX_RETRIES
    retry_backoff_seconds: float = _RETRY_BACKOFF_SECONDS
    record_id: str = ""
    # 知识截止日期的可疑下限（年, 月）。旧于该值记 warning（疑似旧模型顶包）。
    min_knowledge_cutoff: tuple[int, int] = (2024, 1)
    # OCR 级的数字串；留空则运行时随机生成（防记忆）。测试可注入固定值。
    ocr_digits: str = ""


@dataclass
class ForensicsOutcome:
    result_dir: Path
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 传输层：三条探针共用一次封装
# ---------------------------------------------------------------------------


def probe_endpoint(protocol: str, base_url: str, model: str) -> str:
    """取证探针的非流式端点（Gemini 用 :generateContent，不是 SSE 端点）。"""
    root = api_root(protocol, base_url)
    family = protocol_spec(protocol).family
    if family == "openai" or protocol in OPENAI_COMPATIBLE_PROTOCOLS:
        return f"{root}/chat/completions"
    if family == "anthropic":
        return f"{root}/messages"
    if family == "gemini":
        return f"{root}/models/{quote(model, safe='')}:generateContent"
    raise ValueError(f"中转验真不支持协议: {protocol}")


def probe_headers(protocol: str, api_key: str, request_id: str) -> dict[str, str]:
    """非流式请求头：Accept 固定 application/json，鉴权按家族。"""
    family = protocol_spec(protocol).family
    common = {"Content-Type": "application/json", "Accept": "application/json",
              "X-Request-ID": request_id}
    if family == "anthropic":
        common.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
    elif family == "gemini":
        common["x-goog-api-key"] = api_key
    else:
        common["Authorization"] = f"Bearer {api_key}"
    return common


def text_payload(protocol: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    family = protocol_spec(protocol).family
    if family == "anthropic":
        return {"model": model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
    if family == "gemini":
        return {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens}}
    return {"model": model, "stream": False, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}


def image_payload(protocol: str, model: str, prompt: str, png: bytes, max_tokens: int) -> dict[str, Any]:
    """按协议家族构造带图请求体。图片统一 PNG / base64（内嵌生成，无外网）。"""
    family = protocol_spec(protocol).family
    encoded = base64.b64encode(png).decode("ascii")
    if family == "anthropic":
        content: list[dict[str, Any]] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}},
            {"type": "text", "text": prompt},
        ]
        return {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
    if family == "gemini":
        parts = [{"inline_data": {"mime_type": "image/png", "data": encoded}}, {"text": prompt}]
        return {"contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"maxOutputTokens": max_tokens}}
    url = "data:image/png;base64," + encoded
    content = [{"type": "text", "text": prompt},
               {"type": "image_url", "image_url": {"url": url}}]
    return {"model": model, "stream": False, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}]}


def response_text(protocol: str, body: dict[str, Any] | None) -> str:
    """从非流式响应体里取正文（兼容分块 content 与 Gemini parts）。"""
    if not isinstance(body, dict):
        return ""
    family = protocol_spec(protocol).family
    if family == "anthropic":
        blocks = body.get("content") or []
        return "".join(str(block.get("text") or "") for block in blocks if isinstance(block, dict))
    if family == "gemini":
        candidates = body.get("candidates") or []
        parts = ((candidates[0] or {}).get("content") or {}).get("parts") or [] if candidates else []
        return "".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))
    choices = body.get("choices") or []
    message = (choices[0] or {}).get("message") or {} if choices else {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(block.get("text") or "") for block in content if isinstance(block, dict))
    return ""


def error_envelope(protocol: str, body: dict[str, Any] | None, raw: str) -> dict[str, Any]:
    """结构化取错误信封（error.type / error.code / message 一起看，不短路）。"""
    envelope: dict[str, Any] = {"error_type": None, "error_code": None, "error_message": None}
    if not isinstance(body, dict):
        envelope["error_message"] = raw[:400]
        return envelope
    error = body.get("error")
    if isinstance(error, dict):
        envelope["error_type"] = error.get("type") or error.get("name")
        envelope["error_code"] = error.get("code")
        envelope["error_message"] = error.get("message")
    elif isinstance(error, str):
        envelope["error_message"] = error
    if envelope["error_message"] is None:
        envelope["error_message"] = raw[:400]
    return envelope


class Session:
    """取证探针的 HTTP 会话：统一重试与证据落盘，永不抛异常。"""

    def __init__(self, config: ForensicsConfig, audit: AuditLog, client: httpx.Client | None = None):
        self.config = config
        self.audit = audit
        self.seq = 0
        self.rows: list[dict[str, Any]] = []  # 全部 HTTP 证据行（成功与失败都在内）
        self._client = client
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def _client_or_new(self) -> httpx.Client:
        if self._client is None:
            timeout = httpx.Timeout(self.config.request_timeout_seconds,
                                    connect=min(30.0, self.config.request_timeout_seconds))
            self._client = httpx.Client(timeout=timeout, verify=self.config.verify_ssl,
                                        trust_env=self.config.trust_env)
        return self._client

    def post(self, *, case_id: str, probe: str, url: str, body: dict[str, Any] | None = None,
             raw_body: bytes | None = None, extra_headers: dict[str, str] | None = None) -> dict[str, Any]:
        config = self.config
        self.seq += 1
        request_id = f"{case_id}-{self.seq:02d}-{uuid.uuid4().hex[:8]}"
        headers = probe_headers(config.protocol, config.api_key, request_id)
        if extra_headers:
            headers.update(extra_headers)
        payload = body if body is not None else None
        row: dict[str, Any] = {
            "case_id": case_id, "probe": probe, "index": self.seq, "request_id": request_id,
            "protocol": config.protocol, "endpoint": url, "model": config.model,
            "started_at": datetime.now(timezone.utc).isoformat(), "started_epoch": time.time(),
            "completed_epoch": None, "http_status": None, "success": False,
            "transport_error": None, "error_type": None, "error_message": None,
            "attempts": 0, "elapsed_seconds": None,
            "timing": {"ttft_ms": None, "ttfc_ms": None, "e2e_ms": None},
            "request": body if body is not None else {"raw_body_bytes": len(raw_body or b"")},
            "response_body": None, "response_text": "", "response_headers": {},
        }
        started = time.perf_counter()
        attempts = 0
        client = self._client_or_new()
        while attempts <= max(0, config.max_retries):
            attempts += 1
            try:
                if payload is not None:
                    response = client.post(url, json=payload, headers=headers)
                else:
                    response = client.post(url, content=raw_body or b"", headers=headers)
                row["http_status"] = response.status_code
                row["response_headers"] = {
                    key.lower(): value for key, value in response.headers.items()
                    if key.lower() in {"content-type", "server", "via", "x-request-id", "request-id",
                                       "x-trace-id", "x-oneapi-request-id", "cf-ray"}
                }
                raw = response.text
                row["response_text"] = raw[:8000]
                try:
                    row["response_body"] = response.json()
                except ValueError:
                    row["response_body"] = None
                row["timing"]["e2e_ms"] = round((time.perf_counter() - started) * 1000, 3)
                if 200 <= response.status_code < 300:
                    row["success"] = True
                    break
                envelope = error_envelope(config.protocol, row["response_body"], raw)
                row["error_type"] = envelope["error_type"] or f"http_{response.status_code}"
                row["error_message"] = envelope["error_message"]
                if response.status_code < 500:
                    break  # 4xx 是上游明确拒绝，重试没有意义
            except httpx.TimeoutException as exc:
                row["transport_error"] = "timeout"
                row["error_message"] = str(exc)
            except httpx.HTTPError as exc:
                row["transport_error"] = "client_error"
                row["error_message"] = str(exc)
            except OSError as exc:
                row["transport_error"] = "connection_error"
                row["error_message"] = str(exc)
            if attempts <= config.max_retries:
                time.sleep(config.retry_backoff_seconds * attempts)
            else:
                break
        row["attempts"] = attempts
        row["completed_epoch"] = time.time()
        row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        row["transport_status"] = "completed"
        self.rows.append(row)
        self.audit.write(redact(row))
        return row


# ---------------------------------------------------------------------------
# 用例一：隐式身份指纹
# ---------------------------------------------------------------------------

_CODE_PROMPT = (
    "Write a complete Python script that computes the number of days between two dates. "
    "Requirements: (1) start the file with a comment block that MUST contain exactly four "
    "lines: Tool Name, Generated, Model, Author; (2) put the core logic in a function with a "
    "docstring; (3) output the code only, no explanation."
)
_CONTEXT_PROMPT = (
    "What is your maximum context window size in tokens? Reply with just a single integer "
    "(no commas, no units, no explanation), e.g. 200000."
)
_CUTOFF_PROMPT = (
    "What is your knowledge cutoff date? Reply with just the month and year, "
    "e.g. 'March 2024'. No additional explanation."
)

# 族关键词 → 规范族名。声明族与投票族都走同一张表，避免"两把尺子"。
FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "claude": ("claude", "anthropic"),
    "openai": ("gpt-", "gpt ", "openai", "chatgpt"),
    "gemini": ("gemini", "google"),
    "deepseek": ("deepseek", "深度求索"),
    "glm": ("glm", "zhipu", "智谱", "chatglm"),
    "kimi": ("kimi", "moonshot", "月之暗面"),
    "qwen": ("qwen", "通义", "alibaba"),
    "ernie": ("ernie", "文心"),
}
# 档位词：只用于"自述档位 vs 声明档位"的留痕比对，不参与族投票。
TIER_TOKENS: tuple[str, ...] = ("opus", "sonnet", "haiku", "pro", "flash", "mini", "nano", "lite", "turbo")
# 各族上下文窗口的粗区间：只用于族投票（在区间内 = 给声明族投一票），不做档位判定。
CONTEXT_EXPECT: dict[str, tuple[int, int]] = {
    "claude": (180_000, 1_100_000),
    "openai": (100_000, 2_100_000),
    "gemini": (900_000, 10_500_000),
    "deepseek": (60_000, 200_000),
    "glm": (100_000, 300_000),
    "kimi": (200_000, 300_000),
    "qwen": (30_000, 10_500_000),
}
_MONTHS = ("january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december")


def declared_family(model: str) -> str | None:
    lowered = (model or "").lower()
    for family, keywords in FAMILY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return family
    return None


def _family_vote(text: str) -> str | None:
    lowered = (text or "").lower()
    hits = [family for family, keywords in FAMILY_KEYWORDS.items()
            if any(keyword in lowered for keyword in keywords)]
    return hits[0] if len(hits) == 1 else (None if not hits else "ambiguous")


def _tier_tokens(model: str) -> set[str]:
    lowered = (model or "").lower()
    return {token for token in TIER_TOKENS if re.search(rf"(?<![a-z]){token}(?![a-z])", lowered)}


def _parse_context(text: str) -> int | None:
    match = re.search(r"\d[\d,]*", text or "")
    if not match:
        return None
    value = int(match.group(0).replace(",", ""))
    return value if 1_000 <= value <= 100_000_000 else None


def _parse_cutoff(text: str) -> tuple[int, int] | None:
    """解析知识截止自述：支持 "March 2024" / "2024-03" / "2024年3月"。"""
    lowered = (text or "").strip().lower()
    for index, month in enumerate(_MONTHS, start=1):
        if month in lowered:
            year_match = re.search(r"(19|20)\d{2}", lowered)
            if year_match:
                return int(year_match.group(0)), index
            return None
    match = re.search(r"((?:19|20)\d{2})[-/年](\d{1,2})", lowered)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            return year, month
    return None


def run_implicit_identity(config: ForensicsConfig, session: Session) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """三条自述指纹 → 族投票 → 与声明族比对。返回（结果行, 摘要附加信息）。"""
    case = CASES[IMPLICIT_IDENTITY]
    url = probe_endpoint(config.protocol, config.base_url, config.model)
    declared = declared_family(config.model)
    votes: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    # —— 子探针 1：代码签名（Model / Author 自述行）——
    row = session.post(case_id=IMPLICIT_IDENTITY, probe="code_signature", url=url,
                       body=text_payload(config.protocol, config.model, _CODE_PROMPT, _IDENTITY_MAX_TOKENS))
    text = response_text(config.protocol, row["response_body"])
    code_block = text
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    if fence:
        code_block = fence.group(1)
    model_line = re.search(r"(?:model|模型)\s*[:：]\s*([^\n]+)", code_block, re.I)
    author_line = re.search(r"(?:author|作者)\s*[:：]\s*([^\n]+)", code_block, re.I)
    header_text = " ".join(part.group(1).strip() for part in (model_line, author_line) if part)
    vote = _family_vote(header_text) if header_text else None
    model_line_text = model_line.group(1).strip() if model_line else ""
    if row["success"]:
        if vote in (None, "ambiguous"):
            status, passed = "warning", False
            detail = ("代码头注释未给出可判定的模型自述" if not header_text
                      else f"自述同时命中多个族关键词（{header_text[:120]}），无法归一")
        else:
            status, passed = "passed", True
            detail = f"代码头注释自述族 = {vote}（Model 行：{model_line_text[:80] or '—'}）"
    else:
        status, passed = _blocked_or_warning(row)
        detail = f"HTTP {row['http_status']}：{str(row['error_message'] or '')[:160]}"
    if vote and vote != "ambiguous":
        votes.append({"family": vote, "source": "code_signature"})
    rows.append({"case_id": IMPLICIT_IDENTITY, "name": f"{case.title} · 代码签名自述",
                 "step": "code_signature", "status": status, "passed": passed,
                 "duration_seconds": row["elapsed_seconds"], "details": detail,
                 "observations": {"model_line": model_line_text, "author_line": author_line.group(1).strip() if author_line else "",
                                  "http_status": row["http_status"]}})

    # —— 子探针 2：上下文窗口自述 ——
    row = session.post(case_id=IMPLICIT_IDENTITY, probe="context_window", url=url,
                       body=text_payload(config.protocol, config.model, _CONTEXT_PROMPT, 512))
    text = response_text(config.protocol, row["response_body"])
    context = _parse_context(text)
    expect = CONTEXT_EXPECT.get(declared) if declared else None
    if row["success"]:
        if context is None:
            status, passed = "warning", False
            detail = f"未解析出上下文数字（原文：{text[:80]!r}）"
        elif expect and not expect[0] <= context <= expect[1]:
            # 不给声明族投票；是否构成矛盾交给族投票汇总（单条不自证）。
            status, passed = "warning", False
            detail = f"自述上下文 {context} 超出声明族区间 {expect}（留痕，不作单条结论）"
        else:
            status, passed = "passed", True
            detail = (f"自述上下文 {context} 落在声明族（{declared or '未知'}）区间内"
                      if expect else f"自述上下文 {context}（声明族未知，仅留痕）")
        if context is not None and expect and expect[0] <= context <= expect[1] and declared:
            votes.append({"family": declared, "source": "context_window"})
    else:
        status, passed = _blocked_or_warning(row)
        detail = f"HTTP {row['http_status']}：{str(row['error_message'] or '')[:160]}"
    rows.append({"case_id": IMPLICIT_IDENTITY, "name": f"{case.title} · 上下文窗口自述",
                 "step": "context_window", "status": status, "passed": passed,
                 "duration_seconds": row["elapsed_seconds"], "details": detail,
                 "observations": {"context_tokens": context, "declared_family": declared,
                                  "expected_range": list(expect) if expect else None,
                                  "http_status": row["http_status"]}})

    # —— 子探针 3：知识截止日期 ——
    row = session.post(case_id=IMPLICIT_IDENTITY, probe="knowledge_cutoff", url=url,
                       body=text_payload(config.protocol, config.model, _CUTOFF_PROMPT, 512))
    text = response_text(config.protocol, row["response_body"])
    cutoff = _parse_cutoff(text)
    floor = config.min_knowledge_cutoff
    if row["success"]:
        if cutoff is None:
            status, passed = "warning", False
            detail = f"未解析出截止日期（原文：{text[:80]!r}）"
        elif cutoff < floor:
            status, passed = "warning", False
            detail = (f"自述截止 {cutoff[0]}-{cutoff[1]:02d} 早于可疑下限 "
                      f"{floor[0]}-{floor[1]:02d}（疑似旧模型顶包，留痕）")
        else:
            status, passed = "passed", True
            detail = f"自述截止 {cutoff[0]}-{cutoff[1]:02d}，不早于下限 {floor[0]}-{floor[1]:02d}"
    else:
        status, passed = _blocked_or_warning(row)
        detail = f"HTTP {row['http_status']}：{str(row['error_message'] or '')[:160]}"
    rows.append({"case_id": IMPLICIT_IDENTITY, "name": f"{case.title} · 知识截止自述",
                 "step": "knowledge_cutoff", "status": status, "passed": passed,
                 "duration_seconds": row["elapsed_seconds"], "details": detail,
                 "observations": {"cutoff": list(cutoff) if cutoff else None,
                                  "floor": list(floor), "http_status": row["http_status"]}})

    # —— 汇总：族投票 vs 声明族 ——
    # 说明：当前只有 code_signature 会投出"非声明族"的一票（context/cutoff 都是族中立
    # 信号），所以 contradicting 实际只能为 0/1；>=2 的分支是给后续新增投票源
    # （如侧信道自述、风格指纹）预留的信任级口径，别删。
    contradicting = [vote for vote in votes if vote["family"] != declared]
    warned = any(row["status"] == "warning" for row in rows)
    extras: dict[str, Any] = {"declared_family": declared, "family_votes": votes,
                              "contradicting_votes": contradicting}
    if all(row["status"] == "blocked" for row in rows):
        case_status, admission = "blocked", True
        assessment = "三条指纹均未能取证（账号/端点/瞬时故障），无法给出身份结论"
    elif len(contradicting) >= 2:
        case_status, admission = "failed", False
        assessment = (f"信任级：{len(contradicting)} 条独立指纹与声明族 {declared or '未知'} 冲突"
                      f"（{[vote['source'] for vote in contradicting]}），疑似模型替换")
    elif len(contradicting) == 1:
        case_status, admission = "warning", False
        assessment = (f"单条指纹（{contradicting[0]['source']}）与声明族冲突：自述 {contradicting[0]['family']}"
                      f" vs 声明 {declared}；启发式信号，下调准入但不判模型失败")
    elif warned:
        case_status, admission = "warning", False
        assessment = "存在未能归一的指纹（自述缺失/超区间/截止过旧），下调准入但不判模型失败"
    else:
        case_status, admission = "passed", True
        assessment = (f"族投票一致（{[vote['family'] for vote in votes]}），与声明族相符"
                      if votes else "未取得可归一的族自述，未见矛盾信号")
    # 档位自述 vs 声明档位：只留痕。模型自述版本长期不可靠（官方也常自称旧版本），不据此下结论。
    if model_line_text:
        declared_tiers = _tier_tokens(config.model)
        stated_tiers = _tier_tokens(model_line_text)
        mismatch = bool(declared_tiers and stated_tiers and not (declared_tiers & stated_tiers))
        extras["tier_self_report"] = {"declared_tiers": sorted(declared_tiers), "stated_tiers": sorted(stated_tiers),
                                      "mismatch": mismatch}
        if mismatch:
            assessment += (f"；档位自述不符（自述 {'/'.join(sorted(stated_tiers))} vs 声明 "
                           f"{'/'.join(sorted(declared_tiers))}）——自述不可靠，仅留痕")
    return rows, {**extras, "case_status": case_status, "admission": admission, "assessment": assessment}


# ---------------------------------------------------------------------------
# 用例二：图像内容理解
# ---------------------------------------------------------------------------

_COLOR_ALIASES: dict[str, tuple[str, ...]] = {
    "yellow": ("yellow", "黄"),
    "red": ("red", "红"),
    "green": ("green", "绿"),
    "blue": ("blue", "蓝"),
    "black": ("black", "黑"),
    "white": ("white", "白"),
}


def _match_color(answer: str, expected: str) -> bool:
    lowered = (answer or "").lower()
    return any(alias in lowered for alias in _COLOR_ALIASES[expected])


def _match_digits(answer: str, expected: str) -> bool:
    digits = re.sub(r"\D", "", answer or "")
    return digits == expected


def run_image_understanding(config: ForensicsConfig, session: Session) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """五级看图答题：夹具确定，判据确定；OCR 数字随机防记忆。"""
    case = CASES[IMAGE_UNDERSTANDING]
    url = probe_endpoint(config.protocol, config.base_url, config.model)
    digits = config.ocr_digits or str(random.randrange(10000, 100000))
    levels: list[tuple[str, str, bytes, str, Any]] = [
        ("solid_color", "这张图片是什么颜色？只回答颜色名（英文或中文即可）。",
         png_solid(YELLOW), "yellow", _match_color),
        ("chessboard", "这是一张棋盘图。被深色（黑色）方块覆盖的格子一共有多少个？只回答阿拉伯数字。",
         png_chessboard(), "32", lambda answer, expect: _match_digits(answer, expect)),
        ("ocr_digits", "读出图片中的数字，只输出数字本身，不要解释。",
         png_text(digits, scale=6), digits, lambda answer, expect: _match_digits(answer, expect)),
        ("spatial", "图片左半部分和右半部分是两种不同的颜色。右半部分是什么颜色？只回答颜色名。",
         png_halves(GREEN, RED), "red", _match_color),
        ("text_conflict", "忽略图片上的文字内容，这张图片的背景底色是什么颜色？只回答颜色名。",
         png_text("GREEN", fg=BLUE, bg=YELLOW, scale=6), "yellow", _match_color),
    ]
    rows: list[dict[str, Any]] = []
    answered = 0
    correct = 0
    for index, (step, prompt, png, expected, matcher) in enumerate(levels, start=1):
        row = session.post(case_id=IMAGE_UNDERSTANDING, probe=step, url=url,
                           body=image_payload(config.protocol, config.model, prompt, png, _VISION_MAX_TOKENS))
        answer = response_text(config.protocol, row["response_body"]).strip()
        if row["success"]:
            answered += 1
            ok = matcher(answer, expected)
            correct += 1 if ok else 0
            status, passed = ("passed", True) if ok else ("failed", False)
            detail = f"回答 {answer[:60]!r}，期望 {expected!r}"
        else:
            unsupported = any(marker in str(row["error_message"] or "").lower()
                              for marker in _VISION_UNSUPPORTED_MARKERS)
            if row["http_status"] in _BLOCKED_STATUSES or row["transport_error"] or unsupported:
                status, passed = "blocked", False
                detail = f"HTTP {row['http_status']}：视觉输入未被接受（{str(row['error_message'] or '')[:140]}）"
            else:
                status, passed = "failed", False
                detail = f"HTTP {row['http_status']}：{str(row['error_message'] or '')[:160]}"
        rows.append({"case_id": IMAGE_UNDERSTANDING,
                     "name": f"{case.title} · L{index} {step}",
                     "step": step, "status": status, "passed": passed,
                     "duration_seconds": row["elapsed_seconds"], "details": detail,
                     "observations": {"answer": answer[:200], "expected": str(expected),
                                      "http_status": row["http_status"]}})
    extras: dict[str, Any] = {"levels_total": len(levels), "levels_answered": answered,
                              "levels_correct": correct, "ocr_digits": digits}
    if answered == 0:
        case_status, admission = "blocked", True
        assessment = "五级均未取得有效回答（视觉输入未被接受或链路故障），无法评估图像理解能力"
    else:
        rate = correct / answered
        extras["accuracy"] = rate
        if correct == answered:
            case_status, admission = "passed", True
            assessment = f"五级全部答对（{correct}/{answered}）"
        elif rate >= 0.6:
            case_status, admission = "warning", True
            assessment = f"部分答对（{correct}/{answered}），图像理解能力打折但不否决"
        else:
            case_status, admission = "failed", False
            assessment = f"答对率过低（{correct}/{answered}），图像内容理解不达标"
    return rows, {**extras, "case_status": case_status, "admission": admission, "assessment": assessment}


# ---------------------------------------------------------------------------
# 用例三：错误信封泄露
# ---------------------------------------------------------------------------

_HIGH_PATTERNS: tuple[tuple[str, str], ...] = (
    ("secret_key", r"sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|bce-v3/ALTAK[A-Za-z0-9\-_/]*"),
    ("stack_trace", r"Traceback \(most recent call last\)|goroutine \d+|panic: |Exception in thread"),
    ("internal_path", r"/(?:home|root|app|srv|var/www|Users)/[A-Za-z0-9_.\-]{2,}|[A-Za-z]:\\\\[^\"]{3,}"),
    ("source_ref", r"\.(?:py|go|java|js|ts):\d{1,5}"),
    ("private_network", r"\b(?:10|127)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\b192\.168\.\d{1,3}\.\d{1,3}\b"
                        r"|\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"),
    ("sql_error", r"SQLSTATE|ORA-\d{5}|psycopg|pymysql|pq: |SQL syntax|duplicate key value"),
    ("db_dsn", r"(?:postgres(?:ql)?|mysql|redis|mongodb)://\S+"),
)
_MEDIUM_PATTERNS: tuple[tuple[str, str], ...] = (
    ("channel_topology", r"[Nn]o available (?:channel|accounts)|无可用渠道|渠道(?:名|分组)|under group [^\"]{1,40}|channel[#:\s]*\d+"),
    ("internal_hostname", r"\.internal\b|\.local(?:host)?[:：]\d+|localhost:\d+"),
)
_LOW_PATTERNS: tuple[tuple[str, str], ...] = (
    ("relay_software", r"new_api_error|one[_-]api"),
)


def scan_leakage(text: str) -> list[dict[str, str]]:
    """按高/中/低三级扫描错误信封里的内部信息。低危只留痕，不计入裁定。"""
    findings: list[dict[str, str]] = []
    for severity, patterns in (("high", _HIGH_PATTERNS), ("medium", _MEDIUM_PATTERNS), ("low", _LOW_PATTERNS)):
        for name, pattern in patterns:
            match = re.search(pattern, text or "")
            if match:
                findings.append({"severity": severity, "pattern": name,
                                 "evidence": match.group(0)[:120]})
    return findings


def run_error_envelope_leakage(config: ForensicsConfig, session: Session) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """畸形 JSON / 不存在的模型 / 不存在的路径三类 4xx 的信封取证。"""
    case = CASES[ERROR_ENVELOPE_LEAKAGE]
    url = probe_endpoint(config.protocol, config.base_url, config.model)
    root = api_root(config.protocol, config.base_url)
    nonce = uuid.uuid4().hex[:8]
    sub_probes: list[tuple[str, str, str, dict[str, Any] | None, bytes | None]] = [
        ("malformed_json", "畸形 JSON 请求体", url, None,
         b'{"model": "' + nonce.encode() + b'", "messages": ['),
        ("nonexistent_model", "不存在的模型",
         url, text_payload(config.protocol, f"nonexistent-model-{nonce}", "只回复 OK", 16), None),
        ("wrong_path", "不存在的路径", f"{root}/forensics-probe-{nonce}",
         text_payload(config.protocol, config.model, "只回复 OK", 16), None),
    ]
    rows: list[dict[str, Any]] = []
    findings: list[dict[str, str]] = []
    silent_fallback = False
    lenient_parse = False
    # 账号/限额类 4xx = 平台前提不满足（探针根本没到"错误信封取证"这一步），记 blocked。
    _account_statuses = {401, 402, 403, 429}
    for step, name, target, body, raw in sub_probes:
        row = session.post(case_id=ERROR_ENVELOPE_LEAKAGE, probe=step, url=target,
                           body=body, raw_body=raw)
        status_code = row["http_status"]
        step_findings = scan_leakage(row["response_text"])
        findings.extend({"source": step, **finding} for finding in step_findings)
        high = [f for f in step_findings if f["severity"] == "high"]
        medium = [f for f in step_findings if f["severity"] == "medium"]
        extra_note = ""
        if row["success"]:
            # 2xx：错误输入被"成功"处理。malformed/wrong_path = 解析过宽；
            # nonexistent_model = 疑似静默回退/替换（更强的信号）。
            if step == "nonexistent_model":
                silent_fallback = True
                extra_note = "；无通道模型返回了 2xx——疑似静默回退/替换，需人工核验"
            else:
                lenient_parse = True
                extra_note = "；错误输入得到 2xx——网关解析过宽"
            status, passed = "warning", False
        elif row["transport_error"] or (isinstance(status_code, int) and status_code >= 500) \
                or status_code in _account_statuses:
            status, passed = "blocked", False
        elif step == "wrong_path" and status_code == 404:
            status, passed = "passed", True  # 404 正是错误路径的应有行为
        elif high:
            status, passed = "failed", False
        elif medium:
            status, passed = "warning", False  # 中危：单条信封已外泄渠道/拓扑线索
        else:
            status, passed = "passed", True
        summary_bits = [f"HTTP {status_code}"]
        if high:
            summary_bits.append("高危泄露：" + "、".join(sorted({f['pattern'] for f in high})))
        if medium:
            summary_bits.append("中危泄露：" + "、".join(sorted({f['pattern'] for f in medium})))
        if not step_findings and not row["success"]:
            summary_bits.append("未发现内部信息外泄")
        envelope = error_envelope(config.protocol, row["response_body"], row["response_text"])
        rows.append({"case_id": ERROR_ENVELOPE_LEAKAGE, "name": f"{case.title} · {name}",
                     "step": step, "status": status, "passed": passed,
                     "duration_seconds": row["elapsed_seconds"],
                     "details": "，".join(summary_bits) + extra_note,
                     "observations": {"findings": step_findings, "error_type": envelope["error_type"],
                                      "error_code": envelope["error_code"],
                                      "response_preview": row["response_text"][:300],
                                      "http_status": status_code}})
    extras: dict[str, Any] = {"findings": findings, "silent_fallback": silent_fallback,
                              "lenient_parse": lenient_parse}
    if all(row["status"] == "blocked" for row in rows):
        case_status, admission = "blocked", True
        assessment = "三个错误探针均未取得有效信封（账号/端点/瞬时故障），无法评估泄露面"
    elif any(f.get("severity") == "high" for f in findings):
        case_status, admission = "failed", False
        assessment = ("信任级：错误信封外泄高危内部信息（"
                      + "、".join(sorted({f['pattern'] for f in findings if f['severity'] == 'high'})) + "）")
    elif any(f.get("severity") == "medium" for f in findings):
        case_status, admission = "warning", True
        assessment = ("中危：错误信封外泄渠道/拓扑线索（"
                      + "、".join(sorted({f['pattern'] for f in findings if f['severity'] == 'medium'}))
                      + "）；下调观感但不判模型失败")
    else:
        case_status, admission = "passed", True
        assessment = "三类错误信封均未发现内部信息外泄"
    if silent_fallback:
        case_status = "warning" if case_status == "passed" else case_status
        admission = False
        assessment += "；无通道模型被静默放行（疑似回退/替换），不准入"
    return rows, {**extras, "case_status": case_status, "admission": admission, "assessment": assessment}


# ---------------------------------------------------------------------------
# 装配与收尾
# ---------------------------------------------------------------------------


def _blocked_or_warning(row: dict[str, Any]) -> tuple[str, bool]:
    """HTTP 失败行的兜底裁定：账号/瞬时/不支持 → blocked；其余 4xx → warning。"""
    status = row.get("http_status")
    if row.get("transport_error") or (status in _BLOCKED_STATUSES) or (isinstance(status, int) and status >= 500):
        return "blocked", False
    return "warning", False


_RUNNERS = {
    IMPLICIT_IDENTITY: run_implicit_identity,
    IMAGE_UNDERSTANDING: run_image_understanding,
    ERROR_ENVELOPE_LEAKAGE: run_error_envelope_leakage,
}


def _build_report(case: ForensicsCase, rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    counts = {key: sum(1 for row in rows if row["status"] == key)
              for key in ("passed", "failed", "warning", "blocked")}
    table = case_table_report(
        f"{case.title}（{summary['model']} · {summary['protocol']}）",
        total=len(rows),
        passed=counts["passed"],
        failed=counts["failed"],
        warning=counts["warning"],
        blocked=counts["blocked"],
        pass_rate=(counts["passed"] / len(rows)) if rows else None,
        columns=("子探针", "结果", "耗时(s)", "说明"),
        status_column=1,
        rows=[[row["name"], row["status"], f"{row['duration_seconds']:.2f}", row["details"]] for row in rows],
        trailing=["", f"- 本例结论：{summary.get('assessment') or '—'}",
                  f"- 准入：{'通过' if summary.get('admission_passed') else '不通过'}",
                  "", "> 判定口径：信任级矛盾才 failed；启发式信号 warning 不判模型失败；"
                  "账号/瞬时故障/不支持视觉输入记 blocked 只记录不阻断。"],
    )
    return table


def run(config: ForensicsConfig, case_id: str, output_root: Path,
        client: httpx.Client | None = None) -> ForensicsOutcome:
    """跑一个取证用例并落盘（不含收尾；收尾由 ``common.finalizer`` 统一执行）。"""
    case = CASES[case_id]
    runner = _RUNNERS[case_id]
    result_dir = output_root / f"{case_id}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    result_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    audit = AuditLog(result_dir / "requests.jsonl")
    try:
        session = Session(config, audit, client=client)
        try:
            rows, extras = runner(config, session)
            http_rows = list(session.rows)
        finally:
            session.close()
    finally:
        audit.close()
    statuses = [row["status"] for row in rows]
    counts = {key: statuses.count(key) for key in ("passed", "failed", "warning", "blocked")}
    elapsed = round(time.time() - started, 3)
    ok = sum(1 for row in http_rows if row["success"])
    summary: dict[str, Any] = {
        "schema_version": "mltest.forensics.v1",
        "suite": "forensics",
        "test_kind": "forensics",
        "case": case_id,
        "record_id": config.record_id or None,
        "protocol": config.protocol,
        "base_url": config.base_url,
        "model": config.model,
        "endpoint": probe_endpoint(config.protocol, config.base_url, config.model),
        "requested": case.requests,
        "sent": len(rows),
        "completed": len(rows),
        "successful": ok,
        "success_rate": (ok / len(rows)) if rows else 0.0,
        "elapsed_seconds": elapsed,
        "total": len(rows),
        "passed": counts["passed"],
        "failed": counts["failed"],
        "warning": counts["warning"],
        "blocked": counts["blocked"],
        "results": rows,
        "forensics": extras,
        "assessment": extras.get("assessment"),
        "checks": {
            "forensics_verdict": {
                "value": extras.get("case_status"),
                "expected": "无信任级矛盾",
                "passed": extras.get("admission", False),
                "detail": extras.get("assessment"),
            },
        },
        "admission_passed": bool(extras.get("admission")),
        "scope_note": case.admission,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(redact(summary), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (result_dir / "report.md").write_text(_build_report(case, rows, summary), encoding="utf-8")
    return ForensicsOutcome(result_dir=result_dir, summary=summary)
