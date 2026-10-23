"""DeepSeek V4 Pro/Flash 专项兼容性测试。

测试依据 DeepSeek V4 官方 API 文档，使用原始 HTTP 请求，避免 SDK 对扩展字段的过滤。
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import time
import uuid
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
from common.png_fixture import (  # noqa: F401  （再导出，保持 dsv4.core.png_bands 等旧引用可用）
    IMAGE_BOTTOM_RGB,
    IMAGE_TOP_RGB,
    PNG_SIGNATURE,
    image_part,
    png_bands,
    png_data_url,
)
from common.test_framework import HttpExecutor, ResultLog
from common.probe_assertions import arithmetic_function, integer_answer
from .gateway_contract import (
    CAPABILITY_VISION,
    GatewayIssue,
    classify_gateway_issue,
    model_supports_capability,
)

# 官方文档当前主推的模型名（Models & Pricing / Vision / Files API 一致）。
# deepseek-v4-flash 与 deepseek-v4-flash-vision-exp 已退休但仍被接受，请求由最新 Flash 模型服务。
PRIMARY_MODELS = ("deepseek-v4-pro", "deepseek-flash")
RETIRED_ALIASES = ("deepseek-v4-flash",)
MODELS = PRIMARY_MODELS

# 官方文档：Beta 与 Anthropic 兼容端点挂在**根域**下，不带版本后缀。
#   beta      -> https://api.deepseek.com/beta      （chat prefix / FIM / strict tool）
#   anthropic -> https://api.deepseek.com/anthropic （Anthropic 兼容，端点 {base}/v1/messages）
# 而 Web（server.py）与 CLI 传入的 base_url 已被 api_root() 规范化成 ".../v1"，
# 若直接拼接会得到 ".../v1/beta/chat/completions" 这类不存在的路径（线上 404）。
# 因此派生前先剥掉已知的版本后缀。
_API_VERSION_SUFFIXES = ("/api/v3", "/v1beta", "/v1")


def _api_parent(base_url: str) -> str:
    """返回 API 根的父地址，供 beta/anthropic 端点派生使用。"""
    base = base_url.rstrip("/")
    for suffix in _API_VERSION_SUFFIXES:
        if base.endswith(suffix):
            return base[: -len(suffix)].rstrip("/")
    return base


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    beta_base_url: str | None = None
    anthropic_base_url: str | None = None
    timeout: float = 300.0
    verify_ssl: bool = True
    trust_env: bool = False

    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        key = os.getenv("DS_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("缺少 DS_API_KEY（或 DEEPSEEK_API_KEY）")
        base = (os.getenv("DS_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        return cls(
            api_key=key,
            base_url=base,
            beta_base_url=os.getenv("DS_BETA_BASE_URL"),
            anthropic_base_url=os.getenv("DS_ANTHROPIC_BASE_URL"),
            timeout=float(os.getenv("DS_REQUEST_TIMEOUT_SECONDS", "300")),
            verify_ssl=os.getenv("DS_VERIFY_SSL", "true").lower() not in {"0", "false", "no"},
            trust_env=os.getenv("DS_TRUST_ENV", "false").lower() in {"1", "true", "yes"},
        )

    def url(self, path: str, family: str = "openai") -> str:
        base = self.base_url.rstrip("/")
        if family == "beta":
            root = (self.beta_base_url or f"{_api_parent(base)}/beta").rstrip("/")
        elif family == "anthropic":
            root = (self.anthropic_base_url or f"{_api_parent(base)}/anthropic").rstrip("/")
        elif family == "root":
            # 官方文档：GET /models、/user/balance、POST /files 均挂在根域下，不带版本后缀。
            root = _api_parent(base)
        else:
            root = base
        return f"{root}/{path.lstrip('/')}"


@dataclass
class ApiResponse:
    status: int
    body: dict[str, Any] | None
    text: str
    elapsed: float
    headers: dict[str, str] = field(default_factory=dict)
    upstream_request_id: str | None = None
    stream: dict[str, Any] | None = None


@dataclass
class CaseResult:
    case_id: str
    name: str
    model: str
    passed: bool
    status: str = "passed"
    details: str = ""
    duration_seconds: float = 0.0
    observations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    name: str
    priority: str
    handler: str
    beta: bool = False
    destructive: bool = False


CASE_SPECS = (
    CaseSpec("DS-V4-01", "模型别名可调用", "P0", "basic"),
    CaseSpec("DS-V4-02", "默认思考模式", "P0", "default_thinking"),
    CaseSpec("DS-V4-03", "关闭思考模式", "P0", "disabled_thinking"),
    CaseSpec("DS-V4-04", "思考强度映射", "P0", "effort_mapping"),
    CaseSpec("DS-V4-05", "思考模式兼容参数静默接受", "P0", "ignored_sampling"),
    CaseSpec("DS-V4-06", "流式思考与正文分离", "P0", "stream_thinking"),
    CaseSpec("DS-V4-07", "无工具多轮回传思考", "P0", "replay_without_tools"),
    CaseSpec("DS-V4-08", "思考模式工具调用完整回传", "P0", "tool_replay"),
    CaseSpec("DS-V4-09", "工具调用漏传 reasoning_content 的校验行为", "P0", "tool_missing_reasoning"),
    CaseSpec("DS-V4-10", "未进入工具链时 reasoning 回传行为", "P0", "tools_plain_turn_replay"),
    CaseSpec("DS-V4-11", "JSON Output", "P0", "json_output"),
    CaseSpec("DS-V4-12", "JSON Prompt 缺少关键字时可控失败", "P0", "json_missing_keyword"),
    CaseSpec("DS-V4-13", "JSON 小输出截断可检测", "P0", "json_truncation"),
    CaseSpec("DS-V4-14", "自动 Context Cache 命中", "P0", "cache_hit"),
    CaseSpec("DS-V4-15", "Context Cache 完整前缀规则", "P0", "cache_prefix_rule"),
    CaseSpec("DS-V4-16", "user_id 缓存隔离", "P0", "user_isolation"),
    CaseSpec("DS-V4-17", "非法 user_id 线上行为观察", "P0", "invalid_user_id"),
    CaseSpec("DS-V4-18", "同身份多轮 Session 连续性与缓存命中", "P0", "session_continuity_cache"),
    CaseSpec("DS-V4-19", "Strict Tool Call", "P1", "strict_tool", beta=True),
    CaseSpec("DS-V4-20", "Strict 非法 Schema 拒绝", "P1", "strict_invalid", beta=True),
    CaseSpec("DS-V4-21", "Chat Prefix Completion", "P1", "prefix", beta=True),
    CaseSpec("DS-V4-22", "Chat Prefix 非法放置的处理行为", "P1", "prefix_invalid", beta=True),
    CaseSpec("DS-V4-23", "FIM Completion", "P1", "fim", beta=True),
    CaseSpec("DS-V4-24", "Anthropic 模型映射", "P1", "anthropic_mapping"),
    CaseSpec("DS-V4-25", "Responses API 无状态字段", "P1", "responses_stateless"),
    CaseSpec("DS-V4-26", "非流式 Usage 完整性", "P2", "usage_nonstream"),
    CaseSpec("DS-V4-27", "流式 Usage 与结束事件", "P2", "usage_stream"),
    CaseSpec("DS-V4-28", "Stop 停止序列", "P2", "stop_sequence"),
    CaseSpec("DS-V4-29", "Temperature 合法边界", "P2", "temperature_bounds"),
    CaseSpec("DS-V4-30", "Temperature 越界拒绝", "P2", "temperature_invalid"),
    CaseSpec("DS-V4-31", "非法 Messages 拒绝", "P2", "messages_invalid"),
    CaseSpec("DS-V4-32", "Tool Choice none", "P2", "tool_choice_none"),
    CaseSpec("DS-V4-33", "Tool Choice required", "P2", "tool_choice_required"),
    CaseSpec("DS-V4-34", "Responses API 基础响应", "P2", "responses_basic"),
    CaseSpec("DS-V4-35", "Anthropic API 基础响应", "P2", "anthropic_basic"),
    # ---- 2026-09-24 覆盖度评审补充：多模态、端点面、参数面、错误契约、跨协议矩阵 ----
    CaseSpec("DS-V4-36", "图片输入识别（内联 base64）", "P0", "image_input"),
    CaseSpec("DS-V4-37", "模型别名口径与退休别名路由", "P0", "model_alias"),
    CaseSpec("DS-V4-38", "思考强度取值域与关闭语义", "P0", "effort_full_set"),
    CaseSpec("DS-V4-39", "max_tokens 上限与越界拒绝", "P0", "max_tokens_bounds"),
    CaseSpec("DS-V4-40", "无效 API Key 鉴权契约", "P0", "invalid_api_key"),
    CaseSpec("DS-V4-41", "Files API 上传与 file_id 引用闭环", "P0", "files_lifecycle", destructive=True),
    CaseSpec("DS-V4-42", "图片出现在非 user 消息应被拒", "P1", "image_wrong_role"),
    CaseSpec("DS-V4-43", "模型列表端点与声明一致性", "P1", "models_listing"),
    CaseSpec("DS-V4-44", "stream_options 缺少 stream 的契约", "P1", "stream_options_without_stream"),
    CaseSpec("DS-V4-45", "logprobs / top_logprobs 结构", "P1", "logprobs_structure"),
    CaseSpec("DS-V4-46", "top_p 边界与思考模式钳制", "P1", "top_p_bounds"),
    CaseSpec("DS-V4-47", "FIM 与思考模式互斥", "P1", "fim_thinking_conflict"),
    CaseSpec("DS-V4-48", "Anthropic system 与 stop_sequences", "P1", "anthropic_system_stop"),
    CaseSpec("DS-V4-49", "Anthropic thinking 与 output_config", "P1", "anthropic_thinking"),
    CaseSpec("DS-V4-50", "Responses API 流式语义事件", "P1", "responses_stream"),
    CaseSpec("DS-V4-51", "Responses 不支持字段静默忽略", "P1", "responses_ignored_fields"),
    CaseSpec("DS-V4-52", "并行工具调用观测", "P1", "parallel_tool_calls"),
    CaseSpec("DS-V4-53", "图片 detail 档位", "P2", "image_detail_levels"),
    CaseSpec("DS-V4-54", "Files API 列表与检索结构", "P2", "files_listing"),
    CaseSpec("DS-V4-55", "账户余额端点", "P2", "user_balance"),
    CaseSpec("DS-V4-56", "stop 多序列与上限边界", "P2", "stop_multi_sequences"),
    CaseSpec("DS-V4-57", "废弃采样参数静默忽略", "P2", "deprecated_penalties"),
    CaseSpec("DS-V4-58", "FIM 与 JSON mode 互斥", "P2", "fim_json_conflict"),
    CaseSpec("DS-V4-59", "prefix 缺少 prefix 标志的契约", "P2", "prefix_without_flag", beta=True),
    CaseSpec("DS-V4-60", "Anthropic tool_choice 与工具闭环", "P2", "anthropic_tool_loop"),
    CaseSpec("DS-V4-61", "Responses 工具类型矩阵", "P2", "responses_tool_matrix"),
)


def _message(body: dict[str, Any] | None) -> dict[str, Any]:
    choices = (body or {}).get("choices") or []
    return (choices[0].get("message") or {}) if choices else {}


def _usage(body: dict[str, Any] | None) -> dict[str, Any]:
    return (body or {}).get("usage") or {}


def _is_rejection(status: int) -> bool:
    """4xx 且非 404 —— 404 表示端点不存在（链路/配置问题），不构成"参数被正确拒绝"。"""
    return 400 <= status < 500 and status != 404


def _endpoint_missing_reason(status: int, family: str) -> str | None:
    """端点不存在时给出可自解释的原因，避免把链路问题读成模型行为。"""
    if status != 404:
        return None
    return f"{family} 端点不存在（HTTP 404），用例目标特性未被触达；属链路/配置问题（检查 base_url 与端点派生），不是被测对象的行为结论"


def _cache_tokens(body: dict[str, Any] | None) -> int:
    usage = _usage(body)
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    return int(usage.get("prompt_cache_hit_tokens") or details.get("cached_tokens") or 0)


def _error_body(r: "ApiResponse") -> dict[str, Any]:
    """取错误体，兼容 {error:{message,type,code}} 与裸 {message}。"""
    body = r.body if isinstance(r.body, dict) else {}
    err = body.get("error")
    return err if isinstance(err, dict) else body


def _error_message(r: "ApiResponse") -> str:
    """错误文案：优先 error.message，其次 message，最后原始响应文本（供网关口径识别）。"""
    body = _error_body(r)
    for key in ("message", "msg", "error_msg", "detail", "error_description"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    error = body.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    return (r.text or "").strip()[:400]


def _gateway_note(issue: GatewayIssue | None) -> str:
    return f"｜{issue.reason}" if issue else ""


def _anthropic_text(r: "ApiResponse", block_type: str = "text") -> str:
    """拼接 Anthropic content 数组中指定类型块的可读文本。"""
    parts: list[str] = []
    for block in (r.body or {}).get("content") or []:
        if not isinstance(block, dict) or block.get("type") != block_type:
            continue
        parts.append(str(block.get("text") or block.get("thinking") or ""))
    return "".join(parts)


def _anthropic_block_types(r: "ApiResponse") -> list[str]:
    return [str(b.get("type")) for b in (r.body or {}).get("content") or [] if isinstance(b, dict)]


# 确定性图片夹具（PNG_SIGNATURE / png_bands / png_data_url / image_part 等）
# 已抽到 common/png_fixture.py，由本模块顶部 import 再导出，旧引用路径不变。
class DeepSeekV4Runner:
    def __init__(self, config: DeepSeekConfig, client: httpx.AsyncClient | None = None, audit_dir: Path | None = None):
        self.config = config
        self._external_client = client
        self.client = client
        self.executor: HttpExecutor | None = None
        # 执行证据：每个请求的完整 record 实时落到 <audit_dir>/requests.jsonl，
        # 供统一准入校验的 evidence_present 使用。
        self.audit_dir = audit_dir
        self.log: ResultLog | None = None
        self.current_case_id: str | None = None
        # 中转网关口径：记录当前用例期间命中的网关/平台侧问题，供 run() 统一降级 failed。
        self.current_model: str | None = None
        self._case_gateway_issues: list[GatewayIssue] = []

    async def __aenter__(self) -> "DeepSeekV4Runner":
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
                trust_env=self.config.trust_env,
                headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            )
        self.executor = HttpExecutor(timeout=self.config.timeout, verify_ssl=self.config.verify_ssl, trust_env=self.config.trust_env, client=self.client)
        await self.executor.__aenter__()
        if self.audit_dir is not None:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            self.log = ResultLog(self.audit_dir / "requests.jsonl")
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self.log:
            self.log.close()
            self.log = None
        if self.executor:
            await self.executor.__aexit__(*args)
        if self._external_client is None and self.client:
            await self.client.aclose()

    # —— 中转网关口径（gateway verdict）——
    # 判定原则见 dsv4/gateway_contract.py：前提不满足/能力面缺失 ⇒ blocked；功能可用但与文档有差异
    # ⇒ warning；只有请求确实进入模型、且「影响正常使用」的能力不达标才判 failed。

    def _gateway_issue(self, r: ApiResponse, *, family: str = "openai",
                       capability: str | None = None, model: str | None = None) -> GatewayIssue | None:
        return classify_gateway_issue(r.status, _error_message(r), family=family, capability=capability, model=model)

    def _gateway_verdict(self, r: ApiResponse, detail: str, observations: dict[str, Any], *,
                         family: str = "openai", capability: str | None = None, model: str | None = None):
        """非 200 的统一兜底：先过网关口径，命中则降级为 blocked/warning，否则判 failed。"""
        issue = self._gateway_issue(r, family=family, capability=capability, model=model)
        if issue:
            return False, f"{detail}{_gateway_note(issue)}", {**observations, "gateway_issue": issue.kind}, issue.status
        return False, detail, observations

    def _note_gateway(self, r: ApiResponse, family: str) -> None:
        """记录本次请求命中的网关/平台侧问题（非 200 才判），供用例结束后统一降级。"""
        if r.status == 200:
            return
        issue = classify_gateway_issue(r.status, _error_message(r), family=family, model=self.current_model)
        if issue:
            self._case_gateway_issues.append(issue)

    def _downgrade_gateway_failure(self, status: str, details: str) -> tuple[str, str]:
        """统一兜底：判了 failed、但失败请求命中网关/平台特征时，降级为 blocked / warning。

        单点保证「同一现象在不同用例里判法一致」——不必每个用例各自识别网关限制。
        命中条件严格限定为 404 端点缺失 / 429 限流 / 5xx 基础设施 / 账号前提未满足 /
        网关不支持该内容能力 / 网关自有取值域；模型自身返回的参数错误不带这些特征，仍是 failed。
        """
        if status != "failed" or not self._case_gateway_issues:
            return status, details
        issue = self._case_gateway_issues[-1]
        return issue.status, f"{details}｜网关口径降级（{issue.kind}）：{issue.reason}"

    def _request_headers(self, family: str, extra: dict[str, str] | None = None) -> dict[str, str]:
        request_headers = {"X-Request-ID": f"DSV4-{uuid.uuid4().hex[:16]}"}
        if family == "anthropic":
            request_headers.update({"x-api-key": self.config.api_key, "anthropic-version": "2023-06-01"})
        if extra:
            request_headers.update(extra)
        return request_headers

    async def request(self, path: str, payload: dict[str, Any], *, family: str = "openai", stream: bool = False, headers: dict[str, str] | None = None) -> ApiResponse:
        if self.executor is None:
            self.executor = HttpExecutor(timeout=self.config.timeout, verify_ssl=self.config.verify_ssl, trust_env=self.config.trust_env, client=self.client)
            await self.executor.__aenter__()
        request_headers = self._request_headers(family, headers)
        record = await self.executor.execute(case_id=payload.get("_case_id") or self.current_case_id or "DSV4-request", url=self.config.url(path, family), payload={k:v for k,v in payload.items() if not k.startswith("_")}, headers=request_headers, stream=stream)
        if self.log:
            await self.log.write(record)
        if not stream:
            response = ApiResponse(record.http_status or 0, record.response_body if isinstance(record.response_body, dict) else None, record.response_text, record.timing.get("e2e_ms", 0)/1000, record.response_headers, record.provider_request_id)
            self._note_gateway(response, family)
            return response
        info = {"reasoning_content": record.reasoning, "content": record.content, "usage": record.usage or {}, "done": record.done_received, "invalid_events": record.invalid_sse_events, "finish_reasons": record.finish_reasons, "raw_sse_lines": record.raw_sse_lines, "raw_sse_events": record.raw_sse_events, "received_events": record.parsed_sse_events}
        response = ApiResponse(record.http_status or 0, None, record.raw_response_body or record.response_text, record.timing.get("e2e_ms", 0)/1000, record.response_headers, record.provider_request_id, stream=info)
        self._note_gateway(response, family)
        return response

    async def request_raw(self, method: str, path: str, *, family: str = "openai", params: dict[str, Any] | None = None,
                          json_body: dict[str, Any] | None = None, files: dict[str, Any] | None = None,
                          data: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> ApiResponse:
        """GET / DELETE / multipart 通道，证据同样写入 requests.jsonl。"""
        if self.executor is None:
            self.executor = HttpExecutor(timeout=self.config.timeout, verify_ssl=self.config.verify_ssl, trust_env=self.config.trust_env, client=self.client)
            await self.executor.__aenter__()
        request_headers = self._request_headers(family, headers)
        record = await self.executor.execute_raw(case_id=self.current_case_id or "DSV4-request", method=method, url=self.config.url(path, family),
                                                 headers=request_headers, params=params, json_body=json_body, files=files, data=data)
        if self.log:
            await self.log.write(record)
        response = ApiResponse(record.http_status or 0, record.response_body if isinstance(record.response_body, dict) else None, record.response_text, record.timing.get("e2e_ms", 0)/1000, record.response_headers, record.provider_request_id)
        self._note_gateway(response, family)
        return response

    async def root_request(self, method: str, path: str, **kwargs: Any) -> ApiResponse:
        """根域端点（/models、/user/balance、/files）。

        文档口径是根域地址，但部分中网关只暴露带版本前缀的路径。
        先打根域，仅在 404 时回退到带版本地址，并把实际命中的地址写进证据，避免把网关差异读成能力缺失。
        """
        response = await self.request_raw(method, path, family="root", **kwargs)
        if response.status != 404:
            return response
        return await self.request_raw(method, path, family="openai", **kwargs)

    async def chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> ApiResponse:
        payload = {"model": model, "messages": messages, "max_tokens": 2048, **kwargs}
        family = payload.pop("_family", "openai")
        headers = payload.pop("_headers", None)
        return await self.request("chat/completions", payload, family=family, stream=bool(payload.get("stream")), headers=headers)

    async def run(self, models: tuple[str, ...] = MODELS, priorities: tuple[str, ...] = ("P0", "P1"), case_ids: set[str] | None = None, faces: tuple[str, ...] | None = None) -> list[CaseResult]:
        # 面过滤（默认 model+verify）：通道面（Beta/第二协议/附加端点/纯参数校验）
        # 在中转链路上大面积不通过，默认不跑；要跑得显式开 channel。
        from common.faces import ALL_FACES, filter_specs, selected_faces
        from .faces import CASE_FACES

        # 定向复测（case_ids）是用户显式点名，面过滤让位——否则点通道面用例会被静默跳过。
        active = ALL_FACES if case_ids else selected_faces(faces)
        selected = [spec for spec in CASE_SPECS
                    if spec.priority in priorities and (not case_ids or spec.case_id in case_ids)]
        results: list[CaseResult] = []
        for model in models:
            self.current_model = model
            for spec in filter_specs(selected, CASE_FACES, active):
                if spec.priority not in priorities or (case_ids and spec.case_id not in case_ids):
                    continue
                started = time.perf_counter()
                self.current_case_id = spec.case_id
                self._case_gateway_issues = []
                try:
                    outcome = await getattr(self, f"case_{spec.handler}")(model)
                    if len(outcome) == 4:
                        passed, details, observations, status = outcome
                    else:
                        passed, details, observations = outcome
                        status = "passed" if passed else "failed"
                    # 统一网关口径收口：failed 但失败原因属平台/网关侧 ⇒ 降级为 blocked/warning。
                    status, details = self._downgrade_gateway_failure(status, details)
                    results.append(CaseResult(spec.case_id, spec.name, model, passed, status, details, time.perf_counter() - started, observations))
                except Exception as exc:
                    results.append(CaseResult(spec.case_id, spec.name, model, False, "error", f"{type(exc).__name__}: {exc}", time.perf_counter() - started))
        self.current_case_id = None
        return results

    async def case_basic(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"})
        return r.status == 200 and bool(_message(r.body).get("content")), f"HTTP {r.status}", {"response_model": (r.body or {}).get("model")}

    async def case_default_thinking(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "计算 17*23，只给最终答案。"}])
        rc = _message(r.body).get("reasoning_content")
        final = _message(r.body).get("content") or ""
        return r.status == 200 and isinstance(rc, str) and bool(rc) and integer_answer(final, 391), "默认请求应返回 reasoning_content，且最终答案精确为391", {"reasoning_chars": len(rc or ""), "answer": final, "assertion_version": "exact-answer-1"}

    async def case_disabled_thinking(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"})
        rc = _message(r.body).get("reasoning_content")
        return r.status == 200 and not rc, "关闭思考后 reasoning_content 应为空", {}

    async def case_effort_mapping(self, model: str):
        """思考强度映射（DS-V4-04）。

        官方无歧义开启档位是 low/high/max；medium/xhigh 为兼容别名（→high，两处文档都认可）。
        网关若把取值域收窄到只认 low/high/max，属"取值域差异、不影响使用" ⇒ warning，不判失败。
        """
        canonical = ("low", "high", "max")
        aliases = ("medium", "xhigh")
        responses: dict[str, ApiResponse] = {}
        for effort in canonical + aliases:
            responses[effort] = await self.chat(model, [{"role": "user", "content": "计算 19*31，只给答案。"}],
                                                thinking={"type": "enabled"}, reasoning_effort=effort, max_tokens=2048)
        statuses = {k: r.status for k, r in responses.items()}
        if any(r.status == 404 for r in responses.values()):
            return False, _endpoint_missing_reason(404, "openai") or "端点不存在", statuses, "blocked"
        core_bad = {k: statuses[k] for k in canonical if statuses[k] != 200}
        if core_bad:
            sample = responses[next(iter(core_bad))]
            issue = self._gateway_issue(sample, model=model)
            if issue:
                return False, f"官方无歧义档位 {list(canonical)} 未被全部接受：{core_bad}{_gateway_note(issue)}", {**statuses, "gateway_issue": issue.kind}, issue.status
            return False, f"官方无歧义档位 {list(canonical)} 未被全部接受：{core_bad}", statuses
        alias_bad = {k: statuses[k] for k in aliases if statuses[k] != 200}
        if alias_bad:
            return True, (f"low/high/max 三档均被接受；兼容别名 {list(alias_bad)} 未被接受（{alias_bad}）："
                          "官方将其列为兼容别名，网关取值域更窄，不影响正常使用"), statuses, "warning"
        return True, "low/high/max 及兼容别名 medium/xhigh 均被接受", statuses

    async def case_ignored_sampling(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "enabled"}, temperature=1.7, top_p=0.2, presence_penalty=1.0, frequency_penalty=1.0)
        return r.status == 200, "思考模式下兼容参数应静默接受", {"http_status": r.status}

    async def case_stream_thinking(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "计算 29*13，只给最终答案。"}], thinking={"type": "enabled"}, stream=True, stream_options={"include_usage": True})
        s = r.stream or {}
        ok = r.status == 200 and bool(s.get("reasoning_content")) and integer_answer(s.get("content") or "", 377) and s.get("done") and s.get("invalid_events") == 0
        return ok, "SSE 中 reasoning/content 应可独立拼接、收到 [DONE] 且最终答案精确为377", {k: len(v) if isinstance(v, str) else v for k, v in s.items() if k != "usage"}

    async def case_replay_without_tools(self, model: str):
        first = await self.chat(model, [{"role": "user", "content": "记住数字 731，并回复收到。"}])
        assistant = _message(first.body)
        second = await self.chat(model, [{"role": "user", "content": "记住数字 731，并回复收到。"}, assistant, {"role": "user", "content": "刚才的数字是什么？只回复数字。"}])
        return first.status == second.status == 200 and "731" in str(_message(second.body).get("content")), "无 tools 时回传 assistant 消息应成功", {}

    @staticmethod
    def tools() -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "get_number", "description": "返回测试数字", "parameters": {"type": "object", "properties": {}, "required": []}}}]

    # thinking 模式不支持 tool_choice="required"（上游返回 400
    # "Thinking mode does not support this tool_choice"），因此只能用 auto + 强提示诱导调用。
    # 两条提示依次增强，第一条未触发工具调用时再试第二条。
    TOOL_PRIMING_PROMPTS = (
        "必须调用 get_number 获取测试数字，不要自行回答。",
        "禁止直接回答。请先调用 get_number 工具获取测试数字，再基于工具返回结果作答。",
    )

    async def _tool_first(self, model: str) -> tuple[ApiResponse, dict[str, Any], str]:
        """建立「思考模式 + 工具调用」的前置轮次，返回 (响应, assistant 消息, 实际使用的提示词)。"""
        last: tuple[ApiResponse, dict[str, Any], str] | None = None
        for prompt in self.TOOL_PRIMING_PROMPTS:
            r = await self.chat(model, [{"role": "user", "content": prompt}], tools=self.tools(), tool_choice="auto", thinking={"type": "enabled"})
            msg = _message(r.body)
            last = (r, msg, prompt)
            if r.status != 200:
                break  # 上游拒绝该请求（如参数组合不被支持），重试无意义，交由调用方按 blocked 处理
            if (msg.get("tool_calls") or []) and msg.get("reasoning_content"):
                return last
        return last  # type: ignore[return-value]

    @staticmethod
    def _tool_precondition_obs(r: ApiResponse, msg: dict[str, Any], calls: list[Any]) -> dict[str, Any]:
        return {"http_status": r.status, "has_tool_calls": bool(calls), "has_reasoning": bool(msg.get("reasoning_content")), "response": r.text[:500]}

    @staticmethod
    def _tool_precondition_reason(r: ApiResponse, msg: dict[str, Any], calls: list[Any]) -> str:
        """把「前置轮次不可用」的原因写清楚，避免把上游限制误读成模型行为。"""
        if r.status != 200:
            return f"前置轮次被上游拒绝（HTTP {r.status}）：{r.text[:200]}；后续回传链路无法验证"
        if not calls:
            return "前置轮次未产生 tool_calls（thinking 模式不支持 tool_choice=required，已改用 auto+强提示诱导仍未触发）；后续回传链路无法验证"
        return "前置轮次产生了 tool_calls 但未同时返回 reasoning_content；后续回传链路无法验证"

    async def case_tool_replay(self, model: str):
        first, msg, prompt = await self._tool_first(model)
        calls = msg.get("tool_calls") or []
        if first.status != 200 or not calls or not msg.get("reasoning_content"):
            return False, self._tool_precondition_reason(first, msg, calls), self._tool_precondition_obs(first, msg, calls), "blocked"
        messages = [{"role": "user", "content": prompt}, msg, {"role": "tool", "tool_call_id": calls[0]["id"], "content": "731"}]
        second = await self.chat(model, messages, tools=self.tools(), thinking={"type": "enabled"})
        return second.status == 200 and "731" in str(_message(second.body).get("content")), "完整回传后应生成工具结果答案", {"second_status": second.status, "first_status": first.status}

    async def case_tool_missing_reasoning(self, model: str):
        first, msg, prompt = await self._tool_first(model)
        calls = msg.get("tool_calls") or []
        if first.status != 200 or not calls or not msg.get("reasoning_content"):
            return False, self._tool_precondition_reason(first, msg, calls) + "；本负向用例不计失败", self._tool_precondition_obs(first, msg, calls), "blocked"
        msg = dict(msg); msg.pop("reasoning_content", None)
        second = await self.chat(model, [{"role": "user", "content": prompt}, msg, {"role": "tool", "tool_call_id": calls[0]["id"], "content": "731"}], tools=self.tools(), thinking={"type": "enabled"})
        observations = {"http_status": second.status, "response": second.text[:300]}
        if second.status == 400:
            return True, "漏传 reasoning_content 时上游返回 400，符合官方文档的严格校验要求", observations
        if second.status == 200:
            # 官方文档（Thinking Mode → Tool Calls）承诺"未正确回传 reasoning_content 会返回 400"，
            # 线上实际容忍并正常作答。与 DS-V4-10 属同类信号：记为文档/线上行为差异，不判模型失败。
            return True, "线上未执行文档要求的严格校验：漏传 reasoning_content 仍返回 200 并正常作答（文档承诺 400，属线上行为差异）", observations, "warning"
        return False, f"漏传 reasoning_content 时返回非预期状态 HTTP {second.status}", observations

    async def case_tools_plain_turn_replay(self, model: str):
        first = await self.chat(model, [{"role": "user", "content": "不要调用工具，只回复 OK。"}], tools=self.tools(), tool_choice="none", thinking={"type": "enabled"})
        msg = _message(first.body)
        if first.status != 200 or not msg.get("reasoning_content"):
            return False, "无法建立携带 tools 但未调用工具的前置轮次", {"http_status": first.status, "response": first.text[:500]}, "blocked"
        without_reasoning = dict(msg); without_reasoning.pop("reasoning_content", None)
        second = await self.chat(model, [{"role": "user", "content": "不要调用工具，只回复 OK。"}, without_reasoning, {"role": "user", "content": "再回复一次 OK。"}], tools=self.tools(), tool_choice="none", thinking={"type": "enabled"})
        observations = {"http_status": second.status, "response": second.text[:300]}
        if second.status == 400:
            return True, "未进入工具调用链时仍执行 reasoning_content 严格校验", observations
        if second.status == 200:
            return True, "未进入工具调用链时允许省略 reasoning_content（官方线上可接受行为）", observations, "warning"
        return False, "未进入工具调用链时返回非预期状态", observations

    async def case_json_output(self, model: str):
        r = await self.chat(model, [{"role": "system", "content": '请仅输出 json，例如 {"value":731}'}, {"role": "user", "content": "以 json 输出 value=731"}], response_format={"type": "json_object"}, thinking={"type": "disabled"})
        text = _message(r.body).get("content") or ""
        try: parsed = json.loads(text)
        except json.JSONDecodeError: parsed = None
        return r.status == 200 and isinstance(parsed, dict) and parsed.get("value") == 731, "响应必须是可解析 JSON", {"content": text[:300]}

    async def case_json_missing_keyword(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "返回一个包含 value=731 的对象。"}], response_format={"type": "json_object"}, thinking={"type": "disabled"}, max_tokens=32)
        # 官方要求 Prompt 包含 json；实现可以明确拒绝，也可能返回空或有效结构，关键是请求可终止且可分类。
        controlled = r.status == 400 or (r.status == 200 and isinstance(_message(r.body).get("content"), str))
        return controlled, "缺少 json 关键字时必须得到可分类响应，不能协议异常", {"http_status": r.status, "content": str(_message(r.body).get("content"))[:200]}

    async def case_json_truncation(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": '输出 json，包含键 items，值为 100 个整数。'}], response_format={"type": "json_object"}, thinking={"type": "disabled"}, max_tokens=2)
        text = _message(r.body).get("content") or ""
        finish = (((r.body or {}).get("choices") or [{}])[0]).get("finish_reason")
        detectable = r.status == 200 and (finish == "length" or not text or _invalid_json(text))
        return detectable, "小 max_tokens 导致的截断/空内容必须可检测", {"finish_reason": finish, "content": text}

    async def case_cache_hit(self, model: str):
        """自动 Context Cache 命中（官方口径：best-effort，**不保证第 2 跳必命中**）。

        官方《Context Caching》：每个请求在「用户输入结束位置」与「模型输出结束位置」产生 cache
        prefix unit，后续请求须**完整匹配**某个 unit 才命中；缓存构建需数秒，且明确
        "works on a best-effort basis and does not guarantee a 100% cache hit rate"
        （官方 Example 2 中前两次请求都不命中）。故断言改为**多跳内任一次命中即通过**；
        全程未命中记 blocked（未观测到前提），与 DS-V4-16/18 同口径，不判模型失败。
        """
        prefix = "缓存专项固定前缀-" + ("A1B2C3D4。" * 800)
        first = await self.chat(model, [{"role": "user", "content": prefix + "只回复一。"}], thinking={"type": "disabled"}, max_tokens=8)
        seed = [{"role": "user", "content": prefix + "只回复一。"}]
        if first.status == 200:
            seed = seed + [_message(first.body)]
        hits: list[int] = []
        responses: list[ApiResponse] = []
        for tail in ("只回复二。", "只回复三。"):
            await asyncio.sleep(3)
            r = await self.chat(model, seed + [{"role": "user", "content": tail}], thinking={"type": "disabled"}, max_tokens=8)
            responses.append(r)
            hits.append(_cache_tokens(r.body))
        observations = {"first_status": first.status, "followup_status": [r.status for r in responses], "cache_hit_tokens": hits}
        if first.status != 200 or any(r.status != 200 for r in responses):
            bad = next((r for r in responses if r.status != 200), first)
            return self._gateway_verdict(bad, f"缓存探测请求返回非预期状态 HTTP {bad.status}", observations)
        if any(hit > 0 for hit in hits):
            return True, f"重复完整前缀后观测到缓存命中（各跳命中 tokens={hits}）", observations
        return False, (f"多跳内未观测到缓存命中（命中 tokens={hits}）：官方为 best-effort 且不保证 100% 命中，"
                       "该现象不足以判模型缺陷，记为「未观测到前提」"), observations, "blocked"

    async def case_cache_prefix_rule(self, model: str):
        """完整前缀匹配规则（官方 Example 2：公共前缀需在检测持久化后才能命中，best-effort）。"""
        common = "公共缓存前缀-" + ("PREFIX731。" * 700)
        a = await self.chat(model, [{"role": "user", "content": common + "分支A"}], thinking={"type": "disabled"}, max_tokens=8)
        await asyncio.sleep(3)
        b = await self.chat(model, [{"role": "user", "content": common + "分支B"}], thinking={"type": "disabled"}, max_tokens=8)
        await asyncio.sleep(3)
        c = await self.chat(model, [{"role": "user", "content": common + "分支C"}], thinking={"type": "disabled"}, max_tokens=8)
        hit_b, hit_c = _cache_tokens(b.body), _cache_tokens(c.body)
        observations = {"second_hit": hit_b, "third_hit": hit_c, "statuses": [a.status, b.status, c.status]}
        if any(r.status != 200 for r in (a, b, c)):
            bad = next(r for r in (a, b, c) if r.status != 200)
            return self._gateway_verdict(bad, f"缓存前缀探测请求返回非预期状态 HTTP {bad.status}", observations)
        if hit_b > 0 or hit_c > 0:
            return True, f"公共前缀被持久化后可命中（第 2 跳={hit_b}、第 3 跳={hit_c} tokens），符合完整前缀规则", observations
        return False, (f"多跳内未观测到公共前缀命中（第 2 跳={hit_b}、第 3 跳={hit_c}）："
                       "官方缓存为 best-effort，记为「未观测到前提」"), observations, "blocked"

    async def case_user_isolation(self, model: str):
        """user_id 缓存隔离（官方「限速与隔离」：`user_id` 用于 KVCache 隔离）。

        **这条最容易误判**：命中可能来自三个完全不同的地方，只有第一种该判模型 failed。

        1. DeepSeek 原生 KV 缓存按 user_id 隔离 → 换 user_id 仍命中**且**响应带原生字段
           `prompt_cache_hit_tokens` ⇒ 真·跨 user 命中（模型/上游平台侧隐私问题）。
        2. 网关整段响应回放（缓存 key 只哈希 messages+model、不带 user_id）⇒ 换 user_id 拿到的
           响应 `id`/`created` 与上一跳**完全相同** ⇒ 网关侧行为，与模型无关。
        3. 网关/上游服务层的前缀缓存（OpenAI 形状 `prompt_tokens_details.cached_tokens`）⇒
           换 user_id 照样命中，但**从不出**原生字段，响应 `id` 每次都新。
           实测本网关即此类：`cached_tokens` 随前缀长度按 64 对齐增长（384/896/1920/4096），
           换 user_id 数值不变，内容每次都新 ⇒ 无法归因到 DeepSeek 原生 KV，记 blocked。
        """
        prefix = "隔离测试固定前缀-" + ("XYZ731。" * 600)
        uid1, uid2 = f"ds-test-{uuid.uuid4().hex[:8]}", f"ds-test-{uuid.uuid4().hex[:8]}"
        await self.chat(model, [{"role": "user", "content": prefix}], user_id=uid1, thinking={"type": "disabled"}, max_tokens=8)
        await asyncio.sleep(2)
        same = await self.chat(model, [{"role": "user", "content": prefix}], user_id=uid1, thinking={"type": "disabled"}, max_tokens=8)
        other = await self.chat(model, [{"role": "user", "content": prefix}], user_id=uid2, thinking={"type": "disabled"}, max_tokens=8)
        same_hit, other_hit = _cache_tokens(same.body), _cache_tokens(other.body)
        same_body = same.body if isinstance(same.body, dict) else {}
        other_body = other.body if isinstance(other.body, dict) else {}
        native_present = any(
            isinstance(_usage(body).get(key), int)
            for body in (same_body, other_body)
            for key in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
        )
        replayed = bool(same_body.get("id")) and other_body.get("id") == same_body.get("id")
        observations = {
            "same_user_hit": same_hit,
            "other_user_hit": other_hit,
            "cache_field_shape": "deepseek_native" if native_present else "openai_cached_tokens",
            "native_cache_fields_present": native_present,
            "same_response_id": same_body.get("id"),
            "other_response_id": other_body.get("id"),
            "gateway_replayed_response": replayed,
        }
        if same.status != 200 or other.status != 200:
            return False, "user_id 缓存隔离请求失败", {**observations, "same_status": same.status, "other_status": other.status}
        if same_hit == 0:
            return True, "本轮未观测到同 user_id 缓存命中，无法判定隔离（缓存为 best-effort）", observations, "blocked"
        if other_hit > 0:
            if replayed:
                return False, (
                    "不同 user_id 拿到与同 user 请求**完全相同**的响应（同 id）⇒ 网关整段响应回放（缓存 key 未含 "
                    "user_id），属网关侧行为，与模型/DeepSeek 原生 KV 无关"
                ), observations, "blocked"
            if not native_present:
                return False, (
                    f"不同 user_id 同样命中前缀缓存（{other_hit} tokens），但该链路只上报 OpenAI 形状 "
                    "`prompt_tokens_details.cached_tokens`、从不返回 DeepSeek 原生 `prompt_cache_hit_tokens`/"
                    "`prompt_cache_miss_tokens`，且响应 id 每次都新 ⇒ 命中发生在网关/上游服务层，"
                    "**无法归因到 DeepSeek 原生 KV**，user_id 隔离在该链路上不可验证"
                ), observations, "blocked"
            return False, "已观测同 user_id 命中，但不同 user_id 也发生命中（原生字段）", observations
        return True, "相同 user_id 命中且不同 user_id 未命中", observations

    async def case_session_continuity_cache(self, model: str):
        """同一身份按会话追加消息，验证上下文连续性与前缀缓存是否同时成立。"""
        session_id = f"ds-session-{uuid.uuid4().hex[:10]}"
        prefix = "SESSION-CONTINUITY-DSV4-" + ("固定会话背景。" * 700)
        turns = [
            [{"role": "system", "content": prefix}, {"role": "user", "content": "记住会话暗号：ORBIT-731。只回复已记住。"}],
            [{"role": "system", "content": prefix}, {"role": "user", "content": "记住会话暗号：ORBIT-731。只回复已记住。"}, {"role": "assistant", "content": "已记住。"}, {"role": "user", "content": "会话暗号是什么？只回复暗号。"}],
            [{"role": "system", "content": prefix}, {"role": "user", "content": "记住会话暗号：ORBIT-731。只回复已记住。"}, {"role": "assistant", "content": "已记住。"}, {"role": "user", "content": "会话暗号是什么？只回复暗号。"}, {"role": "assistant", "content": "ORBIT-731"}, {"role": "user", "content": "再次确认暗号，只回复暗号。"}],
        ]
        rows = []
        for i, messages in enumerate(turns):
            r = await self.chat(model, messages, user_id=session_id, thinking={"type": "disabled"}, max_tokens=16)
            text = _message(r.body).get("content") or ""
            rows.append({"turn": i + 1, "status": r.status, "content": text[:100], "cache_hit_tokens": _cache_tokens(r.body), "prompt_tokens": _usage(r.body).get("prompt_tokens")})
            if i < len(turns) - 1:
                await asyncio.sleep(2)
        continuity = rows[1]["status"] == rows[2]["status"] == 200 and all("ORBIT-731" in rows[i]["content"] for i in (1, 2))
        cache_hits = [r["cache_hit_tokens"] for r in rows[1:]]
        cache_observed = any(x > 0 for x in cache_hits)
        ok = all(r["status"] == 200 for r in rows) and continuity and cache_observed
        status = "passed" if ok else ("blocked" if continuity and not cache_observed else "failed")
        return ok, f"Session 连续性={'通过' if continuity else '失败'}，后续轮缓存命中={cache_hits}", {"session_id": session_id, "turns": rows, "continuity": continuity, "cache_observed": cache_observed}, status

    async def case_invalid_user_id(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], user_id="非法 user id!", thinking={"type": "disabled"})
        observations = {"http_status": r.status, "response": r.text[:300]}
        missing = _endpoint_missing_reason(r.status, "openai")
        if missing:
            return False, missing, observations, "blocked"
        if _is_rejection(r.status):
            return True, "接口拒绝非法 user_id", observations
        if r.status == 200:
            return True, "接口接受非法 user_id；记录为文档/线上行为差异，不判模型失败", observations, "warning"
        return False, "非法 user_id 返回非预期状态", observations

    async def case_strict_tool(self, model: str):
        tool = {"type": "function", "function": {"name": "create_user", "strict": True, "description": "创建测试用户", "parameters": {"type": "object", "properties": {"email": {"type": "string", "format": "email"}, "score": {"type": "integer", "minimum": 1, "maximum": 5}}, "required": ["email", "score"], "additionalProperties": False}}}
        r = await self.chat(model, [{"role": "user", "content": "调用 create_user，邮箱 a@example.com，评分 5。"}], _family="beta", tools=[tool], tool_choice={"type": "function", "function": {"name": "create_user"}}, thinking={"type": "disabled"})
        calls = _message(r.body).get("tool_calls") or []
        try: args = json.loads(calls[0]["function"]["arguments"])
        except (IndexError, KeyError, json.JSONDecodeError): args = {}
        observations = {"arguments": args, "http_status": r.status}
        missing = _endpoint_missing_reason(r.status, "beta")
        if missing:
            return False, missing, observations, "blocked"
        return r.status == 200 and args == {"email": "a@example.com", "score": 5}, "Strict 参数必须严格符合 Schema", observations

    async def case_strict_invalid(self, model: str):
        tool = {"type": "function", "function": {"name": "bad", "strict": True, "description": "非法测试", "parameters": {"type": "object", "properties": {"x": {"type": "string", "minLength": 2}}, "required": [], "additionalProperties": True}}}
        r = await self.chat(model, [{"role": "user", "content": "调用 bad"}], _family="beta", tools=[tool], thinking={"type": "disabled"})
        observations = {"http_status": r.status, "error": r.text[:300]}
        missing = _endpoint_missing_reason(r.status, "beta")
        if missing:
            return False, missing, observations, "blocked"
        if _is_rejection(r.status):
            return True, "不支持的 Strict Schema 应返回 4xx", observations
        return False, "非法 Strict Schema 未返回预期 4xx", observations

    async def case_prefix(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "写一个 Python 加法函数，只输出代码。"}, {"role": "assistant", "content": "def add(a, b):\n", "prefix": True}], _family="beta", thinking={"type": "disabled"}, stop=["```"], max_tokens=64)
        text = _message(r.body).get("content") or ""
        observations = {"content": text[:300], "http_status": r.status}
        missing = _endpoint_missing_reason(r.status, "beta")
        if missing:
            return False, missing, observations, "blocked"
        correct = arithmetic_function("def add(a, b):\n" + text, "add", [((2, 3), 5), ((-4, 2), -2), ((0, 0), 0)])
        return r.status == 200 and correct, "前缀补全的加法函数应通过正数、负数和零值校验", observations

    async def case_prefix_invalid(self, model: str):
        """负向用例：prefix 被放在非末尾 assistant 消息上时的处理行为。

        官方《Chat Prefix Completion (Beta)》注意事项只写"用户需确保 messages 里最后一条
        消息的 role 为 assistant 且 prefix 为 True"，即**调用方义务**，并未承诺对错误放置
        返回 4xx。线上实测为静默忽略该参数并正常作答。因此 4xx（严格校验型拒绝）与 200
        （宽松忽略）都不构成模型能力失败；200 记为 warning，保留"上游不校验该参数"这一观察，
        避免调用方写法错误长期不被发现。对照 DS-V4-20：strict schema 才是上游明确执行的校验。
        """
        r = await self.chat(model, [{"role": "user", "content": "写代码", "prefix": True}], _family="beta", thinking={"type": "disabled"}, max_tokens=32)
        content = _message(r.body).get("content") or ""
        observations = {"http_status": r.status, "content": content[:200], "response": r.text[:300]}
        missing = _endpoint_missing_reason(r.status, "beta")
        if missing:
            return False, missing, observations, "blocked"
        if _is_rejection(r.status):
            return True, "上游对非法放置的 prefix 返回 4xx，属校验型拒绝", observations
        if r.status == 200 and content:
            return True, "官方文档仅要求调用方自行确保 prefix 位于最后一条 assistant 消息，未承诺拒绝；线上静默忽略该参数并正常作答（文档未定义的宽松行为，非模型能力问题）", observations, "warning"
        return False, f"非法放置的 prefix 未得到可解释的处理结果（HTTP {r.status}）", observations

    async def case_fim(self, model: str):
        payload = {"model": model, "prompt": "def fib(n):\n", "suffix": "\n    return fib(n-1) + fib(n-2)", "max_tokens": 128}
        r = await self.request("completions", payload, family="beta")
        text = (((r.body or {}).get("choices") or [{}])[0]).get("text") or ""
        observations = {"content": text[:300], "http_status": r.status}
        missing = _endpoint_missing_reason(r.status, "beta")
        if missing:
            return False, missing, observations, "blocked"
        correct = arithmetic_function(payload["prompt"] + text + payload["suffix"], "fib", [((0,), 0), ((1,), 1), ((2,), 1), ((5,), 5), ((8,), 21)])
        return r.status == 200 and correct, "Beta FIM 补全应正确处理 Fibonacci 基例和递归结果", observations

    async def case_anthropic_mapping(self, model: str):
        alias = "claude-opus-4-test" if model.endswith("pro") else "claude-sonnet-4-test"
        payload = {"model": alias, "max_tokens": 32, "messages": [{"role": "user", "content": "只回复 OK"}]}
        r = await self.request("v1/messages", payload, family="anthropic")
        response_model = (r.body or {}).get("model")
        expected = "pro" if model.endswith("pro") else "flash"
        observations = {"requested": alias, "response_model": response_model, "http_status": r.status}
        missing = _endpoint_missing_reason(r.status, "anthropic")
        if missing:
            return False, missing, observations, "blocked"
        return r.status == 200 and expected in str(response_model).lower(), "Claude 别名应映射到目标 DeepSeek 型号", observations

    async def case_responses_stateless(self, model: str):
        """官方 Responses 契约：无状态 API —— `store` 恒 false、`previous_response_id` 恒 null。

        官方《Using the Responses API》把 `previous_response_id` / `store` 列为 **Not supported**，
        并明确"Unsupported parameters are silently ignored and do not cause errors"。
        网关若改成强校验（4xx）属"比官方更严"的实现差异，不影响正常使用 ⇒ warning；
        404/429/5xx 走网关口径 ⇒ blocked。
        """
        payload = {"model": model, "input": "只回复 OK", "store": True, "previous_response_id": "resp_nonexistent", "max_output_tokens": 32}
        r = await self.request("responses", payload)
        body = r.body or {}
        observations = {"http_status": r.status, "store": body.get("store"),
                        "previous_response_id": body.get("previous_response_id"), "error": _error_message(r)[:200]}
        if r.status in (404, 429) or r.status >= 500:
            return self._gateway_verdict(r, f"Responses 无状态字段请求返回 HTTP {r.status}", observations, model=model)
        if r.status != 200:
            return True, (f"网关对文档声明「静默忽略」的字段做了强校验（HTTP {r.status}）：{_error_message(r)[:150]}；"
                          "与官方承诺不符但属更严格的实现，不影响正常使用"), observations, "warning"
        if body.get("store") is False and body.get("previous_response_id") is None:
            return True, "按官方契约返回固定值 store=false / previous_response_id=null", observations
        return False, (f"响应固定值不符：store={body.get('store')}、previous_response_id={body.get('previous_response_id')}"
                       "（官方：store 恒 false、previous_response_id 恒 null）"), observations



    async def case_responses_basic(self, model: str):
        r = await self.request("responses", {"model": model, "input": "只回复 OK", "max_output_tokens": 32})
        body = r.body or {}
        output = body.get("output")
        return r.status == 200 and isinstance(body.get("id"), str) and isinstance(output, list), "Responses API 应返回 id 与 output 数组", {"id": body.get("id"), "output_items": len(output or [])}

    async def case_usage_nonstream(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"})
        usage = _usage(r.body)
        prompt, completion, total = usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")
        ok = r.status == 200 and all(isinstance(v, int) and v >= 0 for v in (prompt, completion, total)) and total == prompt + completion
        return ok, "Usage 应包含非负 prompt/completion/total，且 total 等于两者之和", usage

    async def case_usage_stream(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"}, stream=True, stream_options={"include_usage": True})
        info, usage = r.stream or {}, (r.stream or {}).get("usage") or {}
        ok = r.status == 200 and info.get("done") and info.get("invalid_events") == 0 and isinstance(usage.get("total_tokens"), int) and bool(info.get("finish_reasons"))
        return ok, "流式响应应有合法 SSE、[DONE]、finish_reason 与 Usage", {"usage": usage, "finish_reasons": info.get("finish_reasons"), "done": info.get("done")}

    async def case_stop_sequence(self, model: str):
        marker = "<END>"
        r = await self.chat(model, [{"role": "user", "content": f"严格输出 ABC{marker}XYZ"}], thinking={"type": "disabled"}, stop=[marker], max_tokens=32)
        content = str(_message(r.body).get("content") or "")
        return r.status == 200 and marker not in content and "XYZ" not in content, "Stop 命中后不得输出停止串及其后内容", {"content": content}

    async def case_temperature_bounds(self, model: str):
        statuses = []
        for value in (0.0, 2.0):
            r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"}, temperature=value, max_tokens=16)
            statuses.append(r.status)
        return all(status == 200 for status in statuses), "temperature=0 与 2 均应接受", {"statuses": statuses}

    async def case_temperature_invalid(self, model: str):
        """temperature 越界（官方只给出范围 [0,2]，**未承诺 4xx 拒绝**）。"""
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"}, temperature=2.1, max_tokens=16)
        observations = {"http_status": r.status, "error": _error_message(r)[:300]}
        if _is_rejection(r.status):
            return True, f"temperature=2.1 被 4xx 拒绝（HTTP {r.status}）", observations
        if r.status == 200:
            return True, "temperature=2.1 未被拒绝（HTTP 200）：官方只给出取值范围、未承诺拒绝，属宽松放行", observations, "warning"
        return self._gateway_verdict(r, f"temperature 越界返回非预期状态 HTTP {r.status}", observations, model=model)

    async def case_messages_invalid(self, model: str):
        """messages 结构非法（负向：应被拒绝而非 200）。"""
        r = await self.request("chat/completions", {"model": model, "messages": "not-an-array", "max_tokens": 16})
        observations = {"http_status": r.status, "error": _error_message(r)[:300]}
        if _is_rejection(r.status):
            return True, f"结构非法的 messages 被 4xx 拒绝（HTTP {r.status}）", observations
        if r.status == 200:
            return True, "messages 非数组未被拒绝（HTTP 200）：属宽松放行，留痕", observations, "warning"
        return self._gateway_verdict(r, f"非法 messages 返回非预期状态 HTTP {r.status}", observations, model=model)

    async def case_tool_choice_none(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "不要调用工具，只回复 OK"}], tools=self.tools(), tool_choice="none", thinking={"type": "disabled"})
        calls = _message(r.body).get("tool_calls") or []
        return r.status == 200 and not calls, "tool_choice=none 不得返回工具调用", {"tool_calls": calls}

    async def case_tool_choice_required(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "调用 get_number"}], tools=self.tools(), tool_choice="required", thinking={"type": "disabled"})
        calls = _message(r.body).get("tool_calls") or []
        return r.status == 200 and bool(calls) and calls[0].get("function", {}).get("name") == "get_number", "tool_choice=required 必须调用声明的工具", {"tool_calls": calls[:1]}

    # ==================================================================================
    # 2026-09-24 覆盖度评审补充用例（DS-V4-36 ~ DS-V4-61）
    #
    # 判定口径（2026-10-14 起统一为「中转口径」，实现见 dsv4/gateway_contract.py）：
    #   - **网关/平台侧**问题——端点缺失 404、限流 429、账号/平台前提未满足（5xx OptInRequired 等）、
    #     网关不收某类内容、网关自有取值域、被测模型本就不提供的能力面（如 Pro 的视觉）——
    #     一律记 blocked/warning，绝不判 failed；由 `run()` 的 `_downgrade_gateway_failure()`
    #     统一收口，保证「同一现象在各用例里判法一致」。
    #   - 文档明确承诺 4xx 的负向用例：4xx → passed；宽松放行 200 → warning（不阻断准入）。
    #   - 文档承诺支持的正向用例：200 且满足断言 → passed；4xx → failed（除非命中上面网关特征）。
    #   - 文档只给"取值范围/调用方义务"、未承诺拒绝的边界：线上宽松接受记 warning，不判失败。
    # ==================================================================================

    async def case_image_input(self, model: str):
        """图片输入识别：图文请求能否真的读出图上内容。

        官方口径：图片输入（Vision）归属 `deepseek-flash`（现行版本 V4.1-Flash-1017），
        `deepseek-v4-pro` 与视觉开放前的 `deepseek-v4-flash-0731` 均未声明支持图片。
        被测模型不提供该能力面时只能记 blocked/warning——**不可验证**，不得把
        "网关/模型不收图片"或"答不出图"读成"模型视觉能力不达标"。

        注意 `prompt_tokens` 增量**不能**单独作为"图片被理解"的证据：2026-10-21 实测
        `api.oinone.top` / domestic-ark 分组上，纯文本上游（v4-flash-0731、v4-pro-0813）收到图片块
        同样返回 200 且 prompt_tokens 都 +75（24 → 99），但都答不出图；真正看图的
        V4.1-Flash 同请求是 24 → 208。故只有「内容识别正确」才算通过，token 增量仅作留痕。
        """
        prompt = "这张图上半部分和下半部分分别是什么颜色？只回答两个颜色词，用中文。"
        vision_capable = model_supports_capability(model, CAPABILITY_VISION)
        baseline = await self.chat(model, [{"role": "user", "content": prompt}], thinking={"type": "disabled"}, max_tokens=32)
        with_image = await self.chat(model, [{"role": "user", "content": [{"type": "text", "text": prompt}, image_part(png_data_url())]}],
                                     thinking={"type": "disabled"}, max_tokens=32)
        content = str(_message(with_image.body).get("content") or "")
        base_tokens = _usage(baseline.body).get("prompt_tokens")
        image_tokens = _usage(with_image.body).get("prompt_tokens")
        observations = {"http_status": with_image.status, "content": content[:200], "baseline_prompt_tokens": base_tokens,
                        "image_prompt_tokens": image_tokens, "vision_capable_model": vision_capable,
                        "error": _error_message(with_image)[:200]}
        identified = "红" in content and "蓝" in content
        counted = isinstance(base_tokens, int) and isinstance(image_tokens, int) and image_tokens > base_tokens
        if with_image.status == 200 and identified and counted:
            return True, "图片被识别（上红下蓝）且 prompt_tokens 高于同提示的纯文本请求（图片已计入）", observations
        if with_image.status == 200:
            detail = (f"图片请求返回 200 但未正确识别图像内容（期望同时出现「红」「蓝」）：{content[:120]}"
                      if not identified else
                      f"图片请求返回 200 但 prompt_tokens 未高于纯文本基线（疑似图片被丢弃）：{base_tokens} -> {image_tokens}")
            if not vision_capable:
                return True, f"{detail}；该模型未声明支持视觉（官方归属 deepseek-flash），不判模型失败", observations, "warning"
            return False, detail, observations
        detail = f"图片输入请求返回非预期状态 HTTP {with_image.status}（文本请求 HTTP {baseline.status}）"
        if not vision_capable:
            return False, (f"{detail}；官方文档把图片输入归属 deepseek-flash，被测模型 {model} 未声明支持视觉"
                           " ⇒ 该能力面在本模型上不可验证（换 deepseek-flash 复跑即可取证）"), observations, "blocked"
        return self._gateway_verdict(with_image, detail, observations, model=model, capability=CAPABILITY_VISION)

    async def case_model_alias(self, model: str):
        """模型别名口径与退休别名路由。"""
        primary = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"}, max_tokens=8)
        alias = RETIRED_ALIASES[0]
        retired = await self.chat(alias, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"}, max_tokens=8)
        drifted = model not in PRIMARY_MODELS and model not in RETIRED_ALIASES
        observations = {"configured_model": model, "configured_response_model": (primary.body or {}).get("model"),
                        "retired_alias": alias, "retired_status": retired.status, "retired_response_model": (retired.body or {}).get("model"),
                        "primary_models_per_doc": list(PRIMARY_MODELS), "configured_model_off_doc": drifted}
        missing = _endpoint_missing_reason(primary.status, "openai")
        if missing:
            return False, missing, observations, "blocked"
        if primary.status != 200:
            return False, f"配置的模型名 {model} 请求失败（HTTP {primary.status}）", observations
        if retired.status == 200:
            note = f"；注意：当前配置模型 {model} 不在官方主推名 {list(PRIMARY_MODELS)} 内（口径漂移，建议核对）" if drifted else ""
            return True, f"配置模型可调用，退休别名 {alias} 仍被接受（由最新 Flash 模型服务）{note}", observations
        if retired.status == 404:
            # 同一端点上配置模型请求成功，故 404 表示"模型/别名不存在"，不是端点缺失。
            return True, f"退休别名 {alias} 已不可用（HTTP 404 模型不存在；同端点配置模型正常，故非端点缺失），与文档「仍被接受」不符", observations, "warning"
        if _is_rejection(retired.status):
            return True, f"退休别名 {alias} 已被拒绝（HTTP {retired.status}）：文档称仍接受，属文档/线上差异", observations, "warning"
        return False, f"退休别名 {alias} 返回非预期状态 HTTP {retired.status}", observations

    async def case_effort_full_set(self, model: str):
        """思考强度取值域与关闭语义（对齐官方《Thinking Mode》）。

        官方两处文档口径不完全一致（2026-10-14 复核）：
        - 《Thinking Mode》控制参数表：OpenAI 形态的**开关**是 `{"thinking":{"type":"enabled/disabled"}}`，
          **effort** 写作 `{"reasoning_effort":"low/high/max"}`；其下映射表又列出 requested effort
          `minimal→low`、`medium→high`、`xhigh→high`、`ultra→max`（**含 minimal/ultra**）。
        - 《Create Chat Completion》API 参考：`reasoning_effort` 的 Possible values 为
          `none / low / high / max`（`none` 关闭思考；`minimal→low`、`medium/xhigh→high` 为兼容别名），
          **未提 ultra**。

        两处合取后的**无歧义开启档位是 low / high / max**；`medium / xhigh` 两处都认可（兼容别名）；
        `minimal` / `ultra` / `none` 存在文档分歧 ⇒ 本用例只对无歧义档位做正向断言，其余**仅作观察项**
        （被拒或被接受都不计入失败）。Chat 侧关思考按两处都认可的 `thinking={"type":"disabled"}`
        验证（DS-V4-03 亦覆盖）。原实现把 `minimal/ultra` 当"必须被接受"、把 `none` 当 Chat 侧关思考，
        均与官方文档冲突，已改。
        """
        official = ("low", "medium", "high", "xhigh", "max")
        rows: dict[str, Any] = {}
        responses: dict[str, ApiResponse] = {}
        for effort in official:
            r = await self.chat(model, [{"role": "user", "content": "计算 23*17，只给最终答案。"}],
                                thinking={"type": "enabled"}, reasoning_effort=effort, max_tokens=512)
            responses[effort] = r
            rows[effort] = {"status": r.status, "reasoning_chars": len(str(_message(r.body).get("reasoning_content") or ""))}
        off = await self.chat(model, [{"role": "user", "content": "计算 23*17，只给最终答案。"}],
                              thinking={"type": "disabled"}, max_tokens=512)
        off_chars = len(str(_message(off.body).get("reasoning_content") or ""))
        rows["thinking=disabled"] = {"status": off.status, "reasoning_chars": off_chars}
        undocumented: dict[str, int] = {}
        for effort in ("minimal", "ultra"):
            probe = await self.chat(model, [{"role": "user", "content": "只回复 OK。"}],
                                    thinking={"type": "enabled"}, reasoning_effort=effort, max_tokens=32)
            undocumented[effort] = probe.status
        rows["undocumented_efforts"] = undocumented

        if any(rows[e]["status"] == 404 for e in official):
            return False, _endpoint_missing_reason(404, "openai") or "端点不存在", rows, "blocked"
        rejected = {e: rows[e]["status"] for e in official if rows[e]["status"] != 200}
        if rejected:
            sample = responses[next(iter(rejected))]
            detail = f"官方文档列出的 effort 取值未被全部接受：{rejected}"
            issue = self._gateway_issue(sample, model=model)
            if issue:
                return False, f"{detail}{_gateway_note(issue)}", {**rows, "gateway_issue": issue.kind}, issue.status
            return False, detail, rows
        if off.status != 200:
            return self._gateway_verdict(off, f"thinking=disabled 未被接受（HTTP {off.status}）", rows, model=model)
        if off_chars > 0:
            return True, (f"官方取值域均被接受；但 thinking=disabled 仍返回 reasoning_content（{off_chars} 字符），"
                          "与文档「disabled 关闭思考」不符"), rows, "warning"
        return (True, "官方取值域 low/medium/high/xhigh/max 均被接受；thinking=disabled 正确关闭思考（reasoning_content 为空）"
                      f"；文档未列出的 minimal/ultra 观察结果={undocumented}（仅留痕）", rows)

    async def case_max_tokens_bounds(self, model: str):
        """max_tokens 文档范围为 1–384K(393216)：上限应接受，0 与越界应被拒。"""
        upper = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"}, max_tokens=393216)
        zero = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"}, max_tokens=0)
        over = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"}, max_tokens=393217)
        observations = {"upper_status": upper.status, "zero_status": zero.status, "over_status": over.status, "zero_error": _error_body(zero).get("message"), "over_error": _error_body(over).get("message")}
        missing = _endpoint_missing_reason(upper.status, "openai")
        if missing:
            return False, missing, observations, "blocked"
        if upper.status != 200:
            return False, f"文档上限 393216 未被接受（HTTP {upper.status}）：{_error_body(upper).get('message')}", observations
        rejects = (_is_rejection(zero.status), _is_rejection(over.status))
        if all(rejects):
            return True, "上限 393216 被接受，0 与 393217 均被 4xx 拒绝", observations
        lenient = [name for name, status, rejected in (("0", zero.status, rejects[0]), ("393217", over.status, rejects[1])) if not rejected]
        if lenient:
            return True, f"max_tokens={','.join(lenient)} 未被拒绝（HTTP {'/'.join(str(s) for s in (zero.status, over.status))}）：文档给出 1–393216 范围，属宽松放行", observations, "warning"
        return False, "max_tokens 边界返回非预期状态", observations

    async def case_invalid_api_key(self, model: str):
        """鉴权契约：无效 Key 应返回 401，且错误体含 message/type。"""
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"}, max_tokens=8,
                            _headers={"Authorization": "Bearer sk-invalid-dsv4-admission-probe"})
        err = _error_body(r)
        observations = {"http_status": r.status, "error_type": err.get("type"), "error_code": err.get("code"), "error_message": str(err.get("message"))[:200]}
        missing = _endpoint_missing_reason(r.status, "openai")
        if missing:
            return False, missing, observations, "blocked"
        if r.status == 401:
            if err.get("message"):
                return True, "无效 API Key 返回 401 且错误体包含 message", observations
            return True, "无效 API Key 返回 401，但错误体缺少 message 字段", observations, "warning"
        if r.status == 200:
            return False, "无效 API Key 竟返回 200：鉴权未生效（严重）", observations
        return False, f"无效 API Key 返回非预期状态 HTTP {r.status}（期望 401）", observations

    async def case_files_lifecycle(self, model: str):
        """Files API 全链路：上传 → 以 file_id 引用 → 删除。

        属破坏性用例（占用账号配额）：无论成败都在 finally 中尝试删除，并带 1 小时过期兜底。
        """
        png = png_bands()
        upload = await self.root_request("POST", "files", files={"file": ("dsv4-probe.png", png, "image/png")},
                                         data={"purpose": "user_data", "expires_after[anchor]": "created_at", "expires_after[seconds]": "3600"})
        file_id = str((upload.body or {}).get("id") or "")
        observations: dict[str, Any] = {"upload_status": upload.status, "file_id": file_id, "upload_response": (upload.text or "")[:300],
                                        "error": _error_message(upload)[:200]}
        if upload.status != 200 or not file_id.startswith("file-api-"):
            # 上传失败先过网关口径：账号未开通对象存储（5xx OptInRequired）、限流、端点缺失都属平台侧，
            # 记 blocked；只有真属协议/模型问题才判 failed。
            return self._gateway_verdict(upload, f"文件上传失败（HTTP {upload.status}，file_id={file_id or '空'}）",
                                         observations, family="root", model=model, capability=CAPABILITY_VISION)
        chat: ApiResponse | None = None
        deleted: ApiResponse | None = None
        try:
            prompt = "这张图上半部分和下半部分分别是什么颜色？只回答两个颜色词，用中文。"
            chat = await self.chat(model, [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "file", "file_id": file_id}]}],
                                   thinking={"type": "disabled"}, max_tokens=32)
        finally:
            deleted = await self.root_request("DELETE", f"files/{file_id}")
        content = str(_message((chat.body if chat else None)).get("content") or "")
        observations.update({"chat_status": chat.status if chat else None, "content": content[:200],
                             "delete_status": deleted.status if deleted else None, "deleted_flag": (deleted.body or {}).get("deleted") if deleted else None})
        if chat is None or chat.status != 200:
            if chat is None:
                return False, "以 file_id 引用上传文件失败（无响应）", observations
            return self._gateway_verdict(chat, f"以 file_id 引用上传文件失败（HTTP {chat.status}）",
                                         observations, model=model, capability=CAPABILITY_VISION)
        if not ("红" in content and "蓝" in content):
            if not model_supports_capability(model, CAPABILITY_VISION):
                return True, (f"file_id 引用返回 200 但未识别图片内容；该模型未声明支持视觉（官方归属 deepseek-flash），"
                              f"不判模型失败：{content[:120]}"), observations, "warning"
            return False, f"file_id 引用返回 200 但未识别图片内容：{content[:120]}", observations
        if deleted and deleted.status == 200 and (deleted.body or {}).get("deleted") is True:
            return True, "上传 → file_id 引用识别 → 删除全链路通过，且返回 deleted=true", observations
        return True, f"上传与引用识别通过，但删除返回异常（HTTP {deleted.status if deleted else 'n/a'}），文件可能残留并占用配额", observations, "warning"


    async def case_image_wrong_role(self, model: str):
        """文档承诺：图片只允许出现在 user 消息中，放在 system/assistant 返回 400。

        可判定性前提：只有当该模型/网关**真的接受** image_url 内容块时，"role 放错被拒"才有意义。
        在整体不收图片的被测面上，这条负向断言拿到的 4xx 与 role 无关 —— 直接记 blocked，
        否则会产出一条**假通过**（DS-V4-42 在千帆上就是这样拿到的 400）。
        """
        vision_capable = model_supports_capability(model, CAPABILITY_VISION)
        r = await self.chat(model, [{"role": "system", "content": [{"type": "text", "text": "你是助手"}, image_part(png_data_url())]},
                                    {"role": "user", "content": "描述这张图。"}], thinking={"type": "disabled"}, max_tokens=32)
        observations = {"http_status": r.status, "content": str(_message(r.body).get("content") or "")[:150],
                        "error": _error_message(r)[:200], "vision_capable_model": vision_capable}
        if not vision_capable:
            return False, (f"被测模型 {model} 未声明支持视觉（官方归属 deepseek-flash），该模型上图片整体不被接受"
                           f"（HTTP {r.status}）：无法区分拒绝原因是否为「role 放错」，本用例在该面上不可验证"), observations, "blocked"
        if _is_rejection(r.status):
            # 假通过防护：若 4xx 的文案说明「该面上图片整体不被接受」，则无法归因到「role 放错」，
            # 此时记 passed 就是假通过（真实批次里 DS-V4-42 曾拿到与 DS-V4-36 完全相同的 400）。
            # 例外：错误文案**明确点名放置位置**（如千帆对 v4.1-flash 返回
            # "Images in system messages are not supported."）⇒ 拒绝可归因到 role 放错，
            # 正是文档承诺的负向行为，应判 passed。
            err_text = _error_message(r).lower()
            role_named = any(k in err_text for k in ("system message", "assistant message", "user message"))
            if role_named:
                return True, (f"图片放在 system 消息中被 4xx 拒绝（HTTP {r.status}），错误明确指向放置位置"
                              f"（{_error_message(r)[:120]}），符合文档承诺"), observations
            issue = self._gateway_issue(r, model=model, capability=CAPABILITY_VISION)
            if issue and issue.kind in ("capability_not_in_model", "capability_unsupported_by_gateway"):
                return False, (f"图片放在 system 消息中返回 HTTP {r.status}，但错误表明图片在该面上整体不被接受"
                               f"（{_error_message(r)[:120]}）：无法归因到「role 放错」⇒ 该负向断言不可判定，"
                               "记 blocked 以免假通过"), observations, "blocked"
            return True, f"图片放在 system 消息中被 4xx 拒绝（HTTP {r.status}），符合文档承诺", observations
        if r.status == 200:
            return True, "图片放在 system 消息中未被拒绝（HTTP 200）：文档明确承诺返回 400，属宽松放行，模型可能忽略了该图片", observations, "warning"
        return self._gateway_verdict(r, f"图片放在 system 消息中返回非预期状态 HTTP {r.status}", observations, model=model, capability=CAPABILITY_VISION)

    async def case_models_listing(self, model: str):
        r = await self.root_request("GET", "models")
        body = r.body or {}
        data = body.get("data") if isinstance(body.get("data"), list) else []
        ids = [str(x.get("id")) for x in data if isinstance(x, dict)]
        fields_ok = all(isinstance(x.get("id"), str) and "object" in x for x in data if isinstance(x, dict))
        family_token = model.split("-")[0]
        observations = {"http_status": r.status, "object": body.get("object"), "model_ids": ids[:20],
                        "configured_model_listed": model in ids, "family_token_listed": any(family_token in i for i in ids)}
        if r.status == 200 and body.get("object") == "list" and data and fields_ok:
            if not any(PRIMARY_MODELS[1] in i or "flash" in i.lower() for i in ids):
                return True, f"模型列表结构正确，但未出现 Flash 系列模型名（返回 {ids[:8]}），与文档主推口径存在差异", observations, "warning"
            return True, f"GET /models 返回 {len(ids)} 个模型，object=list 且条目字段完整", observations
        # 404 端点缺失 / 429 限流 / 5xx 基础设施都归网关口径（blocked），不污染成模型失败。
        return self._gateway_verdict(r, f"模型列表响应结构不符合预期（HTTP {r.status}，object={body.get('object')}，条目数={len(data)}）",
                                     observations, family="root", model=model)

    async def case_stream_options_without_stream(self, model: str):
        """文档明确：stream_options 必须与 stream=true 同用，否则返回 400。"""
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"},
                            stream_options={"include_usage": True}, max_tokens=16)
        observations = {"http_status": r.status, "error": str(_error_body(r).get("message"))[:200]}
        if r.status == 404:
            return False, _endpoint_missing_reason(404, "openai") or "端点不存在", observations, "blocked"
        if _is_rejection(r.status):
            return True, f"未设置 stream 却传 stream_options 被拒绝（HTTP {r.status}），符合文档承诺", observations
        if r.status == 200:
            return True, "未设置 stream 却传 stream_options 未被拒绝（HTTP 200）：文档承诺返回 400，属宽松放行", observations, "warning"
        return False, f"stream_options 缺少 stream 时返回非预期状态 HTTP {r.status}", observations

    async def case_logprobs_structure(self, model: str):
        """logprobs/top_logprobs：非思考模式应返回对数概率；思考模式按文档不生效但不报错。"""
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"},
                            logprobs=True, top_logprobs=5, max_tokens=16)
        choice = ((r.body or {}).get("choices") or [{}])[0]
        content = ((choice.get("logprobs") or {}).get("content") or [])
        samples = [c for c in content if isinstance(c, dict)]
        has_tokens = bool(samples) and all("token" in c and "logprob" in c for c in samples[:3])
        thinking = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "enabled"},
                                   logprobs=True, top_logprobs=5, max_tokens=16)
        thinking_lp = ((((thinking.body or {}).get("choices") or [{}])[0]).get("logprobs") or {}).get("content") or []
        observations = {"http_status": r.status, "logprob_items": len(samples), "sample": samples[:1],
                        "thinking_status": thinking.status, "thinking_logprob_items": len(thinking_lp),
                        "error": _error_message(r)[:200]}
        if r.status != 200:
            return self._gateway_verdict(r, f"非思考模式 logprobs 请求失败（HTTP {r.status}）：{_error_message(r)[:150]}",
                                         observations, model=model)
        if not has_tokens:
            # 官方 Chat Completions 支持 logprobs；网关不透传 token 级对数概率时，功能可用性不受影响
            # （属能力/实现差异），按中转口径记 warning 而非 failed。
            return True, (f"logprobs=true 但未返回 token 级对数概率（items={len(samples)}）：官方文档承诺返回，"
                          "该网关不透传，不影响模型正常使用，记为差异"), observations, "warning"
        if thinking.status != 200:
            return True, f"非思考模式对数概率正常（{len(samples)} 项）；思考模式下传 logprobs 返回 HTTP {thinking.status}，文档称「不报错但不生效」，属线上差异", observations, "warning"
        return True, f"非思考模式返回 {len(samples)} 项 token 级对数概率；思考模式下按文档不生效且不报错（items={len(thinking_lp)}）", observations

    async def case_top_p_bounds(self, model: str):
        """top_p 越界应拒；思考模式下限 0.95，非思考模式恒 1.0（均表现为不报错）。"""
        over = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"}, top_p=1.2, max_tokens=16)
        thinking = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "enabled"}, top_p=0.5, max_tokens=16)
        plain = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"}, top_p=0.5, max_tokens=16)
        observations = {"over_status": over.status, "thinking_status": thinking.status, "plain_status": plain.status, "over_error": str(_error_body(over).get("message"))[:200]}
        if over.status == 404:
            return False, _endpoint_missing_reason(404, "openai") or "端点不存在", observations, "blocked"
        if _is_rejection(over.status):
            if thinking.status == 200 and plain.status == 200:
                return True, f"top_p=1.2 被 4xx 拒绝；top_p=0.5 在思考（按下限 0.95 处理）与非思考（恒 1.0 忽略）模式下均被接受", observations
            return False, f"top_p 合法值被拒（思考 {thinking.status} / 非思考 {plain.status}）", observations
        if over.status == 200:
            return True, "top_p=1.2 超出文档上限 1 但未被拒绝（HTTP 200），属宽松放行", observations, "warning"
        return False, f"top_p 越界返回非预期状态 HTTP {over.status}", observations

    async def case_fim_thinking_conflict(self, model: str):
        r = await self.request("completions", {"model": model, "prompt": "def fib(n):\n", "suffix": "\n    return fib(n-1)+fib(n-2)", "max_tokens": 32, "thinking": {"type": "enabled"}}, family="beta")
        observations = {"http_status": r.status, "error": str(_error_body(r).get("message"))[:200]}
        if r.status == 404:
            return False, _endpoint_missing_reason(404, "beta") or "beta 端点不存在", observations, "blocked"
        if _is_rejection(r.status):
            return True, f"FIM 携带思考模式被 4xx 拒绝（HTTP {r.status}），符合文档「FIM 仅支持非思考模式」", observations
        if r.status == 200:
            return True, "FIM 携带思考模式未被拒绝（HTTP 200）：文档称 FIM 仅支持非思考模式，属宽松放行（thinking 可能被静默忽略）", observations, "warning"
        return False, f"FIM + thinking 返回非预期状态 HTTP {r.status}", observations

    async def case_anthropic_system_stop(self, model: str):
        marker = "<END>"
        payload = {"model": model, "max_tokens": 64, "system": "你只输出用户指定的字符串，不要解释、不要加引号。",
                   "messages": [{"role": "user", "content": [{"type": "text", "text": f"严格输出 ABC{marker}XYZ"}]}],
                   "stop_sequences": [marker]}
        r = await self.request("v1/messages", payload, family="anthropic")
        text = _anthropic_text(r)
        observations = {"http_status": r.status, "block_types": _anthropic_block_types(r), "text": text[:200], "stop_reason": (r.body or {}).get("stop_reason")}
        if r.status == 404:
            return False, _endpoint_missing_reason(404, "anthropic") or "anthropic 端点不存在", observations, "blocked"
        if r.status != 200:
            return False, f"Anthropic system/stop_sequences 请求失败（HTTP {r.status}）：{str(_error_body(r).get('message'))[:150]}", observations
        if marker in text or "XYZ" in text:
            return False, f"stop_sequences 未生效：输出仍包含停止串或其后续内容（{text[:80]}）", observations
        return True, "Anthropic 顶层 system 生效且 stop_sequences 命中后正确截断", observations

    async def case_anthropic_thinking(self, model: str):
        payload = {"model": model, "max_tokens": 1024,
                   "thinking": {"type": "enabled", "budget_tokens": 1024},
                   "output_config": {"effort": "high"},
                   "messages": [{"role": "user", "content": [{"type": "text", "text": "计算 27*31，只给答案。"}]}]}
        r = await self.request("v1/messages", payload, family="anthropic")
        types = _anthropic_block_types(r)
        observations = {"http_status": r.status, "block_types": types, "error": str(_error_body(r).get("message"))[:200]}
        if r.status == 404:
            return False, _endpoint_missing_reason(404, "anthropic") or "anthropic 端点不存在", observations, "blocked"
        if r.status != 200:
            return False, f"Anthropic thinking/output_config 请求失败（HTTP {r.status}）：budget_tokens 应被忽略且 thinking 受支持", observations
        if "thinking" in types:
            return True, "Anthropic thinking 块正常返回，budget_tokens 被忽略、output_config.effort 被接受", observations
        return True, "请求成功但响应未包含 thinking 块（文档称 thinking 受支持，仅 budget_tokens 被忽略）", observations, "warning"

    async def case_responses_stream(self, model: str):
        """Responses 流式是语义事件流：event + sequence_number，结束于 response.completed，无 [DONE]。"""
        r = await self.request("responses", {"model": model, "input": "只回复 OK", "max_output_tokens": 32, "stream": True}, stream=True)
        info = r.stream or {}
        events = [e for e in (info.get("received_events") or []) if isinstance(e, dict)]
        types = [str(e.get("type")) for e in events]
        seqs = [e.get("sequence_number") for e in events if isinstance(e.get("sequence_number"), int)]
        monotonic = bool(seqs) and all(b > a for a, b in zip(seqs, seqs[1:]))
        text = "".join(str(e.get("delta") or "") for e in events if e.get("type") == "response.output_text.delta")
        raw = info.get("raw_sse_events") or []
        observations = {"http_status": r.status, "event_types": types[:12], "event_count": len(events),
                        "sequence_monotonic": monotonic, "text": text[:120], "has_done_marker": "[DONE]" in raw}
        if r.status == 404:
            return False, _endpoint_missing_reason(404, "openai") or "Responses 端点不存在", observations, "blocked"
        if r.status != 200:
            return False, f"Responses 流式请求失败（HTTP {r.status}）", observations
        if "response.failed" in types:
            return False, f"流式以 response.failed 结束：{str(events[-1].get('response', {}).get('error'))[:150]}", observations
        problems: list[str] = []
        if "response.created" not in types:
            problems.append("缺少首事件 response.created")
        if "response.completed" not in types:
            problems.append(f"未以 response.completed 结束（实际 {types[-1] if types else '无事件'}）")
        if not monotonic:
            problems.append("sequence_number 非单调递增")
        if "[DONE]" in raw:
            problems.append("出现 [DONE]（Responses 规范无该结束标记）")
        if not text:
            problems.append("未收到 response.output_text.delta 文本")
        if problems:
            return False, "；".join(problems), observations
        return True, f"语义事件流正常：{len(events)} 个事件、sequence_number 单调递增、以 response.completed 收尾且无 [DONE]", observations

    async def case_responses_ignored_fields(self, model: str):
        """文档：不支持的顶层参数一律静默忽略，响应固定 store=false / previous_response_id=null / parallel_tool_calls=true。"""
        ignored = {"previous_response_id": "resp_nonexistent", "conversation": "conv_dsv4", "store": True,
                   "metadata": {"probe": "dsv4"}, "include": ["reasoning.encrypted_content"],
                   "prompt": {"id": "pmt_dsv4"}, "truncation": "auto", "service_tier": "auto",
                   "stream_options": {"include_usage": True}, "prompt_cache_key": "dsv4-key",
                   "prompt_cache_retention": "24h", "background": True, "safety_identifier": "dsv4",
                   "max_tool_calls": 3, "context_management": {"type": "auto"}}
        r = await self.request("responses", {"model": model, "input": "只回复 OK", "max_output_tokens": 32, **ignored})
        body = r.body or {}
        observations = {"http_status": r.status, "store": body.get("store"), "previous_response_id": body.get("previous_response_id"),
                        "parallel_tool_calls": body.get("parallel_tool_calls"), "injected_fields": sorted(ignored), "error": str(_error_body(r).get("message"))[:200]}
        if r.status in (404, 429) or r.status >= 500:
            return self._gateway_verdict(r, f"Responses 不支持字段注入请求返回 HTTP {r.status}", observations, model=model)
        if r.status != 200:
            return True, (f"网关未按官方承诺静默忽略不支持的字段（HTTP {r.status}）：{_error_message(r)[:150]}；"
                          "属更严格的实现差异，不影响正常使用"), observations, "warning"
        fixed_ok = body.get("store") is False and body.get("previous_response_id") is None and body.get("parallel_tool_calls") is True
        if fixed_ok:
            return True, f"{len(ignored)} 个不支持字段全部被静默忽略，固定值 store=false / previous_response_id=null / parallel_tool_calls=true", observations
        return False, f"不支持字段虽未报错，但响应固定值不符：store={body.get('store')}、previous_response_id={body.get('previous_response_id')}、parallel_tool_calls={body.get('parallel_tool_calls')}", observations

    async def case_parallel_tool_calls(self, model: str):
        tools = self.tools() + [{"type": "function", "function": {"name": "get_text", "description": "返回测试文本", "parameters": {"type": "object", "properties": {}, "required": []}}}]
        r = await self.chat(model, [{"role": "user", "content": "必须一次性同时调用 get_number 和 get_text 两个工具，不要只调用其中一个。"}],
                            tools=tools, tool_choice="required", thinking={"type": "disabled"}, max_tokens=256)
        calls = _message(r.body).get("tool_calls") or []
        names = [str((c.get("function") or {}).get("name")) for c in calls]
        observations = {"http_status": r.status, "tool_call_count": len(calls), "names": names}
        if r.status == 404:
            return False, _endpoint_missing_reason(404, "openai") or "端点不存在", observations, "blocked"
        if r.status != 200:
            return False, f"并行工具调用请求失败（HTTP {r.status}）", observations
        if len(calls) >= 2:
            return True, f"单轮返回 {len(calls)} 个工具调用（{names}），并行工具调用可用", observations
        return True, f"单轮仅返回 {len(calls)} 个工具调用（{names}）：chat 端点未承诺并行调用，仅作观测", observations, "warning"

    async def case_anthropic_basic(self, model: str):
        payload = {"model": model, "max_tokens": 32, "messages": [{"role": "user", "content": "只回复 OK"}]}
        r = await self.request("v1/messages", payload, family="anthropic", headers={"x-api-key": self.config.api_key, "anthropic-version": "2023-06-01"})
        body = r.body or {}
        observations = {"role": body.get("role"), "model": body.get("model"), "http_status": r.status}
        missing = _endpoint_missing_reason(r.status, "anthropic")
        if missing:
            return False, missing, observations, "blocked"
        return r.status == 200 and body.get("role") == "assistant" and isinstance(body.get("content"), list), "Anthropic Messages 应返回 assistant 与 content 数组", observations

    async def case_image_detail_levels(self, model: str):
        """detail 档位（官方：low 降采样到 512×512；high / original / auto 保留原图）。

        同 DS-V4-36：官方把图片输入归属 `deepseek-flash`（V4.1-Flash-1017），在不提供视觉
        能力面的被测模型（`deepseek-v4-pro`、视觉开放前的 `deepseek-v4-flash-0731`）上本用例
        不可验证 ⇒ blocked/warning，不把"模型不收图/不会看图"读成"detail 不支持"。
        """
        url = png_data_url()
        prompt = "这张图上半部分和下半部分分别是什么颜色？只回答两个颜色词，用中文。"
        vision_capable = model_supports_capability(model, CAPABILITY_VISION)
        rows: dict[str, Any] = {}
        responses: list[ApiResponse] = []
        for detail in ("low", "high", "original", "auto"):
            r = await self.chat(model, [{"role": "user", "content": [{"type": "text", "text": prompt}, image_part(url, detail)]}],
                                thinking={"type": "disabled"}, max_tokens=32)
            responses.append(r)
            content = str(_message(r.body).get("content") or "")
            rows[detail] = {"status": r.status, "recognized": "红" in content and "蓝" in content}
        rows["vision_capable_model"] = vision_capable
        rejected = {k: v["status"] for k, v in rows.items() if isinstance(v, dict) and v["status"] != 200}
        if rejected:
            detail_text = f"以下 detail 档位未被接受：{rejected}"
            if not vision_capable:
                return False, (f"{detail_text}；官方文档把图片输入归属 deepseek-flash，被测模型 {model} 未声明支持视觉"
                               " ⇒ 该能力面在本模型上不可验证"), rows, "blocked"
            return self._gateway_verdict(next(r for r in responses if r.status != 200), detail_text, rows,
                                         model=model, capability=CAPABILITY_VISION)
        unrecognized = [k for k, v in rows.items() if isinstance(v, dict) and not v["recognized"]]
        if unrecognized:
            if not vision_capable:
                return True, f"detail={'/'.join(unrecognized)} 返回 200 但未识别图像内容；该模型未声明支持视觉，不判模型失败", rows, "warning"
            return False, f"detail={'/'.join(unrecognized)} 返回 200 但未识别图像内容", rows
        return True, "detail=low/high/original/auto 四档均被接受且都能正确识别图像内容", rows

    async def case_files_listing(self, model: str):
        listing = await self.root_request("GET", "files", params={"limit": 5})
        body = listing.body or {}
        data = body.get("data") if isinstance(body.get("data"), list) else []
        items_ok = all(isinstance(x.get("id"), str) and "filename" in x and "purpose" in x for x in data if isinstance(x, dict))
        observations: dict[str, Any] = {"list_status": listing.status, "object": body.get("object"), "count": len(data),
                                        "has_more": body.get("has_more"), "first_item": data[:1]}
        if listing.status != 200 or body.get("object") != "list" or not items_ok:
            return self._gateway_verdict(listing, f"文件列表结构不符合预期（HTTP {listing.status}，object={body.get('object')}）",
                                         observations, family="root", model=model)
        if not data:
            return True, "文件列表可访问且结构正确（当前账号无已上传文件，未校验单文件检索）", observations
        file_id = str(data[0].get("id"))
        detail = await self.root_request("GET", f"files/{file_id}")
        detail_body = detail.body or {}
        observations.update({"retrieve_status": detail.status, "retrieve_fields": sorted(detail_body)})
        if detail.status == 200 and detail_body.get("id") == file_id and "bytes" in detail_body:
            return True, f"列表返回 {len(data)} 个文件且字段完整；单文件检索也返回正确记录", observations
        return self._gateway_verdict(detail, f"单文件检索不符合预期（HTTP {detail.status}，字段={sorted(detail_body)}）",
                                     observations, family="root", model=model)

    async def case_user_balance(self, model: str):
        r = await self.root_request("GET", "user/balance")
        body = r.body or {}
        infos = body.get("balance_infos") if isinstance(body.get("balance_infos"), list) else []
        required = {"currency", "total_balance", "granted_balance", "topped_up_balance"}
        infos_ok = all(required.issubset(set(x)) for x in infos if isinstance(x, dict)) if infos else False
        observations = {"http_status": r.status, "is_available": body.get("is_available"),
                        "currencies": [x.get("currency") for x in infos if isinstance(x, dict)],
                        "error": _error_message(r)[:200]}
        if r.status != 200:
            return self._gateway_verdict(r, f"余额端点返回非预期状态 HTTP {r.status}", observations, family="root", model=model)
        if not isinstance(body.get("is_available"), bool):
            return False, f"余额响应缺少布尔字段 is_available（{body.get('is_available')}）", observations
        if not infos_ok:
            return False, f"balance_infos 结构不完整（需要 {sorted(required)}）：{infos[:1]}", observations
        if body.get("is_available") is False:
            return True, "余额端点可用，但账号余额不足（is_available=false）——准入前需先充值", observations, "warning"
        return True, f"余额端点返回 is_available=true 且 balance_infos 字段完整（币种 {observations['currencies']}）", observations

    async def case_stop_multi_sequences(self, model: str):
        markers = ["<A>", "<B>", "<C>"]
        r = await self.chat(model, [{"role": "user", "content": "严格按顺序输出：AB<A>Q1<B>Q2<C>Q3"}],
                            thinking={"type": "disabled"}, stop=markers, max_tokens=64)
        content = str(_message(r.body).get("content") or "")
        over = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"},
                               stop=[f"<S{i}>" for i in range(17)], max_tokens=16)
        observations = {"multi_status": r.status, "content": content, "over_limit_status": over.status,
                        "over_limit_error": str(_error_body(over).get("message"))[:200]}
        if r.status == 404:
            return False, _endpoint_missing_reason(404, "openai") or "端点不存在", observations, "blocked"
        if r.status != 200:
            return False, f"多序列 stop 请求失败（HTTP {r.status}）", observations
        hit = [m for m in markers if m in content]
        leaked = "Q1" in content or "Q2" in content or "Q3" in content
        if hit or leaked:
            return False, f"多序列 stop 未生效（命中串={hit}，后续内容泄漏={leaked}）：{content[:120]}", observations
        if _is_rejection(over.status):
            return True, f"多序列 stop 命中即截断；超过 16 条（17 条）被 4xx 拒绝（HTTP {over.status}），符合文档上限", observations
        if over.status == 200:
            return True, "多序列 stop 命中即截断；17 个停止序列未被拒绝（文档仅写 up to 16，属宽松放行）", observations, "warning"
        return False, f"超过 16 条停止序列时返回非预期状态 HTTP {over.status}", observations

    async def case_deprecated_penalties(self, model: str):
        """文档：frequency_penalty / presence_penalty 已废弃，传入不产生任何效果（也不应报错）。"""
        prompt = "只回复 OK"
        base = await self.chat(model, [{"role": "user", "content": prompt}], thinking={"type": "disabled"}, max_tokens=16)
        penalized = await self.chat(model, [{"role": "user", "content": prompt}], thinking={"type": "disabled"},
                                    max_tokens=16, frequency_penalty=1.8, presence_penalty=-1.8)
        observations = {"baseline_status": base.status, "penalized_status": penalized.status,
                        "baseline_content": str(_message(base.body).get("content") or "")[:40],
                        "penalized_content": str(_message(penalized.body).get("content") or "")[:40]}
        if penalized.status == 404:
            return False, _endpoint_missing_reason(404, "openai") or "端点不存在", observations, "blocked"
        if penalized.status == 200 and base.status == 200:
            return True, "已废弃的 frequency_penalty / presence_penalty 在非思考模式下被静默接受（按文档不产生效果）", observations
        if _is_rejection(penalized.status):
            return True, f"已废弃的采样参数被 4xx 拒绝（HTTP {penalized.status}）：文档称传入不报错、无效果，属线上差异", observations, "warning"
        return False, f"已废弃采样参数返回非预期状态 HTTP {penalized.status}", observations

    async def case_fim_json_conflict(self, model: str):
        r = await self.request("completions", {"model": model, "prompt": "def fib(n):\n", "suffix": "\n    return fib(n-1)+fib(n-2)",
                                              "max_tokens": 32, "response_format": {"type": "json_object"}}, family="beta")
        observations = {"http_status": r.status, "error": str(_error_body(r).get("message"))[:200]}
        if r.status == 404:
            return False, _endpoint_missing_reason(404, "beta") or "beta 端点不存在", observations, "blocked"
        if _is_rejection(r.status):
            return True, f"FIM 与 JSON mode 同时使用被 4xx 拒绝（HTTP {r.status}），符合文档「二者不支持同时使用」", observations
        if r.status == 200:
            return True, "FIM 与 JSON mode 同时使用未被拒绝（HTTP 200）：文档称二者不支持同时使用，属宽松放行", observations, "warning"
        return False, f"FIM + JSON mode 返回非预期状态 HTTP {r.status}", observations

    async def case_prefix_without_flag(self, model: str):
        """文档 422 场景之一：期望前缀续写但最后一条 assistant 消息未设 prefix=true。"""
        r = await self.chat(model, [{"role": "user", "content": "写一个 Python 加法函数，只输出代码。"},
                                    {"role": "assistant", "content": "def add(a, b):\n"}], _family="beta", thinking={"type": "disabled"}, max_tokens=32)
        content = str(_message(r.body).get("content") or "")
        observations = {"http_status": r.status, "content": content[:200], "error": str(_error_body(r).get("message"))[:200]}
        if r.status == 404:
            return False, _endpoint_missing_reason(404, "beta") or "beta 端点不存在", observations, "blocked"
        if _is_rejection(r.status):
            return True, f"末条 assistant 消息缺少 prefix=true 时被 4xx 拒绝（HTTP {r.status}），符合文档 422 场景", observations
        if r.status == 200:
            return True, "末条 assistant 消息缺少 prefix=true 时未被拒绝（HTTP 200）：文档未承诺拒绝（属调用方义务），记为观测", observations, "warning"
        return False, f"prefix 缺少标志时返回非预期状态 HTTP {r.status}", observations

    async def case_anthropic_tool_loop(self, model: str):
        tool = {"name": "lookup_order", "description": "查询订单状态",
                "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}
        first = await self.request("v1/messages", {"model": model, "max_tokens": 256, "tools": [tool],
                                                    "tool_choice": {"type": "any"},
                                                    "messages": [{"role": "user", "content": [{"type": "text", "text": "查询订单 ORD-731 的状态，必须调用工具。"}]}]}, family="anthropic")
        blocks = (first.body or {}).get("content") or []
        tool_uses = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        observations: dict[str, Any] = {"first_status": first.status, "block_types": _anthropic_block_types(first), "tool_use_count": len(tool_uses)}
        if first.status == 404:
            return False, _endpoint_missing_reason(404, "anthropic") or "anthropic 端点不存在", observations, "blocked"
        if first.status != 200:
            return False, f"Anthropic tool_choice=any 请求失败（HTTP {first.status}）：{str(_error_body(first).get('message'))[:150]}", observations
        if not tool_uses:
            return False, f"tool_choice=any 未产生 tool_use 块（块类型 {observations['block_types']}）", observations
        call = tool_uses[0]
        observations["tool_name"] = call.get("name")
        if call.get("name") != "lookup_order":
            return False, f"tool_use 名称不符：{call.get('name')}", observations
        second = await self.request("v1/messages", {"model": model, "max_tokens": 256, "tools": [tool],
                                                     "messages": [{"role": "user", "content": [{"type": "text", "text": "查询订单 ORD-731 的状态，必须调用工具。"}]},
                                                                  {"role": "assistant", "content": blocks},
                                                                  {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call.get("id"), "content": "已发货"}]}]}, family="anthropic")
        text = _anthropic_text(second)
        observations.update({"second_status": second.status, "second_blocks": _anthropic_block_types(second), "text": text[:120]})
        if second.status == 200 and text:
            return True, "Anthropic tool_choice=any 触发 tool_use，tool_result 回传后模型正常作答（闭环成立）", observations
        return False, f"工具结果回传后未正常作答（HTTP {second.status}，块类型 {observations['second_blocks']}）", observations

    # Responses 侧工具矩阵。
    #
    # 官方文档两处关键口径：
    #   1) 《Thinking Mode》控制参数表：Responses 形态的思考开关是
    #      `{"reasoning": {"effort": "none/low/high/max"}}`，**`none` 关闭思考模式**；
    #      思考默认开启，默认档位 high。
    #   2) 《Chat Completions API》tool_choice 段：「思考模式下不支持 required 和指定具体 tool 的用法，
    #      API 会返回 400 错误。请先关闭思考模式。」
    #
    # 即「默认思考模式下 required/指定工具被 4xx 拒绝」是**文档承诺的负向行为**，
    # 「先关闭思考再强制调用」是**文档给出的解法**。因此本用例做两面对称验证，而不是把 400 当成缺陷：
    #   - 负向（文档承诺）：默认思考模式下 required 与指定工具均应 4xx（放宽记 warning）；
    #   - 正向（文档解法）：`reasoning.effort=none` 关闭思考后 required 应稳定触发 function_call。
    # 与 Chat 侧 DS-V4-33「关思考 + required」同一口径，不因该限制本身记 warning。
    #
    # 2026-09-24 15:10 真实批次证据（results/DS_V4_P2-20260924-151006）：
    #   required + 默认思考模式 → 400 "Thinking mode does not support this tool_choice"
    #   required + reasoning.effort=none → 200 且 output=['function_call']（响应回显 reasoning.effort=none）
    RESPONSES_TOOL_PRIMING_PROMPTS = (
        "必须调用 get_number 获取测试数字，不要自行回答。",
        "禁止直接回答。请先调用 get_number 工具获取测试数字，再基于工具返回结果作答。",
    )
    # 思考模式下推理内容会占用 max_output_tokens；给足额度，避免 function_call 被截断成假失败
    # （DS-V4-34 用 32 时 output 只剩一个 reasoning 项且 status=incomplete）。
    RESPONSES_TOOL_MAX_OUTPUT = 1024
    # Responses 形态的思考开关：effort=none 即关闭思考（文档《Thinking Mode》控制参数表）。
    RESPONSES_THINKING_OFF = {"effort": "none"}

    @staticmethod
    def _responses_output_types(r: ApiResponse) -> list[str]:
        return [str(i.get("type")) for i in ((r.body or {}).get("output") or []) if isinstance(i, dict)]

    @classmethod
    def _responses_function_call(cls, r: ApiResponse) -> bool:
        return r.status == 200 and "function_call" in cls._responses_output_types(r)

    async def case_responses_tool_matrix(self, model: str):
        fn_tool = {"type": "function", "name": "get_number", "description": "返回测试数字", "parameters": {"type": "object", "properties": {}, "required": []}}
        pinned_choice = {"type": "function", "name": "get_number"}
        base = {"model": model, "input": self.RESPONSES_TOOL_PRIMING_PROMPTS[0],
                "max_output_tokens": self.RESPONSES_TOOL_MAX_OUTPUT, "tools": [fn_tool]}

        # 负向 1/2：默认（思考）模式下 required 与指定工具都应被 4xx 拒绝（文档承诺）
        default_required = await self.request("responses", {**base, "tool_choice": "required"})
        if default_required.status == 404:
            return False, _endpoint_missing_reason(404, "openai") or "Responses 端点不存在", {"required_status": 404}, "blocked"
        default_pinned = await self.request("responses", {**base, "tool_choice": pinned_choice})

        # 正向：关闭思考（reasoning.effort=none）后 required 应强制触发 function_call
        forced = await self.request("responses", {**base, "tool_choice": "required", "reasoning": self.RESPONSES_THINKING_OFF})
        ladder: list[tuple[str, ApiResponse]] = [("required+effort=none", forced)]
        trigger = "required+effort=none" if self._responses_function_call(forced) else None

        # 兜底 1：指定工具 + 关思考
        if trigger is None:
            fallback_pinned = await self.request("responses", {**base, "tool_choice": pinned_choice, "reasoning": self.RESPONSES_THINKING_OFF})
            ladder.append(("指定工具+effort=none", fallback_pinned))
            if self._responses_function_call(fallback_pinned):
                trigger = "指定工具+effort=none"
        # 兜底 2：auto + 强提示诱导（与 Chat 侧 DS-V4-08 同一手法，两轮递增）
        if trigger is None:
            for prompt in self.RESPONSES_TOOL_PRIMING_PROMPTS:
                auto = await self.request("responses", {**base, "input": prompt, "tool_choice": "auto"})
                ladder.append(("auto+强提示", auto))
                if self._responses_function_call(auto):
                    trigger = "auto+强提示"
                    break

        # 矩阵其余两格：custom 非 apply_patch 必须被拒绝；内置工具必须被静默忽略（都不依赖思考开关）
        bad = await self.request("responses", {"model": model, "input": "执行 patch", "max_output_tokens": 64,
                                                "tools": [{"type": "custom", "name": "not_apply_patch"}]})
        ignored = await self.request("responses", {"model": model, "input": "只回复 OK",
                                                    "max_output_tokens": self.RESPONSES_TOOL_MAX_OUTPUT,
                                                    "tools": [{"type": "web_search"}]})
        ignored_types = self._responses_output_types(ignored)
        observations = {"tool_choice_ladder": {name: r.status for name, r in ladder}, "triggered_by": trigger,
                        "default_required": {"status": default_required.status,
                                             "output_types": self._responses_output_types(default_required),
                                             "error": str(_error_body(default_required).get("message"))[:200]},
                        "default_pinned": {"status": default_pinned.status,
                                           "output_types": self._responses_output_types(default_pinned),
                                           "error": str(_error_body(default_pinned).get("message"))[:200]},
                        "custom_bad_status": bad.status, "custom_bad_error": str(_error_body(bad).get("message"))[:200],
                        "builtin_ignored_status": ignored.status, "builtin_ignored_output_types": ignored_types}

        problems: list[str] = []
        notes: list[str] = []
        # 文档承诺「思考模式下 required / 指定工具 → 400」。线上若未按承诺拒绝，按项目既有口径记 warning
        # （先例 DS-V4-58/59：文档承诺的 4xx 没被执行属"文档/线上差异"，不升级为 failed）。
        # 但要区分两种 200：直接触发（能力具备，只是没报错）vs 静默忽略（强制调用失效）。
        for label, probe in (("required", default_required), ("指定工具", default_pinned)):
            if self._responses_function_call(probe):
                notes.append(f"默认（思考）模式下 tool_choice={label} 已被上游直接放行并触发 function_call，"
                             "未按文档返回 400——文档/线上差异，能力本身具备")
            elif probe.status == 200:
                notes.append(f"默认（思考）模式下 tool_choice={label} 未按文档返回 4xx，且未触发 function_call"
                             f"（HTTP 200，output {self._responses_output_types(probe)}）：强制调用被静默忽略"
                             "——文档/线上差异，需人工判断影响面")
        # 网关/平台侧统一兜底：429、5xx 降级 blocked（先例：千帆 Responses 面只认自有工具类型白名单，
        # 对 custom/web_search 直接 400 —— 那是网关契约差异，不是模型能力缺陷）。
        for label, probe in (("required", default_required), ("指定工具", default_pinned),
                             ("required+关思考", forced), ("custom", bad), ("内置工具", ignored)):
            if probe.status == 429 or probe.status >= 500:
                return self._gateway_verdict(probe, f"Responses 工具矩阵探测（{label}）返回 HTTP {probe.status}",
                                             observations, model=model)
        if trigger is None:
            if all(r.status != 200 for _, r in ladder):
                return self._gateway_verdict(ladder[0][1], "关闭思考后 function 工具仍未被触发（"
                                             + "、".join(f"{name}→HTTP {r.status}" for name, r in ladder) + "）",
                                             observations, model=model)
            problems.append("关闭思考后 function 工具仍未被触发（"
                            + "、".join(f"{name}→HTTP {r.status}" for name, r in ladder)
                            + "）：文档承诺「请先关闭思考模式」后即可强制调用")
        if bad.status == 200:
            notes.append("custom 非 apply_patch 未被拒绝（HTTP 200）：文档承诺 400，属宽松放行")
        elif not _is_rejection(bad.status):
            problems.append(f"custom 非 apply_patch 未被拒绝（HTTP {bad.status}，文档承诺 400）")
        if ignored.status != 200:
            notes.append(f"内置工具 web_search 未被静默忽略（HTTP {ignored.status}）：网关只认自有工具类型集合"
                         f"（{_error_message(ignored)[:120]}），属文档/网关差异，不影响正常使用")
        elif "web_search_call" in ignored_types:
            problems.append("内置工具 web_search 被执行（output 出现 web_search_call，文档承诺忽略）")
        if problems:
            return False, "；".join(problems), observations

        if trigger == "required+effort=none" and not notes:
            detail = ("关闭思考（reasoning.effort=none）后 tool_choice=required 稳定触发 function_call；"
                      "默认思考模式下 required 与指定工具按文档返回 4xx；custom 非 apply_patch 被 4xx 拒绝；"
                      "内置 web_search 被静默忽略（无 web_search_call）")
            return True, detail, observations
        if trigger == "required+effort=none":
            return True, ("关闭思考后 tool_choice=required 触发 function_call；custom 非 apply_patch 被 4xx 拒绝；"
                          "内置 web_search 被静默忽略；" + "；".join(notes)), observations, "warning"
        detail = (f"function 工具经 {trigger} 触发 function_call、custom 非 apply_patch 被 4xx 拒绝、内置工具被静默忽略；"
                  f"但文档解法（reasoning.effort=none 关思考 + required）未生效（HTTP {forced.status}："
                  f"{str(_error_body(forced).get('message'))[:120]}），属文档/线上差异")
        return True, detail, observations, "warning"


def _invalid_json(text: str) -> bool:
    try: json.loads(text); return False
    except json.JSONDecodeError: return True


def summarize_results(results: list[CaseResult]) -> dict[str, Any]:
    counts = {name: sum(r.status == name for r in results) for name in ("passed", "failed", "warning", "blocked", "error")}
    hard_failed = counts["failed"] + counts["error"]
    evaluated = counts["passed"] + hard_failed
    return {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "total": len(results),
            "passed": counts["passed"], "failed": hard_failed, "warning": counts["warning"], "blocked": counts["blocked"],
            "error": counts["error"], "pass_rate": counts["passed"] / evaluated if evaluated else 0.0,
            "all_passed": hard_failed == 0, "results": [asdict(r) for r in results]}


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# DeepSeek V4 专项测试报告", "", f"- 总计：{summary['total']}", f"- 通过：{summary['passed']}", f"- 失败：{summary['failed']}", f"- 警告：{summary['warning']}", f"- 阻塞/未观测：{summary['blocked']}", f"- 已评估用例通过率：{summary['pass_rate']:.1%}", "", "| 用例 | 模型 | 结果 | 耗时(s) | 说明 |", "|---|---|---:|---:|---|"]
    icons = {"passed": "✅ 通过", "failed": "❌ 失败", "error": "❌ 错误", "warning": "⚠️ 警告", "blocked": "⏸ 未验证"}
    for r in summary["results"]:
        detail = str(r["details"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {r['case_id']} {r['name']} | {r['model']} | {icons.get(r['status'], r['status'])} | {r['duration_seconds']:.2f} | {detail} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
