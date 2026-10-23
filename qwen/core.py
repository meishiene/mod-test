"""千问（Qwen 3.6/3.7/3.8）专项用例执行与判定。

设计原则（与 ``glm53/core.py`` 同一套取证/准入分离口径）：
* 正向探针遇到 4xx / 401 / 429 / 5xx / 连接错一律先记 **blocked**（平台侧前提不满足），
  不把「端点没实现该参数」读成「模型能力不达标」；
* 负向契约用例（期待 4xx）收到 4xx 才算通过，收到 2xx 只记 warning（契约放宽）；
* 取证类结论（档位自述、上下文自述、错误信封）用 ``admission`` 与 ``status`` 分离：
  warning 仍可能不准入（例如不存在的模型被静默放行 = 疑似回退）。
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from common.png_fixture import BLUE, YELLOW, data_url, png_chessboard, png_solid, png_text, png_halves
from common.report_style import case_table_report
from common.test_framework import HttpExecutor, ResultLog

from .cases import PRIORITY_THINKING, PRIORITY_TIER, PRIORITY_VISION, model_spec
from .relay_profile import relay_verdict
from common.model_probes import ModelProbeMixin
from common.model_validation import (DEFAULT_SEED, PROBE_DEFINITIONS, assess_validation,
                                     validation_report_stats)
from common.probe_assertions import (valid_usage, valid_tools, stream_tools, exact_answer,
                                     integer_answer, identity_match)

UNSUPPORTED_HINTS = ("unsupported", "not support", "unknown field", "invalid parameter",
                     "invalidparameter", "unrecognized", "不支持", "不支持该参数", "未知参数",
                     # 百炼对「不接受的 content 分片类型」的原话：图片块发给纯文本型号就是这句。
                     "unexpected item type", "does not support image")
# 高/中危泄露特征：只做「是否外泄」的判定，不追求穷举。
_HIGH_RISK = (
    ("堆栈", re.compile(r"traceback \(most recent call last\)|file \"/[^\"]+\", line \d+", re.I)),
    ("内部路径", re.compile(r"/(?:home|app|srv|opt|root)/[A-Za-z0-9_./-]+")),
    ("内网 IP", re.compile(r"\b(?:10|127|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b")),
    ("数据库报错", re.compile(r"psycopg|sqlstate|duplicate key value|syntax error at or near", re.I)),
    ("密钥", re.compile(r"sk-[A-Za-z0-9]{16,}")),
)
_MEDIUM_RISK = (
    ("渠道分组", re.compile(r"no available (?:accounts?|channels?)|channel not found|group[ _-]?id", re.I)),
    ("中转组件标识", re.compile(r"new[ _-]?api|one[ _-]?api|relay[ _-]?error", re.I)),
)


@dataclass(frozen=True)
class QwenConfig:
    api_key: str
    base_url: str          # 已是 api_root（例如 https://dashscope.aliyuncs.com/compatible-mode/v1）
    timeout: float = 180.0
    verify_ssl: bool = True
    trust_env: bool = False
    seed: int = DEFAULT_SEED


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
    # 结论与准入分离：None=沿用 status 推断，False=明确不准入（如无通道模型被静默放行）。
    admission: bool | None = None


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    name: str
    priority: str
    handler: str


CASE_SPECS: tuple[CaseSpec, ...] = (
    CaseSpec("QWEN-01", "基础对话与模型回显", "P0", "basic"),
    CaseSpec("QWEN-02", "非流式 Usage 完整性", "P0", "usage"),
    CaseSpec("QWEN-03", "流式输出与结束事件", "P0", "stream"),
    CaseSpec("QWEN-04", "结构化 JSON 输出", "P0", "json_output"),
    CaseSpec("QWEN-05", "多轮上下文保持", "P0", "multi_turn"),
    CaseSpec("QWEN-06", "长上下文稳定响应", "P0", "long_context"),
    CaseSpec("QWEN-07", "采样参数取值域契约", "P0", "sampling"),
    CaseSpec("QWEN-08", "max_completion_tokens 口径", "P0", "completion_tokens"),
    CaseSpec("QWEN-T01", "默认思考开启", PRIORITY_THINKING, "thinking_default"),
    CaseSpec("QWEN-T02", "enable_thinking=false 关闭思考", PRIORITY_THINKING, "thinking_off"),
    CaseSpec("QWEN-T03", "thinking_budget 限制推理长度", PRIORITY_THINKING, "thinking_budget"),
    CaseSpec("QWEN-T04", "流式 reasoning_content 与 usage", PRIORITY_THINKING, "thinking_stream"),
    CaseSpec("QWEN-T05", "思考模式输出预算与截断", PRIORITY_THINKING, "thinking_max_tokens"),
    CaseSpec("QWEN-T06", "非流式推理内容载体", PRIORITY_THINKING, "thinking_carrier"),
    CaseSpec("QWEN-A01", "工具调用基础链路", "P1", "tool_call"),
    CaseSpec("QWEN-A02", "tool_choice=required", "P1", "tool_required"),
    CaseSpec("QWEN-A03", "工具结果多轮回传", "P1", "tool_replay"),
    CaseSpec("QWEN-A04", "工具流式输出", "P1", "tool_stream"),
    CaseSpec("QWEN-A05", "并行工具调用", "P1", "tool_parallel"),
    CaseSpec("QWEN-F01", "型号回显一致性", PRIORITY_TIER, "tier_identity"),
    CaseSpec("QWEN-F02", "档位与代际自述", PRIORITY_TIER, "tier_self_report"),
    CaseSpec("QWEN-F03", "上下文窗口自述与真实长输入", PRIORITY_TIER, "tier_context"),
    CaseSpec("QWEN-F04", "采样参数取值域指纹", PRIORITY_TIER, "tier_sampling"),
    CaseSpec("QWEN-F05", "错误信封泄露", PRIORITY_TIER, "tier_envelope"),
    CaseSpec("QWEN-F06", "档位指纹汇总", PRIORITY_TIER, "tier_summary"),
    CaseSpec("QWEN-V01", "图像内容理解（五级看图答题）", PRIORITY_VISION, "vision"),
)

CASE_SPECS += tuple(CaseSpec(f"QWEN-{suffix}", name, priority, handler)
                    for suffix, name, priority, handler, _ in PROBE_DEFINITIONS)

# ---------------------------------------------------------------------------
# 响应解析助手
# ---------------------------------------------------------------------------

def msg(body: dict | None) -> dict:
    return (((body or {}).get("choices") or [{}])[0].get("message") or {})


def content(body: dict | None) -> str:
    value = msg(body).get("content", "")
    return value if isinstance(value, str) else ""


def usage_ok(body: dict | None) -> bool:
    return valid_usage(body)


def reasoning_text(body: dict | None) -> str:
    """非流式推理内容的两个已知载体：reasoning_content / reasoning_details。"""
    message = msg(body)
    direct = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(direct, str) and direct.strip():
        return direct
    details = message.get("reasoning_details") or []
    if isinstance(details, list):
        return "".join(str(item.get("text") or "") for item in details if isinstance(item, dict))
    return ""


def reasoning_tokens(body: dict | None) -> int | None:
    usage = (body or {}).get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    value = details.get("reasoning_tokens")
    return value if type(value) is int and value >= 0 else None


def _text_of(body: dict | None) -> str:
    return json.dumps(body or {}, ensure_ascii=False).lower()


def _unsupported(status: int, body: dict | None, extra: str = "") -> bool:
    # 404 不在此列：端点上没有这条路径是路由问题（记 blocked），不是「参数不支持」。
    if status not in (400, 422, 501):
        return False
    text = _text_of(body) + " " + str(extra or "").lower()
    return any(hint in text for hint in UNSUPPORTED_HINTS)


def _non_json_response(record: Any) -> bool:
    """判断「200 但根本不是模型响应」：典型是网关把未知路径兜成前端 index.html。

    只看响应正文首字符 + JSON 解析失败，避免误伤流式文本（SSE 以 ``data:`` 开头）。
    """
    body = getattr(record, "response_body", None)
    if isinstance(body, (dict, list)):
        return False
    text = (getattr(record, "response_text", None) or getattr(record, "raw_response_body", None) or "")
    if not isinstance(text, str):
        return False
    head = text.lstrip()[:15].lower()
    return head.startswith("<!doctype") or head.startswith("<html")


def _blocked_status(status: int) -> bool:
    """平台侧/瞬时故障：只记录不阻断。404 也在内——新参数/新端点未开放不是模型结论。"""
    return status in (0, 401, 403, 404, 408, 409, 429) or status >= 500


def _rejected_as_expected(status: int) -> bool:
    """负向契约用例：期待「参数被拒」。

    只有 400/422 这类**请求语义错误**才算拒绝——401/403/404/429/5xx 是账号、路由或
    限流问题，把限流读成「越界参数被拒」会得出假通过（历史坑：429 混进 4xx 判定）。
    """
    return status in (400, 422)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class QwenRunner(ModelProbeMixin):
    def __init__(self, config: QwenConfig, audit_dir: Path | None = None,
                 client: httpx.AsyncClient | None = None):
        self.config = config
        self.audit_dir = audit_dir
        self.current_case_id: str | None = None
        self._external_client = client
        self.client: httpx.AsyncClient | None = client
        self.executor: HttpExecutor | None = None
        self.log: ResultLog | None = None
        if audit_dir:
            audit_dir.mkdir(parents=True, exist_ok=True)
            self.audit_path: Path | None = audit_dir / "requests.jsonl"
        else:
            self.audit_path = None

    async def __aenter__(self):
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=self.config.timeout, verify=self.config.verify_ssl,
                trust_env=self.config.trust_env,
                headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            )
        self.executor = HttpExecutor(timeout=self.config.timeout, verify_ssl=self.config.verify_ssl,
                                     trust_env=self.config.trust_env, client=self.client)
        await self.executor.__aenter__()
        if self.audit_dir:
            self.log = ResultLog(self.audit_dir / "requests.jsonl")
        return self

    async def __aexit__(self, *args):
        if self.log:
            self.log.close()
        if self.executor:
            await self.executor.__aexit__(*args)
        # 外部注入的 client 由调用方负责关闭（测试里常见）。
        if self.client is not None and self._external_client is None:
            await self.client.aclose()

    def endpoint(self) -> str:
        return self.config.base_url.rstrip("/") + "/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json",
                "X-Request-ID": f"QWEN-{uuid.uuid4().hex[:12]}"}

    async def request(self, model: str, messages: list[dict[str, Any]], *, stream: bool = False,
                      **kwargs: Any):
        if self.executor is None:
            await self.__aenter__()
        payload: dict[str, Any] = {"model": model, "messages": messages,
                                   "max_tokens": kwargs.pop("max_tokens", 2048), "stream": stream}
        payload.update({key: value for key, value in kwargs.items() if value is not None})
        # 显式传 None 表示「不发这个字段」（例如只验证 max_completion_tokens 的用例）。
        payload = {key: value for key, value in payload.items() if value is not None}
        record = await self.executor.execute(case_id=self.current_case_id or "QWEN-request",
                                             url=self.endpoint(), payload=payload,
                                             headers=self._headers(), stream=stream)
        if self.log:
            await self.log.write(record)
        elapsed = record.timing.get("e2e_ms", 0) / 1000
        # 网关会把「不认识的路径」兜成前端 index.html 并返回 200（api.oinone.top 实测）。
        # 这种响应里根本没有模型输出，必须记 blocked —— 否则会被读成「模型答了空字符串」，
        # 把路径写错变成一串 failed（本轮复测踩过：/compatible-mode/v1 被 SPA 兜 200）。
        if _non_json_response(record):
            return 0, {"error": {"type": "non_json_response",
                                 "message": "网关返回非 JSON 响应（疑似前端 SPA 兜底 200），请求未到达模型"}}, \
                elapsed, {"non_json": True}
        if not stream:
            body = record.response_body if isinstance(record.response_body, dict) else None
            return record.http_status or 0, body, elapsed, {}
        parsed = {"content": record.content, "reasoning": record.reasoning,
                  "done": record.done_received, "invalid_events": record.invalid_sse_events,
                  "finish_reasons": record.finish_reasons, "usage": record.usage or {},
                  "tool_calls": stream_tools(record.parsed_sse_events)}
        return record.http_status or 0, None, elapsed, parsed

    async def raw_request(self, url: str, body: str):
        """发送原始请求体（畸形 JSON 探针必须绕过 json= 序列化）。"""
        if self.client is None:
            await self.__aenter__()
        response = await self.client.post(url, content=body.encode("utf-8"), headers=self._headers())
        text = response.text
        try:
            parsed = response.json()
        except (ValueError, json.JSONDecodeError):
            parsed = None
        self._audit_row({"case_id": self.current_case_id, "url": url, "raw_body": body,
                         "http_status": response.status_code, "response_body": parsed,
                         "response_text": text[:4000]})
        return response.status_code, parsed, text[:4000]

    def _audit_row(self, row: dict[str, Any]) -> None:
        """原始请求的取证行（不经 ExecutionRecord）。"""
        if not self.audit_path:
            return
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def tools(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {
            "name": "get_weather", "description": "查询指定城市天气",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                           "required": ["city"], "additionalProperties": False}}}]

    def tools_parallel(self) -> list[dict[str, Any]]:
        return [*self.tools(), {"type": "function", "function": {
            "name": "get_time", "description": "查询指定时区的当前时间",
            "parameters": {"type": "object", "properties": {"zone": {"type": "string"}},
                           "required": ["zone"], "additionalProperties": False}}}]

    async def run(self, model: str, priorities: tuple[str, ...] | list[str],
                  case_ids: set[str] | None = None,
                  faces: tuple[str, ...] | None = None) -> list[CaseResult]:
        # 面过滤（默认 model+verify）：通道契约面在中转链路上大面积不通过，默认不跑。
        from common.faces import ALL_FACES, filter_specs, selected_faces
        from .faces import CASE_FACES

        # 定向复测（case_ids）是用户显式点名，面过滤让位——否则点通道面用例会被静默跳过。
        active = ALL_FACES if case_ids else selected_faces(faces)
        selected = [spec for spec in CASE_SPECS
                    if spec.priority in priorities and (not case_ids or spec.case_id in case_ids)]
        results: list[CaseResult] = []
        for spec in filter_specs(selected, CASE_FACES, active):
            self.current_case_id = spec.case_id
            started = time.perf_counter()
            try:
                outcome = await getattr(self, "case_" + spec.handler)(model)
                admission = None
                if len(outcome) == 5:
                    passed, detail, observations, status, admission = outcome
                else:
                    passed, detail, observations, status = outcome
            except Exception as exc:  # 网络/解析异常：记 error，不静默吞掉
                passed, detail, observations, status = False, f"{type(exc).__name__}: {exc}", {}, "error"
                admission = None
            results.append(CaseResult(spec.case_id, spec.name, model, bool(passed), status, detail,
                                      time.perf_counter() - started, observations, admission))
        self.current_case_id = None
        return results

    # -- 内部工具 ---------------------------------------------------------
    @staticmethod
    def _http_status(status: int, body: dict | None, soft_ok: bool = True) -> str:
        """正向探针的裁定：HTTP 200 才可能 passed；4xx/5xx 一律 blocked。

        4xx 可能是我们的报文写错，也可能是端点不支持该参数——两者都无法据此
        判定模型能力，故只记录不阻断（与项目既有「网关侧问题不判 failed」口径一致）。
        """
        if status == 200:
            return "passed" if soft_ok else "failed"
        if status == 0 or status >= 500 or 400 <= status < 500:
            return "blocked"
        return "failed"

    # -- 核心对话面 -------------------------------------------------------
    async def case_basic(self, model: str):
        status, body, _, _ = await self.request(model, [{"role": "user", "content": "只回复 OK"}],
                                                enable_thinking=False)
        returned = (body or {}).get("model")
        ok = status == 200 and content(body).strip() == "OK"
        return ok, f"HTTP {status}，返回模型={returned or '未返回'}", {"returned_model": returned}, \
            self._http_status(status, body, ok)

    async def case_usage(self, model: str):
        status, body, _, _ = await self.request(model, [{"role": "user", "content": "只回复 OK"}],
                                                enable_thinking=False)
        ok = status == 200 and usage_ok(body)
        return ok, f"HTTP {status}，Usage {'完整' if ok else '不完整'}", {"usage": (body or {}).get("usage")}, \
            self._http_status(status, body, ok)

    async def case_stream(self, model: str):
        status, _, _, parsed = await self.request(model, [{"role": "user", "content": "只回复 OK"}],
                                                  stream=True, enable_thinking=False)
        ok = status == 200 and bool(parsed.get("done")) and parsed.get("content", "").strip() == "OK" and not parsed.get("invalid_events") and "stop" in parsed.get("finish_reasons", [])
        return ok, f"HTTP {status}，结束事件={parsed.get('done')}，非法事件={parsed.get('invalid_events', 0)}", \
            parsed, self._http_status(status, None, ok)

    async def case_json_output(self, model: str):
        status, body, _, _ = await self.request(
            model, [{"role": "user", "content": "返回 JSON，字段 answer，值为 OK，不要任何额外文字"}],
            enable_thinking=False, response_format={"type": "json_object"})
        raw = content(body)
        try:
            ok = status == 200 and json.loads(raw).get("answer") == "OK"
        except (ValueError, AttributeError):
            ok = False
        return ok, f"HTTP {status}，JSON 可解析且字段匹配={ok}", {"content": raw[:500]}, \
            self._http_status(status, body, ok)

    async def case_multi_turn(self, model: str):
        status, body, _, _ = await self.request(model, [
            {"role": "user", "content": "记住代号是 ORBIT-38"},
            {"role": "assistant", "content": "好的"},
            {"role": "user", "content": "代号是什么？只回复代号"},
        ], enable_thinking=False)
        ok = status == 200 and "ORBIT-38" in content(body)
        return ok, f"HTTP {status}，代号命中={ok}", {"content": content(body)[:200]}, \
            self._http_status(status, body, ok)

    async def case_long_context(self, model: str):
        return await self.case_quality_context(model)

    async def case_sampling(self, model: str):
        st_zero, body_zero, _, _ = await self.request(model, [{"role": "user", "content": "只回复 OK"}],
                                                      enable_thinking=False, temperature=0)
        st_bad, body_bad, _, _ = await self.request(model, [{"role": "user", "content": "只回复 OK"}],
                                                    enable_thinking=False, temperature=2.5)
        zero_ok = st_zero == 200 and bool(content(body_zero))
        rejected = _rejected_as_expected(st_bad)
        passed = zero_ok and rejected
        status = "passed" if passed else ("warning" if zero_ok and not rejected else self._http_status(st_zero, body_zero, zero_ok))
        return passed, f"temperature=0 → HTTP {st_zero}；temperature=2.5 → HTTP {st_bad}（应 4xx）", \
            {"temperature_zero_status": st_zero, "out_of_range_status": st_bad, "rejected": rejected}, status

    async def case_completion_tokens(self, model: str):
        status, body, _, _ = await self.request(model, [{"role": "user", "content": "用一句话介绍你自己"}],
                                                enable_thinking=False, max_tokens=None,
                                                max_completion_tokens=1024)
        ok = status == 200 and bool(content(body))
        return ok, f"HTTP {status}，max_completion_tokens 可用={ok}", \
            {"http_status": status, "response": _text_of(body)[:400]}, self._http_status(status, body, ok)

    # -- 思考模式 ---------------------------------------------------------
    async def case_thinking_default(self, model: str):
        status, body, _, _ = await self.request(
            model, [{"role": "user", "content": "计算 918273*817263。思考后最终正文只输出整数。"}], max_tokens=4096)
        text = reasoning_text(body)
        tokens = reasoning_tokens(body)
        ok = status == 200 and (bool(text.strip()) or bool(tokens)) and integer_answer(content(body), 918273*817263)
        return ok, f"HTTP {status}，默认思考：推理字符={len(text)}，reasoning_tokens={tokens}", \
            {"reasoning_chars": len(text), "reasoning_tokens": tokens}, \
            self._http_status(status, body, ok)

    async def case_thinking_off(self, model: str):
        status, body, _, _ = await self.request(
            model, [{"role": "user", "content": "计算 918273*817263。思考后最终正文只输出整数。"}],
            enable_thinking=False, max_tokens=2048)
        text = reasoning_text(body)
        tokens = reasoning_tokens(body)
        disabled = not text.strip() and not tokens and integer_answer(content(body), 918273*817263)
        if status != 200:
            return False, f"HTTP {status}，无法验证关闭思考", {"http_status": status}, self._http_status(status, body, False)
        # 参数未生效属能力级：只记 warning，不否决准入。
        return disabled, f"HTTP 200，reasoning 字符={len(text)}，reasoning_tokens={tokens}", \
            {"reasoning_chars": len(text), "reasoning_tokens": tokens}, "passed" if disabled else "warning"

    async def case_thinking_budget(self, model: str):
        budget = 256
        status, body, _, _ = await self.request(
            model, [{"role": "user", "content": "求 1 到 100 所有质数的和，先思考再作答。"}],
            enable_thinking=True, thinking_budget=budget, max_tokens=2048)
        tokens = reasoning_tokens(body)
        text_len = len(reasoning_text(body))
        if status != 200:
            return False, f"HTTP {status}，无法验证 thinking_budget", {"http_status": status}, self._http_status(status, body, False)
        if tokens is None:
            # 端点不返回 reasoning_tokens：无法验证预算，只记录。
            return False, f"HTTP 200，但 usage 缺 reasoning_tokens（推理字符={text_len}）", \
                {"reasoning_tokens": None, "reasoning_chars": text_len}, "blocked"
        enforced = 0 < tokens <= budget * 2  # 允许 2 倍容差：截断边界与计费口径各家略有差异
        return enforced, f"budget={budget}，reasoning_tokens={tokens}（≤{budget * 2} 视为生效）", \
            {"budget": budget, "reasoning_tokens": tokens}, "passed" if enforced else "warning"

    async def case_thinking_stream(self, model: str):
        status, _, _, parsed = await self.request(
            model, [{"role": "user", "content": "计算 918273*817263。思考后最终正文只输出整数。"}],
            stream=True, enable_thinking=True, max_tokens=4096,
            stream_options={"include_usage": True})
        usage = parsed.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        rt = details.get("reasoning_tokens")
        ok = (status == 200 and bool(parsed.get("reasoning")) and bool(parsed.get("done"))
              and not parsed.get("invalid_events") and "stop" in parsed.get("finish_reasons", [])
              and type(rt) is int and rt > 0
              and valid_usage({"usage": usage})
              and rt <= usage.get("completion_tokens", 0)
              and integer_answer(parsed.get("content", ""), 918273*817263))
        return ok, f"HTTP {status}，推理字符={len(parsed.get('reasoning') or '')}，reasoning_tokens={rt}，DONE={parsed.get('done')}", \
            {"reasoning_chars": len(parsed.get("reasoning") or ""), "reasoning_tokens": rt,
             "usage": usage, "invalid_events": parsed.get("invalid_events")}, \
            self._http_status(status, None, ok)

    async def case_thinking_max_tokens(self, model: str):
        # Test an explicit budget, never assume a shared vendor ceiling for all tiers.
        budget = 64
        status, body, _, _ = await self.request(
            model, [{"role":"user", "content":"请连续列出1至10000的整数，不要省略。"}],
            enable_thinking=True, max_tokens=budget)
        usage = (body or {}).get("usage") or {}
        used = usage.get("completion_tokens")
        finish = (((body or {}).get("choices") or [{}])[0]).get("finish_reason")
        if status != 200 or type(used) is not int:
            return False, "输出预算缺乏有效 usage 证据", {"http_status":status}, "blocked"
        ok = 0 < used <= budget and finish == "length"
        return ok, f"预算={budget}，输出计费tokens={used}，finish_reason={finish}", \
            {"budget":budget,"completion_tokens":used,"finish_reason":finish}, "passed" if ok else "warning"

    async def case_thinking_carrier(self, model: str):
        status, body, _, _ = await self.request(
            model, [{"role": "user", "content": "用两步推理回答：17*23 等于多少？"}],
            enable_thinking=True, max_tokens=4096)
        text = reasoning_text(body)
        tokens = reasoning_tokens(body)
        if status != 200:
            return False, f"HTTP {status}，无法验证推理载体", {"http_status": status}, self._http_status(status, body, False)
        if text.strip():
            return True, f"推理载体存在（{len(text)} 字符），reasoning_tokens={tokens}", \
                {"carrier_chars": len(text), "reasoning_tokens": tokens}, "passed"
        if tokens:
            return False, f"无推理正文但 reasoning_tokens={tokens}（载体字段缺失）", \
                {"carrier_chars": 0, "reasoning_tokens": tokens}, "warning"
        return False, "既无推理正文也无 reasoning_tokens", {"carrier_chars": 0, "reasoning_tokens": None}, "warning"

    # -- Agent / 工具面 ---------------------------------------------------
    async def case_tool_call(self, model: str):
        status, body, _, _ = await self.request(model, [{"role": "user", "content": "请查询北京天气"}],
                                                tools=self.tools(), enable_thinking=False)
        calls = msg(body).get("tool_calls") or []
        ok = status == 200 and valid_tools(calls, self.tools(), {"get_weather": {"city": "北京"}})
        return ok, f"HTTP {status}，工具调用数={len(calls)}", {"tool_calls": calls[:1]}, \
            self._http_status(status, body, ok)

    async def case_tool_required(self, model: str):
        status, body, _, _ = await self.request(model, [{"role": "user", "content": "请查询天气"}],
                                                tools=self.tools(), tool_choice="required",
                                                enable_thinking=False)
        ok = status == 200 and valid_tools(msg(body).get("tool_calls"), self.tools())
        return ok, f"HTTP {status}，强制调用={'命中' if ok else '未命中'}", {"tool_calls": msg(body).get("tool_calls")}, \
            self._http_status(status, body, ok)

    async def case_tool_replay(self, model: str):
        tool = self.tools()[0]
        status, body, _, _ = await self.request(model, [{"role": "user", "content": "查询北京天气"}],
                                                tools=[tool], tool_choice="required", enable_thinking=False)
        calls = msg(body).get("tool_calls") or []
        if status != 200 or not valid_tools(calls, [tool], {"get_weather": {"city": "北京"}}):
            return False, f"首轮 HTTP {status} 未产生工具调用", {"first_status": status}, \
                self._http_status(status, body, False)
        call = calls[0]
        messages = [
            {"role": "user", "content": "查询北京天气"},
            {"role": "assistant", "content": None, "tool_calls": calls},
            {"role": "tool", "tool_call_id": call.get("id", "call_1"), "content": "晴，25 度"},
        ]
        status2, body2, _, _ = await self.request(model, messages, tools=[tool], enable_thinking=False)
        ok = status2 == 200 and "晴" in content(body2) and "25" in content(body2)
        return ok, f"首轮/回传 HTTP {status}/{status2}，最终正文={'有' if ok else '无'}", \
            {"first_status": status, "replay_status": status2}, self._http_status(status2, body2, ok)

    async def case_tool_stream(self, model: str):
        status, _, _, parsed = await self.request(model, [{"role": "user", "content": "查询北京天气，调用工具"}],
                                                  tools=self.tools(), stream=True, enable_thinking=False)
        ok = (status == 200 and bool(parsed.get("done")) and not parsed.get("invalid_events")
              and "tool_calls" in parsed.get("finish_reasons", [])
              and valid_tools(parsed.get("tool_calls"), self.tools()))
        return ok, f"HTTP {status}，流式结束事件={parsed.get('done')}", parsed, \
            self._http_status(status, None, ok)

    async def case_tool_parallel(self, model: str):
        status, body, _, _ = await self.request(
            model, [{"role": "user", "content": "同时查询北京天气和上海时间（时区参数 Asia/Shanghai），两个工具都要调用"}],
            tools=self.tools_parallel(), parallel_tool_calls=True, enable_thinking=False)
        calls = msg(body).get("tool_calls") or []
        names = {((call.get("function") or {}).get("name")) for call in calls}
        ok = status == 200 and valid_tools(calls, self.tools_parallel(),
             {"get_weather": {"city": "北京"}, "get_time": {"zone": "Asia/Shanghai"}})
        if status == 200 and not ok:
            return False, f"HTTP 200，但只产生 {len(calls)} 个调用（未并行）", {"tool_calls": calls}, "warning"
        return ok, f"HTTP {status}，并行调用={sorted(n for n in names if n)}", {"tool_calls": calls[:2]}, \
            self._http_status(status, body, ok)

    # -- 档位与身份指纹 ---------------------------------------------------
    async def case_tier_identity(self, model: str):
        status, body, _, _ = await self.request(model, [{"role": "user", "content": "只回复 OK"}],
                                                enable_thinking=False)
        returned = (body or {}).get("model")
        if status != 200:
            return False, f"HTTP {status}，无法读取返回模型", {"http_status": status}, self._http_status(status, body, False)
        if not returned:
            return False, "HTTP 200，但响应未返回 model 字段", {"returned_model": None}, "warning"
        requested = str(model).strip().lower().rsplit("/", 1)[-1]
        actual = str(returned).strip().lower().rsplit("/", 1)[-1]
        if identity_match(model, returned):
            return True, f"返回模型 {returned} 与请求一致", {"requested_model": model, "returned_model": returned}, "passed"
        if "qwen" in actual and "qwen" in requested:
            return False, f"返回模型 {returned} 与请求 {model} 同族但不同名（别名/版本差异，仅留痕）", \
                {"requested_model": model, "returned_model": returned}, "warning"
        return False, f"返回模型 {returned} 与请求 {model} 异族（疑似替换）", \
            {"requested_model": model, "returned_model": returned}, "failed"

    async def case_tier_self_report(self, model: str):
        status, body, _, _ = await self.request(
            model, [{"role": "user", "content": "你当前是哪个模型？请只回复模型名，例如 qwen3.8-max。"}],
            enable_thinking=False, max_tokens=64)
        text = content(body).strip()
        spec = model_spec(model)
        if status != 200 or not text:
            return False, f"HTTP {status}，模型未给出可解析的档位自述", {"self_report": text[:100]}, \
                ("warning" if status == 200 else self._http_status(status, body, False))
        generation = re.search(r"3\.[678]", text.lower())
        tier = next((name for name in ("flash", "plus", "max") if name in text.lower()), None)
        expected_gen = spec.generation if spec else None
        expected_tier = spec.tier if spec else None
        agrees = generation is not None and tier is not None and (expected_gen is None or generation.group(0) == expected_gen) and (expected_tier is None or tier == expected_tier)
        # 自述版本长期不可靠：只留痕，永不判 failed。
        return agrees, f"自述「{text[:40]}」·声明 {expected_gen or '?'}/{expected_tier or '?'}·" \
            f"{'一致' if agrees else '不一致（仅留痕）'}", \
            {"self_report": text[:120], "generation": generation.group(0) if generation else None,
             "tier": tier, "expected_generation": expected_gen, "expected_tier": expected_tier}, \
            "passed" if agrees else "warning"

    async def case_tier_context(self, model: str):
        # ① 自述窗口（参考）；② 真实约 6K 字符输入 + 标记召回。
        status, body, _, _ = await self.request(
            model, [{"role": "user", "content": "请只回复你的最大上下文窗口 token 数（只回数字）"}],
            enable_thinking=False, max_tokens=32)
        claimed_text = content(body)
        numbers = [int(item) for item in re.findall(r"\d{4,7}", claimed_text.replace(",", ""))]
        claimed = max(numbers) if numbers else None
        filler = "长上下文探针填充内容，请忽略。" * 500 + "\n标记：CTX-QWEN-38"
        st2, body2, _, _ = await self.request(model, [
            {"role": "user", "content": filler},
            {"role": "user", "content": "只回复上面出现的标记"},
        ], enable_thinking=False, max_tokens=128)
        found = st2 == 200 and "CTX-QWEN-38" in content(body2)
        spec = model_spec(model)
        rows = [
            {"name": "上下文窗口自述", "status": "passed" if claimed else "warning",
             "passed": bool(claimed), "duration_seconds": 0.0,
             "details": f"自述 {claimed or '未给出'}" + (f"；矩阵参考 {spec.context_window}" if spec else "")},
            {"name": "真实长输入召回", "status": "passed" if found else ("warning" if st2 == 200 else "blocked"),
             "passed": found, "duration_seconds": 0.0,
             "details": f"HTTP {st2}，标记命中={found}（输入 {len(filler)} 字符）"},
        ]
        low_claim = claimed is not None and claimed < 131072
        passed = found and not low_claim
        if st2 != 200 and (_blocked_status(st2) or _unsupported(st2, body2)):
            status_out = "blocked"
        elif found:
            status_out = "warning" if low_claim else "passed"
        else:
            status_out = "warning"
        return passed, f"自述窗口={claimed or '未给出'}，长输入召回={found}（HTTP {st2}）", \
            {"claimed_context_window": claimed, "marker_found": found, "rows": rows}, status_out

    async def case_tier_sampling(self, model: str):
        st_seed, body_seed, _, _ = await self.request(model, [{"role": "user", "content": "只回复 OK"}],
                                                      enable_thinking=False, seed=42)
        st_temp, body_temp, _, _ = await self.request(model, [{"role": "user", "content": "只回复 OK"}],
                                                      enable_thinking=False, temperature=3)
        st_top_p, body_top_p, _, _ = await self.request(model, [{"role": "user", "content": "只回复 OK"}],
                                                       enable_thinking=False, top_p=1.5)
        seed_ok = st_seed == 200 and bool(content(body_seed))
        rejected = {name: _rejected_as_expected(code)
                    for name, code in (("temperature", st_temp), ("top_p", st_top_p))}
        passed = seed_ok and all(rejected.values())
        if not seed_ok:
            status = self._http_status(st_seed, body_seed, False)
        elif passed:
            status = "passed"
        else:
            status = "warning"
        return passed, f"seed=42 → {st_seed}；temperature=3 → {st_temp}；top_p=1.5 → {st_top_p}（后两者应 4xx）", \
            {"seed_status": st_seed, "temperature_out_of_range_status": st_temp,
             "top_p_out_of_range_status": st_top_p, "rejected": rejected}, status

    async def case_tier_envelope(self, model: str):
        base = self.config.base_url.rstrip("/")
        findings: list[dict[str, str]] = []
        rows: list[dict[str, Any]] = []

        st_mal, body_mal, text_mal = await self.raw_request(base + "/chat/completions", '{"model": "')
        rows.append({"name": "畸形 JSON", "step": "malformed_json", "status": "passed" if _rejected_as_expected(st_mal) else "warning",
                     "passed": _rejected_as_expected(st_mal), "duration_seconds": 0.0, "details": f"HTTP {st_mal}"})
        if st_mal == 200:
            rows[-1]["details"] = f"HTTP 200（解析过宽，疑似把畸形输入当正常请求）"
        self._scan_leakage(f"malformed_json:{st_mal}", text_mal, findings)

        ghost = f"{model}-nonexistent-{uuid.uuid4().hex[:6]}"
        st_ghost, body_ghost, _, _ = await self.request(ghost, [{"role": "user", "content": "只回复 OK"}],
                                                        enable_thinking=False)
        ghost_text = json.dumps(body_ghost or {}, ensure_ascii=False)
        rows.append({"name": "不存在的模型", "step": "missing_model",
                     "status": "passed" if _rejected_as_expected(st_ghost) else "warning",
                     "passed": _rejected_as_expected(st_ghost), "duration_seconds": 0.0,
                     "details": f"HTTP {st_ghost}" + ("（2xx，疑似静默回退）" if st_ghost == 200 else "")})
        self._scan_leakage(f"missing_model:{st_ghost}", ghost_text, findings)

        st_path, body_path, text_path = await self.raw_request(base + "/qwen-probe-not-found", "{}")
        rows.append({"name": "不存在的路径", "step": "wrong_path", "status": "passed" if st_path == 404 else "warning",
                     "passed": st_path == 404, "duration_seconds": 0.0, "details": f"HTTP {st_path}"})
        self._scan_leakage(f"wrong_path:{st_path}", text_path, findings)

        high = [item for item in findings if item["severity"] == "high"]
        medium = [item for item in findings if item["severity"] == "medium"]
        silent_fallback = st_ghost == 200
        admission = not high and not silent_fallback
        if high:
            status, passed = "failed", False
        elif medium or silent_fallback or any(not row["passed"] for row in rows):
            status, passed = "warning", False
        else:
            status, passed = "passed", True
        detail = f"泄露高危={len(high)} 中危={len(medium)}；畸形 JSON={st_mal}，不存在模型={st_ghost}，错误路径={st_path}"
        if silent_fallback:
            detail += "；不存在的模型被静默放行（疑似回退，不准入）"
        return passed, detail, {"findings": findings, "rows": rows, "silent_fallback": silent_fallback}, status, admission

    @staticmethod
    def _scan_leakage(step: str, text: str, findings: list[dict[str, str]]) -> None:
        lowered = (text or "").lower()
        for label, pattern in _HIGH_RISK:
            if pattern.search(lowered):
                findings.append({"step": step, "severity": "high", "label": label})
        for label, pattern in _MEDIUM_RISK:
            if pattern.search(lowered):
                findings.append({"step": step, "severity": "medium", "label": label})

    async def case_tier_summary(self, model: str):
        return False, "本地汇总不构成远端证据", {"synthetic": True}, "blocked"

    # -- 图像内容理解 -----------------------------------------------------
    async def _vision(self, model: str, png: bytes, prompt: str, expect: tuple[str, ...]):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url(png)}},
        ]}]
        status, body, _, _ = await self.request(model, messages, enable_thinking=False, max_tokens=64)
        text = content(body).strip()
        lowered = text.lower()
        ok = status == 200 and exact_answer(text, expect)
        if status != 200:
            status_out = "blocked" if _blocked_status(status) or _unsupported(status, body) else "failed"
        else:
            status_out = "passed" if ok else "failed"
        return ok, f"HTTP {status}，回答「{text[:40]}」，期望 {expect}", \
            {"http_status": status, "answer": text[:120], "expected": list(expect)}, status_out

    async def case_vision(self, model: str):
        """五级看图答题按**卡片聚合**判定：全对 passed、≥60% warning、<60% failed。

        逐题结果进 observations.rows，报告里逐行展开，避免只看一个汇总数字无法复核。
        """
        spec = model_spec(model)
        if spec is not None and not spec.image_input:
            return False, "型号声明不提供图像输入，判不适用，本次不发送视觉请求", {
                "applicable":False,"declared_image_input":False,"rows":[]}, "blocked"
        rng = self.probe_rng("vision")
        digits = str(rng.randrange(1000,10000))
        right_yellow = bool(rng.getrandbits(1))
        levels = (
            ("纯色识别", png_solid(YELLOW), "这张图片是什么颜色？只回复颜色词。", ("黄", "yellow")),
            ("棋盘计数", png_chessboard(cells=8), "数一数图中有多少个深色格子，只回复数字。", ("32",)),
            # 蓝字写在黄底上：正确读图应答「黄」，只读文字先验会答「蓝/绿」。
            ("图文冲突压制", png_text("GREEN", fg=BLUE, bg=YELLOW),
             "图片的背景是什么颜色？只回复颜色词。", ("黄", "yellow")),
            ("随机数字 OCR", png_text(digits), "只回复图片中的四位数字，不要解释。", (digits,)),
            ("左右空间关系", png_halves(BLUE if right_yellow else YELLOW, YELLOW if right_yellow else BLUE),
             "图片右半部分是什么颜色？只回复颜色词。", ("黄", "yellow") if right_yellow else ("蓝", "blue")),
        )
        rows: list[dict[str, Any]] = []
        correct = 0
        blocked = 0
        for name, png, prompt, expect in levels:
            ok, detail, _obs, level_status = await self._vision(model, png, prompt, expect)
            correct += int(ok)
            blocked += int(level_status == "blocked")
            rows.append({"name": name, "status": level_status, "passed": ok,
                         "duration_seconds": 0.0, "details": detail})
        total = len(levels)
        spec = model_spec(model)
        declared_vision = True if spec is None else spec.image_input
        if blocked == total:
            if not declared_vision:
                # 型号本身不提供图像输入（如 qwen3.7-max 纯文本旗舰）：被拒是预期行为，
                # 不能记成「网关能力缺口」，也不能当成失败。
                return False, (f"该型号不提供图像输入（image_input=False，纯文本档位）："
                               f"{blocked}/{total} 探针被端点拒绝，属能力边界、判不适用"), \
                    {"rows": rows, "levels_correct": correct,
                     "declared_image_input": False, "applicable": False}, "blocked"
            return False, (f"视觉输入全部被端点拒绝（{blocked}/{total}），但型号声明 image_input=True "
                           f"⇒ 网关/上游能力缺口"), \
                {"rows": rows, "levels_correct": correct,
                 "declared_image_input": True, "applicable": True}, "blocked"
        ratio = correct / total
        if correct == total:
            status, passed = "passed", True
        elif ratio >= 0.6:
            status, passed = "warning", False   # 能力打折但不否决准入
        else:
            status, passed = "failed", False
        return passed, f"答对 {correct}/{total}" + ("（能力打折，不否决准入）" if status == "warning" else ""), \
            {"rows": rows, "levels_correct": correct, "seed":self.probe_seed(), "levels_total":total}, status


# ---------------------------------------------------------------------------
# 汇总与报告
# ---------------------------------------------------------------------------

def summarize_results(results: list[CaseResult], model: str | None = None) -> dict[str, Any]:
    counts = {name: sum(result.status == name for result in results)
              for name in ("passed", "failed", "warning", "blocked", "error")}
    hard = counts["failed"] + counts["error"]
    evaluated = counts["passed"] + hard
    tier_results = [result for result in results
                    if result.case_id.startswith("QWEN-F") and result.case_id != "QWEN-F06"]
    tier_warnings = [result for result in tier_results if result.status == "warning"]
    tier_hard = [result for result in tier_results if result.status in {"failed", "error"}]
    if tier_hard:
        tier_level = "存在硬失败（疑似替换/回退）"
    elif tier_warnings:
        tier_level = "部分指纹不一致，不能认定同款"
    elif {r.case_id for r in tier_results if r.status == "passed" and r.passed} == {f"QWEN-F0{i}" for i in range(1, 6)}:
        tier_level = "已测接口指纹相容（不证明同款或满血）"
    else:
        tier_level = "无法判定"
    blocked_admission = [result.case_id for result in results if result.admission is False]
    # 两套口径并存：``all_passed`` 是官方严格口径（逐条对齐百炼文档，保留可追溯），
    # ``relay_verdict`` 是中转接入口径 —— 对外唯一准入结论（见 relay_profile.py）。
    relay = relay_verdict([asdict(result) for result in results], model)
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total": len(results),
        "passed": counts["passed"],
        "failed": hard,
        "warning": counts["warning"],
        "blocked": counts["blocked"],
        "error": counts["error"],
        "pass_rate": counts["passed"] / evaluated if evaluated else 0,
        "all_passed": hard == 0 and not blocked_admission and relay["evaluated"] > 0,
        "official_all_passed": hard == 0 and not blocked_admission and relay["evaluated"] > 0,
        "admission_blocked": blocked_admission,
        "verdict_contract": "relay",
        "relay_verdict": relay,
        "results": [asdict(result) for result in results],
        "model_validation": assess_validation(results, model or (results[0].model if results else "")),
        "tier_assessment": {
            "level": tier_level,
            "tested": len(tier_results),
            "warnings": len(tier_warnings),
            "hard_failures": len(tier_hard),
            "diagnosis": [{"case_id": result.case_id, "name": result.name, "status": result.status,
                           "details": result.details} for result in tier_results if result.status != "passed"],
        },
    }


def write_report(out: Path, summary: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows: list[list[Any]] = []
    for result in summary["results"]:
        rows.append([f"{result['case_id']} {result['name']}", result["model"], result["status"],
                     f"{result.get('duration_seconds', 0):.2f}", result["details"]])
        # 子探针逐行展开：结论行在上、证据行在下，避免只看结论无法复核。
        for sub in (result.get("observations") or {}).get("rows") or []:
            rows.append([f"　↳ {sub.get('name', '')}", result["model"], sub.get("status", ""),
                         f"{sub.get('duration_seconds') or 0:.2f}", sub.get("details", "")])
    extra_stats: list[str] = validation_report_stats(summary)
    relay = summary.get("relay_verdict") or {}
    if relay:
        extra_stats.append("准入口径：中转接入（relay）——核心面 + 模型一致性面失败才阻断，"
                           "高级面缺口只记录")
        extra_stats.append(f"中转口径结论：{'准入' if relay.get('admission_passed') else '不准入'}"
                           f"｜{relay.get('summary_line', '')}")
        if relay.get("relay_gap"):
            extra_stats.append("高级面缺口（不阻断）：" + "、".join(
                f"{item['case_id']}({item['face']})" for item in relay["relay_gap"]))
        if relay.get("vision_undeclared"):
            extra_stats.append("视觉未开放（矩阵声明无图像输入）：" + "、".join(relay["vision_undeclared"]))
        if summary.get("case_filter"):
            extra_stats.append("定向复测用例：" + "、".join(summary["case_filter"]))
    tier = summary.get("tier_assessment") or {}
    if tier.get("tested"):
        extra_stats += [f"档位指纹结论：{tier.get('level', '未执行')}",
                        f"已测指纹：{tier.get('tested', 0)} 项；不一致/异常："
                        f"{tier.get('warnings', 0) + tier.get('hard_failures', 0)} 项"]
    if summary.get("admission_blocked"):
        extra_stats.append(f"明确不准入的用例：{', '.join(summary['admission_blocked'])}")
    report = case_table_report(
        "千问（Qwen 3.6 / 3.7 / 3.8）专项测试报告",
        total=summary["total"], passed=summary["passed"], failed=summary["failed"],
        warning=summary.get("warning", 0), blocked=summary.get("blocked", 0),
        pass_rate=summary.get("pass_rate"), extra_stats=extra_stats, rows=rows, status_column=2)
    (out / "report.md").write_text(report, encoding="utf-8")
