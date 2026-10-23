#!/usr/bin/env python3
"""Claude Messages 面探测脚本的**共用基建**（HTTP 通道 / 判定对象 / 状态词表）。

本模块是「Claude 测试」模块下自包含签名校验 CLI 的公共底座：

  fable5/proto_signature_check.py  PROTOBUF_SIGNATURE_CHECK 签名两步校验

两者此前共用 ``fable5/protobuf_check.py`` 里的通道与判定对象；PB/PBR 编解码用例
移除后，这部分基建独立成本模块，避免重复实现与口径分叉（**唯一权威实现**）。

口径约束（沿用项目既有纪律）：

- **瞬时故障先重试再 blocked，绝不判 failed**：5xx 与连接类超时按指数退避重试
  （默认 2 次），耗尽后由调用方记 ``blocked``；本模块只负责重试并如实回报
  ``attempts`` / ``transport_error``，不做判定。
- **不跟随重定向**：避免被 3xx 跳转掩盖上游真实行为。
- 判定对象 ``Verdict`` 的 ``scope`` 区分 ``endpoint``（计入端点准入）与
  ``harness``（工具自身负控，**不计入准入**）——两者不得混淆。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field as dc_field
from typing import Any

import httpx

from volcperf.protocols import api_root

__all__ = [
    "STATUS_TIERS", "PASSED", "FAILED", "WARNING", "BLOCKED", "SKIPPED", "INFO",
    "Resp", "Client", "Verdict", "verdict", "JSON_CT",
]

# ---------------------------------------------------------------------------
# 状态词表（全平台四档 + 工具侧 skipped/info）
# ---------------------------------------------------------------------------

# 平台判定词表：这四档参与准入统计（warning/blocked 只记录不阻断）。
STATUS_TIERS = ("passed", "failed", "warning", "blocked")
PASSED, FAILED, WARNING, BLOCKED, SKIPPED, INFO = (
    "passed", "failed", "warning", "blocked", "skipped", "info",
)
# 平台词表里没有 info；收尾拍平时会把它压成 skipped（见各脚本的 summary 组装）。

JSON_CT = "application/json"


# ---------------------------------------------------------------------------
# 一、HTTP 通道（带瞬时故障重试；不跟随重定向）
# ---------------------------------------------------------------------------

@dataclass
class Resp:
    status: int
    headers: dict[str, str]
    content: bytes
    elapsed: float
    attempts: int = 1
    transport_error: str | None = None

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "")

    def json(self) -> dict[str, Any] | None:
        try:
            parsed = json.loads(self.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    @property
    def text_head(self) -> str:
        return self.content[:200].decode("utf-8", "replace")


_TRANSIENT_EXC = (
    httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
    httpx.WriteTimeout, httpx.PoolTimeout, httpx.ReadError, httpx.RemoteProtocolError,
)


class Client:
    """Anthropic Messages 面客户端；``transport`` 可注入，便于用 MockTransport 测试。"""

    def __init__(self, base_url: str, api_key: str, *, model: str, auth: str = "x-api-key",
                 anthropic_version: str = "2023-06-01", timeout: float = 300.0,
                 max_retries: int = 2, backoff: float = 1.0, trust_env: bool = False,
                 verify_ssl: bool = True, transport: httpx.BaseTransport | None = None) -> None:
        self.root = api_root("anthropic", base_url)
        self.url = f"{self.root}/messages"
        self.api_key = api_key
        self.model = model
        self.auth = auth
        self.anthropic_version = anthropic_version
        self.max_retries = max_retries
        self.backoff = backoff
        self.client = httpx.Client(timeout=timeout, verify=verify_ssl, trust_env=trust_env,
                                   follow_redirects=False, transport=transport)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _headers(self, accept: str, content_type: str,
                 extra: dict[str, str] | None = None) -> dict[str, str]:
        rid = f"PROBE-{os.urandom(8).hex()}"
        head = {
            "Content-Type": content_type,
            "Accept": accept,
            "X-Request-ID": rid,
            "anthropic-version": self.anthropic_version,
        }
        if self.auth == "bearer":
            head["Authorization"] = f"Bearer {self.api_key}"
        else:
            head["x-api-key"] = self.api_key
        if extra:
            head.update(extra)
        return head

    def post(self, body: bytes, *, accept: str, content_type: str = JSON_CT,
             extra_headers: dict[str, str] | None = None,
             retry_5xx: bool = True, retry_transport: bool | None = None) -> Resp:
        """POST /messages。``retry_5xx=False`` 供**负向变异**请求使用：瞬时 5xx 若被重试成
        200，会把"该拒未拒"误判成 failed（假失败），故一次定型、由调用方分派。

        ``retry_transport`` 单独控制**连接类异常**是否重试（默认跟随 ``retry_5xx``）。
        负向变异请求用 ``retry_5xx=False, retry_transport=True``：重发同一请求结果语义
        不变（断开只是"没拿到答案"，不是"上游放行"），但 5xx 绝不重试成 200。
        """
        headers = self._headers(accept, content_type, extra_headers)
        retry_transport = retry_5xx if retry_transport is None else retry_transport
        attempts, started = 0, time.monotonic()
        while True:
            attempts += 1
            try:
                resp = self.client.post(self.url, headers=headers, content=body)
                if resp.status_code >= 500 and retry_5xx and attempts <= self.max_retries:
                    time.sleep(self.backoff * (2 ** (attempts - 1)))
                    continue
                return Resp(resp.status_code, dict(resp.headers), resp.content,
                            time.monotonic() - started, attempts)
            except _TRANSIENT_EXC as exc:
                if not retry_transport or attempts > self.max_retries:
                    return Resp(0, {}, b"", time.monotonic() - started, attempts,
                                f"{type(exc).__name__}: {exc}")
                time.sleep(self.backoff * (2 ** (attempts - 1)))

    def get(self, path: str, *, accept: str = JSON_CT) -> Resp:
        """GET 探测（如 /models 模型列表佐证）：尽力一次、不重试，只作留痕不定判据。"""
        headers = self._headers(accept, JSON_CT)
        started = time.monotonic()
        try:
            resp = self.client.get(f"{self.root}{path}", headers=headers)
            return Resp(resp.status_code, dict(resp.headers), resp.content,
                        time.monotonic() - started)
        except _TRANSIENT_EXC as exc:
            return Resp(0, {}, b"", time.monotonic() - started, 1,
                        f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 二、判定对象
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    case_id: str
    title: str
    group: str                      # 用例分组（各脚本自定，如 request/response/harness）
    status: str
    detail: str
    observations: dict[str, Any] = dc_field(default_factory=dict)
    scope: str = "endpoint"         # endpoint | harness（harness 不计入端点准入）
    duration_seconds: float = 0.0   # 该用例主请求耗时（秒），供 Web 用例明细列展示

    @property
    def passed_flag(self) -> bool | None:
        if self.status == PASSED:
            return True
        if self.status in (FAILED,):
            return False
        return None

    def as_dict(self) -> dict[str, Any]:
        # name/details 是 web 端用例明细表的通用字段名（与 authenticity 等套件契约
        # 一致）；title/detail 保留为本模块报告与测试的内部叫法，两套别名同时输出。
        return {
            "case_id": self.case_id, "name": self.title, "title": self.title,
            "group": self.group, "status": self.status, "passed": self.passed_flag,
            "scope": self.scope, "detail": self.detail, "details": self.detail,
            "duration_seconds": self.duration_seconds, "observations": self.observations,
        }


def verdict(case_id: str, title: str, group: str, status: str, detail: str,
            obs: dict[str, Any] | None = None, scope: str = "endpoint") -> Verdict:
    return Verdict(case_id, title, group, status, detail, obs or {}, scope)


# 各脚本沿用既有短名 `_v`，这里给出别名以保持调用点不变。
_v = verdict
