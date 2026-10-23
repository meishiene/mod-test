"""中转计费探针的**模拟上游**（仅测试用）。

用途有两处，避免两套 mock 各自漂移：
1. `tests/test_relay_billing_probe.py` 用 `httpx.MockTransport` 挂上去做判定分支测试；
2. 端到端验证时由 `/tmp` 里的 HTTP 包装器 import 本模块，起一个真·HTTP 上游。

模式矩阵是刻意设计的**两个正交轴**（计费口径 × 用量真实性），这样才能证明探针
能把它们分别判出来：

| 模式 | 上报 usage | 实际扣款 | 期望结论 |
|---|---|---|---|
| `honest` | 实测 | 按实测用量 | 全通过 |
| `fake_usage` | 伪造＝max_tokens | 按真实用量 | U01/U02/U03 不通过，B01/B02 通过 |
| `cap_charge` | 实测 | 按下单上限 | U 系列通过，B01/B02 不通过 |
| `both` | 伪造＝max_tokens | 按下单上限 | 全部不通过（复刻 i-token.ai） |
| `no_oracle` | 实测 | 按实测用量 | 计费类 blocked（余额口径不可得），U 系列通过 |
| `channel_down` | — | — | 全部 blocked |

两个额外开关（默认关闭，按真实批次校准）：

- `zero_thinking=True` 复刻 **2026-10-09 `claude-sonnet-5` 批次**
  （`results/RELAY-BILLING-PROBE-20261009-111647`）：`content` 只有 `text` 块、
  `usage.output_tokens_details.thinking_tokens` **恒为 0（但 key 存在）** —— 用于检验
  "key 存在但值为 0" 不会被误读成"仍带思考计费"。
- `transient_5xx=N` 让前 N 次请求返回 `502`（空 body，复刻该批次的 `server: Caddy` 网关故障），
  用于检验瞬时故障重试与 blocked 文案是否带上网关状态。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

MODE_HONEST = "honest"
MODE_FAKE_USAGE = "fake_usage"
MODE_CAP_CHARGE = "cap_charge"
MODE_BOTH = "both"
MODE_NO_ORACLE = "no_oracle"
MODE_CHANNEL_DOWN = "channel_down"

MODES = (MODE_HONEST, MODE_FAKE_USAGE, MODE_CAP_CHARGE, MODE_BOTH, MODE_NO_ORACLE, MODE_CHANNEL_DOWN)

LONG_MARKER = "数字 1 到 300"
LONG_ANSWER = " ".join(str(index) for index in range(1, 301))
SHORT_ANSWER = "OK"


@dataclass
class FakeRelay:
    """一个「可以被精确控制口径」的中转上游。"""

    mode: str = MODE_HONEST
    initial_balance: float = 10.0
    rate_per_m: float = 50.0
    dashboard: bool = True
    model: str = "claude-fable-5"
    protocol: str = "anthropic"
    small_balance_floor: float = 0.0
    zero_thinking: bool = False
    transient_5xx: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)
    used_usd: float = 0.0

    # ---------------- 内部口径 ---------------- #
    @property
    def honoring_thinking_switch(self) -> bool:
        """伪造 usage 的通道通常连思考都关不掉。"""
        return self.mode in (MODE_HONEST, MODE_CAP_CHARGE, MODE_NO_ORACLE)

    @property
    def faking_usage(self) -> bool:
        return self.mode in (MODE_FAKE_USAGE, MODE_BOTH)

    @property
    def faking_input(self) -> bool:
        return self.mode == MODE_BOTH

    @property
    def charging_by_cap(self) -> bool:
        return self.mode in (MODE_CAP_CHARGE, MODE_BOTH)

    def reserve(self, max_tokens: int) -> float:
        return max_tokens * self.rate_per_m / 1_000_000

    def _real_output_tokens(self, prompt: str) -> int:
        return 600 if LONG_MARKER in prompt else 6

    # ---------------- 路由 ---------------- #
    def handle(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        self.calls.append({"method": method, "path": path, "body": body or {}})
        if self.transient_5xx > 0:
            # 复刻 i-token.ai 2026-10-09 批次：网关侧 502、空 body、server: Caddy
            self.transient_5xx -= 1
            return 502, {}
        if self.mode == MODE_CHANNEL_DOWN:
            return 401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid api key"}}
        path = path.split("?", 1)[0]
        normalized = re.sub(r"^/v\d+", "", path)
        if self.mode == MODE_NO_ORACLE and normalized.startswith("/dashboard/billing"):
            return 404, {"type": "error", "error": {"type": "not_found_error", "message": "Invalid URL"}}
        if not self.dashboard and normalized.startswith("/dashboard/billing"):
            return 404, {"type": "error", "error": {"type": "not_found_error", "message": "Invalid URL"}}
        if normalized == "/models":
            return 200, {"data": [{"id": self.model}, {"id": "claude-opus-5"}]}
        if normalized == "/dashboard/billing/subscription":
            return 200, {"hard_limit_usd": self.initial_balance}
        if normalized == "/dashboard/billing/usage":
            return 200, {"total_usage": round(self.used_usd * 100, 6)}
        if normalized in ("/messages", "/chat/completions"):
            return self._message(body or {})
        return 404, {"type": "error", "error": {"type": "not_found_error", "message": f"unknown path {path}"}}

    def _message(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        max_tokens = int(body.get("max_tokens") or 0)
        messages = body.get("messages") or []
        prompt = ""
        for item in messages:
            content = item.get("content")
            if isinstance(content, str):
                prompt += content
            elif isinstance(content, list):
                prompt += "".join(part.get("text", "") for part in content if isinstance(part, dict))
        remaining = self.initial_balance - self.used_usd
        if self.reserve(max_tokens) > remaining:
            need = self.reserve(max_tokens)
            return 403, {"type": "error", "error": {
                "type": "i_token_error",
                "message": (f"token quota is not enough, token remain quota: ${remaining:.6f}, "
                            f"need quota: ${need:.6f} (request id: 202610081119049139355368268d9d6oS3swKcA)")}}

        disabled = isinstance(body.get("thinking"), dict) and body["thinking"].get("type") == "disabled"
        real_out = self._real_output_tokens(prompt)
        real_think = 0 if self.zero_thinking else real_out // 2
        text = LONG_ANSWER if LONG_MARKER in prompt else SHORT_ANSWER
        truncated = real_out > max_tokens
        if truncated:
            text = text[: max(1, max_tokens * 2)]

        if self.faking_usage:
            reported_out, reported_think = max_tokens, max(1, max_tokens - 1)
            stop = "end_turn"
        elif disabled and self.honoring_thinking_switch:
            reported_out, reported_think = max(2, real_out // 3), None
            stop = "end_turn"
        else:
            reported_out, reported_think = min(real_out, max_tokens), real_think
            stop = "max_tokens" if truncated else "end_turn"

        charged = self.reserve(max_tokens) if self.charging_by_cap else real_out * self.rate_per_m / 1_000_000
        self.used_usd += charged

        if self.faking_input:
            input_tokens = 10
        else:
            input_tokens = 12 + len(prompt) // 3

        if self.protocol == "openai":
            usage: dict[str, Any] = {"prompt_tokens": input_tokens, "completion_tokens": reported_out}
            if reported_think is not None:
                usage["completion_tokens_details"] = {"reasoning_tokens": reported_think}
            return 200, {"id": "chatcmpl-mock", "model": self.model,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                      "finish_reason": stop}],
                         "usage": usage}
        usage = {"input_tokens": input_tokens, "output_tokens": reported_out,
                 "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        if reported_think is not None:
            usage["output_tokens_details"] = {"thinking_tokens": reported_think}
        content: list[dict[str, Any]] = []
        if reported_think and not (disabled and self.honoring_thinking_switch):
            content.append({"type": "thinking", "thinking": "", "signature": "mock"})
        content.append({"type": "text", "text": text})
        return 200, {"id": "msg_mock", "type": "message", "role": "assistant", "model": self.model,
                     "content": content, "stop_reason": stop, "stop_sequence": None, "usage": usage}

    # ---------------- 给测试用的适配 ---------------- #
    def as_httpx_handler(self):
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            body = None
            if request.content:
                try:
                    body = json.loads(request.content.decode())
                except (ValueError, UnicodeDecodeError):
                    body = None
            status, payload = self.handle(request.method, request.url.path, body)
            return httpx.Response(status, json=payload, headers={"content-type": "application/json"})

        return handler
