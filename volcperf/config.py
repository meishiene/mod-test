from __future__ import annotations

import os
from dataclasses import dataclass

_TRUTHY = {"1", "true", "yes", "on"}


def trust_env_enabled() -> bool:
    """是否允许 httpx 读取环境中的代理变量（由 .env 的 VOLC_TRUST_ENV 控制）。

    默认 false —— 平台直连目标地址。判定逻辑集中在这里，是为了让 Web 层的
    连通性探针、模型列表查询与正式压测共用同一份开关，避免"模型能查到、
    连通性检查却走直连"这类前后不一致。
    """
    return os.getenv("VOLC_TRUST_ENV", "false").lower() in _TRUTHY


@dataclass(frozen=True)
class ApiConfig:
    api_key: str
    base_url: str
    model: str
    protocol: str = "volcengine"
    chat_path: str = "/chat/completions"
    request_timeout_seconds: float = 300.0
    connect_timeout_seconds: float = 30.0
    max_connections: int = 1200
    verify_ssl: bool = True
    trust_env: bool = False
    include_usage: bool = True
    reasoning_effort: str | None = None

    @property
    def endpoint(self) -> str:
        from .protocols import endpoint
        return endpoint(self.protocol, self.base_url, self.model, self.chat_path)

    @classmethod
    def from_env(cls) -> "ApiConfig":
        required = ("VOLC_API_KEY", "VOLC_BASE_URL", "VOLC_MODEL")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise ValueError(f"缺少环境变量: {', '.join(missing)}")
        falsy = {"0", "false", "no", "off"}
        verify = os.getenv("VOLC_VERIFY_SSL", "true").lower() not in falsy
        return cls(
            api_key=os.environ["VOLC_API_KEY"],
            base_url=os.environ["VOLC_BASE_URL"],
            model=os.environ["VOLC_MODEL"],
            protocol=os.getenv("VOLC_PROTOCOL", "volcengine").lower(),
            chat_path=os.getenv("VOLC_CHAT_PATH", "/chat/completions"),
            request_timeout_seconds=float(os.getenv("VOLC_REQUEST_TIMEOUT_SECONDS", "300")),
            connect_timeout_seconds=float(os.getenv("VOLC_CONNECT_TIMEOUT_SECONDS", "30")),
            max_connections=int(os.getenv("VOLC_MAX_CONNECTIONS", "1200")),
            verify_ssl=verify,
            trust_env=trust_env_enabled(),
            include_usage=os.getenv("VOLC_STREAM_INCLUDE_USAGE", "true").lower() not in falsy,
            reasoning_effort=(os.getenv("VOLC_REASONING_EFFORT") or "").strip() or None,
        )
