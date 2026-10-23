from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ApiConfig:
    api_key: str
    base_url: str
    model: str
    chat_path: str = "/chat/completions"
    request_timeout_seconds: float = 300.0
    connect_timeout_seconds: float = 30.0
    max_connections: int = 1200
    verify_ssl: bool = True
    trust_env: bool = False
    failure_backoff_seconds: float = 2.0
    failure_backoff_max_seconds: float = 30.0
    worker_start_stagger_seconds: float = 0.1

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.chat_path.lstrip('/')}"

    @classmethod
    def from_env(cls) -> "ApiConfig":
        missing = [name for name in ("K3_API_KEY", "K3_BASE_URL", "K3_MODEL") if not os.getenv(name)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        return cls(
            api_key=os.environ["K3_API_KEY"],
            base_url=os.environ["K3_BASE_URL"],
            model=os.environ["K3_MODEL"],
            chat_path=os.getenv("K3_CHAT_PATH", "/chat/completions"),
            request_timeout_seconds=float(os.getenv("K3_REQUEST_TIMEOUT_SECONDS", "300")),
            connect_timeout_seconds=float(os.getenv("K3_CONNECT_TIMEOUT_SECONDS", "30")),
            max_connections=int(os.getenv("K3_MAX_CONNECTIONS", "1200")),
            verify_ssl=os.getenv("K3_VERIFY_SSL", "true").lower() not in {"0", "false", "no"},
            trust_env=os.getenv("K3_TRUST_ENV", "false").lower() in {"1", "true", "yes"},
            failure_backoff_seconds=float(os.getenv("K3_FAILURE_BACKOFF_SECONDS", "2")),
            failure_backoff_max_seconds=float(os.getenv("K3_FAILURE_BACKOFF_MAX_SECONDS", "30")),
            worker_start_stagger_seconds=float(os.getenv("K3_WORKER_START_STAGGER_SECONDS", "0.1")),
        )
