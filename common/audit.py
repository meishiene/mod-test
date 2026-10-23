"""统一的请求证据记录与结果封装。

所有测试集都可以把任意 transport record 写入同一份 JSONL；日志实时 flush，
中断时已完成的请求仍可审计。敏感认证头默认脱敏。
"""
from __future__ import annotations
import asyncio, json, threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

SENSITIVE_HEADERS = {"authorization", "x-api-key", "api-key", "proxy-authorization", "cookie", "set-cookie"}

def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("***" if str(k).lower() in SENSITIVE_HEADERS else redact(v)) for k, v in value.items()}
    if isinstance(value, list): return [redact(v) for v in value]
    return value

def to_jsonable(value: Any) -> Any:
    if is_dataclass(value): return to_jsonable(asdict(value))
    if isinstance(value, dict): return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [to_jsonable(v) for v in value]
    return value

class AuditLog:
    """线程/协程安全的实时 JSONL 证据日志。"""
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True); self.path = path
        self.fp = path.open("w", encoding="utf-8", buffering=1)
        self.lock = threading.Lock()
    def write(self, record: Any) -> None:
        obj = redact(to_jsonable(record))
        with self.lock:
            self.fp.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n"); self.fp.flush()
    async def awrite(self, record: Any) -> None:
        await asyncio.to_thread(self.write, record)
    def close(self) -> None: self.fp.close()

class JsonlWriter:
    """兼容旧 runner 的 write(dict)/close 接口。"""
    def __init__(self, path: Path): self.log = AuditLog(path)
    def write(self, row: dict[str, Any]) -> None: self.log.write(row)
    def close(self) -> None: self.log.close()
