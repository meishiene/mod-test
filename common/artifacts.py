"""Common result/artifact contract used by load, capability and benchmark suites."""
from __future__ import annotations
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "mltest.result.v3"
STATUSES = {"passed", "failed", "warning", "blocked", "error", "not_evaluable", "interrupted", "skipped"}

def jsonable(value: Any) -> Any:
    if is_dataclass(value): return jsonable(asdict(value))
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    return value

def result_envelope(*, record_id: str | None, suite: str, model: str | None, protocol: str | None,
                    records: list[dict[str, Any]], summary: dict[str, Any] | None = None,
                    status: str = "completed", **metadata: Any) -> dict[str, Any]:
    if status not in {"completed", "running", "interrupted", "failed"}: raise ValueError(status)
    summary = summary or {}
    return {"schema_version": SCHEMA_VERSION, "record_id": record_id, "suite": suite,
            "model": model, "protocol": protocol, "status": status,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": jsonable(summary), "records": jsonable(records), **jsonable(metadata)}

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as fp:
        fp.write(json.dumps(jsonable(value), ensure_ascii=False, default=str) + "\n"); fp.flush()
