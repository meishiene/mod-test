"""Public shared runner API.

Suites provide a request builder and validators; this module owns transport,
logging and the invariant that verdicts come only from validators.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable
from .artifacts import append_jsonl
from .test_framework import HttpExecutor, ExecutionRecord, validate_case
from .audit import redact

class Runner:
    def __init__(self, *, output: Path, timeout: float = 300, max_connections: int = 100, verify_ssl: bool = True, trust_env: bool = False):
        self.output = output; self.executor = HttpExecutor(timeout=timeout, max_connections=max_connections, verify_ssl=verify_ssl, trust_env=trust_env)
        self.log_path = output / "execution.jsonl"
    async def __aenter__(self): await self.executor.__aenter__(); return self
    async def __aexit__(self, *args): await self.executor.__aexit__(*args)
    async def execute(self, *, case_id: str, url: str, payload: dict[str, Any], headers: dict[str, str], stream: bool, validators: list[Callable] | None = None) -> ExecutionRecord:
        record: ExecutionRecord = await self.executor.execute(case_id=case_id, url=url, payload=payload, headers=headers, stream=stream)
        if validators is not None:
            record.validation = validate_case(record, validators)
        append_jsonl(self.log_path, record)
        return record

    async def request(self, *, case_id: str, url: str, payload: dict[str, Any], headers: dict[str, str],
                     validators: list[Callable], stream: bool) -> dict[str, Any]:
        record = await self.execute(case_id=case_id, url=url, payload=payload, headers=headers, stream=stream, validators=validators)
        result = record.validation
        return {"case_id": case_id, "case_status": result.status, "case_passed": result.passed,
                "checks": [asdict(check) for check in result.checks], "failure_reasons": result.reasons,
                "execution": redact(record.jsonable())}
