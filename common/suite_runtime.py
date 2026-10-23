"""Small integration layer for existing suite runners.

It lets legacy/domain runners keep their scheduling and official harnesses while
using the common evidence and finalization contracts.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .audit import AuditLog
from .finalizer import finalize_result_dir

class SuiteArtifacts:
    def __init__(self, directory: Path):
        self.directory=directory; directory.mkdir(parents=True, exist_ok=True)
        self.execution=AuditLog(directory/"execution.jsonl")
    def record(self, item: Any) -> None: self.execution.write(item)
    def close(self) -> None: self.execution.close()
    def finalize(self, *, summary: dict[str,Any] | None = None) -> dict[str,Any]:
        if summary is not None:
            (self.directory/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
        self.close()
        return finalize_result_dir(self.directory)

def finalize(directory: Path, summary: dict[str,Any] | None = None) -> dict[str,Any]:
    if summary is not None:
        (directory/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
    return finalize_result_dir(directory)
