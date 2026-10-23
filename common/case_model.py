"""跨测试集的案例定义协议。旧案例可通过 adapter 映射到该模型。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
from .verdict import Check, Verdict, verdict
@dataclass(frozen=True)
class CaseDefinition:
    case_id: str; name: str; category: str = "general"; priority: str = "P1"
    request: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    validator: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
@dataclass
class CaseOutcome:
    case: CaseDefinition; execution: dict[str, Any]; validation: Verdict
    def jsonable(self): return {"case": self.case.__dict__, "execution": self.execution, "validation": self.validation.jsonable()}
Validator = Callable[[CaseDefinition, dict[str, Any]], list[Check]]
def evaluate(case: CaseDefinition, execution: dict[str, Any], validators: list[Validator]) -> CaseOutcome:
    checks=[]
    for fn in validators: checks.extend(fn(case, execution))
    result=verdict(checks); return CaseOutcome(case, execution, result)
