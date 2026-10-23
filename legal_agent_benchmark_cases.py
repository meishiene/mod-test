"""Harvey Legal Agent Benchmark official open benchmark metadata."""
from dataclasses import dataclass
@dataclass(frozen=True)
class LegalAgentBenchmarkCase:
    case_id: str = 'LEGAL_AGENT_BENCHMARK'
    title: str = 'Legal Agent Benchmark·法律工作 Agent'
    description: str = '使用 Harvey 官方开源 Legal Agent Benchmark（Harvey Labs）任务、材料、rubric、harness 与 evaluator。'
    repository: str = 'https://github.com/harveyai/harvey-labs/tree/v1.0'
    protocols: tuple[str,...] = ('openai','volcengine')
    cost: str = '极高'
CASE=LegalAgentBenchmarkCase()
