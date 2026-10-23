"""Explicit, bounded workload profiles; these are test budgets, not model limits."""
from dataclasses import dataclass
import re

from .probe_assertions import normalized_model

VERSION = 'economy-validation-1'
FAMILIES = {'deepseek': 'DeepSeek', 'glm': 'GLM', 'qwen': '千问',
            'kvv': 'Kimi K3', 'fable': 'Claude', 'gpt': 'GPT'}
ECONOMY_CASES = {f'{kind.upper()}_ECONOMY': kind for kind in FAMILIES}


@dataclass(frozen=True)
class EconomyProfile:
    family: str
    model: str
    transport: str
    max_requests: int = 11
    max_output_tokens: int = 2048
    max_input_chars: int = 64000
    context_probe_chars: int = 8192
    challenge_samples: int = 3

    @property
    def request_options(self):
        if self.family == 'glm':
            return {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'low'}
        if self.family == 'qwen':
            return {'enable_thinking': False}
        if self.family == 'deepseek':
            return {'thinking': {'type': 'disabled'}}
        if self.family == 'kvv':
            return {'thinking': {'type': 'enabled', 'effort': 'low', 'keep': 'all'}}
        if self.family == 'gpt':
            return {'reasoning': {'effort': 'low'}}
        from fable5.model_profile import resolve_thinking_mode, ENABLED
        if resolve_thinking_mode(self.model) == ENABLED:
            # Objective answers do not require a signed thinking trace.
            return {}
        return {'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'low'}}


def economy_profile(kind: str, model: str) -> EconomyProfile:
    name = normalized_model(model)
    if kind in {'glm', 'qwen'}:
        from .model_validation import validation_profile
        p = validation_profile(name)
        valid = p is not None and p.family == kind
    elif kind == 'deepseek':
        valid = bool(re.fullmatch(r'deepseek-(?:v4[.\w-]*|flash|pro)', name))
    elif kind == 'kvv':
        valid = bool(re.fullmatch(r'(?:kimi-)?k3(?:[.-][a-z0-9-]+)?', name))
    elif kind == 'gpt':
        valid = bool(re.fullmatch(r'gpt-(?:5\.6(?:-(?:sol|terra|luna))?|6-(?:sol|astra)|6\.1(?:-sol)?)', name))
    elif kind == 'fable':
        from fable5.model_profile import normalize_model_id
        name = normalize_model_id(name)
        valid = bool(re.fullmatch(r'claude-(?:opus|sonnet|haiku|fable)-(?:4-[56]|5)(?:-[\w-]+)?', name))
    else:
        valid = False
    if not valid:
        raise ValueError(f'经济验证未登记该型号配置：{kind} / {model}')
    transport = 'messages' if kind == 'fable' else 'responses' if kind == 'gpt' else 'chat'
    return EconomyProfile(kind, model, transport)


def economy_case_payload(case_id: str) -> dict:
    kind = ECONOMY_CASES[case_id]
    return {'id': case_id, 'test_kind': kind, 'title': f'{FAMILIES[kind]} 经济能力验证',
            'description': '客观推理、代码、结构化抽取、8K字符记忆、工具恢复/并行、高级约束题；复用响应验型号，追加路由反例。不跑压测或高成本专项。',
            'cost': '低（最多11次请求；单次输出上限2048 tokens）',
            'duration': '按11次请求上限执行', 'request_count': 11,
            'requests': 11, 'request_text': '最多11次；并发1；不自动重试', 'concurrency': 1,
            'output_tokens': '单次≤2048；整套请求上限合计≤20544',
            'input_tokens': '按字符限额：累计序列化请求≤64000字符（非token数）',
            'protocols': ['anthropic'] if kind == 'fable' else ['openai', 'volcengine'],
            'model_scope': True, 'economy_validation': True,
            'model_scope_hint': '每个所选模型独立执行，最多11次请求；选择多个模型会按数量增加总成本。',
            'executor': 'scripts/run_economy_validation.py',
            'source': '本地客观题集 economy-validation-1',
            'concurrency_text': '串行1；不自动重试',
            'warmup': None, 'steady': None,
            'admission': '仅验证经济题集；缺证据不通过，不证明全部能力满血或厂商身份。',
            'always_visible': True, 'faces': None}
