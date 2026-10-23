"""Low-cost objective probes shared across six model families.

All network attempts are counted before sending. No automatic retry, paid tools,
model-generated exec, or silent output/context budget increase.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

import httpx

from .audit import AuditLog
from .economy_profiles import EconomyProfile, VERSION
from .model_probes import ModelProbeMixin, answer, outcome, row
from .probe_assertions import identity_match, strict_json


def encode_request(profile, model, messages, options):
    """Translate canonical Chat history, retaining native signed/reasoning items."""
    options = dict(options)
    cap = options.pop('max_tokens')
    tools = options.pop('tools', None)
    response_format = options.pop('response_format', None)
    base = {'model': model, **profile.request_options, **options}
    if profile.transport == 'chat':
        base.update(messages=[{k: v for k, v in m.items() if not k.startswith('_economy')} for m in messages], max_tokens=cap)
        if tools: base['tools'] = tools
        if response_format: base['response_format'] = response_format
        return 'chat/completions', base
    if profile.transport == 'responses':
        inputs, instructions = [], []
        for m in messages:
            if m['role'] == 'system':
                instructions.append(m['content'])
            elif m['role'] == 'tool':
                inputs.append({'type': 'function_call_output', 'call_id': m['tool_call_id'], 'output': m['content']})
            elif m.get('_economy_native') is not None:
                inputs.extend(m['_economy_native'])
            else:
                inputs.append({'role': m['role'], 'content': m.get('content') or ''})
        base.update(input=inputs, max_output_tokens=cap, store=False, include=['reasoning.encrypted_content'])
        if instructions: base['instructions'] = '\n'.join(instructions)
        if tools:
            base['tools'] = [{'type': 'function', **t['function']} for t in tools]
        if response_format: base['text'] = {'format': response_format}
        return 'responses', base
    converted, system = [], []
    for m in messages:
        if m['role'] == 'system':
            system.append(m['content'])
            continue
        role = 'user' if m['role'] == 'tool' else m['role']
        if m['role'] == 'tool':
            blocks = [{'type': 'tool_result', 'tool_use_id': m['tool_call_id'], 'content': m['content']}]
        elif m.get('_economy_native') is not None:
            blocks = m['_economy_native']
        else:
            blocks = [{'type': 'text', 'text': m.get('content') or ''}]
        if converted and converted[-1]['role'] == role:
            converted[-1]['content'].extend(blocks)
        else:
            converted.append({'role': role, 'content': list(blocks)})
    base.update(messages=converted, max_tokens=cap)
    if system: base['system'] = '\n'.join(system)
    if tools:
        base['tools'] = [{'name': t['function']['name'], 'description': t['function']['description'],
                          'input_schema': t['function']['parameters']} for t in tools]
    return 'messages', base


def decode_response(transport, body):
    """Return canonical response and completion status. Invalid envelopes are gaps."""
    if not isinstance(body, dict): return None, 'invalid_envelope'
    if transport == 'chat':
        choices = body.get('choices')
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return None, 'invalid_envelope'
        choice = choices[0]
        if choice.get('finish_reason') == 'length': return None, 'output_budget_exhausted'
        if choice.get('finish_reason') not in ('stop', 'tool_calls', 'function_call', 'content_filter'):
            return None, 'invalid_envelope'
        if not isinstance(choice.get('message'), dict): return None, 'invalid_envelope'
        return body, None
    if transport == 'responses':
        if body.get('status') == 'incomplete': return None, 'output_budget_exhausted'
        if body.get('status') != 'completed' or not isinstance(body.get('output'), list):
            return None, 'invalid_envelope'
        native = body['output']
        text = ''.join(c.get('text', '') for item in native if item.get('type') == 'message'
                       for c in item.get('content', []) if c.get('type') == 'output_text')
        calls = [{'id': item.get('call_id'), 'type': 'function',
                  'function': {'name': item.get('name'), 'arguments': item.get('arguments')}}
                 for item in native if item.get('type') == 'function_call']
    else:
        if body.get('stop_reason') == 'max_tokens': return None, 'output_budget_exhausted'
        if body.get('stop_reason') not in ('end_turn', 'tool_use', 'stop_sequence', 'refusal'):
            return None, 'incomplete_message'
        if body.get('type') != 'message' or not isinstance(body.get('content'), list):
            return None, 'invalid_envelope'
        native = body['content']
        text = ''.join(item.get('text', '') for item in native if item.get('type') == 'text')
        calls = [{'id': item.get('id'), 'type': 'function',
                  'function': {'name': item.get('name'), 'arguments': json.dumps(item.get('input'))}}
                 for item in native if item.get('type') == 'tool_use']
    usage = body.get('usage') or {}
    msg = {'role': 'assistant', 'content': text, '_economy_native': native}
    if calls: msg['tool_calls'] = calls
    return {'model': body.get('model'), 'choices': [{'message': msg}],
            'usage': {'prompt_tokens': usage.get('input_tokens'), 'completion_tokens': usage.get('output_tokens')}}, None


CHECKS = (
    ('Q01', '三题客观推理', 'ability', 'thinking', 'quality_reasoning'),
    ('Q02', '代码生成与隔离执行', 'ability', 'quality', 'quality_code'),
    ('Q03', '中文抽取与严格结构', 'ability', 'quality', 'quality_schema'),
    ('Q04', '8K字符头中尾召回', 'ability', 'context', 'quality_context'),
    ('Q05', '工具错误恢复与真实结果利用', 'full_strength', 'tools', 'quality_agent'),
    ('Q06', '双工具精确参数', 'full_strength', 'tools', 'quality_parallel'),
    ('R03', '不可信文档中的指令隔离', 'ability', 'quality', 'verify_instruction'),
    ('Q07', '高级约束任务正确性', 'full_strength', 'quality', 'quality_retention'),
    ('R02', '未知型号路由反例', 'authenticity', 'identity', 'economy_route'),
    ('R01', '复用响应的精确型号一致性', 'authenticity', 'identity', 'economy_identity'),
)


class EconomyRunner(ModelProbeMixin):
    def __init__(self, profile: EconomyProfile, base_url: str, api_key: str, directory: Path,
                 *, client: httpx.AsyncClient, seed: int = 20261023):
        self.profile, self.base_url, self.api_key = profile, base_url.rstrip('/'), api_key
        self.directory, self.client = directory, client
        self.config = SimpleNamespace(seed=seed)
        self.requests = self.input_chars = self.requested_output_tokens = 0
        self.responses = []
        self.current_case = ''
        self.audit = AuditLog(directory / 'requests.jsonl')
        self.gaps = []

    def probe_profile(self, model):
        return self.profile

    def probe_meta(self, **extra):
        return {'workload_version': VERSION, 'seed': self.probe_seed(), **extra}

    def matches_model(self, returned):
        if self.profile.family == 'fable':
            from fable5.model_profile import same_model_id
            return same_model_id(returned, self.profile.model)
        return identity_match(self.profile.model, returned)

    async def probe_request(self, model, messages, **kwargs):
        # Inherited probes may request more; this workload never silently escalates.
        kwargs['max_tokens'] = min(kwargs.get('max_tokens', self.profile.max_output_tokens), self.profile.max_output_tokens)
        path, payload = encode_request(self.profile, model, messages, kwargs)
        encoded = json.dumps(payload, ensure_ascii=False)
        if self.requests >= self.profile.max_requests or self.input_chars + len(encoded) > self.profile.max_input_chars:
            self.gaps.append({'case_id': self.current_case, 'reason': 'request_or_input_budget_exhausted'})
            return 0, None, 0, {}
        self.requests += 1
        self.input_chars += len(encoded)
        self.requested_output_tokens += kwargs['max_tokens']
        headers = ({'x-api-key': self.api_key, 'anthropic-version': '2023-06-01'}
                   if self.profile.transport == 'messages' else {'Authorization': f'Bearer {self.api_key}'})
        headers['Content-Type'] = 'application/json'
        started = time.perf_counter()
        status, raw, reason = 0, None, None
        try:
            response = await self.client.post(f'{self.base_url}/{path}', content=encoded.encode(), headers=headers, timeout=60)
            status = response.status_code
            try: raw = response.json()
            except ValueError: reason = 'invalid_json'
        except httpx.HTTPError as exc:
            reason = type(exc).__name__
        elapsed = time.perf_counter() - started
        normalized = None
        if status == 200 and reason is None:
            try: normalized, reason = decode_response(self.profile.transport, raw)
            except (TypeError, ValueError, AttributeError, KeyError): reason = 'invalid_envelope'
        if reason:
            self.gaps.append({'case_id': self.current_case, 'reason': reason})
        evidence = {'case_id': self.current_case, 'request_index': self.requests, 'request': payload,
                    'request_headers': headers, 'http_status': status, 'response_body': raw,
                    'duration_seconds': elapsed, 'reason': reason, 'success': status == 200 and reason is None}
        self.audit.write(evidence)
        if model == self.profile.model:
            self.responses.append({'case_id': self.current_case, 'status': status,
                                   'model': (raw or {}).get('model') if isinstance(raw, dict) else None,
                                   'valid_response': normalized is not None,
                                   'has_answer': bool(answer(normalized)) if normalized else False})
        # Non-2xx bodies must remain available to the explicit route-negative check.
        return (0 if reason else status), normalized if status == 200 else raw, elapsed, {}

    async def case_quality_reasoning(self, model):
        rng = self.probe_rng('economy_reasoning')
        a, b, c = rng.randint(101, 999), rng.randint(21, 89), rng.randint(2, 19)
        units, ca, cb = rng.randint(16, 29), rng.randint(8, 12), rng.randint(13, 18)
        minimum = min(x*ca+y*cb for x in range(units+1) for y in range(units+1) if 3*x+5*y >= units)
        expected = {'arithmetic': a*b-c, 'minimum_cost': minimum, 'logic': 2**12-12**3}
        prompt = (f'只输出JSON，精确三个整数字段：arithmetic=({a}*{b})-{c}；'
                  f'minimum_cost：A包3件{ca}元、B包5件{cb}元，至少买{units}件，最少费用；'
                  'logic=2^12-12^3（^表示乘方）。不要解释，不要额外字段。')
        st, body, _, _ = await self.probe_request(model, [{'role': 'user', 'content': prompt}])
        return outcome([row('三题一次请求', st, strict_json(answer(body), {'const': expected}),
                            expected=expected, answer=answer(body)[:1000])], **self.probe_meta(samples=3))

    async def case_quality_retention(self, model):
        rng = self.probe_rng('economy_retention')
        orders = [{'id': f'O{i}', 'qty': rng.randint(1, 7), 'price': rng.randint(9, 39),
                   'cancelled': i in (1, 4)} for i in range(7)]
        eligible = [o for o in orders if not o['cancelled']]
        ranked = sorted(eligible, key=lambda o: (-o['qty']*o['price'], o['id']))
        expected = {'ids': [o['id'] for o in ranked[:3]],
                    'total': sum(o['qty']*o['price'] for o in eligible),
                    'count': len(eligible), 'checksum': sum(o['qty'] for o in ranked[:3])}
        # Independent exact-answer task; no reasoning-length/response-size proxy.
        prompt = ('高级约束题：排除cancelled=true的订单，按qty*price降序，同额按id升序。'
                  '只输出JSON：ids为前3个id，total为所有未取消订单总金额，count为未取消订单数，'
                  'checksum为前3个订单qty之和。禁止额外字段。数据：' + json.dumps(orders))
        st, body, _, _ = await self.probe_request(model, [{'role': 'user', 'content': prompt}])
        return outcome([row('过滤、排序、计算与约束联合校验', st, strict_json(answer(body), {'const': expected}),
                            expected=expected, answer=answer(body)[:1000])],
                       **self.probe_meta(reference_comparison=False, vendor_full_strength_proven=False))

    async def case_economy_route(self, model):
        controls = [r for r in self.responses if r['valid_response'] and r['has_answer'] and self.matches_model(r['model'])]
        if not controls: return outcome([], reason='没有型号一致且有正文的正对照，未发送负例')
        ghost = f'{model}-nonexistent-{uuid.uuid4().hex}'
        st, body, _, _ = await self.probe_request(ghost, [{'role': 'user', 'content': '只回复 OK'}], max_tokens=64)
        error = body.get('error') if isinstance(body, dict) else None
        text = json.dumps(error, ensure_ascii=False).lower() if error else ''
        recognized = (st in (400, 404, 422) and any(s in text for s in ('model', '模型'))
                      and any(s in text for s in ('not found', 'not exist', 'unknown model', 'invalid model', 'model_not_found', '不存在', '未找到'))
                      and not any(s in text for s in ('quota', 'rate limit', 'permission', 'authentication', 'access', '限流', '额度', '余额', '权限', '鉴权')))
        result = {'name': '不存在型号负对照', 'status': 'passed' if recognized else 'failed' if 200 <= st < 300 else 'blocked',
                  'passed': recognized, 'http_status': st, 'error': error, 'control_case_id': controls[0]['case_id']}
        return outcome([result], **self.probe_meta())

    async def case_economy_identity(self, model):
        observed = [r for r in self.responses if r['valid_response']]
        missing = any(not r['model'] for r in observed)
        conflict = any(r['model'] and not self.matches_model(r['model']) for r in observed)
        enough = len({r['case_id'] for r in observed}) >= 3
        state = 'failed' if conflict else 'passed' if enough and not missing else 'blocked'
        return state == 'passed', '复用至少三个独立任务的响应型号；不额外请求', {
            **self.probe_meta(), 'responses': observed, 'cryptographic_proof': False,
        }, state

    async def run(self):
        results = []
        try:
            for suffix, name, group, dimension, method in CHECKS:
                self.current_case = f'ECO-{self.profile.family.upper()}-{suffix}'
                started = time.perf_counter()
                gap_start = len(self.gaps)
                try:
                    passed, detail, observations, state = await getattr(self, f'case_{method}')(self.profile.model)
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    passed, detail, observations, state = False, f'无有效证据：{type(exc).__name__}', {}, 'blocked'
                results.append({'case_id': self.current_case, 'name': name, 'model': self.profile.model,
                                'passed': passed, 'status': state, 'details': detail,
                                'duration_seconds': time.perf_counter()-started,
                                'observations': {**observations, 'evidence_gaps': self.gaps[gap_start:]},
                                'scoring_group': group, 'capability': dimension,
                                'workload_version': VERSION})
        finally:
            self.audit.close()
        passed = all(r['status'] == 'passed' for r in results) and len(results) == len(CHECKS)
        summary = {'test_kind': self.profile.family, 'case': f'{self.profile.family.upper()}_ECONOMY',
                   'model': self.profile.model, 'base_url': self.base_url, 'results': results,
                   'total': len(results), 'passed': sum(r['passed'] for r in results),
                   'failed': sum(r['status'] == 'failed' for r in results),
                   'blocked': sum(r['status'] == 'blocked' for r in results), 'admission_passed': passed,
                   'economy_validation': {'version': VERSION, 'profile': asdict(self.profile),
                       'seed': self.probe_seed(), 'requests_sent': self.requests,
                       'serialized_input_chars': self.input_chars, 'requested_output_tokens': self.requested_output_tokens,
                       'vendor_full_strength_proven': False, 'scope': '经济文本/Agent题集；不含视觉、完整通道、厂商验真或极限负载',
                       'excluded': ['KVV_TOOL_SCHEMA', 'load', 'signature_extraction', 'image_generation']}}
        (self.directory / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
        return summary
