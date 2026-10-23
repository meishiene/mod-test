"""Versioned workload profiles and conservative evidence assessment.

Probe sizes are test budgets, NOT vendor context limits. Profiles are repository
expectations pending provider calibration. No API response proves backend weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from collections import Counter
import hashlib
import json
from pathlib import Path

from .probe_assertions import normalized_model

VALIDATION_VERSION = 'qwen-glm-validation-1'
DEFAULT_SEED = 20261023


@dataclass(frozen=True)
class ValidationProfile:
    model_id: str
    family: str
    tier: str
    context_probe_chars: int
    challenge_samples: int
    image_input: bool | None = None
    source: str = 'repository test policy; not a verified vendor limit'

    @property
    def prefix(self) -> str:
        return 'QWEN' if self.family == 'qwen' else 'GLM-53'

    @property
    def request_options(self) -> dict:
        return {'enable_thinking': False} if self.family == 'qwen' else {'thinking': {'type': 'enabled'}}


PROFILES = {
    p.model_id: p for p in (
        ValidationProfile('glm-5.3', 'glm', 'flagship', 65536, 5),
        ValidationProfile('glm-5.3-flash', 'glm', 'flash', 32768, 3),
        # FlashX's proprietary limits are unverified; use the SAME common workload
        # as Flash and require exact identity. Do not invent a speed/quality ordering.
        ValidationProfile('glm-5.3-flashx', 'glm', 'flashx', 32768, 3),
        ValidationProfile('qwen3.8-max', 'qwen', 'max', 131072, 5, True),
        ValidationProfile('qwen3.7-max', 'qwen', 'max', 65536, 5, False),
        ValidationProfile('qwen3.8-flash', 'qwen', 'flash', 32768, 3, True),
        ValidationProfile('qwen3.7-plus', 'qwen', 'plus', 32768, 3, True),
        ValidationProfile('qwen3.6-plus', 'qwen', 'plus', 32768, 3, True),
        ValidationProfile('qwen3.6-flash', 'qwen', 'flash', 32768, 3, True),
    )
}


def validation_profile(model: str) -> ValidationProfile | None:
    return PROFILES.get(normalized_model(model))


# Shared cases are individually auditable and independently scored.
PROBE_DEFINITIONS = (
    ('Q01', '数学与逻辑客观题', 'PQUAL', 'quality_reasoning', 'thinking'),
    ('Q02', '代码生成与隔离解释执行', 'PQUAL', 'quality_code', 'quality'),
    ('Q03', '中文业务抽取与严格 Schema', 'PQUAL', 'quality_schema', 'quality'),
    ('Q04', '长上下文头中尾多针召回', 'PQUAL', 'quality_context', 'context'),
    ('Q05', '工具错误恢复与结果利用', 'PQUAL', 'quality_agent', 'tools'),
    ('Q06', '双工具选择与参数正确性', 'PQUAL', 'quality_parallel', 'tools'),
    ('R01', '随机挑战与逐响应型号一致性', 'PVERIFY', 'verify_identity', 'trust'),
    ('R02', '不存在型号拒绝与静默回退检测', 'PVERIFY', 'verify_route', 'trust'),
    ('R03', '系统指令优先级与干扰隔离', 'PVERIFY', 'verify_instruction', 'quality'),
)


def required_case_ids(model: str) -> tuple[str, ...]:
    p = validation_profile(model)
    if not p:
        return ()
    ids = [f'{p.prefix}-{suffix}' for suffix, *_ in PROBE_DEFINITIONS]
    if p.family == 'qwen':
        ids += [f'QWEN-{i:02d}' for i in range(1,7)]
        ids += ['QWEN-A01','QWEN-A03','QWEN-A04','QWEN-A05','QWEN-F01','QWEN-T01','QWEN-T02','QWEN-T04']
    else:
        ids += [f'GLM-53-{i:02d}' for i in (1,2,3,4,6,7,8,9,11,12)]
        ids += [f'GLM-53-F{i:02d}' for i in (1,2,4,5,7)]
    if p.image_input is True:
        ids.append('QWEN-V01')
    return tuple(ids)


def load_reference(path: Path | None) -> dict | None:
    if path is None:
        return None
    raw = path.read_bytes()
    summary = json.loads(raw)
    if not isinstance(summary, dict):
        raise ValueError('参考报告必须为 summary.json 对象')
    return {'summary': summary, 'sha256': hashlib.sha256(raw).hexdigest(), 'path': str(path.resolve())}


def assess_validation(results, model: str, reference: dict | None = None, *, seed: int = DEFAULT_SEED) -> dict:
    profile = validation_profile(model)
    rows = [asdict(r) if is_dataclass(r) else r for r in results]
    required = required_case_ids(model)
    by_id = {r.get('case_id'): r for r in rows if isinstance(r, dict)}
    duplicates = {cid for cid,count in Counter(r.get('case_id') for r in rows if isinstance(r,dict)).items() if count>1}
    passed = [cid for cid in required if cid not in duplicates and by_id.get(cid, {}).get('status') == 'passed'
              and by_id[cid].get('passed') is True
              and normalized_model(by_id[cid].get('model', '')) == normalized_model(model)]
    failed = [cid for cid in required if by_id.get(cid, {}).get('status') in {'failed', 'error'}]
    missing = [cid for cid in required if cid not in passed and cid not in failed]
    complete = bool(required) and len(passed) == len(required)
    identity_ids = [f'{profile.prefix}-{s}' for s in ('R01', 'R02')] if profile else []
    identity_state = ('behavior_conflict' if any(i in failed for i in identity_ids) else
                      'consistent_observations' if identity_ids and all(i in passed for i in identity_ids) else
                      'insufficient_evidence')
    state = 'capability_gap' if failed else 'profile_passed' if complete else 'insufficient_evidence'
    comparison = {'state': 'not_supplied', 'note': '未提供同型号、同版本、同种子的可信参考报告'}
    if reference is not None:
        other = reference.get('summary', {})
        old = other.get('model_validation') or {}
        eligible = (bool(profile) and other.get('admission_passed') is True
                    and old.get('admission_passed') is True
                    and other.get('case') == ('QWEN_FULL' if profile.family == 'qwen' else 'GLM_53_FULL')
                    and normalized_model(other.get('model', '')) == profile.model_id
                    and old.get('version') == VALIDATION_VERSION and old.get('seed') == seed
                    and old.get('profile') == asdict(profile))
        old_rows = {r.get('case_id'): r for r in other.get('results', [])}
        eligible = eligible and len(old_rows)==len(other.get('results', [])) and all(old_rows.get(cid, {}).get('status') == 'passed'
                                    and old_rows[cid].get('passed') is True
                                    and normalized_model(old_rows[cid].get('model', '')) == profile.model_id for cid in required)
        comparison = {'state': 'consistent' if eligible and complete else 'regression' if eligible and failed else
                      'incomplete' if eligible else 'incompatible',
                      'sha256': reference.get('sha256'), 'path': reference.get('path'),
                      'endpoint': other.get('endpoint'),
                      'note': '参考来源可信度由操作者确认；仅比较本测试集，不能证明底层权重身份'}
        if eligible and complete:
            state = 'reference_consistent'
        elif not eligible:
            state = 'reference_incompatible'
    return {'version': VALIDATION_VERSION, 'seed': seed,
            'profile': asdict(profile) if profile else None,
            'admission_passed': complete and (reference is None or comparison['state'] == 'consistent'),
            'coverage': {'required': len(required), 'passed': len(passed), 'failed': failed, 'missing_or_blocked': missing, 'duplicate_case_ids':sorted(duplicates & set(required))},
            'authenticity': {'state': identity_state, 'proven': False,
                            'note': '随机挑战、精确型号及路由反例是行为证据；无法从可改写的接口证明真实权重或厂商来源'},
            'full_strength': {'state': state, 'vendor_full_strength_proven': False,
                              'note': '通过仅代表已满足该型号的版本化测试标准，非厂商全部能力/上下文极限证明'},
            'reference': comparison}


def apply_validation_gate(summary: dict, comprehensive: bool) -> None:
    """Keep existing relay/face adapters from overwriting comprehensive evidence gate."""
    assessment = summary['model_validation']
    if not comprehensive:
        return
    verdict = summary.get('relay_verdict') or summary.get('face_verdict')
    allowed = assessment['admission_passed'] and (bool(verdict.get('admission_passed')) if verdict else True)
    summary['admission_passed'] = allowed
    if verdict is not None:
        verdict['admission_passed'] = allowed
        verdict['summary_line'] = ('综合校验通过（限已测能力，身份非密码学证明）' if allowed else
                                   '综合校验未通过：存在失败或必测证据不足')


def validation_report_stats(summary: dict) -> list[str]:
    value = summary.get('model_validation') or {}
    if not value:
        return []
    coverage = value['coverage']
    labels = {
        'profile_passed':'已达到本型号测试标准', 'reference_consistent':'与指定参考样本一致',
        'reference_incompatible':'参考样本不匹配', 'capability_gap':'存在已测能力缺口',
        'insufficient_evidence':'证据不足', 'behavior_conflict':'发现行为矛盾',
        'consistent_observations':'已测行为一致', 'not_supplied':'未提供参考样本',
        'consistent':'已测项目一致', 'regression':'相对参考出现回归',
        'incomplete':'当前证据不足', 'incompatible':'型号/版本/种子/完整性不匹配',
    }
    label = lambda state: labels.get(state, state)
    return [f"综合能力证据：{coverage['passed']}/{coverage['required']}；满血评估：{label(value['full_strength']['state'])}",
            f"真实性：{label(value['authenticity']['state'])}（行为取证，不证明底层权重身份）",
            f"参考对照：{label(value['reference']['state'])}；测试版本：{value['version']}；种子：{value['seed']}",
            '缺失或阻断：' + ('、'.join(coverage['missing_or_blocked']) or '无')]


def create_run_directory(root: Path, case: str) -> Path:
    """Atomic unique artifact directory for concurrent model batches."""
    import time
    import uuid
    root.mkdir(parents=True, exist_ok=True)
    while True:
        path = root / f'{case}-{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:8]}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
