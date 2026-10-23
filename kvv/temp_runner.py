#!/usr/bin/env python3
"""Kimi 测试模块下 K3 临时案例集的执行与报告。

本执行器只负责装配：判定全部由 ``common.validators`` 给出，每个请求都写成
可审计记录，自身不含 pass/fail 业务逻辑。

入口是 ``scripts/run_kvv_temp.py``（Web 端派发用）或 ``python -m kvv.temp_runner``；
K3 走 Chat Completions 面，故与官方 KVV pytest 套件（``kvv/runner.py``）分开两条
执行链，但同属一个「Kimi 测试」模块。
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time, uuid
from dataclasses import asdict
from pathlib import Path

# 允许 `python -m kvv.temp_runner` 与脚本入口执行：需要仓库根在 sys.path 上
# 才能 import common/volcperf（Web 端另外注入 PYTHONPATH，二者不冲突）。
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from common.suite import ManifestCase, execute_manifest_case
from common.report_style import stats_lines
from common.test_framework import HttpExecutor, ResultLog, check, validate_case
from common.validators import status, status_2xx, status_error, content, usage, sse, json_content, tool_calls
from volcperf.protocols import api_root
from .temp_suite import MANIFEST_PATH as MANIFEST, TempCase, find_case, load_manifest, manifest_cases, merge_cases

def _thinking_not_closable():
    """K3 思考关不掉（官方契约）：显式传 `thinking={"type":"disabled"}` 时，

    原厂要么直接 400 拒绝该形态，要么忽略该参数、思考照旧产生——两者都算达标。
    只有「HTTP 200 且完全没有产生思考」才判不达标（等价于真的把思考关掉了）；
    非 400/200 属取证前提未成立，记 not_evaluable，不判模型失败。
    """
    def validate(record):
        if record.http_status == 400:
            return check('thinking_disabled', True, '400 拒绝 或 200 仍思考', record.http_status, '')
        if record.http_status != 200:
            return check('thinking_disabled', None, '400 拒绝 或 200 仍思考', record.http_status,
                         '响应既非 400 也非 200，无法判定该观察项')
        body = record.response_body if isinstance(record.response_body, dict) else {}
        choices = body.get('choices') or []
        message = (choices[0].get('message') or {}) if choices else {}
        reasoning = record.reasoning or message.get('reasoning_content') or message.get('reasoning') or ''
        ok = bool(reasoning)
        return [check('thinking_disabled', ok, '400 拒绝 或 200 仍思考',
                      {'http_status': record.http_status, 'reasoning_chars': len(str(reasoning))},
                      '思考被真的关掉了（200 且无 reasoning），与 K3「思考关不掉」的原厂行为不符'),
                usage('prompt_tokens', 'completion_tokens')(record)]
    return validate


def validators_for(case: ManifestCase):
    """Translate the legacy manifest's human assertions into explicit checks.

    New manifests should use ``expected`` directly.  This compatibility layer
    prevents old cases from silently becoming HTTP-200-only tests.
    """
    cid=case.id; out=[]
    if cid in {'K3-ERR-001','K3-ERR-002','K3-JSON-002'}:
        out.append(status(400))
        out.append(status_error(400, error_types=('invalid_request_error','invalid_request'),
                                message_contains=('temperature',) if cid=='K3-ERR-001' else ()))
    elif cid=='K3-PARAM-002':
        # K3 官方契约把 temperature/top_p 锁为固定值单点（1 / 0.95），非固定值必须被拒。
        out.append(status(400))
        out.append(status_error(400, error_types=('invalid_request_error','invalid_request'),
                                message_contains=('temperature','top_p')))
    elif cid=='K3-AUTH-001':
        out += [status(401), status_error(401, message_contains=('auth','key','unauthorized'))]
    elif cid=='K3-AUTH-002':
        out += [status(429), status_error(429, message_contains=('rate','limit','quota','too many'))]
    elif cid=='K3-THINK-002':
        # K3 思考关不掉：显式传 disabled 要么被拒（400），要么被忽略但思考照旧产生。
        out.append(_thinking_not_closable())
    else:
        out.append(status_2xx())
        if case.request.get('stream'): out.append(sse())
        if cid=='K3-CONN-001' or 'usage' in ' '.join(case.assertions).lower(): out.append(usage('prompt_tokens','completion_tokens'))
        if cid=='K3-JSON-001': out.append(json_content())
        if cid=='K3-PARAM-001': out.append(content(exact='PARAM-OK'))
        if cid=='K3-TOOL-001': out.append(tool_calls(count=1,names=('lookup_order',),args={'order_id':'ORD-000731'}))
    return out

def aggregate(plan: TempCase, group_results: list[dict]) -> dict:
    """把一个合并案例下的多次请求汇成一行结论。

    单条案例直接复用原判定；合并案例要求全部通过才算通过，任一失败即失败，
    既无失败也无通过（例如全部 not_evaluable）时保持 not_evaluable。
    """
    if not plan.merged and len(group_results)==1:
        return {**group_results[0], 'case_id': plan.id, 'category': plan.category, 'case_ids': list(plan.case_ids), 'repetitions': 1, 'merged': plan.merged}
    statuses={item['status'] for item in group_results}
    status_value='passed' if statuses=={'passed'} else 'failed' if 'failed' in statuses else 'not_evaluable'
    reasons=[(item['case_id'], '；'.join(item['failure_reasons'])) for item in group_results if item.get('failure_reasons')]
    # 合并案例下多条请求往往是同一个原因（例如整批都缺 messages），逐个铺开会
    # 把报告撑成一行巨文；同因时折叠成一条并标明次数。
    distinct={text for _, text in reasons}
    if len(reasons)>1 and len(distinct)==1:
        detail=[f"{len(reasons)} 次请求同一原因：{distinct.pop()}"]
    else:
        detail=[f"{case_id}: {text}" for case_id, text in reasons]
    return {'case_id':plan.id, 'category':plan.category, 'case_ids':list(plan.case_ids), 'repetitions':len(group_results),
            'merged':plan.merged, 'status':status_value, 'passed':status_value=='passed', 'checks':[],
            'failure_reasons':detail}


CACHE_HIT_RATIO = 0.80       # 单次探测：cached_tokens / prompt_tokens 达到该值记为一次命中
CACHE_HIT_PROBE_RATE = 0.80  # 该档聚合：命中 probe 占比达到该值才算达标（清单 admission_checklist A04）


def _cache_ratio(record):
    usage_value = record.usage or {}
    details = usage_value.get('prompt_tokens_details') or usage_value.get('input_tokens_details') or {}
    prompt = usage_value.get('prompt_tokens', usage_value.get('input_tokens'))
    cached = details.get('cached_tokens', usage_value.get('cached_tokens'))
    try:
        return (float(cached) / float(prompt)) if prompt and cached is not None else None
    except (TypeError, ValueError):
        return None


def _cache_sample_check(record):
    """逐条 probe 只记录样本，不做判定。

    准入标准的口径是「90% 前缀重复的命中率 ≥80%」（清单 A04），即**命中 probe 占比**；
    逐条严判会把「新前缀组前 1~2 次基本不命中」这类上游缓存的正常表现直接判成用例失败。
    运输层与取证的硬约束仍逐条生效（见 probe_validators 里的 status/usage）。
    """
    return check('cache_ratio_sample', True, '记录命中比例（按档聚合判定）', _cache_ratio(record), '')


def aggregate_cache(plan: TempCase, group_results: list[dict]) -> dict:
    """缓存档聚合：先按常规口径确认运输与取证，再按「命中 probe 占比 ≥80%」判定。"""
    row = aggregate(plan, group_results)
    ratios = [item.get('cache_ratio') for item in group_results]
    probes = len(ratios)
    hits = sum(1 for ratio in ratios if ratio is not None and ratio >= CACHE_HIT_RATIO)
    hit_rate = (hits / probes) if probes else None
    row.update({'cache_hits': hits, 'cache_probes': probes, 'cache_hit_rate': hit_rate,
                'cache_ratios': [None if r is None else round(r, 4) for r in ratios]})
    if row['status'] != 'passed':
        return row
    ok = hit_rate is not None and hit_rate >= CACHE_HIT_PROBE_RATE
    row['status'] = 'passed' if ok else 'failed'
    row['passed'] = ok
    if not ok:
        row['failure_reasons'] = [
            f"缓存命中率 {hits}/{probes} = {0.0 if hit_rate is None else hit_rate:.1%} < {CACHE_HIT_PROBE_RATE:.0%}"
            "（单次 cached_tokens / prompt_tokens ≥80% 记为一次命中；缺字段按未命中计）"
        ]
    return row


async def execute_cache_probe_case(executor: HttpExecutor, log: ResultLog, plan: TempCase, *, model: str, url: str, base_headers: dict[str, str], wait_seconds: int = 0) -> list[dict]:
    """Execute a cache case as one warm request followed by same-prefix probes."""
    target_tokens = 70000
    prefix = ('K3 缓存稳定前缀。请保留并识别这段固定上下文。' * (target_tokens * 4 // 24 + 1))[:target_tokens * 4]
    warm_validators = [status_2xx(), usage('prompt_tokens')]
    probe_validators = [status_2xx(), usage('prompt_tokens'), _cache_sample_check]
    results = []
    warm_id = f'K3-{plan.id}-WARM'
    requests = [(warm_id, prefix)] + [
        (case_id, prefix + f'\n本次探测尾部：{index}')
        for index, case_id in enumerate(plan.case_ids, 1)
    ]
    for case_id, content_text in requests:
        if case_id != warm_id and wait_seconds:
            await asyncio.sleep(wait_seconds)
        payload = {
            'model': model, 'stream': True, 'stream_options': {'include_usage': True},
            'max_tokens': 16, 'messages': [{'role': 'user', 'content': content_text}],
        }
        headers = dict(base_headers)
        headers['X-Request-ID'] = f'{case_id}-{uuid.uuid4().hex[:10]}'
        record = await executor.execute(case_id=case_id, url=url, payload=payload, headers=headers, stream=True)
        verdict = validate_case(record, probe_validators if case_id != warm_id else warm_validators)
        record.validation = verdict
        await log.write(record)
        if case_id != warm_id:
            results.append({'case_id': case_id, 'status': verdict.status, 'passed': verdict.passed,
                            'checks': [asdict(item) for item in verdict.checks],
                            'failure_reasons': verdict.reasons, 'request_id': record.request_id,
                            'cache_ratio': _cache_ratio(record)})
    return results

async def main_async(args):
    atomic_cases=manifest_cases(load_manifest(MANIFEST))
    merged_cases=merge_cases(atomic_cases)
    selected=os.getenv('VOLC_TEMP_CASE','').strip() or None
    selected_case=find_case(merged_cases,selected)
    if selected and selected_case is None:
        raise SystemExit(f'未知临时测试案例: {selected}')
    # 案例集按同类项合并后执行：一个合并案例下的原子请求逐条发起（证据不丢），
    # 结果按合并案例汇总成一行。
    plans=[selected_case] if selected_case else merged_cases
    raw_by_id={str(item['id']):item for item in atomic_cases}
    output_root=Path(args.output_root).expanduser().resolve() if args.output_root else ROOT/'results'
    out=output_root/f"K3_TEMP_SUITE-{time.strftime('%Y%m%d-%H%M%S')}"; out.mkdir(parents=True)
    log=ResultLog(out/'execution.jsonl'); results=[]
    protocol=os.getenv('VOLC_PROTOCOL','volcengine')
    base_url=api_root(protocol,os.environ['VOLC_BASE_URL'])+'/chat/completions'
    base_headers={'Authorization':f"Bearer {os.environ['VOLC_API_KEY']}",'Content-Type':'application/json','Accept':'text/event-stream'}
    async with HttpExecutor(timeout=300,max_connections=10) as executor:
        for plan in plans:
            if plan.id in {'CACHE-HIT-7', 'CACHE-TTL-3'}:
                wait_seconds = 30 if plan.id == 'CACHE-TTL-3' else 0
                group_results = await execute_cache_probe_case(executor, log, plan, model=os.environ['VOLC_MODEL'], url=base_url, base_headers=base_headers, wait_seconds=wait_seconds)
                results.append(aggregate_cache(plan, group_results)); print(results[-1]['case_id'],results[-1]['status'],flush=True); continue
            group_results=[]
            for case_id in plan.case_ids:
                case=ManifestCase(raw_by_id[case_id])
                case.validators=lambda v=validators_for(case): v
                group_results.append(await execute_manifest_case(executor,log,case,model=os.environ['VOLC_MODEL'],url=base_url,base_headers=base_headers))
            results.append(aggregate(plan,group_results)); print(results[-1]['case_id'],results[-1]['status'],flush=True)
    log.close()
    passed=sum(x['passed'] for x in results); evaluated=sum(x['status'] in {'passed','failed'} for x in results)
    atomic_requests=sum(x['repetitions'] for x in results)
    # 场景描述类案例没有可执行请求，plan 次数不等于实际发起次数，分开记。
    executed_requests=sum(1 for _ in (out/'execution.jsonl').open(encoding='utf-8'))
    summary={'schema_version':'volcperf.result.v2','record_id':os.getenv('VOLC_RECORD_ID'),'case':selected_case.id if selected_case else 'K3_TEMP_SUITE','test_kind':'kvv','model':os.environ['VOLC_MODEL'],'base_url':os.environ['VOLC_BASE_URL'],'total':len(results),'merged_cases':len(results),'atomic_requests':atomic_requests,'executed_requests':executed_requests,'manifest_cases':len(atomic_cases),'passed':passed,'failed':sum(x['status']=='failed' for x in results),'not_evaluable':sum(x['status']=='not_evaluable' for x in results),'validator_errors':sum(x['status']=='error' for x in results),'pass_rate':passed/evaluated if evaluated else 0,'admission_passed':bool(results) and all(x['status']=='passed' for x in results),'results':results,'evidence':'execution.jsonl','note':'Transport status and case verdict are separate; not_evaluable never counts as pass. Same category + same request.batch rows are merged into one case; every atomic request is still executed and logged.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# Kimi 测试 · K3 临时案例集报告（v2 校验流水线）','',
           *stats_lines(len(results),passed,summary['failed'],0,summary['not_evaluable']+summary['validator_errors'],summary['pass_rate'],
                        [f"结论：{'通过' if summary['admission_passed'] else '未通过/数据不足'}",
                         f"用例：{len(results)} 个合并案例（清单原始 {len(atomic_cases)} 条，同类项已合并）",
                         f"请求：计划 {atomic_requests} 次，实际发起 {executed_requests} 次（仅场景描述的案例不发请求）"]),
           '', '| 用例 | 类别 | 次数 | 状态 | 失败/不可评估原因 |','|---|---|---|---|---|']
    for x in results:
        note='；'.join(x['failure_reasons']) or '—'
        if x.get('cache_probes'):
            rate=x.get('cache_hit_rate')
            hit=f"命中 {x.get('cache_hits')}/{x['cache_probes']} = {'—' if rate is None else format(rate,'.1%')}"
            note=f"{hit}；{note}" if x['failure_reasons'] else hit
        lines.append(f"| `{x['case_id']}` | {x.get('category','') or '—'} | {x['repetitions']} | {x['status']} | {note} |")
    lines += ['', '> 同类项合并规则：`category` 与 `request.batch`（含档位与次数）相同视为同一逻辑案例；合并只影响案例集与汇总粒度，合并案例下的每个原子请求仍单独发起、单独留证。', '', '## 证据', '', '- `execution.jsonl`：请求、响应、原始 SSE、时间点和校验明细（按原子案例逐条）', '- `summary.json`：结构化汇总和准入判定']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(f'结果目录: {out.resolve()}'); return 0 if summary['admission_passed'] else 2

def main():
    p=argparse.ArgumentParser(description='运行 K3 manifest 测试')
    p.add_argument('--output-root', help='结果输出根目录，默认使用项目 results 目录')
    try: raise SystemExit(asyncio.run(main_async(p.parse_args())))
    except KeyError as e: raise SystemExit(f'缺少环境变量: {e}')
if __name__=='__main__': main()
