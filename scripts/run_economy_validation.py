#!/usr/bin/env python3
"""Run a bounded economy card, using the same connection environment as Web."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from common.economy_profiles import ECONOMY_CASES, economy_profile, economy_case_payload
from common.economy_validation import EconomyRunner
from common.faces import selected_faces
from common.model_validation import create_run_directory
from common.report_style import case_table_report
from common.suite_runtime import finalize
from volcperf.protocols import api_root, supports_test_kind


def write_report(directory, summary):
    info=summary['economy_validation']
    report=case_table_report(economy_case_payload(summary['case'])['title'],
        total=summary['total'],passed=summary['passed'],failed=summary['failed'],blocked=summary['blocked'],
        extra_stats=[f"经济题集版本：{info['version']}；种子：{info['seed']}",
                     f"请求：{info['requests_sent']} / 11；累计输入字符：{info['serialized_input_chars']} / 64000",
                     f"请求的输出预算上限合计：{info['requested_output_tokens']} tokens（不是实际花费）",
                     info['scope'], '本题集不出旧制120分，不宣称原厂身份或全部能力满血。'],
        rows=[(r['case_id']+' '+r['name'],r['model'],r['status'],f"{r['duration_seconds']:.2f}",
               r['details']+'；'+','.join(g['reason'] for g in r['observations'].get('evidence_gaps',[])))
              for r in summary['results']],status_column=2)
    (directory/'report.md').write_text(report,encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=ECONOMY_CASES)
    parser.add_argument('--output-root',type=Path,default=Path('results'))
    parser.add_argument('--seed',type=int,default=20261023)
    args=parser.parse_args(argv)
    for key in ('VOLC_BASE_URL','VOLC_API_KEY','VOLC_MODEL'):
        if not os.environ.get(key): parser.error(f'缺少 {key}')
    profile=economy_profile(ECONOMY_CASES[args.case],os.environ['VOLC_MODEL'])
    if not {'model','verify'} <= set(selected_faces()):
        parser.error('经济验证是固定模型能力+验真题集，请同时选择 model,verify')
    protocol=os.environ.get('VOLC_PROTOCOL','anthropic' if profile.family=='fable' else 'openai')
    try:
        supported = supports_test_kind(protocol, profile.family, economy_case_payload(args.case)['protocols'])
    except ValueError:
        supported = False
    if not supported:
        parser.error('经济验证不支持该协议；Claude使用anthropic，其余家族使用OpenAI兼容文字协议')
    base=api_root(protocol,os.environ['VOLC_BASE_URL'])
    directory=create_run_directory(args.output_root,args.case)
    async def execute():
        async with httpx.AsyncClient(trust_env=False,follow_redirects=False) as client:
            runner=EconomyRunner(profile,base,os.environ['VOLC_API_KEY'],directory,client=client,seed=args.seed)
            return await runner.run()
    summary=asyncio.run(execute())
    summary.update(record_id=os.environ.get('VOLC_RECORD_ID',directory.name),protocol=protocol,
                   executed_faces=['model','verify'])
    write_report(directory,summary)
    finalized=finalize(directory,summary)
    print(json.dumps({'result_dir':str(directory),'passed':finalized['admission_passed']},ensure_ascii=False))
    return 0 if finalized['admission_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
