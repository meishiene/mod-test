from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from common.faces import annotate_summary, selected_faces
from common.model_validation import (DEFAULT_SEED, apply_validation_gate, assess_validation,
                                     create_run_directory, load_reference, validation_profile)
from common.suite_runtime import finalize
from volcperf.protocols import api_root
from .cases import CASES
from .core import CASE_SPECS, GLM53Runner, GLMConfig, summarize_results, write_report
from .faces import CASE_FACES


def main(argv=None):
    parser = argparse.ArgumentParser(description='运行 GLM 5.3 / Flash / FlashX 专项测试')
    parser.add_argument('case', choices=sorted(CASES))
    parser.add_argument('--output-root', type=Path, default=Path('results'))
    parser.add_argument('--faces', default=None, help='model,verify（默认）/ channel / all')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='可复现客观题种子')
    parser.add_argument('--reference', type=Path, help='可信的同型号综合 summary.json，仅用于 FULL')
    args = parser.parse_args(argv)
    if args.reference and args.case != 'GLM_53_FULL':
        raise ValueError('--reference 仅支持 GLM_53_FULL 综合校验')
    reference = load_reference(args.reference)
    key = os.getenv('GLM_API_KEY') or os.getenv('VOLC_API_KEY')
    base = os.getenv('GLM_BASE_URL') or os.getenv('VOLC_BASE_URL')
    model = os.getenv('GLM_MODEL') or os.getenv('VOLC_MODEL')
    if not key or not base or not model:
        raise ValueError('缺少 GLM_API_KEY/GLM_BASE_URL/GLM_MODEL')
    if args.case == 'GLM_53_FULL' and (not validation_profile(model) or validation_profile(model).family != 'glm'):
        raise ValueError('综合校验需要已登记的型号；请使用 GLM 5.3 / Flash / FlashX 的精确型号')
    base = api_root(os.getenv('VOLC_PROTOCOL', 'volcengine'), base)
    out = create_run_directory(args.output_root, args.case)
    faces = selected_faces(args.faces)
    case = CASES[args.case]

    async def go():
        config = GLMConfig(key, base, seed=args.seed,
                           verify_ssl=os.getenv('GLM_VERIFY_SSL', 'true').lower() not in {'0', 'false'})
        async with GLM53Runner(config, audit_dir=out) as runner:
            results = await runner.run(model, case.priorities, faces)
        summary = summarize_results(results)
        summary.update(record_id=os.getenv('VOLC_RECORD_ID'), test_kind='glm',
                       case=args.case, model=model, endpoint=base)
        annotate_summary(summary, results, CASE_FACES, faces)
        verdict = summary['face_verdict']
        summary['admission_passed'] = verdict['admission_passed']
        summary['model_validation'] = assess_validation(results, model, reference, seed=args.seed)
        if args.case in {'GLM_53_CAPABILITY', 'GLM_53_VERIFY'}:
            expected = {s.case_id for s in CASE_SPECS if s.priority in case.priorities}
            passed = expected == {r.case_id for r in results if r.status == 'passed' and r.passed}
            summary['admission_passed'] = verdict['admission_passed'] = passed
            verdict['summary_line'] = '本卡证据全部通过' if passed else '本卡存在失败或证据不足'
        apply_validation_gate(summary, args.case == 'GLM_53_FULL')
        write_report(out, summary)
        summary = finalize(out)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f'结果目录: {out.resolve()}')
        return 0 if summary['admission_passed'] else 2

    return asyncio.run(go())
