"""按已有执行证据构建报告轮次明细；只改变展示，不回写裁定或原文件。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from common.scoring import score_batch, score_summary

LIMIT = 200
VERDICTS = {'passed', 'failed', 'error', 'warning', 'blocked', 'skipped', 'not_evaluable', 'interrupted'}
TASK_KINDS = {'deepswe', 'swe-atlas-tw', 'swe-atlas-rf', 'legal-agent-benchmark'}


def _first(row: dict, *keys: str) -> Any:
    return next((row[k] for k in keys if row.get(k) is not None and row[k] != ''), None)


def _text(value: Any) -> str:
    if value is None or value == '':
        return '—'
    if isinstance(value, bool):
        return '是' if value else '否'
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {'true', 'false'}:
        return value.lower() == 'true'
    return None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _seconds(row: dict, name: str) -> str:
    timing = row.get('timing') if isinstance(row.get('timing'), dict) else {}
    value = _number(row.get(f'{name}_seconds'))
    if value is None:
        ms = _number(_first(timing, f'{name}_ms'))
        if ms is None:
            ms = _number(row.get(f'{name}_ms'))
        if ms is not None:
            value = ms / 1000
    if value is None and name == 'e2e':
        value = _number(_first(row, 'duration_seconds', 'elapsed_seconds'))
        if value is None and _number(row.get('latency_ms')) is not None:
            value = float(row['latency_ms']) / 1000
    return '—' if value is None else f'{value:.3f}s'


def _status(row: dict, *, transport: bool = False) -> str:
    for key in ('case_status', 'verdict', 'status'):
        if isinstance(row.get(key), str) and row[key] in VERDICTS:
            return row[key]
    passed = _bool(_first(row, 'case_passed', 'passed'))
    if passed is not None:
        return 'passed' if passed else 'failed'
    if row.get('status') == 'ok' and isinstance(row.get('correct'), bool):
        return 'passed' if row['correct'] else 'failed'
    if transport:
        success = _bool(row.get('success'))
        if success is not None:
            return 'completed' if success else 'failed'
        if _first(row, 'transport_error', 'error_message', 'error'):
            return 'error'
        http = _number(row.get('http_status'))
        if http is not None:
            return 'completed' if 200 <= http < 300 else 'failed'
        if row.get('status') in {'ok', 'completed'}:
            return 'completed'
    return 'not_evaluable'


def _dicts(value: Any) -> list[dict]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _safe_file(directory: Path | None, name: str) -> Path | None:
    if directory is None:
        return None
    path = directory / name
    if path.is_symlink() or not path.is_file() or directory.resolve() not in path.resolve().parents:
        return None
    return path


def _evidence(directory: Path | None, notes: list[str]) -> tuple[list[dict], str | None]:
    for name in ('requests.jsonl', 'execution.jsonl', 'requests.csv'):
        path = _safe_file(directory, name)
        if path is None:
            continue
        rows = []
        invalid = 0
        try:
            with path.open(encoding='utf-8-sig', errors='replace') as stream:
                for raw in (csv.DictReader(stream) if name.endswith('.csv') else stream):
                    if isinstance(raw, str) and not raw.strip():
                        continue
                    try:
                        row = raw if isinstance(raw, dict) else json.loads(raw)
                    except ValueError:
                        invalid += 1
                        continue
                    if not isinstance(row, dict):
                        invalid += 1
                        continue
                    rows.append(row)
                    if len(rows) > LIMIT:
                        notes.append(f'明细仅展示前 {LIMIT} 条；完整记录请查看请求明细或下载全部。')
                        break
        except OSError:
            notes.append(f'{name} 无法读取。')
            continue
        if invalid:
            notes.append(f'{name} 中有 {invalid} 条记录无法解析，已跳过。')
        if rows:
            return rows[:LIMIT], name
    return [], None


def _request_rows(rows: list[dict]) -> list[dict]:
    result = []
    for index, row in enumerate(rows, 1):
        phase = {'warmup': '预热', 'measured': '正式', 'baseline': '基线'}.get(row.get('phase'), row.get('phase') or '未标注阶段')
        counted = _bool(row.get('counted'))
        if counted is not None:
            phase += ' / 计入统计' if counted else ' / 不计入统计'
        usage = row.get('usage') if isinstance(row.get('usage'), dict) else {}
        tokens_in = _first(row, 'prompt_tokens', 'input_tokens')
        tokens_out = _first(row, 'completion_tokens', 'output_tokens')
        timing = f'TTFT {_seconds(row, "ttft")}；TTFC {_seconds(row, "ttfc")}；E2E {_seconds(row, "e2e")}'
        timing += f'；Token 输入 {_text(tokens_in if tokens_in is not None else _first(usage, "prompt_tokens", "input_tokens"))}'
        timing += f' / 输出 {_text(tokens_out if tokens_out is not None else _first(usage, "completion_tokens", "output_tokens"))}'
        tpot = _number(_first(row, 'tpot_ms_per_visible_token', 'tpot_ms_per_token'))
        if tpot is not None:
            timing += f'；TPOT {tpot:.3f}ms/token'
        detail = _first(row, 'error_message', 'transport_error', 'error', 'verdict_detail', 'details', 'detail')
        if detail is None:
            bits = [f'{label}：{_text(row[key])}' for key, label in (
                ('finish_reason', '结束原因'), ('image_count', '返回图片'), ('image_save_error', '图片保存错误'),
                ('task_id', '视频任务'), ('task_status', '任务状态'), ('video_url', '视频链接'),
            ) if row.get(key) is not None]
            detail = '；'.join(bits) or None
        result.append(dict(id=_text(_first(row, 'request_id', 'case_id', 'id', 'index', 'sequence') if
                                    _first(row, 'request_id', 'case_id', 'id', 'index', 'sequence') is not None else index),
                           phase=phase, status=_status(row, transport=True), http=_text(row.get('http_status')),
                           timing=timing, detail=_text(detail)))
    return result


def _case_rows(rows: list[dict]) -> list[dict]:
    return [dict(id=_text(_first(row, 'case_id', 'id', 'task', 'index') if _first(row, 'case_id', 'id', 'task', 'index') is not None else index),
                 name=_text(_first(row, 'name', 'title')), status=_status(row), duration=_seconds(row, 'e2e'),
                 detail=_text(_first(row, 'details', 'detail', 'error', 'message', 'reason')))
            for index, row in enumerate(rows, 1)]


def _task_rows(rows: list[dict]) -> list[dict]:
    output = []
    for index, row in enumerate(rows, 1):
        status = _status(row)
        if row.get('run_exit_code') not in (None, 0) or row.get('eval_exit_code') not in (None, 0):
            status = 'error'
        elif row.get('evaluated') is False:
            status = 'not_evaluable'
        elif status == 'not_evaluable':
            # 使用运行器原有阈值判断结论，分数本身不进入报告展示数据。
            value = _number(_first(row, 'reward', 'pass_rate', 'score'))
            if value is not None:
                status = 'passed' if value >= 1 else 'failed'
        detail = '；'.join(f'{label}：{_text(row[key])}' for key, label in (
            ('run_id', '运行 ID'), ('evaluated', '已评测'), ('run_exit_code', '运行退出码'),
            ('eval_exit_code', '评测退出码'), ('error', '错误'), ('message', '说明'),
        ) if row.get(key) is not None)
        artifact = _first(row, 'artifact', 'score_artifact')
        # 证据路径用于定位任务；不读取或展开其中的评分内容。
        task_id = _first(row, 'task', 'task_id', 'id', 'case_id')
        if task_id is None and artifact:
            task_id = str(Path(str(artifact)).parent)
        output.append(dict(id=_text(task_id if task_id is not None else index), status=status,
                           duration=_seconds(row, 'e2e'), detail=detail or '—', artifact=_text(artifact)))
    return output


def build_report_details(item: dict, kind: str, result_dir: Path | None) -> dict:
    summary = item.get('summary') if isinstance(item.get('summary'), dict) else {}
    details: dict[str, Any] = {'sections': [], 'notes': [], 'links': []}
    notes = details['notes']

    def add(title: str, columns: list[tuple[str, str]], rows: list[dict]):
        if rows:
            details['sections'].append({'title': title, 'columns': [{'key': key, 'label': label} for key, label in columns], 'rows': rows[:LIMIT]})
            if len(rows) > LIMIT:
                notes.append(f'{title}仅展示前 {LIMIT} 条；完整记录请查看指标明细或下载全部。')

    cases = _dicts(summary.get('results')) or _dicts(summary.get('records'))
    suites = _dicts(summary.get('suites'))
    needs_requests = not cases or kind in {'load', 'strict', 'gpt-image', 'gemini-image', 'arxivmath', 'moderation'}
    rows, _ = _evidence(result_dir, notes) if needs_requests and not suites else ([], None)
    if kind == 'arxivmath':
        add('题目结果', [('id', '题目'), ('name', '名称'), ('status', '结果'), ('duration', '耗时'), ('expected', '标准答案'), ('actual', '实际答案'), ('detail', '错误')],
            [dict(id=_text(_first(r, 'index', 'id', 'paper_id')), name=_text(r.get('title')), status=_status(r),
                  duration=_seconds(r, 'e2e'), expected=_text(r.get('gold_answer')), actual=_text(r.get('predicted_answer')),
                  detail=_text(r.get('error'))) for r in (cases or rows)])
    elif kind == 'moderation':
        review_rows = []
        for i, r in enumerate(cases or rows, 1):
            actual = r.get('parsed_result') if isinstance(r.get('parsed_result'), dict) else {}
            checks = r.get('checks') if isinstance(r.get('checks'), dict) else {}
            failures = [name for name, passed in checks.items() if passed is False]
            review_rows.append(dict(id=_text(_first(r, 'case_id', 'id', 'request_id') or i), status=_status(r), duration=_seconds(r, 'e2e'),
                expected=f'{_text(r.get("expected_decision"))} / {_text(r.get("expected_category"))}',
                actual=f'{_text(actual.get("decision"))} / {_text(actual.get("category"))}',
                detail='；'.join(filter(None, [str(_first(r, 'error', 'parse_error') or ''), str(actual.get('reason') or ''),
                                             '未通过：' + '、'.join(failures) if failures else ''])) or '—'))
        add('审核结果', [('id', '用例'), ('status', '结果'), ('duration', '耗时'), ('expected', '预期判定 / 分类'), ('actual', '实际判定 / 分类'), ('detail', '原因')], review_rows)
    elif kind in TASK_KINDS:
        add('任务结果', [('id', '任务'), ('status', '结果'), ('duration', '耗时'), ('detail', '执行信息'), ('artifact', '证据位置')], _task_rows(cases or rows))
    elif cases:
        add('用例结果', [('id', '用例'), ('name', '名称'), ('status', '状态'), ('duration', '耗时'), ('detail', '结论说明')], _case_rows(cases))

    if suites:
        suite_rows = []
        failures = []
        for suite in suites:
            failed = any((_number(suite.get(k)) or 0) > 0 for k in ('failed', 'errors')) or suite.get('exit_code') not in (None, 0)
            status = 'failed' if failed else 'passed' if (_number(suite.get('passed')) or 0) > 0 else 'not_evaluable'
            suite_rows.append(dict(name=_text(suite.get('suite')), status=status, duration=_seconds(suite, 'e2e'),
                counts='；'.join(f'{label} {_text(suite.get(key))}' for key, label in (
                    ('total', '总计'), ('passed', '通过'), ('failed', '失败'), ('errors', '错误'), ('skipped', '跳过'))), exit_code=_text(suite.get('exit_code'))))
            failures.extend(dict(name=_text(r.get('name')), suite=_text(suite.get('suite')), status='failed', detail=_text(_first(r, 'message', 'error')))
                            for r in _dicts(suite.get('failed_cases')))
        add('官方子套件', [('name', '子套件'), ('status', '结果'), ('duration', '耗时'), ('counts', '用例统计'), ('exit_code', '退出码')], suite_rows)
        add('失败用例', [('name', '用例'), ('suite', '子套件'), ('status', '结果'), ('detail', '失败原因')], failures)

    if rows and kind not in TASK_KINDS | {'arxivmath', 'moderation'}:
        add('请求明细', [('id', '请求'), ('phase', '阶段 / 统计范围'), ('status', '请求状态'), ('http', 'HTTP'), ('timing', '耗时 / Token'), ('detail', '说明')], _request_rows(rows))

    checks = summary.get('checks') if isinstance(summary.get('checks'), dict) else {}
    check_rows = [dict(v, name=k) if isinstance(v, dict) else {'name': k, 'passed': v} for k, v in checks.items()]
    if not check_rows:
        validation = summary.get('validation') if isinstance(summary.get('validation'), dict) else {}
        check_rows = _dicts(validation.get('checks'))
    display_checks = []
    for check in check_rows:
        if any(word in str(check.get('name')).lower() for word in ('score', '评分', '得分')):
            continue
        actual = _first(check, 'value', 'actual', 'http_status')
        expected = check.get('expected')
        if expected is None:
            expected = '；'.join(f'{label} {_text(check[key])}' for key, label in [('minimum', '最小值'), ('maximum', '最大值')] if check.get(key) is not None) or None
        # 统一收尾校验的 detail 是失败解释，即便 passed=True 也保留在原数据里。
        # 展示成功项时不能把“缺少证据”等失败文案一起显示。
        explanation = None if _bool(check.get('passed')) is True else _first(check, 'error', 'diagnosis', 'detail')
        display_checks.append(dict(name=_text(check.get('name')), status=_status({'passed': check.get('passed')}),
                                   actual=_text(actual), expected=_text(expected), detail=_text(explanation)))
    # 正常能力套件已有逐例结论，无需重复统一收尾校验；无逐例结果时展示实际判定依据。
    if not cases or kind in {'load', 'strict', 'gpt-image', 'gemini-image'}:
        add('校验结果', [('name', '检查项'), ('status', '结果'), ('expected', '要求'), ('actual', '实际'), ('detail', '说明')], display_checks)

    if not cases and not rows and not suites:
        add('执行统计', [('name', '项目'), ('value', '记录值')], [
            {'name': label, 'value': _text(summary[key])} for key, label in (
                ('dataset', '数据集'), ('runner', '运行器'), ('requested', '计划请求'),
                ('sent', '实际请求'), ('successful', '成功请求'), ('failed', '失败数'),
                ('total_cases', '用例数'), ('passed_cases', '通过用例'), ('failed_cases', '失败用例'),
                ('requested_tasks', '计划任务'), ('evaluated_tasks', '已评测任务'),
                ('passed_tasks', '通过任务'), ('total_problems', '题目数'), ('evaluated', '有效回答'),
                ('exit_code', '进程退出码'),
            ) if summary.get(key) is not None
        ])

    error = _first(summary, 'error', 'error_message') or item.get('error')
    if error or not details['sections']:
        state = item.get('status') or summary.get('status') or 'not_evaluable'
        info = _text(error) if error else '暂无已保存的执行明细；任务可能尚未执行或未产出结果。'
        add('执行情况', [('name', '测试项'), ('status', '执行状态'), ('detail', '说明')],
            [dict(name=_text(item.get('case') or summary.get('case')), status=state, detail=info)])
    run_id = quote(str(item.get('record_id') or ''), safe='')
    if run_id:
        for name, label, suffix in [('requests.jsonl', '完整请求明细', 'requests'), ('summary.json', '指标明细', 'summary')]:
            if _safe_file(result_dir, name):
                details['links'].append({'label': label, 'url': f'/api/runs/{run_id}/{suffix}'})
        if item.get('log_path'):
            details['links'].append({'label': '执行日志', 'url': f'/api/runs/{run_id}/log'})
        if result_dir is not None:
            details['links'].append({'label': '下载完整执行数据', 'url': f'/api/runs/{run_id}/download'})
    return details


def ensure_report_details(payload: dict, root: Path, results_root: Path) -> None:
    """读取时补齐单项和历史批次的展开明细；不修改原 summary。"""
    resolved = []
    for item in _dicts(payload.get('results')):
        directory = None
        value = item.get('result_dir')
        if value:
            candidate = (root / str(value)).resolve()
            if results_root.resolve() in candidate.parents and candidate.is_dir():
                directory = candidate
        summary = item.get('summary') if isinstance(item.get('summary'), dict) else {}
        if not summary:
            path = _safe_file(directory, 'summary.json')
            if path:
                try:
                    loaded = json.loads(path.read_text(encoding='utf-8'))
                    summary = loaded if isinstance(loaded, dict) else {}
                except (OSError, ValueError):
                    pass
        kind = summary.get('test_kind') or item.get('test_kind') or payload.get('test_kind') or ''
        item['report_details'] = build_report_details(dict(item, summary=summary), kind, directory)
        resolved.append(dict(item, summary=summary, test_kind=kind))

    # Use the complete batch's weights, including merged scope and latest retests.
    # Never derive contributions from the truncated top-deductions list.
    reports = score_batch(resolved)
    if len(resolved) == 1 and not resolved[0]['summary'].get('results'):
        reports = [score_summary(resolved[0]['summary'])]
    scores = {(r.get('test_kind') or '', r.get('model') or ''): r for r in reports if r}
    latest = {}
    for index, item in enumerate(resolved):
        summary = item['summary']
        group = (summary.get('test_kind') or item.get('test_kind') or '', item.get('model') or summary.get('model') or '')
        for position, row in enumerate(_dicts(summary.get('results')) or _dicts(summary.get('records'))):
            cid = row.get('case_id') or row.get('id')
            if cid:
                latest[group + (str(cid),)] = (index, position)
    for index, item in enumerate(resolved):
        summary = item['summary']
        group = (summary.get('test_kind') or item.get('test_kind') or '', item.get('model') or summary.get('model') or '')
        report = scores.get(group) or {}
        for section in item['report_details']['sections']:
            if section['title'] != '用例结果':
                continue
            section['columns'].insert(1, {'key': 'points', 'label': '分值'})
            for position, row in enumerate(section['rows']):
                points = report.get('case_scores', {}).get(row['id'])
                row['points'] = dict(points, superseded=latest.get(group + (row['id'],)) != (index, position)) if points else None
            item['report_details']['notes'].append('分值按本报告权重计算：+ 为实际得分，− 为相对该用例满分的失分；总分累加实际得分。同一用例复测仅计最后一次，未测项不计得分。')
            if report.get('valid') is False:
                item['report_details']['notes'].append('信任校验存在失败项，报告总评分无效；下列分值仅展示各项原始得失分。')
