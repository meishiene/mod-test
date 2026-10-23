import json
from pathlib import Path

import pytest

from web.report_details import build_report_details, ensure_report_details


def evidence(tmp_path, rows):
    (tmp_path / 'requests.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))


def section(details, title):
    return next(s for s in details['sections'] if s['title'] == title)


def test_load_requests_distinguish_phase_verdict_and_preserve_zero(tmp_path):
    evidence(tmp_path, [
        {'request_id': 'warmup-1', 'phase': 'warmup', 'counted': False, 'success': True, 'http_status': 200,
         'e2e_seconds': 0, 'ttft_seconds': 0, 'ttfc_seconds': 0.2, 'prompt_tokens': 10, 'completion_tokens': 0,
         'tpot_ms_per_visible_token': 0,
         'request_headers': {'Authorization': 'secret'}, 'raw_response_body': 'private'},
        {'request_id': 'formal-1', 'phase': 'measured', 'counted': True, 'success': False,
         'http_status': 503, 'error_message': '上游不可用'},
    ])
    details = build_report_details({'summary': {'checks': {'success_rate': {'passed': False, 'value': 0.5, 'minimum': 0.99}}}}, 'load', tmp_path)
    rows = section(details, '请求明细')['rows']
    assert '预热' in rows[0]['phase'] and '不计入' in rows[0]['phase']
    assert rows[0]['status'] == 'completed'
    assert '0.000s' in rows[0]['timing'] and '输出 0' in rows[0]['timing']
    assert 'TPOT 0.000ms/token' in rows[0]['timing']
    assert '正式' in rows[1]['phase'] and rows[1]['status'] == 'failed'
    assert rows[1]['detail'] == '上游不可用'
    assert section(details, '校验结果')['rows'][0]['status'] == 'failed'
    text = json.dumps(details, ensure_ascii=False)
    assert 'secret' not in text and 'private' not in text


def test_official_kvv_suites_show_counts_and_failure_evidence():
    item = {'summary': {'suites': [dict(suite='params', total=12, passed=10, failed=1, errors=0,
             skipped=1, duration_seconds=2.5, exit_code=1,
             failed_cases=[{'name': 'test_schema', 'class': 'params.schema', 'message': 'expected 400, got 200'}])]}}
    details = build_report_details(item, 'kvv', None)
    row = section(details, '官方子套件')['rows'][0]
    assert row['name'] == 'params' and row['status'] == 'failed'
    assert '通过 10' in row['counts'] and '跳过 1' in row['counts']
    assert section(details, '失败用例')['rows'][0]['detail'] == 'expected 400, got 200'


def test_math_reads_answers_from_evidence(tmp_path):
    evidence(tmp_path, [{'index': 0, 'title': '题目 A', 'status': 'ok', 'correct': False,
                         'gold_answer': '42', 'predicted_answer': '41', 'latency_ms': 1500}])
    rows = section(build_report_details({'summary': {}}, 'arxivmath', tmp_path), '题目结果')['rows']
    assert rows[0]['id'] == '0' and rows[0]['status'] == 'failed'
    assert rows[0]['expected'] == '42' and rows[0]['actual'] == '41'
    assert rows[0]['duration'] == '1.500s'


def test_moderation_uses_domain_status_not_http_status(tmp_path):
    evidence(tmp_path, [{'id': 'MOD-1', 'status': 'ok', 'case_status': 'failed', 'http_status': 200,
        'expected_decision': 'review', 'expected_category': 'risk',
        'parsed_result': {'decision': 'allow', 'category': 'normal', 'reason': 'mistake'},
        'checks': {'decision': False, 'category': False}, 'latency_ms': 250}])
    row = section(build_report_details({'summary': {}}, 'moderation', tmp_path), '审核结果')['rows'][0]
    assert row['status'] == 'failed'
    assert 'review' in row['expected'] and 'allow' in row['actual']
    assert 'decision' in row['detail']


@pytest.mark.parametrize('kind', ['deepswe', 'swe-atlas-tw', 'swe-atlas-rf', 'legal-agent-benchmark'])
def test_external_tasks_show_conclusions_without_scores(tmp_path, kind):
    evidence(tmp_path, [{'task': 'fix-a', 'reward': 1, 'score': 0.8, 'artifact': 'task-a/reward.json'},
                        {'task': 'fix-b', 'evaluated': False, 'run_exit_code': 1, 'score': 0.9}])
    details = build_report_details({'summary': {}}, kind, tmp_path)
    rows = section(details, '任务结果')['rows']
    assert rows[0]['id'] == 'fix-a' and rows[0]['status'] == 'passed'
    assert rows[1]['status'] == 'error'
    text = json.dumps(details, ensure_ascii=False)
    assert '0.8' not in text and '0.9' not in text
    assert '得分' not in text and '评分' not in text


def test_existing_cases_use_legacy_ids_and_boolean_verdicts():
    details = build_report_details({'summary': {'records': [{'id': 'legacy-1', 'name': '兼容性', 'passed': True,
                                  'duration_seconds': 0, 'details': '真实证据'}]}}, 'awsb', None)
    row = section(details, '用例结果')['rows'][0]
    assert row['id'] == 'legacy-1' and row['status'] == 'passed' and row['duration'] == '0.000s'


def test_failure_without_results_shows_actual_error_and_unknown_is_not_pass():
    details = build_report_details({'status': 'failed', 'error': '进程退出 2', 'summary': {}}, 'fable', None)
    assert '进程退出 2' in json.dumps(details, ensure_ascii=False)
    unknown = build_report_details({'status': 'running', 'summary': {}}, 'load', None)
    row = section(unknown, '执行情况')['rows'][0]
    assert row['status'] == 'running'
    assert '暂无' in row['detail']


def test_summary_only_run_keeps_known_counts_without_inventing_requests():
    details = build_report_details({'summary': {'sent': 0, 'failed': 0, 'requested': 10, 'score': 100}}, 'load', None)
    rows = section(details, '执行统计')['rows']
    assert {'name': '实际请求', 'value': '0'} in rows
    assert not any(s['title'] == '请求明细' for s in details['sections'])
    assert 'score' not in json.dumps(details)


def test_image_without_saved_files_keeps_request_failure(tmp_path):
    evidence(tmp_path, [{'index': 1, 'success': False, 'error_message': '图片请求超时', 'http_status': 504}])
    details = build_report_details({'image_fingerprints': [], 'summary': {}}, 'gemini-image', tmp_path)
    assert section(details, '请求明细')['rows'][0]['detail'] == '图片请求超时'


def test_bounded_and_malformed_evidence_is_explicit(tmp_path):
    (tmp_path / 'requests.jsonl').write_text('broken\n42\n' + ''.join(
        json.dumps({'request_id': str(i), 'success': True}) + '\n' for i in range(205)))
    details = build_report_details({'summary': {}}, 'load', tmp_path)
    assert len(section(details, '请求明细')['rows']) == 200
    assert any('200' in n for n in details['notes'])
    assert any('无法解析' in n for n in details['notes'])


def test_csv_fallback_and_validation_checks(tmp_path):
    (tmp_path / 'requests.csv').write_text('request_id,success,counted,e2e_seconds\nr1,False,True,0\n')
    details = build_report_details({'summary': {'validation': {'checks': [
        {'name': 'evidence_present', 'passed': False, 'expected': '请求记录', 'actual': [], 'detail': '证据缺失'}]}}}, 'load', tmp_path)
    assert section(details, '请求明细')['rows'][0]['status'] == 'failed'
    assert section(details, '校验结果')['rows'][0]['status'] == 'failed'


def test_successful_validation_does_not_show_failure_only_message():
    details = build_report_details({'summary': {'validation': {'checks': [
        {'name': 'evidence_present', 'passed': True, 'actual': ['requests.jsonl'], 'detail': '缺少执行证据'},
    ]}}}, 'load', None)
    row = section(details, '校验结果')['rows'][0]
    assert row['status'] == 'passed' and row['detail'] == '—'


def test_manifest_enrichment_guards_paths_and_does_not_mutate_summary(tmp_path):
    root = tmp_path; results = root / 'results'; out = results / 'run'; out.mkdir(parents=True)
    evidence(out, [{'request_id': 'r1', 'success': True}])
    original = {'checks': {'connectivity': {'passed': False, 'http_status': 401, 'error': '授权失败'}}}
    (out / 'summary.json').write_text(json.dumps(original))
    payload = {'record_type': 'batch', 'test_kind': 'load', 'results': [
        {'record_id': 'r1', 'result_dir': 'results/run', 'summary': {}},
        {'record_id': 'outside', 'result_dir': str(tmp_path.parent), 'summary': {}},
    ]}
    ensure_report_details(payload, root, results)
    assert section(payload['results'][0]['report_details'], '请求明细')['rows'][0]['id'] == 'r1'
    assert payload['results'][0]['summary'] == {}
    assert json.loads((out / 'summary.json').read_text()) == original
    assert not any(s['title'] == '请求明细' for s in payload['results'][1]['report_details']['sections'])
    outside = tmp_path / 'outside.jsonl'; outside.write_text('{"request_id":"secret"}\n')
    (out / 'requests.jsonl').unlink(); (out / 'requests.jsonl').symlink_to(outside)
    ensure_report_details(payload, root, results)
    assert 'secret' not in json.dumps(payload)


def test_single_and_historical_batch_http_manifests_include_details(tmp_path, monkeypatch):
    import threading
    from urllib.request import urlopen
    from web import server

    results = tmp_path / 'results'; out = results / 'run'; out.mkdir(parents=True)
    evidence(out, [{'request_id': 'http-req-1', 'success': False, 'http_status': 502, 'error_message': 'upstream failure'}])
    summary = {'test_kind': 'load', 'case': 'T0', 'sent': 1, 'success_rate': 0, 'results': []}
    (out / 'summary.json').write_text(json.dumps(summary))
    batch = {'record_type': 'batch', 'test_kind': 'load', 'results': [
        {'record_id': 'single', 'case': 'T0', 'result_dir': 'results/run', 'summary': summary}]}
    (out / 'aggregation_manifest.json').write_text(json.dumps(batch))
    monkeypatch.setattr(server, 'ROOT', tmp_path)
    monkeypatch.setattr(server, 'RESULTS', results)
    monkeypatch.setattr(server, 'DB_PATH', tmp_path / 'runs.db')
    server.initialize()
    with server.db() as db:
        for record, manifest in [('single', None), ('batch', 'results/run/aggregation_manifest.json')]:
            db.execute('INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at,result_dir,summary_path,manifest_path,is_batch) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (record, 'load', 'T0', 'openai', 'https://example.com', 'm', 'completed', 0, '2026-10-22',
                        'results/run', 'results/run/summary.json', manifest, int(record == 'batch')))
    httpd = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True); thread.start()
    try:
        for record in ('single', 'batch'):
            with urlopen(f'http://127.0.0.1:{httpd.server_port}/api/runs/{record}/manifest') as response:
                payload = json.load(response)
            details = payload['results'][0]['report_details']
            row = section(details, '请求明细')['rows'][0]
            assert row['id'] == 'http-req-1' and row['detail'] == 'upstream failure'
    finally:
        httpd.shutdown(); httpd.server_close(); thread.join()
