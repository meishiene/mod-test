"""Execute the actual report renderer in Node, without browser or network dependencies."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def render_report_table(manifest):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required to execute the report renderer')
    script = r'''
const fs = require('node:fs');
const source = fs.readFileSync('web/static/app.js', 'utf8');
const helper = name => source.split('\n').find(line => line.startsWith(`function ${name}(`) || line.startsWith(`const ${name} =`));
eval(helper('escapeHtml') + '\n' + helper('statusLabel') + '\n' +
     source.slice(source.indexOf('const CASE_STATUS_META'), source.indexOf('function renderBatchReport')) +
     '\nprocess.stdout.write(batchSuiteTable(JSON.parse(process.argv[1])));');
'''
    return subprocess.run([node, '-e', script, json.dumps(manifest)], cwd=ROOT,
                          check=True, text=True, capture_output=True).stdout


@pytest.mark.parametrize('case_name,signer,generator', [
    ('GPT_IMAGE_2', 'OpenAI Media Service', 'OpenAI Media Service API · gpt-image'),
    ('GEMINI_IMAGE_GEN', 'Google Test Image Signer', 'Test Google Generator · Gemini'),
])
def test_image_round_renders_each_fingerprint_and_escapes_metadata(case_name, signer, generator):
    images = [dict(name=f'image-{i}.png', url=f'/api/runs/r1/images/image-{i}.png', fields=[
        {'label': '图片指纹', 'value': '内容签名（C2PA）'},
        {'label': '签名主体', 'value': signer},
        {'label': '签发者', 'value': '<script>issuer & CA</script>'},
        {'label': '指纹生成器', 'value': generator},
        {'label': '签名校验', 'value': 'ES256 · 校验通过 · 有不可见水印'},
    ]) for i in range(10)]
    html = render_report_table({'results': [{'record_id': 'r1', 'case': case_name,
                                           'image_fingerprints': images}]})
    assert html.count('class="report-image-fingerprint"') == 10
    assert '无用例级结果' not in html
    assert 'image-9.png' in html and '查看原图' in html
    assert html.count('ES256 · 校验通过 · 有不可见水印') == 10
    assert '<script>issuer' not in html
    assert '&lt;script&gt;issuer &amp; CA&lt;/script&gt;' in html
    assert html.count(f'<dd>{signer}</dd>') == 10
    assert html.count(f'<dd>{generator}</dd>') == 10
    if case_name == 'GEMINI_IMAGE_GEN':
        assert 'OpenAI' not in html


@pytest.mark.parametrize('case_name', ['GPT_IMAGE_2', 'GEMINI_IMAGE_GEN'])
def test_image_round_without_saved_images_has_specific_empty_state(case_name):
    html = render_report_table({'results': [{'case': case_name, 'image_fingerprints': []}]})
    assert '暂无已保存的图片，无法读取图片指纹' in html
    assert '无用例级结果' not in html


def test_ordinary_suite_still_renders_case_results():
    html = render_report_table({'results': [{'case': 'FABLE5_P0', 'summary': {'results': [
        {'case_id': 'FABLE5-E01', 'name': '普通用例', 'status': 'passed', 'details': '原有证据'},
    ]}}]})
    assert '普通用例' in html and '原有证据' in html
    assert 'report-image-fingerprint' not in html


def test_round_sections_display_real_evidence_and_do_not_render_empty_case_table():
    details = {'sections': [{'title': '请求明细', 'columns': [
        {'key': 'id', 'label': '请求'}, {'key': 'status', 'label': '请求状态'}, {'key': 'detail', 'label': '说明'}],
        'rows': [{'id': 'req-1', 'status': 'completed', 'detail': 'TTFT 0.000s；输出 0'},
                 {'id': 'req-2', 'status': 'failed', 'detail': '<script>bad</script>'}]}],
        'notes': ['仅展示前 200 条'], 'links': [
            {'label': '完整请求明细', 'url': '/api/runs/r1/requests'},
            {'label': '恶意链接', 'url': 'javascript:alert(1)'}]}
    html = render_report_table({'results': [{'case': 'T0', 'report_details': details}]})
    assert 'req-1' in html and 'TTFT 0.000s；输出 0' in html
    assert '已完成' in html and '仅展示前 200 条' in html
    assert '无用例级结果' not in html and '暂无已保存' not in html
    assert '&lt;script&gt;bad&lt;/script&gt;' in html and '<script>bad' not in html
    assert '/api/runs/r1/requests' in html and 'javascript:' not in html


def test_missing_images_and_request_errors_are_both_visible():
    details = {'sections': [{'title': '请求明细', 'columns': [{'key': 'detail', 'label': '失败原因'}],
                            'rows': [{'detail': '图片请求超时'}]}]}
    html = render_report_table({'results': [{'case': 'GEMINI_IMAGE_GEN', 'image_fingerprints': [], 'report_details': details}]})
    assert '暂无已保存的图片' in html and '图片请求超时' in html


def test_legacy_case_boolean_verdict_and_zero_duration():
    html = render_report_table({'results': [{'summary': {'results': [
        {'id': 'old-case', 'passed': True, 'duration_seconds': 0, 'details': '证据'},
    ]}}]})
    assert 'old-case' in html and '通过' in html and '>0.00<' in html
