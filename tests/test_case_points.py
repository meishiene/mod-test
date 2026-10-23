"""Per-case contributions must reconcile with the authoritative 100 + 20 score."""
import copy

import pytest

from common.score_registry import CaseMeta
from common.scoring import score_rows, score_batch
from web.report_details import ensure_report_details
from test_image_report_ui import render_report_table


def test_contributions_include_all_deductions_and_reconcile_each_side():
    inv = {f'M{i}': CaseMeta(f'M{i}', '', 'P0', 'core') for i in range(12)}
    inv['C'] = CaseMeta('C', '', 'P0', 'core')
    rows = [{'case_id': cid, 'status': 'failed'} for cid in inv]
    rows[0]['status'] = 'passed'
    rows[-1]['status'] = 'warning'
    report = score_rows(rows, test_kind='qwen', inventory_override=inv, case_faces={'C': 'channel'})
    assert len(report['deductions']) == 8
    points = report['case_scores']
    assert len(points) == 13
    assert points['C']['earned'] == points['C']['lost'] == 10
    for side in ('model', 'channel'):
        selected = [p for p in points.values() if p['side'] == side]
        assert sum(p['earned'] for p in selected) == pytest.approx(report['components'][side]['earned'], abs=0.0001)
        assert sum(p['earned'] + p['lost'] for p in selected) == pytest.approx(report['components'][side]['max_score'])


def test_missing_evidence_and_veto_do_not_invent_deductions():
    inv = {cid: CaseMeta(cid, '', 'P0', face) for cid, face in [('M','core'), ('C','core'), ('V','trust'), ('P','performance')]}
    report = score_rows([{'case_id':'M','status':'blocked'}, {'case_id':'V','status':'failed'},
                         {'case_id':'P','status':'passed'}], test_kind='qwen', inventory_override=inv,
                        case_faces={'C':'channel'})
    assert report['case_scores']['M']['earned'] is None
    assert report['case_scores']['M']['lost'] is None
    assert report['case_scores']['C']['lost'] is None
    assert report['case_scores']['V']['veto_failed']
    assert report['case_scores']['P']['weight'] == 0
    assert report['valid'] is False


def test_batch_details_match_parent_weights_and_latest_retest(tmp_path):
    items = [{'model': model, 'summary': {'test_kind':'deepseek', 'model':model,
              'results':[{'case_id':'DS-V4-26','status':status}, {'case_id':'DS-V4-30','status':'warning'}]}}
             for model, status in [('m','passed'), ('m','failed'), ('other','passed')]]
    payload = {'results':items}
    originals = copy.deepcopy(items)
    expected = {(r['test_kind'],r['model']):r for r in score_batch(items)}
    ensure_report_details(payload, tmp_path, tmp_path / 'results')
    for index, item in enumerate(items):
        details = item['report_details']
        section = details['sections'][0]
        assert [c['key'] for c in section['columns']][:3] == ['id','points','name']
        point = section['rows'][0]['points']
        assert point['superseded'] is (index == 0)
        assert point['weight'] == expected['deepseek',item['model']]['case_scores']['DS-V4-26']['weight']
        assert item['summary'] == originals[index]['summary']
    assert items[1]['report_details']['sections'][0]['rows'][0]['points']['lost'] > 0
    assert items[2]['report_details']['sections'][0]['rows'][0]['points']['earned'] > 0
    html = render_report_table(payload)
    assert '复测已替代' in html and 'case-points-loss' in html and 'case-points-gain' in html
    assert '通道侧' in html
    assert 'data-col="duration"' in html


def test_points_renderer_covers_partial_missing_and_zero_weight():
    rows = [dict(id=str(i), points=p) for i,p in enumerate([
        dict(weight=4, earned=2, lost=2, side='model'),
        dict(weight=4, earned=None, lost=None, side='channel'),
        dict(weight=0, earned=0, lost=0, side='model'),
        dict(weight=0, earned=0, lost=0, veto_failed=True),
    ])]
    html = render_report_table({'results':[{'report_details':{'sections':[{'title':'用例结果',
        'columns':[{'key':'id','label':'用例'},{'key':'points','label':'分值'}], 'rows':rows}]}}]})
    assert '+2.00' in html and '−2.00' in html
    assert '模型侧' in html and '通道侧' in html
    assert '未计分' in html and '不计分' in html and '评分无效' in html
