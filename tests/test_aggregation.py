import json
import tempfile
import unittest
from pathlib import Path
from web.aggregation import aggregate_batch,build_item_metrics,ensure_batch_metrics,enrich_summary,suite_request_metrics,SCHEMA_VERSION

class AggregationTests(unittest.TestCase):
    def test_enrich_and_aggregate(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); child=root/'child';child.mkdir();summary=child/'summary.json'
            summary.write_text(json.dumps({'sent':10,'success_rate':1,'ttft_p90_seconds':1.25,'ttfc_p90_seconds':3.5,'admission_passed':True}),encoding='utf-8')
            enriched=enrich_summary(summary,record_id='r1',batch_id='b1',test_kind='load',case_name='T0',protocol='volcengine',base_url='https://x',model='m')
            self.assertEqual(enriched['schema_version'],SCHEMA_VERSION);self.assertEqual(enriched['record_id'],'r1')
            out=root/'batch';batch={'id':'b1','test_kind':'load','protocol':'volcengine','base_url':'https://x','model':'m'}
            childrow={'id':'r1','case_name':'T0','status':'completed','passed':1,'result_dir':str(child),'report_path':str(child/'report.md'),'summary_path':str(summary)}
            result=aggregate_batch(out,batch,[childrow])
            self.assertTrue(result['admission_passed']);self.assertEqual(result['child_record_ids'],['r1'])
            self.assertTrue((out/'aggregation_manifest.json').exists());report=(out/'report.md').read_text();self.assertIn('r1',report);self.assertIn('TTFT P90',report);self.assertIn('1.250',report)
            # 压测 summary 自带字段应进入 manifest 的 metrics
            self.assertEqual(result['results'][0]['metrics']['ttft_p90_seconds'],1.25)
            self.assertEqual(result['results'][0]['metrics']['requests'],10)

    def test_metrics_from_request_evidence(self):
        """能力专项 summary 没有压测字段时，指标应从 requests.jsonl 的 timing 还原。"""
        with tempfile.TemporaryDirectory() as td:
            child=Path(td); rows=[
                {'http_status':200,'transport_error':None,'timing':{'ttft_ms':1000.0,'ttfc_ms':1500.0,'e2e_ms':2000.0}},
                {'http_status':200,'transport_error':None,'timing':{'ttft_ms':2000.0,'ttfc_ms':2500.0,'e2e_ms':3000.0}},
                {'http_status':500,'transport_error':'boom','timing':{'ttft_ms':None,'ttfc_ms':None,'e2e_ms':9000.0}},
            ]
            (child/'requests.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
            metrics=suite_request_metrics(child)
            self.assertEqual(metrics['requests'],3);self.assertEqual(metrics['http_success'],2)
            self.assertAlmostEqual(metrics['success_rate'],2/3)
            self.assertAlmostEqual(metrics['ttft_p90_seconds'],2.0)
            self.assertAlmostEqual(metrics['ttfc_p90_seconds'],2.5)
            self.assertAlmostEqual(metrics['e2e_p90_seconds'],9.0)
            merged=build_item_metrics({'results':[{'status':'passed'},{'status':'warning'},{'status':'blocked'},{'status':'failed'}]},child)
            self.assertEqual(merged['case_total'],4);self.assertEqual(merged['case_passed'],1)
            self.assertEqual(merged['case_warning'],1);self.assertEqual(merged['case_blocked'],1);self.assertEqual(merged['case_failed'],1)
            self.assertAlmostEqual(merged['ttft_p90_seconds'],2.0)

    def test_ensure_batch_metrics_backfills_legacy_manifest(self):
        """旧批次 manifest 没有 metrics 字段时，ensure_batch_metrics 应能补齐。"""
        payload={'record_type':'batch','results':[{'record_id':'r1','result_dir':None,'summary':{}}]}
        ensure_batch_metrics(payload)
        self.assertIn('metrics',payload['results'][0])
        # 非 batch 结构应原样返回，不报错
        self.assertEqual(ensure_batch_metrics({'record_type':'single'}),{'record_type':'single'})

    def test_multi_model_batch_report_groups_by_model(self):
        """同用例 × 多模型批次：manifest 带 model，报告首列按模型分行并给模型范围。"""
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            children=[]
            for index,model in enumerate(('claude-fable-5','claude-opus-5')):
                child=root/f'child-{index}';child.mkdir()
                (child/'summary.json').write_text(json.dumps({'admission_passed':True,'results':[
                    {'case_id':'SIG-01','name':'原签名回传','status':'passed','details':'ok'},
                    {'case_id':'SIG-02','name':'变异签名被拒','status':'passed','details':'400'},
                ]}),encoding='utf-8')
                children.append({'id':f'r{index}','case_name':'PROTOBUF_SIGNATURE_CHECK','model':model,
                                 'status':'completed','passed':1,'result_dir':str(child),
                                 'report_path':str(child/'report.md'),'summary_path':str(child/'summary.json')})
            out=root/'batch'
            batch={'id':'b1','test_kind':'fable','protocol':'anthropic','base_url':'https://x','model':'2 个模型'}
            result=aggregate_batch(out,batch,children)
            self.assertEqual([x['model'] for x in result['results']],['claude-fable-5','claude-opus-5'])
            report=(out/'report.md').read_text(encoding='utf-8')
            self.assertIn('| 模型 | 子记录 ID |',report)
            self.assertIn('| claude-fable-5 | `r0` |',report)
            self.assertIn('| claude-opus-5 | `r1` |',report)
            self.assertIn('- 模型范围：claude-fable-5、claude-opus-5',report)
            self.assertNotIn('| 轮次 |',report)

    def test_single_model_batch_report_keeps_case_rows(self):
        """普通批次（子记录 model 与父一致）：报告维持按用例/轮次分行。"""
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);child=root/'child';child.mkdir()
            (child/'summary.json').write_text(json.dumps({'admission_passed':True}),encoding='utf-8')
            out=root/'batch'
            batch={'id':'b1','test_kind':'load','protocol':'volcengine','base_url':'https://x','model':'m'}
            childrow={'id':'r1','case_name':'T0','model':'m','status':'completed','passed':1,
                      'result_dir':str(child),'report_path':str(child/'report.md'),'summary_path':str(child/'summary.json')}
            aggregate_batch(out,batch,[childrow])
            report=(out/'report.md').read_text(encoding='utf-8')
            self.assertIn('| 轮次 | 子记录 ID |',report)
            self.assertIn('| T0 | `r1` |',report)
            self.assertNotIn('模型范围',report)

if __name__=='__main__':unittest.main()
