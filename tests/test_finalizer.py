import json,tempfile,unittest
from pathlib import Path
from common.finalizer import finalize_result_dir

def _row(status): return {'case_id':'C1','status':status,'case_status':status}

class FinalizerTests(unittest.TestCase):
 def test_warmup_only_load_cannot_pass(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td); (p/'summary.json').write_text(json.dumps({'admission_passed':True,'sent':0,'checks':{}})); (p/'requests.jsonl').write_text('{}\n')
   x=finalize_result_dir(p); self.assertFalse(x['admission_passed']); self.assertTrue((p/'validation.json').exists())
 def test_external_without_scores_cannot_pass(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td); (p/'summary.json').write_text(json.dumps({'admission_passed':True,'scored_tasks':0})); (p/'run.log').write_text('ran')
   self.assertFalse(finalize_result_dir(p)['admission_passed'])
 def test_blocked_does_not_block_admission(self):
  """2026-10-10 口径：blocked 只记录不阻断准入（failed/error/not_evaluable 才拦）。"""
  with tempfile.TemporaryDirectory() as td:
   p=Path(td); (p/'summary.json').write_text(json.dumps({'admission_passed':True,'results':[_row('passed'),_row('blocked')]})); (p/'requests.jsonl').write_text('{}\n')
   x=finalize_result_dir(p)
   self.assertTrue(x['admission_passed'])
   check=[c for c in x['validation']['checks'] if c['name']=='case_verdicts_complete'][0]
   self.assertTrue(check['passed'])
 def test_all_blocked_batch_still_cannot_pass(self):
  """整批只有 blocked：没有任何有效结论，准入无依据，仍不准入。"""
  with tempfile.TemporaryDirectory() as td:
   p=Path(td); (p/'summary.json').write_text(json.dumps({'admission_passed':True,'results':[_row('blocked'),_row('blocked')]})); (p/'requests.jsonl').write_text('{}\n')
   self.assertFalse(finalize_result_dir(p)['admission_passed'])
