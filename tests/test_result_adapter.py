import unittest
from common.artifacts import SCHEMA_VERSION
from common.result_adapter import normalize_summary
class ResultAdapterTests(unittest.TestCase):
 def test_not_evaluable_blocks_admission(self):
  x=normalize_summary({'schema_version':'old','admission_passed':True,'results':[{'status':'not_evaluable'}]})
  self.assertFalse(x['admission_passed']); self.assertEqual(x['schema_version'],SCHEMA_VERSION)
 def test_blocked_does_not_block_admission(self):
  # 2026-10-10 口径：blocked 只记录不阻断准入（failed/error/not_evaluable 才拦）。
  x=normalize_summary({'schema_version':'old','admission_passed':True,'results':[{'status':'passed'},{'status':'blocked'}]})
  self.assertTrue(x['admission_passed'])
 def test_legacy_pass_is_explicit(self):
  x=normalize_summary({'results':[{'passed':True,'status':'passed'}]})
  self.assertTrue(x['results'][0]['case_passed'])
