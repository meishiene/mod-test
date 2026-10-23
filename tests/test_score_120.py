import copy
import json
import tempfile
import unittest
from pathlib import Path

from common.score_registry import CaseMeta
from common.scoring import rating_grade, score_rows, score_summary, score_batch
from web.score_reports import current_score

INV = {"M": CaseMeta("M", "model", "P0", "core"), "C": CaseMeta("C", "channel", "P0", "core")}

class Score120Tests(unittest.TestCase):
    def test_grade_boundaries(self):
        for number, grade in [(None,None),(0,"不可用"),(59.999,"不可用"),(60,"可用"),(69.999,"可用"),
                              (70,"一般"),(79.999,"一般"),(80,"良好"),(89.999,"良好"),(90,"优秀"),
                              (99.999,"优秀"),(100,"十分优秀"),(110,"十分优秀"),(110.001,"官方"),(120,"官方")]:
            with self.subTest(number=number):self.assertEqual(rating_grade(number),grade)

    def test_unselected_channel_is_not_a_free_20_points(self):
        report=score_rows([{"case_id":"M","status":"passed"}],test_kind="qwen",inventory_override=INV,
                          case_faces={"C":"channel"},executed_faces=["model","verify"])
        self.assertEqual(report["score"],100)
        self.assertIsNone(report["channel_score"])
        self.assertEqual(report["grade"],"十分优秀")
        self.assertEqual(report["components"]["channel"]["status"],"not_selected")

    def test_channel_only_does_not_claim_model_acceptance(self):
        report=score_rows([{"case_id":"C","status":"passed"}],test_kind="qwen",inventory_override=INV,
                          case_faces={"C":"channel"},executed_faces=["channel"])
        self.assertEqual(report["channel_score"],20)
        self.assertIsNone(report["model_score"])
        self.assertIsNone(report["score"])
        self.assertIsNone(report["rating_passed"])

    def test_channel_missing_evidence_does_not_earn_points(self):
        report=score_rows([{"case_id":"M","status":"passed"},{"case_id":"C","status":"blocked"}],
                          test_kind="qwen",inventory_override=INV,case_faces={"C":"channel"})
        self.assertEqual(report["score"],100)
        self.assertIsNone(report["grade"])
        self.assertEqual(report["range"],[100,120])

    def test_original_summary_not_mutated(self):
        summary={"test_kind":"qwen","model":"qwen3.6-flash","admission_passed":False,
                 "results":[{"case_id":"QWEN-01","status":"failed"}],"score":{"scoring_version":"old"}}
        original=copy.deepcopy(summary)
        score_summary(summary)
        self.assertEqual(summary,original)

    def test_historical_read_ignores_stale_score_and_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);result=root/'results'/'run';result.mkdir(parents=True)
            summary=result/'summary.json';old=result/'score.json'
            old.write_text(json.dumps({"scoring_version":"conformance-1.0","score":100,"grade":"A+"}))
            payload={"test_kind":"qwen","model":"qwen3.6-flash","results":[{"case_id":"QWEN-01","status":"passed"}]}
            summary.write_text(json.dumps(payload));before=summary.read_bytes();old_before=old.read_bytes()
            row={"result_dir":"results/run"}
            score=current_score(row,root,root/'results')
            self.assertEqual(score['scoring_version'],'conformance-2.1')
            self.assertNotEqual(score['score'],100)
            self.assertEqual(summary.read_bytes(),before);self.assertEqual(old.read_bytes(),old_before)
            payload['results'][0]['status']='failed';summary.write_text(json.dumps(payload))
            self.assertEqual(current_score(row,root,root/'results')['score'],0)

    def test_legacy_score_without_evidence_is_not_relabelled_as_new(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);result=root/'results'/'run';result.mkdir(parents=True)
            (result/'score.json').write_text('{"scoring_version":"conformance-1.0","score":100}')
            self.assertIsNone(current_score({"result_dir":"results/run"},root,root/'results'))

class ScoreAPIIntegrationTests(unittest.TestCase):
    def test_historical_list_detail_and_score_filter_share_new_rating(self):
        import threading
        from contextlib import ExitStack
        from urllib.request import urlopen
        from unittest.mock import patch
        from common.score_registry import inventory
        from web import server
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root=Path(folder);results=root/'results';result=results/'old';result.mkdir(parents=True)
            payload={"test_kind":"qwen","model":"qwen3.6-flash","admission_passed":False,
                     "results":[{"case_id":cid,"status":"passed"} for cid in inventory('qwen','qwen3.6-flash')]}
            summary=result/'summary.json';summary.write_text(json.dumps(payload));before=summary.read_bytes()
            (result/'score.json').write_text('{"scoring_version":"conformance-1.0","score":70,"grade":"C"}')
            for name,value in [('ROOT',root),('RESULTS',results),('DB_PATH',root/'runs.db')]:
                stack.enter_context(patch.object(server,name,value))
            server.initialize()
            with server.db() as conn:
                conn.execute("INSERT INTO runs(id,test_kind,case_name,base_url,model,status,created_at,result_dir,summary_path,passed,score,score_grade) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                             ('old','qwen','suite','https://example.test','qwen3.6-flash','completed','2026-10-22','results/old','results/old/summary.json',0,70,'C'))
            httpd=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
            thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
            def get(path):
                with urlopen(f'http://127.0.0.1:{httpd.server_port}'+path) as reply:return json.load(reply)
            try:
                record=get('/api/runs/old');score=get('/api/runs/old/score')
                self.assertEqual(record['score'],120);self.assertEqual(record['score_grade'],'官方')
                self.assertEqual(record['model_score'],100);self.assertEqual(record['channel_score'],20)
                self.assertFalse(record['passed']);self.assertTrue(record['rating_passed'])
                self.assertEqual(score['score'],record['score'])
                self.assertEqual(get('/api/runs?conclusion=score_pass')['total'],1)
                self.assertEqual(get('/api/runs?conclusion=score_fail')['total'],0)
                self.assertEqual(get('/api/runs?conclusion=score_pending')['total'],0)
                self.assertEqual(summary.read_bytes(),before)
                with server.db() as conn:
                    legacy=conn.execute('SELECT passed,score,score_grade FROM runs WHERE id="old"').fetchone()
                    self.assertEqual(tuple(legacy),(0,70,'C'))
            finally:
                httpd.shutdown();httpd.server_close();thread.join()
