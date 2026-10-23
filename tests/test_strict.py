import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from strictperf.cases import CASES
from strictperf import runner


class StrictRunnerTests(unittest.TestCase):
    def test_cases_match_k3test_strict_rounds(self):
        self.assertEqual([case.upstream_case for case in CASES.values()], ["T0", "L1", "L2", "L3", "L4", "C1"])
        self.assertEqual(CASES["STRICT_L1"].concurrency, 50)
        self.assertEqual(CASES["STRICT_C1"].request_count, 7200)

    def test_execute_calls_upstream_and_renames_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); home = root / "k3test"; output = root / "results"
            (home / "scripts").mkdir(parents=True); (home / ".venv/bin").mkdir(parents=True)
            (home / "scripts/run_T0.py").write_text("")
            (home / ".venv/bin/python").write_text("")
            def fake_run(command, cwd, env):
                out = output / "T0-20260902-180000"; out.mkdir(parents=True)
                (out / "summary.json").write_text(json.dumps({"case":"T0","admission_passed":True}))
                (out / "report.md").write_text("# report")
                return type("Process", (), {"returncode": 0})()
            variables = {"VOLC_API_KEY":"secret", "VOLC_BASE_URL":"https://example.com/v1", "VOLC_MODEL":"m", "K3TEST_HOME":str(home)}
            with patch.dict(os.environ, variables, clear=True), patch.object(runner.subprocess, "run", side_effect=fake_run) as called:
                out, summary, code = runner.execute("STRICT_T0", output)
            self.assertEqual(out.name, "STRICT_T0-20260902-180000")
            self.assertEqual(summary["upstream_case"], "T0")
            self.assertTrue(summary["strict_official_method"])
            self.assertEqual(code, 0)
            self.assertEqual(called.call_args.kwargs["env"]["K3_API_KEY"], "secret")
            self.assertNotIn("secret", called.call_args.args[0])
