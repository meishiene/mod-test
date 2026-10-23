import json
import hashlib
import os
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from kvv.cases import CASES
from kvv.runner import DEFAULT_KVV_HOME, SUITE_PATHS, build_command, official_python, parse_junit, validate_runtime, execute, upstream_info


class OfficialKvvTests(unittest.TestCase):
    def test_cases_map_to_official_suites(self):
        self.assertEqual(set(CASES), {"KVV_K3_FEATURES", "KVV_K3_DYNAMIC_TOOLS", "KVV_K3_THINKING_EFFORT", "KVV_PROMPT_TOKENS", "KVV_PARAMS", "KVV_TOOL_SCHEMA"})
        self.assertEqual(CASES["KVV_K3_FEATURES"].request_count,46)
        self.assertEqual(CASES["KVV_PARAMS"].request_count,18)
        self.assertEqual(CASES["KVV_TOOL_SCHEMA"].request_count,408)
        self.assertNotIn("KVV_K3_FINGERPRINT", CASES)

    def test_all_official_test_files_are_reachable(self):
        selected = set()
        for case in CASES.values():
            for suite in case.suites:
                for relative in SUITE_PATHS[suite]:
                    path = DEFAULT_KVV_HOME / relative
                    selected.update(path.rglob("test_*.py") if path.is_dir() else [path])
        self.assertEqual(selected, set((DEFAULT_KVV_HOME / "tests").rglob("test_*.py")))

    def test_official_suites_collect_offline_with_real_data(self):
        result = subprocess.run(
            [str(official_python(DEFAULT_KVV_HOME)), "-m", "pytest", "tests", "--collect-only", "-q"],
            cwd=DEFAULT_KVV_HOME, capture_output=True, text=True, timeout=60,
            env={**os.environ, "KVV_PARAM_CONTRACT": "k3", "KIMI_API_KEY": "offline", "KIMI_BASE_URL": "http://127.0.0.1:1/v1"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("test_dynamic_tools.py: 54", result.stdout)
        self.assertIn("test_thinking_effort.py: 26", result.stdout)
        self.assertIn("test_prompt_tokens.py: 59", result.stdout)

    def test_prompt_token_assets_match_official_lfs_hashes(self):
        from PIL import Image

        info = upstream_info(DEFAULT_KVV_HOME)
        for relative, expected in info["prompt_token_data_sha256"].items():
            data = (DEFAULT_KVV_HOME / relative).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected, relative)
            self.assertFalse(data.startswith(b"version https://git-lfs"))
        data_dir = DEFAULT_KVV_HOME / "testdata/prompt_token_cases"
        for name, count in (("cases.jsonl", 44), ("vision_cases.jsonl", 11)):
            rows = [json.loads(line) for line in (data_dir / name).read_text().splitlines() if line.strip()]
            self.assertEqual(len(rows), count)
        with Image.open(data_dir / "images/vision_fixture.png") as image:
            image.verify()

    def test_custom_checkout_has_no_bundled_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(upstream_info(Path(td)), {})

    def test_missing_and_lfs_placeholder_data_fail_before_requests(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            for name in ("k3_features", "params", "prompt_tokens"):
                (home / "tests" / name).mkdir(parents=True)
            python = official_python(DEFAULT_KVV_HOME)
            with self.assertRaisesRegex(ValueError, "数据缺失"):
                validate_runtime(home, python, ("prompt_tokens",))
            data = home / "testdata/prompt_token_cases"
            data.mkdir(parents=True)
            (data / "cases.jsonl").write_text("version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(ValueError, "Git LFS 指针"):
                validate_runtime(home, python, ("prompt_tokens",))

    def test_relative_output_and_all_skipped_suite(self):
        def fake_pytest(command, **kwargs):
            junit = Path(next(x.split("=", 1)[1] for x in command if x.startswith("--junitxml=")))
            self.assertTrue(junit.is_absolute())
            junit.write_text('<testsuite tests="26" failures="0" errors="0" skipped="26" time="0"/>')
            return subprocess.CompletedProcess(command, 0)

        env = {"VOLC_BASE_URL": "https://example.com/v1", "VOLC_MODEL": "kimi-k3", "VOLC_API_KEY": "offline"}
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as td, patch.dict(os.environ, env), \
                patch("kvv.runner.subprocess.run", side_effect=fake_pytest), patch("kvv.runner.finalize"):
            out, summary = execute("KVV_K3_THINKING_EFFORT", Path(td).relative_to(Path.cwd()))
            self.assertTrue(out.is_absolute())
            self.assertFalse(summary["admission_passed"])
            self.assertEqual(summary["skipped"], 26)
            self.assertEqual(summary["upstream"]["revision"], "66092cf444c97356c0e11c5078c67116390615d9")

    def test_existing_official_runtime(self):
        python=official_python(DEFAULT_KVV_HOME)
        validate_runtime(DEFAULT_KVV_HOME,python)
        self.assertTrue(python.is_file())

    def test_command_directly_runs_official_pytest_without_key_in_argv(self):
        env={"VOLC_BASE_URL":"https://example.com/v1","VOLC_MODEL":"model-x","VOLC_API_KEY":"secret"}
        with patch.dict(os.environ,env,clear=False):
            cmd=build_command(DEFAULT_KVV_HOME,official_python(DEFAULT_KVV_HOME),"k3_features_core",Path("junit.xml"),Path("tool.json"))
        self.assertEqual(cmd[0],str(DEFAULT_KVV_HOME/".venv/bin/python"))
        self.assertEqual(cmd[1:3],["-m","pytest"])
        self.assertIn("tests/k3_features/test_response_format.py",cmd)
        self.assertNotIn("tests/k3_features/test_dynamic_tools.py",cmd)
        self.assertNotIn("tests/k3_features/test_thinking_effort.py",cmd)
        self.assertNotIn("secret",cmd)

    def test_tool_schema_command_uses_official_parallel_and_report_options(self):
        env={"VOLC_BASE_URL":"https://example.com/v1","VOLC_MODEL":"model-x","VOLC_API_KEY":"secret"}
        with patch.dict(os.environ,env,clear=False):
            cmd=build_command(DEFAULT_KVV_HOME,official_python(DEFAULT_KVV_HOME),"tool_call_json_schema",Path("junit.xml"),Path("tool.json"))
        self.assertIn("-n",cmd); self.assertIn("--selection",cmd); self.assertIn("--thinking",cmd)
        self.assertTrue(any(x.startswith("--tool-json-report=") for x in cmd))

    def test_parse_junit(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"junit.xml"
            root=ET.Element("testsuites")
            suite=ET.SubElement(root,"testsuite",tests="3",failures="1",errors="0",skipped="1",time="1.5")
            ET.SubElement(suite,"testcase",name="ok")
            bad=ET.SubElement(suite,"testcase",name="bad");ET.SubElement(bad,"failure",message="no")
            ET.SubElement(suite,"testcase",name="skip")
            ET.ElementTree(root).write(path)
            result=parse_junit(path,"params")
        self.assertEqual((result["total"],result["passed"],result["failed"],result["skipped"]),(3,1,1,1))

if __name__=="__main__":unittest.main()


def test_manual_schema_cli_runs_without_channel_toggle():
    from kvv.runner import main
    with patch.dict(os.environ, {'VOLC_FACES': 'model,verify'}), \
            patch('kvv.runner.execute', return_value=(Path('/tmp/schema-test'), {'admission_passed': True})) as run:
        assert main(['KVV_TOOL_SCHEMA']) == 0
    assert run.call_args.args[0] == 'KVV_TOOL_SCHEMA'


def test_schema_execution_records_scoring_exclusion():
    def fake_pytest(command, **kwargs):
        junit = Path(next(x.split('=', 1)[1] for x in command if x.startswith('--junitxml=')))
        junit.write_text('<testsuite tests="408" failures="0" errors="0" skipped="0" time="0"/>')
        return subprocess.CompletedProcess(command, 0)
    env = {'VOLC_BASE_URL': 'https://example.test/v1', 'VOLC_MODEL': 'kimi-k3', 'VOLC_API_KEY': 'offline'}
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, env), \
            patch('kvv.runner.subprocess.run', side_effect=fake_pytest), patch('kvv.runner.finalize'):
        _, summary = execute('KVV_TOOL_SCHEMA', Path(directory))
    assert summary['scoring_excluded'] is True
    assert summary['passed'] == 408
    assert summary['executed_faces'] == ['channel']
