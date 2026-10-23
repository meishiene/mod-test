"""调用项目内置的官方 Kimi Vendor Verifier pytest。

官方版本见 vendor_verifier/UPSTREAM.json；保留参数契约档位和
response_format 的已有 K3 兼容补丁，具体差异见 LOCAL_CHANGES.md。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cases import CASES, OPTIONAL_UNSCORED_CASES
from .faces import CASE_FACES
from common.faces import face_of, selected_faces
from common.suite_runtime import finalize
from common.report_style import stats_lines
from volcperf.protocols import api_root

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KVV_HOME = ROOT / "vendor_verifier"
SUITE_PATHS = {
    "k3_features_core": ("tests/k3_features/test_response_format.py", "tests/k3_features/test_tokenization_groundtruth.py", "tests/k3_features/test_tool_choice.py"),
    "k3_dynamic_tools": ("tests/k3_features/test_dynamic_tools.py",),
    "k3_thinking_effort": ("tests/k3_features/test_thinking_effort.py",),
    "prompt_tokens": ("tests/prompt_tokens",),
    "params": ("tests/params",),
    "tool_call_json_schema": ("tests/tool_call_json_schema",),
}


def kvv_home() -> Path:
    return Path(os.getenv("KVV_HOME", str(DEFAULT_KVV_HOME))).expanduser().resolve()


def official_python(home: Path) -> Path:
    override = os.getenv("KVV_PYTHON")
    if override:
        return Path(override).expanduser().resolve()
    # Windows 虚拟环境的解释器在 Scripts\python.exe，POSIX 在 bin/python。
    if os.name == "nt":
        return home / ".venv" / "Scripts" / "python.exe"
    return home / ".venv" / "bin" / "python"


def think_mode() -> str:
    value = os.getenv("KVV_THINK_MODE", "kimi")
    if value not in {"kimi", "opensource", "none"}:
        raise ValueError("KVV_THINK_MODE 仅支持 kimi、opensource、none")
    return value


def param_contract() -> str:
    """官方参数套件使用的契约档位（见 `vendor_verifier/tests/params/test_params.py`）。

    本模块是 K3 准入链路，默认按 **K3 契约** 判定（`temperature` 固定 1.0）；
    显式设 `KVV_PARAM_CONTRACT=k2` 可回到快照原始的 K2.5 期望。
    """
    value = os.getenv("KVV_PARAM_CONTRACT", "k3").strip().lower()
    if value not in {"k2", "k3"}:
        raise ValueError("KVV_PARAM_CONTRACT 仅支持 k2、k3")
    return value


def validate_runtime(home: Path, python: Path, suites: tuple[str, ...] | None = None) -> None:
    if not (home / "tests" / "k3_features").is_dir() or not (home / "tests" / "params").is_dir():
        raise ValueError(f"未找到官方 KVV 测试目录：{home}")
    if not python.is_file():
        raise ValueError(f"未找到官方 KVV Python：{python}")
    for suite in suites or ():
        for relative in SUITE_PATHS[suite]:
            if not (home / relative).exists():
                raise ValueError(f"官方 KVV 套件缺少文件：{home / relative}")
        if suite == "prompt_tokens":
            data = home / "testdata" / "prompt_token_cases"
            for relative in ("cases.jsonl", "vision_cases.jsonl", "images/vision_fixture.png"):
                path = data / relative
                if not path.is_file():
                    raise ValueError(f"官方 Prompt Token 数据缺失：{path}")
                with path.open("rb") as file:
                    if file.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
                        raise ValueError(f"官方 Prompt Token 数据仍为 Git LFS 指针，请下载实际数据：{path}")


def upstream_info(home: Path) -> dict[str, Any]:
    """Read provenance from the selected checkout, including custom KVV_HOME."""
    path = home / "UPSTREAM.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def build_command(home: Path, python: Path, suite: str, junit_path: Path, tool_report: Path) -> list[str]:
    base_url = api_root(os.getenv("VOLC_PROTOCOL", "volcengine"), os.environ["VOLC_BASE_URL"])
    command = [
        str(python), "-m", "pytest", *SUITE_PATHS[suite],
        "--base-url", base_url,
        "--smoke-model", os.environ["VOLC_MODEL"],
        "--think-mode", think_mode(),
        f"--junitxml={junit_path.resolve()}", "-ra", "-v",
    ]
    if suite == "tool_call_json_schema":
        command += ["-n", os.getenv("KVV_TOOL_WORKERS", "4"), "--selection", "all",
                    "--reruns", os.getenv("KVV_RERUNS", "3"), "--reruns-delay", "2",
                    f"--tool-json-report={tool_report.resolve()}"]
        if os.getenv("KVV_TOOL_THINKING", "true").lower() not in {"0", "false", "no"}:
            command.append("--thinking")
    return command


def parse_junit(path: Path, suite: str) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    nodes = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    total = sum(int(node.attrib.get("tests", 0)) for node in nodes)
    failures = sum(int(node.attrib.get("failures", 0)) for node in nodes)
    errors = sum(int(node.attrib.get("errors", 0)) for node in nodes)
    skipped = sum(int(node.attrib.get("skipped", 0)) for node in nodes)
    duration = sum(float(node.attrib.get("time", 0)) for node in nodes)
    failed_cases = []
    for case in root.iter("testcase"):
        problem = case.find("failure")
        if problem is None:
            problem = case.find("error")
        if problem is not None:
            failed_cases.append({"name": case.attrib.get("name"), "class": case.attrib.get("classname"), "message": (problem.attrib.get("message") or problem.text or "")[:1000]})
    return {"suite": suite, "total": total, "passed": total - failures - errors - skipped,
            "failed": failures, "errors": errors, "skipped": skipped, "duration_seconds": duration,
            "failed_cases": failed_cases}


def execute(case_id: str, output_root: Path) -> tuple[Path, dict[str, Any]]:
    for required in ("VOLC_API_KEY", "VOLC_BASE_URL", "VOLC_MODEL"):
        if not os.getenv(required):
            raise ValueError(f"缺少环境变量：{required}")
    meta = CASES[case_id]
    home = kvv_home(); python = official_python(home); validate_runtime(home, python, meta.suites)
    out = output_root.resolve() / f"{case_id}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({"KIMI_API_KEY": os.environ["VOLC_API_KEY"], "KIMI_BASE_URL": api_root(os.getenv("VOLC_PROTOCOL", "volcengine"), os.environ["VOLC_BASE_URL"]),
                "MODEL_NAME": os.environ["VOLC_MODEL"], "THINK_MODE": think_mode(),
                "KVV_PARAM_CONTRACT": param_contract()})
    suite_results=[]; commands=[]
    with (out / "pytest.log").open("w", encoding="utf-8", buffering=1) as log:
        for suite in meta.suites:
            junit = out / f"junit-{suite}.xml"; tool_report = out / "tool-call-schema-report.json"
            command = build_command(home, python, suite, junit, tool_report)
            commands.append([part if part != os.environ.get("VOLC_API_KEY") else "***" for part in command])
            log.write(f"\n===== official KVV: {suite} =====\n$ {' '.join(command)}\n")
            started=time.perf_counter()
            process=subprocess.run(command, cwd=home, env=env, stdout=log, stderr=subprocess.STDOUT)
            if junit.exists(): result=parse_junit(junit,suite)
            else: result={"suite":suite,"total":0,"passed":0,"failed":0,"errors":1,"skipped":0,"duration_seconds":time.perf_counter()-started,"failed_cases":[{"name":"pytest","message":f"未生成 JUnit XML，退出码 {process.returncode}"}]}
            result["exit_code"]=process.returncode; suite_results.append(result)
    total=sum(x["total"] for x in suite_results); passed=sum(x["passed"] for x in suite_results)
    failed=sum(x["failed"]+x["errors"] for x in suite_results); skipped=sum(x["skipped"] for x in suite_results)
    admission=bool(suite_results) and all(x["exit_code"]==0 and x["passed"]>0 and x["failed"]==0 and x["errors"]==0 for x in suite_results)
    summary={"record_id":os.getenv("VOLC_RECORD_ID"),"test_kind":"kvv","case":case_id,
             "official_kvv":True,"kvv_home":str(home),"model":os.environ["VOLC_MODEL"],"base_url":os.environ["VOLC_BASE_URL"],
             "upstream":upstream_info(home),
             "think_mode":think_mode(),"param_contract":param_contract(),"total":total,"passed":passed,"failed":failed,"skipped":skipped,
             "pass_rate":passed/total if total else 0.0,"admission_passed":admission,"suites":suite_results,
             "commands":commands,"generated_at":datetime.now(timezone.utc).isoformat()}
    if case_id in OPTIONAL_UNSCORED_CASES:
        summary.update(scoring_excluded=True, scoring_exclusion_reason="手动专项，仅保留测试结果，不参与模型评分",
                       executed_faces=[face_of(CASE_FACES, case_id)])
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    write_report(out,summary)
    finalize(out)
    return out,summary


def write_report(out: Path, summary: dict[str, Any]) -> None:
    lines=["# 官方 Kimi Vendor Verifier 测试报告","",
           *stats_lines(summary['total'],summary['passed'],summary['failed'],0,summary['skipped'],summary.get('pass_rate'),
                        [f"结论：{'通过' if summary['admission_passed'] else '未通过'}",
                         f"模型：`{summary['model']}`；Base URL：`{summary['base_url']}`；Thinking 格式：`{summary['think_mode']}`；参数契约：`{summary.get('param_contract','k2')}`",
                         f"官方工具目录：`{summary['kvv_home']}`",
                         f"官方快照提交：`{summary.get('upstream', {}).get('revision', '未记录（自定义套件）')}`",
                         "本地兼容补丁：" + ("、".join(summary.get('upstream', {}).get('local_patches', [])) or "未记录")]),
           "","## 官方套件结果","","| 官方 pytest 套件 | 总数 | 通过 | 失败 | 错误 | 跳过 | 耗时(s) |","|---|---:|---:|---:|---:|---:|---:|"]
    for item in summary["suites"]:
        lines.append(f"| `{' + '.join(SUITE_PATHS[item['suite']])}` | {item['total']} | {item['passed']} | {item['failed']} | {item['errors']} | {item['skipped']} | {item['duration_seconds']:.2f} |")
    failures=[(item["suite"],case) for item in summary["suites"] for case in item["failed_cases"]]
    if failures:
        lines += ["","## 失败用例",""]
        for suite,case in failures[:100]: lines.append(f"- `{suite}::{case.get('name')}`：{str(case.get('message','')).replace(chr(10),' ')[:500]}")
    lines += ["","## 产物","","- `pytest.log`：官方 pytest 完整输出","- `junit-*.xml`：官方 pytest JUnit 报告","- `tool-call-schema-report.json`：Tool JSON Schema 官方明细（执行该套件时生成）"]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description="直接执行官方 Kimi Vendor Verifier pytest 套件")
    parser.add_argument("case",choices=sorted(CASES));parser.add_argument("--output-root",type=Path,default=Path("results"));parser.add_argument("--faces",default=None,help="执行面：model,verify（默认）/ channel / all");args=parser.parse_args(argv)
    # 卡片级面过滤：KVV 走官方 pytest，用例是 pytest 节点而非 CASE_SPEC，只能按卡片拦。
    # 显式选中的不计分专项可以直接运行；其余卡片继续遵循执行面选择。
    faces = selected_faces(args.faces)
    face = face_of(CASE_FACES, args.case)
    if face not in faces and args.case not in OPTIONAL_UNSCORED_CASES:
        print(json.dumps({"error": "face_not_selected", "case": args.case, "face": face,
                          "executed_faces": list(faces),
                          "hint": "该用例属通道契约面，已默认停用；如需执行请设 --faces all 或 VOLC_FACES=all"},
                         ensure_ascii=False))
        return 3
    out,summary=execute(args.case,args.output_root);print(json.dumps(summary,ensure_ascii=False,indent=2));print(f"结果目录: {out.resolve()}")
    return 0 if summary["admission_passed"] else 2

if __name__=="__main__":
    try: raise SystemExit(main())
    except ValueError as exc: print(f"配置错误: {exc}",file=os.sys.stderr);raise SystemExit(2)
