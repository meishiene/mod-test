#!/usr/bin/env python3
"""串行执行文档推荐节奏；涉及高额 token 消耗，必须显式传 --confirm-cost。"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "scripts" / "run_test.py"


def execute(case: str, extra: list[str] | None = None) -> int:
    command = [sys.executable, str(RUN), case, "--output-root", str(ROOT / "results"), *(extra or [])]
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-cost", action="store_true", help="确认整套测试约消耗3000万～3500万token")
    parser.add_argument("--include-l5", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()
    if not args.confirm_cost:
        parser.error("整套测试成本很高；确认凭据、限额和预算后传 --confirm-cost")
    cases = ["T0", "B1", "L1", "L2", "L3", "L4"]
    if args.include_l5: cases.append("L5")
    cases += ["O1", "C0", "C1"]
    failed, baselines = [], {}
    baseline_map = {"L1": "B1", "L2": "B1", "L3": "B1", "L4": "B1"}
    for case in cases:
        extra = []
        if case in baseline_map and baseline_map[case] in baselines:
            extra = ["--baseline-summary", str(baselines[baseline_map[case]])]
        code = execute(case, extra)
        matches = sorted((ROOT / "results").glob(f"{case}-*/summary.json"), key=lambda p: p.stat().st_mtime)
        if case == "B1" and matches:
            baselines[case] = matches[-1]
        if code != 0:
            failed.append(case)
            if args.stop_on_failure: break
        if case in {"L1", "L2", "L3", "L4"}: time.sleep(60)
        if case == "C0": time.sleep(120)
    print("失败/数据不足轮次:", ", ".join(failed) if failed else "无")
    return 2 if failed else 0


if __name__ == "__main__": raise SystemExit(main())
