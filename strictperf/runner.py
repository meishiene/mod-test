"""执行项目内置的严格压测脚本，不依赖外部 k3test 目录。"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .cases import CASES
from common.suite_runtime import finalize

DEFAULT_HOME = Path(__file__).resolve().parent.parent


def k3test_home() -> Path:
    return Path(os.getenv("K3TEST_HOME", str(DEFAULT_HOME))).expanduser().resolve()


def k3test_python(home: Path) -> Path:
    override = os.getenv("K3TEST_PYTHON")
    return Path(override).expanduser().resolve() if override else Path(sys.executable).resolve()


def validate_runtime(home: Path, python: Path, upstream_case: str) -> Path:
    script = home / "scripts" / f"run_{upstream_case}.py"
    if not script.is_file():
        raise ValueError(f"未找到 k3test 严格测试脚本：{script}")
    if not python.is_file():
        raise ValueError(f"未找到 k3test Python：{python}")
    return script


def build_env() -> dict[str, str]:
    missing = [name for name in ("VOLC_API_KEY", "VOLC_BASE_URL", "VOLC_MODEL") if not os.getenv(name)]
    if missing:
        raise ValueError(f"缺少环境变量：{', '.join(missing)}")
    env = os.environ.copy()
    env.update({
        "K3_CUSTOM_SAMPLES": os.getenv("VOLC_CUSTOM_REQUESTS", ""),
        "K3_CUSTOM_CONCURRENCY": os.getenv("VOLC_CUSTOM_CONCURRENCY", ""),
        "K3_API_KEY": os.environ["VOLC_API_KEY"],
        "K3_BASE_URL": os.environ["VOLC_BASE_URL"],
        "K3_MODEL": os.environ["VOLC_MODEL"],
    })
    # 可按部署需要覆盖，默认使用项目自身的 /chat/completions。
    if os.getenv("VOLC_CHAT_PATH"):
        env["K3_CHAT_PATH"] = os.environ["VOLC_CHAT_PATH"]
    return env


def newest_result(root: Path, upstream_case: str, since: float) -> Path | None:
    candidates = [p for p in root.glob(f"{upstream_case}-*") if p.is_dir() and p.stat().st_mtime >= since - 2]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def execute(case_id: str, output_root: Path) -> tuple[Path | None, dict, int]:
    meta = CASES[case_id]
    home = k3test_home()
    python = k3test_python(home)
    script = validate_runtime(home, python, meta.upstream_case)
    env = build_env()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    command = [str(python), str(script), "--output-root", str(output_root.resolve())]
    process = subprocess.run(command, cwd=home, env=env)
    source = newest_result(output_root, meta.upstream_case, started)
    if source is None:
        return None, {}, process.returncode
    suffix = source.name[len(meta.upstream_case):]
    target = output_root / f"{case_id}{suffix}"
    if target.exists():
        target = output_root / f"{case_id}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    shutil.move(str(source), str(target))
    summary_path = target / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    if summary:
        summary.update({
            "case": case_id,
            "upstream_case": meta.upstream_case,
            "strict_official_method": True,
            "k3test_home": str(home),
            "embedded_runtime": True,
        })
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if summary and target:
        finalize(target)
    return target, summary, process.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="直接执行 k3test 严格多轮压测")
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    out, summary, returncode = execute(args.case, args.output_root)
    if out:
        print(f"严格测试结果目录: {out.resolve()}")
    if not summary:
        print(f"k3test 未生成 summary.json，退出码 {returncode}", file=os.sys.stderr)
        return returncode or 2
    return 0 if summary.get("admission_passed") else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"配置错误: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
