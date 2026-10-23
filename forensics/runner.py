"""中转验真 CLI 装配。

Web 层通过环境变量注入配置（与其它套件一致），脚本只做参数/环境解析与收尾：

* ``FORENSICS_API_KEY`` / ``FORENSICS_BASE_URL`` / ``FORENSICS_MODEL`` / ``FORENSICS_PROTOCOL``
  （缺省回退 ``VOLC_API_KEY`` / ``VOLC_BASE_URL`` / ``VOLC_MODEL`` / ``VOLC_PROTOCOL``）
* ``VOLC_RECORD_ID``（Web 记录 ID，用于回填 summary.record_id）

退出码：0 = 用例裁定未发现信任级问题；2 = 不通过（含 blocked/warning 下调/failed）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from common.finalizer import finalize_result_dir
from volcperf.protocols import api_root

from .cases import CASES, FORENSICS_PROTOCOLS
from .core import ForensicsConfig, run

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def config_from_env() -> ForensicsConfig:
    api_key = (os.getenv("FORENSICS_API_KEY") or os.getenv("VOLC_API_KEY") or "").strip()
    base_url = (os.getenv("FORENSICS_BASE_URL") or os.getenv("VOLC_BASE_URL") or "").strip()
    model = (os.getenv("FORENSICS_MODEL") or os.getenv("VOLC_MODEL") or "").strip()
    protocol = (os.getenv("FORENSICS_PROTOCOL") or os.getenv("VOLC_PROTOCOL") or "openai").strip().lower()
    if not api_key or not base_url or not model:
        raise ValueError("缺少 FORENSICS_API_KEY / FORENSICS_BASE_URL / FORENSICS_MODEL（Web 启动时由界面配置自动注入）")
    if protocol not in FORENSICS_PROTOCOLS:
        raise ValueError(f"中转验真不支持协议 {protocol}（只支持文字面：{', '.join(FORENSICS_PROTOCOLS)}）")
    # Web 传进来的已是带版本前缀的根（或带自定义路径前缀）；api_root 幂等，重复传入不会叠前缀。
    base_url = api_root(protocol, base_url)
    try:
        timeout = float(os.getenv("FORENSICS_REQUEST_TIMEOUT_SECONDS", "120"))
    except ValueError as exc:
        raise ValueError("FORENSICS_REQUEST_TIMEOUT_SECONDS 必须是数字") from exc
    if timeout <= 0:
        raise ValueError("FORENSICS_REQUEST_TIMEOUT_SECONDS 必须大于 0")
    cutoff = (os.getenv("FORENSICS_MIN_CUTOFF") or "2024-01").strip()
    match = __import__("re").fullmatch(r"(\d{4})-(\d{1,2})", cutoff)
    if not match:
        raise ValueError("FORENSICS_MIN_CUTOFF 必须形如 YYYY-MM")
    return ForensicsConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        protocol=protocol,
        request_timeout_seconds=timeout,
        verify_ssl=(os.getenv("FORENSICS_VERIFY_SSL", "true").lower() not in _FALSY),
        trust_env=(os.getenv("VOLC_TRUST_ENV", "false").lower() in _TRUTHY),
        max_retries=int(os.getenv("FORENSICS_MAX_RETRIES", "2")),
        retry_backoff_seconds=float(os.getenv("FORENSICS_RETRY_BACKOFF_SECONDS", "1.0")),
        record_id=os.getenv("VOLC_RECORD_ID") or "",
        min_knowledge_cutoff=(int(match.group(1)), int(match.group(2))),
        ocr_digits=os.getenv("FORENSICS_OCR_DIGITS", ""),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="中转验真：隐式身份 / 图像理解 / 错误信封泄露")
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    try:
        config = config_from_env()
    except ValueError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    outcome = run(config, args.case, args.output_root)
    # 收尾可能下调 admission_passed（证据缺失等），退出码一律以收尾后的磁盘结论为准。
    summary = finalize_result_dir(outcome.result_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果目录: {outcome.result_dir.resolve()}")
    return 0 if summary.get("admission_passed") else 2


def cli() -> None:
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    cli()
