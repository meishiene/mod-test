"""对话窗口 CLI 装配。

Web 层通过环境变量注入配置（与其它套件一致），脚本只做参数/环境解析与收尾：

* ``CHAT_API_KEY`` / ``CHAT_BASE_URL`` / ``CHAT_MODEL`` / ``CHAT_PROTOCOL``
* ``CHAT_PROMPT``（必填，用户消息原文）
* ``CHAT_SYSTEM``（可选，系统提示词）
* ``CHAT_MAX_TOKENS``（可选，默认 8192）
* ``CHAT_TEMPERATURE``（可选，留空则不下发该参数）
* ``VOLC_RECORD_ID``（Web 记录 ID，用于回填 summary.record_id）

退出码：0 = 本次对话拿到正文；2 = 未通过（含 blocked/failed/warning）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from common.finalizer import finalize_result_dir
from volcperf.protocols import api_root

from .cases import (
    CASES,
    CHAT_PROTOCOLS,
    DEFAULT_MAX_TOKENS,
    MAX_MAX_TOKENS,
    MAX_PROMPT_CHARS,
    MAX_SYSTEM_CHARS,
)
from .core import ChatConfig, run_chat

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _float_or_none(raw: str | None) -> float | None:
    value = (raw or "").strip()
    if not value:
        return None
    return float(value)


def config_from_env() -> ChatConfig:
    api_key = (os.getenv("CHAT_API_KEY") or os.getenv("VOLC_API_KEY") or "").strip()
    base_url = (os.getenv("CHAT_BASE_URL") or os.getenv("VOLC_BASE_URL") or "").strip()
    model = (os.getenv("CHAT_MODEL") or os.getenv("VOLC_MODEL") or "").strip()
    protocol = (os.getenv("CHAT_PROTOCOL") or os.getenv("VOLC_PROTOCOL") or "openai").strip().lower()
    if not api_key or not base_url or not model:
        raise ValueError("缺少 CHAT_API_KEY / CHAT_BASE_URL / CHAT_MODEL（Web 启动时由界面配置自动注入）")
    if protocol not in CHAT_PROTOCOLS:
        raise ValueError(f"对话窗口不支持协议 {protocol}（只支持文字面：{', '.join(CHAT_PROTOCOLS)}）")
    # Web 传进来的是 host-only 地址；这里按协议家族规范化成带版本前缀的根，
    # 与 gemini_image / gemini3 等套件的做法一致（api_root 幂等，重复传入也不会叠前缀）。
    base_url = api_root(protocol, base_url)
    prompt = os.getenv("CHAT_PROMPT") or ""
    if not prompt.strip():
        raise ValueError("用户消息不能为空")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"用户消息过长（{len(prompt)} 字符，上限 {MAX_PROMPT_CHARS}）")
    system = os.getenv("CHAT_SYSTEM") or ""
    if len(system) > MAX_SYSTEM_CHARS:
        raise ValueError(f"系统提示词过长（{len(system)} 字符，上限 {MAX_SYSTEM_CHARS}）")
    try:
        max_tokens = int(os.getenv("CHAT_MAX_TOKENS") or DEFAULT_MAX_TOKENS)
    except ValueError as exc:
        raise ValueError("CHAT_MAX_TOKENS 必须是整数") from exc
    if not 1 <= max_tokens <= MAX_MAX_TOKENS:
        raise ValueError(f"CHAT_MAX_TOKENS 必须在 1～{MAX_MAX_TOKENS} 之间")
    try:
        temperature = _float_or_none(os.getenv("CHAT_TEMPERATURE"))
    except ValueError as exc:
        raise ValueError("CHAT_TEMPERATURE 必须是数字") from exc
    if temperature is not None and not 0 <= temperature <= 2:
        raise ValueError("CHAT_TEMPERATURE 必须在 0～2 之间")
    try:
        timeout = float(os.getenv("CHAT_REQUEST_TIMEOUT_SECONDS", "300"))
    except ValueError as exc:
        raise ValueError("CHAT_REQUEST_TIMEOUT_SECONDS 必须是数字") from exc
    if timeout <= 0:
        raise ValueError("CHAT_REQUEST_TIMEOUT_SECONDS 必须大于 0")
    return ChatConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        protocol=protocol,
        prompt=prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        request_timeout_seconds=timeout,
        verify_ssl=(os.getenv("CHAT_VERIFY_SSL", "true").lower() not in _FALSY),
        # 是否走环境代理与压测共用同一开关，避免"连通性通了、正式请求却绕远"。
        trust_env=(os.getenv("VOLC_TRUST_ENV", "false").lower() in _TRUTHY),
        max_retries=int(os.getenv("CHAT_MAX_RETRIES", "2")),
        retry_backoff_seconds=float(os.getenv("CHAT_RETRY_BACKOFF_SECONDS", "1.0")),
        record_id=os.getenv("VOLC_RECORD_ID") or "",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="对话窗口：发送一次自由对话并归档完整响应")
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--system", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args(argv)
    try:
        config = config_from_env()
    except ValueError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    overrides = {
        "max_tokens": args.max_tokens,
        "system": args.system,
        "temperature": args.temperature,
    }
    if any(value is not None for value in overrides.values()):
        from dataclasses import replace
        config = replace(config, **{key: value for key, value in overrides.items() if value is not None})
    outcome = run_chat(config, args.output_root)
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
