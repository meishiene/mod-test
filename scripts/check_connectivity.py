#!/usr/bin/env python3
"""Send one small streaming request to verify NewAPI connectivity and compatibility."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from k3perf.client import stream_chat
from k3perf.config import ApiConfig


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="发送一次小型流式请求，检查 NewAPI 连通性和 OpenAI-compatible 响应"
    )
    parser.add_argument("--max-tokens", type=int, default=16, help="最大输出 token，默认 16")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选：保存完整检查结果；默认不落盘",
    )
    args = parser.parse_args()

    try:
        config = ApiConfig.from_env()
    except ValueError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        print("请设置 K3_API_KEY、K3_BASE_URL 和 K3_MODEL。", file=sys.stderr)
        return 1

    timeout = httpx.Timeout(config.request_timeout_seconds, connect=config.connect_timeout_seconds)
    limits = httpx.Limits(max_connections=2, max_keepalive_connections=2)
    prompt = (
        "这是一次 API 连通性检查。请只回复：connectivity ok。"
        "不要调用工具，不要输出其他内容。"
    )

    print("开始连通性检查")
    print(f"接口：{config.endpoint}")
    print(f"模型：{config.model}")
    print("请求：stream=true，max_tokens=%d" % args.max_tokens)

    started = time.time()
    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        verify=config.verify_ssl,
        headers={"Accept": "text/event-stream"},
        trust_env=config.trust_env,
    ) as session:
        row = await stream_chat(
            session=session,
            config=config,
            request_id=f"CONNECTIVITY-{uuid.uuid4().hex[:12]}",
            content=prompt,
            max_tokens=args.max_tokens,
            case_name="CONNECTIVITY",
            dataset_id="small-inline-prompt",
            phase="connectivity",
            counted=True,
        )

    report = {
        "checked_at_epoch": started,
        "endpoint": config.endpoint,
        "model": config.model,
        "http_status": row.get("http_status"),
        "stream_compatible": row.get("ttft_seconds") is not None,
        "usage_available": row.get("prompt_tokens") is not None and row.get("completion_tokens") is not None,
        "done_marker_received": row.get("error_type") != "missing_done_marker",
        "success": row.get("success"),
        "error_type": row.get("error_type"),
        "error_message": row.get("error_message"),
        "ttft_seconds": row.get("ttft_seconds"),
        "e2e_seconds": row.get("e2e_seconds"),
        "prompt_tokens": row.get("prompt_tokens"),
        "completion_tokens": row.get("completion_tokens"),
        "request_id": row.get("request_id"),
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"完整结果：{args.output.resolve()}")

    print("\n检查结果")
    print(f"HTTP 状态：{report['http_status']}")
    print(f"收到非空流式内容：{'是' if report['stream_compatible'] else '否'}")
    print(f"收到 usage：{'是' if report['usage_available'] else '否'}")
    print(f"TTFT：{report['ttft_seconds'] if report['ttft_seconds'] is not None else '-'} 秒")
    print(f"E2E：{report['e2e_seconds'] if report['e2e_seconds'] is not None else '-'} 秒")
    print(f"Prompt/Completion Token：{report['prompt_tokens']}/{report['completion_tokens']}")
    print(f"请求 ID：{report['request_id']}")

    if report["success"]:
        print("\n结论：连通成功，流式响应和 usage 满足当前压测脚本要求。")
        return 0

    if report["http_status"] is not None and 200 <= report["http_status"] < 300:
        print(f"\n结论：接口已连通，但兼容性检查未通过：{report['error_type']}。")
        if report["error_type"] == "missing_usage":
            print("NewAPI 已返回流式内容，但没有返回 usage；TPM、TPOT 和输入档位校准暂时无法正式统计。")
        return 2

    print(f"\n结论：连通失败：{report['error_type']}。")
    if report["error_message"]:
        print(f"错误：{report['error_message'][:500]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
