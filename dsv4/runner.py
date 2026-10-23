from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from common.suite_runtime import finalize
from common.model_contracts import model_matches_family
from common.faces import annotate_summary, parse_faces, selected_faces
from volcperf.protocols import api_root

from .cases import CASES
from .core import CASE_SPECS, DeepSeekConfig, DeepSeekV4Runner, summarize_results, write_report
from .faces import CASE_FACES


def config_from_env() -> DeepSeekConfig:
    api_key = os.getenv("DS_API_KEY") or os.getenv("VOLC_API_KEY")
    base_url = os.getenv("DS_BASE_URL") or os.getenv("VOLC_BASE_URL")
    if not api_key or not base_url:
        raise ValueError("缺少 DS_API_KEY/DS_BASE_URL（Web 启动时由界面配置自动注入）")
    return DeepSeekConfig(
        api_key=api_key,
        base_url=api_root(os.getenv("VOLC_PROTOCOL", "volcengine"), base_url),
        beta_base_url=os.getenv("DS_BETA_BASE_URL"),
        anthropic_base_url=os.getenv("DS_ANTHROPIC_BASE_URL"),
        timeout=float(os.getenv("DS_REQUEST_TIMEOUT_SECONDS", "300")),
        verify_ssl=os.getenv("DS_VERIFY_SSL", "true").lower() not in {"0", "false", "no"},
        trust_env=os.getenv("DS_TRUST_ENV", "false").lower() in {"1", "true", "yes"},
    )


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 DeepSeek V4 专项测试")
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--faces", default=None,
                        help="执行面：model,verify（默认）/ channel / all；也可用 VOLC_FACES 注入")
    args = parser.parse_args(argv)
    config = config_from_env()
    model = os.getenv("DS_MODEL") or os.getenv("VOLC_MODEL")
    if not model:
        raise ValueError("缺少 DS_MODEL（Web 启动时由界面配置自动注入）")
    # 模型门禁：误配其他家族的模型（如 gpt-*）会让正向用例被上游 400 拒绝，
    # 而负向用例的 4xx 断言恰被该 400 满足产生假通过。发请求前直接拦截。
    allowed, gate_reason = model_matches_family("deepseek", model)
    if not allowed:
        print(json.dumps({"error": "model_gate_rejected", "message": gate_reason, "model": model, "case": args.case}, ensure_ascii=False))
        return 3
    meta = CASES[args.case]
    # 定向复测：DS_CASE_IDS="DS-V4-19,DS-V4-41" 只跑列出的用例（其余不探测、不落证据）。
    only_ids = {c.strip() for c in os.getenv("DS_CASE_IDS", "").split(",") if c.strip()}
    priorities = meta.priorities
    case_ids = None
    if only_ids:
        known = {spec.case_id for spec in CASE_SPECS}
        unknown = sorted(only_ids - known)
        if unknown:
            print(json.dumps({"error": "unknown_case_ids", "unknown": unknown}, ensure_ascii=False))
            return 3
        in_suite = {spec.case_id for spec in CASE_SPECS if spec.priority in priorities}
        if not (only_ids & in_suite):
            print(json.dumps({"error": "case_ids_not_in_suite", "case": args.case,
                              "requested": sorted(only_ids)}, ensure_ascii=False))
            return 3
        case_ids = only_ids
    # 执行面：默认只跑模型能力+验真。定向复测（DS_CASE_IDS）是用户显式点名，
    # 一律放开到全部面，否则点了一条通道面用例会被过滤成空跑。
    faces = parse_faces("all") if case_ids else selected_faces(args.faces)
    # 目录必须先于请求创建：runner 会把每个请求的证据实时写入该目录下的 requests.jsonl。
    # 同一秒内的并发批次会撞名（多模型并行复测实测踩过），撞名时加序号避免证据互相覆盖。
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.output_root / f"{args.case}-{stamp}"
    suffix = 1
    while out.exists():
        suffix += 1
        out = args.output_root / f"{args.case}-{stamp}-{suffix}"
    async with DeepSeekV4Runner(config, audit_dir=out) as runner:
        results = await runner.run((model,), priorities, case_ids, faces)
    summary = summarize_results(results)
    summary.update({
        "record_id": os.getenv("VOLC_RECORD_ID"),
        "test_kind": "deepseek",
        "case": args.case,
        "model": model,
        "endpoint": config.base_url,
        "case_filter": sorted(case_ids) if case_ids else None,
        "admission_passed": summary["all_passed"],
    })
    # 判据仍以 relay_verdict 为唯一权威（通道面失败＝relay_gap 不阻断），
    # 这里只补执行范围与每条用例的面归属，供报告分组。
    annotate_summary(summary, results, CASE_FACES, faces, with_verdict=False)
    write_report(out, summary)
    # write_report 已写入结果；补写 Web 聚合需要的扩展字段。
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finalize(out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果目录: {out.resolve()}")
    return 0 if summary["admission_passed"] else 2


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(argv))


if __name__ == "__main__":  # `python -m dsv4.runner DS_V4_P0` 之前静默什么都不做，补上入口
    raise SystemExit(main())
