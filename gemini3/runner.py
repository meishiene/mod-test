from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from common.suite_runtime import finalize
from common.faces import annotate_summary, selected_faces
from volcperf.protocols import api_root

from .cases import CASES
from .core import MODELS, Gemini3Config, Gemini3Runner, summarize_results, tier_for_model, write_report
from .faces import CASE_FACES

# Gemini 文本面只走原生 GenerateContent（gemini 家族根 {base}/v1beta）。
FAMILIES = {"gemini"}
# 明确不属于本套件的模型家族前缀：命中即拦回，避免把别的厂商模型派到 Gemini 面。
_FOREIGN_PREFIXES = ("gpt-", "o1", "o3", "o4", "claude-", "kimi-", "moonshot", "deepseek-", "glm-", "agnes")


def config_from_env() -> Gemini3Config:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL") or os.getenv("GEMINI_BASE")
    if not api_key or not base_url:
        raise ValueError("缺少 GEMINI_API_KEY/GEMINI_BASE_URL（Web 启动时由界面配置自动注入）")
    # Web 传入的是 host-only 地址；gemini 家族按 api_root() 规范化成 ".../v1beta"。
    protocol = os.getenv("GEMINI_PROTOCOL", "gemini")
    if protocol not in FAMILIES:
        protocol = "gemini"
    return Gemini3Config(
        api_key=api_key,
        base_url=api_root(protocol, base_url),
        timeout=float(os.getenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "300")),
        verify_ssl=os.getenv("GEMINI_VERIFY_SSL", "true").lower() not in {"0", "false", "no"},
        trust_env=os.getenv("GEMINI_TRUST_ENV", "false").lower() in {"1", "true", "yes"},
        # 瞬时故障（5xx / 连接错误）重试：默认 2 次、退避 1s/2s；置 0 可关闭（回归对照用）。
        max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "2")),
        retry_backoff_seconds=float(os.getenv("GEMINI_RETRY_BACKOFF_SECONDS", "1.0")),
    )


def read_finalized_summary(result_dir: Path) -> dict[str, object]:
    """读回收尾后的 summary.json 作为权威结论。

    ``finalize()``（common.finalizer.finalize_result_dir）会把统一证据/裁定校验
    写回该文件，并可能**下调** ``admission_passed``——典型是"整批只有 blocked/skipped、
    没有任何有效结论"的批次：收尾前内存里的 summary 由 ``all_passed``（仅 failed+error）
    推出 True，收尾后因 ``case_verdicts_complete`` 校验（无有效结论）变为 False。
    注意 blocked 本身**不阻断准入**（2026-10-10 口径），只有整批都拿不到有效结论时才拦。
    因此退出码与打印都必须用这份收尾后的结论，否则 CLI/CI 会把不准入的批次当成 0。
    """
    return json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))


def guard_model(model: str) -> str | None:
    """跨套件误派发拦回：返回错误说明；None 表示可用。

    档位相关期望值按被测模型解析，因此"网关自定义 ID"是被允许的（回落并标记 assumed）；
    但把 Claude / GPT / Kimi 等**别的厂商模型**派到 Gemini 面则一定是选型错误，直接拦回。
    """
    lowered = (model or "").strip().lower()
    if not lowered:
        return None
    for prefix in _FOREIGN_PREFIXES:
        if lowered.startswith(prefix):
            return (f"模型 {model!r} 不属于 Gemini 家族，与套件 {sorted(CASES)} 不匹配；"
                    f"请改用 Gemini 3.x 模型（如 {', '.join(MODELS)}）或对应的专项套件")
    return None


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 Gemini 3.x 专项测试")
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--faces", default=None,
                        help="执行面：model,verify（默认）/ channel / all；也可用 VOLC_FACES 注入")
    args = parser.parse_args(argv)
    config = config_from_env()
    model = os.getenv("GEMINI_MODEL")
    if not model:
        raise ValueError("缺少 GEMINI_MODEL（Web 启动时由界面配置自动注入）")
    error = guard_model(model)
    if error:
        raise ValueError(error)
    meta = CASES[args.case]
    variant = os.getenv("GEMINI_MODEL_VARIANT") or getattr(meta, "model_variant", "gemini3")
    # 目录必须先于请求创建：runner 会把每个请求的证据实时写入该目录下的 requests.jsonl。
    out = args.output_root / f"{args.case}-{time.strftime('%Y%m%d-%H%M%S')}"
    faces = selected_faces(args.faces)
    async with Gemini3Runner(config, audit_dir=out) as runner:
        results = await runner.run((model,), meta.priorities, variant=variant, faces=faces)
    summary = summarize_results(results)
    tier = tier_for_model(model)
    summary.update({
        "record_id": os.getenv("VOLC_RECORD_ID"),
        "test_kind": "gemini",
        "case": args.case,
        "model": model,
        "model_variant": variant,
        "model_tier": tier.model_id if tier else None,
        "endpoint": config.base_url,
        "admission_passed": summary["all_passed"],
    })
    # 面判定：模型/验真面失败才阻断；通道面（内置工具/档位矩阵/端点契约）失败只记缺口。
    annotate_summary(summary, results, CASE_FACES, faces)
    summary["admission_passed"] = summary["face_verdict"]["admission_passed"]
    write_report(out, summary)
    # write_report 已写入结果；补写 Web 聚合需要的扩展字段。
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finalize(out)
    # 收尾可能下调 admission_passed（如"整批只有 blocked"，或证据缺失），
    # 退出码和打印一律以收尾后的磁盘结论为准。
    summary = read_finalized_summary(out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果目录: {out.resolve()}")
    return 0 if summary.get("admission_passed") else 2


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(argv))
