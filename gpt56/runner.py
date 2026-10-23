from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from common.suite_runtime import finalize
from common.faces import annotate_summary, selected_faces
from volcperf.protocols import api_root, protocol_spec

from .cases import CASES, PELICAN_CASE_IDS
from .core import (GPT56Config, GPT56Runner, PELICAN_MAX_CONCURRENCY, PELICAN_MAX_REQUESTS,
                   summarize_results, tier_for_model, write_report)
from .faces import CASE_FACES

FAMILIES = {"openai", "openai_responses"}


def resolve_protocol(value: str | None) -> str:
    """把界面传来的协议 id 收敛到本套件支持的族，并原样保留 id。

    FAMILIES 描述的是「线上形状」（openai / openai_responses），而协议 id 还带着各自的
    api root 前缀：openai=/v1、火山云=/api/v3、百度千帆=/v2。因此必须按族判断、并**保留原 id**——
    若像以前那样把不认识的 id 一律改写成 "openai"，千帆就会被拼成 .../v1 这个不存在的路径。
    """
    protocol = (value or "openai").lower()
    try:
        family = protocol_spec(protocol).family
    except ValueError:
        return "openai"
    return protocol if family in FAMILIES else "openai"


def config_from_env() -> GPT56Config:
    api_key = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("GPT_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise ValueError("缺少 GPT_API_KEY/GPT_BASE_URL（Web 启动时由界面配置自动注入）")
    # Web 传入的是 host-only 地址；按协议 id 的 api root 前缀（.../v1、/api/v3、/v2）规范化。
    protocol = resolve_protocol(os.getenv("GPT_PROTOCOL"))
    return GPT56Config(
        api_key=api_key,
        base_url=api_root(protocol, base_url),
        timeout=float(os.getenv("GPT_REQUEST_TIMEOUT_SECONDS", "300")),
        verify_ssl=os.getenv("GPT_VERIFY_SSL", "true").lower() not in {"0", "false", "no"},
        trust_env=os.getenv("GPT_TRUST_ENV", "false").lower() in {"1", "true", "yes"},
        # 瞬时故障（5xx / 连接错误）重试：默认 2 次、退避 1s/2s；置 0 可关闭（回归对照用）。
        max_retries=int(os.getenv("GPT_MAX_RETRIES", "2")),
        retry_backoff_seconds=float(os.getenv("GPT_RETRY_BACKOFF_SECONDS", "1.0")),
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


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 GPT 专项测试（GPT-5.6 / GPT-6.1）")
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    # 鹈鹕骑车 SVG（产物类用例）支持「一次跑多轮」：同一提示词重复 N 次，按并发数同时发出。
    # 其余用例的请求数是写死的，给了这两个参数只会被忽略——所以下面显式拦回，避免"以为调了"。
    parser.add_argument("--requests", type=int, default=None,
                        help=f"鹈鹕骑车 SVG 用例的请求轮数（1～{PELICAN_MAX_REQUESTS}，默认 1）")
    parser.add_argument("--concurrency", type=int, default=None,
                        help=f"鹈鹕骑车 SVG 用例的并发数（1～{PELICAN_MAX_CONCURRENCY}，默认 1）")
    parser.add_argument("--faces", default=None,
                        help="执行面：model,verify（默认）/ channel / all；也可用 VOLC_FACES 注入")
    args = parser.parse_args(argv)
    meta = CASES[args.case]
    rounds = 1 if args.requests is None else args.requests
    concurrency = 1 if args.concurrency is None else args.concurrency
    if not 1 <= rounds <= PELICAN_MAX_REQUESTS:
        raise ValueError(f"请求轮数必须是 1～{PELICAN_MAX_REQUESTS} 的整数（当前 {rounds}）")
    if not 1 <= concurrency <= PELICAN_MAX_CONCURRENCY:
        raise ValueError(f"并发数必须是 1～{PELICAN_MAX_CONCURRENCY} 的整数（当前 {concurrency}）")
    if (rounds != 1 or concurrency != 1) and args.case not in PELICAN_CASE_IDS:
        raise ValueError(f"请求轮数/并发数只对产物类用例（{'、'.join(PELICAN_CASE_IDS)}）生效，"
                         f"当前用例 {args.case} 的请求数固定")
    config = config_from_env()
    model = os.getenv("GPT_MODEL") or os.getenv("OPENAI_MODEL")
    if not model:
        raise ValueError("缺少 GPT_MODEL（Web 启动时由界面配置自动注入）")
    # 模型档位决定装配哪一组 CASE_SPECS：GPT-5.6 / GPT-6.1 / GPT-6 家族判定口径不同，不得串档。
    # 以 cases.py 的元数据为准（GPT61_* → gpt6.1，GPT6_* → gpt6），GPT_MODEL_VARIANT 仅作显式覆盖。
    variant = os.getenv("GPT_MODEL_VARIANT") or getattr(meta, "model_variant", "gpt5.6")
    # 串档拦截：模型画像与套件档位不一致时判据期望值必然错位（例如拿 6.1 套件跑 gpt-6-sol，
    # 会把"接受 none"当成身份失败），因此在入口直接拦回并指路正确套件。
    tier = tier_for_model(model)
    if tier is not None and tier.variant != variant:
        right = sorted(name for name, case in CASES.items()
                       if getattr(case, "model_variant", "gpt5.6") == tier.variant)
        raise ValueError(
            f"模型 {model} 属于 {tier.variant} 档位，与套件 {args.case}（{variant}）不匹配；"
            f"请改用 {right or ['（对应档位套件）']}，或设 GPT_MODEL_VARIANT 显式覆盖"
        )
    # 网关未开放该档位通道（如 gpt-6-luna）时由 runner 整批 blocked，不在此拦回——
    # 需要让批次留下"通道不存在"的可审计证据，而不是静默跳过。
    # 目录必须先于请求创建：runner 会把每个请求的证据实时写入该目录下的 requests.jsonl。
    out = args.output_root / f"{args.case}-{time.strftime('%Y%m%d-%H%M%S')}"
    faces = selected_faces(args.faces)
    async with GPT56Runner(config, audit_dir=out) as runner:
        runner.pelican_rounds = rounds
        runner.pelican_concurrency = concurrency
        results = await runner.run((model,), meta.priorities, variant=variant, faces=faces)
    summary = summarize_results(results)
    summary.update({
        "record_id": os.getenv("VOLC_RECORD_ID"),
        "test_kind": "gpt",
        "case": args.case,
        "model": model,
        "model_variant": variant,
        "model_tier": tier.model_id if tier else None,
        "endpoint": config.base_url,
        "admission_passed": summary["all_passed"],
    })
    # 面判定：模型/验真面失败才阻断；通道面（内置工具/beta/端点/校验契约）失败只记缺口。
    annotate_summary(summary, results, CASE_FACES, faces)
    summary["admission_passed"] = summary["face_verdict"]["admission_passed"]
    if args.case in PELICAN_CASE_IDS:
        # 轮数/并发进摘要：批次结论必须能自证"这是在什么规模下跑出来的"。
        summary["pelican_rounds"] = rounds
        summary["pelican_concurrency"] = min(concurrency, rounds) if rounds else concurrency
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
