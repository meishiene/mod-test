from __future__ import annotations
import argparse, asyncio, json, os
from pathlib import Path
from common.suite_runtime import finalize
from common.faces import annotate_summary, parse_faces, selected_faces
from volcperf.protocols import api_root
from .cases import CASES
from .core import CASE_SPECS, QwenRunner, QwenConfig, summarize_results, write_report
from .faces import CASE_FACES
from common.model_validation import (DEFAULT_SEED, assess_validation, load_reference, apply_validation_gate,
                                     create_run_directory, validation_profile)

def main(argv=None):
    parser = argparse.ArgumentParser(description="运行千问（Qwen 3.6 / 3.7 / 3.8）专项测试")
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--faces", default=None,
                        help="执行面：model,verify（默认）/ channel / all；也可用 VOLC_FACES 注入")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="可复现客观题种子")
    parser.add_argument("--reference", type=Path, help="操作者确认可信的同型号综合 summary.json")
    args = parser.parse_args(argv)
    if args.reference and args.case != "QWEN_FULL":
        raise ValueError("--reference 仅支持 QWEN_FULL 综合校验")
    reference = load_reference(args.reference)
    key = os.getenv("QWEN_API_KEY") or os.getenv("VOLC_API_KEY")
    base = os.getenv("QWEN_BASE_URL") or os.getenv("VOLC_BASE_URL")
    model = os.getenv("QWEN_MODEL") or os.getenv("VOLC_MODEL")
    if not key or not base or not model:
        raise ValueError("缺少 QWEN_API_KEY/QWEN_BASE_URL/QWEN_MODEL")
    if args.case == "QWEN_FULL" and (not validation_profile(model) or validation_profile(model).family != "qwen"):
        raise ValueError("综合校验需要已登记的精确千问型号")
    # Web 派发时 base 已是 api_root（含 /compatible-mode/v1）；CLI 直连时可只给域名，
    # 这里按协议补一次（带路径的地址 api_root 原样返回，不会重复拼接）。
    base = api_root(os.getenv("VOLC_PROTOCOL") or "qwen", base)

    # 定向复测：QWEN_CASE_IDS="QWEN-A02,QWEN-V01" 只跑列出的用例（其余不探测、不落证据）。
    only_ids = {item.strip() for item in os.getenv("QWEN_CASE_IDS", "").split(",") if item.strip()}
    case_ids: set[str] | None = None
    if only_ids:
        known = {spec.case_id for spec in CASE_SPECS}
        unknown = sorted(only_ids - known)
        if unknown:
            print(json.dumps({"error": "unknown_case_ids", "unknown": unknown}, ensure_ascii=False))
            return 3
        in_suite = {spec.case_id for spec in CASE_SPECS
                    if spec.priority in CASES[args.case].priorities}
        if not (only_ids & in_suite):
            print(json.dumps({"error": "case_ids_not_in_suite", "case": args.case,
                              "requested": sorted(only_ids)}, ensure_ascii=False))
            return 3
        case_ids = only_ids

    # 执行面：默认只跑模型能力+验真；定向复测（QWEN_CASE_IDS）是显式点名，放开到全部面。
    faces = parse_faces("all") if case_ids else selected_faces(args.faces)

    out = create_run_directory(args.output_root, args.case)

    async def go():
        config = QwenConfig(key, base, seed=args.seed,
                            verify_ssl=os.getenv("QWEN_VERIFY_SSL", "true").lower() not in {"0", "false"})
        async with QwenRunner(config, audit_dir=out) as runner:
            results = await runner.run(model, CASES[args.case].priorities, case_ids, faces)
        summary = summarize_results(results, model)
        # 中转接入口径是千问的**唯一对外口径**（分组 domestic 这类中转会收窄能力面，
        # 官方严格口径会把「网关没实现」读成「模型不行」）。官方口径值仍留在 summary 里可追溯。
        relay = summary["relay_verdict"]
        summary.update({"record_id": os.getenv("VOLC_RECORD_ID"), "test_kind": "qwen",
                        "case": args.case, "model": model, "endpoint": base,
                        "case_filter": sorted(case_ids) if case_ids else None,
                        "verdict_contract": "relay",
                        "admission_passed": relay["admission_passed"],
                        "admission_basis": relay["summary_line"]})
        # 判据仍以 relay_verdict 为唯一权威（高级面失败＝relay_gap 不阻断），
        # 这里只补执行范围与面归属。
        annotate_summary(summary, results, CASE_FACES, faces, with_verdict=False)
        summary["model_validation"] = assess_validation(results, model, reference, seed=args.seed)
        if args.case in {"QWEN_CAPABILITY", "QWEN_VERIFY"}:
            expected = {s.case_id for s in CASE_SPECS if s.priority in CASES[args.case].priorities}
            passed = expected == {r.case_id for r in results if r.status == "passed" and r.passed}
            summary["admission_passed"] = relay["admission_passed"] = passed
            relay["summary_line"] = "本卡证据全部通过" if passed else "本卡存在失败或证据不足"
        apply_validation_gate(summary, args.case == "QWEN_FULL")
        summary["admission_basis"] = relay["summary_line"]
        write_report(out, summary)
        summary = finalize(out)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"结果目录: {out.resolve()}")
        return 0 if summary["admission_passed"] else 2

    return asyncio.run(go())


if __name__ == "__main__":
    raise SystemExit(main())
