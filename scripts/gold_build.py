#!/usr/bin/env python3
"""重建原厂金标（tokenizer + 原生契约），**不需要任何模型 API key**。

用法：

    .venv/bin/python scripts/gold_build.py
    .venv/bin/python scripts/gold_build.py --only qwen,deepseek,glm
    .venv/bin/python scripts/gold_build.py --selfcheck

产出：
    gold/data/tokenizer_gold.json   词表物证 + 固定探针 token 数 + 判别矩阵
    gold/data/lineage.json          每条物证的 repo / revision / sha256

第一次跑会从 HuggingFace 下载 tokenizer.json（约 2–12 MB/家），缓存在 ``gold/.cache/``。
国内网络可设 ``GOLD_HF_ENDPOINT=https://hf-mirror.com`` 换镜像（sha256 会照常校验比对）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gold.build import GOLD_PATH, LINEAGE_PATH, build  # noqa: E402
from gold.compare import selfcheck  # noqa: E402
from gold.registry import MODELS  # noqa: E402


def _print_summary(payload: dict) -> None:
    print(f"\n金标语料版本 {payload['corpus_version']}  ·  {payload['built_at']}")
    print(f"{'模型':24} {'状态':11} {'词表':>7} {'训练窗口':>10} {'映射':9} 判别")
    print("-" * 92)
    for e in payload["entries"]:
        tok = e.get("tokenizer") or {}
        contract = e.get("contract") or {}
        status = "ok" if e["status"] == "ok" else "uncollected"
        win = contract.get("native_train_context")
        proxied = e["mapping_confidence"] in ("proxy", "none")
        win_s = f"{win}{'*' if proxied else ''}" if win else "-"
        print(f"{e['slug']:24} {status:11} {str(tok.get('vocab_size') or '-'):>7} "
              f"{win_s:>10} {e['mapping_confidence']:9} "
              f"{'✓' if e['usable_for_discrimination'] else '—'}")
        if e["status"] != "ok":
            print(f"    ↳ 原因：{e.get('reason', '')[:100]}")
    if any(e["mapping_confidence"] in ("proxy", "none") for e in payload["entries"]):
        print("  * 该行的训练窗口来自同为该厂的**其他**开源权重，不能当作被测 API 型号的窗口。")
    disc = payload.get("discrimination", {})
    print(f"\n判别矩阵（共同探针不一致数，总 {disc.get('probe_total')} 条；0 = 纯 token 数分不开）")
    slugs = list((disc.get("pairs") or {}).keys())
    if slugs:
        width = max(len(s) for s in slugs) + 2
        print(" " * width + "".join(f"{s.split(':')[-1][:9]:>11}" for s in slugs))
        for s in slugs:
            row = (disc["pairs"].get(s) or {})
            print(f"{s:<{width}}" + "".join(
                f"{('·' if t == s else str(row.get(t, ''))):>11}" for t in slugs))
    for grp in disc.get("indistinguishable_groups", []):
        print(f"  ⚠ 纯词表无法区分：{' ≡ '.join(grp)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔的 vendor 或 slug，如 qwen,glm 或 glm:glm-4.6")
    ap.add_argument("--selfcheck", action="store_true", help="构建后跑无 key 判别自检")
    args = ap.parse_args()

    models = MODELS
    if args.only:
        wanted = {w.strip().lower() for w in args.only.split(",") if w.strip()}
        models = tuple(m for m in MODELS
                       if m.vendor.lower() in wanted or m.slug.lower() in wanted
                       or m.key.lower() in wanted)
        if not models:
            print(f"--only {args.only} 没匹配到任何注册条目", file=sys.stderr)
            return 2

    payload = build(models)
    _print_summary(payload)
    print(f"\n写入 {GOLD_PATH.relative_to(ROOT)}")
    print(f"写入 {LINEAGE_PATH.relative_to(ROOT)}")

    if args.selfcheck:
        res = selfcheck(json.loads(GOLD_PATH.read_text(encoding="utf-8")))
        print(f"\n自检：{res['passed']}/{res['total']} 通过（人工 offset 后仍能认回本族）")
        for slug, best in res["ties"]:
            print(f"  · {slug} 与 {best} 词表同族分不开（符合预期，二者不该进判别分母）")
        for row in res["failed"]:
            print(f"  ✗ {row['slug']} offset={row['offset']} → {row['verdict']} best={row['best']}")
        return 0 if res["passed"] == res["total"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
