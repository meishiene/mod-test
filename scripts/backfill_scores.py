"""给历史批次补算一致性评分（模型 100 + 通道 20，总分 120）。

新跑的批次在收尾时会自动写 `score.json` 并落库；这个脚本只处理**上线前**的老记录：

    .venv/bin/python scripts/backfill_scores.py            # 只补算，打印变化
    .venv/bin/python scripts/backfill_scores.py --write    # 写 score.json 并回填数据库两列

口径与线上完全一致（都走 `common.scoring.score_summary`），不另写一套算法。

`--write` 只新增 `score.json` 与数据库列，**不改动 summary.json / requests.jsonl 等证据文件**：
老批次的证据原样保留，评分随时可以按当时口径重算。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.scoring import headline, score_summary  # noqa: E402

RESULTS = ROOT / "results"
DB_PATH = ROOT / "web" / "data" / "runs.db"


def iter_summaries(limit: int | None = None):
    paths = sorted(RESULTS.glob("*/summary.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for index, path in enumerate(paths):
        if limit is not None and index >= limit:
            return
        yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="为历史结果补算一致性评分")
    parser.add_argument("--write", action="store_true", help="真正写盘（默认只试算）")
    parser.add_argument("--limit", type=int, default=None, help="只处理最近 N 个结果目录")
    parser.add_argument("--quiet", action="store_true", help="只打印有分数的记录")
    args = parser.parse_args()

    updated = 0
    scored = 0
    conn = sqlite3.connect(DB_PATH) if args.write and DB_PATH.is_file() else None
    if conn is not None:
        # Web 服务还没重启过时，迁移列可能不存在（脚本要能独立跑）。
        columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
        for name, kind in (("score", "REAL"), ("score_grade", "TEXT"), ("score_coverage", "REAL")):
            if name not in columns:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {name} {kind}")
    try:
        for path in iter_summaries(args.limit):
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            report = score_summary(summary)
            if not report:
                continue
            scored += 1
            # 批量多模型时 score_summary 返回列表：列表整体进 scores，DB 两列只认单模型。
            primary = report[0] if isinstance(report, list) and len(report) == 1 else (
                None if isinstance(report, list) else report)
            changed = not (path.parent / "score.json").is_file()
            if args.write:
                (path.parent / "score.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                if conn is not None:
                    conn.execute(
                        "UPDATE runs SET score=?, score_grade=?, score_coverage=? WHERE result_dir=?",
                        ((primary or {}).get("score"), (primary or {}).get("grade"),
                         (primary or {}).get("coverage"), str(path.parent.relative_to(ROOT))))
                updated += 1
            if not args.quiet or primary:
                marker = "写入" if args.write and changed else "变化" if changed else "不变"
                label = headline(primary) if primary else f"{len(report)} 个模型，各自评分"
                print(f"[{marker}] {path.parent.name:44s} {label}")
        if conn is not None:
            conn.commit()
    finally:
        if conn is not None:
            conn.close()
    action = "已写入" if args.write else "试算（加 --write 落盘）"
    print(f"\n扫描完成：{scored} 份可评分结果，{action}{updated if args.write else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
