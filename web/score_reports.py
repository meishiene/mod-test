"""Read-only, versioned scores for new and historical records.

Recompute from archived evidence without rewriting summary, requests, or old score files.
File signatures invalidate the cache as running records finish writing their evidence.
"""
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

from common.scoring import score_batch, score_summary
from common.score_registry import SCORING_VERSION


def _signature(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = path.stat()
        return (str(path), stat.st_mtime_ns, stat.st_size) if path.is_file() else None
    except OSError:
        return None


@lru_cache(maxsize=512)
def _compute(signatures: tuple[tuple[str, int, int], ...], version: str) -> dict[str, Any] | None:
    for filename, _mtime, _size in signatures:
        try:
            payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if Path(filename).name == "score.json":
            reports = payload.get("scores") or [payload]
            if reports and all(isinstance(r, dict) and r.get("scoring_version") == version for r in reports):
                return payload
            continue
        if payload.get("record_type") == "batch" or payload.get("is_batch"):
            reports = score_batch(r for r in payload.get("results", []) if isinstance(r, dict))
            if reports:
                return reports[0] if len(reports) == 1 else {"scoring_version": version, "scores": reports}
        else:
            report = score_summary(payload)
            if report:
                return report
    return None


def current_score(row: Mapping[str, Any], root: Path, results: Path) -> dict[str, Any] | None:
    data = dict(row)
    folder = data.get("result_dir")
    candidates = [data.get("summary_path"), data.get("manifest_path")]
    if folder:
        candidates += [str(Path(folder) / "summary.json"), str(Path(folder) / "aggregation_manifest.json"),
                       str(Path(folder) / "score.json")]
    signatures = []
    seen = set()
    for filename in candidates:
        if not filename:
            continue
        path = (root / filename).resolve()
        if results.resolve() not in path.parents or path in seen:
            continue
        seen.add(path)
        signature = _signature(path)
        if signature:
            signatures.append(signature)
    return _compute(tuple(signatures), SCORING_VERSION) if signatures else None
