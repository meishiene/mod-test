"""把「中转侧实测的 token 数」和「原厂金标」对齐，做同族判别。

为什么不能直接比绝对值
----------------------
中转普遍会在前面塞一段固定 system / 模板，``usage.prompt_tokens`` 因此整段偏移一个**常数**。
绝对值一比就全错。两个办法消掉它：

1. **两点差分（首选）**：``gold/probes.PAIRS`` 是「正文」与「填充 + 正文」成对样本，
   两块相减后常数项自己抵消，剩下的是纯切分差异。这条**不依赖中转透传原始 usage**，
   只要它按实际收到的字符串计数就有效。
2. **中位数定心**：拿所有探针残差的中位数当 offset，看残差是否恒定。
   残差恒定 ⇒ 词表同族；残差**不恒定**（散布大）⇒ 要么是别家词表，要么中转在按内容
   动态改写前缀（后者本身就是可疑证据，但**不能**据此判掉包）。

判定口径（与平台一致：宁可说"无法判定"也不给假结论）
----------------------------------------------------
- ``matched``        声明家族 == 最优匹配，且残差全 0（或 ≤ tol）；
- ``family_mismatch`` 最优匹配是**另一家**，且差距显著（≥2 个探针不一致）；
- ``inconsistent``   残差不恒定 / 最优与次优打平 ⇒ 无法归因，只记录；
- ``no_gold``        中转模型名在注册表里没有可用金标 ⇒ 必须显式说"没得判"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from .probes import PAIRS

DEFAULT_TOL = 0


@dataclass
class Candidate:
    slug: str
    vendor: str
    score: int          # 越小越好：定心后残差绝对值的最大值
    offset: int         # 估计的固定偏移
    spread: int         # 定心后残差的极差（0 = 完全恒定）
    mismatched: list[str] = field(default_factory=list)
    pair_score: int | None = None  # 两点差分得分（仅当实测含 PAIRS 时）

    def as_dict(self) -> dict:
        return {
            "slug": self.slug,
            "vendor": self.vendor,
            "score": self.score,
            "offset": self.offset,
            "spread": self.spread,
            "mismatched": self.mismatched,
            "pair_score": self.pair_score,
        }


@dataclass
class CompareResult:
    verdict: str
    declared_slug: str | None
    best: Candidate | None
    runner_up: Candidate | None
    candidates: list[Candidate]
    notes: list[str] = field(default_factory=list)
    measured: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "declared_slug": self.declared_slug,
            "best": self.best.as_dict() if self.best else None,
            "runner_up": self.runner_up.as_dict() if self.runner_up else None,
            "candidates": [c.as_dict() for c in self.candidates],
            "notes": self.notes,
            "measured": self.measured,
        }


def _pair_deltas(counts: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for short_id, long_id in PAIRS:
        if short_id in counts and long_id in counts:
            out[short_id] = counts[long_id] - counts[short_id]
    return out


def compare(measured: dict[str, int], gold: dict, tol: int = DEFAULT_TOL,
            declared_slug: str | None = None) -> CompareResult:
    """``gold`` 为 ``gold/data/tokenizer_gold.json`` 的解析结果。"""
    measured_deltas = _pair_deltas(measured)
    candidates: list[Candidate] = []
    notes: list[str] = []

    # 「没采到数据」与「金标里没有这家」是两件事，必须分开报，否则会被读成"没得判模型"。
    if len(measured) < 3:
        return CompareResult("unmeasured", declared_slug, None, None, [],
                             [f"只拿到 {len(measured)} 条探针的 token 数，不足以判别"
                              "（检查 key / 模型名 / 是否返回 usage）"], measured)
    for entry in gold.get("entries", []):
        tok = entry.get("tokenizer") or {}
        counts = tok.get("probe_counts") or {}
        common = sorted(set(counts) & set(measured))
        if len(common) < 3:
            continue  # 共同探针太少，任何结论都是噪声
        residuals = [measured[k] - counts[k] for k in common]
        offset = int(median(residuals))
        centered = [r - offset for r in residuals]
        pair_score = None
        gold_deltas = _pair_deltas(counts)
        if measured_deltas and gold_deltas:
            shared = sorted(set(measured_deltas) & set(gold_deltas))
            if shared:
                pair_score = sum(abs(measured_deltas[k] - gold_deltas[k]) for k in shared)
        candidates.append(Candidate(
            slug=entry["slug"],
            vendor=entry["vendor"],
            score=max(abs(c) for c in centered),
            offset=offset,
            spread=max(centered) - min(centered),
            mismatched=[k for k, c in zip(common, centered) if abs(c) > tol],
            pair_score=pair_score,
        ))

    if not candidates:
        return CompareResult("no_gold", declared_slug, None, None, [], 
                             ["金标里没有可与本次实测对齐的条目（共同探针不足 3 条）"], measured)

    # 排序：优先用两点差分（不受 offset 假设影响），其次定心残差
    candidates.sort(key=lambda c: (c.pair_score if c.pair_score is not None else 10**9, c.score))
    best, runner = candidates[0], (candidates[1] if len(candidates) > 1 else None)

    if best.pair_score is not None:
        notes.append("判别主用两点差分（填充段相减），不依赖中转透传原始 usage。")

    # 打平 / 残差不恒定 ⇒ 不下结论
    if runner and best.score == runner.score == 0 and best.vendor != runner.vendor:
        return CompareResult("inconsistent", declared_slug, best, runner, candidates,
                             notes + [f"{best.slug} 与 {runner.slug} 同时完全匹配，无法归因"], measured)

    if declared_slug and best.slug != declared_slug:
        declared_entry = next((e for e in gold.get("entries", []) if e["slug"] == declared_slug), None)
        declared_cand = next((c for c in candidates if c.slug == declared_slug), None)
        gap = (declared_cand.score - best.score) if declared_cand else None
        if gap is not None and gap >= 2:
            return CompareResult("family_mismatch", declared_slug, best, declared_cand, candidates,
                                 notes + [f"声明 {declared_slug}，但最像 {best.slug}（残差差 {gap} 个 token）"],
                                 measured)
        return CompareResult("inconsistent", declared_slug, best, declared_cand, candidates,
                             notes + ["最优匹配与声明不一致但差距不足 2 个 token，可能是版本漂移"], measured)

    if best.score == 0 or best.score <= tol:
        if best.spread == 0:
            return CompareResult("matched", declared_slug or best.slug, best, runner, candidates,
                                 notes, measured)
        return CompareResult("inconsistent", declared_slug, best, runner, candidates,
                             notes + [f"最优匹配但残差散布 {best.spread}，疑动态改写前缀"], measured)

    return CompareResult("inconsistent", declared_slug, best, runner, candidates,
                         notes + [f"所有金标都有偏差，最优 {best.slug} 也有 {best.score} 个 token 不符"],
                         measured)


def selfcheck(gold: dict, offsets: tuple[int, ...] = (0, 7, 23, -11)) -> dict:
    """无 key 自检：给每条金标加**人工 offset**，看判别器能否把它认回自己。

    这是方法有效性的唯一诚实证明方式——它证明的是"判别器能消掉常数偏移并认清同族"，
    **不证明**"某家中转就是原厂"。两者别混。

    只对 ``usable_for_discrimination`` 的条目期待"认回自己"。参照条目（上一代词表）
    本来就和后继同族，正确行为是 ``inconsistent``（分不开），不该算失败——把它们单列。
    """
    rows = []
    ties = []
    for entry in gold.get("entries", []):
        counts = (entry.get("tokenizer") or {}).get("probe_counts") or {}
        if not counts:
            continue
        for off in offsets:
            measured = {k: v + off for k, v in counts.items()}
            res = compare(measured, gold, declared_slug=entry["slug"])
            row = {
                "slug": entry["slug"],
                "offset": off,
                "verdict": res.verdict,
                "best": res.best.slug if res.best else None,
                "expect_match": bool(entry.get("usable_for_discrimination")),
            }
            row["ok"] = (res.verdict == "matched" and res.best is not None
                         and res.best.slug == entry["slug"]) if row["expect_match"] \
                else res.verdict in ("inconsistent", "matched")
            rows.append(row)
            if not row["expect_match"] and res.best is not None and res.best.slug != entry["slug"]:
                ties.append({"slug": entry["slug"], "best": res.best.slug})
    total = len(rows)
    passed = sum(1 for r in rows if r["ok"])
    return {
        "total": total,
        "passed": passed,
        "failed": [r for r in rows if not r["ok"]],
        "ties": sorted({(t["slug"], t["best"]) for t in ties}),
    }
