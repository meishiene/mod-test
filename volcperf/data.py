from __future__ import annotations

import gzip
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from .prompts import Workload, render_prompt, render_workload


@dataclass(frozen=True)
class PromptRecord:
    id: str
    dataset: str
    target_prompt_tokens: int
    marker: str
    word_count: int
    word_offset: int

    def content(self, fixed_output: bool = False) -> str:
        return render_prompt(self.marker, self.word_count, self.word_offset, "fixed" if fixed_output else "normal")

    def workload(self, fixed_output: bool = False) -> Workload:
        try:
            variant = int(self.id.rsplit("-", 1)[-1])
        except ValueError:
            variant = int(self.marker[:8], 16)
        return render_workload(self.marker, self.word_count, self.word_offset, variant, fixed_output=fixed_output)


def load_prompts(path: Path) -> list[PromptRecord]:
    if not path.exists():
        raise FileNotFoundError(f"测试数据不存在: {path}；请先运行 python scripts/prepare_data.py")
    opener = gzip.open if path.suffix == ".gz" else open
    records: list[PromptRecord] = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                records.append(PromptRecord(raw["id"], raw["dataset"], int(raw["target_prompt_tokens"]), raw["marker"], int(raw["word_count"]), int(raw["word_offset"])))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"无效数据 {path}:{number}: {exc}") from exc
    if not records:
        raise ValueError(f"测试数据为空: {path}")
    return records


def generate_random_prompts(
    count: int,
    *,
    minimum_tokens: int = 1000,
    maximum_tokens: int = 20000,
    seed: int | None = None,
) -> list[PromptRecord]:
    """Generate synthetic prompt recipes with independently randomized token targets."""
    if count < 1:
        raise ValueError("随机测试案例数必须大于 0")
    if minimum_tokens < 1 or maximum_tokens < minimum_tokens:
        raise ValueError("随机输入 Token 范围无效")
    rng = random.Random(seed)
    records: list[PromptRecord] = []
    for index in range(1, count + 1):
        target = rng.randint(minimum_tokens, maximum_tokens)
        marker_source = f"random-load:{index}:{target}:{rng.getrandbits(128)}"
        records.append(
            PromptRecord(
                id=f"RANDOM-{index:05d}-{target}",
                dataset="RANDOM_1K_20K",
                target_prompt_tokens=target,
                marker=hashlib.sha256(marker_source.encode()).hexdigest(),
                word_count=max(1, target - 115),
                word_offset=rng.randrange(22),
            )
        )
    return records
