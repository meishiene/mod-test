from __future__ import annotations
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from .prompts import render_cached_prompt, render_prompt

@dataclass(frozen=True)
class PromptRecord:
    id: str
    dataset: str
    target_prompt_tokens: int
    marker: str
    word_count: int
    word_offset: int
    stored_content: str | None = None
    cache_mode: bool = False
    common_word_count: int = 0
    unique_word_count: int = 0

    @property
    def content(self) -> str:
        if self.stored_content is not None:
            return self.stored_content
        if self.cache_mode:
            return render_cached_prompt(self.marker, self.common_word_count, self.unique_word_count, self.word_offset)
        return render_prompt(self.marker, self.word_count, self.word_offset)

def load_prompts(path: Path) -> list[PromptRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run the corresponding prepare_data script first.")
    opener = gzip.open if path.suffix == ".gz" else open
    records = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip(): continue
            raw = json.loads(line)
            try:
                records.append(PromptRecord(
                    raw["id"], raw["dataset"], int(raw["target_prompt_tokens"]), raw.get("marker", raw["id"]),
                    int(raw.get("word_count", 0)), int(raw.get("word_offset", 0)), raw.get("content"),
                    bool(raw.get("cache_mode", False)), int(raw.get("common_word_count", 0)), int(raw.get("unique_word_count", 0)),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid record at {path}:{line_number}: {exc}") from exc
    if not records: raise ValueError(f"Dataset is empty: {path}")
    return records
