#!/usr/bin/env python3
from __future__ import annotations
import gzip
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"
SPECS = {"I1": (7500, 500), "I2": (12000, 500), "I3": (24000, 500), "I4": (48000, 500), "C1": (10000, 7300)}
WORD_VOCAB_SIZE = 22

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"generator_version": 2, "format": "deterministic-prompt-recipe-v1", "seed_scheme": "k3-admission-v1:<dataset>:<index>", "datasets": {}}
    for name, (target, records) in SPECS.items():
        path = OUT / f"{name}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as stream:
            for index in range(records):
                seed = f"k3-admission-v1:{name}:{index}"
                record = {"id": f"{name}-{index:05d}", "dataset": name, "target_prompt_tokens": target, "marker": hashlib.sha256(seed.encode()).hexdigest(), "word_count": max(1, target - 115), "word_offset": random.Random(seed).randrange(WORD_VOCAB_SIZE)}
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["datasets"][name] = {"path": path.name, "target_prompt_tokens": target, "records": records, "sha256_compressed_file": digest, "compressed_bytes": path.stat().st_size}
        print(f"{name}: {records} prompt recipes -> {path} ({path.stat().st_size:,} bytes)")
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
