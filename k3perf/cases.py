from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    data_file: Path
    prompt_min: int
    prompt_max: int
    max_tokens: int
    mode: str
    concurrency: int | None = None
    successful_samples: int | None = None
    warmup_seconds: int = 0
    rate_stages: tuple[tuple[str, float, int, bool], ...] = ()
    ttft_p50_limit: float | None = None
    ttft_p90_limit: float | None = None
    mean_tpot_ms_limit: float | None = None
    e2e_p90_limit: float | None = None
    cache_ratio_min: float | None = None
    cache_ratio_max: float | None = None


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE_DATA = ROOT / "data_cache"

CASES = {
    # thinking 无法真正关闭（reasoning_effort=low 仍会产生思考 token，且计入
    # max_tokens 预算）：上限给小了思考就会吃满预算、零可见正文，请求被判
    # no_nonempty_chunk 失败。统一提到 4096 给思考+正文留足余量。
    "T0": Case("T0", DATA / "I1.jsonl.gz", 7200, 7900, 4096, "closed", concurrency=2, successful_samples=10),
    "L1": Case("L1", DATA / "I1.jsonl.gz", 7200, 7900, 4096, "closed", concurrency=50, successful_samples=300, warmup_seconds=120, ttft_p50_limit=2.5, ttft_p90_limit=6.0, mean_tpot_ms_limit=32.0, e2e_p90_limit=20.0),
    "L2": Case("L2", DATA / "I2.jsonl.gz", 11500, 12500, 4096, "closed", concurrency=30, successful_samples=300, warmup_seconds=120, ttft_p50_limit=4.0, ttft_p90_limit=10.0, mean_tpot_ms_limit=40.0),
    "L3": Case("L3", DATA / "I3.jsonl.gz", 23000, 25000, 4096, "closed", concurrency=20, successful_samples=300, warmup_seconds=120, ttft_p50_limit=4.0, ttft_p90_limit=9.0),
    "L4": Case("L4", DATA / "I4.jsonl.gz", 46000, 50000, 4096, "closed", concurrency=10, successful_samples=300, warmup_seconds=120, ttft_p50_limit=6.5, ttft_p90_limit=14.0),
    # C1 原 128：思考必然吃满、零正文必挂。提到 1024；输入仍占 ~10K/请求，
    # 10 RPS 下输入贡献 ~6.0M TPM，实际生成只会让实测 TPM 更高，不会压低结论。
    "C1": Case("C1", DATA / "C1.jsonl.gz", 9800, 10200, 1024, "open", rate_stages=(("warmup", 2.0, 60, False), ("ramp_4", 4.0, 60, False), ("ramp_6", 6.0, 60, False), ("ramp_8", 8.0, 60, False), ("steady", 10.0, 600, True))),
}

CACHE_CASES = {
    "CACHE_T0": Case("CACHE_T0", CACHE_DATA / "CI1.jsonl.gz", 7200, 7900, 4096, "closed", concurrency=2, successful_samples=10, cache_ratio_min=0.80, cache_ratio_max=0.90),
    "CACHE_L1": Case("CACHE_L1", CACHE_DATA / "CI1.jsonl.gz", 7200, 7900, 4096, "closed", concurrency=50, successful_samples=300, warmup_seconds=120, ttft_p50_limit=2.5, ttft_p90_limit=6.0, mean_tpot_ms_limit=32.0, e2e_p90_limit=20.0, cache_ratio_min=0.80, cache_ratio_max=0.90),
    "CACHE_L2": Case("CACHE_L2", CACHE_DATA / "CI2.jsonl.gz", 11500, 12500, 4096, "closed", concurrency=30, successful_samples=300, warmup_seconds=120, ttft_p50_limit=4.0, ttft_p90_limit=10.0, mean_tpot_ms_limit=40.0, cache_ratio_min=0.80, cache_ratio_max=0.90),
    "CACHE_L3": Case("CACHE_L3", CACHE_DATA / "CI3.jsonl.gz", 23000, 25000, 4096, "closed", concurrency=20, successful_samples=300, warmup_seconds=120, ttft_p50_limit=4.0, ttft_p90_limit=9.0, cache_ratio_min=0.80, cache_ratio_max=0.90),
    "CACHE_L4": Case("CACHE_L4", CACHE_DATA / "CI4.jsonl.gz", 46000, 50000, 4096, "closed", concurrency=10, successful_samples=300, warmup_seconds=120, ttft_p50_limit=6.5, ttft_p90_limit=14.0, cache_ratio_min=0.80, cache_ratio_max=0.90),
    "CACHE_C1": Case("CACHE_C1", CACHE_DATA / "CC1.jsonl.gz", 9800, 10200, 1024, "open", rate_stages=(("warmup", 2.0, 60, False), ("ramp_4", 4.0, 60, False), ("ramp_6", 6.0, 60, False), ("ramp_8", 8.0, 60, False), ("steady", 10.0, 600, True)), cache_ratio_min=0.80, cache_ratio_max=0.90),
}
CASES.update(CACHE_CASES)
