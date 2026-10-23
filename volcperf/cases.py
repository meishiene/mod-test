from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@dataclass(frozen=True)
class Case:
    name: str
    dataset: str
    target_prompt_tokens: int
    prompt_min: int
    prompt_max: int
    max_tokens: int
    mode: str
    concurrency: int = 1
    samples: int = 0
    warmup_seconds: int = 0
    measured_seconds: int = 0
    rps_stages: tuple[tuple[str, float, int, bool], ...] = ()
    ttfc_p50_limit: float | None = None
    ttfc_p90_limit: float | None = None
    tpot_p90_limit: float | None = None
    e2e_p90_limit: float | None = None
    min_visible_tokens: int = 128
    max_visible_tokens: int | None = None
    core: bool = False
    # 正式请求注入的思考档位（None = 不发该字段，走供应商默认）。
    #
    # 压测要量的是"服务把可见正文吐多快"，而思考 token 计入输出上限：走供应商默认思考时
    # deepseek-v4.1-flash 会把 L1–L5 的 1024 预算全部吃进推理，模型还没开始写正文就
    # finish_reason=length ⇒ 零可见正文（2026-10-14 批次 06707685f5044526 六档全灭的主因，
    # 105/300 请求命中，全部满足 reasoning_tokens == completion_tokens == max_tokens）。
    #
    # 这里与前置连通性探针（connectivity_payload 对 Chat Completions 家族固定发
    # reasoning_effort=low）对齐：探针能过 ⇒ 该网关认这个字段 ⇒ 正式请求可以安全复用同一
    # 取值。"探针能过、正式请求却大面积截断"正是同口径缺失造成的。
    # 优先级：环境变量 VOLC_REASONING_EFFORT（非空）> 本字段；协议不支持时一律不注入。
    reasoning_effort: str | None = "low"

    @property
    def data_file(self) -> Path:
        return DATA / f"{self.dataset}.jsonl.gz"


CASES = {
    # —— 输出上限（max_tokens）统一按「正文 + 思考」双份预算给 ——
    # 思考 token 计入输出上限：只按正文长度给预算时，思考会把整段预算吃进推理，模型还没开始
    # 写正文就 finish_reason=length ⇒ 零可见正文。2026-10-14 批次 06707685f5044526（当时
    # L1–L5=1024 / O1=2048）六档全灭即此因，105/300 请求命中，全部满足
    # reasoning_tokens == completion_tokens == max_tokens；FULL_LOAD 复跑（批次
    # bdb8e94cc40c4fcf，已注入 reasoning_effort=low）flash 仍 22–26%/档 吃满 1024
    # ⇒ 光靠降档不够，上限本身必须留思考余量。
    # 现值：L1–L5 / B1 / RANDOM_LOAD = 4096，O1 = 6144（O1 正文目标 400–700，另留思考余量）。
    # T0 冒烟：同上原因，4096 作为余量保留（冒烟档只判"10 条全成功"，宽松无副作用）。
    "T0": Case("T0", "I1", 7500, 7200, 7900, 4096, "closed", concurrency=2, samples=10, core=True),
    "RANDOM_LOAD": Case("RANDOM_LOAD", "RANDOM_1K_20K", 10500, 1000, 20000, 4096, "closed", concurrency=10, samples=100),
    # B1 合并执行 2K/8K/16K/32K，每档固定 4 次，合计 16 次；并发统一为 2。
    # 上限与 L1–L4 对齐：基线要与被测档位同一预算，否则 TTFC 相对退化比较不成立。
    "B1": Case("B1", "I1", 7500, 7200, 50000, 4096, "closed", concurrency=2, samples=16),
    "L1": Case("L1", "I1", 7500, 7200, 7900, 4096, "closed", 50, 200, 60, 180, ttfc_p50_limit=5, ttfc_p90_limit=15, tpot_p90_limit=60, core=True),
    "L2": Case("L2", "I2", 12000, 11500, 12500, 4096, "closed", 40, 200, 60, 180, ttfc_p50_limit=7, ttfc_p90_limit=20, tpot_p90_limit=65, core=True),
    "L3": Case("L3", "I3", 24000, 23000, 25000, 4096, "closed", 25, 200, 60, 180, ttfc_p50_limit=10, ttfc_p90_limit=28, tpot_p90_limit=70, core=True),
    "L4": Case("L4", "I4", 48000, 46000, 50000, 4096, "closed", 15, 150, 60, 180, ttfc_p50_limit=15, ttfc_p90_limit=40, tpot_p90_limit=80, core=True),
    "L5": Case("L5", "I4", 48000, 46000, 50000, 4096, "closed", 8, 100, 60, 180, ttfc_p50_limit=22, ttfc_p90_limit=60, tpot_p90_limit=90),
    "O1": Case("O1", "I2", 12000, 11500, 12500, 6144, "closed", 10, 100, 60, 180, ttfc_p90_limit=20, tpot_p90_limit=65, e2e_p90_limit=60, min_visible_tokens=400, max_visible_tokens=700, core=True),
    "C0": Case("C0", "C1", 10000, 9800, 10200, 1024, "open", rps_stages=(("ramp_4", 4.0, 30, True), ("ramp_7", 7.0, 30, True), ("ramp_10_5", 10.5, 30, True))),
    "C1": Case("C1", "C1", 10000, 9800, 10200, 1024, "open", rps_stages=(("steady", 10.5, 180, True),), core=True),
}
