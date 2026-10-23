#!/usr/bin/env python3
"""Kimi K3 全量跑批 token / 费用估算（可复核、可调参）。

口径说明（改这里即可重算）：
- 单价取 Kimi K3 官方 API 公开价：输入(缓存未命中) ¥20/MTok、缓存命中 ¥2/MTok、输出 ¥100/MTok。
  中转/供应商议价与此不同，改 IN_MISS / IN_HIT / OUT 即可。
- 输入 token 按各用例的 input_tokens_target / 数据集 prompt 档位；输出按 max_tokens 上限。
  真实账单取决于实际生成长度与真实缓存命中率，本模型给的是**标称上界**。
"""
from __future__ import annotations

IN_MISS, IN_HIT, OUT = 20.0, 2.0, 100.0  # 元 / 百万 token


def money(in_miss: float, in_hit: float, out: float) -> float:
    return in_miss / 1e6 * IN_MISS + in_hit / 1e6 * IN_HIT + out / 1e6 * OUT


# (套件, 请求数, 输入总 token, 其中缓存命中, 输出总 token)
ROWS: list[tuple[str, int, float, float, float]] = []

# A. KVV 官方套件（Web 三个 case）+ 公共契约探针
for name, n, per_in, per_out in (
    ("KVV_K3_FEATURES", 46, 400, 120),
    ("KVV_PARAMS", 18, 250, 80),
    ("KVV_TOOL_SCHEMA", 408, 800, 800),  # thinking 开启，max_tokens=2048
):
    ROWS.append((name, n, per_in * n, 0, per_out * n))
ROWS.append(("KVV_CONTRACT", 5, 2000, 0, 1000))

# B. K3 供应商自测基线（68 原子 -> 66 实际请求：4 条 not_evaluable 不发请求，
#    2 个缓存合并案例各补 1 次 warm 请求）
func_in, func_out = 600, 1824
ttft_in = 10 * 16000 + 10 * 64000 + 10 * 100000
ttft_out = 30 * 512
tpot_in, tpot_out = 10 * 16000, 10 * 512
cache_total, cache_hit = 12 * 70000, 10 * 70000
ROWS.append((
    "K3_TEMP_SUITE", 66,
    func_in + ttft_in + tpot_in + cache_total,
    cache_hit,
    func_out + ttft_out + tpot_out + 12 * 16,
))

# C. load 性能压测（run_suite.py 推荐节奏 T0/B1/L1-L5/O1/C0/C1）
LOAD = (
    ("T0", 10, 7500, 4096), ("B1", 16, 14500, 1024),
    ("L1", 100, 7500, 1024), ("L2", 100, 12000, 1024), ("L3", 100, 24000, 1024),
    ("L4", 80, 48000, 1024), ("L5", 50, 48000, 1024), ("O1", 50, 12000, 550),
    ("C0", 645, 10000, 1024), ("C1", 1890, 10000, 1024),
)
ROWS.append(("load_perf", sum(x[1] for x in LOAD),
             sum(x[1] * x[2] for x in LOAD), 0, sum(x[1] * x[3] for x in LOAD)))

# D. k3perf 缓存压测（CACHE_T0/L1-L4/C1），命中率按档位下限 0.85
CP = (
    ("CACHE_T0", 10, 7500, 1024), ("CACHE_L1", 300, 7500, 1024),
    ("CACHE_L2", 300, 12000, 1024), ("CACHE_L3", 300, 24000, 1024),
    ("CACHE_L4", 300, 48000, 1024), ("CACHE_C1", 7200, 10000, 1024),
)
cp_in = sum(x[1] * x[2] for x in CP)
ROWS.append(("k3perf_cache", sum(x[1] for x in CP), cp_in, cp_in * 0.85,
             sum(x[1] * x[3] for x in CP)))


def main() -> int:
    header = f"{'套件':<18}{'请求数':>9}{'输入token':>15}{'命中':>14}{'输出token':>13}{'费用¥':>11}"
    print(header)
    print("-" * len(header.encode("gbk", "ignore")))
    tot = [0, 0, 0, 0, 0.0]
    for name, n, i, ih, o in ROWS:
        cost = money(i - ih, ih, o)
        print(f"{name:<18}{n:>9,}{i:>15,.0f}{ih:>14,.0f}{o:>13,.0f}{cost:>11.2f}")
        tot[0] += n
        tot[1] += i
        tot[2] += ih
        tot[3] += o
        tot[4] += cost
    print("-" * len(header.encode("gbk", "ignore")))
    print(f"{'合计':<18}{tot[0]:>9,}{tot[1]:>15,.0f}{tot[2]:>14,.0f}{tot[3]:>13,.0f}{tot[4]:>11.2f}")
    print(f"\n总 token ≈ {tot[1] + tot[3]:,.0f}（输入 {tot[1]:,.0f} / 输出 {tot[3]:,.0f}）")
    k3_only = sum(money(i - ih, ih, o) for n_, n, i, ih, o in ROWS
                  if not n_.startswith(("load", "k3perf")))
    load_only = sum(money(i - ih, ih, o) for n_, n, i, ih, o in ROWS if n_.startswith("load"))
    cache_only = sum(money(i - ih, ih, o) for n_, n, i, ih, o in ROWS if n_.startswith("k3perf"))
    print(f"\n档 1  K3 专属套件（KVV×3 + 契约 + 供应商基线）≈ ¥{k3_only:.2f}")
    print(f"档 2  档1 + load 性能压测（Web 可派发全集）    ≈ ¥{k3_only + load_only:.2f}")
    print(f"档 3  档2 + k3perf 缓存长稳态（CLI）           ≈ ¥{tot[4]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
