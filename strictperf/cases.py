"""Web 严格压测选项；参数完全对应 k3test/k3perf/cases.py。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrictCase:
    id: str
    upstream_case: str
    title: str
    description: str
    load: str
    duration: str
    cost: str
    request_count: int
    concurrency: int
    protocols: tuple[str, ...] = ("openai", "volcengine")


CASES = {
    "STRICT_T0": StrictCase("STRICT_T0", "T0", "严格冒烟测试", "I1 真实数据（7200–7900 token），要求 10 个成功请求且 usage 完整", "并发 2 · 10 成功请求", "约 2 分钟", "中", 10, 2),
    "STRICT_L1": StrictCase("STRICT_L1", "L1", "严格 L1 日常上下文", "I1 真实数据；预热 120 秒后采集 300 个成功样本并执行全部准入门槛", "并发 50 · 300 成功请求", "至少 4 分钟", "高", 300, 50),
    "STRICT_L2": StrictCase("STRICT_L2", "L2", "严格 L2 长上下文", "I2 真实数据（11500–12500 token）；预热 120 秒后采集 300 个成功样本", "并发 30 · 300 成功请求", "至少 4 分钟", "很高", 300, 30),
    "STRICT_L3": StrictCase("STRICT_L3", "L3", "严格 L3 重型上下文", "I3 真实数据（23000–25000 token）；预热 120 秒后采集 300 个成功样本", "并发 20 · 300 成功请求", "至少 4 分钟", "极高", 300, 20),
    "STRICT_L4": StrictCase("STRICT_L4", "L4", "严格 L4 超长上下文", "I4 真实数据（46000–50000 token）；预热 120 秒后采集 300 个成功样本", "并发 10 · 300 成功请求", "至少 4 分钟", "极高", 300, 10),
    "STRICT_C1": StrictCase("STRICT_C1", "C1", "严格 C1 容量测试", "C1 真实数据；2/4/6/8 RPS 爬坡后，以 10 RPS 连续稳态 600 秒逐分钟验收", "最高 10 RPS · 约 7200 请求", "约 14 分钟+排空", "极高", 7200, 10),
}
