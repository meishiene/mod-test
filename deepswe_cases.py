"""DeepSWE official open benchmark metadata; task corpus is not copied into this repo."""
from dataclasses import dataclass

@dataclass(frozen=True)
class DeepSWECase:
    case_id: str = "DEEPSWE"
    title: str = "DeepSWE·长程代码 Agent"
    description: str = "使用 DataCurve 官方开源 DeepSWE 任务集（113 个软件工程任务）及其程序化 verifier；运行时从官方 GitHub 获取。"
    repository: str = "https://github.com/datacurve-ai/deep-swe.git"
    protocols: tuple[str, ...] = ("openai", "volcengine")
    cost: str = "极高"

CASE = DeepSWECase()
