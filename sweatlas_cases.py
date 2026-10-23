"""SWE-Atlas Test Writing official benchmark metadata."""
from dataclasses import dataclass

@dataclass(frozen=True)
class SWEAtlasTWCase:
    case_id: str = "SWE_ATLAS_TW"
    title: str = "SWE-Atlas TW·代码测试编写"
    description: str = "使用 Scale AI 官方开源 SWE-Atlas Test Writing 任务集；任务环境与 verifier 均由官方项目提供。"
    repository: str = "https://github.com/scaleapi/SWE-Atlas"
    dataset_path: str = "data/tw"
    protocols: tuple[str, ...] = ("openai", "volcengine")
    cost: str = "极高"

CASE = SWEAtlasTWCase()
