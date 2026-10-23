"""SWE-Atlas Refactoring official benchmark metadata."""
from dataclasses import dataclass
@dataclass(frozen=True)
class SWEAtlasRFCase:
    case_id: str = 'SWE_ATLAS_RF'
    title: str = 'SWE-Atlas RF·代码重构'
    description: str = '使用 Scale AI 官方开源 SWE-Atlas Refactoring 任务集；任务环境与 verifier 均由官方项目提供。'
    repository: str = 'https://github.com/scaleapi/SWE-Atlas'
    dataset_path: str = 'data/rf'
    protocols: tuple[str,...] = ('openai','volcengine')
    cost: str = '极高'
CASE=SWEAtlasRFCase()
