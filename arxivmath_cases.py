"""MathArena ArXivMath benchmark metadata.

The questions/answers are intentionally not vendored here.  The runner downloads
MathArena/arxivmath from Hugging Face at execution time.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class ArxivMathCase:
    case_id: str = "ARXIVMATH"
    title: str = "ArXivMath·研究级数学最终答案"
    description: str = "直接使用 MathArena 官方 Hugging Face 数据集 MathArena/arxivmath；不修改、不重写题目。"
    dataset_id: str = "MathArena/arxivmath"
    split: str = "train"
    protocols: tuple[str, ...] = ("openai", "volcengine", "anthropic", "gemini")
    cost: str = "高"

CASE = ArxivMathCase()
