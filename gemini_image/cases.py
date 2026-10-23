"""Gemini 图像生成压测套件的用例与参数矩阵。

参数口径（与 Gemini 原生 generateContent 字段一一对应）：

- **强度** → ``generationConfig.imageConfig.imageSize``：输出分辨率档位。
- **尺寸** → ``generationConfig.imageConfig.aspectRatio``：画面比例。
- **生成数量** → ``generationConfig.candidateCount``：单次请求的候选图片数。
- **模型** → URL 路径里的 ``models/{model}``。

Gemini 原生只有 imageSize 与 aspectRatio 两个几何字段，因此这里不做两个
重复的 imageSize 选择器：强度承载分辨率（决定像素量与图像 token 成本），
尺寸承载画面构图，二者合起来唯一确定输出图的规格。
"""
from __future__ import annotations

from dataclasses import dataclass


# 强度：分辨率档位。官方文档中 1K / 2K / 4K 分别为长边约 1024 / 2048 / 4096 px。
GEMINI_IMAGE_QUALITIES = ("1K", "2K", "4K")

# 尺寸：画面比例。与官方 imageConfig.aspectRatio 支持列表一致。
GEMINI_IMAGE_ASPECT_RATIOS = (
    "1:1",
    "3:2",
    "2:3",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
)

# 生成数量：单次请求的候选图片数（candidateCount 默认 1）。
GEMINI_IMAGE_COUNTS = (1, 2, 3, 4)

# 各模型实际支持的分辨率档位。显式声明，避免生成必然失败的请求组合；
# 列表外的自定义模型不在此约束内（按用户所选档位原样发送）。
MODEL_QUALITY_SUPPORT: dict[str, tuple[str, ...]] = {
    "gemini-3-pro-image": ("1K", "2K", "4K"),
    "gemini-3.1-flash-image": ("1K", "2K", "4K"),
}


@dataclass(frozen=True)
class Case:
    case_id: str
    title: str
    description: str
    project_model: str
    models: tuple[str, ...]
    qualities: tuple[str, ...]
    aspect_ratios: tuple[str, ...]
    counts: tuple[int, ...]
    cost: str = "高"
    protocols: tuple[str, ...] = ("gemini",)


GEMINI_IMAGE_GEN = Case(
    case_id="GEMINI_IMAGE_GEN",
    title="Gemini 图像生成",
    description=(
        "按官方 Gemini 图像接口并发生成；模型、强度（分辨率 1K/2K/4K）、"
        "尺寸（画面比例）与每次生成张数均从下方多选中逐请求随机抽取。"
    ),
    project_model="gemini-3-pro-image + gemini-3.1-flash-image",
    models=("gemini-3-pro-image", "gemini-3.1-flash-image"),
    qualities=GEMINI_IMAGE_QUALITIES,
    aspect_ratios=GEMINI_IMAGE_ASPECT_RATIOS,
    counts=GEMINI_IMAGE_COUNTS,
)

LOAD_CASES: dict[str, Case] = {
    GEMINI_IMAGE_GEN.case_id: GEMINI_IMAGE_GEN,
}

CASES: dict[str, Case] = dict(LOAD_CASES)
