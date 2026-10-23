from __future__ import annotations

from dataclasses import dataclass


GPT_IMAGE_2_QUALITIES = ("auto", "low", "medium", "high")
GPT_IMAGE_25_QUALITIES = ("auto", "low", "medium", "high", "xhigh", "max")
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
GPT_IMAGE_ASPECT_RATIOS = ("1:1", "3:2", "2:3", "4:3", "3:4", "5:4", "4:5", "16:9", "9:16", "21:9")


def model_quality_choices(models: list[str], qualities: list[str]) -> dict[str, list[str]]:
    choices = {}
    for model in models:
        allowed = GPT_IMAGE_2_QUALITIES if model == "gpt-image-2" else GPT_IMAGE_25_QUALITIES
        choices[model] = [quality for quality in qualities if quality in allowed]
        if not choices[model]:
            raise ValueError(f"{model} 不支持所选 quality，请选择兼容的 quality 或取消该模型")
    return choices

# GPT Image 官方规格尺寸池，按画幅比例分组后展开为 API 使用的
# "宽x高" 字符串。每个请求会从当前子项目的完整尺寸池中随机抽取。
GPT_IMAGE_2_SIZES = (
    "1K", "2K", "4K",
    # 1:1 正方形
    "1024x1024",
    "2048x2048",
    # 3:2 / 2:3
    "1536x1024",
    "1024x1536",
    # 4:3 / 3:4
    "1536x1152",
    "2048x1536",
    "1152x1536",
    "1536x2048",
    # 16:9 / 9:16
    "1536x864",
    "2048x1152",
    "2560x1440",
    "3840x2160",
    "864x1536",
    "1152x2048",
    "1440x2560",
    "2160x3840",
)

# GPT Image 2.5 supports the same standard size set and the same custom-size
# family.  Keep the test matrix explicit so every generated request is
# inspectable and reproducible from its request log.
GPT_IMAGE_25_SIZES = GPT_IMAGE_2_SIZES


@dataclass(frozen=True)
class Case:
    case_id: str
    title: str
    description: str
    project_model: str
    qualities: tuple[str, ...]
    sizes: tuple[str, ...]
    model_choices: tuple[str, ...]
    cost: str = "高"
    protocols: tuple[str, ...] = ("openai", "volcengine")


GPT_IMAGE_2 = Case(
    case_id="GPT_IMAGE_2",
    title="gpt-image-2 / 2.5",
    description="统一选择 image 2、2.5 Flare / Sunburst；按官方标准尺寸随机生成，quality 按模型支持范围抽取（xhigh / max 仅限 2.5）。",
    project_model="gpt-image-2",
    qualities=GPT_IMAGE_25_QUALITIES,
    sizes=GPT_IMAGE_2_SIZES,
    model_choices=("gpt-image-2", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst"),
)

GPT_IMAGE_25 = Case(
    case_id="GPT_IMAGE_25",
    title="gpt-image-2.5",
    description="按官方标准尺寸随机生成；quality 支持 auto / low / medium / high / xhigh / max；模型可在 Flare / Sunburst 间随机抽取。",
    project_model="gpt-image-2.5",
    qualities=GPT_IMAGE_25_QUALITIES,
    sizes=GPT_IMAGE_25_SIZES,
    model_choices=("gpt-image-2.5-flare", "gpt-image-2.5-sunburst"),
)

GPT_IMAGE_4K_TOKEN_VERIFY = Case(
    case_id="GPT_IMAGE_4K_TOKEN_VERIFY",
    title="4K · n=10 · Token 验真",
    description="gpt-image-2/high、gpt-image-2.5-flare/max、gpt-image-2.5-sunburst/max 各请求一次；固定 4K、n=10，核对 usage 与实际提示词/图片换算结果。",
    project_model="gpt-image-2 + gpt-image-2.5-flare + gpt-image-2.5-sunburst",
    qualities=("high", "max"),
    sizes=("3840x2160",),
    model_choices=("gpt-image-2", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst"),
    cost="极高",
)

LOAD_CASES: dict[str, Case] = {
    GPT_IMAGE_2.case_id: GPT_IMAGE_2,
    GPT_IMAGE_25.case_id: GPT_IMAGE_25,
}

CASES: dict[str, Case] = {
    **LOAD_CASES,
    GPT_IMAGE_4K_TOKEN_VERIFY.case_id: GPT_IMAGE_4K_TOKEN_VERIFY,
}
