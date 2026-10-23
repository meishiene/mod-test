"""图片生成的可选自定义提示词，Web 与 CLI 使用同一校验规则。"""

MAX_IMAGE_PROMPT_CHARS = 8000


def normalize_image_prompt(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("图片提示词必须是文本")
    if len(value) > MAX_IMAGE_PROMPT_CHARS:
        raise ValueError(f"图片提示词过长（上限 {MAX_IMAGE_PROMPT_CHARS} 字符）")
    if "\x00" in value:
        raise ValueError("图片提示词不能包含空字符")
    return value if value.strip() else None
