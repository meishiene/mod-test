"""Readable artifact names for GPT image tests."""
import re
from datetime import datetime


def image_filename_stem(model: str, quality: str, generated_at: datetime | None = None) -> str:
    model_name = re.sub(r"^gpt-image-", "image", model.strip().lower())
    safe = lambda value: re.sub(r"[^a-z0-9._-]+", "-", value).strip(".-")[:80] or "unknown"
    stamp = (generated_at or datetime.now().astimezone()).strftime("%Y%m%d%H%M")
    return f"{safe(model_name)}-{safe(quality.strip().lower())}-{stamp}"
