"""视觉能力面探针：判定「某个模型名到底会不会看图」，与平台用例解耦。

背景（2026-10-21）
------------------
`dsv4` 的 DS-V4-36/53 曾把 `deepseek-v4-flash-0731` 判成「视觉不达标」（failed）。
实测发现该判法有两处问题：

1. **能力面归属错了**：官方定价页「图像理解」只给 `deepseek-flash`（模型版本
   DeepSeek-V4.1-Flash-1017）打勾，`deepseek-v4-pro` 打叉；`deepseek-v4-flash-0731`
   是 2026-07-31 的纯文本版本，早于视觉首次对外开放（2026-08-21 的 vision-exp）。
2. **`prompt_tokens` 增量不是视觉证据**：网关对纯文本上游收到图片块同样 +75 tokens
   （24 → 99）并回 200，但内容答不出；真正看图的 V4.1-Flash 是 24 → 208。

所以本探针用「同一模型 × 三组请求」判能力面，**只看内容能否读对**：
纯文本基线 → 双色带图（红/蓝）→ 点阵数字图（OCR）。两项全对才算有视觉。

用法::

    DS_API_KEY=... DS_BASE_URL=https://api.oinone.top \\
      .venv/bin/python scripts/probe_ds_vision_capability.py \\
      deepseek-v4-flash-0731 deepseek-v4-pro-0813 deepseek-v4.1-flash-ark

输出 JSON 打印到 stdout，并写一份到 `results/DS_VISION_PROBE-<时间戳>/vision_probe.json`。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from common.png_fixture import data_url, image_part, png_data_url, png_text  # noqa: E402
from dsv4.gateway_contract import CAPABILITY_VISION, model_supports_capability  # noqa: E402

COLOR_PROMPT = "这张图上半部分和下半部分分别是什么颜色？只回答两个颜色词，用中文。"
OCR_PROMPT = "图中显示的4位数字是什么？只回答数字。"
OCR_DIGITS = "4237"
MODELS_DEFAULT = ("deepseek-v4-flash-0731", "deepseek-v4-pro-0813", "deepseek-v4.1-flash-ark")


def _chat(client: httpx.Client, url: str, model: str, content) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 32,
        "thinking": {"type": "disabled"},
    }
    response = client.post(url, json=body)
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    message = ((payload.get("choices") or [{}])[0].get("message") or {})
    usage = payload.get("usage") or {}
    return {
        "http_status": response.status_code,
        "content": str(message.get("content") or "")[:160],
        "prompt_tokens": usage.get("prompt_tokens"),
        "response_model": payload.get("model"),
        "error": None if payload.get("choices") else json.dumps(payload, ensure_ascii=False)[:200],
    }


def probe(model: str, client: httpx.Client, url: str) -> dict:
    baseline = _chat(client, url, model, COLOR_PROMPT)
    bands = _chat(client, url, model, [{"type": "text", "text": COLOR_PROMPT}, image_part(png_data_url())])
    ocr = _chat(client, url, model, [{"type": "text", "text": OCR_PROMPT}, image_part(data_url(png_text(OCR_DIGITS)))])
    colors_ok = "红" in bands["content"] and "蓝" in bands["content"]
    ocr_ok = OCR_DIGITS in ocr["content"]
    base_tokens, image_tokens = baseline.get("prompt_tokens"), bands.get("prompt_tokens")
    token_delta = (
        image_tokens - base_tokens
        if isinstance(base_tokens, int) and isinstance(image_tokens, int)
        else None
    )
    return {
        "model": model,
        "declared_vision_per_docs": model_supports_capability(model, CAPABILITY_VISION),
        "baseline": baseline,
        "bands_image": {**bands, "colors_correct": colors_ok},
        "ocr_image": {**ocr, "digits_correct": ocr_ok, "expected_digits": OCR_DIGITS},
        "token_delta": token_delta,
        # 判定：内容读对才算有视觉；token 增量只留痕（纯文本上游同样 +75）
        "vision_verified": bool(colors_ok and ocr_ok),
        "conclusion": (
            "vision_working"
            if colors_ok and ocr_ok
            else ("text_only_upstream" if not colors_ok and not ocr_ok else "partial_vision")
        ),
    }


def main(argv: list[str]) -> int:
    api_key = os.getenv("DS_API_KEY") or os.getenv("VOLC_API_KEY")
    base_url = (os.getenv("DS_BASE_URL") or os.getenv("VOLC_BASE_URL") or "").rstrip("/")
    if not api_key or not base_url:
        print(json.dumps({"error": "need DS_API_KEY / DS_BASE_URL"}, ensure_ascii=False))
        return 2
    models = [m for m in argv if not m.startswith("-")] or list(MODELS_DEFAULT)
    url = f"{base_url}/v1/chat/completions"
    records = []
    with httpx.Client(timeout=180, headers={"Authorization": f"Bearer {api_key}"}) as client:
        for model in models:
            records.append(probe(model, client, url))
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endpoint": url,
        "models": records,
        "note": (
            "视觉能力面按官方文档归属 deepseek-flash（V4.1-Flash-1017）；"
            "prompt_tokens 增量不能作为图片被理解的证据，只有内容读对才算。"
        ),
    }
    out_dir = REPO / "results" / f"DS_VISION_PROBE-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vision_probe.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    payload["evidence_dir"] = str(out_dir.relative_to(REPO))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
