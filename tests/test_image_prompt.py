import asyncio
import base64
import io
import json
import os
from unittest.mock import patch

import httpx
import pytest
from PIL import Image

from gpt_image import runner as gpt_runner
from gemini_image import runner as gemini_runner


@pytest.mark.parametrize("runner,case", [(gpt_runner, "GPT_IMAGE_2"), (gemini_runner, "GEMINI_IMAGE_GEN")])
@pytest.mark.parametrize("prompt", ["猫" * 8001, "cat\x00dog"])
def test_image_prompt_cli_rejects_invalid_text_before_generating_requests(runner, case, prompt):
    with pytest.raises(SystemExit) as error:
        asyncio.run(runner.main_async([case, "--requests", "1", "--concurrency", "1", "--prompt", prompt]))
    assert error.value.code == 2


@pytest.mark.parametrize("runner,case,model", [
    (gpt_runner, "GPT_IMAGE_2", "gpt-image-2"),
    (gemini_runner, "GEMINI_IMAGE_GEN", "gemini-3-pro-image"),
])
@pytest.mark.parametrize("source", ["environment", "cli", "blank"])
def test_image_prompt_reaches_actual_requests_and_saved_evidence(tmp_path, runner, case, model, source):
    prompt = '  一只橘猫坐在窗边，柔和的晨光。\n保留 "引号"、`反引号` 和 $(文字)。  '
    prefix = runner.__name__.split(".")[0].upper()
    env = {
        prefix + "_API_KEY": "test-secret",
        prefix + "_BASE_URL": "https://example.test",
        prefix + "_MODELS": model,
        prefix + "_PROMPT": " \n " if source == "blank" else prompt,
    }
    argv = [case, "--output-root", str(tmp_path), "--requests", "3", "--concurrency", "2", "--seed", "7"]
    if source == "cli":
        env[prefix + "_PROMPT"] = "旧环境提示词"
        argv += ["--prompt", prompt]
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        actual = payload["prompt"] if runner is gpt_runner else payload["contents"][0]["parts"][0]["text"]
        seen.append(actual)
        if runner is gpt_runner:
            response = {"data": [{"b64_json": encoded}], "model": model}
        else:
            count = payload["generationConfig"]["candidateCount"]
            response = {"candidates": [{"content": {"parts": [
                {"inlineData": {"mimeType": "image/png", "data": encoded}} for _ in range(count)
            ]}, "finishReason": "STOP"}], "modelVersion": model}
        return httpx.Response(200, json=response)

    original_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return original_client(*args, **kwargs)

    with patch.dict(os.environ, env, clear=True), patch.object(runner.httpx, "AsyncClient", side_effect=client_factory):
        code = asyncio.run(runner.main_async(argv))
    assert code == 0
    assert len(seen) == 3
    result = next(path for path in tmp_path.iterdir() if path.is_dir())
    metadata = json.loads((result / "metadata.json").read_text())
    evidence = [json.loads(line) for line in (result / "requests.jsonl").read_text().splitlines()]
    planned = [json.loads(line) for line in (result / "generated_requests.jsonl").read_text().splitlines()]
    # 并发 2 下 ``seen`` 是到达顺序，证据文件按 index 排序 ⇒ 只能按集合比较，
    # 顺序敏感的逐位相等会在并发抖动时假失败（与提示词透传无关）。
    assert sorted(row["prompt"] for row in evidence) == sorted(seen)
    assert sorted(row["prompt"] for row in planned) == sorted(seen)
    if source == "blank":
        assert all(value in runner.PROMPTS for value in seen)
        assert metadata["prompt_count"] == len(runner.PROMPTS)
        assert metadata["prompt_mode"] == "random"
    else:
        assert seen == [prompt] * 3
        assert metadata["prompt_count"] == 1
        assert metadata["prompt_mode"] == "custom"
