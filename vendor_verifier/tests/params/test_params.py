"""API parameter constraint tests.

Verifies that immutable parameters use their default values and that wrong
values are rejected by the vendor API. Covers both thinking and non-thinking
modes.

**本地补丁（2026-10-14）——契约档位按模型切换**

原始快照只承载 Kimi-K2.5 代契约（`temperature` 1.0(think)/0.6(non-think) 可改），
与同一快照 `verify_params.py` 的文件头说明一致；但 Kimi-K3 官方把 `temperature`
固定为 **1.0**（任何非 1 取值原厂同样 400）。用 K2.5 的期望值去测 K3，会把
「与原厂一致」的行为判成失败——2026-10-13 `KVV_PARAMS` 的 2 条 `failed` 就是这样
产生的假阳性。

因此这里按被测模型名选择契约档位：

- 显式 `KVV_PARAM_CONTRACT=k3` / `=k2` 优先（平台 `kvv/runner.py` 走这条，默认 k3，
  不依赖被测模型名——供应商侧模型名未必含 `k3`，例如 `qianfan-vl-v1`）。
- 未显式指定时按模型名猜：含 `k3` → **K3 契约**；否则 → **K2.5 契约**（与快照原行为一致）。
- **K3 契约**：`temperature` 只接受 1.0，0.6 / 0.0 属负向；`thinking` 用官方唯一合法
  显式形态 `{"type":"enabled","keep":"all"}`，且因「K3 思考关不掉」，non-thinking
  分支改为**不传** `thinking`。

注意：`thinking` 的 extra_body 会随每一个请求发送，形态不对会让整套参数用例
集体 400（16 条假失败），所以档位切换必须同时切换它会用的形态。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import openai
import pytest


@dataclass
class ParamSpec:
    name: str
    think_default: Any
    non_think_default: Any
    wrong_value: Any


# Kimi-K2.5 契约：temperature 1.0(think) / 0.6(non-think)。
K2_IMMUTABLE_PARAMS: list[ParamSpec] = [
    ParamSpec("temperature", 0.6, 0.6, 1.1),
    ParamSpec("temperature", 0.0, 0.6, 2.0),
    ParamSpec("temperature", 1.0, 0.6, -0.1),
    ParamSpec("top_p", 0.95, 0.95, 0.8),
    ParamSpec("presence_penalty", 0, 0, 0.5),
    ParamSpec("frequency_penalty", 0, 0, 0.5),
    ParamSpec("n", 1, 1, 2),
]

# Kimi-K3 契约：temperature 固定 1.0；0.6 / 0.0 都是负向（同区间内的非固定值也要拒）。
K3_IMMUTABLE_PARAMS: list[ParamSpec] = [
    ParamSpec("temperature", 1.0, 1.0, 0.6),
    ParamSpec("temperature", 1.0, 1.0, 0.0),
    ParamSpec("top_p", 0.95, 0.95, 0.8),
    ParamSpec("presence_penalty", 0, 0, 0.5),
    ParamSpec("frequency_penalty", 0, 0, 0.5),
    ParamSpec("n", 1, 1, 2),
]


def _smoke_model() -> str:
    """取被测模型名：MODEL_NAME / KIMI_MODEL 环境变量，其次 ``--smoke-model`` 参数。"""
    for key in ("MODEL_NAME", "KIMI_MODEL"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    argv = list(sys.argv)
    for index, item in enumerate(argv):
        if item == "--smoke-model" and index + 1 < len(argv):
            return argv[index + 1].strip()
        if item.startswith("--smoke-model="):
            return item.split("=", 1)[1].strip()
    return ""


def active_contract(model: str | None = None) -> str:
    """返回 ``"k3"`` 或 ``"k2"``；显式 ``KVV_PARAM_CONTRACT`` 优先。"""
    explicit = os.environ.get("KVV_PARAM_CONTRACT", "").strip().lower()
    if explicit:
        if explicit in {"k3", "kimi-k3"}:
            return "k3"
        if explicit in {"k2", "k2.5", "kimi-k2.5"}:
            return "k2"
        raise ValueError(f"KVV_PARAM_CONTRACT 仅支持 k2 / k3，收到：{explicit}")
    target = (model if model is not None else _smoke_model()).lower()
    return "k3" if "k3" in target else "k2"


CONTRACT = active_contract()
IMMUTABLE_PARAMS = K3_IMMUTABLE_PARAMS if CONTRACT == "k3" else K2_IMMUTABLE_PARAMS


PARAM_IDS = [
    f"{param.name}={param.wrong_value}"
    for param in IMMUTABLE_PARAMS
]


def _get_thinking_extra_body(thinking: bool, think_mode: str) -> dict[str, Any]:
    if think_mode == "none":
        return {}
    if think_mode == "opensource":
        return {"chat_template_kwargs": {"thinking": thinking}}
    if CONTRACT == "k3":
        # K3：显式形态只有 enabled+keep=all；思考关不掉，non-thinking 分支不传该参数。
        return {"thinking": {"type": "enabled", "keep": "all"}} if thinking else {}
    return {"thinking": {"type": "enabled" if thinking else "disabled"}}


def _make_request(
    client: openai.Client,
    model: str,
    thinking: bool,
    think_mode: str,
    extra_params: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'OK' and nothing else."}],
        "extra_body": _get_thinking_extra_body(thinking, think_mode),
    }
    if extra_params:
        kwargs.update(extra_params)

    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content[:50] if response.choices else "no content"
        return True, f"OK: {content}"
    except openai.BadRequestError as exc:
        return False, f"Rejected(400): {exc.message}"
    except Exception as exc:
        return False, f"Error: {type(exc).__name__}: {exc}"


@pytest.mark.parametrize("thinking", [False, True], ids=["non-thinking", "thinking"])
def test_no_param_succeeds(
    client: openai.Client,
    model: str,
    thinking: bool,
    think_mode: str,
) -> None:
    success, msg = _make_request(client, model, thinking, think_mode)
    assert success, f"request without optional params failed: {msg}"


@pytest.mark.parametrize("thinking", [False, True], ids=["non-thinking", "thinking"])
def test_default_param_accepted(
    client: openai.Client,
    model: str,
    thinking: bool,
    think_mode: str,
) -> None:
    for param in IMMUTABLE_PARAMS:
        default_value = param.think_default if thinking else param.non_think_default
        success, msg = _make_request(
            client, model, thinking, think_mode, {param.name: default_value}
        )
        assert success, f"{param.name}={default_value} should be accepted: {msg}"


@pytest.mark.parametrize("param", IMMUTABLE_PARAMS, ids=PARAM_IDS)
@pytest.mark.parametrize("thinking", [False, True], ids=["non-thinking", "thinking"])
def test_wrong_param_rejected(
    client: openai.Client,
    model: str,
    thinking: bool,
    think_mode: str,
    param: ParamSpec,
) -> None:
    default_value = param.think_default if thinking else param.non_think_default
    if param.wrong_value == default_value:
        pytest.skip("wrong_value equals default_value")

    success, msg = _make_request(
        client, model, thinking, think_mode, {param.name: param.wrong_value}
    )
    assert not success, (
        f"{param.name}={param.wrong_value} should be rejected, but request succeeded: {msg}"
    )
