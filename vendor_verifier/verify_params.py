"""
Verify Kimi API parameter constraints before running benchmarks.

Immutable parameters — 契约档位按被测模型切换（本地补丁 2026-10-14）：

- **K2.5 契约**：temperature 1.0 (think) / 0.6 (non-think)，与快照原行为一致
- **K3 契约**：temperature **固定 1.0**（任何非 1 取值原厂同样 400）；
  显式 `thinking` 只有 `{"type":"enabled","keep":"all"}`，且思考关不掉
- 两档共用：top_p 0.95、presence_penalty 0、frequency_penalty 0、n 1

档位选择：`KVV_PARAM_CONTRACT=k2|k3` > 模型名（含 `k3` ⇒ K3）> 默认 K2.5。
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any

import httpx
from openai import BadRequestError, OpenAI


@dataclass
class ParamSpec:
    name: str
    think_default: Any
    non_think_default: Any
    wrong_value: Any


K2_IMMUTABLE_PARAMS: list[ParamSpec] = [
    ParamSpec("temperature", 1.0, 0.6, 0.5),
    ParamSpec("top_p", 0.95, 0.95, 0.8),
    ParamSpec("presence_penalty", 0, 0, 0.5),
    ParamSpec("frequency_penalty", 0, 0, 0.5),
    ParamSpec("n", 1, 1, 2),
]

K3_IMMUTABLE_PARAMS: list[ParamSpec] = [
    ParamSpec("temperature", 1.0, 1.0, 0.6),
    ParamSpec("top_p", 0.95, 0.95, 0.8),
    ParamSpec("presence_penalty", 0, 0, 0.5),
    ParamSpec("frequency_penalty", 0, 0, 0.5),
    ParamSpec("n", 1, 1, 2),
]


def active_contract(model: str | None = None) -> str:
    """返回 ``"k3"`` 或 ``"k2"``；显式 KVV_PARAM_CONTRACT 优先。"""
    explicit = os.environ.get("KVV_PARAM_CONTRACT", "").strip().lower()
    if explicit:
        if explicit in {"k3", "kimi-k3"}:
            return "k3"
        if explicit in {"k2", "k2.5", "kimi-k2.5"}:
            return "k2"
        raise ValueError(f"KVV_PARAM_CONTRACT 仅支持 k2 / k3，收到：{explicit}")
    return "k3" if "k3" in str(model or "").lower() else "k2"


def immutable_params(contract: str) -> list[ParamSpec]:
    return K3_IMMUTABLE_PARAMS if contract == "k3" else K2_IMMUTABLE_PARAMS


IMMUTABLE_PARAMS = immutable_params("k2")


def get_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.Client(timeout=60.0),
    )


def get_thinking_extra_body(thinking: bool, think_mode: str, contract: str = "k2") -> dict:
    if think_mode == "none":
        return {}
    if think_mode == "opensource":
        return {"chat_template_kwargs": {"thinking": thinking}}
    if contract == "k3":
        # K3：显式形态只有 enabled+keep=all；思考关不掉 ⇒ non-think 分支不传该参数。
        return {"thinking": {"type": "enabled", "keep": "all"}} if thinking else {}
    return {"thinking": {"type": "enabled" if thinking else "disabled"}}


def make_request(
    client: OpenAI,
    model: str,
    thinking: bool,
    think_mode: str = "kimi",
    extra_params: dict | None = None,
    contract: str = "k2",
) -> tuple[bool, str]:
    """Send test request. Returns (success, message)."""
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'OK' and nothing else."}],
        "extra_body": get_thinking_extra_body(thinking, think_mode, contract),
    }
    if extra_params:
        kwargs.update(extra_params)

    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content[:50] if response.choices else "no content"
        return True, f"OK: {content}"
    except BadRequestError as e:
        return False, f"Rejected(400): {e.message}"
    except Exception as e:
        return False, f"Error: {type(e).__name__}: {e}"


def test_param_rejected(
    client: OpenAI,
    model: str,
    thinking: bool,
    param: ParamSpec,
    think_mode: str = "kimi",
    contract: str = "k2",
) -> tuple[bool, str]:
    """Test that wrong param value is rejected."""
    default_value = param.think_default if thinking else param.non_think_default
    if param.wrong_value == default_value:
        return True, "Skip: wrong_value == default_value"

    success, _ = make_request(client, model, thinking, think_mode, {param.name: param.wrong_value}, contract)
    if success:
        return False, f"❌ FAIL: {param.name}={param.wrong_value} should be rejected"
    return True, f"✓ PASS: {param.name}={param.wrong_value} correctly rejected"


def test_param_accepted(
    client: OpenAI,
    model: str,
    thinking: bool,
    param: ParamSpec,
    think_mode: str = "kimi",
    contract: str = "k2",
) -> tuple[bool, str]:
    """Test that correct param value is accepted."""
    default_value = param.think_default if thinking else param.non_think_default
    success, msg = make_request(client, model, thinking, think_mode, {param.name: default_value}, contract)
    if success:
        return True, f"✓ PASS: {param.name}={default_value} accepted"
    return False, f"❌ FAIL: {param.name}={default_value} rejected: {msg}"


def test_no_param(
    client: OpenAI,
    model: str,
    thinking: bool,
    think_mode: str = "kimi",
    contract: str = "k2",
) -> tuple[bool, str]:
    """Test request without optional params."""
    success, msg = make_request(client, model, thinking, think_mode, contract=contract)
    if success:
        return True, "✓ PASS: request without params succeeded"
    return False, f"❌ FAIL: request without params failed: {msg}"


def run_verification(
    base_url: str,
    api_key: str,
    model: str,
    thinking: bool,
    think_mode: str = "kimi",
    test_reject: bool = True,
    test_accept: bool = True,
    contract: str | None = None,
) -> bool:
    """Run full verification. Returns True if all tests pass."""
    contract = contract or active_contract(model)
    params = immutable_params(contract)
    client = get_client(base_url, api_key)
    mode_str = "think" if thinking else "non-think"

    print(f"\n{'='*60}")
    print(f"Mode: {mode_str} (think-mode: {think_mode}, param-contract: {contract})")
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"{'='*60}\n")

    all_passed = True
    results = []

    # Test 1: no optional params
    print("[1] Test without optional params...")
    passed, msg = test_no_param(client, model, thinking, think_mode, contract)
    results.append(passed)
    if not passed:
        all_passed = False
    print(f"    {msg}")

    # Test 2: correct default values accepted
    if test_accept:
        print("\n[2] Test correct default values...")
        for param in params:
            passed, msg = test_param_accepted(client, model, thinking, param, think_mode, contract)
            results.append(passed)
            if not passed:
                all_passed = False
            print(f"    {msg}")

    # Test 3: wrong values rejected
    if test_reject:
        print("\n[3] Test wrong values (should be rejected)...")
        for param in params:
            passed, msg = test_param_rejected(client, model, thinking, param, think_mode, contract)
            results.append(passed)
            if not passed:
                all_passed = False
            print(f"    {msg}")

    # Summary
    print(f"\n{'='*60}")
    passed_count = sum(results)
    status = "✓ ALL PASSED" if all_passed else "❌ SOME FAILED"
    print(f"Result: {status} ({passed_count}/{len(results)})")
    print(f"{'='*60}\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify Kimi API parameter constraints (K2.5 / K3 双契约)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_params.py --model kimi-k2.5 --thinking --think-mode kimi
  python verify_params.py --model kimi-k3 --all --param-contract k3
  python verify_params.py --model kimi-k2.5 --think-mode opensource --base-url http://localhost:8000/v1
""",
    )
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        help="API base URL (default: $KIMI_BASE_URL or https://api.moonshot.cn/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("KIMI_API_KEY"),
        help="API key (default: $KIMI_API_KEY)",
    )
    parser.add_argument("--thinking", action="store_true", help="Verify thinking mode")
    parser.add_argument(
        "--think-mode",
        choices=["kimi", "opensource"],
        default="kimi",
        help="Thinking param format: kimi (SaaS) or opensource (vLLM/SGLang/KTransformers)",
    )
    parser.add_argument(
        "--param-contract",
        choices=["k2", "k3"],
        default=None,
        help="参数契约档位（默认 $KVV_PARAM_CONTRACT，其次按模型名判断）",
    )
    parser.add_argument("--only-reject", action="store_true", help="Only test wrong values rejected")
    parser.add_argument("--only-accept", action="store_true", help="Only test correct values accepted")
    parser.add_argument("--all", action="store_true", help="Verify both thinking and non-thinking modes")

    args = parser.parse_args()

    if not args.api_key:
        print("Error: Set KIMI_API_KEY env var or use --api-key")
        sys.exit(1)

    contract = args.param_contract or active_contract(args.model)
    test_reject = not args.only_accept
    test_accept = not args.only_reject
    modes = [False, True] if args.all else [args.thinking]

    all_passed = True
    for thinking in modes:
        if not run_verification(
            args.base_url,
            args.api_key,
            args.model,
            thinking,
            think_mode=args.think_mode,
            test_reject=test_reject,
            test_accept=test_accept,
            contract=contract,
        ):
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
