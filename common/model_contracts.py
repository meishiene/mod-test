"""跨模型可复用的契约探针。

只复用测试方法；每个模型通过 Profile 提供自己的字段、数据和验收标准。
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any
from pathlib import Path

@dataclass(frozen=True)
class ModelProfile:
    family: str
    model: str
    thinking_on: dict[str, Any] | None
    thinking_off: dict[str, Any] | None
    expected_model_tokens: tuple[str, ...]
    answer_marker: str
    tool_name: str
    supports_reasoning: bool
    reasoning_efforts: tuple[str, ...] = ("low", "medium", "high")

def load_profile_data(family: str) -> dict[str, Any]:
    # 每个 family 的可选覆盖文件；缺失时返回空 dict，所有调用方均以 PROFILES 默认值兜底。
    path = Path(__file__).parent / "data" / f"{family}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def model_matches_family(family: str, model: str) -> tuple[bool, str]:
    """运行前门禁：配置的模型名必须属于该家族。

    背景（2026-09-24）：把 gpt-5.6-sol 误配进 DS_V4_P0 后，所有正向用例被上游 400
    "unknown model" 拒绝，而负向用例的「4xx 且非 404」断言恰好被这个 400 满足，产生假通过。
    因此入口脚本在发任何请求前先校验模型名：须包含 family 名或该家族白名单令牌之一。
    """
    lowered = (model or "").strip().lower()
    if not lowered:
        return False, "模型名为空，请在界面配置模型后再运行"
    data = load_profile_data(family)
    profile = PROFILES.get(family)
    defaults = profile.expected_model_tokens if profile else ()
    tokens = [str(t).strip().lower() for t in (data.get("model_tokens", defaults) or ()) if str(t).strip()]
    if family.lower() in lowered or any(t in lowered for t in tokens):
        return True, ""
    return False, (
        f"模型名 {model!r} 不属于 {family} 家族（要求名称包含 “{family}” 或白名单令牌 {tokens} 之一）。"
        f"误配模型会使所有正向用例被上游 400 拒绝，而负向用例的 4xx 断言恰好被该 400 满足产生假通过，"
        f"本次运行已阻止。请修正模型名后重跑。"
    )

PROFILES = {
    "kvv": ModelProfile("kvv", "", {"thinking": {"type": "enabled", "keep": "all"}}, None, ("kimi", "k3"), "KVV-CONTRACT-OK", "lookup_order", True, ("low", "high", "max")),
    "deepseek": ModelProfile("deepseek", "", {"thinking": {"type": "enabled"}}, {"thinking": {"type": "disabled"}}, ("deepseek-v4",), "DS-CONTRACT-OK", "lookup_order", True),
    "glm": ModelProfile("glm", "", {"thinking": {"type": "enabled"}}, None, ("glm",), "GLM-CONTRACT-OK", "lookup_order", True),
}

@dataclass(frozen=True)
class ProbeSpec:
    id: str
    name: str
    layer: str
    description: str

PROBES = (
    ProbeSpec("CONTRACT-01", "身份与路由", "identity", "模型返回标识、alias 和未知模型路由"),
    ProbeSpec("CONTRACT-02", "参数错误契约", "boundary", "非法 temperature、messages 类型应得到 4xx"),
    ProbeSpec("CONTRACT-03", "流式协议完整性", "protocol", "SSE 内容、结束标记和 usage"),
    ProbeSpec("CONTRACT-04", "工具调用多轮闭环", "protocol", "tool_choice、tool call 参数和 tool result"),
    ProbeSpec("CONTRACT-05", "Thinking 与 reasoning_effort", "capability", "固定开启 thinking，逐档验证模型支持的 reasoning_effort"),
)

def text(body: dict[str, Any] | None) -> str:
    msg = (((body or {}).get("choices") or [{}])[0].get("message") or {})
    value = msg.get("content", "")
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

def message(body: dict[str, Any] | None) -> dict[str, Any]:
    return (((body or {}).get("choices") or [{}])[0].get("message") or {})

def json_object(value: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(value)
        return obj if isinstance(obj, dict) else None
    except (TypeError, ValueError):
        return None

def validate_identity(status: int, body: dict[str, Any] | None, expected: tuple[str, ...]) -> tuple[bool, dict[str, Any]]:
    returned = str((body or {}).get("model", ""))
    return status == 200 and bool(returned) and any(token.lower() in returned.lower() for token in expected), {"response_model": returned}

def validate_usage(body: dict[str, Any] | None) -> bool:
    usage = (body or {}).get("usage") or {}
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return all(isinstance(usage.get(k), (int, float)) and usage[k] >= 0 for k in keys) and usage.get("total_tokens") == usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
