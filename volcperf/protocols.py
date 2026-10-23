from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlparse


@dataclass(frozen=True)
class Protocol:
    id: str
    provider: str
    name: str
    default_base_url: str
    path_template: str
    description: str
    model_hint: str
    family: str
    test_kinds: tuple[str, ...] = ()
    # api_root() 拼给该协议的版本前缀。默认 /v1；火山方舟是 /api/v3，Gemini 是 /v1beta，
    # 千帆是 /v2。显式挂在协议上，避免再靠 family 反推（family 表达的是「请求/响应长什么样」，
    # 不是「路径怎么写」）。
    api_root_suffix: str = "/v1"


# OpenAI Chat Completions 线格式 + Bearer 鉴权的协议集合。
# 这些协议的 endpoint / headers / payload / SSE 解析形状完全一致，只有 api_root 前缀不同。
# 新增同类网关（如百度千帆）时只改这一处，不要在各地散落 {"openai", "volcengine"} 字面量。
OPENAI_COMPATIBLE_PROTOCOLS = ("openai", "volcengine", "qianfan")

# 图像生成面：OpenAI Images 线格式（Bearer 鉴权 + {root}/images/generations）。
IMAGE_COMPATIBLE_PROTOCOLS = (*OPENAI_COMPATIBLE_PROTOCOLS, "openai_images")

# 可以在正式请求里注入 reasoning_effort 的协议 = 前置连通性探针已经发过该字段的协议。
# 探针（connectivity_payload）对 Chat Completions 家族固定发 reasoning_effort=low，所以
# 网关若不吃这个字段，压测根本不会启动——正式请求再用同一取值，不引入新的兼容风险。
# Responses 面**故意不在内**：探针为兼容"先实现 /responses、后实现 reasoning 控制"的网关
# 只发最小子集，没有探针背书就不默认注入（需要时用 VOLC_REASONING_EFFORT 显式覆盖）。
REASONING_EFFORT_PROTOCOLS = OPENAI_COMPATIBLE_PROTOCOLS

# 用 Bearer 鉴权的协议家族（不按协议 id 判断：openai_images / seedance 都是独立 id、
# 家族挂在既有家族上，散落字面量会漏掉其中一个导致鉴权头缺失）。
BEARER_PROTOCOL_FAMILIES = ("openai", "volcengine", "seedance")


PROTOCOLS = {
    "openai": Protocol(
        "openai", "OpenAI", "Chat Completions", "https://api.openai.com", "/v1/chat/completions",
        "文字、视觉输入、工具调用与流式输出", "例如 gpt-5-mini", "openai",
        ("load", "kvv", "deepseek", "glm", "qwen", "gpt", "authenticity", "moderation",
         "arxivmath", "deepswe", "swe-atlas-tw", "swe-atlas-rf", "legal-agent-benchmark",
         "chat", "forensics"),
    ),
    "openai_responses": Protocol(
        "openai_responses", "OpenAI", "Responses", "https://api.openai.com", "/v1/responses",
        "统一文本/多模态输入、内置工具与 typed SSE", "例如 gpt-5-mini", "openai",
        ("load", "chat"),
    ),
    "openai_images": Protocol(
        "openai_images", "OpenAI", "Images · Generations", "https://api.openai.com", "/v1/images/generations",
        "图片生成接口（GPT Image）", "例如 gpt-image-2", "openai",
        ("gpt-image",),
    ),
    "anthropic": Protocol(
        "anthropic", "Claude", "Messages", "https://api.anthropic.com", "/v1/messages",
        "文字、视觉输入、Thinking、工具调用与 SSE", "例如 claude-sonnet-4-5", "anthropic",
        # 中转验真（forensics）与计费探针同属跨模型取证面：Messages 是主要取证协议。
        ("load", "fable", "chat", "forensics"),
    ),
    "gemini": Protocol(
        "gemini", "Gemini", "streamGenerateContent", "https://generativelanguage.googleapis.com",
        "/v1beta/models/{model}:streamGenerateContent?alt=sse",
        "文字、视觉/文件输入及支持模型的图像输出", "例如 gemini-2.5-flash", "gemini",
        ("load", "gemini", "chat", "forensics"),
        api_root_suffix="/v1beta",
    ),
    "gemini_images": Protocol(
        "gemini_images", "Gemini", "Images · generateContent", "https://generativelanguage.googleapis.com",
        "/v1beta/models/{model}:generateContent",
        "原生图像生成：imageConfig 控制分辨率与画面比例，candidateCount 控制每次张数",
        "例如 gemini-3-pro-image", "gemini",
        ("gemini-image",),
        api_root_suffix="/v1beta",
    ),
    "volcengine": Protocol(
        "volcengine", "火山云", "Chat Completions", "https://ark.cn-beijing.volces.com",
        "/api/v3/chat/completions", "火山方舟 OpenAI 兼容接口", "填写推理接入点或模型 ID", "volcengine",
        api_root_suffix="/api/v3",
    ),
    # Seedance / 即梦视频生成（火山方舟线格式的**异步任务**面）：
    # POST {root}/contents/generations/tasks 提交任务 → GET 同一路径 /{task_id} 轮询状态，
    # 任务成功后返回视频下载链接。请求体不是 Chat Completions 的 messages，而是
    # ``content: [{type: text|image_url}]`` + resolution / ratio / duration / watermark /
    # generate_audio 等顶层字段，因此独立成 family：只有 seedance 用例挂在它下面，
    # 文字/图像套件不会被误放行（family 是「请求/响应长什么样」的表达）。
    "seedance": Protocol(
        "seedance", "Seedance", "视频生成任务", "https://token-hub.tangrankeji.com",
        "/api/v3/contents/generations/tasks",
        "异步视频生成：提交任务拿 task id，轮询任务状态，成功后返回视频下载链接",
        "例如 Seedance-2.0、Seedance-2.5", "seedance",
        ("seedance",),
        api_root_suffix="/api/v3",
    ),
    # 百度千帆 ModelBuilder：v2 版本起对外就是标准 OpenAI Chat Completions 线格式，
    # 鉴权用 Bearer <API Key>（API Key 形如 bce-v3/ALTAK-xxxx/yyyy）。
    # family 取 "openai" 而不是自建：线上形状（请求体、SSE、usage、错误信封）与 OpenAI 完全一致，
    # 只有 api root 是 /v2。这样所有 openai 家族的套件（load/契约/指纹/开源测试集）都能直接选它。
    "qianfan": Protocol(
        "qianfan", "百度千帆", "Chat Completions", "https://qianfan.baidubce.com",
        "/v2/chat/completions",
        "千帆 ModelBuilder OpenAI 兼容接口（Bearer 鉴权，API Key 形如 bce-v3/ALTAK-…/…）",
        "例如 ernie-5.1、deepseek-v3.2-2504 等千帆模型 ID", "openai",
        api_root_suffix="/v2",
    ),
    # 阿里云百炼（DashScope）OpenAI 兼容模式：路径是 /compatible-mode/v1，
    # family 同样取 "openai"——请求体、SSE、usage（含 completion_tokens_details.reasoning_tokens）
    # 与 OpenAI 完全一致，只有 api root 不同；思考开关 enable_thinking / thinking_budget
    # 是 DashScope 的**非标准顶层字段**（SDK 里放 extra_body，裸 HTTP 放 body 顶层）。
    "qwen": Protocol(
        "qwen", "阿里云百炼", "Chat Completions（compatible-mode）",
        "https://dashscope.aliyuncs.com", "/compatible-mode/v1/chat/completions",
        "千问 OpenAI 兼容模式：混合思考、视觉输入、工具调用与流式输出（enable_thinking / thinking_budget）",
        "例如 qwen3.8-max、qwen3.7-plus、qwen3.6-flash", "openai",
        ("qwen", "load", "chat", "forensics"),
        api_root_suffix="/compatible-mode/v1",
    ),
}


# 每个协议在 api_root 之后由执行器/协议函数追加的「操作后缀」。
# Web 端的自定义请求路径若以其中一条结尾，会被剥掉只留根前缀：
# 根前缀拼进 base_url（api_root 对带路径的地址原样返回），操作后缀由执行器自己补回来，
# 因此自定义路径对全部执行链透明，不需要逐套件改造。
OPERATION_SUFFIXES: dict[str, tuple[str, ...]] = {
    "openai": ("/chat/completions",),
    "volcengine": ("/chat/completions",),
    "qianfan": ("/chat/completions",),
    "openai_responses": ("/responses",),
    "openai_images": ("/images/generations",),
    "anthropic": ("/messages",),
    "seedance": ("/contents/generations/tasks",),
    # Gemini 面的路径里嵌着真实模型 ID（{model} 占位符会被替换），后缀匹配不可靠，
    # 改用 /models/ 标记切分。
    "gemini": ("/models/{model}:streamGenerateContent?alt=sse",),
    "gemini_images": ("/models/{model}:generateContent",),
}
GEMINI_MODELS_MARKER = "/models/"


def split_path_override(path_override: str) -> str:
    """把自定义请求路径归一化为「拼进 base_url 的根前缀」。

    规则：
    * 空串 → 返回空串（= 不覆盖，走协议默认路径）；
    * 必须以 / 开头、不含空白字符、长度 ≤ 512，否则抛 ValueError；
    * 完整路径若以该协议的操作后缀结尾（或命中 Gemini 的 /models/ 标记），
      剥掉后缀只留根前缀；否则整段视为根前缀（操作后缀由执行器追加）。
    """
    value = (path_override or "").strip()
    if not value:
        return ""
    if not value.startswith("/") or any(character.isspace() for character in value) or len(value) > 512:
        raise ValueError("自定义路径必须以 / 开头、不含空白字符，且长度不超过 512")
    for suffix in OPERATION_SUFFIXES.values():
        for candidate in suffix:
            if value.endswith(candidate):
                return value[: -len(candidate)]
    if GEMINI_MODELS_MARKER in value:
        return value[: value.index(GEMINI_MODELS_MARKER)]
    return value


def protocol_spec(protocol: str) -> Protocol:
    try:
        return PROTOCOLS[protocol]
    except KeyError as exc:
        raise ValueError(f"不支持的协议: {protocol}") from exc


def normalize_base_url(base_url: str) -> str:
    """Validate and normalize the user-facing, host-only base URL."""
    value = base_url.strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url 必须是有效的 HTTP(S) 地址")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("base_url 只能填写域名（可含端口），不要带 / 或路径")
    if parsed.username or parsed.password:
        raise ValueError("base_url 不能包含用户名或密码")
    return f"{parsed.scheme}://{parsed.netloc}"


def api_root(protocol: str, base_url: str) -> str:
    """Return the SDK-style API root while accepting legacy pathful CLI values."""
    base = base_url.rstrip("/")
    if urlparse(base).path not in {"", "/"}:
        return base
    suffix = protocol_spec(protocol).api_root_suffix
    return base if base.endswith(suffix) else f"{base}{suffix}"


def display_endpoint(protocol: str, base_url: str, model: str = "{model}") -> str:
    spec = protocol_spec(protocol)
    return f"{base_url.rstrip('/')}{spec.path_template.format(model=quote(model, safe=''))}"


def supports_test_kind(protocol: str, test_kind: str, accepted_families: tuple[str, ...] | list[str]) -> bool:
    spec = protocol_spec(protocol)
    return spec.family in accepted_families and (not spec.test_kinds or test_kind in spec.test_kinds)


def endpoint(protocol: str, base_url: str, model: str, chat_path: str = "/chat/completions") -> str:
    root = api_root(protocol, base_url)
    if protocol == "openai_responses":
        return f"{root}/responses"
    if protocol in OPENAI_COMPATIBLE_PROTOCOLS:
        return f"{root}/{chat_path.lstrip('/')}"
    if protocol == "anthropic":
        return f"{root}/messages"
    if protocol == "gemini":
        return f"{root}/models/{quote(model, safe='')}:streamGenerateContent?alt=sse"
    if protocol == "openai_images":
        return f"{root}/images/generations"
    raise ValueError(f"协议 {protocol} 不支持文字流式请求")


def headers(protocol: str, api_key: str, request_id: str) -> dict[str, str]:
    # 非流式协议（图像生成 / 视频任务）不接受 text/event-stream：带错 Accept 会被网关按
    # 「不支持该响应格式」拒绝，与请求本身无关。
    accept = "application/json" if protocol in {"openai_images", "seedance"} else "text/event-stream"
    common = {"Content-Type": "application/json", "Accept": accept, "X-Request-ID": request_id}
    family = protocol_spec(protocol).family
    if family in BEARER_PROTOCOL_FAMILIES:
        common["Authorization"] = f"Bearer {api_key}"
    elif family == "anthropic":
        common.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
    elif family == "gemini":
        common["x-goog-api-key"] = api_key
    return common


def payload(protocol: str, model: str, messages: list[dict[str, str]] | str, max_tokens: int,
            include_usage: bool, reasoning_effort: str | None = None) -> dict[str, Any]:
    system = "You are a performance-test assistant. Follow the final user task exactly; do not call tools."
    normalized = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
    if protocol == "openai_responses":
        body = {"model": model, "stream": True, "max_output_tokens": max_tokens,
                "instructions": system, "input": normalized}
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        return body
    if protocol in OPENAI_COMPATIBLE_PROTOCOLS:
        body: dict[str, Any] = {"model": model, "stream": True, "max_tokens": max_tokens,
                                "messages": [{"role": "system", "content": system}, *normalized]}
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        if include_usage:
            body["stream_options"] = {"include_usage": True}
        return body
    if protocol == "anthropic":
        return {"model": model, "max_tokens": max_tokens, "stream": True, "system": system,
                "messages": normalized}
    if protocol == "gemini":
        role_map = {"assistant": "model", "user": "user"}
        return {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": role_map.get(message["role"], "user"), "parts": [{"text": message["content"]}]} for message in normalized],
                "generationConfig": {"maxOutputTokens": max_tokens}}
    raise ValueError(f"不支持的协议: {protocol}")


def models_endpoint(protocol: str, base_url: str) -> str:
    root = api_root(protocol, base_url)
    if protocol_spec(protocol).family in (*BEARER_PROTOCOL_FAMILIES, "anthropic"):
        return f"{root}/models"
    if protocol_spec(protocol).family == "gemini":
        return f"{root}/models?pageSize=1000"
    raise ValueError(f"不支持的协议: {protocol}")


def models_headers(protocol: str, api_key: str) -> dict[str, str]:
    result = {"Accept": "application/json"}
    family = protocol_spec(protocol).family
    if family in BEARER_PROTOCOL_FAMILIES:
        result["Authorization"] = f"Bearer {api_key}"
    elif family == "anthropic":
        result.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
    elif family == "gemini":
        result["x-goog-api-key"] = api_key
    return result


def parse_models(protocol: str, obj: dict[str, Any]) -> list[dict[str, str]]:
    family = protocol_spec(protocol).family
    raw_models = obj.get("models") if family == "gemini" else obj.get("data")
    raw_models = raw_models or []
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("name") if family == "gemini" else item.get("id")
        if not raw_id:
            continue
        model_id = str(raw_id)
        if family == "gemini" and model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
        if family == "gemini":
            methods = item.get("supportedGenerationMethods") or []
            if methods and not any(method in methods for method in ("generateContent", "streamGenerateContent")):
                continue
        if model_id in seen:
            continue
        seen.add(model_id)
        result.append({"id": model_id, "name": str(item.get("displayName") or model_id)})
    return sorted(result, key=lambda model: model["id"].lower())
