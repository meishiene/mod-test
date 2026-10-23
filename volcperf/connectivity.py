from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import quote

import httpx

from .config import ApiConfig
from .protocols import OPENAI_COMPATIBLE_PROTOCOLS
from .protocols import api_root
from .protocols import headers


_PROTOCOL_NAMES = {
    "openai": "OpenAI Chat Completions",
    "openai_responses": "OpenAI Responses",
    "openai_images": "OpenAI Images",
    "anthropic": "Claude Messages",
    "gemini": "Gemini generateContent",
    "volcengine": "火山云 Chat Completions",
    "qianfan": "百度千帆 Chat Completions",
    "seedance": "Seedance 视频生成任务",
}


def connectivity_endpoint(config: ApiConfig) -> str:
    base = api_root(config.protocol, config.base_url)
    if config.protocol == "openai_images":
        return f"{base}/images/generations"
    if config.protocol == "openai_responses":
        return f"{base}/responses"
    if config.protocol == "seedance":
        return f"{base}/contents/generations/tasks"
    if config.protocol in OPENAI_COMPATIBLE_PROTOCOLS:
        return config.endpoint
    if config.protocol == "anthropic":
        return f"{base}/messages"
    if config.protocol == "gemini":
        return f"{base}/models/{quote(config.model, safe='')}:generateContent"
    raise ValueError(f"不支持的协议: {config.protocol}")


def connectivity_payload(config: ApiConfig) -> dict[str, Any]:
    prompt = "Reply with OK only."
    if config.protocol == "openai_images":
        return {"model": config.model, "prompt": "A small green circle on a white background.",
                "n": 1, "size": "1024x1024", "quality": "low"}
    if config.protocol == "seedance":
        # 视频任务面没有轻量探针端点（GET /tasks 不开放、GET /tasks/{id} 需要先有任务、
        # {root}/models 在 IAM 边缘一律返回 200 信封、探不出鉴权真伪），因此只能真发一次
        # 最小任务：480p / 4 秒 / 不加水印，是官方档位里最便宜的一档。
        # 判定只看"提交是否被接受并返回 task id"，与图像套件用小图探活同一口径。
        return {"model": config.model, "content": [{"type": "text", "text": "connectivity probe: a still blue sky"}],
                "resolution": "480p", "ratio": "16:9", "duration": 4, "watermark": False}
    if config.protocol == "openai_responses":
        # Keep the probe to the portable Responses subset. Third-party gateways
        # often implement /responses before optional reasoning controls.
        return {"model": config.model, "max_output_tokens": 2048, "input": prompt}
    if config.protocol in OPENAI_COMPATIBLE_PROTOCOLS:
        # Thinking 模型的兼容网关可能要求输出预算大于 1024；与随机压测
        # 的默认输出上限保持一致，避免健康探针在正式请求前误判失败。
        return {"model": config.model, "stream": False, "max_tokens": 2048,
                "reasoning_effort": "low", "messages": [{"role": "user", "content": prompt}]}
    if config.protocol == "anthropic":
        # K3/Thinking routes commonly reject output budgets <= 1024 even when
        # the client does not explicitly include a thinking block.
        return {"model": config.model, "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}]}
    if config.protocol == "gemini":
        return {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 32, "thinkingConfig": {"thinkingBudget": 0}}}
    raise ValueError(f"不支持的协议: {config.protocol}")


def connectivity_headers(config: ApiConfig) -> dict[str, str]:
    """Headers for a non-streaming probe, including common gateway auth."""
    result = headers(config.protocol, config.api_key, "connectivity-check")
    result["Accept"] = "application/json"
    if config.protocol == "anthropic":
        # Native Anthropic uses x-api-key. Many multi-protocol gateways accept
        # the same key through Bearer auth, so sending both is the safest
        # compatible shape and does not expose either value in diagnostics.
        result["Authorization"] = f"Bearer {config.api_key}"
    return result


def _request_shape(config: ApiConfig, request_headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": "POST",
        "endpoint": connectivity_endpoint(config),
        "protocol": _PROTOCOL_NAMES.get(config.protocol, config.protocol),
        "model": config.model,
        "header_names": sorted(request_headers),
        "body_keys": sorted(body),
    }


def _upstream_diagnosis(config: ApiConfig, status_code: int, response_text: str) -> str | None:
    lowered = response_text.lower()
    protocol_name = _PROTOCOL_NAMES.get(config.protocol, config.protocol)
    if "invalid character '<' looking for beginning of value" in lowered or "<!doctype html" in lowered or "<html" in lowered:
        return (
            f"请求已到达网关，但网关收到的上游响应是 HTML 而不是 JSON。"
            f"通常是模型路由未启用 {protocol_name}，或该协议的上游地址配置异常；"
            "这不是本地 Base URL 拼装错误。"
        )
    if status_code in {404, 405}:
        return f"网关未开放 {protocol_name} 路径，或该路径不允许 POST。"
    if status_code == 503 and ("memory overloaded" in lowered or "overloaded" in lowered):
        return ("请求已到达网关，是网关自身过载（内存超阈值）而拒绝服务，"
                "属中转侧容量问题、与被测模型和本地配置无关；建议稍后重试或避开高峰。")
    if status_code in {502, 504}:
        return "请求已到达网关，但网关到上游的连接失败或超时；通常是中转上游不稳，建议稍后重试。"
    return None


CONNECTIVITY_TIMEOUT_CEILING = 60.0


def _connectivity_timeout(config: ApiConfig) -> float:
    """Probe read timeout: the caller's budget capped so a health check fails fast.

    Thinking models can burn the whole output budget inside the reasoning phase
    before a non-streaming response produces its first byte, and a proxied hop
    adds to that; raise the ceiling with VOLC_CONNECTIVITY_TIMEOUT_SECONDS when
    60s is genuinely too tight for the target channel.
    """
    ceiling = CONNECTIVITY_TIMEOUT_CEILING
    raw = (os.getenv("VOLC_CONNECTIVITY_TIMEOUT_SECONDS") or "").strip()
    if raw:
        try:
            ceiling = max(1.0, float(raw))
        except ValueError:
            ceiling = CONNECTIVITY_TIMEOUT_CEILING
    return min(config.request_timeout_seconds, ceiling)


def _transport_diagnosis(config: ApiConfig, exc: Exception,
                         timeout_seconds: float | None = None) -> str | None:
    """Explain transport-level failures, which are never model problems.

    The distinction that matters is *where* the request died: a connect failure
    never reached the server (routing/proxy), while a read timeout got a socket
    and then waited too long for the response body.  Reporting both as "未建立
    连接" sends the operator to audit a proxy that is working fine.
    """
    budget = f"{timeout_seconds:.0f}s" if timeout_seconds else "超时上限"
    if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout)):
        return (f"连接已建立，但 {budget} 内没有读回完整响应（{type(exc).__name__}）。"
                "这不是网络不通、也不是代理未生效：非流式请求要等服务端把整段响应写完，"
                "思考型模型的思维链会先耗尽预算再输出正文，经代理时耗时还会叠加。"
                "可放宽 VOLC_CONNECTIVITY_TIMEOUT_SECONDS 后重试；若被测服务支持流式，"
                "改用流式请求可避免整段等待。")
    if isinstance(exc, httpx.PoolTimeout):
        return ("本地 HTTP 连接池已耗尽（PoolTimeout），请求还未发出：调低并发或调大 "
                "VOLC_MAX_CONNECTIONS 后重试。与被测服务无关。")
    if not isinstance(exc, (httpx.ConnectTimeout, httpx.ConnectError, httpx.TimeoutException)):
        return None
    proxy_names = [name for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy") if os.getenv(name)]
    if config.trust_env:
        return ("已按 VOLC_TRUST_ENV 使用环境代理，但仍未建立连接：请确认代理进程正在运行、"
                "已放行该地址，或改为代理客户端的 TUN/增强模式。")
    if proxy_names:
        return ("本次为直连，已忽略环境中的代理变量（" + "、".join(proxy_names) + "）。"
                "若该地址需要经代理访问，请设置 VOLC_TRUST_ENV=true（start.sh 会加载 .env），"
                "或让代理客户端启用 TUN/增强模式后重试。")
    return ("直连失败且未检测到代理配置。请确认目标地址可达；若为境外地址，需先配置代理"
            "（设置 HTTP_PROXY/HTTPS_PROXY 并令 VOLC_TRUST_ENV=true）。")


def check_connectivity(config: ApiConfig) -> dict[str, Any]:
    """Send one tiny request. HTTP 2xx + a valid JSON response means reachable.

    Visible content is deliberately not required: a reasoning model can consume the
    tiny output budget in its thinking phase while the endpoint itself is healthy.
    """
    started = time.perf_counter()
    body = connectivity_payload(config)
    request_headers = connectivity_headers(config)
    timeout_seconds = _connectivity_timeout(config)
    result: dict[str, Any] = {"name": "connectivity", "passed": False, "protocol": config.protocol,
                              "endpoint": connectivity_endpoint(config), "http_status": None,
                              "duration_ms": None, "timeout_seconds": timeout_seconds,
                              "error": None, "diagnosis": None,
                              "request": _request_shape(config, request_headers, body)}
    try:
        with httpx.Client(timeout=timeout_seconds, verify=config.verify_ssl,
                          trust_env=config.trust_env) as client:
            response = client.post(result["endpoint"], headers=request_headers, json=body)
        result["http_status"] = response.status_code
        result["content_type"] = response.headers.get("content-type")
        result["provider_request_id"] = (response.headers.get("x-request-id")
                                         or response.headers.get("request-id")
                                         or response.headers.get("cf-ray"))
        if 200 <= response.status_code < 300:
            try:
                payload = response.json()
                result["passed"] = isinstance(payload, dict) and bool(payload)
                if not result["passed"]:
                    result["error"] = "接口返回了空 JSON 对象"
            except ValueError:
                result["error"] = "接口返回 HTTP 2xx，但响应不是有效 JSON"
        else:
            result["error"] = f"HTTP {response.status_code}: {response.text[:500]}"
            result["diagnosis"] = _upstream_diagnosis(config, response.status_code, response.text)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["diagnosis"] = _transport_diagnosis(config, exc, timeout_seconds)
    result["duration_ms"] = (time.perf_counter() - started) * 1000
    return result
