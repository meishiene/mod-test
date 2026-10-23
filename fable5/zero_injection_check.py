#!/usr/bin/env python3
"""Claude「0 注入中转」判定组（ZI 组四条用例）。

**0 注入中转**（圈内叫法）指网关对请求/消息体不做任何 prompt 注入、不追加系统提示、
不改写上游原始报文，只做协议转发与格式转码：客户端发什么原样透传给官方，官方返回
什么原样回给客户端。对应地，只有 0 注入中转，签名/thinking 块才是官方原生的。

本组从**客户端侧**（我们只能观测网关的 HTTP 面）逐条对上述禁令取证，四条用例一一对应：

  ZI-01  system 金丝雀透传         输入侧「不插 / 不覆盖用户 prompt 与 system」
  ZI-02  缺省 system 差分          输入侧「不偷偷追加前置 / 后置系统提示词」
  ZI-03  未声明工具面              报文 / 输出侧「不篡改、不掺模型」（偷加工具即信任级失败）
  ZI-04  签名原生性                报文 / 输出侧「不篡改 thinking.signature、不换后端」

**判据与归因纪律**（沿用项目既有口径，见 ``fable5/proto_probe.py`` 与 ``docs/``）：

- 信任级缺陷（注入 / 协议改写 / 换后端）才判 ``failed``；取不到证据（端点不支持 thinking、
  usage 缺失、传输错）一律 ``blocked``，**blocked 只记录不阻断准入**；
- 期望不可达时不得判模型失败：金丝雀未回显可能是模型没遵循指令，只有在**对照请求
  回显成功**时才认定是 system 被剥离 ⇒ ``failed``；对照也失败 ⇒ ``warning``；
- 负向 / 差分判定只看**相对增量**，绝对膨胀不设阈值（绝对膨胀归 Claude 指纹套件的
  F09 / F12，避免重复归因）；
- 瞬时 5xx 先重试再 ``blocked``，绝不判 ``failed``。

**与既有卡片的关系**（不重复取证）：

- 「Protobuf 签名两步校验」承担**篡改一个签名字符必须被拒**的负向控制；本组 ZI-04 只取
  **正向原生性**（签名非空 + 原签名可回传 + 模型字段一致），做 0 注入前提的轻量判定；
- 注入取证的三件套（F09 侧信道 / F12 零值 system / F13 未声明工具面）在 Claude 指纹套件内，
  本组是「0 注入」的独立判定入口，互为印证而非重复计数。

用法::

    .venv/bin/python -m fable5.zero_injection_check --self-test          # 离线自检
    .venv/bin/python -m fable5.zero_injection_check --model claude-fable-5 \\
        --base-url https://changhuai.ai --api-key sk-xxx
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fable5.proto_probe import (  # noqa: E402 —— 复用通道与判定对象（唯一权威实现）
    BLOCKED, FAILED, JSON_CT, PASSED, WARNING, Client, Resp, Verdict,
)
from fable5.model_profile import (  # noqa: E402 —— thinking 档位唯一权威
    ADAPTIVE, ADAPTIVE_EFFORT, ENABLED, resolve_thinking_mode, thinking_payload,
)

CASE_ID_ROOT = "ZERO_INJECTION_CHECK"

# 缺省 system（不传）与零值 system（system=""）的 input_tokens 差分阈值：
# 空串理应不产生任何 token，出现正向增量即说明网关在缺省时补了系统提示词。
# 低于该阈值的隐藏提示词与计数抖动不可区分，属本用例已知检测下限（不对绝对膨胀设阈值）。
INJECTION_TOKEN_THRESHOLD = 16

# 通用短文探针：短、确定、不触发拒答。
PROBE_PROMPT = "请只回复 OK 两个字母。"
INTRO_PROMPT = "请用一句话介绍杭州。"
CALC_PROMPT = "请解释快速排序的平均时间复杂度为什么是 O(n log n)，给出递推推导过程。"

# 模型族标记：用于识别「请求 opus、返回 sonnet」这类静默换后端（掺水）。
# 只做**同族/异族**判断，不比对版本号——网关把 claude-fable-5-1 映射成 claude-fable-5
# 属别名，记为留痕（warning 级观察）而不是失败。
_FAMILY_TOKENS = ("opus", "sonnet", "haiku", "fable")

# 上游对 thinking 的支持按**模型族**分裂（4.5/4.6 系只认 enabled、5 系只认 adaptive+effort），
# 唯一权威在 ``fable5/model_profile.py``；本模块只做调用，不再自带映射表。
_FAMILY_TOKENS = ("opus", "sonnet", "haiku", "fable")

_SIGNATURE_ERROR_PATTERNS = (
    r"(?:invalid|incorrect|bad|malformed|corrupt(?:ed)?)\s+(?:thinking\s+)?signature",
    r"signature.{0,50}(?:invalid|incorrect|verification failed|validation failed|does not match)",
    r"(?:failed|unable|cannot|could not)\s+to\s+(?:verify|validate).{0,30}signature",
    r"(?:签名.{0,30}(?:无效|不合法|验证失败|校验失败|不匹配)|(?:无效|不合法).{0,10}签名)",
)
_THINKING_ERROR_PATTERNS = (r"thinking", r"思考", r"budget_tokens")


# ---------------------------------------------------------------------------
# 一、纯函数（判定与解析；全部可离线自检）
# ---------------------------------------------------------------------------

def new_canary() -> str:
    """随机金丝雀：模型不可能凭上下文猜出，回显即证明该文本确实到达了模型。"""
    return "CANARY-" + os.urandom(6).hex().upper()


def canary_echoed(text: str, canary: str) -> bool:
    """金丝雀是否出现在响应正文里（大小写不敏感、允许模型加前后缀）。"""
    return bool(canary) and canary.lower() in (text or "").lower()


def response_text(body: dict[str, Any] | None) -> str:
    """拼接响应里全部 text 块（thinking 块不计入）。"""
    if not isinstance(body, dict):
        return ""
    parts = [block.get("text", "") for block in body.get("content") or []
             if isinstance(block, dict) and block.get("type") == "text"
             and isinstance(block.get("text"), str)]
    return "\n".join(parts)


def block_types(body: dict[str, Any] | None) -> list[str]:
    if not isinstance(body, dict):
        return []
    return [block.get("type") for block in body.get("content") or []
            if isinstance(block, dict) and isinstance(block.get("type"), str)]


def input_tokens(resp: Resp) -> int | None:
    body = resp.json() or {}
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("input_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def model_families(name: str | None) -> set[str]:
    lowered = (name or "").lower()
    return {token for token in _FAMILY_TOKENS if token in lowered}


def thinking_mode_for(model: str) -> str:
    """按模型名解析 thinking 档位（唯一权威在 ``fable5/model_profile.py``）。"""
    return resolve_thinking_mode(model)


def signed_thinking_blocks(body: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(body, dict):
        return []
    return [block for block in body.get("content") or []
            if isinstance(block, dict) and block.get("type") == "thinking"
            and isinstance(block.get("signature"), str) and block["signature"]]


def is_signature_rejection(resp: Resp) -> bool:
    """错误必须**指向签名**才算签名被拒（泛化 4xx 不算，沿用签名校验既有口径）。"""
    error = (resp.json() or {}).get("error")
    if not isinstance(error, dict):
        return False
    message = str(error.get("message") or "").lower().replace("`", "")
    if any(word in message for word in ("unsupported", "not supported", "不支持", "not allowed")):
        return False
    return any(re.search(pattern, message) for pattern in _SIGNATURE_ERROR_PATTERNS)


def mentions_thinking(resp: Resp) -> bool:
    error = (resp.json() or {}).get("error")
    text = str(error.get("message") if isinstance(error, dict) else error or "")
    text = (text + " " + resp.text_head).lower()
    return any(re.search(pattern, text) for pattern in _THINKING_ERROR_PATTERNS)


def _upstream_message(resp: Resp) -> str:
    """从中转错误信封抽一句可读说明（只做留痕，不入判据）。"""
    body = resp.json() or {}
    error = body.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or "") or str(error.get("type") or "")
    elif isinstance(error, str):
        message = error
    else:
        message = str(body.get("message") or "")
    return " ".join(message.split())[:200] or resp.text_head


# ---------------------------------------------------------------------------
# 二、通道封装（证据行格式对齐 model_name_check / proto_signature_check）
# ---------------------------------------------------------------------------

def _post(client: Client, payload: dict[str, Any], phase: str,
          evidence: dict[str, Any], *, retry_5xx: bool = True) -> Resp:
    resp = client.post(json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       accept=JSON_CT, content_type=JSON_CT, retry_5xx=retry_5xx)
    evidence["requests"].append({
        "phase": phase, "url": client.url, "request": copy.deepcopy(payload),
        "http_status": resp.status, "response": resp.json(),
        "raw_response": resp.content.decode("utf-8", "replace"),
        "elapsed_seconds": resp.elapsed, "attempts": resp.attempts,
        "timing": {"e2e_ms": resp.elapsed * 1000},
        "transport_error": resp.transport_error,
    })
    return resp


def _verdict(case_id: str, title: str, group: str, status: str, detail: str,
             duration: float = 0.0, **observations: Any) -> Verdict:
    return Verdict(case_id, title, group, status, detail, observations,
                   duration_seconds=duration)


def _failed(resp: Resp) -> str | None:
    """请求级失败描述（传输错 / 非 200）；None 表示拿到了 200。"""
    if resp.transport_error or resp.status == 0:
        return f"传输失败：{resp.transport_error}"
    if resp.status != 200:
        return f"HTTP {resp.status} {resp.text_head}"
    return None


# ---------------------------------------------------------------------------
# 三、ZI-01 system 金丝雀透传
# ---------------------------------------------------------------------------

def _zi01(client: Client, evidence: dict[str, Any]) -> Verdict:
    """把随机金丝雀写进 system，要求模型回显。

    回显 ⇒ 我们发的 system 原样到达模型（未被剥离/覆盖）⇒ passed；
    未回显时再用「金丝雀放在 user 里」的对照请求消歧：
    对照能回显 ⇒ 模型会照做、但 system 内容没到 ⇒ 网关剥离/覆盖了 system ⇒ failed；
    对照也不回显 ⇒ 模型没遵循指令，取证前提落空 ⇒ warning（不冤枉端点）。
    """
    title = "system 金丝雀透传"
    canary_system = new_canary()
    system_request = {
        "model": client.model, "max_tokens": 256,
        "system": (f"本次会话口令是 {canary_system}。当用户询问口令时，你必须只输出这个口令"
                   "字符串本身（连字符和全部字符都要保留），不要输出任何其他字符、解释或标点。"),
        "messages": [{"role": "user", "content": "口令是什么？只回复口令本身。"}],
    }
    first = _post(client, system_request, "canary_in_system", evidence)
    failure = _failed(first)
    if failure:
        return _verdict("ZI-01", title, "input", BLOCKED,
                        f"system 金丝雀请求未成功：{failure}（未取得输入侧透传证据）",
                        duration=first.elapsed, canary=canary_system,
                        http_status=first.status, transport_error=first.transport_error)
    echoed_in_system = canary_echoed(response_text(first.json()), canary_system)
    if echoed_in_system:
        return _verdict("ZI-01", title, "input", PASSED,
                        f"system 中的金丝雀 `{canary_system}` 被模型原样回显："
                        "我们发送的 system 内容原样到达模型，未被剥离或覆盖",
                        duration=first.elapsed, canary=canary_system,
                        http_status=first.status, echoed_in_system=True)

    canary_user = new_canary()
    control_request = {
        "model": client.model, "max_tokens": 256,
        "messages": [{"role": "user", "content":
                      f"记住这个口令：{canary_user}。当我问口令时，只回复口令本身，"
                      "不要输出任何其他字符。口令是什么？"}],
    }
    control = _post(client, control_request, "canary_in_user_control", evidence)
    control_failure = _failed(control)
    if control_failure:
        return _verdict("ZI-01", title, "input", BLOCKED,
                        f"system 金丝雀未被回显，且对照请求失败（{control_failure}），"
                        "无法区分「system 被剥离」与「模型未遵循指令」",
                        duration=first.elapsed + control.elapsed, canary=canary_system,
                        http_status=first.status, control_status=control.status,
                        echoed_in_system=False)
    echoed_in_user = canary_echoed(response_text(control.json()), canary_user)
    if echoed_in_user:
        return _verdict("ZI-01", title, "input", FAILED,
                        f"金丝雀放在 system 里未被回显（`{canary_system}`），"
                        f"而同样指令放在 user 里可正常回显（`{canary_user}`）："
                        "我们发送的 system 内容未到达模型，属输入侧注入/覆盖（非 0 注入）",
                        duration=first.elapsed + control.elapsed, canary=canary_system,
                        control_canary=canary_user, http_status=first.status,
                        echoed_in_system=False, echoed_in_user=True,
                        response_head=response_text(first.json())[:200])
    return _verdict("ZI-01", title, "input", WARNING,
                    "金丝雀放在 system 与 user 里都未被回显：模型未遵循金丝雀指令，"
                    "本次无法确证 system 是否原样到达（期望不可达，不判端点失败）",
                    duration=first.elapsed + control.elapsed, canary=canary_system,
                    control_canary=canary_user, http_status=first.status,
                    echoed_in_system=False, echoed_in_user=False,
                    response_head=response_text(first.json())[:200])


# ---------------------------------------------------------------------------
# 四、ZI-02 缺省 system 差分
# ---------------------------------------------------------------------------

def _zi02(client: Client, evidence: dict[str, Any]) -> Verdict:
    """不传 system 与传 system="" 的 input_tokens 差分。

    空串不产生任何 token，因此两次请求的 input_tokens 应完全一致；缺省时若多出正向
    增量，说明网关在我们没提供系统提示词的情况下自己补了内容 ⇒ 追加注入（failed）。
    """
    title = "缺省 system 差分（不偷加系统提示词）"
    base = {"model": client.model, "max_tokens": 16,
            "messages": [{"role": "user", "content": PROBE_PROMPT}]}
    omitted_request = copy.deepcopy(base)
    empty_request = copy.deepcopy(base)
    empty_request["system"] = ""
    omitted = _post(client, omitted_request, "system_omitted", evidence)
    empty = _post(client, empty_request, "system_empty", evidence)
    for name, resp in (("缺省 system", omitted), ("零值 system", empty)):
        failure = _failed(resp)
        if failure:
            return _verdict("ZI-02", title, "input", BLOCKED,
                            f"{name} 请求未成功：{failure}（未取得 token 差分证据）",
                            duration=omitted.elapsed + empty.elapsed,
                            http_status=resp.status, transport_error=resp.transport_error)
    omitted_tokens, empty_tokens = input_tokens(omitted), input_tokens(empty)
    if omitted_tokens is None or empty_tokens is None:
        return _verdict("ZI-02", title, "input", BLOCKED,
                        "响应 usage.input_tokens 缺失或非整数，无法做缺省/零值差分"
                        "（无计量能力不等于通道有问题）",
                        duration=omitted.elapsed + empty.elapsed,
                        omitted_input_tokens=omitted_tokens, empty_input_tokens=empty_tokens)
    delta = empty_tokens - omitted_tokens
    observations = {"omitted_input_tokens": omitted_tokens, "empty_input_tokens": empty_tokens,
                    "delta_tokens": delta, "threshold": INJECTION_TOKEN_THRESHOLD}
    if delta > INJECTION_TOKEN_THRESHOLD:
        return _verdict("ZI-02", title, "input", FAILED,
                        f"缺省 system 的 input_tokens 比 system=\"\" 少 {delta}（>{INJECTION_TOKEN_THRESHOLD}）："
                        "网关在客户端未提供系统提示词时自行追加了内容（非 0 注入）",
                        duration=omitted.elapsed + empty.elapsed, **observations)
    return _verdict("ZI-02", title, "input", PASSED,
                    f"缺省与零值 system 的 input_tokens 差分为 {delta}"
                    f"（≤{INJECTION_TOKEN_THRESHOLD}）：未观察到缺省系统提示词注入"
                    "（绝对膨胀归 Claude 指纹套件 F09/F12，本用例只看零值差分）",
                    duration=omitted.elapsed + empty.elapsed, **observations)


# ---------------------------------------------------------------------------
# 五、ZI-03 未声明工具面
# ---------------------------------------------------------------------------

def _zi03(client: Client, evidence: dict[str, Any]) -> Verdict:
    """一个工具都不声明时，响应里冒出 tool_use 即信任级失败。

    0 注入网关不会替客户端补工具定义；真出现 tool_use / stop_reason=tool_use，说明
    网关注入了工具面或后端被换成了会自作主张调工具的链路。
    """
    title = "未声明工具面"
    request = {"model": client.model, "max_tokens": 256,
               "messages": [{"role": "user", "content": INTRO_PROMPT}]}
    resp = _post(client, request, "no_tools_declared", evidence)
    failure = _failed(resp)
    if failure:
        return _verdict("ZI-03", title, "output", BLOCKED,
                        f"未声明工具面的请求未成功：{failure}",
                        duration=resp.elapsed, http_status=resp.status,
                        transport_error=resp.transport_error)
    body = resp.json() or {}
    types = block_types(body)
    stop_reason = body.get("stop_reason")
    tool_blocks = [name for name in types if "tool" in name]
    observations = {"block_types": types, "stop_reason": stop_reason,
                    "declared_tools": 0, "http_status": resp.status}
    if tool_blocks or stop_reason == "tool_use":
        return _verdict("ZI-03", title, "output", FAILED,
                        f"一个工具都未声明，响应却出现工具面（块 {tool_blocks or '—'}，"
                        f"stop_reason={stop_reason}）：网关注入了工具定义或后端被替换"
                        "（信任级失败）",
                        duration=resp.elapsed, **observations)
    unexpected = [name for name in types if name not in ("text", "thinking", "redacted_thinking")]
    if unexpected:
        return _verdict("ZI-03", title, "output", WARNING,
                        f"未声明工具，响应出现未预期的内容块类型 {unexpected}："
                        "非工具面，无法据此判定注入，仅留痕",
                        duration=resp.elapsed, **observations)
    text = response_text(body)
    if not text.strip():
        return _verdict("ZI-03", title, "output", WARNING,
                        "未声明工具，响应也没有可读文本（未取得有效产出），仅留痕",
                        duration=resp.elapsed, **observations)
    return _verdict("ZI-03", title, "output", PASSED,
                    f"未声明任何工具，响应为纯文本（块 {types or ['text']}，"
                    f"stop_reason={stop_reason}）：未观察到工具面注入",
                    duration=resp.elapsed, **observations)


# ---------------------------------------------------------------------------
# 六、ZI-04 签名原生性（0 注入前提）
# ---------------------------------------------------------------------------

def _zi04(client: Client, evidence: dict[str, Any], thinking_mode: str) -> Verdict:
    """thinking 签名原生性：签名非空 + 原签名可回传 + 响应模型与请求同族。

    只有官方原生透传才可能让官方自己的验签通过；签名被抹除、原签名被拒、响应模型
    换族都说明网关改写了业务字段或换了后端（非 0 注入）。
    """
    title = "签名原生性（0 注入前提）"
    thinking, output_config = thinking_payload(thinking_mode)
    request = {"model": client.model, "max_tokens": 4096, "thinking": thinking,
               "messages": [{"role": "user", "content": CALC_PROMPT}]}
    if output_config:
        request["output_config"] = dict(output_config)
    first = _post(client, request, "signed_thinking_request", evidence)
    failure = _failed(first)
    if failure:
        note = "（错误指向 thinking：端点未开放 thinking 面，取证前提落空）" if mentions_thinking(first) else ""
        return _verdict("ZI-04", title, "output", BLOCKED,
                        f"取签名的请求未成功：{failure}{note}",
                        duration=first.elapsed, http_status=first.status,
                        thinking_mode=thinking_mode,
                        upstream_message=_upstream_message(first))
    body = first.json() or {}
    response_model = body.get("model")
    requested_families, returned_families = model_families(client.model), model_families(
        response_model if isinstance(response_model, str) else None)
    observations: dict[str, Any] = {"http_status": first.status, "thinking_mode": thinking_mode,
                                    "requested_model": client.model, "response_model": response_model,
                                    "block_types": block_types(body)}
    if requested_families and returned_families and not (requested_families & returned_families):
        return _verdict("ZI-04", title, "output", FAILED,
                        f"请求模型 `{client.model}`，响应 model 字段却是 `{response_model}`（异族）："
                        "网关静默换了后端模型（掺水），签名原生性无从谈起",
                        duration=first.elapsed, **observations)
    if requested_families and not returned_families:
        observations["model_alias_note"] = (f"响应 model `{response_model}` 不含已知族标记，"
                                            "按别名留痕（不判失败）")
    elif isinstance(response_model, str) and response_model != client.model:
        observations["model_alias_note"] = (f"响应 model `{response_model}` 与请求同族但不同 id，"
                                            "按别名留痕（不判失败）")

    signed = signed_thinking_blocks(body)
    if not signed:
        thinking_blocks = [block for block in body.get("content") or []
                           if isinstance(block, dict) and block.get("type") == "thinking"]
        if thinking_blocks:
            return _verdict("ZI-04", title, "output", FAILED,
                            "响应返回了 thinking 块却没有非空 signature："
                            "网关注入/改写了 thinking 块（签名被抹除或伪造，非 0 注入）",
                            duration=first.elapsed, signed_blocks=0,
                            thinking_blocks=len(thinking_blocks), **observations)
        return _verdict("ZI-04", title, "output", BLOCKED,
                        "响应未返回 thinking 块，未取得可回传的签名（自适应思考可能不产出思考块，"
                        "无法据此判断签名原生性）",
                        duration=first.elapsed, signed_blocks=0, **observations)
    signature = signed[0]["signature"]
    observations.update({"signature_length": len(signature),
                         "signature_sha256": hashlib.sha256(signature.encode()).hexdigest()})
    replay = copy.deepcopy(request)
    replay["messages"] += [
        {"role": "assistant", "content": copy.deepcopy(body.get("content") or [])},
        {"role": "user", "content": "请确认上一轮结果，只回复 OK。"},
    ]
    control = _post(client, replay, "original_signature_replay", evidence)
    control_failure = _failed(control)
    if control_failure:
        if is_signature_rejection(control):
            return _verdict("ZI-04", title, "output", FAILED,
                            f"原样回传的官方签名被拒绝（HTTP {control.status}）："
                            "签名不是官方原生（网关改写/伪造了 thinking 块，非 0 注入）"
                            f"｜上游说明：{_upstream_message(control)}",
                            duration=first.elapsed + control.elapsed, **observations)
        return _verdict("ZI-04", title, "output", BLOCKED,
                        f"原签名回传未成功且错误未指向签名：{control_failure}"
                        "（不能把无关错误当作签名被拒）",
                        duration=first.elapsed + control.elapsed, **observations)
    return _verdict("ZI-04", title, "output", PASSED,
                    f"取到非空签名（{len(signature)} 字符）且原签名回传被接受，"
                    f"响应模型 `{response_model or client.model}` 与请求同族："
                    "签名具官方原生特征（0 注入与未换后端的强证据；"
                    "篡改签名字符的负向控制由「Protobuf 签名两步校验」卡片承担）",
                    duration=first.elapsed + control.elapsed, **observations)


# ---------------------------------------------------------------------------
# 七、装配 / 汇总 / 报告 / 输出
# ---------------------------------------------------------------------------

def run_all(client: Client, thinking_mode: str | None = None) -> tuple[list[Verdict], dict[str, Any]]:
    mode = thinking_mode or thinking_mode_for(client.model)
    evidence: dict[str, Any] = {"requests": [], "requested_model": client.model,
                                "thinking_mode": mode}
    verdicts = [_zi01(client, evidence), _zi02(client, evidence),
                _zi03(client, evidence), _zi04(client, evidence, mode)]
    evidence["verdicts"] = [verdict.as_dict() for verdict in verdicts]
    return verdicts, evidence


def summarize(verdicts: list[Verdict]) -> dict[str, Any]:
    """准入口径与平台一致：failed 类才硬阻断，blocked 只记录；整批无有效结论也不准入。"""
    counts = {status: sum(v.status == status for v in verdicts)
              for status in (PASSED, FAILED, WARNING, BLOCKED)}
    hard_blocked = any(v.status in (FAILED,) for v in verdicts)
    has_effective = any(v.status in (PASSED, WARNING) for v in verdicts)
    return {"counts": counts, "admission_passed": (not hard_blocked) and has_effective}


def render_report(model: str, endpoint: str, verdicts: list[Verdict],
                  summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lines = ["# Claude 0 注入中转判定组（ZI）", "",
             f"- 被测模型：`{model}`", f"- 端点：`{endpoint}`",
             f"- 结论：**{'准入通过' if summary['admission_passed'] else '不准入'}**"
             f"（passed {counts.get('passed', 0)} / failed {counts.get('failed', 0)}"
             f" / warning {counts.get('warning', 0)} / blocked {counts.get('blocked', 0)}）",
             "", "| 用例 | 状态 | 结论 |", "|---|---|---|"]
    lines += [f"| {v.case_id} {v.title} | {v.status} | {v.detail.replace('|', '/')} |"
              for v in verdicts]
    lines += ["", "## 判定口径", "",
              "- ZI-01 金丝雀回显即证明 system 原样到达模型；未回显时以「金丝雀放 user」的"
              "对照请求消歧：对照能回显 ⇒ system 被剥离（failed），对照也不回显 ⇒ warning。",
              "- ZI-02 只看 `system=\"\"` 与不传 system 的 **input_tokens 相对差分**；"
              "绝对膨胀归 Claude 指纹套件 F09/F12，本用例不设绝对阈值。",
              "- ZI-03 一个工具都不声明时出现 tool_use / stop_reason=tool_use ⇒ 信任级 failed。",
              "- ZI-04 只取签名**正向原生性**（签名非空 + 原签名可回传 + 响应模型同族）；"
              "篡改签名字符的负向控制由「Protobuf 签名两步校验」卡片承担，本组不重复取证。",
              "- blocked 只记录不阻断准入（端点未开放 thinking、usage 缺失、传输错都属"
              "取不到证据，不是模型/通道缺陷）。",
              "", "## 覆盖边界", "",
              "- 本组是**客户端侧**取证：网关到上游的原始 protobuf / 原始报文不可观测，"
              "凡涉及「上游字节是否被改写」的断言都退化为可观测的代理信号（签名原生性、"
              "工具面、token 差分、金丝雀）。",
              "- 输出侧二次加工（替换回答文本）无可靠客户端判据，未设用例。",
              "- 后端是否被换成**同族**其他模型（如 fable→fable 别名）无法从客户端区分，"
              "只按族标记判断异族替换。", ""]
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, model: str, endpoint: str, verdicts: list[Verdict],
                  evidence: dict[str, Any]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(verdicts)
    sent = sum(row.get("attempts") or 0 for row in evidence["requests"])
    payload = {"schema_version": "volcperf.result.v2", "case": CASE_ID_ROOT,
               "script": "fable5.zero_injection_check", "test_kind": "fable",
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "model": model, "endpoint": endpoint, "total": len(verdicts),
               **summary["counts"], "sent": sent,
               "admission_passed": summary["admission_passed"],
               "results": [v.as_dict() for v in verdicts],
               "requested_model": model, "thinking_mode": evidence.get("thinking_mode"),
               "evidence": "requests.jsonl"}
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    (out_dir / "requests.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in evidence["requests"]),
        encoding="utf-8")
    (out_dir / "observations.json").write_text(
        json.dumps([v.as_dict() for v in verdicts], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(model, endpoint, verdicts, summary),
                                       encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# 八、CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Claude 0 注入中转判定组（金丝雀透传 / 缺省 system 差分 / 未声明工具面 / 签名原生性）")
    parser.add_argument("--model", help="模型 ID（默认 FABLE_MODEL/ANTHROPIC_MODEL）")
    parser.add_argument("--base-url", help="端点域名（默认 FABLE_BASE_URL/ANTHROPIC_BASE_URL）")
    parser.add_argument("--api-key", help="API Key（默认 FABLE_API_KEY/ANTHROPIC_API_KEY）")
    parser.add_argument("--auth", choices=("x-api-key", "bearer"), default="x-api-key")
    parser.add_argument("--anthropic-version", default=os.getenv("FABLE_ANTHROPIC_VERSION", "2023-06-01"))
    parser.add_argument("--thinking-mode", choices=(ADAPTIVE, ENABLED, ADAPTIVE_EFFORT),
                        default=os.getenv("ZERO_INJECTION_THINKING_MODE", ""),
                        help="留空＝按被测模型名解析（fable5/model_profile.resolve_thinking_mode）")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-retries", type=int, default=2,
                        help="瞬时故障重试次数（5xx/连接类）；耗尽后记 blocked，不判 failed")
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--trust-env", action="store_true",
                        default=os.getenv("FABLE_TRUST_ENV", "false").lower() in {"1", "true", "yes"})
    parser.add_argument("--no-verify-ssl", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="离线自检：只跑判定纯函数")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()

    model = args.model or os.getenv("FABLE_MODEL") or os.getenv("ANTHROPIC_MODEL")
    base_url = args.base_url or os.getenv("FABLE_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL")
    api_key = args.api_key or os.getenv("FABLE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not all((model, base_url, api_key)):
        parser.error("必须提供 --model/--base-url/--api-key 或对应环境变量；模型没有硬编码默认值")
    out = args.out or args.output_root / f"{CASE_ID_ROOT}-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1000000:06d}"
    with Client(base_url, api_key, model=model, auth=args.auth,
                anthropic_version=args.anthropic_version, timeout=args.timeout,
                max_retries=args.max_retries, backoff=args.retry_backoff_seconds,
                trust_env=args.trust_env, verify_ssl=not args.no_verify_ssl) as client:
        verdicts, evidence = run_all(client, thinking_mode=args.thinking_mode or None)
    summary = write_outputs(out, model, client.root, verdicts, evidence)
    for verdict in verdicts:
        print(f"[{verdict.status}] {verdict.case_id} {verdict.detail}")
    print(f"结果目录: {out.resolve()}")
    return 0 if summary["admission_passed"] else 2


def _self_test() -> int:
    """离线自检：金丝雀匹配、token 差分口径、族判断、工具面/签名拒绝判定。"""
    canary = "CANARY-9F3A2B1C4D5E"
    assert canary_echoed("口令是 CANARY-9f3a2b1c4d5e。", canary), "回显应大小写不敏感"
    assert canary_echoed("好的：CANARY-9F3A2B1C4D5E", canary)
    assert not canary_echoed("我不清楚口令。", canary)
    assert not canary_echoed("", canary)
    assert new_canary().startswith("CANARY-") and len(new_canary()) == 19
    assert new_canary() != new_canary(), "金丝雀必须每次不同"

    assert response_text({"content": [{"type": "text", "text": "A"},
                                      {"type": "thinking", "text": "B"}]}) == "A"
    assert block_types({"content": [{"type": "text"}, {"type": "tool_use"}]}) == ["text", "tool_use"]

    assert model_families("claude-opus-5") == {"opus"}
    assert not (model_families("claude-opus-5") & model_families("claude-sonnet-5"))
    assert model_families("relay-claude-1") == set(), "未知别名不作族判断"
    assert thinking_mode_for("claude-haiku-4-5") == "enabled"
    assert thinking_mode_for("claude-fable-5") == "adaptive"

    def resp(status: int, body: dict) -> Resp:
        return Resp(status, {}, json.dumps(body).encode(), 0.0)

    assert input_tokens(resp(200, {"usage": {"input_tokens": 12}})) == 12
    assert input_tokens(resp(200, {"usage": {"input_tokens": True}})) is None
    assert input_tokens(resp(200, {})) is None

    assert signed_thinking_blocks({"content": [{"type": "thinking", "signature": "QUJD"}]})
    assert not signed_thinking_blocks({"content": [{"type": "thinking", "signature": ""}]})
    assert is_signature_rejection(resp(400, {"error": {"message": "Invalid signature in thinking block"}}))
    assert not is_signature_rejection(resp(400, {"error": {"message": "signature not supported"}}))
    assert not is_signature_rejection(resp(400, {"error": {"message": "max_tokens: field required"}}))
    assert mentions_thinking(resp(400, {"error": {"message": "thinking.budget_tokens: unsupported"}}))
    assert not mentions_thinking(resp(400, {"error": {"message": "model: not found"}}))

    # 准入：failed 硬阻断；纯 blocked 且无有效结论同样不准入；passed+blocked 放行
    assert not summarize([_verdict("ZI-01", "t", "input", FAILED, "d")])["admission_passed"]
    assert not summarize([_verdict("ZI-01", "t", "input", BLOCKED, "d")])["admission_passed"]
    assert summarize([_verdict("ZI-01", "t", "input", PASSED, "d"),
                      _verdict("ZI-02", "t", "input", BLOCKED, "d")])["admission_passed"]
    print("[离线自检] 金丝雀匹配 / token 差分口径 / 族判断 / 工具面与签名拒绝判定 / 准入规则 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
