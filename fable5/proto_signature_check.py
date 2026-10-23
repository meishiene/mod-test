#!/usr/bin/env python3
"""Two-step Claude signature check using a real response and its thinking signature.

Step 1 obtains a signed thinking response and confirms the original history is
accepted. Step 2 changes exactly one signature character and requires a specific
upstream signature rejection. The contract checks read the real Messages JSON
fields directly — no local Protobuf snapshot is written (2026-10-15, removed by
request: the snapshot was a lossy projection of the JSON with no consumer).

**Protobuf 在本套件里只出现一次**（2026-10-15 澄清 + 收窄，见
``docs/Claude-Protobuf签名两步校验.md``）：签名自身解码出的 protobuf
（``fable5/signature_structure.py``）—— **真正的校验对象**，合成 SIG-03：严格可解析
（伪造判据）/ 块类型与响应一致 / 标准 base64 重编码等价，外加标识一致性（签名明文里的
模型标识 vs 请求模型，一致 / 不一致记 warning / 无标识无结论）。明文不在签名里
（内层 ``#5`` 是密文），内层 12B/12B/48B 带密钥（8 哈希 × 7 段复算 0/20）
⇒ 只能读结构、不能验内容真伪。原始响应以 ``response.json`` 与 ``requests.jsonl`` 留档。

**准入只看两步回传契约**（SIG-01 + SIG-02）。SIG-03 是取证项，任一项不达标（含标识不一致）
只记 warning 留痕、不阻断准入：本地没有官方 schema，无法排除上游签名格式变更或别名写法，
不能据此判 failed。

Transient channel failures — connection errors, 5xx, and HTTP 200 bodies that are
not Messages JSON (observed as gateway-generated SSE shells such as
``msg_fallback_<timestamp>`` with empty content and zero usage) — are retried
``--max-retries`` times (default 2) and then recorded as ``blocked``, never
``failed``: a channel that never delivered a model answer says nothing about the
model. Only a response that *is* Messages JSON but violates the contract (model
mismatch, missing request id, empty text) is ``failed``.

The mutated-signature request is never retried on 5xx (a 5xx retried into 200
would be misread as "the tampered signature was accepted" ⇒ false failure), but it
may be resent after a pure connection error, where the verdict is unchanged.
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

from fable5.proto_probe import BLOCKED, FAILED, PASSED, WARNING, Client, Resp, Verdict
from fable5.model_profile import (ADAPTIVE, ADAPTIVE_EFFORT, ENABLED,
                                  resolve_thinking_mode, same_model_id,
                                  thinking_payload)
from fable5.signature_structure import (looks_like_model_id, model_id_compatible,
                                        read_signature, synthetic_signature)

CASE_ID_ROOT = "PROTOBUF_SIGNATURE_CHECK"
JSON_CT = "application/json"
CASE_TITLES = {1: "正常响应与原签名回传", 2: "修改一个签名字符后回传", 3: "签名内部结构层校验"}
# 准入唯二依据：两步回传契约。SIG-03 是结构层取证项，warning 不阻断。
STEP_CASE_IDS = ("SIG-01", "SIG-02")
BALL_PROMPT = (
    "有 12 个外观完全一样的小球，其中 1 个重量异常（不知道是更重还是更轻），"
    "只用无砝码天平，最多称量 3 次，请给出完整的称量方案，并且证明该方案可以保证找到异常小球，"
    "同时判断它偏重还是偏轻。"
)
QUICKSORT_PROMPT = "请解释快速排序的平均时间复杂度为什么是 O(n log n)，给出递推推导过程。"
SIGNATURE_PROMPTS = (("ball", BALL_PROMPT), ("quicksort", QUICKSORT_PROMPT))


def signature_prompts() -> tuple[tuple[str, str], ...]:
    """取签问题清单（第一个是主问题，其余为「首问没拿到签名」时的回退问题）。

    默认 ``SIGNATURE_PROMPTS``（ball → quicksort）。两道题都需足够难度：``adaptive``
    thinking 对简单问题会直接不思考（``thinking_tokens=0``），因而永远拿不到签名。
    需要「换一个能取到签名的问题」复查时，用环境变量 ``PROTO_SIGNATURE_PROMPTS``
    覆盖这一次运行的问题清单，不改代码：
    ``PROTO_SIGNATURE_PROMPTS='[{"name":"quicksort","text":"..."}]'``。
    """
    raw = os.getenv("PROTO_SIGNATURE_PROMPTS", "").strip()
    if not raw:
        return SIGNATURE_PROMPTS
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"PROTO_SIGNATURE_PROMPTS 不是合法 JSON：{exc}") from exc
    if not isinstance(data, list) or not data:
        raise SystemExit("PROTO_SIGNATURE_PROMPTS 必须是非空 JSON 数组")
    resolved: list[tuple[str, str]] = []
    for item in data:
        if (not isinstance(item, dict) or not isinstance(item.get("name"), str)
                or not item["name"].strip() or not isinstance(item.get("text"), str)
                or not item["text"].strip()):
            raise SystemExit("PROTO_SIGNATURE_PROMPTS 每项必须是含非空 name / text 的对象")
        resolved.append((item["name"], item["text"]))
    return tuple(resolved)


def is_haiku45_model(model: str | None) -> bool:
    """识别 Haiku 4.5 的稳定别名和带日期的快照模型名。

    保留兼容导入；**CLI 已不再用它决定 thinking 档位**——档位统一由
    ``fable5.model_profile.resolve_thinking_mode`` 按模型名解析（4.5 系不止 haiku 一个）。
    """
    value = str(model or "").strip()
    return value == "claude-haiku-4-5" or bool(
        re.fullmatch(r"claude-haiku-4-5-\d{8}", value)
    )


def validate_message(body: dict[str, Any], expected_model: str) -> dict[str, Any]:
    """直接校验真实 Messages JSON 的契约字段（不留本地 protobuf 快照）。

    只读响应里**真实返回**的字段，绝不用请求值补齐：request id 非空、响应模型等于当前
    MODEL、content 里存在非空文本块。契约不符 ⇒ 调用方记 ``failed``；包体根本不是
    Messages JSON 属通道侧异常，由调用方在更早一步记 ``blocked``。

    模型字段只做 **Bedrock 归一后**的相等比较（``anthropic.X-v1:0`` ↔ ``X``，同一模型的
    两种写法，见 ``fable5/model_profile.py``）：归一等价即通过，真正的换模型仍判失败。
    归一放行的写法由 ``model_field_note`` 取回，供调用方原样留痕。
    """
    if not isinstance(body, dict):
        raise ValueError("响应不是 JSON 对象")
    request_id = body.get("id") or body.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("缺少 request_id")
    if not same_model_id(body.get("model"), expected_model):
        raise ValueError(f"模型字段异常: {body.get('model')}，期望 {expected_model}")
    content = body.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("content 为空")
    texts = [block.get("text") for block in content
             if isinstance(block, dict) and block.get("type") == "text"]
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("文本内容为空")
    return body


def model_field_note(body: dict[str, Any], expected_model: str) -> str | None:
    """响应 model 是 Bedrock 写法时返回留痕说明，否则 None（写法一致无需留痕）。"""
    actual = body.get("model")
    if not isinstance(actual, str) or actual.strip().lower() == str(expected_model or "").strip().lower():
        return None
    if same_model_id(actual, expected_model):
        return f"响应 model 为上游写法 {actual}，归一等价于 {expected_model}（同一模型，非换模型）"
    return None


def mutate_signature(signature: str) -> tuple[str, int]:
    """Replace one interior alphanumeric character, preserving length and padding."""
    positions = [i for i, char in enumerate(signature)
                 if char.isascii() and char.isalnum()]
    if not positions:
        raise ValueError("签名没有可保持编码格式的修改位置")
    position = positions[len(positions) // 2]
    replacement = "B" if signature[position] == "A" else "A"
    return signature[:position] + replacement + signature[position + 1:], position


def is_signature_rejection(resp: Resp) -> bool:
    if resp.status not in (400, 422):
        return False
    body = resp.json() or {}
    error = body.get("error")
    if not isinstance(error, dict):
        return False
    # Do not count a generic 4xx, unsupported parameter or echoed invalid signature
    # as a successful negative control. The error must describe signature failure.
    # Gateways may format field names as Markdown, e.g. Invalid `signature`.
    # Normalize the decoration only; keep the original error in the evidence.
    message = str(error.get("message") or "").lower().replace("`", "")
    if any(word in message for word in ("unsupported", "not supported", "不支持", "not allowed")):
        return False
    patterns = (
        r"(?:invalid|incorrect|bad|malformed|corrupt(?:ed)?)\s+(?:thinking\s+)?signature",
        r"signature.{0,50}(?:invalid|incorrect|verification failed|validation failed|does not match)",
        r"(?:failed|unable|cannot|could not)\s+to\s+(?:verify|validate).{0,30}signature",
        r"(?:签名.{0,30}(?:无效|不合法|验证失败|校验失败|不匹配)|(?:无效|不合法).{0,10}签名)",
    )
    return any(re.search(pattern, message) for pattern in patterns)


def _post(client: Client, payload: dict[str, Any], phase: str,
          evidence: dict[str, Any], *, retry_5xx: bool = True,
          retry_transport: bool | None = None) -> Resp:
    resp = client.post(json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       accept=JSON_CT, content_type=JSON_CT,
                       retry_5xx=retry_5xx, retry_transport=retry_transport)
    evidence["requests"].append({
        "phase": phase, "url": client.url, "request": copy.deepcopy(payload),
        "http_status": resp.status, "response": resp.json(),
        "raw_response": resp.content.decode("utf-8", "replace"),
        "content_type": resp.content_type,
        "elapsed_seconds": resp.elapsed, "attempts": resp.attempts,
        "timing": {"e2e_ms": resp.elapsed * 1000},
        "transport_error": resp.transport_error,
    })
    return resp


def _post_messages_json(client: Client, payload: dict[str, Any], phase: str,
                        evidence: dict[str, Any],
                        ) -> tuple[Resp, dict[str, Any] | None, float, int]:
    """POST 并解码 Messages JSON，**只在「HTTP 200 但响应体不是 Messages JSON」时重试**。

    该形态属通道侧一次性异常（实测为网关自造的 SSE 空壳：``id=msg_fallback_<响应时刻
    时间戳>``、``content`` 空、``stop_reason`` 空、usage 全 0、耗时 20s+），既不代表模型
    能力，也不该据此判 failed ⇒ 与连接类故障同口径：按 ``client.max_retries`` 退避重试，
    耗尽后由调用方记 blocked。每次实际发出的请求都会在 evidence 里留一行。

    返回 ``(最后一次响应, 包体或 None, 累计耗时, 重试次数)``。
    """
    budget = max(0, int(getattr(client, "max_retries", 0) or 0))
    elapsed_total, resp = 0.0, None
    for index in range(budget + 1):
        resp = _post(client, payload, phase, evidence)
        elapsed_total += resp.elapsed
        body = resp.json()
        if body is not None or resp.transport_error or resp.status != 200:
            return resp, body, elapsed_total, index
        if index < budget:
            time.sleep(client.backoff * (2 ** index))
    return resp, None, elapsed_total, budget


def _verdict(step: int, status: str, detail: str, duration: float = 0.0,
             **observations: Any) -> Verdict:
    return Verdict(f"SIG-0{step}", CASE_TITLES[step], "response", status, detail, observations,
                   duration_seconds=duration)


def _model_consistency_note(entry: dict[str, Any]) -> str:
    """标识一致性的结论措辞：一致 / 不一致 / 无标识无结论，三档都要如实写。"""
    identifiers = [str(item) for item in (entry.get("model_identifiers") or [])]
    expected = str(entry.get("expected_model") or "")
    match = entry.get("model_match")
    if match is True:
        hit = next((text for text in identifiers if model_id_compatible(text, expected)),
                   identifiers[0] if identifiers else "")
        if hit and hit != expected:
            return f"签名内含模型标识 {hit}，与请求模型 {expected} 一致（按 `-` 边界视为同一型号）"
        return f"签名内含模型标识 {hit}，与请求模型 {expected} 一致"
    if match is False:
        return (f"签名内含模型标识 {'、'.join(identifiers)}，与请求模型 {expected} 不一致 "
                "⇒ 记 warning 留痕（不阻断准入）")
    return "签名内没有模型标识明文，标识一致性无结论（实测 fable 系列签名即如此，不构成失败）"


def _structure_verdict(evidence: dict[str, Any]) -> Verdict:
    """SIG-03：读签名内部结构（base64 → protobuf → 结构三项 + 标识一致性）。

    取证项，**不参与准入**：本地无官方 schema，无法断言签名版式，因此任一项不达标只记
    ``warning`` 留痕（跨中转一旦复现即「疑为网关自造签名，或上游换了签名格式 / 换了模型」），
    不判 ``failed``。无签名可读时记 ``blocked``。本用例不额外发请求。
    """
    entry = evidence.get("signature_structure")
    if not entry:
        return _verdict(3, BLOCKED, "未取得签名，无法读取内部结构")
    structure_problems = entry.get("failures") or []
    model_problems = _model_consistency_problems(entry)
    note = _model_consistency_note(entry)
    observations = {key: value for key, value in entry.items()
                    if key not in ("readable_strings", "failures")}
    observations["structure_failures"] = structure_problems
    observations["model_consistency_failures"] = model_problems
    if not structure_problems and not model_problems:
        return _verdict(
            3, PASSED,
            f"签名解码为可严格解析的 protobuf（{entry['byte_length']} 字节，版本 "
            f"{entry['version']}，块类型 {entry['block_type']}），标准 base64 重编码与原签名"
            f"逐字符相同；{note}",
            **observations)
    return _verdict(
        3, WARNING,
        ("签名结构层核对未通过：" + "；".join(structure_problems) if structure_problems
         else "结构层三项核对通过，但标识一致性未通过：" + "；".join(model_problems))
        + "（属取证项：本地无官方 schema，无法排除上游版式变更或网关改写 ⇒ 记 warning 留痕，"
          "不判 failed、不阻断准入）",
        **observations)


def _model_consistency_problems(entry: dict[str, Any]) -> list[str]:
    """标识一致性核对项：只在签名**确实写了**模型标识且与请求模型不兼容时给出。"""
    if entry.get("model_match") is False:
        identifiers = "、".join(str(item) for item in (entry.get("model_identifiers") or []))
        return [f"签名内模型标识 {identifiers} 与请求模型 "
                f"{entry.get('expected_model')!r} 不一致"]
    return []


def _stop(status: str, detail: str, evidence: dict[str, Any],
          duration: float = 0.0, **obs: Any):
    return [_verdict(1, status, detail, duration=duration, **obs),
            _verdict(2, BLOCKED, "未取得可用于验签的真实响应，未发送变异请求"),
            _structure_verdict(evidence)], evidence


def run_all(client: Client, thinking_mode: str = "adaptive", *,
            allow_prompt_fallback: bool = True,
            max_tokens: int = 4096) -> tuple[list[Verdict], dict[str, Any]]:
    # 档位 → 线上片段由 model_profile 统一给出（enabled / adaptive / adaptive_effort）。
    thinking, output_config = thinking_payload(thinking_mode)
    evidence: dict[str, Any] = {"requests": [], "thinking_mode": thinking_mode,
                                "thinking": thinking, "max_tokens": max_tokens,
                                "output_config": output_config}
    original_request: dict[str, Any] | None = None
    body: dict[str, Any] | None = None
    first: Resp | None = None
    selected_prompt_name = ""
    index = -1
    block: dict[str, Any] | None = None
    signature = ""
    acquisition_elapsed = 0.0

    prompts = signature_prompts()
    for attempt, (prompt_name, prompt) in enumerate(prompts):
        request = {"model": client.model, "max_tokens": max_tokens, "thinking": thinking,
                   "messages": [{"role": "user", "content": prompt}]}
        if output_config:
            request["output_config"] = dict(output_config)
        phase = "initial" if attempt == 0 else f"fallback_{prompt_name}"
        response, candidate, elapsed, shell_retries = _post_messages_json(
            client, request, phase, evidence)
        acquisition_elapsed += elapsed
        if response.transport_error or response.status != 200:
            return _stop(BLOCKED, f"{prompt_name} 问题请求未成功：HTTP {response.status} "
                         f"{response.transport_error or response.text_head}", evidence,
                         duration=elapsed, http_status=response.status,
                         prompt=prompt_name)
        if candidate is None:
            return _stop(BLOCKED,
                         f"{prompt_name} 问题 HTTP 200 但响应体不是 Messages JSON 对象"
                         f"（已重试 {shell_retries} 次）：{response.text_head}；"
                         "通道侧异常，不据此判定模型能力", evidence, duration=elapsed,
                         prompt=prompt_name, http_status=response.status,
                         content_type=response.content_type,
                         raw_response_head=response.text_head, shell_retries=shell_retries)
        try:
            validate_message(candidate, client.model)
        except ValueError as exc:
            return _stop(FAILED, str(exc), evidence, duration=elapsed,
                         expected_model=client.model, response_model=candidate.get("model"),
                         prompt=prompt_name)
        thinking_blocks = [(i, item) for i, item in enumerate(candidate["content"])
                           if item.get("type") == "thinking"
                           and isinstance(item.get("signature"), str)
                           and item["signature"]]
        if not thinking_blocks:
            if attempt == 0 and allow_prompt_fallback:
                continue
            detail = (f"{len(prompts)} 个问题的正常响应都未返回非空 thinking.signature，无法进行签名校验"
                      if allow_prompt_fallback else
                      "正常响应未返回非空 thinking.signature，无法进行签名校验")
            return _stop(BLOCKED, detail,
                         evidence, duration=elapsed, prompt=prompt_name)
        candidate_index, candidate_block = thinking_blocks[0]
        if not isinstance(candidate_block.get("thinking"), str):
            return _stop(FAILED, f"{prompt_name} 问题的已签名 thinking 块缺少字符串 thinking 内容",
                         evidence, duration=elapsed, prompt=prompt_name)
        first = response
        original_request = request
        body = candidate
        selected_prompt_name = prompt_name
        index = candidate_index
        block = candidate_block
        signature = candidate_block["signature"]
        evidence["original_response"] = candidate
        evidence["signature_prompt"] = {"name": prompt_name, "text": prompt}
        # 读签名：解码 → 严格解析 protobuf → 结构层三项核对（不发请求，见 SIG-03）。
        structure = read_signature(signature, expected_model=client.model,
                                   expected_block_type=candidate_block.get("type"))
        evidence["signature_structure"] = {**structure.as_dict(),
                                          "failures": structure.failures()}
        break

    if first is None or original_request is None or body is None or block is None or index < 0:
        return _stop(BLOCKED, "未取得可用于验签的真实响应", evidence)

    try:
        changed_signature, position = mutate_signature(signature)
    except ValueError as exc:
        return _stop(BLOCKED, str(exc), evidence, duration=first.elapsed, prompt=selected_prompt_name)
    replay = copy.deepcopy(original_request)
    replay["messages"] += [
        {"role": "assistant", "content": copy.deepcopy(body["content"])},
        {"role": "user", "content": "请确认上一轮结果，只回复 OK。"},
    ]
    control, control_body, control_elapsed, control_shell_retries = _post_messages_json(
        client, replay, "original_signature_replay", evidence)
    if control.transport_error or control.status != 200:
        return _stop(BLOCKED, f"原签名回传未被接受：HTTP {control.status} {control.transport_error or control.text_head}；"
                     "不能据此判断变异签名是否被验证", evidence, duration=control_elapsed)
    if control_body is None:
        return _stop(BLOCKED,
                     "原签名回传 HTTP 200 但响应体不是 Messages JSON 对象"
                     f"（已重试 {control_shell_retries} 次）：{control.text_head}；"
                     "通道侧异常，不能据此判断变异签名是否被验证", evidence,
                     duration=control_elapsed, http_status=control.status,
                     content_type=control.content_type,
                     raw_response_head=control.text_head, shell_retries=control_shell_retries)
    try:
        validate_message(control_body, client.model)
    except ValueError as exc:
        return _stop(FAILED, f"原签名回传响应异常：{exc}", evidence, duration=control_elapsed)
    # Bedrock 会把响应 model 写成 anthropic.X-v1:0：归一等价即同一模型，通过但必须留痕，
    # 免得后来人把上游写法差异读成"字段异常"。
    alias_note = model_field_note(body, client.model)
    positive_detail = f"{selected_prompt_name} 问题取得真实签名，字段校验通过，原签名回传正常返回"
    if alias_note:
        positive_detail += f"（{alias_note}）"
    positive_obs: dict[str, Any] = {"expected_model": client.model, "response_model": body["model"],
                                    "request_id": body.get("id") or body.get("request_id"),
                                    "signature_sha256": hashlib.sha256(signature.encode()).hexdigest(),
                                    "signature_length": len(signature)}
    if alias_note:
        positive_obs["model_field_alias"] = alias_note
    positive = _verdict(1, PASSED, positive_detail,
                        duration=acquisition_elapsed + control_elapsed,
                        **positive_obs)
    mutated_request = copy.deepcopy(replay)
    mutated_request["messages"][1]["content"][index]["signature"] = changed_signature
    evidence["mutation"] = {"field": f"content[{index}].signature", "character_offset": position,
                            "before": signature[position], "after": changed_signature[position],
                            "changed_characters": 1, "model_unchanged": True}
    # 负向变异请求一次定型：5xx 绝不重试（重试成 200 会被读成「签名被接受」⇒ 假失败），
    # 但纯连接错可以重发（结果语义不变，断开只是没拿到答案）。
    negative_resp = _post(client, mutated_request, "mutated_signature_replay", evidence,
                          retry_5xx=False, retry_transport=True)
    # 变异签名自己也读一遍：用来如实展示「签名能解析 ≠ 签名有效」——改一个字符后多半仍能
    # 被解析成 protobuf，但上游照样拒绝。原先是硬编码 True，现在是观测值。
    mutated_structure = read_signature(changed_signature,
                                       expected_block_type=candidate_block.get("type"))
    obs = {**evidence["mutation"], "http_status": negative_resp.status,
           "error": (negative_resp.json() or {}).get("error"),
           "mutated_signature_still_parseable": mutated_structure.strictly_parseable}
    if is_signature_rejection(negative_resp):
        negative = _verdict(2, PASSED, "只修改一个签名字符，接口明确拒绝无效签名",
                            duration=negative_resp.elapsed, **obs)
    elif negative_resp.status == 200 and not negative_resp.transport_error:
        negative = _verdict(2, FAILED, "修改后的签名仍被 HTTP 200 接受，未满足签名拒绝要求",
                            duration=negative_resp.elapsed, **obs)
    else:
        negative = _verdict(2, BLOCKED, "未取得明确的签名无效错误，不能把其他接口错误当作验签成功",
                            duration=negative_resp.elapsed, **obs)
    return [positive, negative, _structure_verdict(evidence)], evidence


def _cell(value: Any) -> str:
    """markdown 表格单元格：竖线与反引号会破坏表格，统一替换。"""
    return str(value).replace("|", "/").replace("`", "'")


# 签名内块描述符字段号的实测含义（``scripts/probe_signature_structure.py --phase structure``
# 可复现，覆盖 results/ 与 .pytest-tmp/ 里全部真实签名）。只作留痕对照，不作断言：
# 本地没有官方 schema，字段号含义属观察，不是契约。
DESCRIPTOR_FIELD_ROLES = {
    "6": "块标识字段（实测为模型标识）",
    "8": "块类型描述符",
    "11": "会话 / 请求 UUID",
    "19": "嵌套字段（实测 `#19.1` 为模型标识）",
    "27": "嵌套字段（实测 `#27.1` 为 `prefix_before_block_v1`）",
}
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                      r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def _plaintext_role(path: str, text: str, expected_model: str) -> str:
    """给一段明文标出它在签名里的位置与归因。

    路径形态 ``.2.1.<描述符字段号>[.<子字段号>]``：``.2`` 顶层内层 message、``.1`` 块描述符。
    单段归因与 SIG-03 的整体结论同源，都走 ``signature_structure`` 的
    ``looks_like_model_id`` / ``model_id_compatible``（唯一权威实现，避免两处口径漂移）。
    """
    parts = [part for part in path.split(".") if part]
    field = parts[2] if len(parts) > 2 else ""
    role = DESCRIPTOR_FIELD_ROLES.get(field, "块描述符字段" if field else "未知层级")
    if len(parts) > 3 and field not in DESCRIPTOR_FIELD_ROLES:
        role += f" → 子字段 `#{parts[3]}`"
    if expected_model and text == expected_model:
        return f"{role}；与请求模型一致"
    if model_id_compatible(text, expected_model):
        return (f"{role}；与请求模型一致（`{text}` 是 `{expected_model}` 的别名/版本写法，"
                "按 `-` 边界视为同一型号）")
    if _UUID_RE.fullmatch(text):
        return role if "UUID" in role else f"{role}（UUID 形态）"
    if looks_like_model_id(text):
        return (f"{role}；与请求模型 `{expected_model}` 不一致 ⇒ 记 warning 留痕"
                "（标识一致性属取证项，不阻断准入）")
    return role


def _plaintext_lines(entry: dict[str, Any], expected_model: str) -> list[str]:
    """报告里的「签名解码明文」段：把 base64 解码后能读到的明文如实列出。

    只有未被加密的 length-delimited 字段能读出明文；``thinking`` 正文在内层 ``#5`` 密文里，
    签名**不能**用于核对响应正文或请求侧注入，因此这里只做版式留痕与跨中转对照。
    """
    lines = ["## 签名解码明文", ""]
    if not entry:
        return lines + ["未取得签名，没有可解码的明文内容。", ""]
    strings = [item for item in (entry.get("readable_strings") or []) if isinstance(item, dict)]
    byte_length = entry.get("byte_length")
    cipher = entry.get("cipher_length")
    if not strings:
        lines.append(
            f"签名 base64 解码后为 {byte_length} 字节 protobuf，但没有可读明文段："
            "块类型、模型标识等标记都不在明文里，只有内层密文可读。")
        return lines + ["", "> 无明文属正常形态，不构成失败；明文有无只作跨中转对照。", ""]
    lines.append(
        f"签名 base64 解码后为 {byte_length} 字节 protobuf；未加密的 length-delimited 字段"
        f"可直接读出明文，本份签名共读出 **{len(strings)}** 段"
        + (f"（`thinking` 正文所在的内层 `#5` 是 {cipher} 字节密文，不可读）" if cipher else "")
        + "：")
    lines += ["", "| 字段路径 | 明文内容 | 说明 |", "|---|---|---|"]
    for item in strings:
        path = str(item.get("path") or "")
        text = str(item.get("text") or "")
        # 路径与明文是外部数据要转义；说明列由本模块生成，反引号是有意保留的强调。
        lines.append(f"| `{_cell(path)}` | `{_cell(text)}` | "
                     f"{_plaintext_role(path, text, expected_model)} |")
    lines += [
        "",
        "字段路径：`.2` = 顶层第 2 字段（内层 message），`.1` = 内层第 1 字段（块描述符），"
        "其后是块描述符内的字段号链。",
        "",
        "> 明文段只作版式留痕与跨中转对照：模型标识逐段标出与请求模型是否一致，"
        "整体结论见 SIG-03（不一致记 warning 留痕、不阻断准入；签名里没有标识则写「无结论」）。"
        "内层 `#2`/`#3` 与 `#4` 带密钥（8 种哈希 × 7 种候选段复算 0/20 命中），"
        "`thinking` 正文不在签名里 ⇒ 签名只能读结构、不能验内容真伪。",
        "",
    ]
    return lines


def write_outputs(out_dir: Path, model: str, endpoint: str, verdicts: list[Verdict],
                  evidence: dict[str, Any], *, case_id: str = CASE_ID_ROOT,
                  script: str = "fable5.proto_signature_check", test_kind: str = "fable",
                  provider: str = "Claude") -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {status: sum(v.status == status for v in verdicts)
              for status in (PASSED, FAILED, WARNING, BLOCKED)}
    # 准入只看两步回传契约（SIG-01 + SIG-02）。SIG-03 是结构层取证项：本地无官方 schema，
    # 无法断言签名版式，故 warning 只留痕不阻断（与平台「warning 只记录」口径一致）。
    steps = [v for v in verdicts if v.case_id in STEP_CASE_IDS]
    passed = len(steps) == len(STEP_CASE_IDS) and all(v.status == PASSED for v in steps)
    summary = {"schema_version": "volcperf.result.v2", "case": case_id,
               "script": script, "test_kind": test_kind,
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "model": model, "endpoint": endpoint, "total": len(verdicts), **counts,
               "sent": sum(r["attempts"] for r in evidence["requests"]),
               "admission_passed": passed, "results": [v.as_dict() for v in verdicts],
               "thinking_mode": evidence.get("thinking_mode"),
               "max_tokens": evidence.get("max_tokens"),
               "signature_prompt": evidence.get("signature_prompt"),
               "evidence": "requests.jsonl"}
    if "signature_structure" in evidence:
        summary["signature_structure_file"] = "signature_structure.json"
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "requests.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in evidence["requests"]))
    # 响应不再落本地 protobuf 快照：原始 Messages JSON 直接留档，避免"快照是 JSON 的
    # 更少投影、却无消费方"的双份证据（2026-10-15 按需求移除 response.pb / mutated_response.pb）。
    for key, filename in (("original_response", "response.json"), ("mutation", "mutation.json"),
                          ("signature_structure", "signature_structure.json")):
        if key in evidence:
            (out_dir / filename).write_text(json.dumps(evidence[key], ensure_ascii=False, indent=2) + "\n")
    prompt_name = (evidence.get("signature_prompt") or {}).get("name")
    prompt_text = (evidence.get("signature_prompt") or {}).get("text")
    prompt_line = f"- 取签问题：`{prompt_name}`" if prompt_name else "- 取签问题：未取得签名"
    structure = evidence.get("signature_structure") or {}
    lines = [f"# {provider} Protobuf 签名两步校验", "", f"- 被测模型：`{model}`",
             f"- 端点：`{endpoint}`",
             f"- 请求形态：thinking=`{evidence.get('thinking_mode')}`，"
             f"max_tokens={evidence.get('max_tokens')}", prompt_line]
    if prompt_text and prompt_name:
        lines.append(f"- 取签问题内容：{prompt_text}")
    if structure:
        lines.append(
            f"- 签名结构层：{structure.get('byte_length')} 字节 protobuf，顶层字段号 "
            f"{structure.get('top_fields')}，版本 {structure.get('version')}，块类型 "
            f"{structure.get('block_type')}，密文 {structure.get('cipher_length')} 字节，"
            f"标准 base64 重编码等价={structure.get('reencoded_equal')}")
    lines += [f"- 结论：{'通过' if passed else '未通过（失败或无法验证）'}",
              f"- 实际请求数：{summary['sent']}（原签名回传对照属于第一步；SIG-03 结构层核对不发请求）",
              "", "| 用例 | 状态 | 说明 |", "|---|---|---|"]
    lines += [f"| {v.case_id} {v.title} | {v.status} | {v.detail.replace('|', '/')} |" for v in verdicts]
    lines += [""] + _plaintext_lines(structure, model)
    lines += ["",
              "Protobuf 在本套件里只出现一次：签名自身解码出的 protobuf（signature_structure.json，"
              "承担 SIG-03 结构层与标识一致性核对）。响应不再落本地 protobuf 快照——"
              "契约校验直接读真实 Messages JSON 字段，原始响应以 response.json / requests.jsonl 留档"
              "（response.pb / mutated_response.pb 已于 2026-10-15 移除）。",
              "准入只看 SIG-01/SIG-02 两步回传契约。签名校验由回传接口执行；"
              "签名内部只能核对版式、块类型与可解析性，不能验内容真伪（内层字段带密钥），"
              "修改签名后仍可能被解析。",
              "签名里的模型标识与请求模型的一致性是取证项：不一致记 warning 留痕（不阻断准入），"
              "签名里没有标识则写「无结论」——不构成失败，也不用于断言底层身份。",
              "签名摘要仅用于证据关联；本测试不证明指定模型的底层身份。", ""]
    (out_dir / "report.md").write_text("\n".join(lines))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Claude Protobuf 签名两步校验 + 签名内部结构层核对（不修改模型名）")
    parser.add_argument("--model", default=os.getenv("FABLE_MODEL") or os.getenv("ANTHROPIC_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("FABLE_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("FABLE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    parser.add_argument("--thinking-mode", choices=(ADAPTIVE, ENABLED, ADAPTIVE_EFFORT),
                        default=None,
                        help="留空＝按被测模型名解析（fable5/model_profile.resolve_thinking_mode）："
                             "4.5/4.6 系 enabled、5 系 adaptive_effort、其余 adaptive")
    # 输出预算是**上限**不是花费：给足不会多计费，给少了思考会把预算吃满、正文为零，
    # 被契约读成「文本内容为空」⇒ 假 failed（2026-10-20 实测 sonnet-4-6 在 4096 下必然如此，
    # 16000 下正常返回 thinking+text+signature）。故默认取 16000，可显式覆盖。
    parser.add_argument("--max-tokens", type=int,
                        default=int(os.getenv("PROTO_SIGNATURE_MAX_TOKENS", "16000")))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-retries", type=int, default=2,
                        help="瞬时故障重试次数（5xx / 连接类 / HTTP 200 但响应体不是 Messages JSON）；"
                             "耗尽后记 blocked，不判 failed。负向变异请求固定不重试 5xx")
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--auth", choices=("x-api-key", "bearer"), default="x-api-key")
    parser.add_argument("--anthropic-version", default=os.getenv("FABLE_ANTHROPIC_VERSION", "2023-06-01"))
    parser.add_argument("--trust-env", action="store_true",
                        default=os.getenv("FABLE_TRUST_ENV", "false").lower() in {"1", "true", "yes"})
    parser.add_argument("--no-verify-ssl", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        good = {"id": "self-test-id", "model": "self-test-model",
                "content": [{"type": "thinking", "thinking": "更深的推理过程", "signature": "QUJDRA=="},
                            {"type": "text", "text": "OK"}]}
        validate_message(good, "self-test-model")
        for broken in ({"id": "", "model": "self-test-model",
                        "content": [{"type": "text", "text": "OK"}]},
                       {"id": "x", "model": "other-model",
                        "content": [{"type": "text", "text": "OK"}]},
                       {"id": "x", "model": "self-test-model", "content": []},
                       {"id": "x", "model": "self-test-model",
                        "content": [{"type": "text", "text": "   "}]}):
            try:
                validate_message(broken, "self-test-model")
            except ValueError:
                pass
            else:
                raise SystemExit(f"离线自检失败：非法响应未被拦下 {broken}")
        changed, _ = mutate_signature("QUJDREVGR0hJSktMTU5PUA==")
        assert changed and len(changed) == len("QUJDREVGR0hJSktMTU5PUA==")
        assert is_signature_rejection(Resp(400, {}, b'{"error":{"message":"Invalid signature in thinking block"}}', 0))
        assert not is_signature_rejection(Resp(400, {}, b'{"error":{"message":"signature not supported"}}', 0))
        # 签名结构层：合成样本必须三项全过；块类型不符/非法载荷必须落成「问题」而不是异常。
        good = read_signature(synthetic_signature(), expected_model="self-test-model",
                              expected_block_type="thinking")
        assert good.strictly_parseable and good.block_type_matches and good.reencoded_equal
        assert not good.failures()
        mismatched = read_signature(synthetic_signature(block_type="text"),
                                    expected_block_type="thinking")
        assert mismatched.strictly_parseable and mismatched.failures()
        assert read_signature("QUJDREVGR0hJSktMTU5PUA==").failures()
        assert read_signature("").failures() == ["未提供签名"]
        print("离线自检通过：Messages 响应契约校验（无本地快照），签名修改一个字符，"
              "签名错误与其他错误分离；签名结构层可核对版式/块类型/重编码等价/标识一致性")
        return 0
    if not all((args.model, args.base_url, args.api_key)):
        parser.error("必须提供 --model/--base-url/--api-key 或对应环境变量；模型没有硬编码默认值")
    if args.thinking_mode is None:
        # 未显式指定（也没走 --thinking-mode）：按模型名解析，替代原先"只特判 haiku"的做法。
        env_mode = (os.getenv("PROTO_SIGNATURE_THINKING_MODE") or "").strip().lower()
        args.thinking_mode = (env_mode if env_mode in {ADAPTIVE, ENABLED, ADAPTIVE_EFFORT}
                              else resolve_thinking_mode(args.model))
    out = args.out or args.output_root / f"{CASE_ID_ROOT}-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1000000:06d}"
    with Client(args.base_url, args.api_key, model=args.model, auth=args.auth,
                anthropic_version=args.anthropic_version, timeout=args.timeout,
                max_retries=args.max_retries, backoff=args.retry_backoff_seconds,
                trust_env=args.trust_env, verify_ssl=not args.no_verify_ssl) as client:
        verdicts, evidence = run_all(client, thinking_mode=args.thinking_mode,
                                     max_tokens=args.max_tokens)
    summary = write_outputs(out, args.model, client.root, verdicts, evidence)
    for verdict in verdicts:
        print(f"[{verdict.status}] {verdict.case_id} {verdict.detail}")
    print(f"结果目录: {out.resolve()}")
    return 0 if summary["admission_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
