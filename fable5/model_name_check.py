#!/usr/bin/env python3
"""Claude 模型名称两步校验（MN 组两条用例）。

参照 ``proto_signature_check``（签名两步校验）的结构：**第一步取证、第二步变异**。

- **MN-01 正常模型名请求**：用当前被测模型名发一轮 Messages 请求，HTTP 200 且
  响应是合法 Messages JSON ⇒ ``passed``（证明通道与模型名可用，是第二步的前提）。
- **MN-02 修改一个模型名字符后请求**：把模型名**中间位置的一个字符**换成原名里
  不存在的字符（保持长度，如 ``claude-fable-5-1`` → ``claude-qable-5-1``）后再发
  同一请求，要求端点**明确报错**（模型不存在 / 无效 / 未知）。

判定纪律（通道与判定对象见 ``fable5/proto_probe.py``）：

- **明确拒绝**（错误码 ``model_not_found`` / ``invalid_model``——**``error.code`` 与
  ``error.type`` 都要看**，New API 的真实码在 ``code``、``type`` 只是 ``new_api_error``；
  或 4xx 且错误文案指向模型）⇒ ``passed``；
- **HTTP 200 放行** ⇒ 该拒未拒，记 ``failed``（端点未校验模型名）。若响应里的
  ``model`` 字段与请求的变异名不同，说明网关**静默改写/映射**到别的模型，一并留痕；
- **其他 4xx/5xx/传输错** ⇒ ``blocked``（不能把无关的接口错误当成"模型名校验成功"）；
- **变异无效**（改动后的名称恰好出现在端点 ``/v1/models`` 声明列表里）⇒ ``blocked``，
  取证前提落空，不冤枉端点；
- 第二步使用 ``retry_5xx=False``：瞬时 5xx 若被重试成 200，会把"该拒未拒"误判成
  failed（假失败），必须一次定型。

用法::

    .venv/bin/python -m fable5.model_name_check --self-test                 # 离线自检
    .venv/bin/python -m fable5.model_name_check --model claude-fable-5-1 \\
        --base-url https://changhuai.ai --api-key sk-xxx
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fable5.proto_probe import (  # noqa: E402 —— 复用通道与判定对象（唯一权威实现）
    BLOCKED, FAILED, JSON_CT, PASSED, Client, Resp, Verdict,
)

CASE_ID_ROOT = "MODEL_NAME_CHECK"
PROMPT = "请只回复 OK 两个字母。"
# 错误信封的类型标记：命中即认定为「确定性模型名校验拒绝」（与状态码无关）。
MODEL_TYPE_MARKERS = ("model_not_found", "invalid_model")
# 文案正则：必须指向模型，泛化的 400 不算校验成功（同签名校验的既有口径）。
_MODEL_MESSAGE_PATTERNS = (
    r"\bmodel\b.{0,40}(?:not\s+(?:found|exist|supported|available)|invalid|unknown"
    r"|does\s+not\s+exist|is\s+not\s+found|unrecognized)",
    r"(?:invalid|unknown|unsupported|nonexistent|unavailable|unrecognized)\s+model\b",
    r"(?:模型|模型名|模型名称).{0,20}(?:不存在|无效|不合法|未知|不支持|有误)",
    r"(?:不存在|无效|未知|不合法|不支持).{0,10}模型",
)


# ---------------------------------------------------------------------------
# 一、模型名变异（确定性、可复现，不去撞真实模型名）
# ---------------------------------------------------------------------------

def mutate_model_name(name: str) -> tuple[str, int]:
    """把模型名中间位置的一个字母换成原名里没有的字母（长度不变）。

    只替换**中间位置**（不含首尾字符），替换字符取与原名完全不重复的字母，
    尽量降低"改出来的名字恰好是另一个真实模型"的概率；即便如此，第二步仍会先看
    端点 ``/v1/models`` 声明列表，命中即判变异无效（blocked），不冤枉端点。
    """
    positions = [i for i, char in enumerate(name)
                 if char.isascii() and char.isalnum() and 0 < i < len(name) - 1]
    if not positions:
        raise ValueError("模型名没有可保持长度的中间修改位置")
    position = positions[len(positions) // 2]
    original = name[position]
    lowered = name.lower()
    replacement = next((char for char in "qzxwvjk" if char not in lowered), None)
    if replacement is None:
        raise ValueError("模型名占满可用替换字母，无法构造无冲突变异")
    return name[:position] + replacement + name[position + 1:], position


def error_envelope(resp: Resp) -> tuple[str, str, str]:
    """抽出错误信封的 (type, code, message)。

    **必须同时看 ``code``**：New API / One API 一类中转把真正的错误码放在 ``error.code``
    （如 ``model_not_found``），而 ``error.type`` 只是笼统的 ``new_api_error``——
    2026-10-13 实批（180.178.56.190，claude-fable-5）就因只看 type 把明确的
    「未知模型」拒绝误判成 blocked。
    """
    body = resp.json() or {}
    error = body.get("error")
    if isinstance(error, dict):
        return (str(error.get("type") or ""), str(error.get("code") or ""),
                str(error.get("message") or ""))
    if isinstance(error, str):
        return "", "", error
    return "", "", str(body.get("message") or "")


def is_model_rejection(resp: Resp, requested_model: str | None = None) -> bool:
    """是否构成本用例要的**明确模型名拒绝**（泛化 4xx/限流/瞬时错都不算）。"""
    if resp.transport_error:
        return False
    error_type, error_code, message = error_envelope(resp)
    type_lower, code_lower = error_type.lower(), error_code.lower()
    message_lower = message.lower().replace("`", "")
    # ① 中转/官方错误码明确表态模型不可用（type 与 code 都要看）。
    if any(marker in type_lower or marker in code_lower for marker in MODEL_TYPE_MARKERS):
        return True
    if "not_found_error" in type_lower:
        # 404 的资源型错误：只有文案明确指向模型才算（排除路径/端点不存在）。
        return bool(re.search(r"\bmodel\b|模型", message_lower))
    if resp.status in (400, 401, 403, 404, 410, 422):
        return any(re.search(pattern, message_lower) for pattern in _MODEL_MESSAGE_PATTERNS)
    if resp.status >= 500:
        # ② 无错误码时退到文案：必须**点名了本次请求的模型**才算拒绝，防止把
        #    「No available accounts」（渠道整体不可用，与模型名无关）当成校验成功。
        if "model not found" in message_lower or ("模型" in message_lower and "不存在" in message_lower):
            return True
        if requested_model and requested_model.lower() in message_lower:
            return bool(re.search(r"no available (?:channel|account)s? for model"
                                  r"|无可用(?:渠道|通道)", message_lower))
    return False


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


def _response_model(resp: Resp) -> str | None:
    body = resp.json() or {}
    model = body.get("model")
    return model if isinstance(model, str) else None


# ---------------------------------------------------------------------------
# 二、通道封装（证据行格式对齐 proto_signature_check 的 requests.jsonl）
# ---------------------------------------------------------------------------

def _post(client: Client, payload: dict[str, Any], phase: str, evidence: dict[str, Any],
          *, retry_5xx: bool = True) -> Resp:
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


def _declared_models(client: Client, evidence: dict[str, Any]) -> set[str] | None:
    """尽力取端点声明的模型集（/v1/models）；取不到返回 None，只留痕不定判据。"""
    try:
        resp = client.get("/models")
    except Exception as exc:  # noqa: BLE001 —— 探测失败不该影响主判定
        evidence["requests"].append({"phase": "models_probe", "url": f"{client.root}/models",
                                     "http_status": None, "response": None,
                                     "raw_response": "", "elapsed_seconds": None, "attempts": 1,
                                     "timing": {}, "transport_error": f"{type(exc).__name__}: {exc}"})
        return None
    body = resp.json()
    identifiers: set[str] = set()
    if isinstance(body, dict):
        for item in body.get("data") or []:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                identifiers.add(item["id"])
        for item in body.get("models") or []:
            if isinstance(item, str):
                identifiers.add(item)
            elif isinstance(item, dict) and isinstance(item.get("id"), str):
                identifiers.add(item["id"])
    evidence["requests"].append({"phase": "models_probe", "url": f"{client.root}/models",
                                 "http_status": resp.status, "response": body,
                                 "raw_response": resp.content.decode("utf-8", "replace"),
                                 "elapsed_seconds": resp.elapsed, "attempts": resp.attempts,
                                 "timing": {"e2e_ms": resp.elapsed * 1000},
                                 "transport_error": resp.transport_error})
    return identifiers or None


# ---------------------------------------------------------------------------
# 三、两条用例
# ---------------------------------------------------------------------------

def _verdict(step: int, status: str, detail: str, duration: float = 0.0,
             **observations: Any) -> Verdict:
    title = "正常模型名请求" if step == 1 else "修改一个模型名字符后请求"
    return Verdict(f"MN-0{step}", title, "model", status, detail, observations,
                   duration_seconds=duration)


def _stop(status: str, detail: str, duration: float = 0.0, **obs: Any) -> list[Verdict]:
    return [_verdict(1, status, detail, duration=duration, **obs),
            _verdict(2, BLOCKED, "第一步未取得可用通道，未发送变异模型名请求")]


def run_all(client: Client) -> tuple[list[Verdict], dict[str, Any]]:
    evidence: dict[str, Any] = {"requests": [], "requested_model": client.model}
    normal_request = {"model": client.model, "max_tokens": 256,
                      "messages": [{"role": "user", "content": PROMPT}]}
    first = _post(client, normal_request, "initial", evidence)
    if first.transport_error or first.status != 200:
        reason = first.transport_error or f"HTTP {first.status} {first.text_head}"
        return _stop(BLOCKED, f"正常模型名请求未成功：{reason}",
                     duration=first.elapsed,
                     http_status=first.status, requested_model=client.model), evidence
    body = first.json()
    if body is None:
        return _stop(FAILED, "正常模型的响应不是合法的 Messages JSON 对象",
                     duration=first.elapsed, http_status=first.status), evidence
    evidence["normal_response"] = body
    positive = _verdict(1, PASSED,
                        f"正常模型名 `{client.model}` 请求成功（HTTP 200，响应合法 Messages JSON）",
                        duration=first.elapsed,
                        http_status=first.status, requested_model=client.model,
                        response_model=body.get("model"))
    try:
        mutated, position = mutate_model_name(client.model)
    except ValueError as exc:
        return [positive, _verdict(2, BLOCKED, str(exc), duration=first.elapsed)], evidence
    evidence["mutation"] = {"field": "model", "character_offset": position,
                            "before": client.model[position], "after": mutated[position],
                            "changed_characters": 1,
                            "from": client.model, "to": mutated}
    declared = _declared_models(client, evidence)
    evidence["declared_models_known"] = declared is not None
    if declared is not None and mutated in declared:
        # 变异不出来一个"端点认定无效"的名字（恰好撞上声明列表）⇒ 取证前提落空。
        return [positive, _verdict(2, BLOCKED,
                                   f"变异后的名称 `{mutated}` 恰在端点 /v1/models 声明列表中，"
                                   "变异无效、无法取证（不据此判端点失败）",
                                   duration=first.elapsed,
                                   mutated_model=mutated, declared_count=len(declared),
                                   **evidence["mutation"])], evidence
    mutated_request = copy.deepcopy(normal_request)
    mutated_request["model"] = mutated
    resp = _post(client, mutated_request, "mutated_model_request", evidence, retry_5xx=False)
    error_type, error_code, _ = error_envelope(resp)
    obs = {**evidence["mutation"], "http_status": resp.status,
           "requested_model": mutated, "response_model": _response_model(resp),
           "error": (resp.json() or {}).get("error"),
           "error_type": error_type or None, "error_code": error_code or None,
           "declared_models_known": declared is not None}
    if is_model_rejection(resp, requested_model=mutated):
        evidence_marker = error_code or error_type
        return [positive, _verdict(2, PASSED,
                                   f"变异模型名 `{mutated}` 被明确拒绝：HTTP {resp.status}"
                                   + (f"（{evidence_marker}）" if evidence_marker else "")
                                   + (f"｜上游说明：{_upstream_message(resp)}"
                                      if _upstream_message(resp) else ""),
                                   duration=resp.elapsed, **obs)], evidence
    if resp.transport_error or resp.status == 0:
        return [positive, _verdict(2, BLOCKED,
                                   f"变异模型名请求传输失败：{resp.transport_error}（未取到拒绝证据）",
                                   duration=resp.elapsed, **obs)], evidence
    if resp.status == 200:
        served = obs["response_model"]
        extra = (f"；响应 model 字段为 `{served}`，与请求的变异名不同——网关可能静默改写/映射到其他模型"
                 if served and served != mutated else "")
        note = "" if declared is not None else "（未能取得端点 /v1/models 声明列表佐证）"
        return [positive, _verdict(2, FAILED,
                                   f"未知模型名 `{mutated}` 未被拒绝：HTTP 200 正常返回{extra}{note}",
                                   duration=resp.elapsed, **obs)], evidence
    if resp.status == 429:
        return [positive, _verdict(2, BLOCKED,
                                   f"变异模型名请求被限流（HTTP 429），未取到校验证据",
                                   duration=resp.elapsed, **obs)], evidence
    if resp.status >= 500:
        return [positive, _verdict(2, BLOCKED,
                                   f"变异模型名请求 HTTP {resp.status} 且错误未指向模型，"
                                   "不能当作模型名校验成功（瞬时故障重试后再 blocked）"
                                   + (f"｜上游说明：{_upstream_message(resp)}"
                                      if _upstream_message(resp) else ""),
                                   duration=resp.elapsed, **obs)], evidence
    return [positive, _verdict(2, BLOCKED,
                               f"HTTP {resp.status} 的错误未提及模型，不能当作模型名校验成功"
                               + (f"｜上游说明：{_upstream_message(resp)}"
                                  if _upstream_message(resp) else ""),
                               duration=resp.elapsed, **obs)], evidence


# ---------------------------------------------------------------------------
# 四、汇总 / 报告 / 输出（volcperf.result.v2 兼容，证据文件 requests.jsonl）
# ---------------------------------------------------------------------------

def summarize(verdicts: list[Verdict]) -> dict[str, Any]:
    counts = {status: sum(v.status == status for v in verdicts)
              for status in (PASSED, FAILED, BLOCKED)}
    passed = len(verdicts) == 2 and all(v.status == PASSED for v in verdicts)
    return {"counts": counts, "admission_passed": passed}


def render_report(model: str, endpoint: str, verdicts: list[Verdict],
                  summary: dict[str, Any]) -> str:
    lines = ["# Claude 模型名称两步校验", "", f"- 被测模型：`{model}`",
             f"- 端点：`{endpoint}`",
             f"- 结论：**{'准入通过' if summary['admission_passed'] else '不准入'}**"
             f"（passed {summary['counts'].get('passed', 0)} / failed {summary['counts'].get('failed', 0)}"
             f" / blocked {summary['counts'].get('blocked', 0)}）", "",
             "| 步骤 | 状态 | 说明 |", "|---|---|---|"]
    lines += [f"| {v.case_id} {v.title} | {v.status} | {v.detail.replace('|', '/')} |" for v in verdicts]
    lines += ["", "> 判定纪律：第一步用真实模型名取证；第二步只改模型名一个字符，"
              "端点必须明确报模型不存在/无效。HTTP 200 放行未知模型名 ⇒ failed（未校验模型名）；"
              "泛化 4xx/5xx/限流/传输错 ⇒ blocked（不能把无关错误当作校验成功）；"
              "变异名恰在 /v1/models 声明列表中 ⇒ blocked（变异无效，不冤枉端点）。", ""]
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, model: str, endpoint: str, verdicts: list[Verdict],
                  evidence: dict[str, Any]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(verdicts)
    sent = sum(row.get("attempts") or 0 for row in evidence["requests"]
               if row.get("phase") in {"initial", "mutated_model_request"})
    payload = {"schema_version": "volcperf.result.v2", "case": CASE_ID_ROOT,
               "script": "fable5.model_name_check", "test_kind": "fable",
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "model": model, "endpoint": endpoint, "total": len(verdicts),
               **summary["counts"], "sent": sent,
               "admission_passed": summary["admission_passed"],
               "results": [v.as_dict() for v in verdicts],
               "requested_model": model, "evidence": "requests.jsonl"}
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    (out_dir / "requests.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in evidence["requests"]),
        encoding="utf-8")
    if "normal_response" in evidence:
        (out_dir / "response.json").write_text(
            json.dumps(evidence["normal_response"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    if "mutation" in evidence:
        (out_dir / "mutation.json").write_text(
            json.dumps(evidence["mutation"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(model, endpoint, verdicts, summary), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# 五、CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Claude 模型名称两步校验（改一个模型名字符，要求端点报错）")
    parser.add_argument("--model", help="模型 ID（默认 FABLE_MODEL/ANTHROPIC_MODEL）")
    parser.add_argument("--base-url", help="端点域名（默认 FABLE_BASE_URL/ANTHROPIC_BASE_URL）")
    parser.add_argument("--api-key", help="API Key（默认 FABLE_API_KEY/ANTHROPIC_API_KEY）")
    parser.add_argument("--auth", choices=("x-api-key", "bearer"), default="x-api-key")
    parser.add_argument("--anthropic-version", default=os.getenv("FABLE_ANTHROPIC_VERSION", "2023-06-01"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-retries", type=int, default=2,
                        help="第一步（正常请求）的瞬时故障重试次数；第二步负向请求固定不重试")
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--trust-env", action="store_true",
                        default=os.getenv("FABLE_TRUST_ENV", "false").lower() in {"1", "true", "yes"})
    parser.add_argument("--no-verify-ssl", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="离线自检：只跑变异与判定纯函数")
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
        verdicts, evidence = run_all(client)
    summary = write_outputs(out, model, client.root, verdicts, evidence)
    for verdict in verdicts:
        print(f"[{verdict.status}] {verdict.case_id} {verdict.detail}")
    print(f"结果目录: {out.resolve()}")
    return 0 if summary["admission_passed"] else 2


def _self_test() -> int:
    """离线自检：变异可复现且长度不变；拒绝判定只认"指向模型"的错误。"""
    name = "claude-fable-5-1"
    mutated, position = mutate_model_name(name)
    assert len(mutated) == len(name), mutated
    assert mutated != name and sum(a != b for a, b in zip(name, mutated)) == 1, mutated
    assert mutated[position] not in name.lower(), mutated
    assert mutate_model_name(name) == (mutated, position), "变异必须确定性可复现"
    assert mutate_model_name("claude-opus-5")[0] != "claude-opus-5"

    def resp_of(status: int, body: dict) -> Resp:
        return Resp(status, {}, json.dumps(body).encode(), 0.0)

    assert is_model_rejection(resp_of(400, {"error": {"type": "invalid_request_error",
                                                      "message": "model: claude-qable-5-1 not found"}}))
    assert is_model_rejection(resp_of(404, {"error": {"type": "not_found_error",
                                                      "message": "model `claude-qable-5-1` 不存在"}}))
    assert is_model_rejection(resp_of(503, {"error": {"type": "model_not_found",
                                                      "message": "No available accounts"}}))
    # 2026-10-13 实批形态：New API 把码放在 error.code，type 只是 new_api_error
    assert is_model_rejection(resp_of(503, {"error": {
        "code": "model_not_found", "type": "new_api_error",
        "message": "No available channel for model claude-qable-5 under group claude-cc (distributor)"}}),
        requested_model="claude-qable-5")
    # 无错误码时退到文案，但必须点名被请求的模型
    assert is_model_rejection(resp_of(503, {"error": {
        "type": "new_api_error",
        "message": "No available channel for model claude-qable-5 under group claude-cc"}}),
        requested_model="claude-qable-5")
    # 负控：限流、泛化 400、认证错、传输错都不是"模型名校验成功"
    assert not is_model_rejection(resp_of(429, {"error": {"type": "rate_limit_error", "message": "slow down"}}))
    assert not is_model_rejection(resp_of(400, {"error": {"type": "invalid_request_error",
                                                          "message": "max_tokens: field required"}}))
    assert not is_model_rejection(resp_of(401, {"error": {"type": "authentication_error",
                                                          "message": "invalid api key"}}))
    assert not is_model_rejection(Resp(0, {}, b"", 0.0, 1, "ConnectError: boom"))
    # 负控：渠道整体不可用（未点名模型）不得当成模型名被拒绝
    assert not is_model_rejection(resp_of(503, {"error": {"type": "new_api_error",
                                                          "message": "No available accounts: 无可用渠道"}}),
                                  requested_model="claude-qable-5")
    print(f"[离线自检] 变异确定性通过（{name} → {mutated}）；拒绝判定正例 + 负控全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
