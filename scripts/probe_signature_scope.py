#!/usr/bin/env python3
"""签名覆盖域 / 证明力 / 回放 token 记账专项探测（**独立脚本，未接入平台套件表**）。

背景：`fable5/proto_signature_check.py`（签名两步校验）与 `fable5/zero_injection_check.py`
的 ZI-04 都建立在「thinking.signature 是官方原生」之上，并常被当成「0 注入」的证据。
本脚本只回答一个问题：**签名到底约束了什么**，即网关注入（system / 历史改写）能否被它检出。

三个 phase：

  scope   —— 覆盖域矩阵：取一次真签名后，逐个单变量改动再回传（改签名 / 改 thinking 明文 /
             改正文 / 加 system / 改首条 user 消息 / 去掉 thinking 块），看哪些被拒。
  attest  —— 签名证明力：跨响应搬运（B 轮的真思考明文 + A 轮签名）、伪造思考明文 + 真签名；
             同时用「51 字可数金丝雀 system」校准 input_tokens 是否逐字反映到达上游的内容。
  tokens  —— 回放 token 记账对照：同一签名、同样消息，只改 thinking 明文长度 / 是否带 system /
             system 长度，各条件重复两次，比较 input+cc+cr 总量与缓存命中，判断
             「回传时上游用的是请求体明文还是签名内封存内容」。

判据（脚本自己算，不依赖平台 validators）：
  被拒   = HTTP 400/422 且错误文案指向 signature
  接受   = HTTP 200
  token 总量 = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
              （提示词缓存会影响三项的分布，但三者之和对同一前缀稳定；本脚本只比总量）

用法（API Key 只走环境变量，不写盘、不进日志）：

  CLAUDE_RELAY_KEY=sk-xxx .venv/bin/python scripts/probe_signature_scope.py \
      --model claude-fable-5-1 --base-url http://HOST:PORT --out .pytest-tmp/sig-probe

输出：`evidence-<phase>.json`（逐请求原文与 usage）+ `report.md`（矩阵与结论）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

THINKING = {"type": "adaptive", "display": "summarized"}
Q_QUICKSORT = "请解释快速排序的平均时间复杂度为什么是 O(n log n)，给出递推推导过程。"
Q_BALL = ("有 12 个外观完全一样的小球，其中 1 个重量异常（不知道是更重还是更轻），"
          "只用无砝码天平，最多称量 3 次，请给出完整方案并证明。")
Q_CALC = "请计算 17×23，核对计算过程后给出结果。"
CANARY = "以下是本次会话的固定前缀口令 CANARY-9F2B71E4，请在任何情况下都不要在本轮回复中提及它。"
FOLLOW_UP = "请确认上一轮结果，只回复 OK。"


class Probe:
    def __init__(self, base_url: str, key: str, model: str, timeout: float = 300.0) -> None:
        self.url = base_url.rstrip("/").removesuffix("/v1") + "/v1/messages"
        self.model = model
        self.headers = {"content-type": "application/json", "accept": "application/json",
                        "anthropic-version": "2023-06-01", "x-api-key": key}
        self.timeout = timeout
        self.records: list[dict[str, Any]] = []

    def post(self, tag: str, payload: dict[str, Any], note: str = "") -> dict[str, Any] | None:
        t0 = time.monotonic()
        resp = httpx.post(self.url, headers=self.headers, json=payload,
                          timeout=self.timeout, trust_env=False)
        elapsed = time.monotonic() - t0
        body = None
        if resp.headers.get("content-type", "").startswith("application/json"):
            try:
                body = resp.json()
            except ValueError:
                body = None
        usage = (body or {}).get("usage") or {}
        total = sum(usage.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                "cache_read_input_tokens"))
        error = body.get("error") if isinstance(body, dict) else None
        self.records.append({"tag": tag, "phase": tag.split(":")[0], "status": resp.status_code,
                             "elapsed": round(elapsed, 2), "usage": usage, "total_input": total,
                             "request": payload, "error": error,
                             "raw_head": resp.text[:400]})
        print(f"[{tag:<22}] HTTP {resp.status_code} {elapsed:6.2f}s 总计={total or '-'} "
              f"(in={usage.get('input_tokens')} cc={usage.get('cache_creation_input_tokens')} "
              f"cr={usage.get('cache_read_input_tokens')}) {note}")
        if error:
            print(f"    error = {json.dumps(error, ensure_ascii=False)[:240]}")
        return body

    def evidence(self) -> dict[str, Any]:
        return {"model": self.model, "endpoint": self.url, "records": self.records}


def signed_block(body: dict[str, Any] | None) -> tuple[str, str, str] | tuple[None, None, None]:
    for block in (body or {}).get("content") or []:
        if block.get("type") == "thinking" and block.get("signature"):
            text = next((b.get("text") or "" for b in (body.get("content") or [])
                         if b.get("type") == "text"), "")
            return block["signature"], block.get("thinking") or "", text
    return None, None, None


def base_request(model: str, question: str = Q_QUICKSORT) -> dict[str, Any]:
    return {"model": model, "max_tokens": 4096, "thinking": THINKING,
            "messages": [{"role": "user", "content": question}]}


def replay_request(model: str, question: str, blocks: list[dict[str, Any]],
                   system: Any = "__absent__", follow_up: str = FOLLOW_UP) -> dict[str, Any]:
    payload = {"model": model, "max_tokens": 4096, "thinking": THINKING,
               "messages": [{"role": "user", "content": question},
                            {"role": "assistant", "content": blocks},
                            {"role": "user", "content": follow_up}]}
    if system != "__absent__":
        payload["system"] = system
    return payload


def take_signature(p: Probe, question: str = Q_QUICKSORT) -> tuple[str, str, str]:
    body = p.post(f"signA:{question[:6]}", base_request(p.model, question))
    sig, thinking, text = signed_block(body)
    if not sig:
        raise SystemExit(f"取签失败：{question[:12]} 未返回带签名 thinking 块（types="
                         f"{[b.get('type') for b in (body or {}).get('content') or []]}）")
    print(f"    签名 {len(sig)} 字符 sha256={hashlib.sha256(sig.encode()).hexdigest()[:16]}"
          f" thinking 明文 {len(thinking)} 字符 正文 {len(text)} 字符")
    return sig, thinking, text


# --------------------------------------------------------------------------- scope
def phase_scope(p: Probe) -> None:
    sig, thinking, text = take_signature(p)
    blocks = lambda: [{"type": "thinking", "thinking": thinking, "signature": sig},   # noqa: E731
                      {"type": "text", "text": text}]
    p.post("scope:ident", replay_request(p.model, Q_QUICKSORT, blocks()), "基线（不改）")

    mutated = list(blocks())
    s = mutated[0]["signature"]
    i = next(k for k, c in enumerate(s) if c.isalnum())
    mutated[0]["signature"] = s[:i] + ("A" if s[i] != "A" else "B") + s[i + 1:]
    p.post("scope:sig_tweak", replay_request(p.model, Q_QUICKSORT, mutated), "改签名 1 字符")

    mutated = list(blocks())
    t = mutated[0]["thinking"]
    j = len(t) // 2
    mutated[0]["thinking"] = t[:j] + ("A" if t[j] != "A" else "B") + t[j + 1:]
    p.post("scope:thinking_tweak", replay_request(p.model, Q_QUICKSORT, mutated), "改 thinking 明文")

    mutated = list(blocks())
    mutated[1] = {"type": "text", "text": (mutated[1]["text"] or "") + "（改动）"}
    p.post("scope:text_tweak", replay_request(p.model, Q_QUICKSORT, mutated), "改正文块")

    p.post("scope:system_added", replay_request(p.model, Q_QUICKSORT, blocks(), system=CANARY),
           "顶层加 system（注入等价形态）")
    p.post("scope:user_tweak",
           replay_request(p.model, Q_QUICKSORT + " ", blocks()), "改首条 user 消息")
    p.post("scope:thinking_removed",
           replay_request(p.model, Q_QUICKSORT, [blocks()[1]]), "回传去掉 thinking 块")


# -------------------------------------------------------------------------- attest
def phase_attest(p: Probe) -> None:
    sig_a, think_a, text_a = take_signature(p, Q_QUICKSORT)
    sig_b, think_b, _ = take_signature(p, Q_BALL)
    print(f"    A sig={len(sig_a)} thinking={len(think_a)} | B sig={len(sig_b)} thinking={len(think_b)}")
    p.post("attest:cross_attach",
           replay_request(p.model, Q_QUICKSORT,
                          [{"type": "thinking", "thinking": think_b, "signature": sig_a},
                           {"type": "text", "text": text_a}]),
           "B 轮真思考明文 + A 轮签名")
    p.post("attest:fabricated_thinking",
           replay_request(p.model, Q_QUICKSORT,
                          [{"type": "thinking", "thinking": "这是一段我随手编造的思考内容，与上游无关。",
                            "signature": sig_a},
                           {"type": "text", "text": text_a}]),
           "编造思考明文 + A 轮签名")
    base = base_request(p.model, Q_CALC)
    p.post("attest:calc_no_system", base, "金丝雀校准①不传 system")
    p.post("attest:calc_empty_system", {**base, "system": ""}, '金丝雀校准②system=""')
    p.post("attest:calc_canary_system", {**base, "system": CANARY},
           f"金丝雀校准③system={len(CANARY)} 字")


# -------------------------------------------------------------------------- tokens
def phase_tokens(p: Probe) -> None:
    sig, thinking, text = take_signature(p)
    blocks = lambda t: [{"type": "thinking", "thinking": t, "signature": sig},   # noqa: E731
                        {"type": "text", "text": text}]
    for tag, t, note in (("short#1", "短。", "thinking 明文 2 字"),
                         ("short#2", "短。", "thinking 明文 2 字（重复）"),
                         ("long#1", thinking, f"thinking 明文 {len(thinking)} 字"),
                         ("long#2", thinking, f"thinking 明文 {len(thinking)} 字（重复）")):
        p.post(f"tokens:plain_{tag}", replay_request(p.model, Q_QUICKSORT, blocks(t)), note)
    for tag, system, note in (("nosys", "__absent__", "不传 system"),
                              ("emptysys", "", 'system=""'),
                              ("canary", CANARY, f"system 金丝雀 {len(CANARY)} 字")):
        for run in (1, 2):
            p.post(f"tokens:sys_{tag}#{run}",
                   replay_request(p.model, Q_QUICKSORT, blocks(thinking), system=system),
                   f"{note}（第 {run} 次）")


PHASES = {"scope": phase_scope, "attest": phase_attest, "tokens": phase_tokens}


def write_report(out: Path, evidence: dict[str, Any]) -> None:
    lines = ["# 签名覆盖域 / 证明力 / 回放 token 记账探测", "",
             f"- 端点：`{evidence['endpoint']}`",
             f"- 模型：`{evidence['model']}`", "",
             "| tag | HTTP | 总计 input | in | cc | cr | 耗时 | 错误 |",
             "|---|---|---|---|---|---|---|---|"]
    for rec in evidence["records"]:
        u = rec["usage"] or {}
        err = (rec["error"] or {}).get("message", "")
        lines.append(f"| `{rec['tag']}` | {rec['status']} | {rec['total_input'] or '-'} | "
                     f"{u.get('input_tokens', '-')} | {u.get('cache_creation_input_tokens', '-')} | "
                     f"{u.get('cache_read_input_tokens', '-')} | {rec['elapsed']}s | "
                     f"{err[:70]} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", default="scope",
                        help="逗号分隔：scope,attest,tokens（默认 scope）")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=os.getenv("CLAUDE_RELAY_KEY", ""))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out", type=Path, default=Path(".pytest-tmp/sig-probe"))
    args = parser.parse_args()
    if not args.api_key:
        print("缺少 API Key：用 --api-key 或环境变量 CLAUDE_RELAY_KEY 提供", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    for name in [x.strip() for x in args.phase.split(",") if x.strip()]:
        if name not in PHASES:
            print(f"未知 phase：{name}（可选 {', '.join(PHASES)}）", file=sys.stderr)
            return 2
        print(f"==== phase {name} ====")
        probe = Probe(args.base_url, args.api_key, args.model, args.timeout)
        PHASES[name](probe)
        evidence = probe.evidence()
        dest = args.out / f"evidence-{name}.json"
        dest.write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
        write_report(args.out, evidence)
        print(f"证据写入 {dest} / {args.out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
