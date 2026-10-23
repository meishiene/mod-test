#!/usr/bin/env python3
"""复测第二阶段：对第一阶段出现的三个疑点做定点确认（额度优先，共 8 次请求）

1. K3 缓存：用 ~3500 token 前缀再测一次，排除「前缀太短导致比率天然低于 80%」的测量偏差。
2. K3 采样参数：用 logprobs 对照 temperature=0 / 1.0 / 2.1 的首 token 分布，
   判断「200 接受」是「静默改写为默认值」还是「真的生效」。
3. GLM 5.3 / flash 身份：换一种问法（开发公司 + 基座模型名）复核第一阶段的矛盾自述。
"""
from __future__ import annotations

import os

import json
import time
from datetime import datetime
from pathlib import Path

import requests

BASE = "https://api.oinone.top/v1"
KEYS = {
    "kimi": os.environ.get("OINONE_KIMI_API_KEY", ""),
    "glm": os.environ.get("OINONE_GLM_API_KEY", ""),
}
OUT = Path(__file__).resolve().parent / "results" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-p2")
OUT.mkdir(parents=True, exist_ok=True)
N = {"n": 0}


def call(case_id: str, model: str, key: str, payload: dict) -> dict:
    N["n"] += 1
    body = {"model": model, **payload}
    started = time.perf_counter()
    r = requests.post(BASE + "/chat/completions",
                      headers={"Authorization": "Bearer " + KEYS[key], "Content-Type": "application/json"},
                      json=body, timeout=240)
    rec = {"case_id": case_id, "model": model, "request": body, "http_status": r.status_code,
           "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
           "raw_body": r.text[:9000],
           "gateway_version": r.headers.get("x-new-api-version")}
    try:
        data = r.json()
    except ValueError:
        data = None
    if isinstance(data, dict):
        rec["body"] = data
        u = data.get("usage") or {}
        rec["usage"] = u
        rec["prompt_tokens"] = u.get("prompt_tokens")
        rec["cached_tokens"] = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")
                                or u.get("prompt_cache_hit_tokens") or 0)
        m = ((data.get("choices") or [{}])[0].get("message") or {})
        rec["content"] = m.get("content")
        rec["reasoning"] = m.get("reasoning_content") or ""
        rec["logprobs"] = ((data.get("choices") or [{}])[0].get("logprobs") or {})
        rec["finish_reason"] = (data.get("choices") or [{}])[0].get("finish_reason")
        rec["error"] = data.get("error")
    with (OUT / "results.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"[{N['n']:02d}] {case_id:18s} {model:16s} -> {r.status_code} "
          f"pt={rec.get('prompt_tokens')} cached={rec.get('cached_tokens')} ({rec['elapsed_ms']:.0f} ms)")
    return rec


print("输出", OUT)

# --------------------------------------------------------------------------- #
# 1. K3 缓存 · 大前缀
# --------------------------------------------------------------------------- #
print("\n[1] K3 缓存（~3500 token 稳定前缀）")
prefix = "K3-RETEST-CACHE-BIG-" + ("固定公共上下文用于缓存命中复测，本段内容在三次请求之间完全一致。" * 130)
rows = []
for label in ("warm", "probe1", "probe2"):
    rows.append(call(f"K3-CACHE-BIG-{label}", "kimi-k3", "kimi",
                     {"messages": [{"role": "system", "content": prefix},
                                   {"role": "user", "content": "只回复 OK"}],
                      "max_tokens": 16}))
    time.sleep(3 if label == "warm" else 2)
probes = [r for r in rows[1:] if r.get("prompt_tokens")]
ratios = [r["cached_tokens"] / r["prompt_tokens"] for r in probes]
print(f"  ratios={[round(x, 4) for x in ratios]}  "
      f"prompt={[r.get('prompt_tokens') for r in rows]} cached={[r.get('cached_tokens') for r in rows]}")

# --------------------------------------------------------------------------- #
# 2. K3 采样参数是否真的生效（logprobs 对照）
# --------------------------------------------------------------------------- #
print("\n[2] K3 temperature 是否生效（logprobs 首 token 分布对照）")
BASE_PROMPT = "续写这句话，只输出一个词：今天天气真"
lp = []
for temp in (0.0, 1.0, 2.1):
    rec = call(f"K3-LP-{temp}", "kimi-k3", "kimi",
               {"messages": [{"role": "user", "content": BASE_PROMPT}],
                "max_tokens": 8, "temperature": temp, "logprobs": True, "top_logprobs": 5})
    top = []
    try:
        content_lp = (rec.get("logprobs") or {}).get("content") or []
        if content_lp:
            top = [(t.get("token"), round(t.get("logprob"), 4)) for t in (content_lp[0].get("top_logprobs") or [])]
    except Exception as exc:  # noqa: BLE001
        print("  parse err", exc)
    lp.append({"temp": temp, "http": rec["http_status"], "content": rec.get("content"),
               "top": top, "error": rec.get("error")})
    time.sleep(1)
for x in lp:
    print(f"  temp={x['temp']}: http={x['http']} content={x['content']!r} top={x['top'][:5]} err={x['error']}")

# --------------------------------------------------------------------------- #
# 3. GLM 身份复核（换问法）
# --------------------------------------------------------------------------- #
print("\n[3] GLM 身份复核：开发公司 + 基座模型名")
ident = []
for model, tag in (("glm-5.3", "5.3"), ("glm-5.3-flash", "flash")):
    rec = call(f"G-IDENT2:{tag}", model, "glm",
               {"messages": [{"role": "user", "content":
                              "只回答两行，不要解释：\nA) 你的基座模型由哪家公司训练？\nB) 你的基座模型叫什么名字？"}],
                "thinking": {"type": "enabled"}, "max_tokens": 512})
    ident.append({"tag": tag, "http": rec["http_status"], "content": rec.get("content"),
                  "reasoning_head": (rec.get("reasoning") or "")[:300]})
    time.sleep(1)

print("\n" + "=" * 90)
print(json.dumps({"cache": {"prompt": [r.get("prompt_tokens") for r in rows],
                            "cached": [r.get("cached_tokens") for r in rows],
                            "ratios": ratios},
                  "logprobs": lp, "identity": ident}, ensure_ascii=False, indent=1))
print("总请求", N["n"], "→", OUT / "results.jsonl")
