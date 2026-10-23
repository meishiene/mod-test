"""缓存归属探针（DS-V4-16 取证专用）：区分「DeepSeek 原生 KV 缓存（user_id 未透传）」与「网关响应缓存」。

背景（2026-10-21）
------------------
DS-V4-16 在 `deepseek-v4.1-flash-ark` 上观测到 same_user_hit = other_user_hit = 1792，
原先直接判成「模型侧数据隔离失效」。但这条结论只有在**能证明命中来自 DeepSeek 后端 KV 缓存**时
才成立；若命中来自网关自己的响应缓存（key 只哈希 messages+model、不带 user_id），那属网关实现，
按 gateway_contract 口径不该判模型 failed。

判别依据（本探针采集的字段）：

===============  ==========================================================
 现象             结论
===============  ==========================================================
 响应 `id`/`created`
 与上一跳完全相同   网关直接回放上一次存下的响应（响应缓存）⇒ 与 DeepSeek 无关
 换 user_id 后
 `prompt_cache_hit_tokens` > 0
 且 `id` 全新       真的命中了后端 KV 缓存，且 user_id 没被上游当作隔离维度
 换 user_id 后
 命中 0 且内容不同  隔离成立（本轮）
===============  ==========================================================

用法::

    DS_API_KEY=... DS_BASE_URL=https://api.oinone.top \\
      .venv/bin/python scripts/probe_ds_cache_ownership.py deepseek-v4.1-flash-ark

输出 JSON 打印并落盘 `results/DS_CACHE_PROBE-<时间戳>/cache_probe.json`。
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
PREFIX_BODY = "缓存归属探测固定背景。" * 700


def _one(client: httpx.Client, url: str, model: str, messages: list[dict], user_id: str | None,
         extra: dict | None = None) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": 16, "thinking": {"type": "disabled"}}
    if user_id is not None:
        body["user_id"] = user_id
    if extra:
        body.update(extra)
    started = time.perf_counter()
    response = client.post(url, json=body)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    usage = payload.get("usage") or {}
    message = ((payload.get("choices") or [{}])[0].get("message") or {})
    return {
        "http_status": response.status_code,
        "id": payload.get("id"),
        "created": payload.get("created"),
        "system_fingerprint": payload.get("system_fingerprint"),
        "content": str(message.get("content") or "")[:60],
        "usage": usage,
        "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
        "prompt_cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "elapsed_ms": elapsed_ms,
        "request_user_id": user_id,
    }


def probe(model: str, client: httpx.Client, url: str) -> dict:
    nonce = uuid.uuid4().hex[:10]
    uid_a, uid_b = f"cache-a-{nonce}", f"cache-b-{nonce}"
    prefix = f"[NONCE {nonce}]" + PREFIX_BODY
    messages = [{"role": "user", "content": prefix}]
    rows = {}
    rows["r1_cold_userA"] = _one(client, url, model, messages, uid_a)
    time.sleep(3)
    rows["r2_repeat_userA"] = _one(client, url, model, messages, uid_a)
    rows["r3_other_userB"] = _one(client, url, model, messages, uid_b)
    # 同 messages、换一个请求参数（网关响应缓存若只哈希 messages+model，仍会回放）
    rows["r4_userB_temp"] = _one(client, url, model, messages, uid_b, {"temperature": 0.3})
    # 同前缀但不同的收尾问题：真 KV 缓存应命中前缀，响应缓存不会命中
    rows["r5_prefix_tail"] = _one(
        client, url, model, [{"role": "user", "content": prefix + "\n请只回复：第九个字的颜色？"}], uid_b
    )

    r2, r3, r4, r5 = (rows[k] for k in ("r2_repeat_userA", "r3_other_userB", "r4_userB_temp", "r5_prefix_tail"))
    replay_other = bool(r3.get("id")) and r3.get("id") == r2.get("id")
    replay_temp = bool(r4.get("id")) and r4.get("id") == r2.get("id")
    kv_hit_other = isinstance(r3.get("prompt_cache_hit_tokens"), int) and r3["prompt_cache_hit_tokens"] > 0
    kv_hit_prefix_tail = isinstance(r5.get("prompt_cache_hit_tokens"), int) and r5["prompt_cache_hit_tokens"] > 0
    if replay_other or replay_temp:
        verdict = "gateway_response_cache"
    elif kv_hit_other or kv_hit_prefix_tail:
        verdict = "deepseek_kv_cache_user_id_not_isolated"
    else:
        verdict = "isolated_or_not_observed"
    return {"model": model, "rows": rows, "verdict": verdict,
            "signals": {"id_replayed_for_other_user": replay_other, "id_replayed_ignoring_temperature": replay_temp,
                        "kv_hit_across_user": kv_hit_other, "kv_hit_same_prefix_other_tail": kv_hit_prefix_tail}}


def main(argv: list[str]) -> int:
    api_key = os.getenv("DS_API_KEY") or os.getenv("VOLC_API_KEY")
    base_url = (os.getenv("DS_BASE_URL") or os.getenv("VOLC_BASE_URL") or "").rstrip("/")
    if not api_key or not base_url:
        print(json.dumps({"error": "need DS_API_KEY / DS_BASE_URL"}, ensure_ascii=False))
        return 2
    models = [m for m in argv if not m.startswith("-")] or ["deepseek-v4.1-flash-ark"]
    url = f"{base_url}/v1/chat/completions"
    with httpx.Client(timeout=180, headers={"Authorization": f"Bearer {api_key}"}) as client:
        records = [probe(model, client, url) for model in models]
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endpoint": url,
        "models": records,
        "note": "id/created 相同 ⇒ 网关响应回放；换 user_id 仍 prompt_cache_hit_tokens>0 且 id 全新 ⇒ 后端 KV 缓存未按 user_id 隔离。",
    }
    out_dir = REPO / "results" / f"DS_CACHE_PROBE-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache_probe.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    payload["evidence_dir"] = str(out_dir.relative_to(REPO))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
