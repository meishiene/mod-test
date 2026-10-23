#!/usr/bin/env python3
"""K3 / GLM-5.3 未通过项 · 独立复测（不依赖 web 套件、不依赖 platform runner）

设计原则
--------
1. 只打「上次没通过 / 不可评估」的项，不重跑已通过项（省额度）。
2. 每个用例的 max_tokens 压到能出结论的最小值：参数校验类 16，功能类 256~512。
3. prompts 一律短句；缓存类用 ~1000 token 公共前缀（够触发缓存，比原套件 3500 token 省 3 倍）。
4. 每次请求都落盘（请求、状态码、错误信封、usage、正文/思考长度、耗时），报告只从落盘数据算。
5. 硬预算闸门：超过 MAX_REQUESTS 直接中止，避免脚本跑飞烧额度。

用法：.venv/bin/python scripts/retest_20261021/retest.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests

BASE = "https://api.oinone.top/v1"
KEYS = {
    "kimi": os.environ.get("OINONE_KIMI_API_KEY", ""),
    "glm": os.environ.get("OINONE_GLM_API_KEY", ""),
}
BAD_KEY = "sk-invalid-retest-000000000000000000000000"

MODEL_KIMI = "kimi-k3"
MODEL_GLM = "glm-5.3"
MODEL_GLM_FLASH = "glm-5.3-flash"

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
MAX_REQUESTS = 60
TIMEOUT = 240

_state = {"n": 0, "rows": []}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def call(case_id: str, model: str, key_name: str, payload: dict,
         auth_mode: str = "normal", headers_extra: dict | None = None) -> dict:
    """发一次 /chat/completions，返回完整记录（不含明文 key）。"""
    if _state["n"] >= MAX_REQUESTS:
        raise RuntimeError(f"达到请求预算上限 {MAX_REQUESTS}，中止")
    _state["n"] += 1

    url = BASE + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if auth_mode == "normal":
        headers["Authorization"] = "Bearer " + KEYS[key_name]
    elif auth_mode == "bad":
        headers["Authorization"] = "Bearer " + BAD_KEY
    # auth_mode == "none": 不带头
    if headers_extra:
        headers.update(headers_extra)

    body = {"model": model, **payload}
    started = time.perf_counter()
    rec = {
        "case_id": case_id,
        "model": model,
        "key": key_name,
        "auth_mode": auth_mode,
        "url": url,
        "request": body,
        "request_headers": {k: ("Bearer ***" if k.lower() == "authorization" else v)
                            for k, v in headers.items()},
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=TIMEOUT)
        elapsed = (time.perf_counter() - started) * 1000
        rec["http_status"] = resp.status_code
        rec["elapsed_ms"] = round(elapsed, 1)
        rec["response_headers"] = {k.lower(): v for k, v in resp.headers.items()}
        raw = resp.text
        rec["raw_body"] = raw[:8000]
        try:
            data = resp.json()
        except ValueError:
            data = None
        rec["body"] = data
        if isinstance(data, dict):
            err = data.get("error")
            rec["error"] = err if isinstance(err, dict) else ({"message": str(err)} if err else None)
            u = data.get("usage") or {}
            rec["usage"] = u
            det = (u.get("prompt_tokens_details") or {})
            cdet = (u.get("completion_tokens_details") or {})
            rec["prompt_tokens"] = u.get("prompt_tokens") or u.get("input_tokens")
            rec["completion_tokens"] = u.get("completion_tokens") or u.get("output_tokens")
            rec["cached_tokens"] = (det.get("cached_tokens") or u.get("prompt_cache_hit_tokens")
                                    or u.get("cache_read_input_tokens") or 0)
            rec["reasoning_tokens"] = cdet.get("reasoning_tokens")
            rec["text_tokens"] = cdet.get("text_tokens")
            ch = (data.get("choices") or [{}])[0]
            m = ch.get("message") or {}
            c = m.get("content")
            rec["content"] = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
            rec["reasoning"] = m.get("reasoning_content") or m.get("reasoning") or ""
            rec["tool_calls"] = m.get("tool_calls") or []
            rec["finish_reason"] = ch.get("finish_reason")
            rec["response_model"] = data.get("model")
    except Exception as exc:  # noqa: BLE001
        rec["http_status"] = None
        rec["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        rec["transport_error"] = f"{type(exc).__name__}: {exc}"[:400]

    _state["rows"].append(rec)
    with (OUT / "results.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    st = rec.get("http_status")
    extra = ""
    if isinstance(rec.get("error"), dict) and rec["error"]:
        extra = f" {rec['error'].get('type')}/{rec['error'].get('code')}"
    print(f"  [{_state['n']:02d}] {case_id:22s} {model:16s} -> {st}{extra}"
          f"  ({rec['elapsed_ms']:.0f} ms)")
    return rec


def err_of(rec: dict) -> dict:
    e = rec.get("error")
    return e if isinstance(e, dict) else {}


def err_text(rec: dict) -> str:
    e = err_of(rec)
    return " ".join(str(e.get(k) or "") for k in ("message", "code", "type", "param")).strip()


# --------------------------------------------------------------------------- #
# 用例判定
# --------------------------------------------------------------------------- #
def judge_param(rec: dict, name: str, expect_400: bool = True) -> dict:
    """参数越界：期望 400；返回 200 = 静默接受（红线）。"""
    st = rec.get("http_status")
    e = err_of(rec)
    if st == 400:
        return {"verdict": "passed",
                "reason": f"400 拒绝，error.type={e.get('type')} code={e.get('code')}，指向 {name}="
                          f"{name in err_text(rec)}"}
    if st == 200:
        return {"verdict": "failed",
                "reason": f"200 静默接受非法 {name}（未校验/未改写留痕），finish={rec.get('finish_reason')}"}
    return {"verdict": "failed", "reason": f"HTTP {st}，非 400/200：{err_text(rec)[:160] or rec.get('raw_body', '')[:160]}"}


def judge_json_unsupported(rec: dict) -> dict:
    st = rec.get("http_status")
    txt = err_text(rec).lower()
    if st == 400:
        return {"verdict": "passed" if "format" in txt or "response_format" in txt else "warning",
                "reason": f"400 拒绝，error 语义命中 format={('format' in txt)}"}
    return {"verdict": "failed", "reason": f"HTTP {st}：不支持的 response_format 未拒绝"}


def judge_messages_type(rec: dict) -> dict:
    st = rec.get("http_status")
    if st == 400:
        e = err_of(rec)
        ok = e.get("type") in ("invalid_request_error", "invalid_request")
        return {"verdict": "passed" if ok else "warning",
                "reason": f"400 状态码正确，但 error.type={e.get('type')}（原厂口径 invalid_request_error）"
                          if not ok else "400 + error.type=invalid_request_error"}
    return {"verdict": "failed", "reason": f"HTTP {st}：messages 类型错误未返回 400"}


def judge_auth(rec: dict, expect: str = "reject") -> dict:
    st = rec.get("http_status")
    e = err_of(rec)
    if expect == "reject":
        if st in (401, 403):
            return {"verdict": "passed",
                    "reason": f"{st}，error.type={e.get('type')} msg={(e.get('message') or '')[:80]}"}
        if st == 200:
            return {"verdict": "failed", "reason": "200 伪成功：认证未生效"}
        return {"verdict": "warning", "reason": f"HTTP {st}（非 401/403），msg={(e.get('message') or '')[:80]}"}
    return {"verdict": "warning", "reason": f"HTTP {st}"}


def judge_tool(rec: dict, min_calls: int = 1) -> dict:
    st = rec.get("http_status")
    calls = rec.get("tool_calls") or []
    if st != 200:
        return {"verdict": "failed", "reason": f"HTTP {st}：{err_text(rec)[:120]}"}
    if len(calls) < min_calls:
        return {"verdict": "failed",
                "reason": f"200 但 tool_calls={len(calls)}（<{min_calls}）；正文={ (rec.get('content') or '')[:60]!r}"}
    names = [((c.get("function") or {}).get("name")) for c in calls]
    ids = [c.get("id") for c in calls]
    ok_args = all((c.get("function") or {}).get("arguments") for c in calls)
    return {"verdict": "passed" if ok_args else "warning",
            "reason": f"tool_calls={len(calls)} names={names} id_unique={len(set(ids)) == len(ids)} args={ok_args}"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"输出目录 {OUT}")
    print(f"端点 {BASE}    预算上限 {MAX_REQUESTS} 次请求\n")

    # ---------- 0. 可用模型对照（0 token） ----------
    print("[0] 模型清单")
    for name, key in KEYS.items():
        try:
            r = requests.get(BASE + "/models", headers={"Authorization": "Bearer " + key}, timeout=30)
            ids = [m.get("id") for m in (r.json().get("data") or [])]
            print(f"  {name}: {r.status_code} {ids}")
            _state["rows"].append({"case_id": "SCOPE-models", "key": name,
                                   "http_status": r.status_code, "models": ids})
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: ERR {exc}")

    results: list[dict] = []

    def run(case: dict) -> dict:
        rec = call(case["id"], case["model"], case["key"], case["payload"],
                   auth_mode=case.get("auth", "normal"))
        verdict = case["judge"](rec)
        row = {"case_id": case["id"], "title": case["title"], "model": case["model"],
               "http_status": rec.get("http_status"), **verdict,
               "request_id": rec.get("response_headers", {}).get("x-oneapi-request-id")
                             or rec.get("response_headers", {}).get("x-request-id")}
        results.append(row)
        return rec

    # ===================================================================== #
    # A. Kimi K3 —— 上次 failed / not_evaluable 的项
    # ===================================================================== #
    print("\n[A] Kimi K3 认证与参数校验")
    simple = [{"role": "user", "content": "只回复 OK"}]
    cases_k3 = [
        dict(id="K3-AUTH-01", title="无效 API Key", model=MODEL_KIMI, key="kimi", auth="bad",
             payload={"messages": simple, "max_tokens": 16},
             judge=lambda r: judge_auth(r, "reject")),
        dict(id="K3-AUTH-02", title="缺失 Authorization 头", model=MODEL_KIMI, key="kimi", auth="none",
             payload={"messages": simple, "max_tokens": 16},
             judge=lambda r: judge_auth(r, "reject")),
        dict(id="K3-ERR-01a", title="temperature=2.1（>2 上界）", model=MODEL_KIMI, key="kimi",
             payload={"messages": simple, "max_tokens": 16, "temperature": 2.1},
             judge=lambda r: judge_param(r, "temperature")),
        dict(id="K3-ERR-01b", title="temperature=-1（<0 下界）", model=MODEL_KIMI, key="kimi",
             payload={"messages": simple, "max_tokens": 16, "temperature": -1},
             judge=lambda r: judge_param(r, "temperature")),
        dict(id="K3-ERR-01c", title="top_p=1.5（>1 上界）", model=MODEL_KIMI, key="kimi",
             payload={"messages": simple, "max_tokens": 16, "top_p": 1.5},
             judge=lambda r: judge_param(r, "top_p")),
        dict(id="K3-ERR-02", title="messages 传字符串（类型错误）", model=MODEL_KIMI, key="kimi",
             payload={"messages": "not-an-array", "max_tokens": 16},
             judge=judge_messages_type),
        dict(id="K3-JSON-02", title="response_format=unsupported_type", model=MODEL_KIMI, key="kimi",
             payload={"messages": simple, "max_tokens": 16,
                      "response_format": {"type": "unsupported_type"}},
             judge=judge_json_unsupported),
        dict(id="K3-PARAM-02", title="temperature=0.6 + top_p=0.8（契约锁单值）", model=MODEL_KIMI, key="kimi",
             payload={"messages": [{"role": "user", "content": "只回复 OK"}], "max_tokens": 16,
                      "temperature": 0.6, "top_p": 0.8},
             judge=lambda r: judge_param(r, "temperature")),
    ]
    for c in cases_k3:
        run(c)
        time.sleep(1)

    print("\n[B] Kimi K3 工具调用（原 TOOL-002/003 不可评估 → 补执行器）")
    tools = [
        {"type": "function", "function": {"name": "lookup_order", "description": "查询订单状态",
                                          "parameters": {"type": "object",
                                                         "properties": {"order_id": {"type": "string"}},
                                                         "required": ["order_id"]}}},
        {"type": "function", "function": {"name": "get_logistics", "description": "查询物流轨迹",
                                          "parameters": {"type": "object",
                                                         "properties": {"order_id": {"type": "string"}},
                                                         "required": ["order_id"]}}},
    ]
    rec_tool1 = run(dict(
        id="K3-TOOL-01", title="单工具 tool_choice=required", model=MODEL_KIMI, key="kimi",
        payload={"messages": [{"role": "user", "content": "查询订单 A1001 的状态"}],
                 "tools": tools[:1], "tool_choice": "required", "max_tokens": 256},
        judge=lambda r: judge_tool(r, 1)))
    time.sleep(1)
    rec_tool2 = run(dict(
        id="K3-TOOL-02", title="双工具并行调用", model=MODEL_KIMI, key="kimi",
        payload={"messages": [{"role": "user", "content": "查订单 A1001 的状态和物流，两个都要查"}],
                 "tools": tools, "tool_choice": "required", "max_tokens": 384},
        judge=lambda r: judge_tool(r, 2)))
    time.sleep(1)

    calls = rec_tool2.get("tool_calls") or rec_tool1.get("tool_calls") or []
    if calls:
        msgs = [{"role": "user", "content": "查订单 A1001 的状态和物流，两个都要查"},
                {"role": "assistant", "content": None, "tool_calls": calls}]
        for i, c in enumerate(calls):
            msgs.append({"role": "tool", "tool_call_id": c.get("id"),
                         "content": json.dumps({"result": f"结果{i + 1}：状态已发货，预计明天送达"},
                                               ensure_ascii=False)})
        msgs.append({"role": "user", "content": "用一句话总结上面的查询结果"})

        def judge_replay(r: dict) -> dict:
            st = r.get("http_status")
            if st != 200:
                return {"verdict": "failed", "reason": f"回传 tool 消息后 HTTP {st}：{err_text(r)[:120]}"}
            text = (r.get("content") or "").strip()
            if not text:
                return {"verdict": "warning", "reason": "200 但正文为空（思考吃满预算）"}
            return {"verdict": "passed", "reason": f"200 正文 {len(text)} 字：{text[:60]!r}"}

        run(dict(id="K3-TOOL-03", title="tool 结果回传后给终答", model=MODEL_KIMI, key="kimi",
                 payload={"messages": msgs, "tools": tools, "max_tokens": 512},
                 judge=judge_replay))
        time.sleep(1)
    else:
        results.append({"case_id": "K3-TOOL-03", "title": "tool 结果回传", "model": MODEL_KIMI,
                        "verdict": "not_evaluable", "reason": "前置调用未产生 tool_calls，无可回传内容"})

    print("\n[C] Kimi K3 结构化输出 / 输出预算")
    run(dict(id="K3-JSON-01", title="response_format=json_object", model=MODEL_KIMI, key="kimi",
             payload={"messages": [{"role": "user", "content": "只输出 JSON：{\"ok\":true}"}],
                      "response_format": {"type": "json_object"}, "max_tokens": 256},
             judge=lambda r: {"verdict": "passed" if r.get("http_status") == 200 and (r.get("content") or "").strip().startswith("{") else "failed",
                              "reason": f"HTTP {r.get('http_status')} content={(r.get('content') or '')[:60]!r}"}))
    time.sleep(1)

    # 预算截断机制：同一个短提示，两档 max_tokens 对照（原 64K 档 TTFT-14 的省额度复现）
    run(dict(id="K3-BUDGET-128", title="max_tokens=128 是否被思考吃满", model=MODEL_KIMI, key="kimi",
             payload={"messages": [{"role": "user", "content": "37 乘以 83 等于多少？只回复数字"}],
                      "max_tokens": 128},
             judge=lambda r: {"verdict": "info",
                              "reason": f"finish={r.get('finish_reason')} completion={r.get('completion_tokens')} "
                                        f"reasoning={r.get('reasoning_tokens')} text={r.get('text_tokens')} "
                                        f"content_len={len(r.get('content') or '')}"}))
    time.sleep(1)
    run(dict(id="K3-BUDGET-1024", title="max_tokens=1024 同提示对照", model=MODEL_KIMI, key="kimi",
             payload={"messages": [{"role": "user", "content": "37 乘以 83 等于多少？只回复数字"}],
                      "max_tokens": 1024},
             judge=lambda r: {"verdict": "passed" if (r.get("content") or "").strip() else "failed",
                              "reason": f"finish={r.get('finish_reason')} completion={r.get('completion_tokens')} "
                                        f"reasoning={r.get('reasoning_tokens')} text={r.get('text_tokens')} "
                                        f"content={(r.get('content') or '')[:40]!r}"}))
    time.sleep(1)

    print("\n[D] Kimi K3 缓存命中抽查（1 预热 + 2 探测）")
    prefix = "K3-RETEST-CACHE-" + ("固定公共上下文用于缓存命中复测。" * 60)
    cache_rows = []
    for label in ("warm", "probe1", "probe2"):
        rec = call(f"K3-CACHE-{label}", MODEL_KIMI, "kimi",
                   {"messages": [{"role": "system", "content": prefix},
                                 {"role": "user", "content": "只回复 OK"}],
                    "max_tokens": 32})
        cache_rows.append(rec)
        time.sleep(2 if label != "warm" else 3)
    probes = [r for r in cache_rows[1:] if r.get("prompt_tokens")]
    ratios = [(r["cached_tokens"] / r["prompt_tokens"]) for r in probes]
    avg = sum(ratios) / len(ratios) if ratios else 0
    results.append({"case_id": "K3-CACHE-3x", "title": "稳定前缀缓存命中率", "model": MODEL_KIMI,
                    "verdict": "passed" if ratios and all(x >= 0.8 for x in ratios) else "failed",
                    "reason": f"prompt={[r.get('prompt_tokens') for r in cache_rows]} "
                              f"cached={[r.get('cached_tokens') for r in cache_rows]} "
                              f"probe_ratio={[round(x, 3) for x in ratios]} mean={avg:.1%}"})

    # ===================================================================== #
    # E. GLM 5.3 / 5.3-flash
    # ===================================================================== #
    def glm_block(model: str, tag: str) -> None:
        print(f"\n[E:{tag}] {model}")
        think = {"type": "enabled"}
        run(dict(id=f"G-IDENT:{tag}", title="身份自述（模型名/版本/上下文）", model=model, key="glm",
                 payload={"messages": [{"role": "user", "content":
                                        "请只回答三行：1) 你的模型名称与版本号；2) 你的最大上下文窗口（tokens）；"
                                        "3) 你的训练知识截止月份。"}],
                          "thinking": think, "max_tokens": 1024},
                 judge=lambda r: {"verdict": "info",
                                  "reason": f"HTTP {r.get('http_status')} content={(r.get('content') or '')[:200]!r}"}))
        time.sleep(1)

        for effort in ("low", "medium", "high"):
            run(dict(id=f"G-EFFORT-{effort}:{tag}", title=f"reasoning_effort={effort}", model=model, key="glm",
                     payload={"messages": [{"role": "user", "content": "只回复 OK"}],
                              "thinking": think, "reasoning_effort": effort, "max_tokens": 512},
                     judge=lambda r, e=effort: {
                         "verdict": "passed" if r.get("http_status") == 200 and (r.get("reasoning") or "")
                         else ("failed" if r.get("http_status") == 200 else "note"),
                         "reason": f"HTTP {r.get('http_status')} "
                                   + (f"error={err_of(r).get('type')}/{err_of(r).get('code')} "
                                      f"msg={err_text(r)[:140]}" if r.get("http_status") != 200
                                      else f"reasoning_tokens={r.get('reasoning_tokens')} "
                                           f"content_len={len(r.get('content') or '')}")}))
            time.sleep(1)

        run(dict(id=f"G-TEMP:{tag}", title="temperature=2.1", model=model, key="glm",
                 payload={"messages": simple, "max_tokens": 16, "temperature": 2.1},
                 judge=lambda r: judge_param(r, "temperature")))
        time.sleep(1)
        run(dict(id=f"G-TOPP:{tag}", title="top_p=1.5", model=model, key="glm",
                 payload={"messages": simple, "max_tokens": 16, "top_p": 1.5},
                 judge=lambda r: judge_param(r, "top_p")))
        time.sleep(1)
        run(dict(id=f"G-TOOL-REQ:{tag}", title="tool_choice=required（city 明确）", model=model, key="glm",
                 payload={"messages": [{"role": "user", "content": "查询北京的天气"}],
                          "tools": [{"type": "function", "function": {
                              "name": "get_weather", "description": "查询天气",
                              "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                             "required": ["city"]}}}],
                          "tool_choice": "required", "thinking": think, "max_tokens": 256},
                 judge=lambda r: judge_tool(r, 1)))
        time.sleep(1)

        # 多轮连续性：原 C02 用 max_tokens=32 导致思考吃满，这次给 512
        secret = "ORBIT-731"
        turns = [
            ([{"role": "user", "content": f"记住会话暗号：{secret}。只回复已记住。"}], "已记住", None),
            ([{"role": "user", "content": f"记住会话暗号：{secret}。只回复已记住。"},
              {"role": "assistant", "content": "已记住。"},
              {"role": "user", "content": "会话暗号是什么？只回复暗号。"}], secret, "turn2"),
            ([{"role": "user", "content": f"记住会话暗号：{secret}。只回复已记住。"},
              {"role": "assistant", "content": "已记住。"},
              {"role": "user", "content": "会话暗号是什么？只回复暗号。"},
              {"role": "assistant", "content": secret},
              {"role": "user", "content": "再次确认暗号，只回复暗号。"}], secret, "turn3"),
        ]
        hits = []
        for i, (msgs, want, label) in enumerate(turns):
            rec = call(f"G-MULTI-{i + 1}:{tag}", model, "glm",
                       {"messages": msgs, "thinking": think, "max_tokens": 512})
            text = (rec.get("content") or "")
            hits.append({"turn": i + 1, "http": rec.get("http_status"), "content": text[:80],
                         "reasoning_tokens": rec.get("reasoning_tokens"),
                         "text_tokens": rec.get("text_tokens"),
                         "hit_secret": want in text})
            time.sleep(1)
        ok_cont = all(h["http"] == 200 for h in hits) and all(h["hit_secret"] for h in hits[1:])
        results.append({"case_id": f"G-MULTI:{tag}", "title": "多轮连续性（max_tokens=512）", "model": model,
                        "verdict": "passed" if ok_cont else "failed",
                        "reason": json.dumps(hits, ensure_ascii=False)[:400]})

        # 缓存：1 预热 + 2 探测，前缀约 1000 token
        cprefix = f"GLM53-RETEST-CACHE-{tag}-" + ("固定公共上下文用于缓存命中复测。" * 90)
        rows = []
        for label in ("warm", "probe1", "probe2"):
            rec = call(f"G-CACHE-{label}:{tag}", model, "glm",
                       {"messages": [{"role": "system", "content": cprefix},
                                     {"role": "user", "content": "只回复 OK"}],
                        "thinking": think, "max_tokens": 64})
            rows.append(rec)
            time.sleep(3 if label == "warm" else 2)
        pr = [r for r in rows[1:] if r.get("prompt_tokens")]
        rr = [(r["cached_tokens"] / r["prompt_tokens"]) for r in pr]
        results.append({"case_id": f"G-CACHE:{tag}", "title": "稳定前缀缓存命中率", "model": model,
                        "verdict": "passed" if rr and all(x >= 0.8 for x in rr) else "failed",
                        "reason": f"prompt={[r.get('prompt_tokens') for r in rows]} "
                                  f"cached={[r.get('cached_tokens') for r in rows]} "
                                  f"probe_ratio={[round(x, 3) for x in rr]}"})

    glm_block(MODEL_GLM, "5.3")
    glm_block(MODEL_GLM_FLASH, "flash")

    # ---------- 汇总 ----------
    (OUT / "summary.json").write_text(json.dumps(
        {"base": BASE, "requests": _state["n"], "results": results,
         "rows": _state["rows"]}, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + "=" * 100)
    print(f"{'用例':24s} {'模型':16s} {'判定':10s} 说明")
    print("-" * 100)
    for r in results:
        print(f"{r['case_id']:24s} {r['model']:16s} {r['verdict']:10s} {r['reason'][:110]}")
    print("=" * 100)
    print(f"总请求 {_state['n']} 次；明细 {OUT / 'results.jsonl'}")


if __name__ == "__main__":
    sys.exit(main())
