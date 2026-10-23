#!/usr/bin/env python3
"""K3 自部署版 · thinking 开关真实收益 + 身份真伪 定点探测

要回答两个问题：
 Q1「thinking 关掉后 token 是不是真的减少」
     - 计量层：reasoning_tokens / completion_tokens / prompt_tokens 是否下降
     - 真实层：是否真的不再思考（流式看首 reasoning / 首 content 空窗、耗时、难题正确率）
     - 反差层：关闭态 prompt 反而更少 67 token，这 67 是什么
 Q2「是不是真的 K3」
     - 自述 / 知识截止（弱证据，只留痕不下结论）
     - 自家事实知识（月之暗面 / K2 / K3 谱系）
     - tokenizer 指纹（prompt_tokens 对固定文本的计数形态）
     - 思考链风格指纹（reasoning_content 特征）

设计：全程流式（stream_options.include_usage），只发短 prompt，MAX_REQUESTS 硬闸门。
用法：.venv/bin/python scripts/k3_thinking_probe.py <phase>
      phase = p0 | p1 | p2 | all   （默认 all）
"""
from __future__ import annotations

import os

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

BASE = "https://api.oinone.top/v1"
KEY = os.environ.get("OINONE_API_KEY", "")
MODEL = "kimi-k3"

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "retest_20261021" / "k3_thinking" / datetime.now().strftime("%Y%m%d-%H%M%S")
MAX_REQUESTS = 40
TIMEOUT = 300

_state = {"n": 0, "rows": []}


def _iter_sse(resp):
    """按字节缓冲切行，再逐行 utf-8 解码，产出 JSON chunk。"""
    buf = b""
    for piece in resp.iter_content(chunk_size=None):
        if not piece:
            continue
        buf += piece
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_s = line[5:].strip()
            if data_s == "[DONE]":
                return
            try:
                yield json.loads(data_s)
            except ValueError:
                continue


def call(case_id: str, payload: dict) -> dict:
    if _state["n"] >= MAX_REQUESTS:
        raise RuntimeError(f"达到请求预算上限 {MAX_REQUESTS}")
    _state["n"] += 1

    url = BASE + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}
    body = {"model": MODEL, "stream": True, "stream_options": {"include_usage": True}, **payload}

    rec = {
        "case_id": case_id,
        "request": body,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    t0 = time.perf_counter()
    # 流式分段时间线
    tl = {"first_reasoning": None, "first_content": None, "first_any": None}
    usage = None
    model_echo = None
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    finish = None
    try:
        with requests.post(url, headers=headers, json=body, timeout=TIMEOUT, stream=True) as resp:
            rec["http_status"] = resp.status_code
            rec["response_headers"] = {k.lower(): v for k, v in resp.headers.items()}
            if resp.status_code != 200:
                rec["raw_body"] = resp.text[:4000]
                try:
                    rec["body"] = resp.json()
                except ValueError:
                    pass
            else:
                prev_id = None
                # 必须按字节切行再 utf-8 解码：decode_unicode=True 会按 latin-1 退化解码，
                # 中文跨 chunk 时变成 "åªåæ°å­" 这类乱码，污染思考链指纹。
                for chunk in _iter_sse(resp):
                    if chunk.get("id") and chunk["id"] != prev_id:
                        prev_id = chunk["id"]
                    if chunk.get("model"):
                        model_echo = chunk["model"]
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for ch in chunk.get("choices") or []:
                        d = ch.get("delta") or {}
                        now = (time.perf_counter() - t0) * 1000
                        r = d.get("reasoning_content") or d.get("reasoning")
                        c = d.get("content")
                        if r:
                            reasoning_parts.append(r)
                            if tl["first_reasoning"] is None:
                                tl["first_reasoning"] = round(now, 1)
                            if tl["first_any"] is None:
                                tl["first_any"] = round(now, 1)
                        if c:
                            content_parts.append(c)
                            if tl["first_content"] is None:
                                tl["first_content"] = round(now, 1)
                            if tl["first_any"] is None:
                                tl["first_any"] = round(now, 1)
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
        rec["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as exc:  # noqa: BLE001
        rec["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        rec["transport_error"] = f"{type(exc).__name__}: {exc}"[:400]

    rec["timeline"] = tl
    rec["usage"] = usage or {}
    rec["response_model"] = model_echo
    rec["finish_reason"] = finish
    reasoning = "".join(reasoning_parts)
    content = "".join(content_parts)
    rec["reasoning"] = reasoning
    rec["content"] = content
    rec["reasoning_chars"] = len(reasoning)
    rec["content_chars"] = len(content)
    u = rec["usage"] or {}
    det = u.get("prompt_tokens_details") or {}
    cdet = u.get("completion_tokens_details") or {}
    rec["tokens"] = {
        "prompt": u.get("prompt_tokens"),
        "completion": u.get("completion_tokens"),
        "total": u.get("total_tokens"),
        "reasoning": cdet.get("reasoning_tokens"),
        "text": cdet.get("text_tokens"),
        "cached": det.get("cached_tokens"),
    }

    _state["rows"].append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "results.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    tk = rec["tokens"]
    print(f"  [{_state['n']:02d}] {case_id:26s} {rec.get('http_status')} "
          f"P={tk['prompt']} C={tk['completion']} R={tk['reasoning']} T={tk['text']} "
          f"| 1stR={tl['first_reasoning']} 1stC={tl['first_content']} e2e={rec['elapsed_ms']:.0f} "
          f"| rsn_chars={rec['reasoning_chars']} txt={rec['content_chars']}")
    return rec


def brief(rec: dict, n: int = 260) -> None:
    if rec.get("content"):
        print("        TXT:", rec["content"][:n].replace("\n", " ⏎ "))
    if rec.get("reasoning"):
        print("        RSN:", rec["reasoning"][:n].replace("\n", " ⏎ "))
    if rec.get("transport_error"):
        print("        ERR:", rec["transport_error"])
    if rec.get("raw_body"):
        print("        RAW:", rec["raw_body"][:n])


# --------------------------------------------------------------------------- #
# P0：thinking 三态 + 关闭形态的接受度（短题，验证参数与计量）
# --------------------------------------------------------------------------- #
Q_SIMPLE = "只回复数字：3071 + 0 = ?"

P0 = [
    ("P0-default", {"messages": [{"role": "user", "content": Q_SIMPLE}], "max_tokens": 512}),
    ("P0-disabled", {"messages": [{"role": "user", "content": Q_SIMPLE}], "max_tokens": 512,
                     "thinking": {"type": "disabled"}}),
    ("P0-enabled-all", {"messages": [{"role": "user", "content": Q_SIMPLE}], "max_tokens": 512,
                        "thinking": {"type": "enabled", "keep": "all"}}),
    ("P0-enabled-none", {"messages": [{"role": "user", "content": Q_SIMPLE}], "max_tokens": 512,
                         "thinking": {"type": "enabled", "keep": "none"}}),
    ("P0-think-false", {"messages": [{"role": "user", "content": Q_SIMPLE}], "max_tokens": 512,
                        "thinking": False}),
]


def phase_p0() -> None:
    print("\n== P0 thinking 形态与计量 ==")
    for cid, p in P0:
        brief(call(cid, p))


# --------------------------------------------------------------------------- #
# P1：tokenizer 指纹（max_tokens=1，只取 prompt_tokens）
# --------------------------------------------------------------------------- #
TEXT_ZH = ("北京是中华人民共和国的首都，位于华北平原的北部。故宫、天坛和长城是著名的历史遗迹。"
           "长江与黄河孕育了中华文明，两者流域面积广阔，人口稠密，农业与工业都很发达。")
TEXT_EN = ("The quick brown fox jumps over the lazy dog. Tokenization fingerprinting works "
           "by comparing how many pieces a fixed string is split into under different vocabularies.")
TEXT_CODE = ("def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n")
TEXT_MIX = ("中文与 English 混排，含数字 12345、符号 → ∑、emoji 🚀，以及罕见字 龘靐齉爩。")

P1 = [
    ("P1-zh", TEXT_ZH),
    ("P1-en", TEXT_EN),
    ("P1-code", TEXT_CODE),
    ("P1-mix", TEXT_MIX),
]


def phase_p1() -> None:
    print("\n== P1 tokenizer 计数指纹（max_tokens=1）==")
    for cid, txt in P1:
        rec = call(cid, {"messages": [{"role": "user", "content": txt}], "max_tokens": 1})
        rec["_text_len_bytes"] = len(txt.encode("utf-8"))
        rec["_text_len_chars"] = len(txt)
        print(f"        chars={len(txt)} bytes={len(txt.encode('utf-8'))} "
              f"prompt_tokens={rec['tokens']['prompt']}")


# --------------------------------------------------------------------------- #
# P2：身份与自家事实
# --------------------------------------------------------------------------- #
P2 = [
    ("P2-self", "你是哪家公司训练的什么模型？请只回答两行：公司名 / 模型名。"),
    ("P2-cutoff", "你的训练数据截止到什么时间？只回答年月。"),
    ("P2-lineage", "Kimi K2 是哪个团队发布的、总参数量与激活参数量各是多少？只回答这些事实。"),
    ("P2-k3", "Kimi K3 相比 K2 有哪些公开的技术变化？如果是你自己，请直接说明。"),
]


def phase_p2() -> None:
    print("\n== P2 身份与谱系 ==")
    for cid, q in P2:
        rec = call(cid, {"messages": [{"role": "user", "content": q}], "max_tokens": 700})
        brief(rec, 500)


# --------------------------------------------------------------------------- #
# P3：难度对照 —— 关闭 thinking 后是否真的不思考（质量 + 延迟 + 思考通道）
# 判据：若关闭态在难题上正确率不降、耗时也不降，说明思考仍在（藏起来或不计费）。
# --------------------------------------------------------------------------- #
Q_MID = "甲比乙大 5 岁，乙比丙大 3 岁，三人年龄之和是 62。甲多少岁？只回复数字。"
Q_HARD = "1 到 1000（含）之间，有多少个整数不能被 2、3、5 中任何一个整除？只回复数字。"
Q_TRAP = "笼子里有鸡和兔，共 35 个头、94 只脚。兔有多少只？只回复数字。"
Q_PROB = "从 1 到 20 这 20 个整数里随机取两个不同的数，两数之和为偶数的概率是多少？只回复最简分数。"

P3_CASES = [
    ("P3-mid", Q_MID, ["25"]),
    ("P3-hard", Q_HARD, ["266"]),
    ("P3-trap", Q_TRAP, ["12"]),
    ("P3-prob", Q_PROB, ["9/19"]),
]


def _answer_of(text: str) -> str:
    import re
    nums = re.findall(r"-?\d+(?:/\d+)?", text or "")
    return nums[-1] if nums else ""


def phase_p3() -> None:
    print("\n== P3 难度对照（disabled vs default）==")
    for cid, q, expect in P3_CASES:
        for mode in ("disabled", "default"):
            payload = {"messages": [{"role": "user", "content": q}], "max_tokens": 1536}
            if mode == "disabled":
                payload["thinking"] = {"type": "disabled"}
            rec = call(f"{cid}-{mode}", payload)
            got = _answer_of(rec.get("content", ""))
            rec["expect"] = expect
            rec["got"] = got
            rec["correct"] = got in expect
            print(f"        expect={expect} got={got!r} correct={rec['correct']} "
                  f"finish={rec.get('finish_reason')}")
    print("\n== P3b 关闭态显式要求逐步推理 ==")
    rec = call("P3b-disabled-cot", {
        "messages": [{"role": "user", "content": Q_HARD + " 回答前请逐步详细推理。"}],
        "max_tokens": 1536, "thinking": {"type": "disabled"}})
    print(f"        reasoning_chars={rec['reasoning_chars']} content_chars={rec['content_chars']}")
    brief(rec, 400)


def main() -> None:
    want = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    print(f"输出目录: {OUT}")
    if want in ("p0", "all"):
        phase_p0()
    if want in ("p1", "all"):
        phase_p1()
    if want in ("p2", "all"):
        phase_p2()
    if want in ("p3", "all"):
        phase_p3()
    keys = ("case_id", "http_status", "tokens", "timeline", "elapsed_ms", "finish_reason",
            "response_model", "reasoning_chars", "content_chars", "expect", "got", "correct",
            "content", "request")
    summary = [{k: r.get(k) for k in keys} for r in _state["rows"]]
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n完成，共 {_state['n']} 次请求。落盘: {OUT / 'results.jsonl'}")


if __name__ == "__main__":
    main()
