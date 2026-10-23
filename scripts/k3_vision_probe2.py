#!/usr/bin/env python3
"""K3 型号判别第二批
1) 视觉真实性：换图（5 个蓝圆）+ 无图对照，排除"从问句猜答案"
2) K3 独有 special token 判据：
   K3 词表（vocab_size 163840）新增 <|kimi_image_placeholder|> = id 163605；
   K2 词表（163596）没有该 token。把该串当普通文本发出去，
   K3 计 1 token，K2 会拆成多个 → prompt_tokens 差一个量级。
3) 图像 token 计量：同图不同尺寸，看 prompt_tokens 是否随图变化。
"""
from __future__ import annotations

import os

import base64
import json
import time
from pathlib import Path

import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from k3_vision_probe import make_png, OUT  # noqa: E402

BASE = "https://api.oinone.top/v1"
KEY = os.environ.get("OINONE_API_KEY", "")
MODEL = "kimi-k3"


def call(cid: str, payload: dict, show: int = 400) -> dict:
    body = {"model": MODEL, **payload}
    t0 = time.perf_counter()
    r = requests.post(BASE + "/chat/completions", headers={"Authorization": "Bearer " + KEY},
                      json=body, timeout=180)
    el = round((time.perf_counter() - t0) * 1000, 1)
    try:
        d = r.json()
    except ValueError:
        d = None
    rec = {"case_id": cid, "request": body, "http_status": r.status_code, "elapsed_ms": el, "body": d}
    with (OUT / "results.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    u = (d or {}).get("usage") or {}
    ch = ((d or {}).get("choices") or [{}])[0]
    m = ch.get("message") or {}
    print(f"\n[{cid}] HTTP {r.status_code} prompt={u.get('prompt_tokens')} "
          f"completion={u.get('completion_tokens')} ({el:.0f} ms)")
    if m.get("content"):
        print("  content:", m["content"][:show].replace("\n", " ⏎ "))
    if (d or {}).get("error"):
        print("  error:", json.dumps(d["error"], ensure_ascii=False)[:400])
    return rec


BLUE = (30, 80, 220)
png5 = make_png(120, 120, [(6, 6, 34, 34, BLUE), (43, 6, 71, 34, BLUE), (80, 6, 108, 34, BLUE),
                           (6, 43, 34, 71, BLUE), (43, 43, 71, 71, BLUE)])
b64_5 = base64.b64encode(png5).decode()

# 1) 视觉真实性
call("V3-five-blue", {"messages": [{"role": "user", "content": [
    {"type": "text", "text": "图中蓝色圆形/方块一共有几个？背景是什么颜色？格式：数量,背景色"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64_5}}]}],
    "max_tokens": 120, "thinking": {"type": "disabled"}})

# 2) 无图对照（同样的问句）
call("V4-no-image", {"messages": [{"role": "user", "content":
    "图中红色方块一共有几个？只回复数字。"}], "max_tokens": 120,
    "thinking": {"type": "disabled"}})

# 3) 大图：看 prompt_tokens 是否随图幅增长
big = make_png(512, 512, [(0, 0, 256, 512, BLUE), (256, 0, 512, 512, (250, 200, 20))])
b64_big = base64.b64encode(big).decode()
call("V5-big-two-color", {"messages": [{"role": "user", "content": [
    {"type": "text", "text": "这张图左半边和右半边各是什么颜色？格式：左色,右色"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64_big}}]}],
    "max_tokens": 120, "thinking": {"type": "disabled"}})

# 4) K3 独有 special token 判据
TARGETS = {
    "TOK-k3-image": "<|kimi_image_placeholder|>",
    "TOK-k2-imend": "<|im_end|>",           # K2/K3 都有，作为对照
    "TOK-control": "<|not_a_real_token|>",  # 两边都没有，作为上界
}
for cid, tok in TARGETS.items():
    call(cid, {"messages": [{"role": "user", "content": tok + " ok"}], "max_tokens": 1,
               "thinking": {"type": "disabled"}})

print(f"\n落盘: {OUT / 'results.jsonl'}")
