#!/usr/bin/env python3
"""K3 型号判别：原生多模态判据

依据（ModelScope moonshotai/Kimi-K3 官方 config.json）：
  architectures = KimiK3ForConditionalGeneration
  vision_config 存在；image_placeholder = <|kimi_image_placeholder|>（token 163605）
  text_config.max_position_embeddings = 1048576（1M）
K2（moonshotai/Kimi-K2-Instruct）是纯文本模型，config 无 vision_config。
⇒ 若端点是真 K3，图像输入必须被接受并能识图；纯文本上游会 400 或忽略图片。

用法：.venv/bin/python scripts/k3_vision_probe.py
"""
from __future__ import annotations

import os

import base64
import io
import json
import struct
import time
import zlib
from pathlib import Path

import requests

BASE = "https://api.oinone.top/v1"
KEY = os.environ.get("OINONE_API_KEY", "")
MODEL = "kimi-k3"
OUT = Path(__file__).resolve().parent / "retest_20261021" / "k3_thinking" / "vision"
OUT.mkdir(parents=True, exist_ok=True)


def make_png(w: int, h: int, blocks: list[tuple[int, int, int, int, tuple[int, int, int]]]) -> bytes:
    """白色底 + 若干实心色块。blocks: (x0,y0,x1,y1,(r,g,b))"""
    rows = []
    for y in range(h):
        row = bytearray([0])  # filter type 0
        for x in range(w):
            px = (255, 255, 255)
            for x0, y0, x1, y1, color in blocks:
                if x0 <= x < x1 and y0 <= y < y1:
                    px = color
                    break
            row.extend(px)
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


RED = (220, 40, 40)
png = make_png(96, 96, [(8, 8, 40, 40, RED), (56, 8, 88, 40, RED), (32, 56, 64, 88, RED)])
(OUT / "probe.png").write_bytes(png)
b64 = base64.b64encode(png).decode()
print(f"测试图: 96x96 白底 + 3 个红方块, {len(png)} bytes")


def call(cid: str, payload: dict) -> dict:
    body = {"model": MODEL, **payload}
    t0 = time.perf_counter()
    r = requests.post(BASE + "/chat/completions", headers={"Authorization": "Bearer " + KEY},
                      json=body, timeout=180)
    el = round((time.perf_counter() - t0) * 1000, 1)
    try:
        d = r.json()
    except ValueError:
        d = None
    rec = {"case_id": cid, "request": body, "http_status": r.status_code,
           "elapsed_ms": el, "body": d, "raw": r.text[:2000]}
    with (OUT / "results.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"\n[{cid}] HTTP {r.status_code} ({el:.0f} ms)")
    if isinstance(d, dict):
        u = d.get("usage") or {}
        print("  usage:", {k: u.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")})
        ch = (d.get("choices") or [{}])[0]
        m = ch.get("message") or {}
        print("  finish:", ch.get("finish_reason"))
        print("  content:", (m.get("content") or "")[:400].replace("\n", " ⏎ "))
        rsn = m.get("reasoning_content") or ""
        if rsn:
            print("  reasoning:", rsn[:400].replace("\n", " ⏎ "))
        if d.get("error"):
            print("  error:", json.dumps(d["error"], ensure_ascii=False)[:500])
    else:
        print("  raw:", r.text[:500].replace("\n", " "))
    return rec


# 1) 结构化图像输入（OpenAI 兼容形态）
call("V1-image-count", {
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "这张图里有几个红色方块？只回复数字。"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
    ]}],
    "max_tokens": 200, "thinking": {"type": "disabled"},
})

# 2) 同图，直接问主色（防"数错了但确实看到图"）
call("V2-image-color", {
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "这张图片里出现的彩色图形是什么颜色？背景是什么颜色？各一个词。"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
    ]}],
    "max_tokens": 200, "thinking": {"type": "disabled"},
})

print(f"\n落盘: {OUT / 'results.jsonl'}")
