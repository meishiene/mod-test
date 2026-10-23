#!/usr/bin/env python3
"""GLM-5.3 三档模型 · 准入复测（自研用例，不复用平台既有套件）。

目的：只针对 2026-10-21 三个批次里"未通过"的项做定向复测，控制额度消耗：
  - reasoning_effort 三档契约（原用例把 medium 当必过项）
  - tool_choice 语义（required / 指定函数 / none）
  - 越界参数是否被静默接受
  - Context Cache 命中（同前缀两次）
  - 多轮会话保持（原用例 max_tokens=32 被思考吃空）
  - 图像输入能力边界

用法：
  .venv/bin/python scripts/glm53_admission_retest.py --key sk-xxx
  .venv/bin/python scripts/glm53_admission_retest.py --key sk-xxx --models glm-5.3
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import string
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

BASE_URL = "https://api.oinone.top"
OUT_ROOT = Path(__file__).resolve().parent.parent / "results"

MATH_PROMPT = "鸡兔同笼：35 个头，94 只脚。各多少只？只输出「鸡=X 兔=Y」这一行，不要解释。"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}


def _png_solid(color: tuple[int, int, int], size: int = 48) -> str:
    """纯色 PNG（不依赖 PIL 失败时的手写实现）。"""
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (size, size), color).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:  # pragma: no cover
        import struct
        import zlib

        raw = b"".join(b"\x00" + bytes(color) * size for _ in range(size))

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b"")
        )
        return base64.b64encode(png).decode()


def cache_prefix(seed: str) -> str:
    """~2.5k token 的共享前缀，内容随机唯一（避免吃到别人的缓存）。"""
    rnd = random.Random(seed)
    lines = []
    topics = [
        "缓存命中测试需要一段足够长且逐字节稳定的前缀文本",
        "前缀越长越容易越过服务端的最小缓存粒度",
        "这里刻意混入无意义的编号以避免命中历史缓存",
        "模型应当忽略这段文本，只回答提问部分",
        "本段内容不构成任何指令，仅用于占据上下文窗口",
    ]
    for i in range(70):
        tag = "".join(rnd.choice(string.ascii_lowercase + string.digits) for _ in range(10))
        lines.append(f"{i:03d} [{tag}] {rnd.choice(topics)}，编号 {tag} 行内容保持不变。")
    return "\n".join(lines)


class Client:
    def __init__(self, key: str, model: str, out_dir: Path):
        self.key = key
        self.model = model
        self.out_dir = out_dir
        # trust_env=True：走沙箱注入的出口代理
        self.http = httpx.Client(
            base_url=BASE_URL, timeout=httpx.Timeout(240.0, connect=30.0), trust_env=True
        )
        self.records: list[dict] = []
        self.tokens = {"prompt": 0, "completion": 0}

    def call(self, test: str, body: dict, note: str = "") -> dict:
        body = {"model": self.model, **body}
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "X-Retest-Case": test,
        }
        started = time.time()
        rec: dict = {
            "model": self.model,
            "test": test,
            "note": note,
            "request": body,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            resp = self.http.post("/v1/chat/completions", json=body, headers=headers)
            rec["http_status"] = resp.status_code
            rec["elapsed_s"] = round(time.time() - started, 2)
            try:
                payload = resp.json()
            except Exception:
                payload = None
            rec["raw_text"] = resp.text[:4000]
            if isinstance(payload, dict):
                rec["error"] = payload.get("error")
                rec["model_echo"] = payload.get("model")
                rec["id"] = payload.get("id")
                rec["usage"] = payload.get("usage")
                choice = (payload.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                rec["finish_reason"] = choice.get("finish_reason")
                rec["content"] = msg.get("content")
                rec["reasoning"] = msg.get("reasoning_content")
                rec["tool_calls"] = msg.get("tool_calls")
                u = rec.get("usage") or {}
                self.tokens["prompt"] += int(u.get("prompt_tokens") or 0)
                self.tokens["completion"] += int(u.get("completion_tokens") or 0)
        except Exception as exc:
            rec["http_status"] = 0
            rec["transport_error"] = f"{type(exc).__name__}: {exc}"
            rec["elapsed_s"] = round(time.time() - started, 2)
        self.records.append(rec)
        self._log(rec)
        return rec

    def _log(self, rec: dict) -> None:
        cid = rec["test"]
        print(
            f"  [{cid}] http={rec.get('http_status')} "
            f"finish={rec.get('finish_reason')} "
            f"tool={len(rec.get('tool_calls') or [])} "
            f"{rec.get('elapsed_s')}s",
            flush=True,
        )
        line = json.dumps(rec, ensure_ascii=False)
        with (self.out_dir / "requests.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def close(self) -> None:
        self.http.close()


def _reasoning_len(rec: dict) -> int:
    return len(rec.get("reasoning") or "")


def run_model(model: str, key: str, out_dir: Path) -> dict:
    c = Client(key, model, out_dir)
    print(f"\n=== {model} ===", flush=True)
    findings: list[dict] = []

    def add(test, verdict, severity, detail):
        findings.append(
            {"test": test, "verdict": verdict, "severity": severity, "detail": detail}
        )

    # ---------- A. reasoning_effort 三档契约 + 梯度（每档 2 样本，抗单次噪声）----------
    ladder: dict[str, list[dict]] = {"low": [], "high": [], "max": []}
    for rnd in range(2):
        for effort in ("low", "high", "max"):
            rec = c.call(
                f"A-effort-{effort}-r{rnd + 1}",
                {
                    "messages": [{"role": "user", "content": MATH_PROMPT}],
                    "max_tokens": 1200,
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": effort,
                },
                note=f"档位 {effort} 第 {rnd + 1} 次采样",
            )
            ladder[effort].append(rec)
            time.sleep(1)

    flat = [r for rs in ladder.values() for r in rs]
    ok_tiers = [t for t, rs in ladder.items() if all(r.get("http_status") == 200 for r in rs)]
    bad = {t: rs for t, rs in ladder.items() if any(r.get("http_status") != 200 for r in rs)}
    if len(ok_tiers) == 3:
        lens = {t: [_reasoning_len(r) for r in rs] for t, rs in ladder.items()}
        chars = {t: [len(r.get("content") or "") for r in rs] for t, rs in ladder.items()}
        nonempty = all((r.get("content") or "").strip() for r in flat)
        # 用两样本均值判断是否有方向性梯度
        means = {t: sum(v) / len(v) for t, v in lens.items()}
        grad = means["low"] <= means["high"] <= means["max"]
        spread = (max(means.values()) - min(means.values())) / max(
            max(means.values()), 1
        )
        if nonempty and grad and spread >= 0.15:
            add("A reasoning_effort 三档", "passed", "info",
                f"low/high/max 均 200、正文非空，思考长度均值单调递增 "
                f"low={means['low']:.0f} high={means['high']:.0f} max={means['max']:.0f}"
                f"（区分度 {spread * 100:.0f}%），逐次 {lens}")
        elif nonempty:
            add("A reasoning_effort 三档", "warning", "warning",
                f"三档均可用且正文非空，但思考长度未呈稳定梯度：均值 "
                f"low={means['low']:.0f} high={means['high']:.0f} max={means['max']:.0f}，"
                f"逐次 {lens}；正文长度 {chars}。档位被接受但强弱不可预期（能力级偏差）")
        else:
            add("A reasoning_effort 三档", "warning", "warning",
                f"三档均 200 但存在空正文，思考长度 {lens}，疑似思考吃满预算")
    else:
        add("A reasoning_effort 三档", "failed", "blocker",
            f"可用档位 {ok_tiers}，失败档位与错误："
            + json.dumps(
                {t: (rs[0].get("error") or rs[0].get("transport_error")) for t, rs in bad.items()},
                ensure_ascii=False,
            ))

    rec = c.call(
        "A-effort-medium",
        {
            "messages": [{"role": "user", "content": "只回复 OK"}],
            "max_tokens": 512,
            "thinking": {"type": "enabled"},
            "reasoning_effort": "medium",
        },
        note="medium 是否被拒（厂商契约：仅 low/high/max）",
    )
    time.sleep(1)
    if rec.get("http_status") in (400, 422):
        code = str(((rec.get("error") or {}).get("code")) or "")
        add("A medium 档位拒绝", "passed", "info",
            f"HTTP {rec.get('http_status')} code={code}，与厂商契约一致（仅 low/high/max），"
            f"原批次判 failed 属用例期望错设")
    elif rec.get("http_status") == 200:
        add("A medium 档位拒绝", "warning", "warning",
            "medium 被接受（200），与厂商文档「仅 low/high/max」不一致，属静默兼容")
    else:
        add("A medium 档位拒绝", "warning", "warning",
            f"HTTP {rec.get('http_status')}：{rec.get('error') or rec.get('transport_error')}")

    rec = c.call(
        "A-thinking-disabled",
        {
            "messages": [{"role": "user", "content": "只回复 OK"}],
            "max_tokens": 512,
            "thinking": {"type": "disabled"},
        },
        note="关闭思考是否被拒（模型始终思考）",
    )
    time.sleep(1)
    if rec.get("http_status") in (400, 422):
        add("A 关闭思考拒绝", "passed", "info",
            f"HTTP {rec.get('http_status')}，与「该模型始终思考」一致")
    else:
        add("A 关闭思考拒绝", "warning", "warning",
            f"thinking=disabled 返回 {rec.get('http_status')}，未拒绝")

    # ---------- B. tool_choice 语义 ----------
    rec = c.call(
        "B-required-text-prompt",
        {
            "messages": [{"role": "user", "content": "2 加 2 等于几？直接回答数字，不要调用任何工具。"}],
            "max_tokens": 900,
            "thinking": {"type": "enabled"},
            "tools": [WEATHER_TOOL],
            "tool_choice": "required",
        },
        note="required 在语义不适配时是否仍强制调用",
    )
    time.sleep(1)
    calls = rec.get("tool_calls") or []
    if rec.get("http_status") != 200:
        add("B tool_choice=required（不适配语境）", "warning", "warning",
            f"HTTP {rec.get('http_status')}：{rec.get('error')}")
    elif calls:
        add("B tool_choice=required（不适配语境）", "passed", "info",
            f"强制调用成立，tool={calls[0]['function']['name']}，符合 OpenAI 严格语义")
    else:
        add("B tool_choice=required（不适配语境）", "warning", "warning",
            "200 但未产生 tool_calls：required 属「建议性」而非强制，"
            "调用方若依赖 required 兜底会拿到纯文本（能力级偏差，非模型伪造）")

    rec = c.call(
        "B-forced-named-function",
        {
            "messages": [{"role": "user", "content": "北京今天天气怎么样？"}],
            "max_tokens": 900,
            "thinking": {"type": "enabled"},
            "tools": [WEATHER_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        },
        note="指定函数是否被精确执行",
    )
    time.sleep(1)
    calls = rec.get("tool_calls") or []
    names = [tc["function"]["name"] for tc in calls]
    if rec.get("http_status") == 200 and names == ["get_weather"]:
        detail = f"指定函数被调用，参数={calls[0]['function'].get('arguments')}"
        try:
            args = json.loads(calls[0]["function"].get("arguments") or "{}")
        except Exception:
            args = {}
        if str(args.get("city", "")).find("北京") >= 0:
            add("B 指定函数强制调用", "passed", "info", detail)
        else:
            add("B 指定函数强制调用", "warning", "warning", detail + "（参数未按语境填充）")
    else:
        add("B 指定函数强制调用", "failed" if rec.get("http_status") == 200 else "warning",
            "blocker" if rec.get("http_status") == 200 else "warning",
            f"HTTP {rec.get('http_status')} tool_calls={names} err={rec.get('error')}")

    rec = c.call(
        "B-none",
        {
            "messages": [{"role": "user", "content": "北京今天天气怎么样？简单说说。"}],
            "max_tokens": 900,
            "thinking": {"type": "enabled"},
            "tools": [WEATHER_TOOL],
            "tool_choice": "none",
        },
        note="none 是否被遵守",
    )
    time.sleep(1)
    calls = rec.get("tool_calls") or []
    if rec.get("http_status") == 200 and not calls and (rec.get("content") or "").strip():
        add("B tool_choice=none", "passed", "info", "未调用工具且给出文本回答")
    elif rec.get("http_status") == 200 and calls:
        add("B tool_choice=none", "failed", "blocker", "none 仍调用工具，参数契约被违反")
    else:
        add("B tool_choice=none", "warning", "warning",
            f"HTTP {rec.get('http_status')} err={rec.get('error')}")

    # ---------- C. 越界参数校验 ----------
    over = []
    for name, patch in (
        ("temperature=2.1", {"temperature": 2.1}),
        ("top_p=1.5", {"top_p": 1.5}),
    ):
        rec = c.call(
            f"C-{name}",
            {
                "messages": [{"role": "user", "content": "只回复 OK"}],
                "max_tokens": 400,
                "thinking": {"type": "enabled"},
                **patch,
            },
            note="越界参数是否被拒绝",
        )
        time.sleep(1)
        over.append((name, rec))
    accepted = [n for n, r in over if r.get("http_status") == 200]
    rejected = [n for n, r in over if r.get("http_status") in (400, 422)]
    if not accepted:
        add("C 越界参数校验", "passed", "info", f"越界值均被拒：{rejected}")
    elif not rejected:
        add("C 越界参数校验", "warning", "warning",
            f"越界值 {accepted} 全部返回 200：网关未做取值域校验，"
            "错误配置不会报错（参数静默容忍，不改变结果正确性）")
    else:
        add("C 越界参数校验", "warning", "warning",
            f"部分拒绝：拒绝={rejected}，接受={accepted}（校验不一致）")

    # ---------- D. Context Cache：连发 vs 有间隔（定位原批次 33% 成因）----------
    def cached_of(r: dict) -> tuple[int, int]:
        u = r.get("usage") or {}
        p = int(u.get("prompt_tokens") or 0)
        cd = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
        return cd, p

    def cache_call(tag: str, prefix: str, question: str, note: str) -> dict:
        return c.call(
            tag,
            {
                "messages": [
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": question},
                ],
                "max_tokens": 300,
                "thinking": {"type": "enabled"},
            },
            note=note,
        )

    # D1：冷启动后立刻重发（复刻原批次无间隔节奏）
    p_rapid = cache_prefix(f"{model}-rapid-{uuid.uuid4().hex}")
    cold = cache_call("D1-rapid-cold", p_rapid, "这段文本一共多少行？只回数字。", "冷启动（连发组）")
    hot = cache_call("D1-rapid-warm", p_rapid, "这段文本一共多少行？只回数字。", "冷启动后立刻重发，无间隔")
    cd_rapid, p_rapid_tok = cached_of(hot)
    rate_rapid = round(cd_rapid / p_rapid_tok * 100, 1) if p_rapid_tok else 0.0
    time.sleep(1)

    # D2：冷启动后间隔 3 秒再连发两次
    p_paced = cache_prefix(f"{model}-paced-{uuid.uuid4().hex}")
    cache_call("D2-paced-cold", p_paced, "这段文本第一行是什么？只回该行。", "冷启动（间隔组）")
    time.sleep(3)
    warm_recs = []
    for i in range(2):
        warm_recs.append(
            cache_call(
                f"D2-paced-warm-{i + 1}",
                p_paced,
                "这段文本第一行是什么？只回该行。" if i == 0 else "这段文本一共多少行？只回数字。",
                f"间隔 3 秒后第 {i + 1} 次复用同前缀",
            )
        )
        time.sleep(1)
    paced = [(cached_of(r)[0], cached_of(r)[1]) for r in warm_recs]
    paced_rates = [round(a / b * 100, 1) if b else 0.0 for a, b in paced]
    best_paced = max(paced_rates) if paced_rates else 0.0

    detail = (
        f"连发（无间隔）命中 {rate_rapid}%（cached={cd_rapid}/prompt={p_rapid_tok}）；"
        f"间隔 3 秒后复用命中 {paced_rates}%（cached/prompt={paced}）"
    )
    if best_paced >= 80 and rate_rapid < 50:
        add("D 上下文缓存命中", "passed", "info",
            detail + "。结论：缓存能力真实存在，命中依赖请求间隔（首请求写入未完成即重发会 0 命中），"
            "原批次 C01「33.2% 不达标」是连发节奏导致的测量偏差，不是模型缓存能力缺失")
    elif best_paced >= 80:
        add("D 上下文缓存命中", "passed", "info",
            detail + "。结论：同前缀稳定命中，缓存能力达标")
    elif best_paced > 0:
        add("D 上下文缓存命中", "warning", "warning",
            detail + "。结论：缓存可用但不稳定，符合多上游通道路由且无会话粘性的特征，"
            "非模型能力缺失，但调用方无法稳定拿到缓存折扣")
    else:
        add("D 上下文缓存命中", "warning", "warning",
            detail + "。结论：两次冷启动均未产生可复用缓存，等同无上下文缓存能力（能力级缺失）")

    # ---------- E. 多轮会话保持（修正预算） ----------
    token = "ZQ-" + uuid.uuid4().hex[:4].upper()
    sid = f"retest-session-{uuid.uuid4().hex[:8]}"
    r1 = c.call(
        "E-memory-turn1",
        {
            "messages": [
                {"role": "user", "content": f"请记住口令 {token}，我稍后会问。现在只回复「收到」。"}
            ],
            "max_tokens": 800,
            "thinking": {"type": "enabled"},
            "user": sid,
        },
        note="写入口令",
    )
    time.sleep(1)
    r2 = c.call(
        "E-memory-turn2",
        {
            "messages": [
                {"role": "user", "content": f"请记住口令 {token}，我稍后会问。现在只回复「收到」。"},
                {"role": "assistant", "content": (r1.get("content") or "收到")},
                {"role": "user", "content": "刚才的口令是什么？只回复口令本身。"},
            ],
            "max_tokens": 800,
            "thinking": {"type": "enabled"},
            "user": sid,
        },
        note="取回口令",
    )
    time.sleep(1)
    ans = (r2.get("content") or "")
    if r2.get("http_status") == 200 and token in ans:
        add("E 多轮会话保持", "passed", "info",
            f"口令 {token} 正确取回（turn1 正文={len(r1.get('content') or '')} 字）")
    elif r2.get("http_status") == 200 and ans.strip():
        add("E 多轮会话保持", "warning", "warning",
            f"有回答但未带口令，实际回答={ans[:80]!r}")
    else:
        add("E 多轮会话保持", "warning", "warning",
            f"HTTP {r2.get('http_status')}，正文长度={len(ans)}，"
            f"finish={r2.get('finish_reason')}，"
            "需与 CORE-07 已通过的结论对照（原 C02 用 max_tokens=32 + 强制思考，应答被吃空）")

    # ---------- F. 图像输入边界 ----------
    img = _png_solid((229, 57, 53))
    rec = c.call(
        "F-vision-solid-red",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这张图片的主色是什么？只回答颜色名（中文或英文）。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
                    ],
                }
            ],
            "max_tokens": 600,
            "thinking": {"type": "enabled"},
        },
        note="纯红 48x48 PNG，验证是否接受图像输入",
    )
    time.sleep(1)
    ans = (rec.get("content") or "").lower()
    if rec.get("http_status") == 200 and ans.strip():
        if "红" in ans or "red" in ans:
            add("F 图像输入", "passed", "info", f"识色正确：{ans[:60]!r}")
        else:
            add("F 图像输入", "warning", "warning", f"接受图像但识色偏差：{ans[:60]!r}")
    elif rec.get("http_status") in (400, 422):
        add("F 图像输入", "blocked", "warning",
            f"图像输入被拒 HTTP {rec.get('http_status')}：{(rec.get('error') or {}).get('message')}"
            "（该模型/渠道未开放多模态，属能力边界，不判伪造）")
    else:
        add("F 图像输入", "warning", "warning",
            f"HTTP {rec.get('http_status')} err={rec.get('error') or rec.get('transport_error')}")

    blockers = [f for f in findings if f["verdict"] == "failed"]
    warnings = [f for f in findings if f["verdict"] == "warning"]
    blocked = [f for f in findings if f["verdict"] == "blocked"]
    summary = {
        "model": model,
        "base_url": BASE_URL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "calls": len(c.records),
        "tokens": c.tokens,
        "findings": findings,
        "counts": {
            "passed": len([f for f in findings if f["verdict"] == "passed"]),
            "warning": len(warnings),
            "blocked": len(blocked),
            "failed": len(blockers),
        },
        "admission": "not_admitted" if blockers else "admitted",
    }
    c.close()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=os.environ.get("GLM_RETEST_KEY", ""))
    ap.add_argument("--models", default="glm-5.3,glm-5.3-flash,glm-5.3-flashx")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if not args.key:
        print("缺少 --key", file=sys.stderr)
        return 2

    stamp = args.tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = OUT_ROOT / f"GLM53_RETEST-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        results.append(run_model(model, args.key, out_dir))

    summary = {
        "schema": "glm53.retest.v1",
        "record_id": stamp,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "models": results,
        "total_calls": sum(r["calls"] for r in results),
        "total_tokens": {
            k: sum(r["tokens"][k] for r in results) for k in ("prompt", "completion")
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== 汇总 ===")
    for r in results:
        print(f"{r['model']}: {r['counts']} calls={r['calls']} tokens={r['tokens']} "
              f"admission={r['admission']}")
        for f in r["findings"]:
            print(f"  [{f['verdict']:8}] {f['test']}: {f['detail'][:110]}")
    print(f"\n输出目录：{out_dir}")
    print(f"总调用 {summary['total_calls']} 次，tokens={summary['total_tokens']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
