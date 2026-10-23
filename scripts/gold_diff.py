#!/usr/bin/env python3
"""用**中转侧实测的 token 数**对齐原厂金标，判断"这是不是它自称的那家模型"。

两种用法：

    # 1) 不用任何 key：验证判别器本身有效（加人工 offset 仍能认回本族）
    .venv/bin/python scripts/gold_diff.py --selfcheck

    # 2) 有一把中转 key（你自己的中转就行）：实测一把
    python scripts/gold_diff.py --base-url https://api.example.com/v1 \
        --api-key sk-xxx --model glm-5.3

判据说明（务必先读，否则会误判）：

- 只读 ``usage.prompt_tokens``，不看回答内容；所以 ``max_tokens`` 给小值即可，成本约等于 0。
- 中转若**自己重算 usage**（不透传上游计数），本判据会把它读成"异族"。这是本方法的
  已知盲区——所以结论是 ``inconsistent`` 时只能记"token 数不可信"，**不能**判掉包。
- 声明模型在注册表里没有可用金标（豆包、或金标未采集）时直接 ``no_gold``，不猜。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from gold.build import GOLD_PATH  # noqa: E402
from gold.compare import compare, selfcheck  # noqa: E402
from gold.probes import CORPUS_VERSION, all_samples  # noqa: E402
from gold.registry import resolve  # noqa: E402

OUT_ROOT = ROOT / ".pytest-tmp" / "gold-diff"


def _load_gold() -> dict:
    if not GOLD_PATH.exists():
        raise SystemExit(f"金标不存在：{GOLD_PATH}\n先跑一次 scripts/gold_build.py")
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def _measure(client: httpx.Client, base_url: str, api_key: str, model: str,
             samples: dict[str, str], max_tokens: int, delay: float) -> tuple[dict[str, int], list[dict]]:
    """逐探针发一条极短请求，取 ``usage.prompt_tokens``。失败项记录原因、不进向量。"""
    measured: dict[str, int] = {}
    rows: list[dict] = []
    url = base_url.rstrip("/") + "/chat/completions"
    for name, text in samples.items():
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        started = time.time()
        try:
            resp = client.post(url, json=payload,
                               headers={"Authorization": f"Bearer {api_key}"})
            elapsed = round((time.time() - started) * 1000)
            body = None
            try:
                body = resp.json()
            except Exception:
                body = None
            usage = (body or {}).get("usage") or {}
            pt = usage.get("prompt_tokens")
            row = {
                "probe": name,
                "http": resp.status_code,
                "prompt_tokens": pt,
                "ms": elapsed,
                "error": None if isinstance(pt, int) else _error_hint(resp, body),
            }
            if isinstance(pt, int):
                measured[name] = pt
        except Exception as exc:
            rows.append({"probe": name, "http": None, "prompt_tokens": None,
                         "ms": round((time.time() - started) * 1000),
                         "error": f"{type(exc).__name__}: {exc}"})
        else:
            rows.append(row)
        if delay:
            time.sleep(delay)
    return measured, rows


def _error_hint(resp: httpx.Response, body) -> str:
    """把「200 但非 JSON」「4xx 错误信封」分开写清楚——两者的处置完全不同。"""
    if resp.status_code == 200 and body is None:
        ctype = resp.headers.get("content-type", "")
        return f"200 但响应非 JSON（content-type={ctype!r}），请求可能没到模型"
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return f"HTTP {resp.status_code} error={err.get('type')}/{err.get('code')} {str(err.get('message'))[:80]}"
        if err:
            return f"HTTP {resp.status_code} {str(err)[:100]}"
    return f"HTTP {resp.status_code} 未返回 usage.prompt_tokens"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true", help="不联网，验证判别器")
    ap.add_argument("--base-url", default="", help="如 https://api.example.com/v1")
    ap.add_argument("--api-key", default="", help="也可用 GOLD_PROBE_KEY 环境变量")
    ap.add_argument("--model", default="", help="中转侧模型名，用于取「声明家族」")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--delay", type=float, default=0.2)
    ap.add_argument("--no-proxy", action="store_true", help="直连（不走 HTTP(S)_PROXY）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    gold = _load_gold()

    if args.selfcheck:
        res = selfcheck(gold)
        print(f"自检 {res['passed']}/{res['total']} 通过")
        for slug, best in res["ties"]:
            print(f"  · {slug} ≡ {best}（同族，不进判别分母）")
        for row in res["failed"]:
            print(f"  ✗ {row['slug']} offset={row['offset']} → {row['verdict']} best={row['best']}")
        return 0 if res["passed"] == res["total"] else 1

    api_key = args.api_key or __import__("os").environ.get("GOLD_PROBE_KEY", "")
    if not (args.base_url and api_key and args.model):
        ap.error("需要 --base-url / --api-key（或 GOLD_PROBE_KEY）/ --model")

    declared = resolve(args.model)
    samples = all_samples()
    with httpx.Client(timeout=90, trust_env=not args.no_proxy) as client:
        measured, rows = _measure(client, args.base_url, api_key, args.model,
                                  samples, args.max_tokens, args.delay)

    result = compare(measured, gold, declared_slug=declared.slug if declared else None)
    payload = {
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "corpus_version": CORPUS_VERSION,
        "base_url": args.base_url,
        "model": args.model,
        "declared_entry": declared.slug if declared else None,
        "declared_mapping_confidence": declared.mapping_confidence if declared else None,
        "probes_ok": len(measured),
        "probes_total": len(samples),
        "result": result.as_dict(),
        "rows": rows,
    }
    out_dir = Path(args.out) if args.out else OUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gold_diff.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                            encoding="utf-8")

    print(f"模型 {args.model}｜声明 {payload['declared_entry'] or '无金标'}"
          f"｜实测探针 {len(measured)}/{len(samples)}")
    print(f"判定：{result.verdict}")
    if result.best:
        print(f"  最像 {result.best.slug}（offset={result.best.offset}, 残差={result.best.score}, "
              f"散布={result.best.spread}, 差分={result.best.pair_score}）")
    for note in result.notes:
        print(f"  · {note}")
    failed = [r for r in rows if r["error"]]
    if failed:
        print(f"  ⚠ {len(failed)} 条探针没拿到 usage，例如：{failed[0]['probe']} → {failed[0]['error']}")
    print(f"明细 {out_dir / 'gold_diff.json'}")
    return 0 if result.verdict in ("matched", "family_mismatch") else 2


if __name__ == "__main__":
    raise SystemExit(main())
