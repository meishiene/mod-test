#!/usr/bin/env python3
"""一次性出图任务：sapizzz 网关 gpt-image-2.5 双模型 × 5 档强度 × 2K/4K = 20 张。

不是常驻套件，不登记进 protocols/server.py；产物落在 results/<OUT>/ 下：
  images/*.png        逐张原图
  requests.jsonl      逐条请求证据（含耗时、字节数、sha256、尺寸、usage）
  summary.json        汇总
  report.md           结论表
  gallery.html        缩略图对比页（可离线打开）
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

DEFAULT_BASE_URL = "https://img-api.sapizzz.cc"
# API Key 一律从环境变量 / 命令行读取，不写死在仓库里（历史版本曾硬编码，已移除）
DEFAULT_API_KEY = os.environ.get("SAPIZZZ_API_KEY", "")
MODELS = ("gpt-image-2.5-flare", "gpt-image-2.5-sunburst")
QUALITIES = ("low", "medium", "high", "xhigh", "max")
SIZES = (("2k", "2560x1440"), ("4k", "3840x2160"))
PROMPT_FILE = "results/SAPIZZZ_IMAGE25_PROMPT.txt"

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}
MAX_ATTEMPTS = 3


def _png_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _decode_image(item: dict) -> tuple[bytes | None, str]:
    encoded = item.get("b64_json")
    if isinstance(encoded, str) and encoded:
        part = encoded.split(",", 1)[1] if encoded.startswith("data:") and "," in encoded else encoded
        try:
            return base64.b64decode(part), "b64_json"
        except Exception:  # noqa: BLE001
            return None, "b64_decode_failed"
    if isinstance(item.get("url"), str) and item["url"]:
        return None, "url"
    return None, "empty"


async def _one(
    client: httpx.AsyncClient,
    *,
    sem: asyncio.Semaphore,
    images_dir: Path,
    model: str,
    quality: str,
    size_label: str,
    size: str,
    prompt: str,
    timeout: float,
) -> dict:
    short = model.replace("gpt-image-2.5-", "")
    name = f"{short}-{size_label}-{quality}"
    request_id = f"SAPIZZZ-{name}-{int(time.time())}"
    payload = {"model": model, "prompt": prompt, "size": size, "quality": quality, "n": 1}
    row: dict = {
        "task": name,
        "model": model,
        "size_label": size_label,
        "size": size,
        "quality": quality,
        "n": 1,
        "endpoint": f"{str(client.base_url).rstrip('/')}/v1/images/generations",
        "request_id": request_id,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_chars": len(prompt),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    headers = {
        "Authorization": f"Bearer {DEFAULT_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Request-ID": request_id,
    }
    async with sem:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            row["attempts"] = attempt
            started = time.perf_counter()
            try:
                resp = await client.post(
                    "/v1/images/generations", json=payload, headers=headers, timeout=timeout
                )
                row["http_status"] = resp.status_code
                row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                row["response_headers"] = {
                    k.lower(): v
                    for k, v in resp.headers.items()
                    if k.lower() in {"content-type", "server", "via", "x-request-id", "x-one-api-request-id"}
                }
                if resp.status_code in RETRY_STATUS and attempt < MAX_ATTEMPTS:
                    await asyncio.sleep(2.0 * attempt)
                    continue
                try:
                    body = resp.json()
                except ValueError:
                    row["error"] = "non_json_response"
                    row["body_preview"] = resp.text[:1500]
                    return row
                if not resp.is_success:
                    row["error"] = "http_error"
                    row["error_body"] = json.dumps(body, ensure_ascii=False)[:1500]
                    return row
                items = body.get("data") if isinstance(body, dict) else None
                items = items if isinstance(items, list) else []
                row["image_count"] = len(items)
                row["usage"] = body.get("usage")
                row["response_model"] = body.get("model")
                row["response_id"] = body.get("id")
                row["response_keys"] = sorted(body.keys()) if isinstance(body, dict) else []
                if not items:
                    row["error"] = "no_image_data"
                    row["error_body"] = json.dumps(body, ensure_ascii=False)[:1500]
                    return row
                item = items[0] if isinstance(items[0], dict) else {}
                image_bytes, source = _decode_image(item)
                row["result_source"] = source
                if image_bytes is None and source == "url":
                    fetched = await client.get(item["url"], timeout=timeout)
                    if fetched.is_success:
                        image_bytes = fetched.content
                        row["download_status"] = fetched.status_code
                    else:
                        row["download_status"] = fetched.status_code
                if not image_bytes:
                    row["error"] = "image_payload_empty"
                    return row
                out = images_dir / f"{name}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(image_bytes)
                row["image_file"] = str(out)
                row["image_bytes"] = len(image_bytes)
                row["image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
                dims = _png_dimensions(image_bytes)
                row["image_width"], row["image_height"] = dims if dims else (None, None)
                row["status"] = "passed"
                return row
            except (httpx.TimeoutException, httpx.HTTPError) as exc:
                row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                row["error"] = type(exc).__name__
                row["error_message"] = str(exc)[:500]
                if attempt < MAX_ATTEMPTS:
                    await asyncio.sleep(2.0 * attempt)
                    continue
                return row
            except Exception as exc:  # noqa: BLE001
                row["error"] = type(exc).__name__
                row["error_message"] = str(exc)[:500]
                return row
    return row


def _thumbnail_data_uri(path: Path, max_side: int = 1000) -> str | None:
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=84)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


def build_gallery(rows: list[dict], out_dir: Path, prompt: str) -> Path:
    cells: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        cells[(r["model"], r["quality"], r["size_label"])] = r

    def card(r: dict | None) -> str:
        if r is None:
            return '<div class="card empty">未执行</div>'
        if r.get("status") != "passed":
            err = r.get("error") or "failed"
            body = (r.get("error_body") or r.get("error_message") or "")[:400]
            return (
                f'<div class="card fail"><div class="bad">失败：{err}</div>'
                f'<pre>{body}</pre></div>'
            )
        uri = _thumbnail_data_uri(Path(r["image_file"]))
        rel = os.path.relpath(r["image_file"], out_dir)
        mb = r["image_bytes"] / 1024 / 1024
        meta = (
            f"HTTP {r['http_status']} · {r['elapsed_seconds']}s · {mb:.2f} MB · "
            f"{r['image_width']}×{r['image_height']}"
        )
        img = f'<img loading="lazy" src="{uri}" alt="{r["task"]}">' if uri else "<div>缩略图生成失败</div>"
        return f'<a class="card" href="{rel}" target="_blank">{img}<div class="meta">{meta}</div></a>'

    sections = []
    for model in MODELS:
        short = model.replace("gpt-image-2.5-", "")
        header = "".join(f"<th>{lbl.upper()}</th>" for lbl, _ in SIZES)
        body_rows = []
        for q in QUALITIES:
            tds = "".join(f"<td>{card(cells.get((model, q, lbl)))}</td>" for lbl, _ in SIZES)
            body_rows.append(f"<tr><th class='q'>{q}</th>{tds}</tr>")
        sections.append(
            f'<h2>{short} <span class="sub">{model}</span></h2>'
            f"<table class='grid'><thead><tr><th></th>{header}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody></table>"
        )

    passed = sum(1 for r in rows if r.get("status") == "passed")
    total_mb = sum(r.get("image_bytes") or 0 for r in rows) / 1024 / 1024
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>sapizzz · gpt-image-2.5 双模型 5 档强度 2K/4K 出图</title>
<style>
:root{{--bg:#f6f3ee;--card:#fff;--ink:#34291d;--line:#ded5c6;--green:#59644c;--warn:#8a4b2a}}
*{{box-sizing:border-box}}
body{{margin:0;padding:32px 28px 64px;background:var(--bg);color:var(--ink);
font:14px/1.6 -apple-system,"PingFang SC","Songti SC",serif}}
h1{{font-size:24px;margin:0 0 6px;letter-spacing:.5px}}
.lead{{color:#6b6152;margin:0 0 22px}}
.kpis{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:26px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 16px}}
.kpi b{{font-size:18px}}
h2{{font-size:17px;margin:30px 0 12px;display:flex;align-items:baseline;gap:10px}}
h2 .sub{{font-weight:400;color:#8b8271;font-size:13px}}
table.grid{{width:100%;border-collapse:separate;border-spacing:14px;table-layout:fixed}}
table.grid th{{font-weight:600;color:#6b6152;font-size:13px;text-align:left;padding:0}}
th.q{{width:64px;vertical-align:middle;color:var(--ink)}}
.card{{display:block;background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:8px;text-decoration:none;color:inherit;transition:.15s}}
.card:hover{{border-color:var(--green);box-shadow:0 6px 18px rgba(52,41,29,.10)}}
.card img{{width:100%;border-radius:8px;display:block}}
.card .meta{{font-size:12px;color:#6b6152;padding:8px 2px 2px;font-variant-numeric:tabular-nums}}
.card.fail{{border-color:#c9a38a;color:var(--warn);font-size:12px}}
.card.fail .bad{{font-weight:600;margin-bottom:6px}}
.card.fail pre{{white-space:pre-wrap;word-break:break-all;font-size:11px;margin:0;color:#7d5b45}}
details{{margin-top:34px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 18px}}
summary{{cursor:pointer;font-weight:600}}
pre.prompt{{white-space:pre-wrap;font-size:12px;color:#5c5344;max-height:420px;overflow:auto}}
</style></head>
<body>
<h1>sapizzz · gpt-image-2.5 双模型 · 5 档强度 · 2K / 4K</h1>
<p class="lead">端点 <code>{DEFAULT_BASE_URL}/v1/images/generations</code>　
提示词：高端家居香氛产品广告海报（逐字固定，见文末）</p>
<div class="kpis">
<div class="kpi">成功 <b>{passed}/{len(rows)}</b></div>
<div class="kpi">图片总体积 <b>{total_mb:.1f} MB</b></div>
<div class="kpi">矩阵 <b>2 模型 × 5 档 × 2 分辨率</b></div>
</div>
{''.join(sections)}
<details><summary>请求提示词全文（{len(prompt)} 字，20 次请求逐字一致）</summary>
<pre class="prompt">{prompt}</pre></details>
</body></html>
"""
    path = out_dir / "gallery.html"
    path.write_text(html, encoding="utf-8")
    return path


def build_report(rows: list[dict], out_dir: Path, wall: float) -> Path:
    lines = ["# sapizzz · gpt-image-2.5 双模型 5 档强度 2K/4K 出图", ""]
    passed = [r for r in rows if r.get("status") == "passed"]
    lines.append(f"- 端点：`{DEFAULT_BASE_URL}/v1/images/generations`")
    lines.append(f"- 矩阵：2 模型 × 5 档（low/medium/high/xhigh/max） × 2 分辨率（2K 2560×1440 / 4K 3840×2160） = {len(rows)} 次请求")
    lines.append(f"- 成功 {len(passed)}/{len(rows)}，墙钟 {wall:.0f}s，图片总体积 "
                 f"{sum(r.get('image_bytes') or 0 for r in rows)/1024/1024:.1f} MB")
    lines.append("")
    lines.append("| 模型 | 档位 | 分辨率 | HTTP | 耗时(s) | 体积(MB) | 实际像素 | 状态 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (x["model"], QUALITIES.index(x["quality"]), x["size_label"])):
        mb = (r.get("image_bytes") or 0) / 1024 / 1024
        wh = f"{r.get('image_width')}×{r.get('image_height')}" if r.get("image_width") else "—"
        state = "✅ 成功" if r.get("status") == "passed" else f"❌ {r.get('error')}"
        lines.append(
            f"| {r['model']} | {r['quality']} | {r['size_label']} | {r.get('http_status','—')} "
            f"| {r.get('elapsed_seconds','—')} | {mb:.2f} | {wh} | {state} |"
        )
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_cells(spec: str) -> list[tuple[str, str, str, str]]:
    """解析 --cells，形如 "gpt-image-2.5-flare|low|2k|2560x1440,..."，支持只补跑个别单元格。"""
    out: list[tuple[str, str, str, str]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split("|")]
        if len(parts) != 4:
            raise ValueError(f"--cells 单元格必须为 model|quality|size_label|size，收到：{chunk!r}")
        out.append((parts[0], parts[1], parts[2], parts[3]))
    return out


async def main_async(argv: list[str] | None = None) -> int:
    global DEFAULT_API_KEY
    parser = argparse.ArgumentParser(description="sapizzz gpt-image-2.5 一次性出图")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--prompt-file", default=PROMPT_FILE)
    parser.add_argument("--cells", default="", help="只跑指定单元格，格式 model|quality|size_label|size，逗号分隔")
    parser.add_argument("--merge", action="store_true", help="把本次结果合并进 out-dir 已有的 requests.jsonl 后重建产物")
    args = parser.parse_args(argv)

    if not args.api_key:
        print("缺少 API Key：请设置 SAPIZZZ_API_KEY，或用 --api-key 传入", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parent.parent
    prompt_path = Path(args.prompt_file)
    if not prompt_path.is_absolute():
        prompt_path = root / prompt_path
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    DEFAULT_API_KEY = args.api_key

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = root / "results" / f"SAPIZZZ_IMAGE25-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    tasks = (
        parse_cells(args.cells)
        if args.cells
        else [(m, q, lbl, size) for m in MODELS for q in QUALITIES for lbl, size in SIZES]
    )
    print(f"[i] 输出目录 {out_dir}")
    print(f"[i] {len(tasks)} 个请求，并发 {args.concurrency}，提示词 {len(prompt)} 字")
    sem = asyncio.Semaphore(args.concurrency)
    wall_start = time.perf_counter()
    rows: list[dict] = []
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        timeout=httpx.Timeout(args.timeout, connect=30.0, write=60.0, pool=30.0),
        verify=True,
        trust_env=True,
    ) as client:
        results = await asyncio.gather(
            *(
                _one(
                    client,
                    sem=sem,
                    images_dir=images_dir,
                    model=m,
                    quality=q,
                    size_label=lbl,
                    size=size,
                    prompt=prompt,
                    timeout=args.timeout,
                )
                for m, q, lbl, size in tasks
            )
        )
    wall = time.perf_counter() - wall_start
    rows = list(results)

    jsonl = out_dir / "requests.jsonl"
    if args.merge and jsonl.exists():
        merged: dict[str, dict] = {}
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            old = json.loads(line)
            merged[old["task"]] = old
        for r in rows:
            merged[r["task"]] = r
        rows = list(merged.values())
    rows.sort(key=lambda r: (r["model"], QUALITIES.index(r["quality"]), r["size_label"]))

    with jsonl.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in rows if r.get("status") == "passed"]
    summary = {
        "endpoint": f"{args.base_url}/v1/images/generations",
        "models": list(MODELS),
        "qualities": list(QUALITIES),
        "sizes": {lbl: size for lbl, size in SIZES},
        "requested": len(rows),
        "passed": len(ok),
        "failed": len(rows) - len(ok),
        "wall_seconds": round(wall, 1),
        "total_image_bytes": sum(r.get("image_bytes") or 0 for r in rows),
        "elapsed_by_task": {r["task"]: r.get("elapsed_seconds") for r in rows},
        "failures": [
            {"task": r["task"], "http_status": r.get("http_status"), "error": r.get("error"),
             "error_body": (r.get("error_body") or r.get("error_message") or "")[:400]}
            for r in rows if r.get("status") != "passed"
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = build_report(rows, out_dir, wall)
    gallery = build_gallery(rows, out_dir, prompt)
    print(f"[✓] 成功 {len(ok)}/{len(tasks)}  墙钟 {wall:.0f}s")
    for r in rows:
        if r.get("status") != "passed":
            print(f"    ✗ {r['task']} HTTP {r.get('http_status')} {r.get('error')} "
                  f"{(r.get('error_body') or r.get('error_message') or '')[:160]}")
    print(f"[✓] {report}")
    print(f"[✓] {gallery}")
    print(f"[✓] {jsonl}")
    return 0 if ok else 1


def cli() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    cli()
