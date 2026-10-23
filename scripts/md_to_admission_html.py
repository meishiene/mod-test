#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 RELAY-ADMISSION-REPORT.md 渲染为与项目既有报告同款的单文件 HTML。

只支持本报告用到的 Markdown 子集：h1-h4 / 表格 / 有序无序列表 / 引用块(note) /
分隔线 / 围栏代码块 / 行内 code 与 **粗体** / 状态 emoji 着色。
"""
import html
import re
import sys
from pathlib import Path

CSS = """
  :root{
    --bg:#f7f8fa; --card:#ffffff; --ink:#1f2328; --muted:#5b6472;
    --line:#e3e6ea; --line2:#eef1f4;
    --pass:#1a7f37; --passbg:#e8f5ec;
    --warn:#9a6700; --warnbg:#fff6e0;
    --fail:#b42318; --failbg:#fdecea;
    --block:#5b6472; --blockbg:#eef1f4;
    --accent:#0b5cad;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
    line-height:1.7;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1140px;margin:0 auto;padding:40px 28px 80px}
  h1{font-size:27px;line-height:1.35;margin:0 0 8px;letter-spacing:.2px}
  h2{font-size:21px;margin:46px 0 14px;padding-bottom:8px;border-bottom:2px solid var(--line)}
  h3{font-size:16.5px;margin:30px 0 10px;color:#2b3038}
  h4{font-size:15px;margin:22px 0 8px;color:#2b3038}
  p{margin:10px 0}
  .meta{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:18px 0 8px}
  .meta div{font-size:13.5px;color:var(--muted);margin:3px 0}
  .meta b{color:var(--ink);font-weight:600}
  table{width:100%;border-collapse:separate;border-spacing:0;background:var(--card);
    border:1px solid var(--line);border-radius:12px;overflow:hidden;margin:14px 0;font-size:13.5px;
    table-layout:auto}
  th,td{padding:8px 11px;text-align:left;border-bottom:1px solid var(--line2);vertical-align:top}
  th{background:#f2f4f7;font-weight:600;color:#3a4250;font-size:13px}
  tr:last-child td{border-bottom:none}
  td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
  .st{display:inline-block;padding:0 7px;border-radius:6px;font-size:12.5px;font-weight:600;white-space:nowrap}
  .s-pass{color:var(--pass);background:var(--passbg)}
  .s-warn{color:var(--warn);background:var(--warnbg)}
  .s-fail{color:var(--fail);background:var(--failbg)}
  .s-block{color:var(--block);background:var(--blockbg)}
  .note{background:#f2f7fd;border-left:3px solid var(--accent);border-radius:0 8px 8px 0;
    padding:11px 14px;margin:14px 0;font-size:13.5px;color:#334155}
  .note p{margin:5px 0}
  code{background:#f0f2f5;padding:1px 6px;border-radius:5px;font-size:12.5px;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#b42318}
  pre{background:#0f172a;color:#e2e8f0;border-radius:10px;padding:14px 16px;overflow-x:auto;
    font-size:12.5px;line-height:1.6;margin:14px 0}
  pre code{background:none;color:inherit;padding:0;font-size:12.5px}
  ul{margin:10px 0;padding-left:22px}
  ol{margin:10px 0;padding-left:22px}
  li{margin:5px 0;font-size:13.5px}
  hr{border:none;border-top:1px solid var(--line);margin:30px 0}
  .wrap>p{font-size:13.5px;color:#334155}
  .foot{margin-top:36px;font-size:12.5px;color:#8a93a0;text-align:center}
  td b,th b{white-space:nowrap}
"""

EMOJI = {
    "✅": '<span class="st s-pass">✅</span>',
    "⚠️": '<span class="st s-warn">⚠️</span>',
    "❌": '<span class="st s-fail">❌</span>',
    "⏸": '<span class="st s-block">⏸</span>',
}

PLACEHOLDER = "\x00%d\x00"


def inline(text: str) -> str:
    """行内元素：先抽出 code，再处理粗体，最后放回 code 并着色 emoji。"""
    stash = []

    def stash_code(m):
        stash.append(f"<code>{html.escape(m.group(1))}</code>")
        return PLACEHOLDER % (len(stash) - 1)

    text = re.sub(r"`([^`]+)`", stash_code, text)
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # 只在正文/表格里着色 emoji（避免把 <b> 内容外的都替换掉，这里全量替换是安全的）
    for k, v in EMOJI.items():
        text = text.replace(k, v)
    for i, frag in enumerate(stash):
        text = text.replace(PLACEHOLDER % i, frag)
    return text


def convert(md: str) -> str:
    lines = md.split("\n")
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        # 围栏代码块
        if line.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            continue

        # 分隔线
        if line.strip() == "---":
            out.append("<hr>")
            i += 1
            continue

        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2).strip())}</h{lvl}>")
            i += 1
            continue

        # 表格
        if line.startswith("|"):
            block = []
            while i < n and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            head = [c.strip() for c in block[0].strip().strip("|").split("|")]
            body = block[2:] if len(block) > 1 else []
            t = ["<table><thead><tr>"]
            for c in head:
                t.append(f"<th>{inline(c)}</th>")
            t.append("</tr></thead><tbody>")
            for row in body:
                cells = [c.strip() for c in row.strip().strip("|").split("|")]
                t.append("<tr>")
                for c in cells:
                    t.append(f"<td>{inline(c)}</td>")
                t.append("</tr>")
            t.append("</tbody></table>")
            out.append("".join(t))
            continue

        # 引用块 -> note
        if line.startswith(">"):
            buf = []
            while i < n and lines[i].startswith(">"):
                buf.append(re.sub(r"^>\s?", "", lines[i]))
                i += 1
            parts = [f"<p>{inline(x)}</p>" if x.strip() else "" for x in buf]
            out.append('<div class="note">' + "".join(parts) + "</div>")
            continue

        # 列表
        if re.match(r"^\s*[-*]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
            ordered = bool(re.match(r"^\s*\d+\.\s+", line))
            tag = "ol" if ordered else "ul"
            items = []
            while i < n and (re.match(r"^\s*[-*]\s+", lines[i]) or re.match(r"^\s*\d+\.\s+", lines[i])):
                txt = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", lines[i])
                items.append(f"<li>{inline(txt)}</li>")
                i += 1
            out.append(f"<{tag}>" + "".join(items) + f"</{tag}>")
            continue

        # 空行
        if not line.strip():
            i += 1
            continue

        # 普通段落
        out.append(f"<p>{inline(line)}</p>")
        i += 1
    return "\n".join(out)


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    title = sys.argv[3] if len(sys.argv) > 3 else "中转通道准入测试报告"
    md = src.read_text(encoding="utf-8")

    # 头部元信息块：h1 之后、第一个 --- 之前的 "**X：** Y" 行 -> .meta
    lines = md.split("\n")
    body_start = 0
    meta_lines = []
    for idx, ln in enumerate(lines):
        if ln.startswith("**") and "：" in ln and ln.rstrip().endswith("`") is False:
            pass
    # 更稳的做法：显式扫描 h1 之后连续的 **...**：... 行
    out_lines = []
    idx = 0
    while idx < len(lines) and not lines[idx].startswith("# "):
        idx += 1
    h1 = lines[idx]
    idx += 1
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    while idx < len(lines) and lines[idx].startswith("**"):
        meta_lines.append(lines[idx])
        idx += 1

    meta_html = "".join(f"<div>{inline(x)}</div>" for x in meta_lines)
    rest = "\n".join(lines[idx:])
    body = convert(rest)

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body><div class="wrap">
<h1>{inline(h1[2:].strip())}</h1>
<div class="meta">{meta_html}</div>
{body}
<div class="foot">ml-test-platform · {html.escape(title)} · 2026-10-09</div>
</div></body>
</html>
"""
    dst.write_text(doc, encoding="utf-8")
    print(f"wrote {dst} ({len(doc)} bytes)")


if __name__ == "__main__":
    main()
