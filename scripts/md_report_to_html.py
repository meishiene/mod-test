#!/usr/bin/env python3
"""把准入报告 Markdown 渲染成自包含 HTML（无外部依赖的浅色排版）。"""
from __future__ import annotations

import sys
from pathlib import Path

import markdown

CSS = """
:root{
  --bg:#ffffff; --fg:#1f2328; --muted:#59636e; --line:#d8dee4;
  --accent:#0b6bcb; --soft:#f6f8fa; --warn:#9a6700; --bad:#b42318; --ok:#1a7f37;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:16px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;}
.wrap{max-width:1080px;margin:0 auto;padding:56px 40px 96px;}
h1{font-size:32px;line-height:1.3;margin:0 0 8px;letter-spacing:-.3px}
h2{font-size:23px;margin:52px 0 16px;padding-bottom:8px;border-bottom:1px solid var(--line)}
h3{font-size:18px;margin:32px 0 12px}
p{margin:12px 0}
ul,ol{padding-left:24px;margin:12px 0}
li{margin:6px 0}
hr{border:0;border-top:1px solid var(--line);margin:40px 0}
code{background:var(--soft);padding:2px 6px;border-radius:5px;font-size:.9em;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14.5px}
.tbl{overflow-x:auto;margin:18px 0}
.tbl table{margin:0}
th,td{border:1px solid var(--line);padding:9px 12px;text-align:left;vertical-align:top}
th{background:var(--soft);font-weight:600;white-space:nowrap}
tbody tr:nth-child(even){background:#fbfcfd}
blockquote{margin:18px 0;padding:10px 18px;border-left:3px solid #d0d7de;background:var(--soft);
  color:var(--muted)}
strong{font-weight:600}
.meta{color:var(--muted);font-size:14px;margin-bottom:8px}
"""


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: md_report_to_html.py <input.md> [output.html]", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".html")
    md = markdown.Markdown(extensions=["tables", "sane_lists", "fenced_code"])
    body = md.convert(src.read_text(encoding="utf-8"))
    body = body.replace("<table>", '<div class="tbl"><table>').replace(
        "</table>", "</table></div>"
    )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{src.stem}</title>
<style>{CSS}</style>
</head>
<body><div class="wrap">
{body}
</div></body></html>
"""
    dst.write_text(html, encoding="utf-8")
    print(f"written: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
