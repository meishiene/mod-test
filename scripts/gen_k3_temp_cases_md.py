#!/usr/bin/env python3
"""按清单 JSON 生成 K3 临时案例集的 Markdown（唯一真源 = `kvv/kimi_k3_temp_cases.json`）。

以前 md 是手抄的，已经出现过漏字段（版本号写成模板占位 `{d['version']}`）和期望值与
清单不一致的问题。测试 `tests/test_kvv_temp_validators.py::test_md_matches_manifest`
会比对「重新渲染的结果」与仓库里的 md，两者不一致即失败——改清单后必须跑：

    .venv/bin/python scripts/gen_k3_temp_cases_md.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "kvv" / "kimi_k3_temp_cases.json"
MD_PATH = ROOT / "kvv" / "kimi_k3_temp_cases.md"

CATEGORY_NOTE = {
    "基础连通": "连通、错误、Thinking、参数、流式、JSON、工具、追踪",
    "TTFT": "16K/64K/100K 输入各 10 次，并发 10",
    "TPOT": "16K 输入、并发 10，独立记录后汇总判定",
    "缓存": "90% 前缀命中 7 次 + 30 秒 TTL 3 次",
}
FUNCTIONAL_CATEGORIES = ("基础连通", "认证与错误", "Thinking 模式", "采样参数",
                         "流式与计量", "结构化输出", "工具调用", "可观测性")

CONTRACT_TABLE = (
    ("`temperature`", "固定 `1.0`，其他取值报错；官方建议不显式传入"),
    ("`top_p`", "固定 `0.95`"),
    ("`n` / `presence_penalty` / `frequency_penalty`", "固定 `1` / `0` / `0`"),
    ("`thinking`", "K2.x 专属参数；K3 可省略，显式传只接受 `{\"type\":\"enabled\",\"keep\":\"all\"}`"),
    ("思考开关", "关不掉（K3 始终思考）"),
    ("输出上限", "官方口径为 `max_completion_tokens`（默认 131,072 / 上限 1,048,576），本清单沿用 `max_tokens` 并记录实际用量"),
    ("结构化输出", "`json_schema` 模式必须提供 `json_schema.name` 与 `json_schema.schema`"),
    ("`tool_choice`", "支持 `auto` / `none` / `required`；不支持「指定函数名」（静默忽略，与原厂一致）"),
)


def render(manifest: dict[str, Any]) -> str:
    cases = [item for item in (manifest.get("cases") or []) if item.get("id")]
    counts = Counter(item["category"] for item in cases)
    functional = sum(counts[category] for category in FUNCTIONAL_CATEGORIES)

    lines = [
        f"# {manifest['title']}", "",
        f"- 套件：`{manifest['suite_id']}`",
        f"- 版本：`{manifest['version']}`",
        f"- 契约基准：{manifest.get('contract_baseline', '—')}",
        f"- 范围：{manifest['scope']}",
        "- 说明：文档字段是测试要求，不是发送给模型的提示词。", "",
        "## 执行约定", "",
        f"- {manifest['common']['protocol']}。",
        f"- 默认 `max_tokens={manifest['common']['default_max_tokens']}`、并发 `{manifest['common']['default_concurrency']}`。",
        "- 采样参数按 K3 官方契约固定：`temperature=1`、`top_p=0.95`、`n=1`、`presence_penalty=0`、`frequency_penalty=0`。"
        "正向用例只能用固定值，非固定值属负向用例（期望被拒）。",
        f"- {manifest['official_contract']['thinking_token_budget_note']}",
        "- 性能和缓存按原子请求计数：TTFT 30 条（16K/64K/100K 各 10 条）、TPOT 10 条、缓存 10 条。",
        "- 每条请求保留请求体、原始响应/SSE、状态码、耗时、usage、request_id 和断言结果。", "",
        "## 契约要点（2026-10-14 复核）", "", "| 项 | 官方口径 |", "|---|---|",
    ]
    lines += [f"| {name} | {note} |" for name, note in CONTRACT_TABLE]
    lines += ["", "## 案例数量", "", "| 类别 | 原子案例数 | 说明 |", "|---|---:|---|",
              f"| 接口兼容性 | {functional} | {CATEGORY_NOTE['基础连通']} |"]
    for category in ("TTFT", "TPOT", "缓存"):
        lines.append(f"| {category} | {counts[category]} | {CATEGORY_NOTE[category]} |")
    lines += [f"| **合计** | **{len(cases)}** | **临时测试集原子案例总数** |", "",
              "## 临时案例明细", "",
              "| ID | 类别 | 优先级 | 案例 | 关键参数 | 通过标准摘要 |", "|---|---|---:|---|---|---|"]
    for item in cases:
        request = json.dumps(item["request"], ensure_ascii=False, separators=(",", ":"))
        lines.append(f"| `{item['id']}` | {item['category']} | {item['priority']} | {item['name']} "
                     f"| `{request}` | {'；'.join(item['assertions'])} |")
    lines += ["", "## 不纳入范围", "",
              "- " + "、".join(f"`{value}`" for value in manifest.get("excluded_open_source_evaluations", []))
              + " 等开源评测集。",
              "- 额度和技术支持为人工确认项。",
              "- 本套件为临时自测基线，不替代正式评测。", ""]
    return "\n".join(lines)


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    MD_PATH.write_text(render(manifest), encoding="utf-8")
    print(f"已生成 {MD_PATH.relative_to(ROOT)}（{len(manifest['cases'])} 条原子案例）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
