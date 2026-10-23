"""清单驱动案例集的加载与同类项合并（Kimi 测试模块的 K3 临时案例集）。

清单里的性能与缓存案例是按「实际请求」拆分的原子条目（例如 TTFT 16K 档
10 次 → 10 条 `K3-TTFT-01..10`）。逐条请求不能少，因为每条都是独立证据，
但**案例集本身**不应呈现为 68 条散装条目——同一 `category` 且同一
`request.batch`（batch 名已包含档位与次数，如 `16K-10`）的条目属于同类项，
合并为一个逻辑案例，执行时再展开成该案例下的全部原子请求。

本模块是这份合并规则的唯一实现，`kvv/temp_runner.py`（执行与报告）
和 `web/server.py`（案例集展示与选择校验）都从这里取，避免两处各写一套。
「Claude 测试」模块下的中转计费探针清单与 K3 清单同构，也复用这里的解析
（见 `web/server.py:billing_probe_cases`），因此本模块是通用的清单解析器，
`MANIFEST_PATH` 只是 K3 案例集的默认清单。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = Path(__file__).resolve().parent / "kimi_k3_temp_cases.json"

# 合并后的案例名去掉「第 N 次」后缀，次数改由 repetitions 字段表达。
_REPETITION_SUFFIX = re.compile(r"\s*第\s*\d+\s*次\s*$")


@dataclass(frozen=True)
class TempCase:
    """合并后的逻辑案例：一次执行 = 该案例下的全部原子请求。"""

    id: str
    category: str
    name: str
    priority: str
    batch: str | None
    case_ids: tuple[str, ...]

    @property
    def repetitions(self) -> int:
        """该逻辑案例包含的原子请求数（即要发起的请求次数）。"""
        return len(self.case_ids)

    @property
    def merged(self) -> bool:
        return len(self.case_ids) > 1

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "priority": self.priority,
            "batch": self.batch,
            "repetitions": self.repetitions,
            "request_count": self.repetitions,
            "merged": self.merged,
            "case_ids": list(self.case_ids),
        }


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else MANIFEST_PATH
    return json.loads(source.read_text(encoding="utf-8"))


def manifest_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """清单里带 id 的原始原子案例，保持清单顺序。"""
    return [item for item in (manifest.get("cases") or []) if isinstance(item, dict) and item.get("id")]


def merge_key(raw: dict[str, Any]) -> tuple[str, str]:
    """同类项判定：`category` 相同且 `request.batch` 相同。

    没有 batch 的案例（接口/错误/工具类单条场景）不参与合并，
    用原子 id 兜底保证仍然各自独立。
    """
    category = str(raw.get("category") or "")
    batch = (raw.get("request") or {}).get("batch")
    if batch:
        return category, str(batch)
    return category, f"__atomic__:{raw['id']}"


def _merged_name(group: Sequence[dict[str, Any]]) -> str:
    if len(group) == 1:
        return str(group[0].get("name") or group[0]["id"])
    base = _REPETITION_SUFFIX.sub("", str(group[0].get("name") or "")).strip()
    return base or str(group[0].get("category") or group[0]["id"])


def merge_cases(raw_cases: Iterable[dict[str, Any]]) -> list[TempCase]:
    """把原子案例按同类项合并，返回合并后的逻辑案例（保持首次出现顺序）。"""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    order: list[tuple[str, str]] = []
    for raw in raw_cases:
        key = merge_key(raw)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(raw)

    merged: list[TempCase] = []
    for key in order:
        group = groups[key]
        first = group[0]
        batch = (first.get("request") or {}).get("batch")
        merged.append(
            TempCase(
                id=str(batch) if batch else str(first["id"]),
                category=str(first.get("category") or ""),
                name=_merged_name(group),
                priority=str(first.get("priority") or ""),
                batch=str(batch) if batch else None,
                case_ids=tuple(str(item["id"]) for item in group),
            )
        )
    return merged


def find_case(cases: Sequence[TempCase], selected: str | None) -> TempCase | None:
    """按合并 id 查找；同时兼容直接传原子案例 id。"""
    if not selected:
        return None
    wanted = str(selected).strip()
    for case in cases:
        if case.id == wanted or wanted in case.case_ids:
            return case
    return None
