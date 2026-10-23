"""统一的测试报告 markdown 样式（对齐 dsv4/core.py write_report 的样式基线）。

所有测试项的 report.md 都应使用这里的助手生成，保证 Web 报告弹窗样式一致：
标题 → 统计要点（总计/通过/失败/警告/阻塞/未观测/已评估用例通过率）→ 用例表格。

单项测试（图片类除外）在 finalize 阶段还会调用 `unify_report`，把报告头改写成
聚合报告同款版式：记录 ID / 协议 / 模型 / 总结论 / 通过 x/y + 统一指标表。
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Sequence

from .report_metrics import build_item_metrics

STATUS_ICONS = {
    "passed": "✅ 通过",
    "failed": "❌ 失败",
    "error": "❌ 错误",
    "warning": "⚠️ 警告",
    "blocked": "⏸ 未验证",
}

# 与 Web 聚合报告、前端的套件表保持同一套列与口径。
METRICS_COLUMNS: tuple[str, ...] = ("轮次", "状态", "结论", "请求", "成功率", "用例通过", "TTFT P90", "TTFC P90", "E2E P90", "TPOT(ms/tok)")
METRICS_HINT = "请求数与成功率来自执行证据（requests.jsonl）；TTFT / TTFC P90 仅统计流式请求样本，E2E P90 覆盖全部请求。"
RUN_STATUS_LABELS = {"completed": "已完成", "failed": "失败", "interrupted": "已中断", "running": "执行中", "queued": "排队中"}
IMAGE_TEST_KINDS = {"gpt-image", "gemini-image"}
# 自带完整版式的套件（模板里已有图片/视频清单与产物链接）：不参与报告版式统一。
# Seedance 报告的主产物是**生成视频的链接**，套用统一指标表会把链接表挤到后面，
# 因此与图像类套件同口径——保留套件自己写出的报告。
SELF_STYLED_TEST_KINDS = IMAGE_TEST_KINDS | {"seedance"}

_META_PREFIXES = ("- 记录 ID：", "- 协议：", "- 模型：", "- 总结论：", "- 通过：", "- 请求：", "- 用例数：")
_LEGACY_STATS_PREFIXES = ("- 总计：", "- 通过：", "- 失败：", "- 警告：", "- 阻塞/未观测：", "- 已评估用例通过率：")
_LEGACY_CONCLUSION_PREFIXES = ("- 本轮结论：", "- 准入结论：", "- 结论：")
_DROP_BULLET_PREFIXES = _META_PREFIXES + _LEGACY_STATS_PREFIXES + _LEGACY_CONCLUSION_PREFIXES


def status_icon(status: str) -> str:
    return STATUS_ICONS.get(status, status)


def _escape(cell: Any) -> str:
    return str(cell).replace("|", "\\|").replace("\n", " ")


def stats_lines(
    total: int,
    passed: int,
    failed: int,
    warning: int = 0,
    blocked: int = 0,
    pass_rate: float | None = None,
    extra: Iterable[str] = (),
) -> list[str]:
    """统计要点块。pass_rate 缺省按 (passed)/(passed+failed) 计算。"""
    if pass_rate is None:
        evaluated = passed + failed
        pass_rate = passed / evaluated if evaluated else 0.0
    lines = [
        f"- 总计：{total}",
        f"- 通过：{passed}",
        f"- 失败：{failed}",
        f"- 警告：{warning}",
        f"- 阻塞/未观测：{blocked}",
        f"- 已评估用例通过率：{pass_rate:.1%}",
    ]
    lines.extend(f"- {x}" for x in extra)
    return lines



def case_table_report(
    title: str,
    *,
    total: int,
    passed: int,
    failed: int,
    warning: int = 0,
    blocked: int = 0,
    pass_rate: float | None = None,
    extra_stats: Iterable[str] = (),
    columns: Sequence[str] = ("用例", "模型", "结果", "耗时(s)", "说明"),
    status_column: int | None = 2,
    rows: Sequence[Sequence[Any]],
    trailing: Iterable[str] = (),
) -> str:
    """渲染与 DeepSeek V4 专项测试报告一致的 report.md 全文。

    rows 单元格自动转义；若指定 status_column，该列的状态词（passed/failed/...）
    会自动替换为图标文案。
    """
    lines = [
        f"# {title}",
        "",
        *stats_lines(total, passed, failed, warning, blocked, pass_rate, extra_stats),
        "",
        "| " + " | ".join(columns) + " |",
    ]
    align = ["---"] * len(columns)
    if len(columns) >= 4:
        align[3] = "---:"
    lines.append("|" + "|".join(align) + "|")
    for row in rows:
        cells = [_escape(c) for c in row]
        if status_column is not None and status_column < len(cells):
            cells[status_column] = status_icon(str(cells[status_column]))
        lines.append("| " + " | ".join(cells) + " |")
    tail = list(trailing)
    if tail:
        lines.extend(["", *tail])
    return "\n".join(lines) + "\n"


def _fmt_seconds(value: Any) -> str:
    return f"{value:.3f}s" if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else "—"


def _fmt_rate(value: Any) -> str:
    return f"{value * 100:.1f}%" if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else "—"


def _fmt_number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return str(int(value)) if float(value).is_integer() else f"{value}"


def metrics_table_lines(metrics: dict[str, Any], *, run_label: str, run_status: str, concluded: bool) -> list[str]:
    """聚合报告同款的指标表（单项测试只有一行）。"""
    warnings = metrics.get("case_warning") or 0
    blocked = metrics.get("case_blocked") or 0
    marks = [f"⚠{warnings}"] if warnings else []
    if blocked:
        marks.append(f"⏸{blocked}")
    extras = f"（{' · '.join(marks)}）" if marks else ""
    total, ok = metrics.get("case_total"), metrics.get("case_passed")
    case_text = f"{ok}/{total}{extras}" if isinstance(total, int) and total else "—"
    tpot = metrics.get("tpot_p90_ms_per_token")
    row = [
        run_label,
        RUN_STATUS_LABELS.get(run_status, run_status or "—"),
        "通过" if concluded else "未通过",
        _fmt_number(metrics.get("requests")),
        _fmt_rate(metrics.get("success_rate")),
        case_text,
        _fmt_seconds(metrics.get("ttft_p90_seconds")),
        _fmt_seconds(metrics.get("ttfc_p90_seconds")),
        _fmt_seconds(metrics.get("e2e_p90_seconds")),
        f"{tpot:.1f}" if isinstance(tpot, (int, float)) and not isinstance(tpot, bool) else "—",
    ]
    return [
        "## 指标",
        "",
        "| " + " | ".join(METRICS_COLUMNS) + " |",
        "|" + "|".join(["---"] * len(METRICS_COLUMNS)) + "|",
        "| " + " | ".join(_escape(cell) for cell in row) + " |",
        "",
        f"> {METRICS_HINT}",
    ]


def _generated_block_bounds(lines: list[str]) -> tuple[int, int] | None:
    """定位上一次 unify_report 生成的 `## 指标` 段，保证重复 finalize 幂等。"""
    header = "| " + " | ".join(METRICS_COLUMNS) + " |"
    for index, line in enumerate(lines):
        if line.strip() != "## 指标":
            continue
        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor < len(lines) and lines[cursor].strip() == header:
            end = cursor
            while end < len(lines) and lines[end].startswith("|"):
                end += 1
            while end < len(lines) and not lines[end].strip():
                end += 1
            if end < len(lines) and lines[end].lstrip().startswith(f"> {METRICS_HINT}"):
                end += 1
            return index, end
    return None


def _case_table_lines(summary: dict[str, Any]) -> list[str]:
    results = [row for row in (summary.get("results") or []) if isinstance(row, dict)]
    lines = ["| 用例 | 名称 | 状态 | 耗时(s) | 结论说明 |", "|---|---|---:|---|"]
    for row in results:
        status = str(row.get("status") or ("passed" if row.get("passed") is True else "failed"))
        duration = row.get("duration_seconds")
        lines.append(
            "| " + " | ".join([
                _escape(row.get("case_id") or row.get("id") or "—"),
                _escape(row.get("name") or ""),
                _escape(status_icon(status)),
                f"{duration:.2f}" if isinstance(duration, (int, float)) and not isinstance(duration, bool) else "—",
                _escape(str(row.get("details") or row.get("error") or "—")[:240]),
            ]) + " |"
        )
    return lines


def is_image_summary(summary: dict[str, Any]) -> bool:
    """套件是否自带完整报告版式（图片清单 / 视频链接），不参与版式统一。"""
    summary = summary or {}
    if str(summary.get("test_kind") or "") in SELF_STYLED_TEST_KINDS:
        return True
    return any(key in summary for key in ("images_saved", "image_bytes_saved", "images_dir"))


def _face_head_lines(summary: dict[str, Any]) -> list[str]:
    """报告头的「执行范围 / 网关缺口」两行（没有面信息时返回空）。"""
    from .faces import FACE_CHANNEL, FACE_LABELS

    faces = summary.get("executed_faces")
    if not isinstance(faces, (list, tuple)) or not faces:
        return []
    labels = "、".join(FACE_LABELS.get(str(face), str(face)) for face in faces)
    suffix = "" if FACE_CHANNEL in faces else "（通道契约面未执行：网关端点 / 第二协议 / beta / 参数校验）"
    lines = [f"- 执行范围：{labels}{suffix}"]
    gap: Any = None
    verdict = summary.get("face_verdict")
    if isinstance(verdict, dict):
        gap = verdict.get("channel_gap")
    if not gap:
        relay = summary.get("relay_verdict")
        if isinstance(relay, dict):
            gap = relay.get("relay_gap")
    if gap:
        lines.append(f"- 网关缺口：{len(gap)} 项（只记录，不阻断准入）")
    return lines


def strip_report_scores(report_text: str) -> str:
    """移除历史报告的评分要点和完整评分章节，保留用例证据与其他指标。"""
    kept: list[str] = []
    score_level = 0
    fence = ""
    for line in report_text.splitlines(keepends=True):
        stripped = line.strip()
        marker = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence:
            if not score_level:
                kept.append(line)
            if stripped.startswith(fence) and not stripped[len(fence):].strip():
                fence = ""
            continue
        if marker:
            fence = marker.group(1)
            if not score_level:
                kept.append(line)
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", stripped)
        if heading:
            level = len(heading.group(1))
            if score_level and level <= score_level:
                score_level = 0
            if heading.group(2) == "评分":
                score_level = score_level or level
        if score_level or re.match(r"^- 评分[：:（(]", stripped):
            continue
        kept.append(line)
    return "".join(kept)


def unify_report(report_text: str, summary: dict[str, Any], result_dir: Any = None) -> str:
    """把单项测试报告改写成聚合报告同款版式，并补上统一指标表。

    - 报告头：记录 ID / 协议 / 模型 / 总结论 / 通过 x/y / 请求与成功率，
      原有领域要点（负载画像、请求上限、指纹结论等）保留在要点区内；
    - 紧随其后插入聚合报告同款的 10 列指标表（单项只有一行）；
    - 原有正文（用例表、判定、核心统计、错误等章节）原样保留，
      若正文已有同名 `## 指标` 章节则改名为 `## 指标明细`，避免与统一指标表混淆。
    """
    summary = summary or {}
    lines = strip_report_scores(report_text or "").replace("\r", "").split("\n")
    generated = _generated_block_bounds(lines)
    if generated:
        start, end = generated
        lines = lines[:start] + lines[end:]

    title: str | None = None
    cursor = 0
    for index, line in enumerate(lines):
        if line.startswith("# "):
            title = line.strip()
            cursor = index + 1
            break

    bullets: list[str] = []
    while cursor < len(lines):
        stripped = lines[cursor].strip()
        if not stripped:
            cursor += 1
            continue
        if stripped.startswith("- "):
            bullets.append(stripped)
            cursor += 1
            continue
        break
    body = lines[cursor:]
    while body and not body[0].strip():
        body = body[1:]

    case_label = str(summary.get("case") or summary.get("case_name") or (getattr(result_dir, "name", "") or "") or "测试")
    if not title:
        title = f"# {case_label} 测试报告"

    metrics = build_item_metrics(summary, result_dir)
    passed = bool(summary.get("admission_passed"))
    head: list[str] = []
    record_id = summary.get("record_id") or summary.get("recordId") or getattr(result_dir, "name", "")
    if record_id:
        head.append(f"- 记录 ID：`{record_id}`")
    if summary.get("protocol"):
        head.append(f"- 协议：`{summary['protocol']}`")
    model = summary.get("model") or summary.get("served_model")
    if model:
        head.append(f"- 模型：`{model}`")
    # 执行范围：默认只跑模型能力+验真，通道契约面停用。报告必须自证"用例为什么少"，
    # 否则会被读成"漏跑"；勾选通道面执行时也要写清缺口数不阻断准入。
    head.extend(_face_head_lines(summary))
    head.append(f"- 总结论：{'通过' if passed else '未通过'}")
    total, ok = metrics.get("case_total"), metrics.get("case_passed")
    if isinstance(total, int) and total:
        warnings, blocked = metrics.get("case_warning") or 0, metrics.get("case_blocked") or 0
        marks = [f"⚠{warnings}"] if warnings else []
        if blocked:
            marks.append(f"⏸{blocked}")
        head.append(f"- 通过：{ok}/{total}" + (f"（{' · '.join(marks)}）" if marks else ""))
    if metrics.get("requests") is not None:
        head.append(f"- 请求：{_fmt_number(metrics['requests'])}；成功率：{_fmt_rate(metrics.get('success_rate'))}")
    extras: list[str] = []
    for bullet in bullets:
        if any(bullet.startswith(prefix) for prefix in _DROP_BULLET_PREFIXES):
            continue
        if bullet not in extras:
            extras.append(bullet)

    out = [title, "", *head, *extras, "", *metrics_table_lines(
        metrics,
        run_label=case_label,
        run_status=str(summary.get("status") or ("interrupted" if summary.get("interrupted") else "completed")),
        concluded=passed,
    ), ""]
    renamed = ["## 指标明细" if line.strip() == "## 指标" else line for line in body]
    if renamed and renamed[0].strip().startswith("|"):
        out += ["## 用例结果", "", *renamed]
    elif renamed:
        out += renamed
    else:
        out += ["## 用例结果", "", *_case_table_lines(summary)]
    return "\n".join(out).rstrip() + "\n"

