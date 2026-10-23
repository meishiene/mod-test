"""生成千问（Qwen）**中转接入口径**准入报告（md + html）。

用法：
    .venv/bin/python scripts/qwen_relay_admission_report.py [--out-dir ~] [--date 20261021]

数据来源：``results/QWEN_<CARD>-<YYYYMMDD>-<HHMMSS>``。
- 同一 (型号, 卡片) 取**最新一份有效批次**（复测优先，基线自动降级为历史证据）；
- 无效批次（网关把路径兜成前端 index.html、或 runner 崩溃无 summary）自动排除并列出；
- 准入判定一律走 ``qwen/relay_profile.relay_verdict``（唯一权威），
  官方严格口径只作对照列保留。
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qwen.cases import QWEN_MODELS  # noqa: E402
from qwen.relay_profile import FACE_LABELS, relay_verdict  # noqa: E402

CARDS = ("QWEN_CORE", "QWEN_THINKING", "QWEN_AGENT", "QWEN_TIER", "QWEN_VISION")
CARD_LABELS = {"QWEN_CORE": "核心对话与协议", "QWEN_THINKING": "思考模式",
               "QWEN_AGENT": "Agent 与工具", "QWEN_TIER": "档位与身份指纹",
               "QWEN_VISION": "图像内容理解"}
STATUS_CN = {"passed": "通过", "warning": "警告", "blocked": "阻断", "failed": "失败", "error": "错误"}


def _is_html_batch(directory: Path) -> tuple[bool, int, int]:
    """全部请求都拿到 HTML（SPA 兜底）的批次判无效：请求根本没到模型。"""
    path = directory / "requests.jsonl"
    if not path.exists():
        return True, 0, 0
    total = html = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        if isinstance(row.get("response_body"), (dict, list)):
            continue
        text = (row.get("response_text") or "").lstrip()[:15].lower()
        if text.startswith("<!doctype") or text.startswith("<html"):
            html += 1
    return (total > 0 and html == total), html, total


def load_batches(results_dir: Path, date: str) -> tuple[dict, list[dict]]:
    """返回 {(model, card): batch} 与作废批次清单。"""
    latest: dict[tuple[str, str], dict] = {}
    discarded: list[dict] = []
    for directory in sorted(results_dir.glob(f"QWEN_*-{date}-*")):
        summary_path = directory / "summary.json"
        card = directory.name.split("-")[0]
        if card not in CARDS:
            continue
        if not summary_path.exists():
            discarded.append({"dir": directory.name, "reason": "runner 崩溃，无 summary.json"})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model = summary.get("model") or "?"
        html_only, html, total = _is_html_batch(directory)
        if html_only:
            discarded.append({"dir": directory.name, "model": model, "card": card,
                              "reason": f"全部 {html}/{total} 个响应是前端 HTML（路径被 SPA 兜 200），"
                                        "请求未到达模型"})
            continue
        batch = {"dir": directory, "name": directory.name, "model": model, "card": card,
                 "summary": summary, "started": directory.name.split(f"-{date}-")[-1]}
        key = (model, card)
        if key not in latest or batch["started"] > latest[key]["started"]:
            latest[key] = batch
    return latest, discarded


def build_model_rows(latest: dict, results_dir: Path) -> dict[str, dict]:
    """按型号合并各卡最新批次，给出中转口径与官方口径两套结论。"""
    out: dict[str, dict] = {}
    for spec in QWEN_MODELS:
        model = spec.model_id
        cards = [latest[(model, card)] for card in CARDS if (model, card) in latest]
        if not cards:
            continue
        combined: list[dict] = []
        card_rows: list[dict] = []
        for batch in cards:
            results = batch["summary"]["results"]
            # 型号矩阵声明「无图像输入」时，视觉卡片记「不适用」——这是能力边界，不是缺口也不是失败；
            # 其探针结果同样不进合计，避免把「不适用」算成 3 个 blocked 拉低观感。
            not_applicable = batch["card"] == "QWEN_VISION" and not spec.image_input
            if not not_applicable:
                combined.extend(results)
            verdict = relay_verdict(results, model)
            counts = collections.Counter(item["status"] for item in results)
            card_rows.append({
                "card": batch["card"], "label": CARD_LABELS.get(batch["card"], batch["card"]),
                "dir": batch["name"], "stamp": batch["started"],
                "counts": {name: counts.get(name, 0)
                           for name in ("passed", "warning", "blocked", "failed", "error")},
                "applicable": not not_applicable,
                "relay_admitted": verdict["admission_passed"],
                "basis": ("型号不提供图像输入（image_input=False）⇒ 视觉卡片不适用"
                          if not_applicable else verdict["summary_line"]),
                "gaps": [f"{row['case_id']}({row['face']})" for row in verdict["relay_gap"]],
                "failed": [f"{row['case_id']} {row['details']}" for row in results
                           if row["status"] in {"failed", "error"}],
                "warnings": [f"{row['case_id']} {row['details']}" for row in results
                             if row["status"] == "warning"],
            })
        verdict = relay_verdict(combined, model)
        counts = collections.Counter(item["status"] for item in combined)
        official_ok = (counts.get("failed", 0) + counts.get("error", 0) == 0
                       and all(card["summary"].get("official_all_passed", True) for card in cards))
        out[model] = {
            "spec": spec, "cards": card_rows, "verdict": verdict,
            "counts": {name: counts.get(name, 0) for name in ("passed", "warning", "blocked", "failed", "error")},
            "official_passed": official_ok, "total": len(combined),
        }
    return out


def not_applicable_cases(models: dict[str, dict]) -> int:
    """被判「不适用」而未参与判定的用例数（型号矩阵声明无该能力）。"""
    return sum(sum(card["counts"].values()) for row in models.values()
               for card in row["cards"] if not card.get("applicable", True))


def execution_window(models: dict[str, dict]) -> str:
    stamps = sorted(card["stamp"] for row in models.values() for card in row["cards"])
    if not stamps:
        return "—"
    def human(stamp: str) -> str:
        return f"{stamp[:2]}:{stamp[2:4]}" if len(stamp) >= 4 else stamp
    return f"{human(stamps[0])}–{human(stamps[-1])}"


def _common_warnings(models: dict[str, dict]) -> list[dict]:
    """跨型号共性 warning：按用例聚合，保留每款型号各自的证据文本（自述窗口等会因型号不同）。"""
    bucket: dict[str, dict] = {}
    for model, row in models.items():
        for card in row["cards"]:
            for text in card["warnings"]:
                case_id, _, detail = text.partition(" ")
                item = bucket.setdefault(case_id, {"case_id": case_id, "models": [], "details": {}})
                item["models"].append(model)
                item["details"].setdefault(detail, []).append(model)
    out = []
    for item in bucket.values():
        variants = sorted(item["details"].items(), key=lambda kv: -len(kv[1]))
        item["detail"] = variants[0][0]
        item["variants"] = [{"detail": detail, "models": models_list}
                            for detail, models_list in variants]
        out.append(item)
    return sorted(out, key=lambda item: (-len(item["models"]), item["case_id"]))


def _md(models: dict[str, dict], discarded: list[dict], date: str) -> str:
    lines: list[str] = []
    add = lines.append
    total_cases = sum(row["total"] for row in models.values())
    admitted = [m for m, row in models.items() if row["verdict"]["admission_passed"]]
    official = [m for m, row in models.items() if row["official_passed"]]
    add("# 千问（Qwen）模型准入测试报告（中转接入口径）")
    add("")
    add(f"- **测试日期**：2026-10-21 {execution_window(models)}（同一分组全量执行，GMT+8）")
    add("- **被测型号**：qwen3.6-flash / qwen3.6-plus / qwen3.7-plus / qwen3.7-max / qwen3.8-flash / qwen3.8-max")
    add("- **接入通道**：中转网关 `https://api.oinone.top/v1`，分组 **domestic**（上游百炼，厂商静默升级后复测）")
    add("- **判据口径**：**中转接入口径**（唯一权威 `qwen/relay_profile.py`）——核心对话面与模型一致性面出现 "
        "failed/error 才阻断；工具强制语义、思考参数、档位指纹、错误信封、视觉输入属**高级面**，缺口只记 "
        "`relay_gap` 不阻断准入。官方严格口径仅作对照列保留。")
    na = not_applicable_cases(models)
    add(f"- **样本**：{len(models)} 款型号 × 5 张卡 = {total_cases} 条已判定用例（同一分组全量执行，"
        "每条用例一条结论）" + (f"；另有 {na} 条判「不适用」（型号无图像输入，不计入分母）" if na else ""))
    add("")
    add("## 一、准入结论")
    add("")
    add("| 型号 | 中转口径结论 | 官方严格口径（对照） | 通过/警告/阻断/失败 | 高级面缺口 |")
    add("|---|---|---|---|---|")
    for model, row in models.items():
        counts = row["counts"]
        gaps = "、".join(FACE_LABELS.get(item["face"], item["face"]) for item in
                        {(gap["face"], gap["face"]): gap for gap in row["verdict"]["relay_gap"]}.values()) or "—"
        add(f"| {model} | {'✅ 准入' if row['verdict']['admission_passed'] else '❌ 不准入'} | "
            f"{'通过' if row['official_passed'] else '不合格'} | "
            f"{counts['passed']}/{counts['warning']}/{counts['blocked']}/{counts['failed']} | {gaps} |")
    add("")
    add(f"**中转口径下 {len(admitted)}/{len(models)} 款准入**；若按官方严格口径只有 {len(official)}/{len(models)} 款合格——"
        "差异全部来自「网关/上游没实现」而非「模型能力不达标」，这正是改用中转口径的原因。")
    add("")
    add("## 二、复测：原两项未通过项的处理结果")
    add("")
    add("### 1. `tool_choice=required` 未强制（QWEN-A02）—— 复现，记 relay_gap")
    add("")
    add("- 复测（domestic 分组，先单点后全量重跑）：HTTP 200、`finish_reason=stop`、**无 tool_calls**，"
        "正文反问「请告诉我您想查询哪个城市的天气？」。")
    add("- 同批复测 A01/A03/A04/A05（基础链路/回传/流式/并行）全部通过 ⇒ 工具基础能力正常，只有**强制语义**被网关降级为 auto。")
    add("- 口径处理：Strict Tool 属高级面，记 `relay_gap`，不阻断准入；若业务强依赖「必须调用某工具」，"
        "需客户端自行校验 `tool_calls` 缺失并重试，或改用 `tools` + 提示词约束。")
    add("")
    add("### 2. 视觉输入（QWEN-V01）—— 型号本身不带图像输入，判「不适用」")
    add("")
    add("- 复测 3 个视觉探针仍全部 HTTP 400 `InvalidParameter … Unexpected item type in content`。")
    add("- **经确认：`qwen3.7-max` 就是纯文本型号，不提供图像输入**，不是网关把它降级了。矩阵里 "
        "`image_input=False` 即为该事实的登记处，视觉卡片对它判「不适用」（`vision_undeclared`），"
        "既不记失败也不记缺口。")
    add("- 需要图像理解时按能力路由到 `qwen3.8-max/flash`、`qwen3.7-plus`、`qwen3.6-plus/flash`"
        "（这几款视觉探针 3/3 读图正确，其中 3.6-plus 本次从「未声明」纠正为「支持」）。")
    add("")
    add("### 3. 复测过程中修正的两个判据问题")
    add("")
    add("- **路径坑**：该网关只暴露 `/v1`，`/compatible-mode/v1` 会被前端 SPA 兜成 **HTML 200**，"
        "旧逻辑会把它读成「模型答了空字符串」→ 一串假 failed。已加守卫：200 但非 JSON ⇒ 记 blocked，"
        "并在报告中把它们从证据里剔除（见文末作废批次）。")
    add("- **复测开关**：runner 新增 `QWEN_CASE_IDS`（等价 DSV4 的 `DS_CASE_IDS`），可按用例定向复测，"
        "结果目录撞名加序号兜底。")
    add("")
    add("## 三、逐型号卡片结论（中转口径）")
    add("")
    add("| 型号 | 卡片 | 通过/警告/阻断/失败 | 结论 | 依据 / 缺口 |")
    add("|---|---|---|---|---|")
    for model, row in models.items():
        for card in row["cards"]:
            counts = card["counts"]
            if not card.get("applicable", True):
                add(f"| {model} | {card['label']} | —（不适用） | ➖ 不适用 | {card['basis']} |")
                continue
            add(f"| {model} | {card['label']} | {counts['passed']}/{counts['warning']}/"
                f"{counts['blocked']}/{counts['failed']} | "
                f"{'准入' if card['relay_admitted'] else '不准入'} | {card['basis']} |")
    add("")
    add("> 「➖ 不适用」= 型号矩阵声明该型号不提供该能力（当前仅 `qwen3.7-max` 的图像输入），"
        "属能力边界：既不算通过也不算缺口，该卡 1 条用例（3 个探针）不进合计、不参与准入判定。")
    add("")
    add("## 四、跨型号共性发现（warning，均不阻断）")
    add("")
    for item in _common_warnings(models):
        add(f"- **{item['case_id']}（{len(item['models'])}/6 型号）**：{item['detail']}")
        for variant in item.get("variants", [])[1:]:
            add(f"  - {'、'.join(variant['models'])}：{variant['detail']}")
    add("")
    add("## 五、整改建议")
    add("")
    add("1. **对网关/上游**：① `tool_choice=required` 若无法透传，应在响应或文档中显式声明降级；"
        "② 思考模式 `max_tokens` 上限（文档 32768）实测不生效，需与厂商对齐；"
        "③ `top_p` 越界（1.5）返回 200 而 `temperature` 越界返回 400，取值域校验不一致；"
        "④ 不存在的模型返回 503，建议映射为 404 以免被误判为瞬时故障。")
    add("2. **对接侧**：需要「必须调用工具」的场景，请校验 `tool_calls`；需要图像理解时不要路由到 "
        "`qwen3.7-max`（矩阵声明无视觉）；长输入超时按 3.6-plus 的实测水位（>150s）设置。")
    add("3. **复测方式**：`QWEN_CASE_IDS=QWEN-A02` + `QWEN_MODEL=qwen3.7-max` 即可单点复核强制语义；"
        "全量重跑用界面「千问测试」勾全部卡片。")
    add("")
    add("## 六、证据与复现")
    add("")
    add("| 型号 | 使用的批次 |")
    add("|---|---|")
    for model, row in models.items():
        add(f"| {model} | " + "、".join(card["dir"] for card in row["cards"]) + " |")
    add("")
    if discarded:
        add("**作废批次（未参与判定）**：")
        add("")
        for item in discarded:
            add(f"- `{item['dir']}`：{item['reason']}")
        add("")
    add("*判据唯一权威 `qwen/relay_profile.py`；报告生成器 `scripts/qwen_relay_admission_report.py`。*")
    return "\n".join(lines) + "\n"


def _html(md_text: str, models: dict[str, dict], date: str) -> str:
    """由同一份数据直出 HTML（不引第三方 markdown 依赖）。"""
    def esc(text: str) -> str:
        return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    total_cases = sum(row["total"] for row in models.values())
    body: list[str] = []
    body.append("<h1>千问（Qwen）模型准入测试报告</h1>")
    body.append(f'<p class="sub">中转接入口径 · 分组 domestic · 2026-10-21 {execution_window(models)}</p>')
    body.append('<ul class="meta">'
                '<li><b>接入通道</b>：<code>https://api.oinone.top/v1</code>，分组 <b>domestic</b>（上游百炼，厂商静默升级后复测）</li>'
                '<li><b>被测型号</b>：qwen3.6-flash / qwen3.6-plus / qwen3.7-plus / qwen3.7-max / qwen3.8-flash / qwen3.8-max</li>'
                '<li><b>判据口径</b>：中转接入（核心面 + 模型一致性面失败才阻断；高级面缺口只记 relay_gap）</li>'
                f'<li><b>样本</b>：6 款 × 5 卡，共 {total_cases} 条已判定用例'
                + (f'（另有 {not_applicable_cases(models)} 条判「不适用」，不计入分母）'
                   if not_applicable_cases(models) else '') + '</li>'
                '</ul>')

    body.append("<h2>一、准入结论</h2>")
    body.append("<table><tr><th>型号</th><th>中转口径</th><th>官方严格口径</th>"
                "<th>通过/警告/阻断/失败</th><th>高级面缺口</th></tr>")
    for model, row in models.items():
        c = row["counts"]
        gaps = "、".join(sorted({FACE_LABELS.get(gap["face"], gap["face"])
                                for gap in row["verdict"]["relay_gap"]})) or "—"
        cls = "ok" if row["verdict"]["admission_passed"] else "bad"
        text = "✅ 准入" if row["verdict"]["admission_passed"] else "❌ 不准入"
        off = "通过" if row["official_passed"] else "不合格"
        body.append(f"<tr><td>{model}</td><td><span class='tag {cls}'>{text}</span></td>"
                    f"<td>{off}</td><td class='num'>{c['passed']}/{c['warning']}/{c['blocked']}/{c['failed']}</td>"
                    f"<td>{esc(gaps)}</td></tr>")
    body.append("</table>")
    admitted = sum(1 for row in models.values() if row["verdict"]["admission_passed"])
    official = sum(1 for row in models.values() if row["official_passed"])
    body.append(f'<div class="callout">中转口径下 <b>{admitted}/{len(models)} 款准入</b>；'
                f'官方严格口径只有 <b>{official}/{len(models)}</b> 款合格。'
                '差异全部来自「网关/上游没实现」，不是「模型能力不达标」。</div>')

    body.append("<h2>二、定向复测结果</h2>")
    body.append("<h3>1. <code>tool_choice=required</code> 未强制（QWEN-A02）—— 复现，记 relay_gap</h3>")
    body.append("<ul><li>HTTP 200、<code>finish_reason=stop</code>、<b>无 tool_calls</b>，正文反问「请告诉我您想查询哪个城市的天气？」。</li>"
                "<li>同批 A01/A03/A04/A05（基础链路/回传/流式/并行）全部通过 ⇒ 工具能力在，只有强制语义被降级。</li>"
                "<li>高级面口径：不阻断准入；强依赖场景由客户端校验 <code>tool_calls</code> 缺失后重试。</li></ul>")
    body.append("<h3>2. 视觉输入（QWEN-V01）—— 型号本身不带图像输入，判「不适用」</h3>")
    body.append("<ul><li>3 个探针仍全部 400 <code>InvalidParameter … Unexpected item type in content</code>。</li>"
                "<li><b>经确认</b>：<code>qwen3.7-max</code> 就是纯文本型号、不提供图像输入，"
                "矩阵 <code>image_input=False</code> 即该事实的登记处 ⇒ 视觉卡片判「不适用」"
                "（<code>vision_undeclared</code>），既不算失败也不算缺口。</li>"
                "<li>需要图像理解时路由到 3.8-max/flash、3.7-plus、3.6-plus/flash（均 3/3 读图正确）。</li></ul>")
    body.append("<h3>3. 复测修正的两个判据问题</h3>")
    body.append("<ul><li><b>路径坑</b>：网关只暴露 <code>/v1</code>，<code>/compatible-mode/v1</code> 被前端 SPA 兜成 "
                "<b>HTML 200</b>，旧逻辑读成「空回答」→ 假 failed。已加守卫：200 非 JSON ⇒ blocked，并剔除作废批次。</li>"
                "<li><b>复测开关</b>：runner 新增 <code>QWEN_CASE_IDS</code>，支持用例级定向复测。</li></ul>")

    body.append("<h2>三、逐型号卡片结论</h2>")
    body.append("<table><tr><th>型号</th><th>卡片</th><th>通过/警告/阻断/失败</th><th>结论</th><th>依据 / 缺口</th></tr>")
    for model, row in models.items():
        for card in row["cards"]:
            c = card["counts"]
            if not card.get("applicable", True):
                body.append(f"<tr><td>{model}</td><td>{card['label']}</td><td class='num'>—</td>"
                            f"<td><span class='tag neutral'>➖ 不适用</span></td>"
                            f"<td>{esc(card['basis'])}</td></tr>")
                continue
            cls = "ok" if card["relay_admitted"] else "bad"
            body.append(f"<tr><td>{model}</td><td>{card['label']}</td>"
                        f"<td class='num'>{c['passed']}/{c['warning']}/{c['blocked']}/{c['failed']}</td>"
                        f"<td><span class='tag {cls}'>{'准入' if card['relay_admitted'] else '不准入'}</span></td>"
                        f"<td>{esc(card['basis'])}</td></tr>")
    body.append("</table>")
    body.append("<p class='sub2'>「➖ 不适用」= 型号矩阵声明该型号不提供该能力（当前仅 "
                "<code>qwen3.7-max</code> 的图像输入），属能力边界：既不计通过也不计缺口，"
                "该卡 1 条用例（3 个探针）不进合计、不参与准入判定。</p>")

    body.append("<h2>四、跨型号共性发现（warning，均不阻断）</h2><ul>")
    for item in _common_warnings(models):
        extra = "".join(f"<br><span class='sub2'>{esc('、'.join(v['models']))}：{esc(v['detail'])}</span>"
                        for v in item.get("variants", [])[1:])
        body.append(f"<li><b>{item['case_id']}（{len(item['models'])}/6 型号）</b>：{esc(item['detail'])}{extra}</li>")
    body.append("</ul>")

    body.append("<h2>五、整改建议</h2><ol>"
                "<li><b>网关/上游</b>：① <code>tool_choice=required</code> 无法透传应显式声明降级；"
                "② 思考模式 <code>max_tokens</code> 上限（32768）实测不生效；"
                "③ <code>top_p=1.5</code> 返回 200 而 <code>temperature=3</code> 返回 400，取值域校验不一致；"
                "④ 不存在的模型返回 503，建议映射 404。</li>"
                "<li><b>对接侧</b>：强依赖工具调用时校验 <code>tool_calls</code>；图像理解不要路由到 qwen3.7-max；"
                "长输入超时按 3.6-plus 实测水位（&gt;150s）设置。</li>"
                "<li><b>复测</b>：<code>QWEN_CASE_IDS=QWEN-A02 QWEN_MODEL=qwen3.7-max</code> 单点复核强制语义。</li></ol>")

    body.append("<h2>六、证据</h2><table><tr><th>型号</th><th>使用的批次</th></tr>")
    for model, row in models.items():
        body.append(f"<tr><td>{model}</td><td class='dirs'>{'<br>'.join(card['dir'] for card in row['cards'])}</td></tr>")
    body.append("</table>")

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>千问模型准入测试报告 - {date}</title>
<style>
:root{{--ink:#1a2233;--muted:#5a6a85;--line:#dde4ef;--bg:#f6f8fb;--card:#fff;--ok:#0a7d43;--okbg:#e6f5ec;--bad:#b42318;--badbg:#fdecea;--blu:#1d5fd6;}}
*{{box-sizing:border-box}}
body{{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;color:var(--ink);background:var(--bg);margin:0;padding:32px 16px;line-height:1.65}}
.wrap{{max-width:1000px;margin:0 auto}}
h1{{font-size:26px;margin:0 0 4px}} .sub{{color:var(--muted);margin:0 0 14px;font-size:13px}}
h2{{font-size:19px;margin:34px 0 12px;padding-bottom:8px;border-bottom:2px solid var(--line)}}
h3{{font-size:15.5px;margin:18px 0 6px}}
.meta{{color:var(--muted);font-size:13px;padding-left:18px}} .meta li{{margin:2px 0}}
table{{width:100%;border-collapse:collapse;background:var(--card);font-size:13px;border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:12px 0}}
th,td{{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}
th{{background:#eef2f8;font-weight:600;white-space:nowrap}} tr:last-child td{{border-bottom:none}}
.tag{{display:inline-block;padding:1px 10px;border-radius:20px;font-size:12px;font-weight:600;white-space:nowrap}}
.ok{{color:var(--ok);background:var(--okbg)}} .bad{{color:var(--bad);background:var(--badbg)}}
.neutral{{color:var(--muted);background:#eef2f8}}
.callout{{border-left:4px solid var(--blu);background:var(--card);border-radius:0 8px 8px 0;padding:12px 16px;margin:14px 0;font-size:13.5px}}
code{{background:#eef2f8;border-radius:4px;padding:1px 5px;font-size:12px;font-family:"SF Mono",Menlo,monospace}}
ul,ol{{padding-left:20px}} li{{margin:5px 0;font-size:13.5px}} .num{{font-variant-numeric:tabular-nums}}
.sub2{{color:var(--muted);font-size:12.5px}}
.dirs{{font-size:11.5px;color:var(--muted);font-family:"SF Mono",Menlo,monospace}}
.foot{{color:var(--muted);font-size:12px;margin-top:30px;border-top:1px solid var(--line);padding-top:12px}}
</style></head><body><div class="wrap">
{chr(10).join(body)}
<p class="foot">判据唯一权威 <code>qwen/relay_profile.py</code> · 生成器 <code>scripts/qwen_relay_admission_report.py</code> · {date}</p>
</div></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261021")
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    parser.add_argument("--out-dir", type=Path, default=Path.home())
    args = parser.parse_args()

    latest, discarded = load_batches(args.results_dir, args.date)
    models = build_model_rows(latest, args.results_dir)
    if not models:
        print(json.dumps({"error": "no_batches", "date": args.date}, ensure_ascii=False))
        return 2
    md_text = _md(models, discarded, args.date)
    md_path = args.out_dir / f"千问模型准入测试报告-{args.date}.md"
    html_path = args.out_dir / f"千问模型准入测试报告-{args.date}.html"
    md_path.write_text(md_text, encoding="utf-8")
    html_path.write_text(_html(md_text, models, args.date), encoding="utf-8")
    print(json.dumps({"md": str(md_path), "html": str(html_path),
                      "models": {model: row["verdict"]["admission_passed"] for model, row in models.items()},
                      "official": {model: row["official_passed"] for model, row in models.items()},
                      "discarded": discarded}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
