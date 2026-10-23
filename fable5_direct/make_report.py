#!/usr/bin/env python3
"""把某次直连实测的结果合并「人工复核更正」后，产出最终报告（Markdown + HTML）。

用法：python make_report.py <run_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 复核更正：用例 id -> (修正后状态, 更正依据)。凡与自动判定不同者必须给出可复查的证据来源。
CORRECTIONS: dict[str, tuple[str, str]] = {
    "E17": ("passed", "判据缺陷更正：改用必然出现的分隔符后，stop_reason=stop_sequence，正文在 D4 前精确截断（手工探针复核）"),
    "E18": ("passed", "判据缺陷更正：原判据「必须只输出固定串」实为模型对齐行为（同指令放 user 消息亦被拒）；改用格式前缀后回复以指定 tag 开头（手工探针复核）"),
    "E03": ("warning", "定级更正：文档承诺 400、线上宽松 200，按项目口径属线上差异记 warning，非模型失败"),
    "E04": ("warning", "定级更正：同上，文档承诺 400、线上 200"),
    "S10": ("warning", "定级更正：未观测到 refusal 语义属未取证，不构成模型失败"),
    "E20": ("blocked", "余额不足：403 token quota（need $1.600），请求未进入模型"),
    "S11": ("blocked", "余额不足：403 token quota（need $1.600），请求未进入模型"),
    "C17": ("blocked", "余额不足：403 token quota（need $25.000 预扣），参数校验未被触达"),
}
ICON = {"passed": "✅ 通过", "failed": "❌ 失败", "error": "❌ 错误", "warning": "⚠️ 警告", "blocked": "⏸ 未验证"}
GROUPS = ["基础能力", "增强能力", "专项能力", "契约健壮性", "满血指纹"]

FINDINGS_MD = """
## 一、结论摘要

| 维度 | 判定 | 说明 |
|---|---|---|
| **模型本体能力** | **达标** | 能力面无硬失败。数学、代码（经实际执行验证）、约束推理、工具闭环、结构化输出随机串强绑定、并行工具、流式、视觉、长上下文、前缀缓存、stop_sequences 全部成立 |
| **文档契约一致性** | **不达标** | 官方文档承诺返回 400 的参数校验，该通道**普遍不执行**（静默接受或静默丢弃），见 F4 |
| **用量/计费可信性** | **信任级失败** | `usage` 由 `max_tokens` 推导而非真实计量，见 F1 |
| **截断可判定性** | **不可用** | `output_tokens == max_tokens` 与 `stop_reason=end_turn` 自相矛盾，见 F2 |
| **本轮覆盖完整性** | **不完整** | 9 条用例未取证（8 条通道余额不足 + 1 条端点缺失），见第六节 |

**总判定：不建议使用该通道作为 Fable 5 的准入判定数据源。** 模型能力结论可用（能力面全绿），但凡涉及计费、用量审计、截断判定、参数边界契约的结论**不可用**。

---

## 二、被测对象与方式

- 端点：`https://i-token.ai/v1`（Anthropic Messages 协议，`POST /v1/messages`）
- 模型：`claude-fable-5`
- 方式：**独立直连**（`fable5_direct/main.py`，不经过平台 runner / Web / 任务队列），77 条用例、91 次请求，逐请求落盘 `requests.jsonl`
- 判据来源：Anthropic 官方《Migrating to Claude Fable 5》《Effort》《Messages API》文档 + 项目既有判定口径
  （负向断言要求 4xx 且非 404 且错误文案指向目标字段；文档承诺报错而线上未执行 → 记 warning）

---

## 三、关键发现

### F1【信任级】`usage` 由 `max_tokens` 推导，不是真实计量

决定性证据（同一条提示「只回复 OK」，仅改 `max_tokens`）：

| 请求 | 返回正文 | `output_tokens` | `thinking_tokens` |
|---|---|---:|---:|
| `max_tokens=64` | `OK` | **64** | 63 |
| `max_tokens=100` | `OK` | **100** | 99 |

真实生成量不可能随上限线性变化。旁证：

1. 批量批次 72 条 200 响应中，**38 条的 `output_tokens` 恰好等于 `max_tokens`**，且 `thinking_tokens = output_tokens − 1`（取值集中在 255 / 511 / 1023，即 2ⁿ−1）。
2. **23 条内容完全不同的请求**（S01–S09 的九种 beta、C02/C03/C05/C13、E02/E03/E15/E16…）全部报 `thinking_tokens=255`。真实的思考长度不可能在不同任务上完全一致。
3. 流式对照：`message_start` 报 `output_tokens=1`，`message_delta` 报 `output_tokens=100`（= `max_tokens`）。
4. 最强证据：`max_tokens=100` 要求「输出 1 到 60 的数字」，实测正文 170 字符 / 60 个数字（**可见部分本身就超过 100 token**），usage 仍报 `output=100 / think=99`——数值自相矛盾，不可能是实测。

> 影响：成本核算、用量审计、配额管理、以及"回答是否被截断"的判断全部不可依赖。

### F2【契约级】`stop_reason` 与 `output_tokens` 自相矛盾 → 截断不可判定

38 条响应 `output_tokens == max_tokens`，但 `stop_reason` 仍是 `end_turn`。按官方文档，`max_tokens` 是"思考 + 正文"的硬上限，命中上限时 `stop_reason` 必须为 `max_tokens`。全程只有刻意构造的 `max_tokens=16`（E19）才正确回显 `max_tokens`。

> 影响：长文 / 代码 / 结构化输出类任务若被静默截断，调用方无法察觉。

### F3【平台级】账号按 `max_tokens` 预扣额度，余额已耗尽

403 报错原文形如 `token quota is not enough, token remain quota: $0.913022, need quota: $1.600080`，其中 `need quota` **恒等于 `max_tokens × $50/1M`**，与实际生成量无关——即下单先按上限预扣。

余额消耗轨迹（同一批次内）：`$0.913 → $0.714 → $0.464 → $0.284 → $0.037 → $0.006`，当前约 `$0.012`。

> 影响：`max_tokens` 稍大即被拒，与模型能力无关；8 条用例因此无法取证。这也意味着该通道**无法用于长输出压测**。

### F4【契约级】官方文档承诺的 400，该通道普遍未执行

| 请求形态 | 官方文档 | 线上实测 |
|---|---|---|
| `thinking={"type":"disabled"}` | 400 | **200 静默接受** |
| `thinking={"type":"enabled","budget_tokens":N}` | 400 | **200 静默接受** |
| `temperature` / `top_p` / `top_k` | 400（已移除） | **200 静默丢弃** |
| 末尾 assistant 消息（prefill） | 400 | **200 被接受** |
| 缺失 `max_tokens` | 400（必填） | **200** |
| `messages` 角色非法（`robot`） | 400 | **200** |
| `output_config.effort` 非法值 | 400 | **200 静默接受** |
| 缺失 `anthropic-version` 头 | 400 | **200** |
| `max_tokens` 越界 | 400 | 403（余额预扣拦截，字段校验未被触达） |

**正确执行**的校验：空 `messages` → 400 `field messages is required`；畸形 JSON → 400；无效 Key → 401 `Invalid token`；未知模型 → 503 `无可用渠道`。

> 影响：依赖文档边界做健壮性/准入判定时，该通道的"未报错"不能解读为"参数被正确处理"。特别注意 `thinking=disabled` 被接受的实际后果——调用方以为关掉了思考以省成本，实际思考照常发生且挤占输出预算（E18 实测 `max_tokens=512` 时 `thinking_tokens=511`，正文只剩 1 个 token）。

### F5【能力级】模型本体能力大面成立（真实、可用）

- **基础能力 15/15 全通过**：身份回显 `claude-fable-5`；严格指令遵循逐字符一致；`4387×2963=12,998,681` 正确；中文材料推断正确；**代码生成经实际执行验证**（大小写不敏感计数 + 空串边界均过）；准确定位可变默认参数缺陷；约束推理得唯一解 `CAB`；1–20 质数一个不多一个不少。
- **增强能力**：adaptive thinking 被接受；effort 五档全部接受；工具闭环（`tool_use → tool_result →` 复述随机 marker）；`tool_choice=any` 强制调用；**并行工具调用单轮 2 个**；结构化输出随机串 + 固定值强绑定；流式事件序列完整且无 `[DONE]`；**视觉识别成立并有 token 双判据**（带图 401 vs 纯文本 51）；长上下文 27,406 input tokens；前缀缓存命中增量 5,135 ≥ 2,048；**`stop_sequences` 精确截断**（在 `D4` 前截断，`stop_reason=stop_sequence`）。
- **满血指纹**：随机动态工具被正确理解并调用、参数绑定本次随机串；随机结构化绑定通过；跨轮随机口令准确复述；响应 `model` 未被静默降级。

### F6【能力级】`effort` 档位差异被显著稀释

同一提示 `low` vs `max` 对照：`thinking_tokens` **662 vs 696（+5%）**，`output_tokens` 832 vs 951（+14%）。官方文档称 effort 控制思考深度、工具调用合并与详略，档位间应有显著差异。

叠加 F1（usage 伪造），**effort 是否真实生效无法可靠取证**，本轮记"未充分验证"。

### F7【通道级】思考摘要被剥离

`thinking.display="summarized"` 时，thinking 块的 `thinking` 文本长度**恒为 0**，只返回 `signature`（签名可解出 `claude-fable-5` + `thinking` + UUID 的可读片段，形似真实 Anthropic 签名）；流式下出现 `thinking_delta` 但内容为空。官方文档承诺 `summarized` 时 thinking 块应携带可读摘要。

### F8【通道级】端点缺失与中转指纹

- `POST /v1/messages/count_tokens` → **404 `Invalid URL`**，端点未实现（C12 无法取证）。
- 报错体泄露分组名：`分组 ccmax 下模型 <model> 无可用渠道（distributor）`。
- OpenAI 面 `/v1/chat/completions` 是**协议转译**（非原生）：响应含 `billing_usage.source=claude_messages`、`usage_source=anthropic`、`claude_cache_creation_5_m_tokens` 等非标准字段，顶层 `output_tokens` 恒为 0。
- **Messages 面顶层字段干净**：未命中任何中转标记（`billing_usage`/`access_programs`/`system_fingerprint` 等），未出现 `choices`/`object` 等外来协议结构 → 该面协议未被改写。

### F9【专项级】未观测到 `refusal` 语义

官方文档承诺：安全分类器拒绝时返回 **HTTP 200 + `stop_reason="refusal"` + `stop_details.category`**（cyber / bio / reasoning_extraction）。实测以 reasoning_extraction 域提示探测（索取未摘要的原始思维链），返回的是回避性回答（且 usage 报 1024/1023），**未出现 `refusal` 或 `stop_details`**。

### F10 两条原判"失败"系判据缺陷，已更正

| 用例 | 原判定 | 复核结论 |
|---|---|---|
| E17 `stop_sequences` | 失败 | **判据缺陷**。原写法让模型"遇到标记就停"，模型提前自然收尾，把"停止序列未生效"与"模型自行结束"混为一谈。改用必然出现的分隔符后：`stop_reason=stop_sequence`，正文在 `D4` 前精确截断 → **通过** |
| E18 `system 指令` | 失败 | **判据缺陷**。原判据要求"必须且只能输出固定串"，实测该指令放在 **user** 消息里同样被拒（原文：`I can't follow an instruction to output only a fixed string regardless of context`），属模型既定对齐行为，不是 `system` 未生效。改用格式前缀约束后回复以指定 tag 开头 → **通过** |

> 这两条已回写进 `fable5_direct/cases.py`，避免后续重复误判。

---

## 四、分维度结果（复核后）

| 维度 | 总数 | 通过 | 失败 | 警告 | 未验证 |
|---|---:|---:|---:|---:|---:|
"""

FINDINGS_TAIL_MD = """
> **警告** = 线上行为与官方文档承诺存在差异、但功能可用、不足以判定模型失败（本次全部警告均属此类）。
> **未验证** = 通道侧前提不满足（余额不足 / 端点缺失 / 端点 404），目标特性从未被触达，不计入模型能力结论。

---

## 五、逐条结果

"""

LIMITS_MD = """
---

## 六、未取证清单与重跑条件

| 用例 | 未取证原因 | 重跑条件 |
|---|---|---|
| E20 大 `max_tokens` 接受性 | 余额不足（需预扣 $1.600） | 充值 |
| S11 长输出生成（流式） | 余额不足（需预扣 $1.600） | 充值 |
| C17 `max_tokens` 越界拒绝 | 余额不足（需预扣 $25.000） | 充值 |
| A04 思考真实性（`effort=max`） | 余额不足（需预扣 $0.410） | 充值 |
| A06 原始思维链不得泄漏 | 余额不足（需预扣 $0.103） | 充值 |
| A09 中转注入识别 | 余额不足（需预扣 $0.013） | 充值 |
| A10 非标准字段与协议改写 | 余额不足 | 充值（冒烟轮已通过：Messages 顶层字段干净） |
| A11 流式事件真实性 | 余额不足（需预扣 $0.026） | 充值 |
| A08 同族别名路由（部分） | `claude-fable-5-1` 余额不足 | 充值 |
| C12 `count_tokens` 交叉验证 | 端点 404 `Invalid URL` | 需通道补实现该端点 |

> 补充口径：`S01`–`S09` 的 200 只能证明"参数未被拒绝"。该通道对未支持参数的默认行为是**静默接受**（见 F4），因此 **beta 能力是否真正生效，本次未做效果级验证**，不能据 200 判定 beta 可用。

---

## 七、建议

### 通道侧（使用建议）
1. **不要把该通道的返回作为 Fable 5 的准入/计费依据**：`usage` 伪造 + 文档边界校验普遍未执行，两者都会让判定结论失真。
2. 若目的只是核验**模型本体能力**：结论可用——能力面无硬失败。
3. 若要核验 **beta 效果、refusal 语义、长输出**：需换直连官方（或可提供真实 usage 的通道）复核。
4. 长输出场景在该通道**当前不可用**（预扣额度按 `max_tokens` 计，余额不足即 403）。

### 测试套件侧（修法建议）
1. `fable5_direct/harness.py` 已把 **403 + quota 归类为 `blocked`**（平台前提不满足），不再误记 `failed`。
2. `cases.py` 已修正 `stop_sequences` 与 `system` 两条判据。
3. **建议新增"用量可信性"用例**（本次发现的最有价值的一条）：同一提示分别以 `max_tokens=M` 与 `max_tokens=2M` 调用两次，若 `output_tokens` 随 `max_tokens` 变化（尤其 `== max_tokens`），即判**信任级失败**。
4. **建议新增"截断一致性"检查**：`output_tokens == max_tokens && stop_reason == "end_turn"` 为矛盾组合，命中即记信任级失败。
"""


def main() -> int:
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out").resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    for r in summary["results"]:
        fix = CORRECTIONS.get(r["case_id"])
        if fix:
            r["original_status"] = r["status"]
            r["status"] = fix[0]
            r["passed"] = fix[0] == "passed"
            r["details"] = f"【复核更正】{fix[1]}　｜　自动判定原述：{r['details']}"
    counts = {k: sum(r["status"] == k for r in summary["results"]) for k in ICON}
    hard = counts["failed"] + counts["error"]
    evaluated = counts["passed"] + hard
    by_group: dict[str, dict] = {}
    for r in summary["results"]:
        g = by_group.setdefault(r["group"], {"total": 0, "passed": 0, "failed": 0, "warning": 0, "blocked": 0})
        g["total"] += 1
        g[r["status"]] = g.get(r["status"], 0) + 1

    md = ["# Fable 5 直连实测报告", "",
          f"- 被测端点：`{summary['target']['base_url']}`　模型：`{summary['target']['model']}`",
          f"- 执行方式：**独立直连**（不经平台 runner / Web / 任务队列）",
          f"- 生成时间：{summary['generated_at']}",
          f"- 用例总数：**{len(summary['results'])}**　请求总数：**91**（含重试与手工探针另计）", "", FINDINGS_MD]
    for g in GROUPS:
        s = by_group.get(g)
        if s:
            md.append(f"| {g} | {s['total']} | {s.get('passed', 0)} | "
                      f"{s.get('failed', 0)} | {s.get('warning', 0)} | {s.get('blocked', 0)} |")
    md += ["", f"**合计：通过 {counts['passed']} / 失败 {hard} / 警告 {counts['warning']} / 未验证 "
               f"{counts['blocked']}；已评估用例通过率 {counts['passed'] / evaluated:.1%}**", "", FINDINGS_TAIL_MD]
    for g in GROUPS:
        rows = [r for r in summary["results"] if r["group"] == g]
        if not rows:
            continue
        md += [f"### {g}", "", "| 用例 | 结果 | 耗时(s) | 判定说明 |", "|---|---|---:|---|"]
        for r in rows:
            d = str(r["details"]).replace("|", "\\|").replace("\n", " ")
            md.append(f"| `{r['case_id']}` {r['title']} | {ICON.get(r['status'])} | {r['elapsed']:.1f} | {d} |")
        md.append("")
    md.append(LIMITS_MD)
    (run_dir / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # HTML：把 Markdown 要点渲染成可读页面（表格 + 卡片）
    color = {"passed": "#16a34a", "failed": "#dc2626", "error": "#dc2626",
             "warning": "#d97706", "blocked": "#6b7280"}
    bg = {"passed": "#f0fdf4", "failed": "#fef2f2", "error": "#fef2f2",
          "warning": "#fffbeb", "blocked": "#f3f4f6"}
    secs = []
    for g in GROUPS:
        rows = [r for r in summary["results"] if r["group"] == g]
        if not rows:
            continue
        s = by_group[g]
        trs = "".join(
            f'<tr><td class="cid">{r["case_id"]}</td><td>{r["title"]}</td>'
            f'<td><span class="pill" style="color:{color[r["status"]]};background:{bg[r["status"]]}">'
            f'{ICON[r["status"]]}</span></td><td class="num">{r["elapsed"]:.1f}</td>'
            f'<td class="det">{str(r["details"])[:400]}</td></tr>' for r in rows)
        secs.append(f'<section><h2>{g}<span class="cnt">{s["total"]} 条 · 通过 {s.get("passed",0)} · '
                    f'失败 {s.get("failed",0)} · 警告 {s.get("warning",0)} · 未验证 {s.get("blocked",0)}</span></h2>'
                    f'<table><thead><tr><th>ID</th><th>用例</th><th>结果</th><th>耗时</th><th>判定说明</th></tr>'
                    f'</thead><tbody>{trs}</tbody></table></section>')
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Fable 5 直连实测报告</title><style>
 body{{margin:0;padding:32px 20px;background:#f8fafc;color:#0f172a;line-height:1.65;
  font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}
 .wrap{{max-width:1150px;margin:0 auto}} h1{{font-size:26px;margin:0 0 6px}}
 .sub{{color:#64748b;font-size:13px;margin-bottom:22px}}
 .verdict{{background:#fff;border:1px solid #e2e8f0;border-left:5px solid #dc2626;border-radius:12px;
  padding:16px 20px;margin-bottom:22px}} .verdict b{{color:#dc2626}}
 .kpis{{display:flex;gap:12px;flex-wrap:wrap;margin:0 0 26px}}
 .kpi{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:14px 20px;min-width:110px}}
 .kpi .v{{font-size:26px;font-weight:700}} .kpi .k{{font-size:12px;color:#64748b}}
 section{{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:6px 18px 16px;margin-bottom:20px}}
 h2{{font-size:17px;display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}}
 .cnt{{font-size:12px;font-weight:400;color:#64748b}}
 table{{width:100%;border-collapse:collapse;font-size:13px}}
 th{{text-align:left;color:#64748b;font-weight:600;font-size:12px;padding:8px;border-bottom:1px solid #e2e8f0}}
 td{{padding:9px 8px;border-bottom:1px solid #f1f5f9;vertical-align:top}} tr:last-child td{{border-bottom:none}}
 .cid{{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#475569;white-space:nowrap}}
 .num{{text-align:right;color:#64748b}} .det{{color:#334155}} .pill{{padding:2px 9px;border-radius:999px;font-size:12px;white-space:nowrap}}
 code{{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}}
 .find{{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:4px 22px 18px;margin-bottom:20px}}
</style></head><body><div class="wrap">
<h1>Fable 5 直连实测报告</h1>
<div class="sub">端点 <code>{summary['target']['base_url']}</code> · 模型 <code>{summary['target']['model']}</code> ·
{summary['generated_at']} · 独立直连，未经平台队列</div>
<div class="verdict">模型本体能力<b>达标</b>（能力面无硬失败）；但该通道 <b>usage 计量被伪造</b>、
文档承诺的参数校验<b>普遍未执行</b>，且<b>账号余额已耗尽</b>导致 9 条用例未取证。
<br>结论：<b>不建议用该通道作为 Fable 5 的准入判定数据源。</b></div>
<div class="kpis">
 <div class="kpi"><div class="v">{len(summary['results'])}</div><div class="k">用例总数</div></div>
 <div class="kpi"><div class="v" style="color:#16a34a">{counts['passed']}</div><div class="k">通过</div></div>
 <div class="kpi"><div class="v" style="color:#dc2626">{hard}</div><div class="k">失败</div></div>
 <div class="kpi"><div class="v" style="color:#d97706">{counts['warning']}</div><div class="k">警告（留痕）</div></div>
 <div class="kpi"><div class="v" style="color:#6b7280">{counts['blocked']}</div><div class="k">未验证</div></div>
 <div class="kpi"><div class="v">{counts['passed'] / evaluated:.0%}</div><div class="k">已评估通过率</div></div>
</div>
<div class="find"><h2>关键发现</h2><ol>
<li><b>信任级</b>｜同一提示仅改 <code>max_tokens</code>（64→64、100→100），<code>output_tokens</code> 随上限线性变化；批次内 38/72 条 <code>output_tokens == max_tokens</code>，23 条不同任务的 <code>thinking_tokens</code> 全为 255。→ <b>usage 非真实计量</b>。</li>
<li><b>契约级</b>｜38 条 <code>output_tokens == max_tokens</code> 时 <code>stop_reason</code> 仍为 <code>end_turn</code>。→ <b>截断不可判定</b>。</li>
<li><b>平台级</b>｜403 预扣额度 <code>need quota = max_tokens × $50/M</code>，余额已从 $0.913 耗至约 $0.012。→ <b>长输出场景不可用</b>。</li>
<li><b>契约级</b>｜<code>thinking=disabled</code>、<code>budget_tokens</code>、<code>temperature/top_p/top_k</code>、prefill、缺 <code>max_tokens</code>、非法 role —— 文档承诺 400，线上全部 <b>200 静默接受</b>。</li>
<li><b>能力级</b>｜基础 15/15 全通过；工具闭环、并行工具、结构化强绑定、视觉（401 vs 51 tokens）、长上下文 27406 tokens、缓存命中 5135、<code>stop_sequences</code> 精确截断均成立。</li>
<li><b>通道级</b>｜<code>count_tokens</code> 端点 404；思考摘要被剥离（<code>display=summarized</code> 仍返回空 thinking）；<code>effort</code> 档位差异仅 +5%。</li>
<li><b>更正</b>｜原判失败的 <code>stop_sequences</code> 与 <code>system</code> 两条经复核为<b>判据缺陷</b>，已更正为通过并回写代码。</li>
</ol></div>
{"".join(secs)}
</div></body></html>"""
    (run_dir / "report.html").write_text(html, encoding="utf-8")
    print(f"已生成：{run_dir/'REPORT.md'}、{run_dir/'report.html'}")
    print(f"复核后：通过 {counts['passed']} / 失败 {hard} / 警告 {counts['warning']} / 未验证 {counts['blocked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
