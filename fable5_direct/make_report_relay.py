#!/usr/bin/env python3
"""按「中转口径」产出 Fable 5 通道测评报告（Markdown + HTML）。

与 make_report.py（官方契约口径）的区别：本脚本**不以官方文档的 400 清单作为合格性尺子**，
改以「中转通道职能」立论——忠实转发 + 如实计费。官方 400 未执行只作口径盘点，
唯二红线是「能力保真」与「计费诚实」。

用法：python make_report_relay.py <run_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 复核更正（与官方口径版共用）：用例 id -> (修正后状态, 依据)
CORRECTIONS: dict[str, tuple[str, str]] = {
    "E17": ("passed", "判据缺陷更正：改用必然出现的分隔符后 stop_reason=stop_sequence，正文在 D4 前精确截断（手工探针复核）"),
    "E18": ("passed", "判据缺陷更正：原判据「必须只输出固定串」实为模型对齐行为（同指令放 user 消息亦被拒）；改用格式前缀后回复以指定 tag 开头（手工探针复核）"),
    "E03": ("warning", "中转口径重定性：该用例测的是「文档承诺的 400」，中转从宽接受**不属缺陷**；但响应仍带 thinking 块属**语义失控**，保留留痕（见 G2）"),
    "E04": ("warning", "中转口径重定性：同上——接受不报错非缺陷，预算未受约束属语义失控，保留留痕（见 G2）"),
    "S10": ("warning", "未观测到 refusal 语义属未取证，不构成模型失败"),
    "E20": ("blocked", "余额不足：403 token quota（need $1.600），请求未进入模型"),
    "S11": ("blocked", "余额不足：403 token quota（need $1.600），请求未进入模型"),
    "C17": ("blocked", "余额不足：403 token quota（need $25.000 预扣），参数校验未被触达"),
}

# 中转口径重定性：用例 id -> 该条在「中转标准」下的结论（覆盖官方口径的措辞）
RELAY_RECLASS: dict[str, str] = {
    "E03": "⚠️ **语义失控**：`thinking=disabled` 被接受但**响应仍带 thinking 块**，关闭思考的意图静默失效（见 G2）",
    "E04": "⚠️ **语义失控**：`budget_tokens` 被接受但未约束思考预算，思考量调用方不可控（见 G2）",
    "C03": "✅ 非缺陷：不支持的采样参数被静默丢弃；Fable 5 本就不支持 `temperature`，无实际影响",
    "C04": "✅ 非缺陷：同上（`top_p` 被丢弃，无实际影响）",
    "C05": "✅ 非缺陷：同上（`top_k` 被丢弃，无实际影响）",
    "C06": "⚠️ **语义失控**：末尾 assistant 消息（prefill）**未被沿用**，响应不是 prefill 的续写（见 G4）",
    "C07": "✅ 可接受：缺 `max_tokens` 由内部补默认上限后正常作答（需向调用方公布默认值）",
    "C09": "✅ 非缺陷：非法 role（`robot`）被归一化后正常作答，模型输出正常",
    "C19": "✅ 非缺陷：未知顶层字段被忽略，属前向兼容的常规做法",
    "C20": "✅ 非缺陷：缺 `anthropic-version` 由中转补默认，无实际影响",
    "C02": "✅ 非缺陷：错误体结构为网关自定义（New API 风格），不影响调用方解析状态码",
    "C13": "⚠️ 字段齐全但**数值不实**——`usage` 由 `max_tokens` 推导，字段完整 ≠ 可信（R2 红线，见 G1）",
    "C15": "✅ `stop_reason` 词表正确，`max_tokens` 截断能正确回显",
    "E01": "⚠️ 需知悉：`adaptive` 被接受，但 `display=summarized` 的思考摘要被剥离（thinking 文本恒为空）",
    "E12": "⚠️ 需知悉：流式存在 `thinking_delta` 事件但内容为空（摘要同样被剥离）",
    "E16": "✅ 透传正常：前缀缓存真实生效（`cache_read_input_tokens=5135`）",
    "E17": "✅ 透传正常：`stop_sequences` 真实生效（`stop_reason=stop_sequence`，精确截断）",
    "E18": "✅ 透传正常：`system` 真实生效（格式前缀约束被遵守）",
    "E19": "✅ `max_tokens` 上限被真正强制执行，截断可检测",
    "E14": "✅ 透传正常：图像被真实编码（带图 401 vs 纯文本 51 input tokens）",
    "E15": "✅ 透传正常：长上下文被真实接收（27,406 input tokens）",
    "S01": "⏸ 中性：200 仅证明参数**未被拒绝**；beta 是否真转发到上游，本通道无从证明（同理见 S02–S09）",
    "S10": "⏸ 未取证：未观测到 `refusal` 语义（属未触达，不构成模型失败）",
    "A05": "⚠️ 存疑：`effort=low/max` 输出差异仅 +5%~14%；且受 G1 影响，**档位是否真实生效无法可靠取证**",
    "A07": "✅ 身份保真：`model` 未被静默降级（回显 `claude-fable-5`）",
    "A08": "⚠️ 需知悉：同族别名路由未完全取证（余额不足）",
    "B01": "✅ 能力保真：身份回显正确",
}
for _i in range(2, 10):
    RELAY_RECLASS.setdefault(f"S0{_i}", "⏸ 中性：200 仅证明参数未被拒绝，beta 是否生效未做效果级验证")

ICON = {"passed": "✅ 通过", "failed": "❌ 失败", "error": "❌ 错误", "warning": "⚠️ 警告", "blocked": "⏸ 未验证"}
GROUPS = ["基础能力", "增强能力", "专项能力", "契约健壮性", "满血指纹"]
GROUP_RELAY = {
    "基础能力": "R1 能力保真",
    "增强能力": "R1 能力保真 / M1 参数透传",
    "专项能力": "R1 / M1（beta 参数可信度）",
    "契约健壮性": "M1 参数透传口径",
    "满血指纹": "R3 身份保真 / R2 计费链",
}

FINDINGS_MD = """
## 一、测评立场：本报告按「中转口径」判定

中转通道（API 中转 / 聚合站）的职能是**忠实转发 + 如实计费**，而不是逐条复刻官方 API 的报错契约。
官方文档里的 400 清单（`thinking` 配置非法、`temperature` 已移除、缺 `max_tokens` 必须报错 ……）属于**官方实现的合规项**；
中转普遍采取「宽进归一化」，若拿这份清单当合格性尺子，会派生出大量**假缺陷**（本次原报告里 7 条 warning 中的 5 条即属此类）。

因此本报告的判定基线收敛为**三条红线 + 两项运行指标**：

| 编号 | 维度 | 含义 |
|---|---|---|
| **R1 能力保真** | 转发后模型能力不打折（不降智、不静默截断、不篡改回答） |
| **R2 计费诚实** | `usage` 反映真实生成量，计费模型可预期 |
| **R3 身份/协议保真** | 不注入提示词、不改写协议结构、不静默降级模型 |
| **M1 参数透传口径** | 参数被透传还是被归一化；归一化是否**改变调用方语义** |
| **M2 可用性** | 端点完整性、容量/余额模型、错误可读性 |

判定原则：

1. **R1 / R2 / R3 是合格性红线**——任一不过，该通道不具备生产可用性。
2. **M1 只做口径盘点**：官方 400 未执行**不算缺陷**。但「**静默吞掉会改变语义的参数**」（调用方按文档传了、以为生效、实际没生效且无任何提示）单独标为 **语义失控**，属实质风险。
3. 套件原有的 `warning`（线上与官方文档有差异但功能可用）在中转口径下**多数直接消解为「非缺陷」**。

---

## 二、总判定

| 维度 | 中转口径结论 | 依据 |
|---|---|---|
| **R1 能力保真** | ✅ **达标** | 能力面零硬失败（数学 / 代码经实际执行 / 约束推理 / 工具闭环 / 并行工具 / 结构化强绑定 / 视觉 / 长上下文 / 缓存 / 流式均成立） |
| **R2 计费诚实** | ❌ **不达标（红线）** | `usage` 随 `max_tokens` 变化、且与响应内容自相矛盾（见 G1） |
| **R3 身份/协议保真** | ✅ **达标** | Messages 面顶层字段零污染、无 prompt 注入迹象、`model` 未被降级（见 G6） |
| **M1 参数透传口径** | ⚠️ **主体正常，3 处语义失控** | `thinking` 开关/预算、`prefill` 被静默吞掉（见 G2 / G4） |
| **M2 可用性** | ⚠️ **受限** | 按 `max_tokens` 预扣额度、余额近枯竭；`count_tokens` 端点未实现（见 G7） |

**总判定：**

- 只用来**核验模型本体能力**（准入判定）：**可用**——R1、R3 达标，能力结论可信。
- 用作**生产通道**（涉及成本核算、用量审计、输出长度控制、参数语义控制）：**不可用**——R2 踩红线，M1 存在语义失控。
- **长输出场景当前不可用**——通道按下单 `max_tokens` 预扣额度，余额不足即 403。

---

## 三、关键发现

### G1【R2 红线】`usage` 不是实测值，且可被 `max_tokens` 直接拉动

按证明力排序：

| # | 探针 | 观测 | 说明 |
|---|---|---|---|
| 1 | 同一提示「只回复 OK」，**仅改 `max_tokens`** | 64 → `output_tokens=64 / thinking=63`；100 → `output_tokens=100 / thinking=99` | 正文完全相同，用量却跟着**上限**走（两次独立调用，可复现） |
| 2 | `max_tokens=100`，要求输出 1–60（正文 170 字符） | usage 仍报 `output=100 / think=99` | 可见正文本身已超过 100 token，与 usage **自相矛盾** |
| 3 | 全批 72 条 200 响应 | **38 条**的 `output_tokens == max_tokens` | 命中率高得不像真实分布 |
| 4 | `max_tokens=16`（E19） | `output=16`、`stop_reason=max_tokens`、正文恰为 1..16 | 上限**确实被强制执行**；但 usage 同时声称 `think=15`，与正文 16 行相矛盾 |
| 5 | 反例：A05 `max_tokens=4096` | `output=832 / 951` ≠ 上限 | 该值**既非恒等于上限、也非实测**，缺乏可解释的计量口径 |

> **影响**：成本核算、用量审计、配额管理、以及「回答是否被静默截断」的判断**全部不可依赖**。

### G2【M1·语义失控】请求侧的 `thinking` 配置无法控制响应侧行为

- `thinking={"type":"disabled"}`（E03）→ **HTTP 200，且响应仍带 `thinking` 块**。
- `thinking={"type":"enabled","budget_tokens":2048}`（E04）→ 200，响应同样带 `thinking` 块。
- 全批统计：**71 / 72 条** 200 响应带 `thinking` 块；响应形态只有 `(thinking, text)` 与 `(thinking, tool_use…)` 两种。

→ 中转层对 `thinking` 做了**统一归一化（恒开）**，调用方**无法**通过参数控制思考的开关与预算。
这不是「文档 400 未执行」这种纸面问题，而是**实际语义被改写**：想「关思考省 token / 降延迟」的调用方会静默失效，
而且由于 G1，连「是否多花了钱」都无法核算。

### G3【M1·口径盘点】参数透传对照表

| 请求参数 | 中转行为 | 对调用方语义的影响 | 中转口径评价 |
|---|---|---|---|
| `thinking={"type":"disabled"}` | 200，**响应仍带 thinking 块** | 关闭思考的意图**失效** | ⚠️ 语义失控 |
| `thinking={"type":"enabled","budget_tokens":N}` | 200，预算未受约束 | 思考预算**不可控** | ⚠️ 语义失控 |
| 末尾 assistant 消息（prefill） | 200，但**响应非 prefill 续写** | 锁定输出起点的意图**失效** | ⚠️ 语义失控 |
| `temperature` / `top_p` / `top_k` | 200，参数被丢弃 | 无（Fable 5 本不支持采样参数） | ✅ 非缺陷 |
| 缺 `max_tokens` | 200，内部补默认上限 | 需知悉默认值 | ✅ 可接受 |
| `messages` role 非法（`robot`） | 200，被归一化后正常作答 | 无 | ✅ 非缺陷 |
| 未知顶层字段 | 200，被忽略 | 无（前向兼容） | ✅ 非缺陷 |
| 缺 `anthropic-version` 头 | 200，中转补默认 | 无 | ✅ 非缺陷 |
| `output_config.effort` 非法值 | 200，静默接受 | 可能被归一为默认档 | ⚠️ 需知悉 |
| `output_config.effort` 合法值 | **真实透传**（`low` vs `max` 输出不同） | 无 | ✅ 透传正常 |
| `stop_sequences` | **真实透传**（精确截断） | 无 | ✅ 透传正常 |
| `system` | **真实透传**（格式前缀被遵守） | 无 | ✅ 透传正常 |
| `tools` / `tool_choice` | **真实透传**（闭环成立、`any` 强制调用） | 无 | ✅ 透传正常 |
| `cache_control`（前缀缓存） | **真实生效**（`cache_read=5135`） | 无 | ✅ 透传正常 |
| 图像块（vision） | **真实生效**（带图 401 vs 文本 51 tokens） | 无 | ✅ 透传正常 |
| `max_tokens` 越界 | 403（余额预扣**先**拦截） | 字段校验未被触达 | ⏸ 未验证 |

**小结**：真正被中转支持的参数面（`effort` / `stop_sequences` / `system` / `tools` / `cache_control` / 图像）**透传正常**；
凡**调用方无感知的归一化**中，只有 3 处会**改变语义**（`thinking` 开关、`thinking` 预算、`prefill`），其余均为无害。

### G4【M1·语义失控】`prefill` 未被沿用

请求末条消息为 `{"role":"assistant","content":"尊敬的客户，P-095EA345"}`（C06），
响应正文实为 `您好！您的订单（编号：P-095EA345）已成功完成支付。感谢您的惠顾。`——**并非对 prefill 的续写**。
调用方若依赖 prefill 锁定输出起点（强制 JSON 开头、稳定格式等），会静默失效。

### G5【R1 达标】能力保真——转发后模型能力未打折

- **基础能力 15 / 15 全过**：身份回显；严格指令遵循逐字符一致；`4387×2963=12,998,681` 正确；中文材料推断正确；
  **代码生成经实际执行验证**（大小写不敏感计数 + 空串边界）；准确定位可变默认参数缺陷；约束推理唯一解 `CAB`；质数枚举一个不多一个不少。
- **增强能力**：工具闭环（`tool_use → tool_result →` 复述随机 marker）；`tool_choice=any` 强制调用；**并行工具调用单轮 2 个**；
  结构化输出随机串 + 固定值强绑定；流式事件序列完整且无 `[DONE]`；视觉识别成立；长上下文 27,406 tokens；
  前缀缓存命中 5,135；`stop_sequences` 精确截断。
- **满血指纹**：随机动态工具被正确理解并调用、参数绑定本次随机串；跨轮随机口令准确复述；`model` 未被静默降级。

### G6【R3 达标】身份与协议保真

- **Messages 面顶层字段零污染**：仅 `{content,id,model,role,stop_details,stop_reason,stop_sequence,type,usage}`，
  未出现 `billing_usage` / `system_fingerprint` / `choices` / `object` 等外来协议结构或注入标记。
- **无 prompt 注入迹象**：未传 `instructions` 时未见被替换成固定系统提示；`system` 参数按原样生效。
- **模型未被降级**：所有响应 `model` 恒为 `claude-fable-5`。
- **OpenAI 兼容面是转译而非原生**（`billing_usage.source=claude_messages`）——在中转口径下这是**加分项**（提供 OpenAI 兼容是常规能力），
  只要不污染 Messages 面即可，实测未污染。

### G7【M2 受限】容量模型与端点完整性

- 403 报错原文形如 `token quota is not enough, token remain quota: $0.913022, need quota: $1.600080`，
  其中 **`need quota` 恒等于 `max_tokens × $50/1M`**——即**下单先按上限预扣**，与实际生成量无关。
  余额轨迹：`$0.913 → $0.714 → $0.464 → $0.284 → $0.037 → $0.006`（批次内），当前约 `$0.012`。
- `POST /v1/messages/count_tokens` → **404 `Invalid URL`**，端点未实现。
- 错误体泄露分组名：`分组 ccmax 下模型 <model> 无可用渠道（distributor）`。

> 影响：`max_tokens` 稍大即 403，与模型能力无关；**该通道做不了长输出压测**，8 条用例因此未取证。

### G8【R1 存疑】`effort` 档位差异被稀释

同一提示 `low` vs `max` 对照：`thinking_tokens` **662 vs 696（+5%）**，`output_tokens` 832 vs 951（+14%）。
叠加 G1（usage 不可信），**档位是否真实生效无法可靠取证**，本轮记「存疑」。

### G9 两条原判「失败」系判据缺陷，已更正

| 用例 | 原判定 | 复核结论 |
|---|---|---|
| E17 `stop_sequences` | 失败 | **判据缺陷**。原写法让模型「遇到标记就停」，模型提前自然收尾。改用必然出现的分隔符后：`stop_reason=stop_sequence`，正文在 `D4` 前精确截断 → **通过** |
| E18 `system 指令` | 失败 | **判据缺陷**。原判据要求「只输出固定串」，实测该指令放在 **user** 消息里同样被拒，属模型既定对齐行为。改用格式前缀后回复以指定 tag 开头 → **通过** |

> 两条已回写 `fable5_direct/cases.py`。另：E03 / E04 原按官方口径判「失败」，在中转口径下**改判为「非缺陷 + 语义失控风险」**（见 G2）。

---

## 四、分维度结果（复核后）

| 维度 | 主判中转维度 | 总数 | 通过 | 失败 | 警告 | 未验证 |
|---|---|---:|---:|---:|---:|---:|
"""

FINDINGS_TAIL_MD = """
> 口径说明：本表沿用测试套件分组。「警告」在中转口径下**多数已消解为非缺陷**（官方文档差异，功能可用）；
> 「未验证」= 通道侧前提不满足（余额不足 / 端点缺失），目标特性从未被触达，不计入模型能力结论。

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
| A09 中转注入识别 | 余额不足（需预扣 $0.007） | 充值（冒烟轮已过：`system` 生效、token 未见异常膨胀） |
| A10 非标准字段与协议改写 | 余额不足 | 充值（冒烟轮已过：Messages 顶层字段干净） |
| A11 流式事件真实性 | 余额不足（需预扣 $0.026） | 充值 |
| A08 同族别名路由（部分） | `claude-fable-5-1` 余额不足 | 充值 |
| C12 `count_tokens` 交叉验证 | 端点 404 `Invalid URL` | 需通道补实现该端点 |

> 补充口径：`S01`–`S09` 的 200 **只能证明参数未被拒绝**。该通道对未支持参数的默认行为是**静默接受**，
> 因此 **beta 能力是否真正生效，本次未做效果级验证**，不能据 200 判定 beta 可用。

---

## 七、建议

### 通道侧（怎么用）

1. **可用的部分**：把该通道当作**能力核验通道**——R1 能力保真、R3 身份保真均达标，能力结论可直接采信。
2. **不可用的部分**：不要用它做**成本核算 / 用量审计 / 输出长度控制**——`usage` 不实，且 `max_tokens` 被预扣。
3. **需要显式规避的坑**：不要依赖 `thinking` 开关或预算控制成本与延迟（G2）、不要依赖 `prefill` 锁定输出起点（G4）。
4. **长输出**：当前通道不可用（预扣模型 + 余额近枯竭）。
5. 若要核验 **beta 效果、`refusal` 语义、长输出**，需换直连官方（或可提供真实 usage 的通道）复核。

### 测试套件侧（修法建议）

1. `fable5_direct/harness.py` 已把 **403 + quota 归类为 `blocked`**（通道前提不满足），不再误记 `failed`。
2. `cases.py` 已修正 `stop_sequences` 与 `system` 两条判据。
3. **建议新增「用量可信性」用例**（本轮最有价值的一条）：同一提示分别以 `max_tokens=M` 与 `max_tokens=2M` 调用两次，
   若 `output_tokens` 随 `max_tokens` 变化，即判 **R2 红线失败**。
4. **建议新增「语义失控」用例族**：对「调用方以为会改变行为」的参数（`thinking` 开关 / `budget_tokens` / `prefill`）
   做**行为级**验证，而非只验证是否报错——本次正是靠这类探针才发现 G2 / G4。
5. **建议新增「截断一致性」检查**：`output_tokens == max_tokens && stop_reason == "end_turn"` 为矛盾组合，命中即记 R2 失败。
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

    md = ["# Fable 5 中转通道测评报告（中转口径）", "",
          f"- 被测通道：`{summary['target']['base_url']}`　模型：`{summary['target']['model']}`",
          f"- 判定基线：**中转三红线（R1 能力保真 / R2 计费诚实 / R3 身份保真）+ 两项运行指标（M1 参数透传 / M2 可用性）**",
          f"- 执行方式：**独立直连**（不经平台 runner / Web / 任务队列），逐请求落盘 `requests.jsonl`",
          f"- 生成时间：{summary['generated_at']}",
          f"- 用例总数：**{len(summary['results'])}**　请求总数：**91**（含重试；另有手工对照探针未入档）", "",
          "> 官方文档的 400 断言在本报告中**降级为口径盘点（M1）**，不作合格性要求。", "", FINDINGS_MD]
    for g in GROUPS:
        s = by_group.get(g)
        if s:
            md.append(f"| {g} | {GROUP_RELAY.get(g, '')} | {s['total']} | {s.get('passed', 0)} | "
                      f"{s.get('failed', 0)} | {s.get('warning', 0)} | {s.get('blocked', 0)} |")
    md += ["", f"**合计：通过 {counts['passed']} / 硬失败 {hard} / 警告 {counts['warning']} / 未验证 "
               f"{counts['blocked']}**（已评估用例零硬失败；警告为口径差异或需知悉项，见逐条「中转口径结论」列）", "", FINDINGS_TAIL_MD]
    for g in GROUPS:
        rows = [r for r in summary["results"] if r["group"] == g]
        if not rows:
            continue
        md += [f"### {g}　<span style=\"color:#64748b;font-weight:400;font-size:13px\">（{GROUP_RELAY.get(g, '')}）</span>", "",
               "| 用例 | 结果 | 耗时(s) | 中转口径结论 | 自动判定说明 |", "|---|---|---:|---|---|"]
        for r in rows:
            d = str(r["details"]).replace("|", "\\|").replace("\n", " ")
            rl = RELAY_RECLASS.get(r["case_id"], "").replace("|", "\\|").replace("\n", " ")
            md.append(f"| `{r['case_id']}` {r['title']} | {ICON.get(r['status'])} | {r['elapsed']:.1f} | {rl} | {d} |")
        md.append("")
    md.append(LIMITS_MD)
    (run_dir / "REPORT-relay.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---------- HTML ----------
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
            f'<td class="det">{str(RELAY_RECLASS.get(r["case_id"], "") or "—")}</td>'
            f'<td class="det dim">{str(r["details"])[:300]}</td></tr>' for r in rows)
        secs.append(f'<section><h2>{g}<span class="cnt">{GROUP_RELAY.get(g, "")} · {s["total"]} 条 · 通过 {s.get("passed",0)} · '
                    f'失败 {s.get("failed",0)} · 警告 {s.get("warning",0)} · 未验证 {s.get("blocked",0)}</span></h2>'
                    f'<table><thead><tr><th>ID</th><th>用例</th><th>结果</th><th>耗时</th>'
                    f'<th>中转口径结论</th><th>自动判定说明</th></tr>'
                    f'</thead><tbody>{trs}</tbody></table></section>')

    dim_rows = [
        ("R1 能力保真", "#16a34a", "达标", "能力面零硬失败：数学 / 代码经执行 / 约束推理 / 工具闭环 / 并行工具 / 结构化强绑定 / 视觉 / 长上下文 / 缓存 / 流式均成立"),
        ("R2 计费诚实", "#dc2626", "不达标（红线）", "usage 随 max_tokens 变化、与响应内容自相矛盾（G1）"),
        ("R3 身份/协议保真", "#16a34a", "达标", "Messages 面顶层字段零污染、无注入迹象、model 未降级（G6）"),
        ("M1 参数透传口径", "#d97706", "主体正常，3 处语义失控", "thinking 开关 / 预算、prefill 被静默吞掉（G2 / G4）"),
        ("M2 可用性", "#d97706", "受限", "按 max_tokens 预扣额度、余额近枯竭；count_tokens 未实现（G7）"),
    ]
    dim_html = "".join(
        f'<tr><td><b style="color:{c}">{n}</b></td><td><span class="pill" style="color:{c};background:#fff">{v}</span></td><td class="det">{d}</td></tr>'
        for n, c, v, d in dim_rows)

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Fable 5 中转通道测评报告（中转口径）</title><style>
 body{{margin:0;padding:32px 20px;background:#f8fafc;color:#0f172a;line-height:1.65;
  font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}
 .wrap{{max-width:1200px;margin:0 auto}} h1{{font-size:26px;margin:0 0 6px}}
 .sub{{color:#64748b;font-size:13px;margin-bottom:22px}}
 .verdict{{background:#fff;border:1px solid #e2e8f0;border-left:5px solid #dc2626;border-radius:12px;
  padding:16px 20px;margin-bottom:22px}} .verdict b{{color:#dc2626}}
 .kpis{{display:flex;gap:12px;flex-wrap:wrap;margin:0 0 26px}}
 .kpi{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:14px 20px;min-width:110px}}
 .kpi .v{{font-size:26px;font-weight:700}} .kpi .k{{font-size:12px;color:#64748b}}
 section,.card{{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:6px 18px 16px;margin-bottom:20px}}
 h2{{font-size:17px;display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}}
 .cnt{{font-size:12px;font-weight:400;color:#64748b}}
 table{{width:100%;border-collapse:collapse;font-size:13px}}
 th{{text-align:left;color:#64748b;font-weight:600;font-size:12px;padding:8px;border-bottom:1px solid #e2e8f0}}
 td{{padding:9px 8px;border-bottom:1px solid #f1f5f9;vertical-align:top}} tr:last-child td{{border-bottom:none}}
 .cid{{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#475569;white-space:nowrap}}
 .num{{text-align:right;color:#64748b}} .det{{color:#334155}} .dim{{color:#94a3b8;font-size:12px}}
 .pill{{padding:2px 9px;border-radius:999px;font-size:12px;white-space:nowrap;border:1px solid #e2e8f0}}
 code{{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}}
 .lead{{font-size:14px;color:#475569}}
</style></head><body><div class="wrap">
<h1>Fable 5 中转通道测评报告</h1>
<div class="sub">通道 <code>{summary['target']['base_url']}</code> · 模型 <code>{summary['target']['model']}</code> ·
{summary['generated_at']} · 独立直连，未经平台队列 · <b>判定基线：中转口径</b></div>
<div class="verdict"><div class="lead">中转通道的职能是<b>忠实转发 + 如实计费</b>，本报告据此判定：</div>
<b>R1 能力保真 达标</b>、<b>R3 身份/协议保真 达标</b>；但 <b>R2 计费诚实 不达标（红线）</b>——
<code>usage</code> 随 <code>max_tokens</code> 变化且与响应内容自相矛盾；
<b>M1 参数透传</b> 存在 3 处<b>语义失控</b>（<code>thinking</code> 开关/预算、<code>prefill</code> 被静默吞掉）。
<br>结论：<b>可作能力核验通道，不可作生产计费通道</b>；长输出场景当前不可用。
<div class="lead" style="margin-top:8px;color:#64748b">官方文档的 400 清单已<b>降级为口径盘点</b>，不作合格性要求。</div></div>
<div class="kpis">
 <div class="kpi"><div class="v">{len(summary['results'])}</div><div class="k">用例总数</div></div>
 <div class="kpi"><div class="v" style="color:#16a34a">{counts['passed']}</div><div class="k">通过</div></div>
 <div class="kpi"><div class="v" style="color:#dc2626">{hard}</div><div class="k">失败</div></div>
 <div class="kpi"><div class="v" style="color:#d97706">{counts['warning']}</div><div class="k">警告（多已消解）</div></div>
 <div class="kpi"><div class="v" style="color:#6b7280">{counts['blocked']}</div><div class="k">未验证</div></div>
 <div class="kpi"><div class="v" style="color:#dc2626">1</div><div class="k">红线未过（R2）</div></div>
</div>
<div class="card"><h2>总判定：中转五维</h2>
<table><thead><tr><th>维度</th><th>结论</th><th>依据</th></tr></thead><tbody>{dim_html}</tbody></table></div>
<div class="card"><h2>关键发现</h2><ol>
<li><b>R2 红线</b>｜同一提示仅改 <code>max_tokens</code>（64→64、100→100），<code>output_tokens</code> 随上限走；批次内 38/72 条 <code>output_tokens == max_tokens</code>；<code>max_tokens=100</code> 却输出 170 字符正文。→ <b>usage 非实测，计费不可信</b>。</li>
<li><b>M1 语义失控</b>｜<code>thinking=disabled</code> 被接受但<b>响应仍带 thinking 块</b>（71/72 条如此）。→ 关闭思考的意图静默失效。</li>
<li><b>M1 语义失控</b>｜末尾 assistant 消息（prefill）未被沿用，响应非续写。→ 锁定输出起点的意图静默失效。</li>
<li><b>M1 非缺陷</b>｜<code>temperature/top_p/top_k</code> 被丢弃、未知字段被忽略、缺 <code>anthropic-version</code> 被补默认——中转从宽，均无实际影响。</li>
<li><b>R1 达标</b>｜基础 15/15；代码经实际执行验证；工具闭环 + 并行工具；结构化强绑定；视觉（401 vs 51 tokens）；长上下文 27406；缓存命中 5135；<code>stop_sequences</code> 精确截断。</li>
<li><b>R3 达标</b>｜Messages 面顶层字段干净、无注入、<code>model</code> 未降级；OpenAI 兼容面为转译（正常）。</li>
<li><b>M2 受限</b>｜403 预扣 <code>need quota = max_tokens × $50/M</code>，余额已从 $0.913 耗至约 $0.012；<code>count_tokens</code> 端点 404。</li>
<li><b>更正</b>｜原判失败的 <code>stop_sequences</code> / <code>system</code> 经复核为判据缺陷；E03/E04 原按官方口径判失败，中转口径下改判「非缺陷 + 语义失控风险」。</li>
</ol></div>
{"".join(secs)}
</div></body></html>"""
    (run_dir / "report-relay.html").write_text(html, encoding="utf-8")
    print(f"已生成：{run_dir/'REPORT-relay.md'}、{run_dir/'report-relay.html'}")
    print(f"复核后：通过 {counts['passed']} / 失败 {hard} / 警告 {counts['warning']} / 未验证 {counts['blocked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
