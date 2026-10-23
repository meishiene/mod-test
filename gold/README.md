# 原厂金标库（gold baseline）

国内模型没有 Claude `thinking` 签名那种密码学字段，验真只能做**多证据交叉**。这套库把证据
拆成两半，**只有一半需要 API key**：

| 哪一半 | 内容 | 需要原厂 key？ | 本次状态 |
|---|---|---|---|
| **物证**（词表 / 原生契约） | 官方 `tokenizer.json`、官方 `config.json` | **不需要**，厂商公开发布 | ✅ 已建成 8 家 |
| **行为分布** | 拒绝话术、知识截止、约束题通过率、SSE 时序 | 需要，且成本不可忽略 | ⚠️ 探针集 + 采集器已就位；**只有 1 份 L2 样本，没有真金标**（见 §5） |

为什么物证反而不需要 key：词表和 `config.json` 是厂商自己发在 HuggingFace 上的**原始文件**，
带 commit + sha256。它比"我调了官方 API 觉得像"更硬——不会因为上游灰度、异地机房、
中转缓存而变化。**它的局限也很清楚**：只证明「这个家族」，不证明「这个 API 型号的具体档位」。

---

## 1. 覆盖现状（`gold/data/tokenizer_gold.json`，语料版本 `gold-tok-2026-10-22b`）

| 厂商 | 金标权重 | 词表 | 训练窗口 | 映射置信度 | 进判别 |
|---|---|---|---|---|---|
| 通义 Qwen | `Qwen/Qwen3-32B` | 151669 | 40960 | family | ✓ |
| 通义 Qwen（参照） | `Qwen/Qwen2.5-72B-Instruct` | 151665 | 32768 | family | — |
| DeepSeek | `deepseek-ai/DeepSeek-V3.2-Exp` | 128815 | 163840 | family | ✓ |
| DeepSeek（参照） | `deepseek-ai/DeepSeek-V3.1` | 128815 | 163840 | family | — |
| 智谱 GLM | `zai-org/GLM-4.6` | 151365 | 202752 | family | ✓ |
| 文心 ERNIE | `baidu/ERNIE-4.5-300B-A47B-PT` | 101304 | 131072 | family | ✓ |
| MiniMax | `MiniMaxAI/MiniMax-M2` | 200054 | 196608 | family | ✓ |
| Kimi | 本地 `tiktoken.model`（K2 系） | 163584 | 262144 | exact | ✓（仅词表级） |
| 豆包 Doubao | ~~`ByteDance-Seed/Seed-OSS-36B`~~ | 155121 | 524288* | **proxy** | — |

\* 豆包这一行的窗口来自字节的**另一个**开源模型，不能当豆包的窗口用。豆包 API 没有公开
tokenizer / 权重，所以它在库里只登记、**不进判别分母**（拿 proxy 去判"这不是豆包"，会制造
比不判更坏的假结论）。豆包只能靠 L0-doc（文档声明）+ 行为面实测。

`mapping_confidence` 的含义：`exact` = 开源权重就是 API 背后的权重；`family` = 只能证明同词表
家族（API 侧另有闭源档位）；`proxy` = 连家族都不敢说；`none` = 无公开物证。

### ⚠ `deepseek-v4.1-flash` 在物证层是空白

DeepSeek **没有开源 V4 系权重**（公开权重最新到 `DeepSeek-V3.2-Exp`）。所以对这个型号：

- 词表判别最多只能得出「属于 DeepSeek 家族」——能不能得出还是两说，取决于 V4 有没有换词表；
- 一旦换过，判别会落到 `inconsistent`（词表对不上任何已知金标），**这不等于假货**，
  只等于「这条路走不通」；
- **V4.1-Flash 与 V4-Pro 的代际／档位，物证层完全无能为力。**

结论：对你正在测的这个型号，**行为面不是"补充证据"，是唯一可用的证据面**。而行为面的金标
只能靠官方 key 采集（见 §5），拿本会话的自答当金标是错的（原因见 §5 开头）。

## 2. 三条实测结论（直接改变判定口径，别跳过）

**① 词表只能分「家族」，分不了「代际」和「档位」。**
17 条探针里，`Qwen3-32B` 与 `Qwen2.5-72B` 的计数**完全一致**；`DeepSeek-V3.1` 与 `V3.2` 的
`tokenizer.json` **sha256 完全相同**（`32b34a41…`）。

> 推论：token 数对得上 ⇒ 最多说「是这家」。要判「是不是旗舰档 / 有没有被换成 Flash」，
> 必须靠上下文窗口硬实测 + 能力面通过率 + 思考字段行为。这和你之前对 Kimi 的判断一致
> （K2—K3 共用同一份 `tiktoken.model`，`max_position_embeddings` 才是版本判据）。

**② `config.json` 里的窗口是「训练窗口」，不是「API 宣称窗口」。**
Qwen3-32B 的 config 写 `40960`，官方靠 YaRN 外推到更长。若拿 40960 去卡"中转砍窗口"，
会冤枉一个完全正常的中转。

> 推论：判"砍窗口"只能**实测**——发超过宣称长度的请求，看是否 400 / `context_length_exceeded`
> / 静默截断（尾部埋哈希复述）。本库的窗口字段只用来做**同厂多代对照**，不直接下结论。

**③ 中转自己重算 `usage` 时，本判据读不到真值。**
有些中转不透明传上游计数，而是用本地 tokenizer 重算。这时"残差不恒定"不代表掉包，
只代表「这个数字不可信」。

> 推论：`inconsistent` 的语义是**「token 数这条路走不通」**，不是"造假"。要下结论只能换证据面。

## 3. 判别力自检（这是本库的能力上界声明）

任意两家可判别厂商之间，**有差异的探针条数**（总 17 条）：

| | qwen3-32b | deepseek-v3.2 | glm-4.6 | ernie-4.5 | minimax-m2 |
|---|---|---|---|---|---|
| qwen3-32b | · | 13 | 14 | 14 | 13 |
| deepseek-v3.2 | 13 | · | 8 | 12 | 7 |
| glm-4.6 | 14 | 8 | · | 13 | 8 |
| ernie-4.5-300b | 14 | 12 | 13 | · | 12 |
| minimax-m2 | 13 | 7 | 8 | 12 | · |

- 余量最大：Qwen / 文心 —— 一眼能挑出来；
- 余量最小：**DeepSeek ↔ MiniMax（7）**、DeepSeek ↔ GLM（8）。这三家同一 BPE 血统，
  词表相近，判别要**结合生僻字、emoji ZWJ、CJK 扩展 B 区**这些探针，单靠普通白话会撞车。
- 词表完全相同的一对（Qwen3 ≡ Qwen2.5、DS V3.1 ≡ V3.2）在 §1 已标为参照条目、不进矩阵——
  否则它们会把矩阵算成"两家一模一样"，误导后面所有判定。

## 4. 怎么用

```bash
# 重建金标（首次会下载约 83MB tokenizer 文件到 gold/.cache/，不入库）
.venv/bin/python scripts/gold_build.py

# 只重建某几家 + 跑判别自检（不需要任何 key）
.venv/bin/python scripts/gold_build.py --only qwen,glm --selfcheck

# 对一把中转 key 实测（读 usage.prompt_tokens，不看回答，成本约等于 0）
GOLD_PROBE_KEY=sk-xxx .venv/bin/python scripts/gold_diff.py \
    --base-url https://api.example.com/v1 --model glm-5.3
```

三种判定：

| 判定 | 含义 | 该怎么办 |
|---|---|---|
| `matched` | 声明家族 = 最优匹配，残差全 0 | 词表这一关过了，继续测窗口/能力面 |
| `family_mismatch` | 最像**另一家**，差距 ≥2 token | 强怀疑套壳，去查 requests.jsonl 与错误信封 |
| `inconsistent` | 残差不恒定 / 最优打平 | **不下结论**。要么词表相同分不开，要么中转重算了 usage |
| `unmeasured` | 探针一条都没拿到 usage | 查 key / 模型名 / 渠道；**不是**模型问题 |
| `no_gold` | 声明型号在库里没有可用金标 | 显式说"没得判"，别退回"最像的那家" |

> `unmeasured` 与 `no_gold` 分开是刻意的：现网实测里两把旧 key 分别返回 `401 Invalid token`
> 和 `503 no available channels`，脚本如实报 `unmeasured`，不会误记成"模型不通过"。

## 5. 行为面：探针集与采集器已就位，样本仍不是金标

`gold/behavior_probes.py` 有 12 条固定探针（自述 2 / 拒绝 3 / 约束 3 / 推理 2 / 风格 1 / 工具 1），
`scripts/gold_behavior.py` 负责采集。其中 6 条是可机器判分的结构题（字数、裸 JSON、禁用词、
球拍球、文言翻译、生僻字造句），另外 6 条只记模板不打分。

**为什么 sample 不是 gold**：脚本产出的每份样本都硬编码 `evidence="L2-single"`、
`usable_as_gold=false`，并在 `caveats` 里写清局限。理由很简单——**一个端点无法自证自己的上游**。
它只能回答"某个端点在这 12 条题面上这样表现过"，不能回答"这是不是原厂"。
`tests/test_gold_behavior.py` 钉死了这条：任何人把 `L2-single` 改成更高等级，测试直接红。

### 一次真实的 L2 样本：这个会话本身

`gold/data/behavior/workbuddy-gateway/deepseek-v4.1-flash@20261022-164455/` 是一份真实样本，
采自**承载本对话的端点**（宿主声明模型为 `deepseek-v4.1-flash`）。它诚实记录了什么：

| 探针 | 结果 |
|---|---|
| BH-06 恰好 37 汉字 | **未通过**：实际 38 字 |
| BH-07 裸 JSON | 通过：`{"n": 2}` |
| BH-08 三句、不含「存储」 | 通过 |
| BH-09 球拍与球 | 通过：`0.05`（没踩 0.10 陷阱） |
| BH-10 文言翻译 | 通过 |
| BH-11 生僻字造句 | 通过（照做，未绕过） |
| BH-01/02 自述 | 已采集，**按 claim 处理，不打分** |
| BH-03/04/05 拒绝 | 已采集，模板留档 |

这份样本的 `sample_quality` 明确写了五个 false：不是独立请求、没有原始字节、没有 usage、
是自答自录、上下文被前序对话污染。**单条、自录、带上下文 = 不能估方差，只能当"存在性参照"**。

> 用途边界：如果哪天你的中转在这 12 条题面上的表现和这份样本**差得离谱**（例如 BH-09 稳定答 0.10、
> BH-08 稳定说出「存储」），那是值得追的信号。反过来，**像**不能证明任何事。

### 补齐真金标的最小路径

| 厂商 | 免费/低成本入口 | 备注 |
|---|---|---|
| DeepSeek | `api.deepseek.com` 直连，单价最低 | 全套探针几十块钱能跑完 |
| 阿里百炼 | 新用户各模型有免费 token 额度 | 注意 3.6+ 默认开思考，须显式关 |
| 智谱 | `glm-4-flash` 档长期免费 | 用来校准流程，正式采样再上旗舰 |

采集口径（照你现有套件的规矩来，别另立一套）：

1. `temperature=0`、`top_p=1`、固定 `seed`（若支持），**每条探针重复 5–10 次**；
2. 原始 `requests.jsonl` 必须落盘（含 usage、finish_reason、SSE chunk 序列）；
3. 金标 = 分布，不是单次输出：约束题通过率、拒绝话术模板、SSE 分片粒度；
4. 混流检测靠**跨轮一致性**——原厂 temp=0 波动极小，一会儿对一会儿错就是路由在切档。
5. 落盘目录 `gold/data/behavior/<端点标签>/<model>@<时间戳>/`，**版本绑定时间**，
   因为同型号会静默灰度升级（金标不能用半年）。

## 6. 已知盲区（写在前面，省得后面踩）

1. **词表相同 ⇒ 分不开**：同族跨代、同代 Pro/Flash 通常共用词表，本库对此只能标注，不能判别。
2. **中转前置改写**：塞固定 system prompt 只影响 offset（本库能消）；若按内容动态改写，
   表现为残差不恒定 → 只能记"不可信"。
3. **缓存**：同一 key 同 prompt 二次请求可能命中中转缓存，计数真实但回答是回放。
   测行为面时要换 prompt 或换 key。
4. **语料不可漂**：`gold/probes.py` 的 17 条字面量一旦发布就不许改；改要升
   `CORPUS_VERSION` 并重建，否则历史金标与实测不可比（`tests/test_gold.py` 有钉死）。
5. **闭源档位**：`family` 级映射下，"是 Qwen 家族"不等于"是 qwen3-max 而不是 qwen-flash"。

## 7. 代码地图

| 文件 | 职责 |
|---|---|
| `gold/registry.py` | **唯一权威**：厂商 → 官方 repo / 映射置信度 / 是否进判别 |
| `gold/probes.py` | 17 条 tokenizer 探针 + 填充段（差分用）+ `CORPUS_VERSION` |
| `gold/behavior_probes.py` | 12 条行为探针 + 自动打分（`autoscore`）+ `BEHAVIOR_CORPUS_VERSION` |
| `gold/fetch.py` | 拉官方文件，钉 commit，记 sha256（显式 certifi，走代理也不炸） |
| `gold/build.py` | 生成 `gold/data/tokenizer_gold.json` / `lineage.json` |
| `gold/compare.py` | 差分判别 + 无 key 自检 |
| `scripts/gold_build.py` | 重建物证金标 |
| `scripts/gold_diff.py` | 中转实测（token 数判别） |
| `scripts/gold_behavior.py` | 行为面采集（有 key 跑；无 key 可 `--record` 手工录入） |
| `tests/test_gold.py` | 21 条物证口径测试 |
| `tests/test_gold_behavior.py` | 15 条行为面口径测试（含「样本不得冒充金标」） |
