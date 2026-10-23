# ml-test-platform

面向多供应商大模型验收的端到端测试平台。一套 Web UI + 一组领域测试引擎，把「性能容量、KV 缓存、协议契约、能力专项、官方套件、内容安全、开源基准」放在同一条流水线上跑，并把请求级原始证据与统一准入判定一起落盘。

不是单点压测脚本集合 —— 每个测试项都必须产出可审计的执行证据，并由校验器（而不是 HTTP 200）给出通过/失败结论。

---

## 目录

- [能测什么](#能测什么)
- [快速开始](#快速开始)
- [配置](#配置)
- [测试项总览](#测试项总览)
- [命令行直接运行](#命令行直接运行)
- [结果产物与判定契约](#结果产物与判定契约)
- [架构](#架构)
- [开发约定](#开发约定)
- [已知问题](#已知问题)

---

## 能测什么

六类文本模型新增「经济验证」：每模型最多11次请求、单次输出上限2048 tokens，复用响应验型号；Kimi不包含408项Schema套件。完整题集、预算与使用方法见 [ECONOMY_VALIDATION.md](ECONOMY_VALIDATION.md)。本题集仅提供有限范围证据，不生成旧制120分，也未切换尚待确认的v3评分方案。

| 维度 | 说明 |
|---|---|
| 性能与容量 | TTFT / TTFC / TPOT / E2E 分位、成功率、截断率、开环容量窗口（RPM / TPM）、在途积压斜率 |
| KV 缓存 | 缓存命中率、缓存 token 统计、公共前缀配方下的命中稳定性 |
| 协议契约 | 身份路由、非法参数拒绝、SSE 完整性、静态工具闭环、thinking / reasoning_effort 对照 |
| 模型能力 | 思考回传、工具调用、JSON 输出、Session、Prefix / FIM、Anthropic Messages、Responses |
| 供应商验真 | 官方 KVV 套件（K3 核心 / 动态工具 / Thinking Effort / Prompt Token / params / tool JSON Schema）、DeepSeek 满血指纹 |
| 内容安全 | 上下文劫持、关键词变体绕过、分类一致率、高风险漏判率、误报率、结论翻转率 |
| 图像生成 | 并发压测、4K / n=10 token 核验（PIL 解码 + 字节/token 上限规则） |
| 视频生成 | Seedance 异步任务：提交 → 轮询任务状态 → 归档视频与下载链接；提示词 / 分辨率 / 画面比例 / 时长 / 首帧·参考图 / 参考视频（地址提交、本地仅归档）/ 原生音频 / 水印 |
| 开源基准 | ArXivMath、DeepSWE、SWE-Atlas TW/RF、Legal Agent Benchmark |

支持的协议族：`openai`、`openai_responses`、`openai_images`、`anthropic`、`gemini`、`volcengine`、`seedance`。

`seedance` 是唯一的**异步任务面**协议（`/api/v3/contents/generations/tasks`，提交拿 task id 后轮询），与其余同步/流式协议族不共用请求体形状，也不接受文字聊天测试。

---

## 快速开始

### 环境要求

- Python 3.12+（开发环境使用 3.14）
- 能访问目标模型网关

### 启动 Web 平台

```bash
bash start.sh
# 指定监听地址与端口
bash start.sh --host 127.0.0.1 --port 8085
```

`start.sh` 会自动完成：创建 `.venv` → 按 `requirements.txt` 补齐依赖 → 加载 `.env` → 检测并释放被占用的端口 → 启动 Web 服务。

### Windows 一键启动

双击项目根目录的 `start.bat`，或在 `cmd` 中执行：

```bat
start.bat
start.bat --host 127.0.0.1 --port 8085
```

`start.bat` 会自动完成：定位 64 位 Python 3.12+（找不到时尝试用 `winget` 安装 Python 3.12）→ 创建 `.venv` → 按 `requirements.txt` 安装/补齐依赖 → 加载 `.env` → 释放被占用端口 → 启动 Web 服务。依赖安装结果记录在 `.venv\.mltp-deps.json`，`requirements.txt` 未变化时后续启动会跳过安装。

- 首次启动需要联网下载依赖（其中 C2PA 的 Windows wheel 约 80MB）。网络受限时可用 `PIP_INDEX_URL` 指定镜像；官方 PyPI 安装失败时脚本会自动改用清华镜像重试。
- 若机器禁用 `winget`，请先手动安装 64 位 Python 3.12+，安装时勾选 “Add python.exe to PATH”，再运行 `start.bat`。
- 可选环境变量：`PYTHON_BIN`（指定解释器）、`VENV_DIR`（虚拟环境目录）、`MLTP_AUTO_INSTALL_PYTHON=0`（禁用自动安装 Python）、`MLTP_NO_MIRROR=1`（禁用镜像回退）。
- 只准备环境或只做检查：`start.bat --bootstrap-only` / `start.bat --check`。

启动后打开 `http://127.0.0.1:8085`，页面分四块：接口配置、选择测试项、测试记录、报告弹窗。

选择测试时，可点击左侧“常用组合”一次选好一组，再点整行加入或移出单项；“详情”仅查看配置，“清空”取消当前选择。组合会替换当前选择，只有点击“启动测试”才会执行。清空后刷新仍保留空选状态。

GLM 组合：基础验收 = 核心能力 + 真实性与路由校验；Agent 专项 = 核心能力 + Agent 能力；缓存专项 = 核心能力 + 缓存与 Session；综合验收 = 综合能力与满血证据校验 + 缓存与 Session。其他类别按各自用例配置，GPT / Claude 随当前档位切换；组合按钮显示项数，悬停可查看成员。协议不支持或需启用通道面时，整组禁用；单独运行的压测和 Kimi 临时案例集不会与其他项混选。组合定义集中在 `web/static/case-presets.js`。

> 默认监听 `0.0.0.0`。**平台当前没有入站鉴权**，请勿在不可信网络上直接暴露（见[已知问题](#已知问题)）。

### 跑单元测试

```bash
.venv/bin/python -m pytest tests/ -q
```

---

## 配置

复制 `.env.example` 为 `.env`，按需填写。平台按协议把凭据注入到子进程，**API key 不写入数据库**。

| 变量 | 用途 |
|---|---|
| `VOLC_API_KEY` / `VOLC_BASE_URL` / `VOLC_MODEL` | 主凭据，KVV、性能压测、OpenAI 兼容协议 |
| `VOLC_PROTOCOL` | `openai` / `openai_responses` / `openai_images` / `anthropic` / `gemini` / `volcengine` |
| `VOLC_CHAT_PATH` | chat 补全路径，默认 `/chat/completions` |
| `VOLC_REQUEST_TIMEOUT_SECONDS` / `VOLC_CONNECT_TIMEOUT_SECONDS` | 单请求总超时 / 连接超时 |
| `VOLC_MAX_CONNECTIONS` | 连接池上限 |
| `VOLC_VERIFY_SSL` / `VOLC_TRUST_ENV` | TLS 校验 / 是否信任系统代理环境变量 |
| `VOLC_STREAM_INCLUDE_USAGE` | 是否要求流式响应返回 usage；正式测试应为 `true` |
| `DS_*` / `GLM_*` / `GPT_IMAGE_*` | 各专项套件的独立凭据 |
| `GPT_IMAGE_REQUEST_TIMEOUT_SECONDS` / `GPT_IMAGE_4K_TIMEOUT_SECONDS` | 图像单请求 / 4K 验真超时 |
| `KVV_HOME` / `KVV_PYTHON` / `KVV_THINK_MODE` | 官方 KVV 套件目录、解释器、thinking 模式 |

`.env` 已加入 `.gitignore`，请勿提交。

### 被测网关在境外时：需要显式配置代理

平台默认**直连**目标地址（`VOLC_TRUST_ENV` 默认 `false`，即忽略环境里的代理变量）。若被测网关位于境外、本机需要经本地代理（如系统代理 `127.0.0.1:7897`）才能访问，直连会在 60 秒后报 `ConnectTimeout: timed out`，前置连通性检查失败、正式测试不会启动——**这不是被测模型的问题**，报告里的 `diagnosis` 字段会直接说明这一点。

在 `.env` 中显式指定代理并允许平台读取：

```bash
HTTP_PROXY=http://127.0.0.1:7897
HTTPS_PROXY=http://127.0.0.1:7897
http_proxy=http://127.0.0.1:7897
https_proxy=http://127.0.0.1:7897
VOLC_TRUST_ENV=true
```

大小写两种写法都要写：httpx 读取环境代理时**小写优先**，只写大写会被环境里残留的小写变量覆盖。改完需重启服务（`start.sh` 只在启动时加载 `.env`）。代理客户端的「TUN / 增强模式」是另一条路——它让直连也被透明代理，无需动 `.env`。

---

## 测试项总览

UI 共 14 个分类 tab、90 张用例卡片。

### 性能压测（`load`）

| 用例 | 内容 |
|---|---|
| `T0` | 冒烟，10 次请求且要求恰好 10 次成功 |
| `RANDOM_LOAD` | 随机输入 1,000–20,000 token，成功率 ≥99% |
| `B1` | 多规格基线：2K / 8K / 16K / 32K 各 4 次，产出 L1–L4 的相对退化基线 |
| `L1`–`L5` | 分档负载，并发 50→8，长上下文 |
| `O1` | 定长可见输出，要求 95% 以上落在 400–700 token |
| `C0` / `C1` | 开环容量爬坡，`C1` 目标 600 RPM / 6,000,000 TPM |
| `SIMPLE_LOAD` / `FULL_LOAD` | 预设组合（固定参数，UI 不放开自定义） |

L 档验收阈值（`TTFC P50 / P90`、`TPOT P90`）：

| 用例 | TTFC P50 | TTFC P90 | TPOT P90 |
|---|---:|---:|---:|
| L1 | ≤5s | ≤15s | ≤60 ms/token |
| L2 | ≤7s | ≤20s | ≤65 ms/token |
| L3 | ≤10s | ≤28s | ≤70 ms/token |
| L4 | ≤15s | ≤40s | ≤80 ms/token |
| L5 | ≤22s | ≤60s | ≤90 ms/token |

L1–L4 另需 `TTFC P90 ≤ 同档 B1 基线的 3 倍`，因此**必须先有 B1 结果**并带 `--baseline-summary` 才能形成准入结论。所有性能档位还要求：成功率 ≥99%、usage 完整率 100%、输入 token 落在目标 ±5%、截断率 ≤5%。

### 其余分类

| Tab | 用例 | 内容 |
|---|---|---|
| `kvv` | `KVV_K3_FEATURES`(46) / `KVV_K3_DYNAMIC_TOOLS`(54) / `KVV_K3_THINKING_EFFORT`(26) / `KVV_PROMPT_TOKENS`(55 请求) / `KVV_PARAMS`(16·K3 契约) / `KVV_TOOL_SCHEMA`(408) + `K3_TEMP_SUITE` | 官方 Kimi Vendor Verifier pytest 直接封装（保留参数契约与 response_format 兼容补丁，详见 `vendor_verifier/LOCAL_CHANGES.md`）+ K3 供应商临时案例集；详见下方「Kimi 测试」 |
| `deepseek` | `DS_V4_P0` / `P1` / `P2` / `DS_V4_FINGERPRINT` | 能力专项与满血指纹 |
| `glm` | `GLM_53_CORE` / `AGENT` / `CACHE` / `FINGERPRINT` | 含缓存命中率 ≥70% 门槛 |
| `gpt-image` | `GPT_IMAGE_2` / `GPT_IMAGE_25` / `GPT_IMAGE_4K_TOKEN_VERIFY` | 图像并发压测 + 4K token 验真 |
| `seedance` | `SEEDANCE_VIDEO_GEN` | Seedance 异步视频生成：提交任务 → 轮询到终态 → 归档视频与链接；详见下方「视频生成」与 `docs/Seedance视频生成测试.md` |
| `open-source` | `ARXIVMATH` / `DEEPSWE` / `SWE_ATLAS_TW` / `SWE_ATLAS_RF` / `LEGAL_AGENT_BENCHMARK` | 5 项开源基准，CLI runner 已实现；Web 侧脚本分发仍有未定义常量，见已知问题 |
| `moderation` | `MOD_GENERAL_COMMON` / `MOD_PARTNER_COMMON` / `MOD_PARTNER_SMOKE` | SillyTavern 上下文劫持 + 关键词变体，含不变性分组 |
| `authenticity` | `DS_AUTH_FINGERPRINT` / `DS_V4_FINGERPRINT` | 6 探针：动态 required 工具、随机 JSON Schema、多轮记忆、reasoning 字段、SSE 完整性、静默吞参对照 |

`kvv` / `deepseek` / `glm` 三个 tab 下还各挂一个公共契约探针（`KVV_CONTRACT` / `DS_CONTRACT` / `GLM_CONTRACT`），每个 5 个探针。

### 「Kimi 测试」模块（`test_kind=kvv`）

模块内含两条执行链，判定口径与协议面不同，Web 端按 case id 分流、不允许混批：

| 入口 | 执行链 | 协议面 | 说明 |
|---|---|---|---|
| `KVV_K3_FEATURES` / `KVV_K3_DYNAMIC_TOOLS` / `KVV_K3_THINKING_EFFORT` / `KVV_PROMPT_TOKENS` / `KVV_PARAMS` / `KVV_TOOL_SCHEMA` | `scripts/run_kvv.py`（官方 pytest） | openai / volcengine | 直接跑 `vendor_verifier/` 官方用例，按 pytest 退出码与 failures/errors 判定。**参数套件带契约档位**：默认按 K3 契约定期望（`temperature` 固定 1.0），`KVV_PARAM_CONTRACT=k2` 回到 K2.5 档；档位写进 `summary.json` 与报告 |
| `K3_TEMP_SUITE`（24 个合并案例 / 68 条原子请求） | `scripts/run_kvv_temp.py` → `kvv/temp_runner.py` | openai / volcengine | 供应商自测基线清单驱动；同 `category` 且同 `request.batch` 的重复请求合并为一个逻辑案例，执行时逐条发起、分别留证 |

清单与解析器都在 `kvv/` 包内（`kvv/kimi_k3_temp_cases.json` + `kvv/temp_suite.py`）；`k3` 分组的选择项在 UI 上是该卡片下的下拉框（「全部 K3 合并案例」= 整组）。「中转计费探针」是另一条独立的清单案例组，挂在「Claude 测试」模块下（`scripts/run_relay_billing_probe.py`），详见 `docs/中转计费临时用例设计.md`。

### 「视频生成」模块（`test_kind=seedance`）

唯一用例 `SEEDANCE_VIDEO_GEN`，走火山方舟 **异步任务面**（协议 `seedance`，根路径 `/api/v3`，默认域名 `https://token-hub.tangrankeji.com`）。这是全平台唯一一条「先提交、再轮询」的链路，因此判定与产物形状都和同步套件不同：

| 参数 | 取值域 | 说明 |
|---|---|---|
| 提示词 `seedance_prompt` | 必填，≤2000 字符 | 逐字发送，平台不追加任何系统提示词 |
| 生成方式 `seedance_mode` | `text` 仅文生视频 / `image` 带参考图 / `video` 带参考视频 / `both` 两者都带（默认 `text`） | 界面先选档：只展示该方式用得到的参数与上传区；入队时按模式**强校验**（声明文生视频却带着图片 ⇒ 400，声明带视频却没填地址 ⇒ 400）。它决定 `content` 里出现哪些块 |
| 分辨率 `seedance_resolution` | `720p` / `1080p` / `4k`（默认 `720p`） | `480p` 只属 1.0 系列，2.0/2.5 的可选集里没有它——档位写错会被上游以「任务级参数校验失败」拒绝 |
| 画面比例 `seedance_ratio` | `16:9` / `4:3` / `1:1` / `3:4` / `9:16` / `21:9` / `adaptive`（默认 `adaptive`） | |
| 时长 `seedance_duration` | 4–15 秒（2.0）/ 4–30 秒（2.5），整数（默认 5） | 区间按**模型档位**解析：入队时收紧，档位识别不出来时退化成并集 4–30 |
| 上传图片 `seedance_image_path` | 本服务上传目录内的文件 | 走 `POST /api/uploads` 原始二进制通道（JSON 通道上限 64 KiB，base64 必然超限），≤8 MiB；仅生成方式为 `image` / `both` 时提交 |
| 图片地址 `seedance_image_url` | `http(s)://` 或 `data:` URI | 与上传二选一；两者都填以上传为准 |
| 图片位置 `seedance_image_role` | `first_frame` / `reference_image`（默认首帧） | **只在生成方式带图、且确实带了图片时才写进请求体**；单图合法用法只有首帧与参考图，首尾帧需两张图、当前输入面不支持 |
| 上传参考视频 `seedance_video_path` | 上传目录内 `mp4`/`webm`/`mov`/`mkv`/`m4v`，≤64 MiB | 走同一条上传通道；**只归档留痕（复制进 `inputs/` 并记 sha256），不提交给上游**——该接口只接受 URL |
| 视频地址 `seedance_video_url` | `http(s)://` 或 `data:` URI | **唯一会真正生效的参考视频输入**；与本地文件同时填写时以地址为准。生成方式为 `video` / `both` 时**必须**填它（只上传本地文件不算） |
| 视频位置 `seedance_video_role` | `reference_video` / 空串（= 不带 `role` 字段），默认 `reference_video` | 空串给"上游不认这个 role"留退路；只在生成方式带视频时随视频块出现 |
| 生成音频 / 水印 | `seedance_audio` / `seedance_watermark`，官方默认**音频开、水印关** | 两个开关都以布尔值显式提交 |
| 视频任务数 / 并发 | 1–100 / 1–16，**手动输入** | 单条视频成本极高，不做随机抽取 |

判定口径：提交拿到 task id 且轮询到 `succeeded` 并给出视频链接 ⇒ `passed`；2xx 却无 task id、任务 `failed`、`succeeded` 但无视频链接 ⇒ `failed`；账号 / 权限 / 额度 / 限流 / 瞬时故障 / 观察窗口内未达终态 ⇒ `blocked`（只记录、不阻断准入，但**整批只有 blocked** 时视为无有效结论、不准入）。带参考视频的请求被 4xx 拒绝且错误文案点到 `video`/`视频` ⇒ `blocked`（该模型/网关可能未开放 `video_url` 面，属平台能力缺失，不是模型缺陷）。报告（`report.md`）有独立的「视频链接」小节直接给出可复制的视频地址，结果目录另有 `videos/` 归档、`inputs/` 素材归档与 `requests.jsonl` 里的逐次轮询时间线。**失败取证**：任务 `failed` 时 `verdict_detail` 会带上 `error.code`，`requests.jsonl` 另留 `task_error_code` / `task_body_keys`（上游响应顶层键名，直接回答「上游回显了哪些参数」）/ `task_body_sample`（末次响应体原文，截断 2000 字符），报告只在有未通过用例时多出「未通过用例的上游原文」一节——异步面把参数校验放在执行阶段，笼统的「请求参数校验失败」只靠这些原文才能定位到具体参数。

---

## 命令行直接运行

不走 Web 时可以直接调用，所有 runner 都支持 `--output-root`。

```bash
# 性能压测（闭环 / 开环）
.venv/bin/python scripts/run_test.py T0
.venv/bin/python scripts/run_test.py L1 --samples 100 --concurrency 30
.venv/bin/python scripts/run_test.py L2 --baseline-summary results/B1-20260921-120000/summary.json
.venv/bin/python scripts/run_test.py T0 --quick          # 仅验证脚本，不构成准入结论

# 整套推荐节奏（消耗约 3000 万–3500 万 token，必须显式确认成本）
.venv/bin/python scripts/run_suite.py --confirm-cost [--include-l5] [--stop-on-failure]

# 官方 KVV 套件
.venv/bin/python scripts/run_kvv.py KVV_K3_FEATURES
.venv/bin/python scripts/run_kvv.py KVV_K3_DYNAMIC_TOOLS
.venv/bin/python scripts/run_kvv.py KVV_K3_THINKING_EFFORT
.venv/bin/python scripts/run_kvv.py KVV_PROMPT_TOKENS

# 能力专项 / 契约探针
.venv/bin/python scripts/run_dsv4.py DS_V4_P0
.venv/bin/python scripts/run_glm53.py GLM_53_CORE
.venv/bin/python scripts/run_contracts.py GLM_CONTRACT

# 图像 / 指纹 / 安全
.venv/bin/python scripts/run_gpt_image.py GPT_IMAGE_2 --requests 16 --concurrency 4
.venv/bin/python scripts/run_authenticity.py DS_V4_FINGERPRINT
.venv/bin/python scripts/run_moderation.py MOD_GENERAL_COMMON

# Seedance 视频生成（异步任务：提交 → 轮询 → 归档视频与链接）
SEEDANCE_API_KEY=sk-x SEEDANCE_BASE_URL=https://token-hub.tangrankeji.com \
SEEDANCE_MODEL=doubao-seedance-1-0-pro-250528 \
SEEDANCE_PROMPT='镜头缓慢推进，一只白鹭在浅滩展翅起飞' \
.venv/bin/python scripts/run_seedance.py --requests 1 --concurrency 1 \
  --resolution 1080p --ratio 16:9 --duration 5 --image-role first_frame \
  --image-path web/data/uploads/<file>.png --generate-audio --watermark
# 参考视频：地址会随任务提交；本地文件（--video-path）只归档留痕
.venv/bin/python scripts/run_seedance.py --video-url https://cdn.example/ref.mp4 \
  --video-role reference_video --video-path web/data/uploads/<clip>.mp4

# 清单驱动的 K3 临时案例集（同类项已合并：24 个逻辑案例 / 68 条原子请求）
.venv/bin/python scripts/run_kvv_temp.py --output-root results
VOLC_TEMP_CASE=16K-10 .venv/bin/python scripts/run_kvv_temp.py --output-root /tmp/probe   # 只跑「16K 输入 TTFT」这一个合并案例（10 次请求）
```

`scripts/run_kvv_temp.py` 支持 `--output-root`（默认 `results/`）。用 `VOLC_TEMP_CASE` 指定单个合并案例：取其 `batch` 名（如 `16K-10`、`TPOT-10`、`CACHE-HIT-7`），也兼容原子案例 id（如 `K3-TTFT-12` → 展开为 `64K-10`）。合并只改变案例集与汇总粒度，合并案例下的每个原子请求仍单独发起并单独记录证据。

### 哪些 K3 临时案例跑不了（`not_evaluable`）

当前有 4 条用例只写了场景、没有可发送的请求内容（缺 `messages`），执行器不会为它们伪造请求，直接判 `not_evaluable`：

| 案例 | 场景字段 | 缺什么 |
|---|---|---|
| `K3-AUTH-002`、`K3-TOOL-002`、`K3-TOOL-003` | `scenario` | 需要人为构造限流 / 多工具并发场景 |
| `K3-AUTH-001` | `headers`(无效 key) | 清单未给请求体，补一条 `messages` 即可恢复为可执行 |

`CACHE-HIT-7` 和 `CACHE-TTL-3` 已由 `kvv/temp_runner.py` 的专用执行器支持：先发送一次 warm 请求，再按原子案例逐条发送同前缀 probe；`CACHE-TTL-3` 在 warm 后等待 30 秒。每次请求都写入 `execution.jsonl`。命中判定按**聚合口径**（清单 `admission_checklist` A04）：单次 `cached_tokens / prompt_tokens ≥ 80%` 记为一次命中，**命中 probe 占比 ≥ 80%** 才算该档达标——逐条严判会把「新前缀组前 1~2 次不命中」这类上游缓存的正常表现误判成失败。独立的 `k3perf/` 缓存压测（`CACHE_T0`/`CACHE_L1–L4`/`CACHE_C1`）仍保留，用于并发和长稳态缓存指标。

退出码约定：`0` 通过，`2` 未通过或数据不足。

`scripts/run_contracts.py` 的探针按 family 读取 `common/data/<family>.json` 作为可选覆盖配置；文件缺失时使用内置默认值，不会中断。

---

## 结果产物与判定契约

每次运行在 `results/<CASE>-<时间戳>/` 下落盘：

| 文件 | 内容 |
|---|---|
| `summary.json` | 机器可读统计 + 准入判定，含 `schema_version` 与 `validation` |
| `validation.json` | 统一校验结论：领域 checks + 证据存在性 checks |
| `report.md` | 人类可读报告 |
| `requests.jsonl` | **请求级原始证据**：请求、响应、原始 SSE、usage、provider request-id、时间点、传输异常 |
| `requests.csv` / `errors.csv` / `statistics.csv` | 便于 Excel 筛选的扁平视图 |
| `metadata.json` | 运行参数、模型、端点、环境 |
| `connectivity.json` | 前置连通性探测结果 |

批次运行还会在 `results/BATCH-<id>/` 下产出 `aggregation_manifest.json` 与汇总报告。

### 判定契约

核心原则：**传输成功不等于用例通过。**

- `transport_status = completed` 只表示请求跑完了；`case_status = passed` 才表示断言全部通过。
- `not_evaluable` 永远不计为通过。
- `common/finalizer.py` 负责终审：`admission_passed = 用例自身准入 AND 无硬阻断项 AND 证据存在`。硬阻断集合为 `failed / error / not_evaluable / interrupted`；**`blocked`（平台前提不满足、无法观测）只记录、不阻断准入**，但整批全为 `blocked`（无任何有效结论）时仍不准入。
- 结果 schema 为 `mltest.result.v3`；历史结果的原始格式保留，避免被静默改写。
- 前置连通性失败时，正式测试不会启动，直接产出 `connectivity_failed` 并判不通过。

四层架构（详见 `common/architecture.md`）：`CaseDefinition / RequestBuilder` → `Runner / Transport` → `Evidence Log` → `Validator / Metrics`。

### 测试面：模型能力 / 模型验真 / 通道契约

用例按「它到底在测什么」分三层（`common/faces.py`，各套件 `<suite>/faces.py` 登记）：

| 面 | 测什么 | 默认执行 | 失败后果 |
|---|---|---|---|
| `model` 模型能力 | 推理/代码/长上下文/工具闭环/视觉/缓存命中 | ✅ | **阻断准入** |
| `verify` 模型验真 | 身份回显、指纹挑战、思考真实性、注入识别 | ✅ | **阻断准入** |
| `channel` 通道契约 | 第二协议、beta、Files/余额/models、参数越界是否报错、错误信封 | ❌ 默认不跑 | 只记 `channel_gap`，**永不阻断** |

为什么这么分：走中转时网关通常只承诺 Chat Completions 主面，通道面的 404/400 会把「网关少实现几个端点」读成「模型不行」。
所以**默认只跑模型能力 + 验真**；要测通道契约就在界面勾「含通道面」，或命令行 `--faces all` / 环境变量 `VOLC_FACES=channel`。

- 定向复测（`DS_CASE_IDS` / `QWEN_CASE_IDS`）是显式点名，**不受面过滤影响**；
- 评分同步收窄：未执行的面的用例整条移出分母，`score.json` 会写 `executed_weight / scope_weight` 与 `skipped_cases`，
  分数只代表已测面（详见 `docs/模型测试评分体系设计.md` 12.3）；

---

## 架构

```
前端 (web/static)  ──POST /api/runs──▶  web/server.py
                                          │  参数校验 + 用例白名单 + 入队 SQLite
                                          ▼
                                    worker() 单线程串行
                                          │  subprocess.Popen（每个任务独立进程）
                                          ▼
                    scripts/run_*.py  ──▶  领域引擎
                                          │  volcperf / k3perf / kvv / dsv4 / glm53
                                          │  gpt_image / authenticity / moderation
                                          ▼
                              results/<CASE>-<ts>/  落盘证据
                                          │
                                          ▼
                       common/finalizer.py  统一终审 → mltest.result.v3
                                          │
                                          ▼
                     web/aggregation.py  批次聚合 → 报告回填 runs.db
```

| 目录 | 职责 |
|---|---|
| `web/` | Web 服务、任务队列、批次聚合、静态前端 |
| `common/` | 判定契约、校验器、执行框架、证据日志、终审、结果适配 |
| `volcperf/` | 性能与容量压测引擎 |
| `k3perf/` | KV 缓存专项引擎 |
| `kvv/` | 「Kimi 测试」模块：官方 KVV 套件封装 + K3 供应商临时案例集（清单与解析器也在此） |
| `strictperf/` | 长稳态压测封装 |
| `dsv4/` `glm53/` | DeepSeek / GLM 能力专项 |
| `gpt_image/` | 图像生成压测与 token 验真 |
| `gemini3/` | Gemini 3.x 原生 GenerateContent 专项（文本 / 多模态 / 工具 / 指纹） |
| `seedance/` | Seedance 视频生成（异步任务：提交 / 轮询 / 视频归档 / 报告链接） |
| `chatpane/` | 对话窗口（单条消息原样发送、完整响应归档） |
| `vendor_verifier/` | 官方 Kimi Vendor Verifier 快照（第三方资产，尽量不改写） |
| `moderation_test_project/` `moderation_keywords/` | 内容安全用例生成与报告 |
| `scripts/` | 所有 CLI 入口与一次性工具脚本 |
| `tests/` | 平台自身单元测试 |
| `data/` `data_cache/` | 压测数据集；`data_cache/` 为 85% 公共前缀的缓存配方 |
| `test_cases/` | 清单驱动的原子用例定义 |

---

## 开发约定

- **主干分支 `master` 为只读发布线，日常开发一律在 `dev` 分支进行**，通过合并请求合入 `master`。
- 内部依赖：数据加载失败/文件缺失应降级到默认值，不要硬崩（历史故障：`load_profile_data` 对缺失文件抛 `FileNotFoundError`）。
- 敏感头必须在落盘前脱敏，走 `common/audit.py` 的 `AuditLog`，不要绕过它直接写文件。
- 新增测试项时：结果行必须同时给出布尔 `passed` 与字符串 `status`，并在 `summary` 顶层给出 `admission_passed`，否则会被适配层判成 `not_evaluable` 而阻断准入。
- **新增一整套 Web 套件要同时登记七处**（漏一处就是「后端能跑、界面没入口」）：`volcperf/protocols.py` 的 `PROTOCOLS` 与线格式分组 → `web/server.py` 的 `SCRIPT` 常量 / 用例元数据 / 入队校验 / `_execute_child` 的 env 与派发 / `available` 与 `protocols_for` 白名单 → `web/static/index.html` 的 tab 与参数面板 → `web/static/app.js` 的 `kindLabel`、`kindMatches`、恢复白名单与表单持久化 → `tests/test_web.py` 的协议与 tab 守卫。完整清单与逐点说明见 `docs/Seedance视频生成测试.md` 的「Web 登记点」。
- 改完请跑 `.venv/bin/python -m pytest tests/ -q`。
- **DeepSeek V4 能力专项的唯一权威实现是 `dsv4/` 包**：用例定义在 `dsv4/core.py`（`MODELS` / `CASE_SPECS` / handler），展示元数据在 `dsv4/cases.py`，装配与落盘在 `dsv4/runner.py`，CLI 入口固定为 `scripts/run_dsv4.py`。init 时代的旧副本 `k3perf/deepseek_v4.py` + `scripts/run_deepseek_v4.py` 已删除，**不要在 `k3perf/` 下重建同类并行实现**（它缺 DS-V4-18 与整个 P2 段，只接受 `--priority`/`--case`，位置传参直接 argparse 中止）。

---

## 已知问题

按严重程度排列，修复顺序建议照此进行。

### P0

1. **5 个开源基准与 1 个 GLM 流式复测仍无法从 UI 正常运行**
   `web/server.py` 的脚本分发表仍引用了 6 个未定义的常量：`ARXIVMATH_SCRIPT`、`DEEPSWE_SCRIPT`、`SWE_ATLAS_TW_SCRIPT`、`SWE_ATLAS_RF_SCRIPT`、`LEGAL_AGENT_BENCHMARK_SCRIPT`、`GLM_STREAM_RETEST_SCRIPT`。
   触发时会产生 `NameError`；CLI runner（`scripts/run_arxivmath.py` 等）本身已经实现，可直接从命令行运行。Web worker 现在会捕获任务异常并将任务标为 `failed`，提交新任务时也会自动拉起已退出的 worker，不再因为单个任务异常永久卡死队列。

2. **平台无入站鉴权**，且 `start.sh` 默认监听 `0.0.0.0`。
   同网段任何人都可以提交测试（消耗你的 API 额度）、读取全部报告与 `requests.jsonl`、打包下载。对外暴露前请改为 `127.0.0.1` 或加一层 Bearer 校验。

3. **KVV 依赖独立的官方 Python 环境**。
   `kvv/runner.py` 默认使用 POSIX 的 `vendor_verifier/.venv/bin/python`（Windows 为
   `vendor_verifier\.venv\Scripts\python.exe`），也可以通过
   `KVV_HOME` / `KVV_PYTHON` 显式指定官方套件目录和解释器。当前工作区已具备该环境，
   `tests/test_kvv.py` 可通过；在新环境部署时需先按官方套件要求创建该虚拟环境。
   Windows 下若该环境存在，`start.bat` 会自动设置 `KVV_PYTHON`。

KVV 官方快照已于 2026-10-22 对齐 [MoonshotAI/Kimi-Vendor-Verifier](https://github.com/MoonshotAI/Kimi-Vendor-Verifier/commit/66092cf444c97356c0e11c5078c67116390615d9) 的 main 提交 `66092cf`（2026-09-17）。版本和本地差异见 `vendor_verifier/UPSTREAM.json`、`vendor_verifier/LOCAL_CHANGES.md`。Prompt Token 的 44 个文本、11 个视觉样本及图片原图已随仓库保存，无需 Git LFS；按上游代码允许 `[expected, expected+3]`。动态工具与 Thinking Effort 同时覆盖流式/非流式，保留官方 skip/flaky；检查项数不等于实际请求数，部分项目会多次请求或重试。可在「Kimi 测试」中单独选择新增的三个入口。BEAM 大型 benchmark 数据不在本次 pytest 集成范围内。

### P1

4. **空断言恒过**：`common/test_framework.py:validate_case(record, [])` 对任意 HTTP 状态和传输异常都返回 `passed`；`common/validators.py` 的 `content()` / `tool_calls()` 无参调用恒真。漏挂断言的新用例会静默通过。
5. **单 worker 串行 + 子进程无超时**：一个卡住的测试会阻塞整条队列；`cancel` 对排队中的任务只改数据库状态，不摘队列，任务仍会启动并消耗额度。worker 异常隔离和自动拉起已覆盖，但单任务超时和排队任务取消仍待改进。
6. **KVV 官方跳过项仍单列**：保留上游 skip 标记；平台已要求每个所选套件至少一个测试通过，全 skip 不再准入。独立 `KVV_PROMPT_TOKENS` 提供 55 条文本/视觉基线请求，补足核心套件被跳过的 tokenization 检查。
7. **`authenticity` 永远显示未通过**：其 summary 不含 `admission_passed`，被终审规则 `original AND ...` 判为 False，即使 6 个探针全过。

### P2

8. 指标口径跨引擎不一致：`k3perf` 把首个可见内容记为 `ttft`（对思考模型实为 TTFC）；两引擎 TPOT 基准不同（可见 token vs 含 reasoning），数字不可直接比较。
9. 重试策略不一致：`volcperf` / `strictperf` / `gpt_image` / `dsv4` / `glm53` 均无重试，429 直接计为失败，而 L 档要求成功率 ≥99% —— 一次限流即整体不通过。仅 `k3perf` 有 2 倍尝试预算、`moderation` 有 3 次指数退避。
10. `moderation_test_project/build_report.py` 用 `len(v)==3` 判定「结构完整」，而生成器每组产 5 条 → 该指标恒为 0。
11. `vendor_verifier/verify_params.py` 把任意异常（401 / 429 / 超时）都判为「正确拒绝」；`mmmu_pro_vision.py` 解析失败时随机猜选项。
12. 死代码与重复：`vendor_verifier/utils.py` 整模块无调用；`common/` 下多个模块仅被测试引用；`scripts/run_.py` 文件名残缺。
13. 核心路径缺测试：`validate_case` 空断言、`suite.execute_manifest_case`、`runner.Runner`、`audit.redact`。

完整的代码审计报告（含行号与实测复现步骤）见 `CODE_AUDIT_2026-09-22.md`。
