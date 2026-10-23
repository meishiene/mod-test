# ml-test-platform 代码审计报告

审计日期：2026-09-22
审计基线：`32bd2d0`（master，工作区干净）
审计范围：仓库内 159 个 Python 文件 / 约 17,900 行（排除 `.venv`），含 `web/`、`common/`、`scripts/`、`volcperf/`、`k3perf/`、`kvv/`、`strictperf/`、`dsv4/`、`glm53/`、`gpt_image/`、`vendor_verifier/`、`moderation_*`、`tests/`

> **历史快照说明（复核于 2026-09-23）**：本文保留 2026-09-22 审计时的原始现象和复现步骤，不回写历史结论。后续已修复：worker 异常隔离/自动拉起、Web 列表分页测试夹具，以及 `vendor_verifier/.venv` 运行环境。仍未修复：`web/server.py` 中开源基准和 GLM 流式复测的脚本路径常量未定义、KVV 全 skip 仍可能被判准入通过、authenticity 缺少 `admission_passed` 等问题。

---

## 一、总体判断

这是一个**面向多供应商大模型验收的端到端测试平台**，不是玩具项目：领域测试引擎、官方套件封装、结果契约、Web 编排层、批次聚合、请求级证据留痕都已成型，且前期已排掉多个真实故障（KVV venv、`load_profile_data` 崩溃、`audit_log` 未定义、契约结果格式不符）。

但当前状态是 **"引擎层成熟、编排层断裂"**：

- 领域引擎（压测/能力/安全/图像/官方套件）功能完整、指标口径清晰、并发无竞态、无 TODO 占位；
- 编排层（`web/server.py`）存在一个**已实证的 P0 级断线**：5 个开源基准测试项的脚本路径常量从未定义，触发即 NameError，并会**永久杀死任务队列线程**；
- 质量把关层存在**结构性恒真漏洞**：空断言校验器返回 `passed`；
- 平台**零鉴权 + 默认监听 0.0.0.0**。

一句话：**能跑的部分跑得不错，跑不了的部分不是"没做完"而是"接线掉了"，而且掉线会连带拖死整个队列。**

---

## 二、已实现的功能

### 2.1 平台层（Web）

标准库 `ThreadingHTTPServer`（非 Flask），`bash start.sh` 一键启动（自动建 venv、补依赖、释放端口），默认 `0.0.0.0:8085`。

| 能力 | 实现位置 |
|---|---|
| 任务提交 / 自动成批 | `POST /api/runs`（`web/server.py:850`） |
| 任务取消 | `POST /api/runs/{id}/cancel`（:788） |
| 连通性预检 | `POST /api/connectivity/check`（:806） |
| 模型列表查询 | `POST /api/models/query`（:825） |
| 记录列表 / 详情（批次带 children） | `GET /api/runs`（:665）/ `{id}`（:670） |
| 报告 / 指标 / 日志 / 聚合清单 | `GET /api/runs/{id}/{report,summary,log,manifest}`（:733） |
| 请求级明细 | `GET /api/runs/{id}/requests`（:726） |
| 单条 / 多选 zip 打包下载 | `GET`（:684）/ `POST /api/runs/download`（:752） |
| 静态单页前端 | `web/static/index.html` + `app.js`（9 个测试项 tab、3 秒轮询记录、报告弹窗） |

调度：单 daemon 线程 `worker`（:565）串行消费 `queue.Queue`，每个任务 `subprocess.Popen(..., start_new_session=True)`（:472）。状态持久化在 SQLite `web/data/runs.db`，重启时把 queued/running 置为 interrupted。

### 2.2 测试引擎层

| 引擎 | 测试项 | 内容 |
|---|---|---|
| `volcperf/` | T0、RANDOM_LOAD、B1、L1–L5、O1、C0/C1，+ SIMPLE_LOAD/FULL_LOAD 预设 | 闭环/开环压测；TTFT/TTFC/TPOT/E2E/成功率/截断率；容量窗口含 RPM/TPM 达标、积压斜率；L\* 阈值与 B1 基线相对劣化 3× |
| `k3perf/` | CACHE_T0、CACHE_L1–L4、CACHE_C1 | KV 缓存专项，`data_cache/` 85% 公共前缀配方，缓存命中率门槛 0.8–0.9 |
| `kvv/` | KVV_K3_FEATURES、KVV_PARAMS、KVV_TOOL_SCHEMA | 直接封装官方 KVV pytest 套件，解析 JUnit XML，key 在日志中打码 |
| `strictperf/` | 包装 T0/L1–L5/C1 | 300 样本 / 10 分钟稳态 |
| `dsv4/` | DS_V4_P0/P1/P2/FINGERPRINT（35 条） | 思考回传、工具、JSON、缓存命中、Session、Prefix/FIM、Anthropic/Responses |
| `glm53/` | GLM_53_CORE/AGENT/CACHE/FINGERPRINT | 同上，缓存目标 ≥70% |
| `gpt_image/` | GPT_IMAGE_2 / 2.5 / 4K_TOKEN_VERIFY | 图像压测 + 4K、n=10 的 token 核验（PIL 解码 + 768 B/token 上限规则） |
| `awsb.py` | AWSB-O5-01~04 + CACHE | Anthropic Messages 协议下 Claude 网关自检（身份/thinking/边界/流式工具/缓存） |
| `authenticity.py` | DS_AUTH_FINGERPRINT、DS_V4_FINGERPRINT | 满血指纹 6 探针：动态 required 工具、随机 JSON Schema、多轮记忆、reasoning 字段、SSE 完整性、静默吞参对照 |
| `moderation_*` | MOD_GENERAL_COMMON、MOD_PARTNER_COMMON、MOD_PARTNER_SMOKE | SillyTavern 上下文劫持 + 关键词变体；4 项指标（有效结果率、分类一致率、高风险漏判率、误报率）+ 结论翻转率/劫持率 + 不变性分组 |
| `scripts/run_contracts.py` | KVV/DS/GLM_CONTRACT | 每族 5 探针公共契约 |
| `scripts/run_{arxivmath,deepswe,sweatlas_tw,sweatlas_rf,legal_agent_benchmark}.py` | 5 项开源基准 | arxivmath 用 MathArena 官方 HF 数据集；deepswe 走官方 Pier + 官方 verifier；SWE-Atlas TW/RF、Legal Agent Benchmark 走官方 harness |

UI 可直接列举的用例合计 **40 项**；支持 6 种协议：`openai`、`openai_responses`、`openai_images`、`anthropic`、`gemini`、`volcengine`。

### 2.3 公共框架

- `common/verdict.py`：`Check`/`Verdict` 状态机，`not_evaluable` 不计通过。
- `common/validators.py`：19 个校验器（status / content / usage / sse / tool_calls / metric / cache_ratio / percentile …）。
- `common/finalizer.py`：**这是全项目质量把关最扎实的一环** —— 合并领域检查 + 证据存在性检查，`admission = 原准入 and 无阻断项`，并落 `validation.json`；`mltest.result.v3` 就是在这里盖章的。
- `common/audit.py`：`AuditLog` 实时写 `requests.jsonl`，敏感头自动脱敏。
- `common/architecture.md`：定义了四层契约（CaseDefinition → Runner/Transport → Evidence Log → Validator/Metrics）。

---

## 三、完成度评估

### 3.1 完整可用
`volcperf`、`k3perf`、`kvv`、`strictperf`、`dsv4`、`glm53`、`gpt_image`、`moderation`、`awsb`、`authenticity`、`run_contracts`、Web 平台主体。21 个模块全部可导入，无坏 import，无 TODO / NotImplementedError / 假结果。

### 3.2 断线 / 不可达

| 项 | 状态 |
|---|---|
| 5 项开源基准（arxivmath / deepswe / swe-atlas-tw / swe-atlas-rf / legal-agent） | **runner 脚本存在且是真实实现，但 `web/server.py` 未定义对应脚本路径常量** → 从 UI 侧完全不可达，且触发即 P0 故障 |
| `k3_features_extensions` 官方套件（`test_dynamic_tools.py`、`test_thinking_effort.py`） | 定义在 `SUITE_PATHS` 但没有任何 `KVV_CASES` 引用 → 死配置，无法从 UI 运行 |
| `gpt`、`glm_stream_retest` 分支 | `web/server.py` 中的死分支，`available` 映射不可达 |
| `scripts/run_.py` | 文件名残缺，传参含空格 |

### 3.3 四层契约的实际落地度
架构文档声称"所有测试集统一经过四层"，**实际只有 `scripts/run_temp_cases.py` 完整走通**。其余为"只借一半"：`kvv`/`dsv4`/`glm53`/`strictperf`/`volcperf`/`k3perf` 只 import `finalize` + `AuditLog`，判决仍由各自领域逻辑给出；`moderation_test_project/`、`vendor_verifier/` 基本自成一派。`mltest.result.v3` 是 finalizer 盖在 `summary.json` 上的字段，**没有任何 runner 原生产出 envelope**（`common/artifacts.py:result_envelope()` 仅被测试引用）。

### 3.4 死代码 / 重复
- `vendor_verifier/utils.py`：整个模块零调用点。
- `common/{runner,case_model,case_schema,request_ids,legacy,verdict}.py`：无生产引用。
- `k3perf/awsb.py` 与根 `awsb.py` 双分叉，易漂移。
- `vendor_verifier/pyproject.toml` 引用不存在的 README/LICENSE；`awsb.py` base_url 默认 `api.example.com` 占位。

### 3.5 测试覆盖
`tests/` 覆盖了 finalizer、result_adapter、contracts、protocols、metrics、aggregation、web。**零测试的关键路径**：`validate_case`、`common/suite.py:execute_manifest_case`、`common/runner.py:Runner`、`audit.redact`、`verdict`/`case_model`/`case_schema`/`request_ids`/`legacy`。
`results/` 目录下只有 **1 次真实运行**（`CACHE_T0-20260921-171905`，20 请求全部 `client_error`，成功率 0）——说明端到端链路在近期很少被完整跑通。

---

## 四、风险清单

### P0 —— 会直接破坏可用性

**R1｜5 个未定义常量导致 NameError，并永久杀死任务队列线程**

`web/server.py:446` 引用了 6 个从未定义的常量：

```
ARXIVMATH_SCRIPT / DEEPSWE_SCRIPT / SWE_ATLAS_TW_SCRIPT
SWE_ATLAS_RF_SCRIPT / LEGAL_AGENT_BENCHMARK_SCRIPT / GLM_STREAM_RETEST_SCRIPT
```

AST 静态检查确认这 6 个名字在模块级不存在（同名的 `*_CASE` 常量存在，说明是漏写路径而非漏写整个功能）。实证触发：

```
arxivmath            -> NameError: name 'ARXIVMATH_SCRIPT' is not defined
deepswe              -> NameError: name 'DEEPSWE_SCRIPT' is not defined
swe-atlas-tw         -> NameError: name 'SWE_ATLAS_TW_SCRIPT' is not defined
legal-agent-benchmark-> NameError: name 'LEGAL_AGENT_BENCHMARK_SCRIPT' is not defined
```

爆炸半径之所以是 P0：`worker()` 只有 `try/finally`，**没有任何 `except`**（全文件 68KB 仅 `:544` 一处 `except Exception`，且位于 `script = ...` 之后）：

```python
def worker() -> None:
    while True:
        job = JOB_QUEUE.get()
        try:
            ...
        finally:
            job["api_key"] = ""
            JOB_QUEUE.task_done()      # 无 except → 异常逃逸 → 线程死亡
```

后果：用户在 UI 点任一"开源测试集"tab 提交 → worker 线程终止 → **此后所有任务永久停留在 queued，必须重启进程才能恢复**。前端 `app.js:215` 的 `effectiveTestKind` 正好会解析出这 5 个 `test_kind`。

**R2｜零鉴权 + 默认监听 0.0.0.0**

- `start.sh` 默认 `HOST=0.0.0.0`、`PORT=8085`；`web/server.py` 中 `Authorization`/`token`/`session`/`cookie` 出现次数为 **0**，无任何入站鉴权、无 CORS 头。
- 同网段任何人对 `/api/runs` 发一个 POST，即可**用你的 API key 发起真实测试消耗额度**（key 由请求体带入，服务端不校验来源）。
- 可读取全部报告、日志、`requests.jsonl`（含上游原始 SSE 与响应体）并打包下载。
- 路径穿越已防护（静态与产物均 `resolve()` + 父目录校验）；命令注入不存在（`Popen` 用列表、`shell=False`、case 走白名单）。
- API key 不落库（`tests/test_web.py:109` 验证），但仍通过 env 传给子进程，同机同用户可从进程环境读取。

### P1 —— 会让结论失真

**R3｜空断言恒真漏洞**

```python
def validate_case(record, assertions):
    checks = []
    for validator in assertions: ...
    if failed:  return failed
    if unknown: return not_evaluable
    return ValidationResult("passed", True, checks, [])   # checks 为空 → 直接 passed
```

`validate_case(record, [])` 对任意 HTTP 状态、任意传输异常都返回 `passed`。同理 `validators.content()` / `tool_calls()` **无参调用时恒真**。任何新增用例只要漏挂断言，就会静默"通过"。这与平台"HTTP 成功不等于案例通过"的核心承诺直接冲突。

**R4｜单 worker 串行 + 子进程无超时；取消对排队任务无效**

- 一个卡住的测试（4K 图像单模型超时 1800s）会阻塞整条队列。
- `cancel` 对 queued 任务只把 DB 状态改为 interrupted（:788），**不摘队列、无进程可杀**；worker 拿到它时仍会 `update(status="running")` 并 `Popen` 启动，照常消耗 API 额度，直到跑完才看到 `CANCELLED_RUNS`。

**R5｜KVV 准入忽略 skipped，唯一 groundtruth oracle 被跳过**

- `kvv/runner.py:117`：`admission = exit_code==0 and failed==0 and errors==0`，**skipped 不计入**。pytest 全 skip 时退出码为 0 → admission 仍为 True。
- `vendor_verifier/tests/k3_features/test_tokenization_groundtruth.py:83` 是该套件**唯一独立的 prompt_tokens groundtruth 断言**，被标为 `@pytest.mark.skip("skip, not pass")`，且它正是 `SUITE_PATHS["k3_features_core"]` 的成员。等于核心套件里最硬的"token 计费真值"检查已经失效。
- 同套件 `test_thinking_effort.py` 另有 6 处 skip。

**R6｜authenticity 永远显示"未通过"**

`authenticity.py` 的 summary 不含 `admission_passed`，而 `finalizer` 的规则是 `admission = original and not blocking`，`original=False` → 恒为 False。即使 6 个探针全部 `passed`，报告仍判未通过 / not_evaluable。这属于报告口径缺陷，会掩盖真实结论。

### P2 —— 影响可比性与可维护性

| 编号 | 风险 |
|---|---|
| R7 | **指标口径跨引擎不一致**：`k3perf/client.py:131` 把首个 visible content 记为 `ttft_seconds`（对思考模型实为 TTFC）；TPOT 基准两引擎不同（visible vs 含 reasoning）→ 两引擎数字不可直接比较。 |
| R8 | **重试策略不一致**：`volcperf`/`strictperf`/`gpt_image`/`dsv4`/`glm53` 无重试，429 直接计失败，而 L\*/C1 要求成功率 ≥0.99 → 一次限流即整体不通过（历史批次已多次命中 429）。只有 `k3perf` 有 2× 尝试预算、`moderation` 有 3 次指数退避。 |
| R9 | **`moderation_test_project/build_report.py:29`** 用 `len(v)==3` 判定"结构完整"，而生成器（`generate_cases.py:132`）每组产 **5** 条 → 该指标恒为 0。同文件 `:36` 又会接受 `{3,5}`，前后自相矛盾。 |
| R10 | **`vendor_verifier/verify_params.py` 假通过**：任意异常（401/429/超时）都被判为"正确拒绝"。 |
| R11 | **`vendor_verifier/mmmu_pro_vision.py:160`** 解析失败时随机猜选项 → 分数虚高且不可复现。 |
| R12 | **`gpt_image/token_verify.py`** 的 token 核验依赖 `IMAGE_BYTES_PER_TOKEN = 768` 这一假设常量，属编码进代码的口径约定而非实测，换模型/编码需同步改动。 |
| R13 | **evidence 膨胀**：`requests.jsonl` 保存完整 `raw_sse_lines` / `raw_sse_events` / `raw_response_body`，C1（1890 请求）体量会显著增长。 |
| R14 | **`k3perf/client.py:68-69` 硬编码** `temperature=1` / `reasoning_effort=low`；`awsb.py:153` 要求原生模型 ID ≥400，合法实现会被误判。 |
| R15 | **`kvv/runner.py:35`** 依赖 `vendor_verifier/.venv` 存在性（历史上因非法 venv 导致官方套件 0 用例通过）；`KVV_PYTHON` 需在 Web 注入才生效。 |
| R16 | 死代码 / 双分叉分散注意力（见 3.4）。 |

### 补记（同日新发现，级别 P0）

**R17｜`vendor_verifier/.venv` 已不存在，官方 KVV 套件全部无法执行**

R15 描述的是"依赖 `.venv` 存在性"这一设计脆弱点，本次实测确认该依赖**当前已经断裂**：

```
$ .venv/bin/python -m pytest tests/ -q
FAILED tests/test_kvv.py::OfficialKvvTests::test_existing_official_runtime
E   ValueError: 未找到官方 KVV Python：/Users/liuyeyu/workspace/ml-test-platform/vendor_verifier/.venv/bin/python
1 failed, 99 passed in 1.90s
```

`ls -la vendor_verifier/` 中已无 `.venv`。据 2026-09-10 的工作记录，该路径曾是**指向 `../.venv` 的软链**（当时为修复"官方套件 0 用例通过"而建），推测在后续重克隆 / 清理中丢失。

影响：`KVV_K3_FEATURES`、`KVV_PARAMS`、`KVV_TOOL_SCHEMA` 三个测试项全部无法运行，`kvv/runner.py:validate_runtime()` 直接抛错。

恢复（该路径已在 `.gitignore`，软链不会被提交）：

```bash
ln -s ../.venv vendor_verifier/.venv
```

建议同时评估 R15 提到的 `KVV_PYTHON` 环境变量方案，让依赖不再隐式绑定目录布局。

---

## 五、建议修复顺序

1. **（立刻）补 `web/server.py` 的 5 个脚本路径常量**，同时给 `worker()` 加 `except Exception` 兜底：把任务标 failed + 记录 traceback，保证单任务失败不会终止队列。这是当前唯一"点一下就整站瘫痪"的问题。
2. **（立刻）恢复 `vendor_verifier/.venv`**：`ln -s ../.venv vendor_verifier/.venv`。一条命令让 3 个 KVV 测试项恢复可用（当前完全跑不了，见 R17）；同时按 R15 评估改用 `KVV_PYTHON` 显式指定解释器，别让依赖隐式绑定目录布局。
3. **（立刻）收紧监听**：`start.sh` 默认改 `127.0.0.1`，或加一层 Bearer 校验；对外暴露前不要用 `0.0.0.0`。
4. **（高）修恒真漏洞**：`validate_case` 对空断言返回 `not_evaluable`；`validators.content()` / `tool_calls()` 无参时抛错而非恒真；补上这几条路径的单元测试。
5. **（高）KVV 准入把 `skipped` 纳入判定**；同时决定 `test_tokenization_groundtruth.py` 是修复后启用还是明确标注为"不覆盖"。
6. **（高）authenticity 补 `admission_passed`**，让 finalizer 能给出真实结论。
7. **（中）统一重试策略**，至少在 L\*/C1 前置 429 退避；对齐两引擎的 TTFT/TPOT 口径。
8. **（中）清理死代码与双分叉**（`utils.py`、`k3perf/awsb.py`、`scripts/run_.py`、无生产引用的 `common/*`），减少后续改动踩空。
9. **（中）给 `results/` 做一次完整端到端回归**：目前只有 1 条真实记录，接线类问题（如 R1）正是靠这个才能被发现。

---

## 附：核心结论一句话版

- **功能面**：11 类测试引擎 + 40 个用例 + 6 协议 + 完整 Web 编排与证据留痕，覆盖性能/缓存/能力/安全/图像/官方套件/开源基准。
- **最大缺口**：5 项开源基准的 Web 接线断裂（P0，连带拖死队列）；四层契约只有 1 个模块真正走通。
- **最大风险**：零鉴权暴露（可用你的额度跑测试 + 读全部结果）；空断言恒真（漏挂断言的用例静默通过）；KVV 忽略 skip（唯一 groundtruth 已失效）。
