# GLM / Qwen 能力、真实性与满血证据校验

新增三类卡片，Web 和 CLI 共用执行器：

| 目的 | GLM | 千问 |
|---|---|---|
| 客观能力 | `GLM_53_CAPABILITY` | `QWEN_CAPABILITY` |
| 真实性与路由 | `GLM_53_VERIFY` | `QWEN_VERIFY` |
| 综合能力与满血证据 | `GLM_53_FULL` | `QWEN_FULL` |

综合校验要求该型号**全部必测项有效通过**。blocked、warning、缺失结果、另一个型号的结果、本地汇总项均不能补足必测证据。局部复测不构成综合通过。独立能力/真实性卡片也要求本卡证据完整。

## 型号差异

以下为 `common/model_validation.py` 的版本化测试策略，**不是厂商公布的上下文极限或能力保证**。型号名大小写、供应商前缀可归一；Flash 和 FlashX 精确区分，禁止子串匹配。

| 型号 | 客观推理样本 | 长文本填充预算（字符） | 视觉 |
|---|---:|---:|---|
| `glm-5.3` | 5 | 65,536 | 未取得可靠声明，不计为支持/不支持 |
| `glm-5.3-flash` | 3 | 32,768 | 同上 |
| `glm-5.3-flashx` | 3 | 32,768 | 同上 |
| `qwen3.8-max` | 5 | 131,072 | 按仓库声明执行五级视觉题 |
| `qwen3.7-max` | 5 | 65,536 | 按仓库声明不适用，不发送图片请求 |
| 已登记 Qwen Plus / Flash | 3 | 32,768 | 按仓库现有矩阵 |

Qwen 的视觉适用性沿用 `qwen/cases.py` 现有矩阵（仓库 2026-10-21 观测说明），不是本次重新进行的在线厂商校准。GLM FlashX 的独立上下文/输出限制和性能优势没有可信来源，因此先复用 Flash 工作负载，仅严格区分型号；不虚构专属性能门槛。

千问共用题显式关闭思考；思考开关和流式推理由独立思考用例检查。GLM 共用题开启思考，现有 low/medium/high 档位探针继续检查答案正确性。API 不支持某参数意味着该项缺乏通过证据，不能由此认定底层模型被替换。

## 共用断言

每族新增九个顶层用例：

- `Q01`：整数算术与离散最优化题，计算标准答案，检查最终答案正确；种子控制变体。
- `Q02`：生成 `solve(nums)` 程序，九组固定边界/随机输入验证结果。使用有步数、语法、整数和集合上限的 AST 解释器，不使用 eval/exec，不执行模型代码的系统调用。
- `Q03`：中文订单抽取与总额计算，嵌套 JSON Schema 严格检查字段、值和额外属性。
- `Q04`：头、中、尾三个随机标记召回，保存真实字符数及上游 prompt_tokens。通过只证明本次长度，不证明最大窗口。
- `Q05`：工具选择及参数 → 临时错误 → 重试 → 注入随机工具结果 → 最终精确利用结果。
- `Q06`：一次响应中的两种工具，严格检查工具名称、参数、ID 和调用数量。
- `R01`：三次新随机挑战，对照每次响应的精确型号；缺失 model 字段判证据不足。
- `R02`：真实型号正对照后，发送不存在型号；明确型号不存在错误才通过，2xx 标记静默回退风险，鉴权/限流不能当作正常拒绝。
- `R03`：系统指令优先于用户文档内的冲突命令。仅证明这次挑战，不证明“没有任何隐藏系统提示词”。

现有用例同时修复：工具 SSE 需真实调用片段并正确拼接参数；GLM 空正文不能 passed；长上下文不再是短对话占位；推理需正确答案；视觉增加随种子变化的四位数字 OCR、左右空间关系，数字不能用子串匹配；usage 需非负整数且总和一致；严格 Schema 不允许额外字段；本地汇总不算有效远端证据。

`QWEN-T05` 改为小输出预算及截断行为验证，不再把没有逐型号依据的“32768 上限”套到所有型号。GLM 缓存仍为独立卡片，综合卡不把缓存实现当作厂商身份依据。旧 GLM 视觉用例已删除，雷达不再永久保留不可执行的视觉分母；历史视觉证据存在时仍可显示。

## 使用

先通过既有环境配置设置对应 `GLM_API_KEY / GLM_BASE_URL / GLM_MODEL` 或 `QWEN_API_KEY / QWEN_BASE_URL / QWEN_MODEL`，再运行：

```bash
.venv/bin/python scripts/run_glm53.py GLM_53_FULL --seed 20261023
.venv/bin/python scripts/run_qwen.py QWEN_FULL --seed 20261023
```

Web 可选择综合卡并勾选多个模型，服务端按模型生成独立子任务与报告。GLM 卡片均支持多模型；Qwen 新增的三张卡片及原有档位/视觉卡支持多模型。所有子任务使用原子创建的独立结果目录，不会因同秒执行覆盖证据。

默认执行 model + verify，通道面仍需显式 `--faces all`。卡片请求数为全部面、最大样本档的近似上界；能力卡最多 12 次，真实性卡 6 次；前置步骤失败、型号不适用或默认排除通道面会减少请求。综合卡的长输入会明显高于原先的短探针预算。

## 参考样本对照

使用操作者确认可信的参考端点，先对**同一型号、同一种子**运行综合卡，保存其 `summary.json`。再切换到待测端点：

```bash
.venv/bin/python scripts/run_glm53.py GLM_53_FULL --seed 20261023 --reference /absolute/path/to/reference/summary.json
.venv/bin/python scripts/run_qwen.py QWEN_FULL --seed 20261023 --reference /absolute/path/to/reference/summary.json
```

`--reference` 仅允许用于综合卡。参考报告必须满足型号、测试版本、种子、完整 profile 一致且全部必测项已通过；记录参考文件路径和 SHA-256。参考不兼容或当前证据不足不能判对照通过。系统不会仅凭 URL 名字认定某端点就是官方。

报告 `model_validation` 给出：

- `coverage`：必测数、已通过数、失败 ID、缺失或阻断 ID。
- `authenticity`：行为一致／行为冲突／证据不足，`proven=false`。
- `full_strength`：`profile_passed` 为达到本测试标准；`reference_consistent` 为已测项目与指定参考一致；另外区分能力缺口、证据不足、参考不兼容。
- `reference`：对照状态及参考文件证据。

**API 回显、自述和有限行为题都无法证明底层权重、真实厂商来源或全部能力满血。** 因此 `vendor_full_strength_proven` 保持 false。新流程能发现可观测的替换/回退/能力回归，并保证证据不足不被升级为“满血”；不能排除会伪装的网关或测试范围外的降级。需要更强身份保证时，必须另有厂商可验证签名或受信任执行链。

## 离线验证

```bash
.venv/bin/python -m pytest tests/test_model_validation.py tests/test_qwen.py tests/test_glm53.py tests/test_faces.py tests/test_contracts.py tests/test_forensics.py tests/test_capabilities.py -q
.venv/bin/python -m pytest tests -q
```

测试包含五个目标型号的完整 HTTP MockTransport/SSE/报告/最终准入路径，以及全阻断、错误数学答案、空工具流、错误工具参数、代码越界、同族错档、参考不兼容等反例。实现验证不请求真实模型。
