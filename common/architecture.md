# 统一测试执行架构

## 契约

所有测试集统一经过四层：

1. **CaseDefinition / RequestBuilder**：定义并构造真实请求；不把 `scenario` 降级为普通请求。
2. **Runner / Transport**：只负责 HTTP、SSE、超时、并发和计时；不决定案例是否通过。
3. **Evidence Log**：实时写入 `execution.jsonl`，保存请求、响应、原始 SSE、usage、request-id、时间点和传输异常。
4. **Validator / Metrics**：按案例断言校验 HTTP status、错误语义、响应结构、内容、工具、SSE、usage、TTFT/TTFC/TPOT、缓存和样本完整性。

`transport_status=completed` 只表示请求完成；`case_status=passed` 才表示案例断言全部通过。`not_evaluable` 永远不计为通过。

## 兼容策略

- 旧的 load / KVV / DeepSeek / GLM / moderation / benchmark runner 可以继续使用自己的领域调度和官方执行器，但必须输出统一的 execution record 和 validator verdict。
- 官方 pytest 套件的 pytest assertion 结果作为 validator verdict，不再由外层 runner 根据 HTTP 200 推断。
- 旧结果保留原格式；新执行结果使用 `mltest.result.v3`，避免历史报告被静默改写。
