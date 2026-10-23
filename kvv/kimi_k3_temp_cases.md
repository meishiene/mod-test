# Kimi-K3 供应商临时自测案例集

- 套件：`KIMI-K3-TEMP-20260909`
- 版本：`0.2-temp`
- 契约基准：Kimi-K3 官方契约（2026-10-14 复核）
- 范围：接口兼容性（18条）、性能（40条）、缓存（10条）和自测准入，共68条原子案例；不包含 arxivmath、DeepSWE、SWE-Atlas TW/RF、Legal Agent Benchmark 等开源评测集
- 说明：文档字段是测试要求，不是发送给模型的提示词。

## 执行约定

- OpenAI Chat Completions，stream=true 时使用 SSE。
- 默认 `max_tokens=512`、并发 `10`。
- 采样参数按 K3 官方契约固定：`temperature=1`、`top_p=0.95`、`n=1`、`presence_penalty=0`、`frequency_penalty=0`。正向用例只能用固定值，非固定值属负向用例（期望被拒）。
- 思考常开且思考 token 计入输出上限：需要可见正文的用例必须给足输出预算（≥512），小预算下「零可见正文」是原厂行为，不是模型缺陷。
- 性能和缓存按原子请求计数：TTFT 30 条（16K/64K/100K 各 10 条）、TPOT 10 条、缓存 10 条。
- 每条请求保留请求体、原始响应/SSE、状态码、耗时、usage、request_id 和断言结果。

## 契约要点（2026-10-14 复核）

| 项 | 官方口径 |
|---|---|
| `temperature` | 固定 `1.0`，其他取值报错；官方建议不显式传入 |
| `top_p` | 固定 `0.95` |
| `n` / `presence_penalty` / `frequency_penalty` | 固定 `1` / `0` / `0` |
| `thinking` | K2.x 专属参数；K3 可省略，显式传只接受 `{"type":"enabled","keep":"all"}` |
| 思考开关 | 关不掉（K3 始终思考） |
| 输出上限 | 官方口径为 `max_completion_tokens`（默认 131,072 / 上限 1,048,576），本清单沿用 `max_tokens` 并记录实际用量 |
| 结构化输出 | `json_schema` 模式必须提供 `json_schema.name` 与 `json_schema.schema` |
| `tool_choice` | 支持 `auto` / `none` / `required`；不支持「指定函数名」（静默忽略，与原厂一致） |

## 案例数量

| 类别 | 原子案例数 | 说明 |
|---|---:|---|
| 接口兼容性 | 18 | 连通、错误、Thinking、参数、流式、JSON、工具、追踪 |
| TTFT | 30 | 16K/64K/100K 输入各 10 次，并发 10 |
| TPOT | 10 | 16K 输入、并发 10，独立记录后汇总判定 |
| 缓存 | 10 | 90% 前缀命中 7 次 + 30 秒 TTL 3 次 |
| **合计** | **68** | **临时测试集原子案例总数** |

## 临时案例明细

| ID | 类别 | 优先级 | 案例 | 关键参数 | 通过标准摘要 |
|---|---|---:|---|---|---|
| `K3-CONN-001` | 基础连通 | P0 | 非流式基础调用 | `{"stream":false,"messages":[{"role":"user","content":"只回复 OK"}],"max_tokens":512}` | HTTP 200；返回 choices[0].message.content；usage 含 prompt_tokens、completion_tokens、cached_tokens（cached_tokens 可位于 usage.prompt_tokens_details）；响应可解析为 JSON；响应或响应头存在可追溯 request_id |
| `K3-CONN-002` | 基础连通 | P0 | 流式基础调用 | `{"stream":true,"messages":[{"role":"user","content":"请输出三行：A、B、C"}],"max_tokens":512}` | HTTP 200；Content-Type 为 text/event-stream 或等价 SSE；每个 data 事件均为合法 JSON（[DONE] 除外）；正常收到 [DONE]；存在 finish_reason=stop 或原厂等价结束语义；正文可由 delta.content 按序拼接 |
| `K3-AUTH-001` | 认证与错误 | P0 | 无效认证 | `{"headers":{"Authorization":"Bearer invalid-key"},"stream":false}` | HTTP 401；错误 body 可解析；错误语义明确为认证失败；不返回 HTTP 200 伪成功 |
| `K3-AUTH-002` | 认证与错误 | P0 | 限流错误 | `{"stream":false,"scenario":"在并发/额度受限条件下触发限流"}` | HTTP 429；错误 body 可解析；语义明确为限流/速率超限；不把 429 改写为 400 或 500；如原厂提供 Retry-After，则字段语义一致 |
| `K3-ERR-001` | 认证与错误 | P0 | 非法参数校验 | `{"stream":false,"messages":[{"role":"user","content":"只回复 OK"}],"temperature":2.1,"max_tokens":16}` | HTTP 400；错误格式和字段语义与原厂一致；不静默修改 temperature 后返回 200 |
| `K3-ERR-002` | 认证与错误 | P0 | 非法 messages 类型 | `{"stream":false,"messages":"not-an-array","max_tokens":16}` | HTTP 400；错误格式可解析；错误指向参数校验而非服务端异常 |
| `K3-THINK-001` | Thinking 模式 | P0 | Thinking 官方显式形态 | `{"stream":false,"thinking":{"type":"enabled","keep":"all"},"messages":[{"role":"user","content":"计算 918273×817263，并给出简要过程"}],"max_tokens":1024}` | HTTP 200；产生 reasoning/thinking 内容或原厂对应字段；正文与思考内容字段分离且格式稳定；usage 完整 |
| `K3-THINK-002` | Thinking 模式 | P0 | Thinking 不可关闭（观察项） | `{"stream":false,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"只回复 DISABLED-OK"}],"max_tokens":512}` | HTTP 400 拒绝（该形态不是 K3 契约），或 HTTP 200 但思考仍产生；K3 思考关不掉：200 且 reasoning 为空即视为未达标；若 200，正文与思考内容字段分离且格式稳定；usage 完整 |
| `K3-PARAM-001` | 采样参数 | P0 | 采样参数固定值正向 | `{"stream":false,"messages":[{"role":"user","content":"只回复 PARAM-OK"}],"temperature":1.0,"top_p":0.95,"max_tokens":512}` | HTTP 200；显式传入原厂固定值 temperature=1、top_p=0.95 被接受，不因传参报错；正文为 PARAM-OK；usage 完整且可计量 |
| `K3-PARAM-002` | 采样参数 | P0 | 采样参数非固定值拒绝 | `{"stream":false,"messages":[{"role":"user","content":"连续输出编号 1 到 20，每行一个"}],"temperature":0.6,"top_p":0.8,"max_tokens":512}` | HTTP 400；temperature/top_p 取值域被锁为固定值单点：合法区间内的非固定值（temperature=0.6）同样被拒；错误语义指向 temperature 或 top_p，不静默改写后返回 200；错误格式可解析且与原厂一致 |
| `K3-STREAM-001` | 流式与计量 | P0 | SSE chunk 字段完整性 | `{"stream":true,"messages":[{"role":"user","content":"计算 37×83，并简要说明"}],"max_tokens":512}` | chunk id/model/object 字段符合原厂；choices、delta、index 结构可解析；正文 delta 与 reasoning delta 不串字段；结束 chunk 的 finish_reason 正确；[DONE] 只出现一次且在最后 |
| `K3-STREAM-002` | 流式与计量 | P0 | 流式 usage 计量 | `{"stream":true,"messages":[{"role":"user","content":"请输出 100 字以内的测试说明"}],"max_tokens":512}` | 流式结束阶段返回 usage 或原厂等价计量方式；prompt_tokens、completion_tokens、cached_tokens 均存在且为非负数（cached_tokens 可位于 usage.prompt_tokens_details）；total_tokens 与原厂计量规则一致；流式拼接正文与非流式正文语义一致 |
| `K3-JSON-001` | 结构化输出 | P0 | JSON mode | `{"stream":false,"response_format":{"type":"json_object"},"messages":[{"role":"user","content":"返回 JSON：包含 name=K3、passed=true"}],"max_tokens":512}` | HTTP 200；message.content 为合法 JSON object；JSON 可被标准解析器解析；无 Markdown 代码围栏；usage 完整 |
| `K3-JSON-002` | 结构化输出 | P0 | response_format 参数校验 | `{"stream":false,"response_format":{"type":"unsupported_type"},"messages":[{"role":"user","content":"只回复 OK"}],"max_tokens":16}` | HTTP 400；错误语义为不支持的 response_format；不降级为普通文本成功响应 |
| `K3-TOOL-001` | 工具调用 | P0 | 单工具调用闭环 | `{"stream":false,"tools":[{"type":"function","function":{"name":"lookup_order","description":"查询订单","parameters":{"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"],"additionalProperties":false}}}],"tool_choice":"required","messages":[{"role":"user","content":"查询订单 ORD-000731，必须调用工具"}],"max_tokens":1024}` | HTTP 200；返回 tool_calls；工具名正确；arguments 为合法 JSON；参数含 order_id=ORD-000731；finish_reason=tool_calls 或原厂等价语义 |
| `K3-TOOL-002` | 工具调用 | P0 | 多工具调用 | `{"stream":false,"scenario":"一次请求声明 lookup_order 与 get_logistics 两个工具并要求同时查询"}` | HTTP 200；返回两个可解析 tool_calls；每个 name 与 arguments 一一对应；tool_call_id 唯一且格式稳定；顺序与原厂一致 |
| `K3-TOOL-003` | 工具调用 | P0 | 并行工具调用与工具结果回传 | `{"stream":false,"scenario":"助手 tool_calls 后追加两条 role=tool 消息，再请求最终答复"}` | 支持并行 tool calls；能接受对应 tool_call_id 的 tool 消息；最终返回正文而非再次无故调用工具；多轮消息结构与原厂一致 |
| `K3-OBS-001` | 可观测性 | P0 | request_id 追踪 | `{"stream":false,"headers":{"X-Request-ID":"client-trace-001"},"messages":[{"role":"user","content":"只回复 TRACE-OK"}],"max_tokens":512}` | 响应包含 request_id，或响应头包含可追溯请求标识；标识非空且单次请求唯一；日志/服务端可使用该标识定位请求；客户端传入标识时遵循原厂回显/关联规则 |
| `K3-TTFT-01` | TTFT | P0 | 16K 输入 TTFT 第 1 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":1,"batch":"16K-10"}` | 记录第 1 次首个有效输出 delta 到请求发出的耗时；输入长度命中 16K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 9703ms、P90 ≤ 18000ms（10 次完成后统一判定） |
| `K3-TTFT-02` | TTFT | P0 | 16K 输入 TTFT 第 2 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":2,"batch":"16K-10"}` | 记录第 2 次首个有效输出 delta 到请求发出的耗时；输入长度命中 16K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 9703ms、P90 ≤ 18000ms（10 次完成后统一判定） |
| `K3-TTFT-03` | TTFT | P0 | 16K 输入 TTFT 第 3 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":3,"batch":"16K-10"}` | 记录第 3 次首个有效输出 delta 到请求发出的耗时；输入长度命中 16K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 9703ms、P90 ≤ 18000ms（10 次完成后统一判定） |
| `K3-TTFT-04` | TTFT | P0 | 16K 输入 TTFT 第 4 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":4,"batch":"16K-10"}` | 记录第 4 次首个有效输出 delta 到请求发出的耗时；输入长度命中 16K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 9703ms、P90 ≤ 18000ms（10 次完成后统一判定） |
| `K3-TTFT-05` | TTFT | P0 | 16K 输入 TTFT 第 5 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":5,"batch":"16K-10"}` | 记录第 5 次首个有效输出 delta 到请求发出的耗时；输入长度命中 16K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 9703ms、P90 ≤ 18000ms（10 次完成后统一判定） |
| `K3-TTFT-06` | TTFT | P0 | 16K 输入 TTFT 第 6 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":6,"batch":"16K-10"}` | 记录第 6 次首个有效输出 delta 到请求发出的耗时；输入长度命中 16K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 9703ms、P90 ≤ 18000ms（10 次完成后统一判定） |
| `K3-TTFT-07` | TTFT | P0 | 16K 输入 TTFT 第 7 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":7,"batch":"16K-10"}` | 记录第 7 次首个有效输出 delta 到请求发出的耗时；输入长度命中 16K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 9703ms、P90 ≤ 18000ms（10 次完成后统一判定） |
| `K3-TTFT-08` | TTFT | P0 | 16K 输入 TTFT 第 8 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":8,"batch":"16K-10"}` | 记录第 8 次首个有效输出 delta 到请求发出的耗时；输入长度命中 16K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 9703ms、P90 ≤ 18000ms（10 次完成后统一判定） |
| `K3-TTFT-09` | TTFT | P0 | 16K 输入 TTFT 第 9 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":9,"batch":"16K-10"}` | 记录第 9 次首个有效输出 delta 到请求发出的耗时；输入长度命中 16K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 9703ms、P90 ≤ 18000ms（10 次完成后统一判定） |
| `K3-TTFT-10` | TTFT | P0 | 16K 输入 TTFT 第 10 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":10,"batch":"16K-10"}` | 记录第 10 次首个有效输出 delta 到请求发出的耗时；输入长度命中 16K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 9703ms、P90 ≤ 18000ms（10 次完成后统一判定） |
| `K3-TTFT-11` | TTFT | P0 | 64K 输入 TTFT 第 1 次 | `{"stream":true,"input_tokens_target":64000,"max_tokens":512,"concurrency":10,"repetition":1,"batch":"64K-10"}` | 记录第 1 次首个有效输出 delta 到请求发出的耗时；输入长度命中 64K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8303ms、P90 ≤ 15000ms（10 次完成后统一判定） |
| `K3-TTFT-12` | TTFT | P0 | 64K 输入 TTFT 第 2 次 | `{"stream":true,"input_tokens_target":64000,"max_tokens":512,"concurrency":10,"repetition":2,"batch":"64K-10"}` | 记录第 2 次首个有效输出 delta 到请求发出的耗时；输入长度命中 64K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8303ms、P90 ≤ 15000ms（10 次完成后统一判定） |
| `K3-TTFT-13` | TTFT | P0 | 64K 输入 TTFT 第 3 次 | `{"stream":true,"input_tokens_target":64000,"max_tokens":512,"concurrency":10,"repetition":3,"batch":"64K-10"}` | 记录第 3 次首个有效输出 delta 到请求发出的耗时；输入长度命中 64K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8303ms、P90 ≤ 15000ms（10 次完成后统一判定） |
| `K3-TTFT-14` | TTFT | P0 | 64K 输入 TTFT 第 4 次 | `{"stream":true,"input_tokens_target":64000,"max_tokens":512,"concurrency":10,"repetition":4,"batch":"64K-10"}` | 记录第 4 次首个有效输出 delta 到请求发出的耗时；输入长度命中 64K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8303ms、P90 ≤ 15000ms（10 次完成后统一判定） |
| `K3-TTFT-15` | TTFT | P0 | 64K 输入 TTFT 第 5 次 | `{"stream":true,"input_tokens_target":64000,"max_tokens":512,"concurrency":10,"repetition":5,"batch":"64K-10"}` | 记录第 5 次首个有效输出 delta 到请求发出的耗时；输入长度命中 64K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8303ms、P90 ≤ 15000ms（10 次完成后统一判定） |
| `K3-TTFT-16` | TTFT | P0 | 64K 输入 TTFT 第 6 次 | `{"stream":true,"input_tokens_target":64000,"max_tokens":512,"concurrency":10,"repetition":6,"batch":"64K-10"}` | 记录第 6 次首个有效输出 delta 到请求发出的耗时；输入长度命中 64K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8303ms、P90 ≤ 15000ms（10 次完成后统一判定） |
| `K3-TTFT-17` | TTFT | P0 | 64K 输入 TTFT 第 7 次 | `{"stream":true,"input_tokens_target":64000,"max_tokens":512,"concurrency":10,"repetition":7,"batch":"64K-10"}` | 记录第 7 次首个有效输出 delta 到请求发出的耗时；输入长度命中 64K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8303ms、P90 ≤ 15000ms（10 次完成后统一判定） |
| `K3-TTFT-18` | TTFT | P0 | 64K 输入 TTFT 第 8 次 | `{"stream":true,"input_tokens_target":64000,"max_tokens":512,"concurrency":10,"repetition":8,"batch":"64K-10"}` | 记录第 8 次首个有效输出 delta 到请求发出的耗时；输入长度命中 64K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8303ms、P90 ≤ 15000ms（10 次完成后统一判定） |
| `K3-TTFT-19` | TTFT | P0 | 64K 输入 TTFT 第 9 次 | `{"stream":true,"input_tokens_target":64000,"max_tokens":512,"concurrency":10,"repetition":9,"batch":"64K-10"}` | 记录第 9 次首个有效输出 delta 到请求发出的耗时；输入长度命中 64K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8303ms、P90 ≤ 15000ms（10 次完成后统一判定） |
| `K3-TTFT-20` | TTFT | P0 | 64K 输入 TTFT 第 10 次 | `{"stream":true,"input_tokens_target":64000,"max_tokens":512,"concurrency":10,"repetition":10,"batch":"64K-10"}` | 记录第 10 次首个有效输出 delta 到请求发出的耗时；输入长度命中 64K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8303ms、P90 ≤ 15000ms（10 次完成后统一判定） |
| `K3-TTFT-21` | TTFT | P0 | 100K 输入 TTFT 第 1 次 | `{"stream":true,"input_tokens_target":100000,"max_tokens":512,"concurrency":10,"repetition":1,"batch":"100K-10"}` | 记录第 1 次首个有效输出 delta 到请求发出的耗时；输入长度命中 100K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8190ms、P90 ≤ 15750ms（10 次完成后统一判定） |
| `K3-TTFT-22` | TTFT | P0 | 100K 输入 TTFT 第 2 次 | `{"stream":true,"input_tokens_target":100000,"max_tokens":512,"concurrency":10,"repetition":2,"batch":"100K-10"}` | 记录第 2 次首个有效输出 delta 到请求发出的耗时；输入长度命中 100K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8190ms、P90 ≤ 15750ms（10 次完成后统一判定） |
| `K3-TTFT-23` | TTFT | P0 | 100K 输入 TTFT 第 3 次 | `{"stream":true,"input_tokens_target":100000,"max_tokens":512,"concurrency":10,"repetition":3,"batch":"100K-10"}` | 记录第 3 次首个有效输出 delta 到请求发出的耗时；输入长度命中 100K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8190ms、P90 ≤ 15750ms（10 次完成后统一判定） |
| `K3-TTFT-24` | TTFT | P0 | 100K 输入 TTFT 第 4 次 | `{"stream":true,"input_tokens_target":100000,"max_tokens":512,"concurrency":10,"repetition":4,"batch":"100K-10"}` | 记录第 4 次首个有效输出 delta 到请求发出的耗时；输入长度命中 100K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8190ms、P90 ≤ 15750ms（10 次完成后统一判定） |
| `K3-TTFT-25` | TTFT | P0 | 100K 输入 TTFT 第 5 次 | `{"stream":true,"input_tokens_target":100000,"max_tokens":512,"concurrency":10,"repetition":5,"batch":"100K-10"}` | 记录第 5 次首个有效输出 delta 到请求发出的耗时；输入长度命中 100K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8190ms、P90 ≤ 15750ms（10 次完成后统一判定） |
| `K3-TTFT-26` | TTFT | P0 | 100K 输入 TTFT 第 6 次 | `{"stream":true,"input_tokens_target":100000,"max_tokens":512,"concurrency":10,"repetition":6,"batch":"100K-10"}` | 记录第 6 次首个有效输出 delta 到请求发出的耗时；输入长度命中 100K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8190ms、P90 ≤ 15750ms（10 次完成后统一判定） |
| `K3-TTFT-27` | TTFT | P0 | 100K 输入 TTFT 第 7 次 | `{"stream":true,"input_tokens_target":100000,"max_tokens":512,"concurrency":10,"repetition":7,"batch":"100K-10"}` | 记录第 7 次首个有效输出 delta 到请求发出的耗时；输入长度命中 100K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8190ms、P90 ≤ 15750ms（10 次完成后统一判定） |
| `K3-TTFT-28` | TTFT | P0 | 100K 输入 TTFT 第 8 次 | `{"stream":true,"input_tokens_target":100000,"max_tokens":512,"concurrency":10,"repetition":8,"batch":"100K-10"}` | 记录第 8 次首个有效输出 delta 到请求发出的耗时；输入长度命中 100K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8190ms、P90 ≤ 15750ms（10 次完成后统一判定） |
| `K3-TTFT-29` | TTFT | P0 | 100K 输入 TTFT 第 9 次 | `{"stream":true,"input_tokens_target":100000,"max_tokens":512,"concurrency":10,"repetition":9,"batch":"100K-10"}` | 记录第 9 次首个有效输出 delta 到请求发出的耗时；输入长度命中 100K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8190ms、P90 ≤ 15750ms（10 次完成后统一判定） |
| `K3-TTFT-30` | TTFT | P0 | 100K 输入 TTFT 第 10 次 | `{"stream":true,"input_tokens_target":100000,"max_tokens":512,"concurrency":10,"repetition":10,"batch":"100K-10"}` | 记录第 10 次首个有效输出 delta 到请求发出的耗时；输入长度命中 100K 档；HTTP 200 且 SSE 正常结束；保存本次请求级明细；该档汇总 avg ≤ 8190ms、P90 ≤ 15750ms（10 次完成后统一判定） |
| `K3-TPOT-01` | TPOT | P0 | 流式生成 TPOT 第 1 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":1,"batch":"TPOT-10"}` | HTTP 200 且 SSE 正常结束；记录首 token 后相邻有效输出 token/delta 的间隔；排除首 token 等待时间计算 TPOT；保存本次请求级明细；10 次汇总需满足 avg ≤22ms/token、P50 ≤18ms/token、P90 ≤36ms/token |
| `K3-TPOT-02` | TPOT | P0 | 流式生成 TPOT 第 2 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":2,"batch":"TPOT-10"}` | HTTP 200 且 SSE 正常结束；记录首 token 后相邻有效输出 token/delta 的间隔；排除首 token 等待时间计算 TPOT；保存本次请求级明细；10 次汇总需满足 avg ≤22ms/token、P50 ≤18ms/token、P90 ≤36ms/token |
| `K3-TPOT-03` | TPOT | P0 | 流式生成 TPOT 第 3 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":3,"batch":"TPOT-10"}` | HTTP 200 且 SSE 正常结束；记录首 token 后相邻有效输出 token/delta 的间隔；排除首 token 等待时间计算 TPOT；保存本次请求级明细；10 次汇总需满足 avg ≤22ms/token、P50 ≤18ms/token、P90 ≤36ms/token |
| `K3-TPOT-04` | TPOT | P0 | 流式生成 TPOT 第 4 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":4,"batch":"TPOT-10"}` | HTTP 200 且 SSE 正常结束；记录首 token 后相邻有效输出 token/delta 的间隔；排除首 token 等待时间计算 TPOT；保存本次请求级明细；10 次汇总需满足 avg ≤22ms/token、P50 ≤18ms/token、P90 ≤36ms/token |
| `K3-TPOT-05` | TPOT | P0 | 流式生成 TPOT 第 5 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":5,"batch":"TPOT-10"}` | HTTP 200 且 SSE 正常结束；记录首 token 后相邻有效输出 token/delta 的间隔；排除首 token 等待时间计算 TPOT；保存本次请求级明细；10 次汇总需满足 avg ≤22ms/token、P50 ≤18ms/token、P90 ≤36ms/token |
| `K3-TPOT-06` | TPOT | P0 | 流式生成 TPOT 第 6 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":6,"batch":"TPOT-10"}` | HTTP 200 且 SSE 正常结束；记录首 token 后相邻有效输出 token/delta 的间隔；排除首 token 等待时间计算 TPOT；保存本次请求级明细；10 次汇总需满足 avg ≤22ms/token、P50 ≤18ms/token、P90 ≤36ms/token |
| `K3-TPOT-07` | TPOT | P0 | 流式生成 TPOT 第 7 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":7,"batch":"TPOT-10"}` | HTTP 200 且 SSE 正常结束；记录首 token 后相邻有效输出 token/delta 的间隔；排除首 token 等待时间计算 TPOT；保存本次请求级明细；10 次汇总需满足 avg ≤22ms/token、P50 ≤18ms/token、P90 ≤36ms/token |
| `K3-TPOT-08` | TPOT | P0 | 流式生成 TPOT 第 8 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":8,"batch":"TPOT-10"}` | HTTP 200 且 SSE 正常结束；记录首 token 后相邻有效输出 token/delta 的间隔；排除首 token 等待时间计算 TPOT；保存本次请求级明细；10 次汇总需满足 avg ≤22ms/token、P50 ≤18ms/token、P90 ≤36ms/token |
| `K3-TPOT-09` | TPOT | P0 | 流式生成 TPOT 第 9 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":9,"batch":"TPOT-10"}` | HTTP 200 且 SSE 正常结束；记录首 token 后相邻有效输出 token/delta 的间隔；排除首 token 等待时间计算 TPOT；保存本次请求级明细；10 次汇总需满足 avg ≤22ms/token、P50 ≤18ms/token、P90 ≤36ms/token |
| `K3-TPOT-10` | TPOT | P0 | 流式生成 TPOT 第 10 次 | `{"stream":true,"input_tokens_target":16000,"max_tokens":512,"concurrency":10,"repetition":10,"batch":"TPOT-10"}` | HTTP 200 且 SSE 正常结束；记录首 token 后相邻有效输出 token/delta 的间隔；排除首 token 等待时间计算 TPOT；保存本次请求级明细；10 次汇总需满足 avg ≤22ms/token、P50 ≤18ms/token、P90 ≤36ms/token |
| `K3-CACHE-HIT-01` | 缓存 | P0 | 90% 稳定前缀命中探测第 1 次 | `{"phase":"probe","wait_seconds":0,"warm_input_tokens_target":70000,"same_prefix_byte_exact":true,"append_tail":true,"repetition":1,"batch":"CACHE-HIT-7"}` | warm 请求已完成且稳定前缀 byte-exact 保持不变；记录每次 probe 的 prompt_tokens 与 cached_tokens（缺字段按未命中计）；保存本次 warm/probe 关联信息；该档命中率：cached_tokens / prompt_tokens ≥80% 的 probe 占比 ≥80%（7 次完成后统一判定） |
| `K3-CACHE-HIT-02` | 缓存 | P0 | 90% 稳定前缀命中探测第 2 次 | `{"phase":"probe","wait_seconds":0,"warm_input_tokens_target":70000,"same_prefix_byte_exact":true,"append_tail":true,"repetition":2,"batch":"CACHE-HIT-7"}` | warm 请求已完成且稳定前缀 byte-exact 保持不变；记录每次 probe 的 prompt_tokens 与 cached_tokens（缺字段按未命中计）；保存本次 warm/probe 关联信息；该档命中率：cached_tokens / prompt_tokens ≥80% 的 probe 占比 ≥80%（7 次完成后统一判定） |
| `K3-CACHE-HIT-03` | 缓存 | P0 | 90% 稳定前缀命中探测第 3 次 | `{"phase":"probe","wait_seconds":0,"warm_input_tokens_target":70000,"same_prefix_byte_exact":true,"append_tail":true,"repetition":3,"batch":"CACHE-HIT-7"}` | warm 请求已完成且稳定前缀 byte-exact 保持不变；记录每次 probe 的 prompt_tokens 与 cached_tokens（缺字段按未命中计）；保存本次 warm/probe 关联信息；该档命中率：cached_tokens / prompt_tokens ≥80% 的 probe 占比 ≥80%（7 次完成后统一判定） |
| `K3-CACHE-HIT-04` | 缓存 | P0 | 90% 稳定前缀命中探测第 4 次 | `{"phase":"probe","wait_seconds":0,"warm_input_tokens_target":70000,"same_prefix_byte_exact":true,"append_tail":true,"repetition":4,"batch":"CACHE-HIT-7"}` | warm 请求已完成且稳定前缀 byte-exact 保持不变；记录每次 probe 的 prompt_tokens 与 cached_tokens（缺字段按未命中计）；保存本次 warm/probe 关联信息；该档命中率：cached_tokens / prompt_tokens ≥80% 的 probe 占比 ≥80%（7 次完成后统一判定） |
| `K3-CACHE-HIT-05` | 缓存 | P0 | 90% 稳定前缀命中探测第 5 次 | `{"phase":"probe","wait_seconds":0,"warm_input_tokens_target":70000,"same_prefix_byte_exact":true,"append_tail":true,"repetition":5,"batch":"CACHE-HIT-7"}` | warm 请求已完成且稳定前缀 byte-exact 保持不变；记录每次 probe 的 prompt_tokens 与 cached_tokens（缺字段按未命中计）；保存本次 warm/probe 关联信息；该档命中率：cached_tokens / prompt_tokens ≥80% 的 probe 占比 ≥80%（7 次完成后统一判定） |
| `K3-CACHE-HIT-06` | 缓存 | P0 | 90% 稳定前缀命中探测第 6 次 | `{"phase":"probe","wait_seconds":0,"warm_input_tokens_target":70000,"same_prefix_byte_exact":true,"append_tail":true,"repetition":6,"batch":"CACHE-HIT-7"}` | warm 请求已完成且稳定前缀 byte-exact 保持不变；记录每次 probe 的 prompt_tokens 与 cached_tokens（缺字段按未命中计）；保存本次 warm/probe 关联信息；该档命中率：cached_tokens / prompt_tokens ≥80% 的 probe 占比 ≥80%（7 次完成后统一判定） |
| `K3-CACHE-HIT-07` | 缓存 | P0 | 90% 稳定前缀命中探测第 7 次 | `{"phase":"probe","wait_seconds":0,"warm_input_tokens_target":70000,"same_prefix_byte_exact":true,"append_tail":true,"repetition":7,"batch":"CACHE-HIT-7"}` | warm 请求已完成且稳定前缀 byte-exact 保持不变；记录每次 probe 的 prompt_tokens 与 cached_tokens（缺字段按未命中计）；保存本次 warm/probe 关联信息；该档命中率：cached_tokens / prompt_tokens ≥80% 的 probe 占比 ≥80%（7 次完成后统一判定） |
| `K3-CACHE-TTL-01` | 缓存 | P0 | 30 秒 TTL 命中探测第 1 次 | `{"phase":"probe","wait_seconds":30,"warm_input_tokens_target":70000,"same_prefix_byte_exact":true,"append_tail":true,"repetition":1,"batch":"CACHE-TTL-3"}` | warm 后精确等待 30 秒再发起 probe；稳定前缀 byte-exact 保持不变；保存时间戳、等待时长和 usage；该档命中率：30 秒后 cached_tokens / prompt_tokens ≥80% 的 probe 占比 ≥80%（3 次完成后统一判定） |
| `K3-CACHE-TTL-02` | 缓存 | P0 | 30 秒 TTL 命中探测第 2 次 | `{"phase":"probe","wait_seconds":30,"warm_input_tokens_target":70000,"same_prefix_byte_exact":true,"append_tail":true,"repetition":2,"batch":"CACHE-TTL-3"}` | warm 后精确等待 30 秒再发起 probe；稳定前缀 byte-exact 保持不变；保存时间戳、等待时长和 usage；该档命中率：30 秒后 cached_tokens / prompt_tokens ≥80% 的 probe 占比 ≥80%（3 次完成后统一判定） |
| `K3-CACHE-TTL-03` | 缓存 | P0 | 30 秒 TTL 命中探测第 3 次 | `{"phase":"probe","wait_seconds":30,"warm_input_tokens_target":70000,"same_prefix_byte_exact":true,"append_tail":true,"repetition":3,"batch":"CACHE-TTL-3"}` | warm 后精确等待 30 秒再发起 probe；稳定前缀 byte-exact 保持不变；保存时间戳、等待时长和 usage；该档命中率：30 秒后 cached_tokens / prompt_tokens ≥80% 的 probe 占比 ≥80%（3 次完成后统一判定） |

## 不纳入范围

- `arxivmath`、`DeepSWE`、`Swe-atlas-tw`、`Swe-atlas-rf`、`Legal Agent Benchmark` 等开源评测集。
- 额度和技术支持为人工确认项。
- 本套件为临时自测基线，不替代正式评测。
