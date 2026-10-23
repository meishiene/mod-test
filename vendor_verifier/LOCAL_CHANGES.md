# 官方快照与本地差异

来源：[MoonshotAI/Kimi-Vendor-Verifier](https://github.com/MoonshotAI/Kimi-Vendor-Verifier)，
固定提交 `66092cf444c97356c0e11c5078c67116390615d9`（2026-09-17）。2026-10-22 核对时为官方 main 最新提交。
版本及数据 SHA256 记录在 `UPSTREAM.json`。

本次补齐原快照遗漏的官方说明、许可证、依赖锁文件、重试脚本和 Prompt Token 数据。
测试 Python 文件已对照该提交；保留以下已有本地补丁，没有将其冒充为官方原始断言：

- `tests/params/test_params.py`、`verify_params.py`：K2/K3 参数契约选择；平台默认 K3，支持 `KVV_PARAM_CONTRACT=k2`。
- `tests/k3_features/test_response_format.py`：两个原始请求不再传 `thinking=disabled`，其 token 检查仅要求正整数；严格 token 基线由独立 Prompt Token 套件覆盖。

Prompt Token 的 44 个文本样本、11 个视觉样本和 PNG 原图均从官方 Git LFS 对象下载，
按 LFS 指针的大小和 SHA256 校验，直接存入本仓库，部署时无需 Git LFS。
`cases.partial.jsonl` 同步保留，但官方 pytest 默认不运行这些 partial-prefill 样本。
官方代码对 prompt tokens 接受 `[expected, expected + 3]`；上游 README 中“精确相等”的旧描述不代表当前断言。

官方 skip/flaky 标记保持不变；跳过项在报告中单列，不视作通过。平台要求每个所选套件至少一个测试通过，且无失败/错误。
新增动态工具、Thinking Effort、Prompt Token 三个独立入口，原核心入口保持兼容。

`.gitignore` 保留本地环境规则，但不再忽略运行所需的 Prompt Token 数据。
不引入官方 `.gitattributes`（本仓库直接保存上述小型测试数据），不包含平台 pytest 不使用的 BEAM 大型数据 `beam/data/`。
上游 benchmark 说明与脚本保留；这些 benchmark 不是本次新增的平台入口。

后续同步：拉取官方固定提交，逐文件比较；显式保留或复核上述三个补丁；
下载并校验 LFS 数据后更新 `UPSTREAM.json`，运行平台 KVV/Web 回归和官方 `pytest tests --collect-only`，禁止以 LFS 指针代替测试数据。
