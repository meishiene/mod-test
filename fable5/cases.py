"""Fable 5 专项测试的展示元数据（唯一来源：docs/Fable5专项测试用例设计.md）。

用例实现与判定在 fable5/core.py，装配在 fable5/runner.py；本文件只提供
Web 界面展示所需的标题/描述/成本/优先级/协议，不含任何请求逻辑。

用例集按**能力维度**划分，与 core.py 的 CASE_SPECS 优先级一一对应：
BASIC 基础能力 / P0 增强能力 / P1 专项能力（beta）/ P2 契约健壮性 / AUTH 满血真实性指纹。
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Fable5Case:
    id: str
    title: str
    description: str
    cost: str
    priorities: tuple[str, ...]
    request_count: int
    concurrency: int = 1
    protocols: tuple[str, ...] = ("anthropic",)


CASES = {
    "FABLE5_BASIC": Fable5Case(
        "FABLE5_BASIC", "Fable 5 基础能力",
        "身份与基础问答、严格指令遵循、多位数乘法、中文材料推断、代码生成（实际执行验证）、代码缺陷定位、"
        "约束推理唯一解、长文结构化生成、摘要保真、翻译保真、纯文本 JSON、多轮指令保持、事实性知识、"
        "列举完整性、安全拒答边界",
        "低（短提示为主，单项一次请求）", ("BASIC",), 15, 1, ("anthropic",),
    ),
    "FABLE5_P0": Fable5Case(
        "FABLE5_P0", "Fable 5 增强能力",
        "Messages API 自适应思考生命周期（adaptive 接受/省略/disabled 留痕/enabled+budget 留痕）、effort 五档与默认档、"
        "工具闭环与 tool_choice、结构化输出 schema 强绑定、并行工具调用、响应对象结构契约、流式协议（含 thinking_delta "
        "与无 [DONE] 收尾）、视觉输入（token 双判据）、长上下文接受性、隐式前缀缓存、stop_sequences、system 指令、"
        "max_tokens 截断可检测与大 max_tokens 接受性",
        "中～高（长上下文与缓存用例输入 token 较多）", ("P0",), 20, 1, ("anthropic",),
    ),
    "FABLE5_P1": Fable5Case(
        "FABLE5_P1", "Fable 5 专项能力（beta）",
        "task budget beta、上下文编辑（工具结果清理）beta、压缩（compaction）beta、memory 工具、代码执行工具、"
        "程序化工具调用（PTC）、服务端 fallback beta、1 小时缓存 TTL、数据驻留参数（inference_geo）、"
        "reasoning_extraction 类拒答语义、受限预算连续输出（200项）",
        "中（连续输出用例上限4096 tokens，其余beta专项按各自预算）", ("P1",), 11, 1, ("anthropic",),
    ),
    "FABLE5_P2": Fable5Case(
        "FABLE5_P2", "Fable 5 契约与健壮性",
        "无效 Key 鉴权契约、错误对象结构契约、已移除的采样参数（temperature/top_p/top_k）、assistant prefill、"
        "max_tokens 必填与越界、messages 空数组/非法角色、未知模型拒绝、/v1/models 列表、count_tokens 交叉验证、"
        "Usage 字段完整性、流式 usage 收尾、stop_reason 词表、metadata 透传、畸形 JSON 请求体、未知顶层字段、"
        "缺失 anthropic-version 头、响应对象结构契约",
        "中", ("P2",), 21, 1, ("anthropic",),
    ),
    "FABLE5_FINGERPRINT": Fable5Case(
        "FABLE5_FINGERPRINT", "Fable 5 满血真实性指纹",
        "随机挑战动态工具、结构化输出绑定、多轮记忆、思考真实性、effort 吞参对照、原始思维链不得泄漏、"
        "模型身份未被静默降级、同族别名路由身份、中转注入识别（system 回显 / token 膨胀）、"
        "中转零注入验证（侧信道自述系统指令 / 未声明工具被调用即注入 / 零值 system 差分）、"
        "非标准字段与协议改写检测、流式事件真实性",
        "中～高（effort 双档对照、零注入侧信道与流式组合）", ("AUTH",), 14, 1, ("anthropic",),
    ),
}
