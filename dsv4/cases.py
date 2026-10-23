from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSeekCase:
    id: str
    title: str
    description: str
    cost: str
    priorities: tuple[str, ...]
    request_count: int
    concurrency: int = 1
    protocols: tuple[str, ...] = ("openai", "volcengine")


CASES = {
    "DS_V4_P0": DeepSeekCase("DS_V4_P0", "DeepSeek V4 核心专项", "思考模式、reasoning 回传、工具调用、JSON、缓存、同身份 Session 连续性、user_id 隔离，以及多模态图片输入、模型别名口径、思考强度取值域、max_tokens 边界、鉴权契约与 Files API 闭环", "中0.03", ("P0",), 24),
    "DS_V4_P1": DeepSeekCase("DS_V4_P1", "DeepSeek V4 增强专项", "Strict Tool、Prefix/FIM、Anthropic 映射与 Responses API，以及图片 role 约束、模型列表、stream_options 契约、logprobs、top_p 边界、FIM 模式互斥、Anthropic system/stop/thinking、Responses 流式语义事件与不支持字段忽略、并行工具调用", "中", ("P1",), 18),
    "DS_V4_P2": DeepSeekCase("DS_V4_P2", "DeepSeek V4 契约与健壮性", "Usage、流式结束、Stop、参数边界、非法消息、Tool Choice、Responses 与 Anthropic 基础契约，以及图片 detail 档位、Files 列表检索、账户余额、stop 多序列上限、废弃采样参数、FIM+JSON 互斥、prefix 标志缺失、Anthropic 工具闭环、Responses 工具矩阵", "中", ("P2",), 19),
    "DS_V4_FINGERPRINT": DeepSeekCase("DS_V4_FINGERPRINT", "DeepSeek V4 满血真实性指纹", "随机挑战 Thinking、动态工具、JSON Schema、流式和静默吞参，识别疑似降级或中转特征", "中", ("AUTH",), 6),
}
