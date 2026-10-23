"""Gemini 3.x 专项测试的展示元数据（唯一来源：docs/Gemini3专项测试用例设计.md）。

用例实现与判定在 ``gemini3/core.py``，装配在 ``gemini3/runner.py``；本文件只提供
Web 界面展示所需的标题/描述/成本/优先级/协议，不含任何请求逻辑。

与 Claude（``fable5``）/ GPT（``gpt56``）两个模块的差别：
- 两个模型（3.8 Flash / 3.1 Pro）**共用同一批用例**，档位相关期望值由 ``core.py`` 的
  ``GeminiTier`` 能力矩阵按被测模型解析（同 GPT-6 家族的做法），因此只有一个档位
  ``gemini3``，不按模型拆套件；
- 调用面是 Gemini 原生 GenerateContent（``{base}/v1beta/models/{model}:generateContent``
  与 ``:streamGenerateContent?alt=sse``），协议 family 为 ``gemini``；SSE 无 ``[DONE]``，
  与 Responses / Messages / Chat 三种协议都不同。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gemini3Case:
    id: str
    title: str
    description: str
    cost: str
    priorities: tuple[str, ...]
    request_count: int
    concurrency: int = 1
    protocols: tuple[str, ...] = ("gemini",)
    # 档位：``gemini3``（3.8 Flash + 3.1 Pro 共用，矩阵按被测模型解析）。
    model_variant: str = "gemini3"


CASES = {
    "GEMINI3_BASIC": Gemini3Case(
        "GEMINI3_BASIC", "Gemini 3.x 基础能力",
        "模型身份（以响应 modelVersion 为锚点）、严格指令遵循、多位数乘法、中文材料推断、代码生成（实际执行验证）、"
        "长文结构化生成、原生结构化输出（responseMimeType+responseSchema）、多轮上下文保持（contents 角色 user/model）、"
        "事实性与列举完整性、安全拒答边界（finishReason=SAFETY / promptFeedback.blockReason / 正文拒答三判据）",
        "低（短提示为主，单项一次请求）", ("BASIC",), 11,
    ),
    "GEMINI3_P0": Gemini3Case(
        "GEMINI3_P0", "Gemini 3.x 核心专项",
        "thinkingConfig.thinkingLevel 合法档位逐档接受、省略参数时的默认档位（3.8 Flash=MEDIUM / 3.1 Pro=HIGH）、"
        "minimal 档位下限双向判定、非法档位枚举校验与合法集取证、已弃用 thinkingBudget 的行为、"
        "思考摘要回传（includeThoughts）、函数调用闭环（id/name 严格匹配 + 原样回传签名）、"
        "toolConfig mode=ANY 强制调用、三个内置工具逐项注册（googleSearch/urlContext/codeExecution）、"
        "结构化输出与内置工具组合、流式协议完整性（SSE 无 [DONE]）、视觉输入（识别 + token 双判据）、"
        "mediaResolution、长上下文接受性、隐式前缀缓存、maxOutputTokens 截断可检测、stopSequences、systemInstruction",
        "中～高（长上下文与缓存用例输入 token 较多）", ("P0",), 28,
    ),
    "GEMINI3_P1": Gemini3Case(
        "GEMINI3_P1", "Gemini 3.x 专项能力",
        "思考签名多轮推理上下文、多模态函数响应（payload 内联图片）、一次请求多个函数调用、"
        "多候选 candidateCount 按档位双向（3.8 Flash 报错 / 3.1 Pro 支持）、安全阈值可配置（safetySettings）、"
        "长输出生成（流式）、显式缓存资源（cachedContents）、原生音频/视频输入接受性、"
        "思考 token 计入输出上限、流式与非流式结果等价性",
        "中～高（长输出与显式缓存为多跳）", ("P1",), 13,
    ),
    "GEMINI3_P2": Gemini3Case(
        "GEMINI3_P2", "Gemini 3.x 契约与健壮性",
        "无效 API Key 鉴权契约、错误对象结构契约（Google error.code/status/message）、"
        "已弃用采样参数被忽略（temperature/topP/topK）、frequency_penalty 与 presence_penalty 按档位双向、"
        "contents 为空/角色非法应被拒、未知模型拒绝、/models 列表、:countTokens 交叉验证、"
        "usageMetadata 字段完整性、响应对象结构契约、finishReason 词表、畸形 JSON 请求体、"
        "未知顶层字段的处理、maxOutputTokens 越界",
        "中", ("P2",), 17,
    ),
    "GEMINI3_FINGERPRINT": Gemini3Case(
        "GEMINI3_FINGERPRINT", "Gemini 3.x 满血真实性指纹",
        "随机挑战：随机动态工具（AUTO 诱导 + ANY 兜底）、随机结构化输出绑定、随机多轮记忆；"
        "档位面：思考真实性（thoughtsTokenCount / 思考摘要）、low↔high 档位梯度对照（逐指标同量纲）、"
        "minimal 档位下限反向指纹（「该拒未拒」先探网关声明的合法档位集，与模型口径不符即记 blocked）、"
        "网关档位白名单探测；身份面：modelVersion 未被静默降级；通道面：中转注入识别（systemInstruction 回显 / "
        "promptTokenCount 膨胀）、非标准字段与协议改写检测、流式事件真实性、思考签名真实性",
        "中～高（档位梯度对照 + 随机挑战 + 网关口径探测）", ("AUTH",), 16,
    ),
}
