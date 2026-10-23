from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class GLMCase:
    id: str
    title: str
    description: str
    cost: str
    priorities: tuple[str, ...]
    request_count: int
    concurrency: int = 1
    protocols: tuple[str, ...] = ("openai", "volcengine")
    model_scope: bool = True
    admission: str = "已测能力逐项判定；真实性仅为行为证据，满血结论须检查必测证据完整性。"

CASES = {
    "GLM_53_CORE": GLMCase("GLM_53_CORE", "GLM 5.3 核心能力", "基础对话、流式输出（必须开启深度思考）、结构化输出、长上下文与 Usage", "中", ("P0",), 9),
    "GLM_53_AGENT": GLMCase("GLM_53_AGENT", "GLM 5.3 Agent 能力", "工具调用、tool_choice、工具流式输出、多轮工具链与 Schema", "中", ("P1",), 5),
    "GLM_53_CACHE": GLMCase("GLM_53_CACHE", "GLM 5.3 缓存与 Session 专项", "数学、代码、中文业务三组固定前缀；每组种子请求后执行两次命中探测，并验证同身份多轮 Session 连续性", "中", ("PCACHE",), 12, concurrency=1),
    "GLM_53_FINGERPRINT": GLMCase("GLM_53_FINGERPRINT", "GLM 5.3 接口能力指纹", "记录思考、Schema、流式、上下文和型号一致性；接口指纹相容不等于真实权重或满血证明", "中", ("P2",), 8),
}

# Explicit model registry; FlashX is not matched by substring against Flash.
from common.model_validation import PROFILES
GLM_MODELS = tuple(p for p in PROFILES.values() if p.family == 'glm')
GLM_MODEL_IDS = tuple(p.model_id for p in GLM_MODELS)
GLM_DEFAULT_MODEL = 'glm-5.3'

CASES.update({
    'GLM_53_CAPABILITY': GLMCase(
        'GLM_53_CAPABILITY', 'GLM 5.3 家族客观能力校验',
        '5.3 / Flash / FlashX 共用数学逻辑、代码隔离执行、中文嵌套抽取、头中尾召回、工具恢复和双工具参数题；'
        '旗舰执行5道推理题与64K字符探针，Flash/FlashX执行3道与32K字符探针。测试预算不代表官方上下文上限。',
        '中', ('PQUAL',), 12),
    'GLM_53_VERIFY': GLMCase(
        'GLM_53_VERIFY', 'GLM 5.3 家族真实性与路由校验',
        '随机挑战、精确区分5.3/Flash/FlashX、不存在型号负对照、系统指令优先级；不以自述证明真实权重身份。',
        '低', ('PVERIFY',), 6),
    'GLM_53_FULL': GLMCase(
        'GLM_53_FULL', 'GLM 5.3 家族综合能力与满血证据校验',
        '核心、思考、工具、客观能力与真实性一并校验，必测项缺失或阻断不能通过；缓存另卡验证。'
        '可通过CLI提供同型号同版本同种子参考报告；结果限定在已测范围，不能证明厂商全部能力。',
        '高（含长文本）', ('P0','P1','P2','PQUAL','PVERIFY'), 40),
})
