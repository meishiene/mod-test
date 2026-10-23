"""千问（Qwen 3.6/3.7/3.8）专项测试卡片与型号矩阵。

**判定口径的取舍**（与 GLM 5.3 套件同一套原则）：
- 信任级问题（响应模型被换成别族、错误信封外泄内部信息）才判 failed；
- 能力级问题（思考预算未严格生效、参数被静默忽略、视觉输入不被接受）只记
  warning / blocked —— 中转网关与官方端点在这类参数上的支持面本来就不同，
  把「端点没实现」读成「模型不行」会得出错误的准入结论；
- 档位自述（模型说自己属于哪个代际/档位）只留痕，永远不参与硬判定：
  模型自述版本长期不可靠，同族不同档位（flash/plus/max）的能力差异本身也不是缺陷。

型号矩阵里的上下文窗口与输出上限是**参考值**（用于界面提示与自述区间对照），
不构成硬期望——实测以真实长上下文探针为准。
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class QwenModel:
    model_id: str
    label: str
    generation: str          # 3.6 / 3.7 / 3.8
    tier: str                # flash / plus / max
    context_window: int      # 参考值，仅用于提示与自述区间对照
    max_output: int          # 参考值
    thinking_default: bool = True   # 3.6/3.7/3.8 系列默认开启思考
    # 仓库维护的型号能力声明（不是上游身份凭证）：False = 该型号本身不提供图像输入，视觉卡片判「不适用」，
    # 而不是「网关能力缺口」。取值以实测与厂商声明为准，视觉探针只做交叉验证。
    image_input: bool = True


QWEN_MODELS: tuple[QwenModel, ...] = (
    QwenModel("qwen3.8-max", "Qwen3.8 Max（旗舰）", "3.8", "max", 983_616, 131_072),
    QwenModel("qwen3.8-flash", "Qwen3.8 Flash（轻量）", "3.8", "flash", 262_144, 65_536),
    # 3.7-max 是纯文本旗舰：发图片内容块必 400（Unexpected item type in content），属设计如此。
    QwenModel("qwen3.7-max", "Qwen3.7 Max（旗舰 · 纯文本无视觉）", "3.7", "max", 262_144, 65_536,
              image_input=False),
    QwenModel("qwen3.7-plus", "Qwen3.7 Plus（均衡）", "3.7", "plus", 262_144, 65_536),
    # 3.6-plus 实测三探针 3/3 读图正确（2026-10-21），按能力可用来算，不能因同代 max 无视觉就类推。
    QwenModel("qwen3.6-plus", "Qwen3.6 Plus（均衡）", "3.6", "plus", 262_144, 65_536),
    QwenModel("qwen3.6-flash", "Qwen3.6 Flash（轻量）", "3.6", "flash", 262_144, 65_536),
)
QWEN_MODEL_IDS: tuple[str, ...] = tuple(model.model_id for model in QWEN_MODELS)
QWEN_DEFAULT_MODEL = "qwen3.8-max"

_BY_ID = {model.model_id: model for model in QWEN_MODELS}


def model_spec(model_id: str) -> QwenModel | None:
    """按型号取矩阵条目；网关别名（如 ``qwen/qwen3.8-max``）也能归一。"""
    value = str(model_id or "").strip().lower()
    if value in _BY_ID:
        return _BY_ID[value]
    tail = value.rsplit("/", 1)[-1]
    return _BY_ID.get(tail)


@dataclass(frozen=True)
class QwenCase:
    id: str
    title: str
    description: str
    cost: str
    priorities: tuple[str, ...]
    request_count: int
    concurrency: int = 1
    # protocols 是**协议家族**（不是协议 id）：千问走 OpenAI Chat Completions 线格式，
    # 因此官方 qwen 协议、openai、volcengine、qianfan 都能选中同一批用例。
    protocols: tuple[str, ...] = ("openai",)
    # model_scope：卡片下方渲染「执行范围」多选（选项来自「查询模型」），
    # 多选时服务端按模型拆子任务、聚合报告每个模型一行——千问六款同族不同档，
    # 这个能力用得上（档位与视觉卡片开放，其余保持单选语义）。
    model_scope: bool = False
    # 卡片上的准入标准文案；留空时由 web/server.py 按套件默认口径生成。
    admission: str = ""


PRIORITY_THINKING = "PTHINK"
PRIORITY_TIER = "PTIER"
PRIORITY_VISION = "PVISION"

CASES: dict[str, QwenCase] = {
    "QWEN_CORE": QwenCase(
        "QWEN_CORE", "Qwen 核心对话与协议",
        "关闭思考后的基础能力面：对话与模型回显、Usage 完整性、流式结束事件、JSON 输出、"
        "多轮上下文、长上下文稳定响应，以及 temperature / max_completion_tokens 的取值域契约。"
        "所有用例显式发送 enable_thinking=false——混合思考模型默认会思考，"
        "不关闭会把输出预算耗在推理上，读成「零正文」的假失败。",
        "中", ("P0",), 9, concurrency=1,
        admission="中转接入口径：核心面（本卡 8 条）是唯一阻断面——出现 failed/error 即不准入；"
                  "参数被端点拒绝（4xx）记 blocked、被静默忽略（200 但行为不符）记 warning，两者只记录。",
    ),
    "QWEN_THINKING": QwenCase(
        "QWEN_THINKING", "Qwen 思考模式（enable_thinking / thinking_budget）",
        "混合思考模型的核心开关：① 默认开启（不传参数应产生推理内容）；② enable_thinking=false "
        "应关闭推理；③ thinking_budget 应限制推理 token 数；④ 流式 delta.reasoning_content 与 "
        "usage.reasoning_tokens 一致；⑤ 思考模式下小输出预算及 length 截断契约；"
        "⑥ 非流式推理内容载体（reasoning_content / reasoning_details）。",
        "中", (PRIORITY_THINKING,), 6, concurrency=1,
        admission="中转接入口径：思考参数属高级面——端点不认参数（4xx）记 blocked、参数未生效或"
                  "usage 缺 reasoning_tokens 记 warning、乃至 failed，都只记 relay_gap 不阻断准入。",
    ),
    "QWEN_AGENT": QwenCase(
        "QWEN_AGENT", "Qwen Agent 与工具调用",
        "工具调用基础链路、tool_choice=required 强制调用、工具结果多轮回传、工具流式输出、"
        "以及一次返回多个调用的并行工具面。",
        "中", ("P1",), 6, concurrency=1,
        admission="中转接入口径（唯一权威 qwen/relay_profile.py）：工具面属高级面，"
                  "tool_choice=required 未被强制只记 relay_gap 不阻断准入；"
                  "工具被端点拒绝（4xx/不支持报错）记 blocked 只记录。"
                  "只有基础对话面失败才阻断。",
    ),
    "QWEN_TIER": QwenCase(
        "QWEN_TIER", "Qwen 档位与身份指纹",
        "识别「同族不同档」与「疑似替换」：① 响应 model 回显一致性（异族替换=信任级）；"
        "② 档位/代际自述（只留痕）；③ 上下文窗口自述与真实长输入；④ 采样参数取值域契约；"
        "⑤ 错误信封（畸形 JSON / 不存在的模型 / 不存在的路径）；⑥ 指纹汇总。"
        "开放多模型执行范围：六款型号同批对照，每款一份结论。",
        "低", (PRIORITY_TIER,), 10, concurrency=1, model_scope=True,
        admission="中转接入口径：本卡唯一阻断项是响应模型明显异族（非 qwen 族且与请求不符，属信任级）；"
                  "自述缺失/档位不符/上下文自述偏低/参数被拒/错误信封泄露一律只留痕，不阻断准入。",
    ),
    "QWEN_VISION": QwenCase(
        "QWEN_VISION", "Qwen 图像内容理解",
        "五级看图答题（夹具内嵌生成、结果确定）：① 纯色识别；② 8×8 棋盘数深色格（计数 32）；"
        "③ 图文冲突；④ 随种子变化的四位数字 OCR；⑤ 左右双色空间关系。答对 5/5 通过、3～4/5 记 warning、"
        "≤2/5 记 failed；模型不接受图片输入记 blocked、不阻断准入。",
        "低", (PRIORITY_VISION,), 5, concurrency=1, model_scope=True,
        admission="中转接入口径：视觉属高级面且按矩阵声明能力分流——矩阵声明无图像输入的型号"
                  "（如 qwen3.7-max）视觉被拒只记 vision_undeclared，不阻断准入；"
                  "声明支持却整批被拒记 relay_gap。答题本身：全对 passed、答对 3～4/5 warning、≤2/5 failed（均不单独否决准入）。",
    ),
}

CASES.update({
    'QWEN_CAPABILITY': QwenCase(
        'QWEN_CAPABILITY', 'Qwen 客观能力校验',
        '共用数学与逻辑、代码隔离执行、中文嵌套抽取、头中尾长上下文、工具错误恢复和双工具参数校验；'
        'Max 使用5道推理题，其他档3道；3.8 Max/3.7 Max使用不同长文本测试预算。预算不代表厂商上限。',
        '中', ('PQUAL',), 12, model_scope=True,
        admission='本卡全部适用的客观题有效通过；blocked/缺失均不能作为能力通过。'),
    'QWEN_VERIFY': QwenCase(
        'QWEN_VERIFY', 'Qwen 真实性与路由校验',
        '三次随机挑战及精确型号回显、真实/不存在型号正负对照、系统指令优先级；只提供可审计行为证据，不以自述证明身份。',
        '低', ('PVERIFY',), 6, model_scope=True,
        admission='本卡全部探针有效通过才称行为一致；不证明底层权重、厂商来源或不存在隐藏提示词。'),
    'QWEN_FULL': QwenCase(
        'QWEN_FULL', 'Qwen 综合能力与满血证据校验',
        '核心协议、思考、工具、型号适用的视觉及客观能力和真实性探针一起执行；必测证据不完整不能通过。'
        '可通过CLI提供同型号同版本同种子参考报告，比较是否出现回归。',
        '高（含长文本）', ('P0','P1',PRIORITY_THINKING,PRIORITY_TIER,PRIORITY_VISION,'PQUAL','PVERIFY'),
        54, model_scope=True,
        admission='全部必测项通过仅代表达到本测试标准；参考报告一致不等于厂商身份或全部能力已获证明。'),
})
