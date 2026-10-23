"""厂商 / 模型注册表：金标从哪儿来、能证明到哪一步。

**唯一权威**。新增厂商只改这里 + 跑 ``scripts/gold_build.py``，不要在别处硬编码 repo 名。

三个字段决定"能证明什么"，务必分开看：

- ``evidence``      —— 证据等级（见 ``gold/__init__.py`` 的说明）；
- ``mapping_confidence`` —— 「官方开源权重」与「你要测的 API 型号」是不是同一个东西：
  * ``exact``  —— 开源权重就是 API 背后的权重（如 Kimi-K2）；
  * ``family`` —— 只能证明同一个**词表家族**，API 侧还有闭源档位（如 qwen-max vs qwen-flash）；
  * ``proxy``  —— 连家族都不敢说，只是同厂的另一个开源模型（如豆包 ↔ Seed-OSS）；
  * ``none``   —— 没有任何公开物证；
- ``usable_for_discrimination`` —— 是否允许进「异族判别」的分母。``proxy`` / ``none`` 一律 False：
  拿一个不敢确定的词表去判"这不是豆包"，会得出比不做更坏的结论。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 厂商级元信息：display 用来出报告，doc_url 是闭源型号的「L0-doc」出处。
VENDORS: dict[str, dict[str, str]] = {
    "qwen": {
        "display": "通义千问（阿里云百炼）",
        "doc_url": "https://help.aliyun.com/zh/model-studio/models",
    },
    "deepseek": {
        "display": "DeepSeek（深度求索）",
        "doc_url": "https://api-docs.deepseek.com/zh-cn/",
    },
    "glm": {
        "display": "智谱 GLM",
        "doc_url": "https://docs.bigmodel.cn/cn/guide/start/model-overview",
    },
    "ernie": {
        "display": "文心 ERNIE（百度千帆）",
        "doc_url": "https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Fm2vrveyu",
    },
    "kimi": {
        "display": "月之暗面 Kimi",
        "doc_url": "https://platform.moonshot.cn/docs/guide/start-using-kimi-api",
    },
    "minimax": {
        "display": "MiniMax",
        "doc_url": "https://platform.minimaxi.com/document/guides",
    },
    "doubao": {
        "display": "豆包 Doubao（火山方舟）",
        "doc_url": "https://www.volcengine.com/docs/82379/1099320",
    },
}


@dataclass(frozen=True)
class VendorModel:
    key: str
    vendor: str
    repo: str
    tokenizer_kind: str  # hf-json | tiktoken-bpe | missing
    tokenizer_file: str | None = None
    config_file: str | None = "config.json"
    evidence: str = "L0-weights"
    mapping_confidence: str = "family"
    usable_for_discrimination: bool = True
    api_model_globs: tuple[str, ...] = ()
    note: str = ""
    # 需要读本地已有文件的（Kimi 的 tiktoken.model 平台里已经存了一份）
    local_tokenizer_path: str | None = None
    extra_files: tuple[str, ...] = field(default_factory=tuple)

    @property
    def slug(self) -> str:
        return f"{self.vendor}:{self.key}"


# 顺序即报告顺序。同一厂商可以有多个权重版本，但**判别只用一个**（其余只作漂移参照）。
MODELS: tuple[VendorModel, ...] = (
    VendorModel(
        key="qwen3-32b",
        vendor="qwen",
        repo="Qwen/Qwen3-32B",
        tokenizer_kind="hf-json",
        tokenizer_file="tokenizer.json",
        api_model_globs=("qwen3-max", "qwen-max", "qwen-plus", "qwen-flash", "qwen-turbo",
                         "qwen3.6-plus", "qwen3.7-max", "qwen3.8-max"),
        note="qwen3-max / plus / flash 未开源，词表用同代开源 Qwen3 顶替 ⇒ 只到家族级；"
             "3.6/3.7/3.8 默认开思考，L1 探测必须显式 enable_thinking=false 才拿得到干净计数。",
    ),
    VendorModel(
        key="qwen2.5-72b",
        vendor="qwen",
        repo="Qwen/Qwen2.5-72B-Instruct",
        tokenizer_kind="hf-json",
        tokenizer_file="tokenizer.json",
        api_model_globs=(),
        usable_for_discrimination=False,
        note="上一代词表，只作「版本漂移」参照：同一中转前后两次计数若跨代跳变，说明上游换了权重代。",
    ),
    VendorModel(
        key="deepseek-v3.2",
        vendor="deepseek",
        repo="deepseek-ai/DeepSeek-V3.2-Exp",
        tokenizer_kind="hf-json",
        tokenizer_file="tokenizer.json",
        api_model_globs=("deepseek-chat", "deepseek-reasoner", "deepseek-v3", "deepseek-v3.1",
                         "deepseek-v3.2", "deepseek-v4", "deepseek-v4-pro", "deepseek-flash"),
        note="deepseek-chat 与 deepseek-reasoner 共用词表⇒ 词表无法区分思考/非思考档位，"
             "这两档只能靠 thinking 字段与预算行为分辨。",
    ),
    VendorModel(
        key="deepseek-v3.1",
        vendor="deepseek",
        repo="deepseek-ai/DeepSeek-V3.1",
        tokenizer_kind="hf-json",
        tokenizer_file="tokenizer.json",
        api_model_globs=(),
        usable_for_discrimination=False,
        note="与 V3.2 词表对照用；若两者 tokenizer sha256 相同，说明该代未换词表（本库会自动记录）。",
    ),
    VendorModel(
        key="glm-4.6",
        vendor="glm",
        repo="zai-org/GLM-4.6",
        tokenizer_kind="hf-json",
        tokenizer_file="tokenizer.json",
        api_model_globs=("glm-4.5", "glm-4.6", "glm-5", "glm-5.3", "glm-5.3-flash", "glm-5.3-flashx"),
        note="GLM 的 tokenizer 与 Qwen/DeepSeek 同为 BPE 但词表不同，是判别力最强的一组；"
             "flash/flashx 与旗舰同词表 ⇒ 降档识别必须靠上下文窗口与能力面。",
    ),
    VendorModel(
        key="ernie-4.5-300b",
        vendor="ernie",
        repo="baidu/ERNIE-4.5-300B-A47B-PT",
        tokenizer_kind="hf-json",
        tokenizer_file="tokenizer.json",
        api_model_globs=("ernie-4.5", "ernie-4.0", "ernie-5.0", "ernie-speed", "ernie-tiny",
                         "ernie-lite", "wenxin"),
        note="ERNIE 4.5 是 2025 年开源权重；文心 4.0 / 5.0 闭源档位是否同词表**未经证实**，"
             "因此 mapping_confidence 只能给 family。",
    ),
    VendorModel(
        key="minimax-m2",
        vendor="minimax",
        repo="MiniMaxAI/MiniMax-M2",
        tokenizer_kind="hf-json",
        tokenizer_file="tokenizer.json",
        api_model_globs=("minimax-m2", "minimax-m1", "abab"),
        note="M2/M1 同代词表；abab 系列闭源，映射同为家族级。",
    ),
    VendorModel(
        key="kimi-k2",
        vendor="kimi",
        repo="moonshotai/Kimi-K2-Instruct",
        tokenizer_kind="tiktoken-bpe",
        tokenizer_file=None,
        # 平台内已存了官方 tiktoken.model（sha256 见 lineage），不重复下载
        local_tokenizer_path="vendor_verifier/beam/tokenizer/tiktoken.model",
        api_model_globs=("kimi-k2", "kimi-k3", "kimi-k2.5", "kimi-k2.6", "kimi-k2.7", "moonshot"),
        mapping_confidence="exact",
        note="tiktoken 型词表需要 transformers 的 PreTrainedTokenizer 才能精确切分；"
             "本库**只固化词表级指纹**（sha256 / 词表大小 / 特殊 token），"
             "token 计数标 uncollected —— 不猜一个正则去假装能算，那会污染整张判别矩阵。",
    ),
    VendorModel(
        key="seed-oss-36b",
        vendor="doubao",
        repo="ByteDance-Seed/Seed-OSS-36B-Instruct",
        tokenizer_kind="hf-json",
        tokenizer_file="tokenizer.json",
        mapping_confidence="proxy",
        usable_for_discrimination=False,
        evidence="L0-weights(proxy)",
        api_model_globs=(),
        note="豆包 API 没有公开 tokenizer / 权重。Seed-OSS 是字节的**另一个**开源模型，"
             "拿它当豆包金标会制造假结论，故只登记不进判别。豆包只能用 L0-doc（文档声明）"
             "+ 行为面（上下文窗口硬实测、SSE 字段、拒绝话术）验。",
    ),
)


def resolve(model_name: str) -> VendorModel | None:
    """把中转侧看到的模型名映射到注册表条目（按 globs 前缀/包含匹配）。

    匹配不到返回 ``None``——调用方必须把它当「无金标」处理，不要退回"最像的那家"。
    """
    lowered = (model_name or "").strip().lower()
    if not lowered:
        return None
    # 先精确 token 匹配，再退化为「包含」；长 glob 优先，避免 qwen-max 被 qwen 抢走。
    candidates = sorted(
        (m for m in MODELS if m.api_model_globs),
        key=lambda m: max(len(g) for g in m.api_model_globs),
        reverse=True,
    )
    for model in candidates:
        for glob in model.api_model_globs:
            if glob in lowered:
                return model
    return None


def discriminating_models() -> tuple[VendorModel, ...]:
    """进入「异族判别」的模型：必须有可用物证，且不是 proxy。"""
    return tuple(
        m for m in MODELS
        if m.usable_for_discrimination and m.tokenizer_kind != "missing"
    )
