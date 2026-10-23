"""固定探针语料（tokenizer 侧）。

设计原则
--------
1. **只放"字面稳定、不需要模型回答"的字符串**——tokenizer 计数由纯代码算出，没有采样噪声，
   所以这些探针是唯一能做到「temp、seed、供应商心情都不影响」的金标；
2. 每条探针要**尽量放大厂商间差异**：生僻字 / 文言 / 繁体 / emoji 组合 / 代码 / 纯数字
   在中文各家 BPE 上的切分差异远大于普通白话；
3. ``PAIRS`` 是「短 + 填充」成对样本，用来做**两点差分**：中转如果加了固定前缀，
   两条都多同样的量，相减即消；只有真实的切分差异会留下残差；
4. 语料一旦发布就**不许改字面**——改了要升 ``CORPUS_VERSION``，否则历史金标与实测不可比。

不要把这里改成"难一点的题"。难不难无关，能不能被**逐字节复现**才有关。
"""
from __future__ import annotations

CORPUS_VERSION = "gold-tok-2026-10-22b"

# 探针正文：全部是"字面量"，不含指令、不需要模型理解。
PROBE_STRINGS: dict[str, str] = {
    # 中文常用白话：各家差异小，主要用来估 offset 的稳定度
    "zh_plain": "今天天气不错，我们去公园散步吧。",
    # 生僻字：字表覆盖差异最大的区域
    "zh_rare": "龘靐齉爩麤龗灪厵滟爨癵籱饢驫鱻麣",
    # 文言：单字高频、虚词密集
    "zh_classical": "子曰：学而时习之，不亦说乎？有朋自远方来，不亦乐乎？人不知而不愠，不亦君子乎？",
    # 繁体：简繁是否共用词表一眼看出
    "zh_traditional": "繁體中文測試：臺灣、香港、澳門的資料庫連線設定。",
    # emoji：含 ZWJ 家族序列与肤色修饰符
    "emoji": "🎉🚀🌏🧪💡👨‍👩‍👧‍👦👍🏽",
    # 中英混排 + 代码标识符
    "mixed": "用 Python 写一个 fib(n)，返回 list[int]，不要用递归。",
    # 代码块：缩进与符号密集
    "code": 'def f(x):\n    return {"a": x * 2, "b": [i for i in range(3)]}\n',
    # 纯数字/科学计数/十六进制
    "numeric": "1234567890 3.141592653589793 0x1F 1e-9 2026-10-22T16:25:05+08:00",
    # 全角标点与空白
    "punct": "　，（）。；：！？——……《》【】「」",
    # 英文 pangram：英文侧切分一致性参照（**对照组**，各家常一样，用来校验 offset 估得对不对）
    "en": "The quick brown fox jumps over the lazy dog.",
    # CJK 扩展 B 区：字表覆盖的天花板区，各家差异最大
    "cjk_ext_b": "𠀀𠀁𠀂𠀃𠀄𪚥𪚦𪚧𪚨𪚩",
    # 假名 + 谚文：中文厂商词表对东亚邻语的覆盖差异明显
    "kana_hangul": "こんにちは、カタカナ。안녕하세요. 반갑습니다.",
    # 数学/希腊符号
    "sym_math": "∑∫√≈≠≤≥∞∂∇αβγδεζηθλμπσφψω",
    # 长中文：长度放大的同时保持可复现（与 filler 不同，这条是自然文本）
    "zh_long": "人工智能模型的评测需要可复现的证据链，而不是单次对话的主观感受。" * 6,
}

# 填充段：用于差分。刻意选"重复、无歧义、单字为主"的片段，让不同 tokenizer 的
# 切分行为尽量接近，从而把差分样本的噪声压在最低。
FILLER = "一二三四五六七八九十" * 20

# (short_id, long_id) —— 同一探针配上填充后的版本，两条相减得到"填充段的净 token 数"。
PAIRS: tuple[tuple[str, str], ...] = (
    ("zh_plain", "zh_plain_filled"),
    ("zh_rare", "zh_rare_filled"),
    ("code", "code_filled"),
)

FILLED_SUFFIX = "_filled"


def filled(text: str) -> str:
    """探针 + 填充段（顺序固定：填充在前，正文在后）。

    填充放在前面而不是后面：部分中转会对**开头**的 system/前缀做改写，
    把正文顶到尾部可以少受一点污染（这也是实测里更稳的一侧）。
    """
    return FILLER + text


def all_samples() -> dict[str, str]:
    """返回「探针名 → 待计数文本」的完整映射（含 PAIRS 里的填充版本）。"""
    samples = dict(PROBE_STRINGS)
    for short_id, long_id in PAIRS:
        samples[long_id] = filled(PROBE_STRINGS[short_id])
    return samples
