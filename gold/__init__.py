"""原厂金标库（gold baseline）。

**这个包不产生任何"模型回答"**——它只做两件事：

1. 把**厂商自己发布的物证**（官方 tokenizer 文件、官方 ``config.json``）固化成本地金标，
   每条都记 ``source_url`` / ``hf_revision`` / ``sha256``，可离线复现、可版本绑定；
2. 提供「实测 token 数向量 vs 金标向量」的差分判别方法（消固定 offset，抗中转加壳）。

证据等级（写进每条记录，不允许混用）：

- ``L0-weights``：厂商开源权重的 tokenizer / config ⇒ 原厂物证，置信最高，
  但只证明「这个**家族**的词表/窗口」，不证明「某个 API 型号的具体档位」；
- ``L0-doc``：厂商官方文档声明（闭源型号只能到这里），需记 URL + 抓取时间；
- ``L1-consensus``：多个**独立中转**对同一模型的一致结果 ⇒ 共识基线，不是原厂；
- ``L2-single``：单条中转结果 ⇒ 只能当**待验样本**，不能当金标。

没有 key 也能建 L0-weights 这一层，因为词表和 config 是公开的；需要花钱买 key 的只有
「行为分布」（拒绝话术、知识截止、约束题通过率、SSE 时序），见 ``gold/README.md``。
"""
from __future__ import annotations

from .probes import CORPUS_VERSION, FILLER, PROBE_STRINGS
from .registry import MODELS, VENDORS, resolve

__all__ = [
    "CORPUS_VERSION",
    "FILLER",
    "PROBE_STRINGS",
    "MODELS",
    "VENDORS",
    "resolve",
]
