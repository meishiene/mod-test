#!/usr/bin/env python3
"""K3 词表/模板指纹：用 moonshot 官方 K2 词表与官方 chat template 复算端点 prompt_tokens

原理
----
端点返回的 usage.prompt_tokens 是上游模型真实分词结果，网关无法伪造得那么细。
若「官方词表复算值 + 官方模板开销」逐样本等于端点值，说明上游用的就是 moonshot 家族
的词表与 chat template，即权重来自 K 系，而不是别家模型套壳。

官方物料（ModelScope moonshotai/Kimi-K2-Instruct）：tiktoken.model / tokenization_kimi.py / chat_template.jinja

用法：.venv/bin/python scripts/k3_tokenizer_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe

TOKDIR = Path("/tmp/k2tok")
tiktoken.model  # noqa: B018

# tokenization_kimi.py 里的 pat_str（原样抄）
PAT = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])

SPECIALS = {
    "<|im_end|>": 163586,
    "<|im_user|>": 163587,
    "<|im_assistant|>": 163588,
    "<|start_header_id|>": 163590,
    "<|end_header_id|>": 163591,
    "[EOT]": 163593,
    "<|im_system|>": 163594,
    "<|im_middle|>": 163595,
}

TEXTS = {
    "zh78": ("北京是中华人民共和国的首都，位于华北平原的北部。故宫、天坛和长城是著名的历史遗迹。"
             "长江与黄河孕育了中华文明，两者流域面积广阔，人口稠密，农业与工业都很发达。"),
    "en_words": ("The quick brown fox jumps over the lazy dog. Tokenization fingerprinting works "
                 "by comparing how many pieces a fixed string is split into under different vocabularies."),
    "code": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "en_simple": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen",
}

# 端点实测（取 P6 那批，thinking=disabled，max_tokens=1）
ENDPOINT = {"zh78": 61, "en_words": 50, "code": 55, "en_simple": 33, "baseline": 19}

DEFAULT_SYSTEM = "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>"


def render_user_only(content: str) -> str:
    """官方 template：首条非 system 消息时自动注入默认 system；末尾加 assistant 生成提示。"""
    return (DEFAULT_SYSTEM
            + "<|im_user|>user<|im_middle|>" + content + "<|im_end|>"
            + "<|im_assistant|>assistant<|im_middle|>")


def main() -> None:
    vocab = TOKDIR / "tiktoken.model"
    if not vocab.exists():
        print(f"缺词表: {vocab}", file=sys.stderr)
        sys.exit(1)
    ranks = load_tiktoken_bpe(str(vocab))
    enc = tiktoken.Encoding(name="kimi-k2", pat_str=PAT, mergeable_ranks=ranks,
                            special_tokens=SPECIALS)
    print(f"官方词表: {len(ranks)} mergeable + {len(SPECIALS)} special = n_vocab {enc.n_vocab}")
    print(f"官方 template 默认注入 system: {DEFAULT_SYSTEM[:70]}...\n")

    rows = []
    print(f"{'样本':10s} {'字符':>4s} {'端点':>5s} {'官方词表正文':>7s} {'+模板':>6s} {'差值':>5s}")
    for name, txt in TEXTS.items():
        body = len(enc.encode(txt, allowed_special="all"))
        full = len(enc.encode(render_user_only(txt), allowed_special="all"))
        ep = ENDPOINT[name]
        rows.append({"name": name, "chars": len(txt), "endpoint": ep, "body": body, "full": full})
        print(f"{name:10s} {len(txt):4d} {ep:5d} {body:7d} {full:6d} {ep - full:+5d}")

    base_txt = render_user_only("x")
    base_tokens = enc.encode(base_txt, allowed_special="all")
    print(f"\nbaseline（单字符消息 x）: 官方 template 复算 = {len(base_tokens)} token；端点实测 = {ENDPOINT['baseline']}")
    print("  模板 token 序列:", [enc.decode_single_token_bytes(t).decode("utf-8", "replace") for t in base_tokens])

    ok = all(r["endpoint"] == r["full"] for r in rows)
    print(f"\n判定：官方词表+官方模板复算与端点逐样本{'全部一致' if ok else '存在差异'}"
          f"（{sum(r['endpoint'] == r['full'] for r in rows)}/{len(rows)}）")
    print(f"baseline 一致: {len(base_tokens) == ENDPOINT['baseline']}")

    (TOKDIR / "tokenizer_check.json").write_text(
        json.dumps({"rows": rows, "baseline": {"official": len(base_tokens), "endpoint": ENDPOINT["baseline"]},
                    "template_tokens": len(base_tokens), "all_match": ok}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n落盘: {TOKDIR / 'tokenizer_check.json'}")


if __name__ == "__main__":
    main()
