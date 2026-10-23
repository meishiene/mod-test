"""生成 tokenizer / 原生契约金标（``gold/data/*.json``）。

产出两张表：

- ``tokenizer_gold.json``：每个厂商权重版本的**词表物证 + 固定探针 token 数**；
- ``lineage.json``：每条物证的溯源（repo / revision / 文件 / sha256 / 字节数 / 抓取时间），
  以及失败项的 ``uncollected`` 原因。

**失败不兜底**：拉不到就记 ``status="uncollected"`` 并写清原因，绝不拿近似词表顶替。
本库的价值全在"记录为真"，一次兜底就能让整张判别矩阵失效。
"""
from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import fetch as fetcher
from .probes import CORPUS_VERSION, all_samples
from .registry import MODELS, VENDORS, VendorModel

DATA_DIR = Path(__file__).resolve().parent / "data"
GOLD_PATH = DATA_DIR / "tokenizer_gold.json"
LINEAGE_PATH = DATA_DIR / "lineage.json"


def _and_all(samples: dict[str, str], counter) -> dict[str, int]:
    return {name: int(counter(text)) for name, text in samples.items()}


def _hf_json_fingerprint(path: Path, samples: dict[str, str]) -> dict[str, Any]:
    from tokenizers import Tokenizer  # 延迟导入：没装 tokenizers 时其余功能仍可用

    tok = Tokenizer.from_file(str(path))
    counts = _and_all(samples, lambda text: len(tok.encode(text, add_special_tokens=False).ids))
    vocab = tok.get_vocab()
    added = tok.get_added_tokens_decoder()
    try:  # tokenizers 的 model 名只存在于序列化结果里，属性上没有稳定接口
        model_type = json.loads(tok.to_str()).get("model", {}).get("type", "")
    except Exception:
        model_type = ""
    return {
        "kind": "hf-json",
        "model_type": model_type,
        "vocab_size": tok.get_vocab_size(),
        "added_tokens": len(added),
        "special_tokens": sorted({t.content for t in added.values() if getattr(t, "special", False)})[:40],
        "vocab_head_sha256": hashlib.sha256(
            json.dumps(sorted(vocab.items())[:2048], ensure_ascii=False).encode()
        ).hexdigest(),
        "probe_counts": counts,
    }


def _tiktoken_fingerprint(model: VendorModel, samples: dict[str, str]) -> dict[str, Any]:
    """tiktoken 型词表：只固化词表级指纹，token 计数明确标 uncollected。

    理由见 registry 里 kimi-k2 的 note：精确切分依赖 transformers 的
    ``PreTrainedTokenizer`` 与厂商私有正则，本库不猜。
    """
    from tiktoken.load import load_tiktoken_bpe

    path = Path(__file__).resolve().parent.parent / (model.local_tokenizer_path or "")
    ranks = load_tiktoken_bpe(str(path))
    head = [[base64.b64encode(k).decode("ascii"), v] for k, v in list(ranks.items())[:2048]]
    return {
        "kind": "tiktoken-bpe",
        "model_type": "tiktoken-bpe",
        "vocab_size": len(ranks),
        "added_tokens": None,
        "special_tokens": [],
        "vocab_head_sha256": hashlib.sha256(
            json.dumps(head, ensure_ascii=False).encode()
        ).hexdigest(),
        "probe_counts": {},
        "counts_uncollected_reason": "tiktoken 型词表需 transformers 预分词正则才能精确计数；"
                                     "本库不猜正则，改用「词表 sha256 + 特殊 token + 上下文窗口」判别。",
    }


def _native_config(raw: bytes) -> dict[str, Any]:
    """从官方 config.json 抽原生契约。``max_position_embeddings`` 是上下文窗口的**硬证据**。"""
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError:
        return {"parse_error": True}
    keys = (
        "architectures", "model_type", "max_position_embeddings", "vocab_size",
        "num_hidden_layers", "hidden_size", "rope_scaling", "torch_dtype",
        "tie_word_embeddings",
    )
    out = {k: cfg[k] for k in keys if k in cfg}
    # 多模态/混合权重把文本侧配置塞在 text_config / llm_config 里（Kimi 就是这种）
    for nested_key in ("text_config", "llm_config", "language_config"):
        nested = cfg.get(nested_key)
        if isinstance(nested, dict):
            for k in keys:
                if k not in out and k in nested:
                    out[k] = nested[k]
            out.setdefault("_nested_from", nested_key)
    # 少数权重把窗口写成 "seq_length" / "max_sequence_length"
    for alt in ("seq_length", "max_sequence_length", "n_positions"):
        if "max_position_embeddings" not in out and alt in cfg:
            out["max_position_embeddings"] = cfg[alt]
            out["max_position_embeddings_source_key"] = alt
    return out


def _contract_block(cfg: dict[str, Any]) -> dict[str, Any]:
    """把 config 里的窗口信息整理成一条**带免责说明**的契约记录。

    这是本库最容易被人用错的地方，所以说明写进数据里而不是只写在文档里：
    ``native_train_context`` 是权重训练时的窗口，**不是** API 页面宣称的上下文长度。
    例如 Qwen3-32B 的 config 写 40960，官方靠 YaRN 外推到更长；若拿 40960 去卡
    "中转砍窗口"，就会冤枉一个正常的中转。判砍窗口只能**实测**（发超长请求看 400／截断）。
    """
    rope = cfg.get("rope_scaling")
    rope = rope if isinstance(rope, dict) else {}
    return {
        "native_train_context": cfg.get("max_position_embeddings"),
        "rope_type": rope.get("type") or rope.get("rope_type"),
        "rope_factor": rope.get("factor"),
        "rope_original_context": rope.get("original_max_position_embeddings"),
        "caveat": "native_train_context 来自官方 config.json，只是权重训练窗口；不等于 API 宣称窗口。"
                  "判『砍窗口』必须发超长请求实测，不能拿本字段直接下结论。",
    }


def _family_signature(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    payload = json.dumps(sorted(counts.items()), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_model(model: VendorModel) -> tuple[dict[str, Any], dict[str, Any]]:
    """返回 ``(gold_entry, lineage_entry)``。任何异常都折成 uncollected，不伪造。"""
    samples = all_samples()
    entry: dict[str, Any] = {
        "slug": model.slug,
        "vendor": model.vendor,
        "vendor_display": VENDORS.get(model.vendor, {}).get("display", model.vendor),
        "key": model.key,
        "evidence": model.evidence,
        "mapping_confidence": model.mapping_confidence,
        "usable_for_discrimination": model.usable_for_discrimination,
        "api_model_globs": list(model.api_model_globs),
        "note": model.note,
        "corpus_version": CORPUS_VERSION,
        "status": "ok",
        "tokenizer": None,
        "native_config": None,
        "contract": None,
    }
    lineage: dict[str, Any] = {
        "slug": model.slug,
        "vendor": model.vendor,
        "repo": model.repo,
        "files": [],
        "errors": [],
    }

    if model.tokenizer_kind == "missing":
        entry["status"] = "uncollected"
        entry["reason"] = "厂商未公开 tokenizer / 权重"
        lineage["errors"].append("厂商未公开 tokenizer / 权重")
        return entry, lineage

    try:
        if model.local_tokenizer_path:
            path = Path(__file__).resolve().parent.parent / model.local_tokenizer_path
            if not path.exists():
                raise fetcher.FetchError(f"本地词表不存在：{model.local_tokenizer_path}")
            toks = _tiktoken_fingerprint(model, samples)
            lineage["files"].append({
                "filename": model.local_tokenizer_path,
                "source": "local-vendored",
                "sha256": fetcher.sha256_of(path),
                "bytes": path.stat().st_size,
            })
            entry["tokenizer"] = {**toks, "sha256": fetcher.sha256_of(path)}
        else:
            tok_file = fetcher.fetch(model.repo, model.tokenizer_file or "tokenizer.json")
            toks = _hf_json_fingerprint(tok_file.path, samples)
            lineage["files"].append({
                "filename": tok_file.filename, "source": tok_file.source_url,
                "revision": tok_file.revision, "sha256": tok_file.sha256, "bytes": tok_file.size,
            })
            entry["tokenizer"] = {**toks, "sha256": tok_file.sha256, "repo": model.repo,
                                  "revision": tok_file.revision}
    except Exception as exc:  # 网络/格式/依赖缺失
        entry["status"] = "uncollected"
        entry["reason"] = f"{type(exc).__name__}: {exc}"
        lineage["errors"].append(entry["reason"])

    if model.config_file:
        try:
            if model.local_tokenizer_path:
                cfg_path = Path(__file__).resolve().parent.parent / model.local_tokenizer_path
                cfg_path = cfg_path.parent / "config.json"
                entry["native_config"] = _native_config(cfg_path.read_bytes())
                lineage["files"].append({
                    "filename": "config.json", "source": "local-vendored",
                    "sha256": fetcher.sha256_of(cfg_path), "bytes": cfg_path.stat().st_size,
                })
            else:
                cfg_file = fetcher.fetch(model.repo, model.config_file,
                                         revision=(entry.get("tokenizer") or {}).get("revision"))
                entry["native_config"] = _native_config(cfg_file.path.read_bytes())
                lineage["files"].append({
                    "filename": cfg_file.filename, "source": cfg_file.source_url,
                    "revision": cfg_file.revision, "sha256": cfg_file.sha256, "bytes": cfg_file.size,
                })
        except Exception as exc:
            entry["native_config"] = {"_uncollected": f"{type(exc).__name__}: {exc}"}
            lineage["errors"].append(f"config.json: {exc}")

    if entry["tokenizer"]:
        entry["tokenizer"]["family_signature"] = _family_signature(
            entry["tokenizer"].get("probe_counts") or {}
        )
    if isinstance(entry.get("native_config"), dict) and "_uncollected" not in entry["native_config"]:
        entry["contract"] = _contract_block(entry["native_config"])
    return entry, lineage


def discrimination_matrix(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """判别力自检：只统计「有 count 且允许判别」的模型，两两算可区分探针数。

    这张矩阵是金标库的**能力上界声明**：某两家差异探针数为 0，就说明纯 tokenizer 方法
    永远分不开它们（例如同词表的不同档位），必须换上下文窗口 / 行为面。
    """
    usable = [
        e for e in entries
        if e.get("usable_for_discrimination") and (e.get("tokenizer") or {}).get("probe_counts")
    ]
    names = [e["slug"] for e in usable]
    matrix: dict[str, dict[str, int]] = {n: {} for n in names}
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            ca = a["tokenizer"]["probe_counts"]
            cb = b["tokenizer"]["probe_counts"]
            diff = sum(1 for k in ca if k in cb and ca[k] != cb[k])
            matrix[a["slug"]][b["slug"]] = diff
            matrix[b["slug"]][a["slug"]] = diff
    same_sig: dict[str, list[str]] = {}
    for e in entries:  # 含不进判别的参照条目：它们"分不开"这件事本身就是结论
        sig = (e.get("tokenizer") or {}).get("family_signature")
        if sig and (e.get("tokenizer") or {}).get("probe_counts"):
            same_sig.setdefault(sig, []).append(e["slug"])
    return {
        "probe_total": len(all_samples()),
        "pairs": matrix,
        "indistinguishable_groups": [v for v in same_sig.values() if len(v) > 1],
    }


def build(models: tuple[VendorModel, ...] | None = None) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    lineages: list[dict[str, Any]] = []
    for model in models or MODELS:
        entry, lineage = build_model(model)
        entries.append(entry)
        lineages.append(lineage)

    payload = {
        "schema": "gold/tokenizer_gold@1",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_version": CORPUS_VERSION,
        "probe_names": sorted(all_samples()),
        "entries": entries,
        "discrimination": discrimination_matrix(entries),
    }
    lineage_payload = {
        "schema": "gold/lineage@1",
        "built_at": payload["built_at"],
        "hf_endpoint": fetcher.DEFAULT_ENDPOINT,
        "models": lineages,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LINEAGE_PATH.write_text(json.dumps(lineage_payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return payload
