#!/usr/bin/env python3
"""探测 Claude thinking.signature 的内部结构，判断 protobuf 能承担哪些校验职责。

三个 phase：
  structure  扫描 results/ 里已保存的真实签名，统计 protobuf 版式、可读字段、
             各模型是否含模型标识，并用随机字节做对照估计「严格可解析」的判别力。
  hashes     尝试用常见哈希算法与候选片段复算签名内部的 12B/48B 固定字段，
             判断这些字段是否可在本地验算（无密钥）。
  mutation   真实端点分层变异矩阵：分别改外层 base64 字符、顶层版本 varint、
             块描述符字段、内层 12B/12B/48B、密文中段，观察上游判定粒度。
             需要 --base-url / --api-key（或 CLAUDE_RELAY_KEY），会向端点发请求。

只读：不写 results/，输出仅打印。
"""

from __future__ import annotations

import argparse
import base64
import collections
import glob
import hashlib
import json
import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 解析实现只留一份权威版本：fable5/signature_structure.py（套件 SIG-03 用的同一份）。
from fable5.signature_structure import (  # noqa: E402
    decode_signature,
    parse_wire as parse,
    readable_strings,
)


def collect_signatures() -> dict[str, dict]:
    """从 results/ 与 .pytest-tmp/ 的历史记录里收集全部签名及其上下文。"""
    seen: dict[str, dict] = {}
    files = sorted(glob.glob(str(ROOT / "results/PROTOBUF_SIGNATURE_CHECK-*/requests.jsonl")))
    files += sorted(glob.glob(str(ROOT / ".pytest-tmp/**/requests.jsonl"), recursive=True))
    for path in files:
        try:
            lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip().startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = record.get("response") or {}
            request = record.get("request") or {}
            model = response.get("model") or request.get("model")
            for block in response.get("content") or []:
                if block.get("type") != "thinking" or not block.get("signature"):
                    continue
                # 排除单测夹具：真实签名都在数百字节以上（.pytest-tmp 里有 "ABCDEF" 这类占位串）
                if len(decode_signature(block["signature"])) < 128:
                    continue
                key = hashlib.sha256(block["signature"].encode()).hexdigest()[:12]
                seen.setdefault(key, {
                    "signature": block["signature"],
                    "thinking": block.get("thinking") or "",
                    "model": model,
                    "phase": record.get("phase"),
                    "source": Path(path).parent.name,
                })
    return seen


def phase_structure() -> int:
    signatures = collect_signatures()
    print(f"去重签名 {len(signatures)} 个\n")

    layouts = collections.Counter()
    descriptor = collections.Counter()
    readable_all = collections.Counter()
    inner_shape = collections.Counter()
    fixed_len = collections.defaultdict(set)
    with_model = without_model = parsed = 0
    rows = []

    for key, item in signatures.items():
        raw = decode_signature(item["signature"])
        try:
            top = parse(raw)
        except ValueError as exc:
            layouts[f"严格解析失败: {exc}"] += 1
            continue
        parsed += 1
        top_map = {f: v for f, w, v, _ in top}
        if set(top_map) != {1, 2, 3}:
            layouts[f"顶层字段集 {sorted(top_map)}"] += 1
            continue
        layouts["#1 varint + #2 message + #3 varint"] += 1
        inner = parse(top_map[2])  # type: ignore[arg-type]
        blk = parse(inner[0][2])  # type: ignore[arg-type]
        for field, _wire, value, _off in blk:
            if field == 8:
                descriptor[value] += 1  # type: ignore[index]
        inner_shape[tuple(f for f, _w, _v, _o in inner[1:])] += 1
        for field, _w, value, _off in inner[1:]:
            fixed_len[field].add(len(value))  # type: ignore[arg-type]
        texts = readable_strings(raw)
        for _path, text in texts:
            readable_all[text] += 1
        model = str(item.get("model") or "")
        has_model = any(text == model for _p, text in texts) if model else False
        with_model += bool(has_model)
        without_model += not has_model
        rows.append(dict(key=key, model=model, phase=item["phase"], strings=len(texts),
                         has_model=has_model, cipher=len(inner[-1][2]),  # type: ignore[arg-type]
                         think=len(item["thinking"])))

    print("顶层版式：", dict(layouts))
    print("块描述符 #8（块类型）：", {k: v for k, v in descriptor.items()})
    print("内层字段号序列：", {str(k): v for k, v in inner_shape.items()})
    print("内层固定字段长度：", {k: sorted(v) for k, v in sorted(fixed_len.items())})
    print(f"\n严格可解析 {parsed}/{len(signatures)}")
    print(f"含模型标识明文 {with_model} / 不含 {without_model}")
    print("\n可读字符串全集：", {k: v for k, v in readable_all.most_common()})

    by_model = collections.defaultdict(lambda: [0, 0])
    for row in rows:
        by_model[row["model"]][0 if row["has_model"] else 1] += 1
    print(f"\n{'模型':<32}{'含模型标识':>10}{'仅块类型':>10}")
    for model, (yes, no) in sorted(by_model.items()):
        print(f"{str(model):<32}{yes:>10}{no:>10}")

    print("\n随机字节对照（估计严格可解析的判别力）：")
    for size in (64, 256, 1024, 2378):
        ok = 0
        trials = 20000 if size <= 256 else 4000
        for _ in range(trials):
            try:
                parse(secrets.token_bytes(size))
                ok += 1
            except ValueError:
                pass
        print(f"  {size:>5} 字节随机数据：{ok}/{trials} 可严格解析"
              f"　→ 误判率约 {ok / trials:.2e}")
    return 0


def phase_hashes() -> int:
    signatures = collect_signatures()
    algorithms = {
        "sha256": lambda b: hashlib.sha256(b).digest(),
        "sha384": lambda b: hashlib.sha384(b).digest(),
        "sha512": lambda b: hashlib.sha512(b).digest(),
        "sha512/256": lambda b: hashlib.new("sha512_256", b).digest(),
        "sha3_256": lambda b: hashlib.sha3_256(b).digest(),
        "sha3_384": lambda b: hashlib.sha3_384(b).digest(),
        "blake2b48": lambda b: hashlib.blake2b(b, digest_size=48).digest(),
        "blake2s32": lambda b: hashlib.blake2s(b).digest(),
    }
    checked = matched = 0
    for key, item in list(signatures.items()):
        raw = decode_signature(item["signature"])
        try:
            top = parse(raw)
        except ValueError:
            continue
        top_map = {f: (v, off) for f, w, v, off in top}
        if set(top_map) != {1, 2, 3}:
            continue
        inner_raw, inner_off = top_map[2]
        inner = parse(inner_raw)
        inner_map = {f: (v, off) for f, w, v, off in inner[1:]}
        if not all(f in inner_map for f in (2, 3, 4, 5)):
            continue
        cipher = inner_map[5][0]
        targets = {"#2(12B)": inner_map[2][0], "#3(12B)": inner_map[3][0], "#4(48B)": inner_map[4][0]}
        segments = {
            "raw": raw,
            "inner": inner_raw,
            "cipher": cipher,
            "raw[:-48]": raw[:-48],
            "inner[:-48]": inner_raw[:-48],
            "prefix": inner_raw[:inner_map[5][1] - inner_off],
            "think(utf8)": item["thinking"].encode(),
        }
        checked += 1
        hits = []
        for seg_name, seg in segments.items():
            for alg_name, alg in algorithms.items():
                try:
                    digest = alg(seg)
                except (ValueError, TypeError):
                    continue
                for target_name, target in targets.items():
                    if digest == target:
                        hits.append(f"{target_name} == {alg_name}({seg_name})")
                    elif len(digest) > len(target) and digest[:len(target)] == target:
                        hits.append(f"{target_name} == {alg_name}({seg_name})[:{len(target)}]")
        if hits:
            matched += 1
            print(f"  {key} ({item['model']}): " + "; ".join(hits))
    print(f"\n可本地复算的哈希关系：{matched} / {checked}")
    print("0 命中说明这些字段带密钥（nonce/keystream/MAC），protobuf 只能读结构，不能验内容。")
    return 0


def phase_mutation(base_url: str, api_key: str, model: str) -> int:
    from fable5.proto_probe import Client

    prompt = "请解释快速排序的平均时间复杂度为什么是 O(n log n)，给出递推推导过程。"
    with Client(base_url, api_key, model=model, timeout=300, max_retries=1) as client:
        acquisition = {"model": model, "max_tokens": 4096,
                       "thinking": {"type": "adaptive", "display": "summarized"},
                       "messages": [{"role": "user", "content": prompt}]}
        resp = client.post(json.dumps(acquisition).encode(), accept="application/json")
        print(f"取签 HTTP {resp.status} {resp.elapsed:.2f}s")
        body = resp.json() or {}
        if resp.status != 200:
            print("取签失败：", resp.text_head)
            return 1
        signed = [b for b in body.get("content") or []
                  if b.get("type") == "thinking" and b.get("signature")]
        if not signed:
            print("响应没有非空 thinking.signature，content 类型 =",
                  [b.get("type") for b in body.get("content") or []])
            return 1
        signature = signed[0]["signature"]
        padding = len(signature) - len(signature.rstrip("="))
        raw = decode_signature(signature)
        print(f"签名 b64 {len(signature)} → {len(raw)} 字节　padding={padding}　"
              f"thinking 明文 {len(signed[0].get('thinking') or '')}")

        top = parse(raw)
        top_map = {f: (v, off) for f, w, v, off in top}
        inner = parse(top_map[2][0], base=top_map[2][1])
        inner_map = {f: (v, off) for f, w, v, off in inner}
        block = parse(inner[0][2], base=inner[0][3])
        block_map = {f: (v, off) for f, w, v, off in block}
        print("顶层:", [(f, w, len(v) if w == 2 else v, o) for f, w, v, o in top])
        print("内层:", [(f, w, len(v) if w == 2 else v, o) for f, w, v, o in inner])
        print("块描述符:", [(f, w, v if w == 0 else repr(v[:20]), o) for f, w, v, o in block])

        def mutate(field: int, delta: int = 1, byte_index: int | None = None,
                   xor: int | None = None, setchar: bytes | None = None) -> bytes:
            buf = bytearray(raw)
            value, off = inner_map[field] if field in inner_map else top_map[field]
            if byte_index is not None:
                position = off + byte_index
                buf[position] = buf[position] ^ (xor or 0x01)
            elif setchar is not None:
                buf[off] = setchar[0]
            else:
                buf[off] = (int(value) + delta) % 8
            return bytes(buf)

        variants: dict[str, bytes | None] = {"00_基线（原签名）": raw}
        mid = len(signature) // 2
        while not signature[mid].isalnum():
            mid += 1
        variants["10_外层base64字符"] = None
        variants["20_解码重编码往返"] = raw
        variants["30_顶层#1版本varint"] = mutate(1)
        blk_type = bytearray(raw)
        blk_type[block_map[8][1]] = ord("x")
        variants["40_块描述符#8块类型首字节"] = bytes(blk_type)
        blk_field3 = bytearray(raw)
        blk_field3[block_map[3][1]] = 3
        variants["50_块描述符#3=2→3"] = bytes(blk_field3)
        variants["60_内层#2（12B）"] = mutate(2, byte_index=0)
        variants["70_内层#3（12B）"] = mutate(3, byte_index=0)
        variants["80_内层#4（48B）"] = mutate(4, byte_index=24)
        cipher_len = len(inner_map[5][0])
        variants["90_内层#5密文中段"] = mutate(5, byte_index=cipher_len // 2)

        replay = [{"role": "user", "content": prompt},
                  {"role": "assistant", "content": body["content"]},
                  {"role": "user", "content": "请确认上一轮结果，只回复 OK。"}]
        print(f"\n{'变体':<26}{'改动字节':>8}  HTTP   耗时  判定")
        for name, mutated in variants.items():
            messages = json.loads(json.dumps(replay, ensure_ascii=False))
            if mutated is None:
                changed = signature[:mid] + ("A" if signature[mid] != "A" else "B") + signature[mid + 1:]
                nbytes = 1
            else:
                nbytes = sum(1 for a, b in zip(raw, mutated) if a != b)
                if nbytes == 0:
                    changed = signature
                else:
                    encoded = base64.b64encode(mutated).decode()
                    changed = encoded.rstrip("=") if padding == 0 else encoded
            for block_item in messages[1]["content"]:
                if block_item.get("type") == "thinking" and block_item.get("signature"):
                    block_item["signature"] = changed
            request = {"model": model, "max_tokens": 4096,
                       "thinking": {"type": "adaptive", "display": "summarized"},
                       "messages": messages}
            result = client.post(json.dumps(request, ensure_ascii=False).encode(),
                                 accept="application/json")
            payload = result.json() or {}
            if result.status == 200:
                verdict = "接受"
            else:
                message = (payload.get("error") or {}).get("message") or result.text_head
                verdict = f"拒绝：{message[:80]}".replace("\n", " ")
            print(f"{name:<26}{nbytes:>8}  {result.status:>4}  {result.elapsed:5.2f}s  {verdict}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("structure", "hashes", "mutation"), required=True)
    parser.add_argument("--base-url", default=os.getenv("CLAUDE_RELAY_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("CLAUDE_RELAY_KEY", ""))
    parser.add_argument("--model", default=os.getenv("CLAUDE_RELAY_MODEL", "claude-fable-5-1"))
    args = parser.parse_args()
    if args.phase == "structure":
        return phase_structure()
    if args.phase == "hashes":
        return phase_hashes()
    if not args.base_url or not args.api_key:
        parser.error("mutation 需要 --base-url 与 --api-key（或 CLAUDE_RELAY_BASE_URL / CLAUDE_RELAY_KEY）")
    return phase_mutation(args.base_url, args.api_key, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
