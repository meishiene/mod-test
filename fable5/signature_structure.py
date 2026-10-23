#!/usr/bin/env python3
"""读取 Claude ``thinking.signature`` 的内部结构：base64 → protobuf。

只读模块：不发请求、不写文件。承担三项结构层核对 + 一组版式留痕。

判据全部来自 2026-10-15 实测（``scripts/probe_signature_structure.py`` 可复现）：

1. ``strictly_parseable`` —— 解码后能否严格解析为 protobuf（完整消费全部字节、无非法
   wiretype、无字段号 0；顶层与内层 ``#2`` 都要求通过）。36 个真实签名 36/36 通过；
   随机字节对照 64B 误判约 1.4e-3、1KB 以上 ≈ 0（2378B 实测 0/4000）
   ⇒ **可作「签名是否由上游签发」的伪造判据**。
2. ``block_type_matches`` —— 签名内块描述符 ``#8``（路径 ``.2.1.8``）是否等于响应该块的
   ``type``。实测 36/36 为 ``thinking``。
3. ``reencoded_equal`` —— 标准 base64 重编码是否与原签名字符串逐字符相同。这是「改一个
   字符被上游拒绝」的归因基线：若重编码本身有损耗，那个 400 就说不清是谁造成的。
4. **标识一致性**（``model_match``，取证档，2026-10-15 新增）—— 签名明文里若出现模型标识，
   与请求模型的对比结论分三档：**一致**（含 ``claude-fable-5`` / ``claude-fable-5-1`` 这类按
   ``-`` 边界的别名/版本写法）/ **不一致**（记 ``warning``，不阻断准入）/ **签名里没有标识
   ⇒ 无结论**（实测 ``claude-fable-5`` 的签名就没有模型标识明文，不能当成不一致）。这一段是
   上游自述，本地没有官方 schema 断言它是权威身份，因此**不做 failed 判定**。

**边界（实测，不得越界推断）**：内层 ``#2``/``#3``（各 12B）与 ``#4``（48B）带密钥——
8 种哈希 × 7 种候选段复算 **0/20** 命中；``thinking`` 明文不在签名里（内层 ``#5`` 是密文）。
⇒ 本模块只能读**结构**，不能验**内容真伪**；签名既不能用于核对响应正文，也不能用于判定
请求侧 0 注入（取签与回传是同网关同注入，常量注入恒等抵消）。
"""
from __future__ import annotations

import base64
import binascii
from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_BLOCK_TYPE = "thinking"
# 实测版式：``#1 varint + #2 message + #3 varint``（20 个样本）与 ``#2 + #3``（16 个样本）。
KNOWN_TOP_LAYOUTS = ((1, 2, 3), (2, 3))
MAX_READABLE_STRINGS = 12
MAX_STRING_CHARS = 120
MIN_STRING_CHARS = 4
# 疑似模型标识的厂商/家族前缀。只用来判断「这段明文像不像模型名」，**不用来断言身份**：
# 签名里写谁、写不写，都由上游决定（实测 claude-fable-5 的签名里没有模型标识明文）。
MODEL_ID_PREFIXES = ("claude", "deepseek", "gpt", "gemini", "kimi", "moonshot", "qwen",
                     "glm", "grok", "llama", "mistral", "doubao", "ernie", "hunyuan")


def looks_like_model_id(text: str) -> bool:
    """明文是否形如模型标识：已知厂商前缀 + 小写字母/数字/``.``/``-``/``_``。

    前缀之后必须还有内容（裸 ``claude`` 不算），避免把普通英文单词误认成模型标识。
    """
    if not text:
        return False
    prefix = next((item for item in MODEL_ID_PREFIXES if text.startswith(item)), None)
    if prefix is None:
        return False
    rest = text[len(prefix):]
    return bool(rest) and all(char.islower() or char.isdigit() or char in ".-_" for char in rest)


def model_id_compatible(model_id: str, expected_model: str) -> bool:
    """签名内标识与请求模型是否算一致。

    除完全相等（忽略大小写）外，只额外容忍**按 ``-`` 边界的别名/版本写法**：
    ``claude-fable-5`` 与 ``claude-fable-5-1`` 互为前缀即算一致（网关常回基础型号名）；
    ``claude-opus-5`` 与 ``claude-opus-50`` 不算（边界不是 ``-``，是真不同的型号名）。
    """
    if not model_id or not expected_model:
        return False
    left, right = model_id.lower(), expected_model.lower()
    if left == right:
        return True
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    return longer.startswith(shorter) and longer[len(shorter)] == "-"


class WireError(ValueError):
    """载荷不是合法 protobuf wire 格式。"""


def read_varint(buf: bytes, index: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        if index >= len(buf):
            raise WireError("varint 越界")
        byte = buf[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, index
        shift += 7
        if shift > 63:
            raise WireError("varint 过长")


def parse_wire(buf: bytes, base: int = 0) -> list[tuple[int, int, Any, int]]:
    """严格解析一层 wire 字段，返回 ``[(字段号, wiretype, 值, 值偏移)]``。

    必须完整消费全部字节；字段号 0、已废弃的 group wiretype（3/4）、长度越界一律抛
    ``WireError``。length-delimited 的载荷保持为 ``bytes``（不递归）。
    """
    fields: list[tuple[int, int, Any, int]] = []
    index = 0
    while index < len(buf):
        key, index = read_varint(buf, index)
        number, wire = key >> 3, key & 7
        if number == 0:
            raise WireError("字段号 0 非法")
        if wire == 0:
            start = index
            value, index = read_varint(buf, index)
            fields.append((number, wire, value, base + start))
        elif wire == 1:
            if index + 8 > len(buf):
                raise WireError("fixed64 越界")
            fields.append((number, wire, buf[index:index + 8], base + index))
            index += 8
        elif wire == 2:
            length, index = read_varint(buf, index)
            if index + length > len(buf):
                raise WireError("length-delimited 越界")
            fields.append((number, wire, buf[index:index + length], base + index))
            index += length
        elif wire == 5:
            if index + 4 > len(buf):
                raise WireError("fixed32 越界")
            fields.append((number, wire, buf[index:index + 4], base + index))
            index += 4
        else:
            raise WireError(f"非法 wiretype {wire}")
    return fields


def decode_signature(signature: str) -> bytes:
    """base64 解码（容忍缺失的 ``=`` 填充）。非法字符抛 ``binascii.Error``。"""
    return base64.b64decode(signature + "=" * (-len(signature) % 4))


def encode_signature(raw: bytes) -> str:
    """标准 base64 编码（带 ``=`` 填充），用于等价性基线比较。"""
    return base64.b64encode(raw).decode("ascii")


def readable_strings(buf: bytes, prefix: str = "", depth: int = 0) -> list[tuple[str, str]]:
    """递归收集可读字符串，返回 ``[(字段路径, 文本)]``；路径形如 ``.2.1.8``。"""
    if depth > 6:
        return []
    try:
        fields = parse_wire(buf)
    except WireError:
        return []
    found: list[tuple[str, str]] = []
    for number, wire, value, _offset in fields:
        if wire != 2:
            continue
        path = f"{prefix}.{number}"
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError:
            found += readable_strings(value, path, depth + 1)
            continue
        if len(text) >= MIN_STRING_CHARS and all(char.isprintable() for char in text):
            found.append((path, text))
        else:
            found += readable_strings(value, path, depth + 1)
    return found


@dataclass
class SignatureStructure:
    """一份签名的结构层读解结果。字段全是**观测**，问题清单由 ``failures()`` 给出。"""

    signature_length: int = 0
    byte_length: int = 0
    padding: int = 0
    decode_error: str | None = None
    strictly_parseable: bool = False
    parse_error: str | None = None
    reencoded_equal: bool = False
    top_fields: list[int] = field(default_factory=list)
    known_layout: bool = False
    version: int | None = None
    block_type: str | None = None
    block_type_matches: bool | None = None
    inner_field_numbers: list[int] = field(default_factory=list)
    fixed_lengths: dict[str, int] = field(default_factory=dict)
    cipher_length: int | None = None
    model_identifier: str | None = None
    # 标识一致性（取证档）：``expected_model`` 是请求模型，``model_identifiers`` 是签名里
    # 全部疑似模型标识的明文，``model_match`` 三档 —— True 一致 / False 不一致 / None 无结论。
    expected_model: str | None = None
    model_identifiers: list[str] = field(default_factory=list)
    model_match: bool | None = None
    readable_strings: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def failures(self) -> list[str]:
        """返回未通过的结构层三项核对；空列表表示三项全过。

        **不含标识一致性** —— 那是取证档，单独走 ``model_consistency_problems()``。
        """
        if not self.signature_length:
            return ["未提供签名"]
        if self.decode_error:
            return [f"签名不是合法 base64（{self.decode_error}）"]
        problems: list[str] = []
        if not self.strictly_parseable:
            problems.append(f"解码后不是合法 protobuf（{self.parse_error}）")
        elif self.block_type is None:
            problems.append(f"签名内未找到块类型字段（顶层字段号 {self.top_fields}，"
                            f"已知版式为 1,2,3 / 2,3）")
        elif self.block_type_matches is False:
            problems.append(f"签名内块类型 {self.block_type!r} 与响应块类型不一致")
        if not self.reencoded_equal:
            problems.append("标准 base64 重编码与原签名不一致（解码/编码存在损耗）")
        return problems

    def model_consistency_problems(self) -> list[str]:
        """标识一致性核对项：只在**确实不一致**（签名写了模型标识且与请求模型不兼容）时才给。

        签名里没有模型标识明文（实测 fable 系列如此）不算问题 —— 没有证据不等于反证。
        """
        if self.model_match is False:
            return [f"签名内模型标识 {'、'.join(self.model_identifiers)} 与请求模型 "
                    f"{self.expected_model!r} 不一致"]
        return []


def _first_block_type(inner: list[tuple[int, int, Any, int]]) -> str | None:
    """从内层取第一个块的描述符 ``#8``（实测为块类型，``thinking``）。"""
    if not inner or inner[0][0] != 1 or inner[0][1] != 2:
        return None
    try:
        descriptor = parse_wire(inner[0][2])
    except WireError:
        return None
    for number, wire, value, _offset in descriptor:
        if number == 8 and wire == 2:
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return None
    return None


def read_signature(signature: str, *, expected_model: str | None = None,
                   expected_block_type: str | None = DEFAULT_BLOCK_TYPE,
                   ) -> SignatureStructure:
    """解析一份签名并完成结构层核对。

    ``expected_block_type`` 传入响应该块的 ``type``（本套件恒为 ``thinking``）；
    传 ``None`` 表示不做块类型核对（``block_type_matches`` 保持 ``None``）。
    """
    result = SignatureStructure(signature_length=len(signature),
                                padding=len(signature) - len(signature.rstrip("=")))
    if not signature:
        result.decode_error = result.parse_error = "签名为空"
        return result
    try:
        raw = decode_signature(signature)
    except (binascii.Error, ValueError) as exc:
        result.decode_error = result.parse_error = f"{type(exc).__name__}: {exc}"
        return result
    result.byte_length = len(raw)
    result.reencoded_equal = encode_signature(raw) == signature
    try:
        top = parse_wire(raw)
    except WireError as exc:
        result.parse_error = f"顶层：{exc}"
        return result
    result.top_fields = [number for number, _wire, _value, _off in top]
    result.known_layout = tuple(result.top_fields) in KNOWN_TOP_LAYOUTS
    top_map = {number: value for number, _wire, value, _off in top}
    if isinstance(top_map.get(1), int):
        result.version = top_map[1]
    inner_raw = top_map.get(2)
    if not isinstance(inner_raw, bytes):
        result.parse_error = f"顶层 #2 不是 length-delimited 字段（字段号 {result.top_fields}）"
        return result
    try:
        inner = parse_wire(inner_raw)
    except WireError as exc:
        result.parse_error = f"内层 #2：{exc}"
        return result
    result.strictly_parseable = True
    result.inner_field_numbers = [number for number, _wire, _value, _off in inner]
    for number, wire, value, _off in inner[1:]:
        if wire == 2:
            result.fixed_lengths[str(number)] = len(value)
    cipher = next((value for number, wire, value, _off in inner
                   if number == 5 and wire == 2), None)
    if isinstance(cipher, bytes):
        result.cipher_length = len(cipher)
    result.block_type = _first_block_type(inner)
    if expected_block_type is not None:
        result.block_type_matches = result.block_type == expected_block_type
    strings = readable_strings(raw)
    result.readable_strings = [{"path": path, "text": text[:MAX_STRING_CHARS]}
                               for path, text in strings[:MAX_READABLE_STRINGS]]
    if expected_model:
        result.expected_model = expected_model
        # 只在**全部可读明文**里找标识（不受 MAX_READABLE_STRINGS 截断影响），
        # 排序去重保证同一份签名多次读解得到同一结论。
        candidates = sorted({text for _path, text in strings if looks_like_model_id(text)})
        result.model_identifiers = candidates
        if any(model_id_compatible(text, expected_model) for text in candidates):
            result.model_match = True
        elif candidates:
            result.model_match = False
        # 兼容字段：恰好等于请求模型的那段明文（历史口径，一字不改）。
        result.model_identifier = next((text for text in candidates
                                        if text == expected_model), None)
    return result


def encode_varint(value: int) -> bytes:
    """protobuf varint 编码（长度前缀必须用它：>127 字节的单字节写法会被解析成续字节）。"""
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def synthetic_signature(*, block_type: str = DEFAULT_BLOCK_TYPE, version: int = 4,
                        cipher_length: int = 16, model_identifier: str | None = None) -> str:
    """构造**合成**签名样本，仅供离线自检与单测使用，不参与任何判定。

    版式对齐实测：顶层 ``#1 varint + #2 message{ 块描述符 + 12B + 12B + 48B + 密文 } + #3 varint``，
    块描述符固定 ``#1=18 / #3=2 / #7=1 / #8=<块类型>``；``model_identifier`` 非空时按实测位置
    补一个 ``#6``（真实签名里就是模型标识明文，路径 ``.2.1.6``）。
    """
    descriptor = (b"\x08\x12" + b"\x18\x02"
                  + (b"\x32" + encode_varint(len(model_identifier))
                     + model_identifier.encode("utf-8") if model_identifier else b"")
                  + b"\x38\x01"
                  + b"\x42" + encode_varint(len(block_type)) + block_type.encode("utf-8"))
    inner = (b"\x0a" + encode_varint(len(descriptor)) + descriptor
             + b"\x12\x0c" + b"\x00" * 12
             + b"\x1a\x0c" + b"\x00" * 12
             + b"\x22\x30" + b"\x00" * 48
             + b"\x2a" + encode_varint(cipher_length) + b"\x00" * cipher_length)
    return encode_signature(b"\x08" + bytes([version]) + b"\x12"
                            + encode_varint(len(inner)) + inner + b"\x18\x01")
