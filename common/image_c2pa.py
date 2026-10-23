"""C2PA 内容签名：图片「官方指纹」的读取与汇总。

背景：OpenAI gpt-image 等官方出图会在文件内嵌 C2PA（JUMBF）清单，里面有
签名主体、签发者、生成器、不可见水印动作与完整性校验结果。报告里呈现为：

* 图片指纹：官方模型签名（C2PA）；
* 签名主体 / 签发者：签名证书链的 DN；
* 指纹生成器：claim generator + softwareAgent；
* 签名校验：算法与内容签名校验结果；
* 水印：`c2pa.watermarked.*` 动作。

口径（**重要**）：

* C2PA 数据只作**证据**，不参与准入判定；
* 本机信任库未必内置厂商证书链，`signingCredential.untrusted` 一类提示只说明
  「本机不认识这张证书」，不能据此认定签名伪造——内容是否被篡改以
  `claimSignature.validated` / `claimSignature.insideValidity` 为准；
* 无 C2PA 清单 ≠ 无水印：SynthID 一类不可见水印不以 C2PA 记录。
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# 报告阶段逐图读取 C2PA 是 O(n) 次文件解析：超过上限只扫前 N 张，
# 其余按未扫描记录（完整逐图数据可后续用脚本补扫）。
C2PA_SCAN_MAX_IMAGES = 100
# 逐图清单里列出的文件上限（完整分组数据在 summary.json 的 image_c2pa）。
PROFILE_MEMBER_LIMIT = 8

try:  # c2pa / cryptography 缺失时整个模块降级为「不可用」，不影响出报告
    from c2pa import Reader as _C2paReader
except Exception:  # pragma: no cover - 环境相关
    _C2paReader = None

try:
    from cryptography import x509 as _x509
except Exception:  # pragma: no cover - 环境相关
    _x509 = None

# 签名主体/签发者等 DN 里常见 OID 的可读缩写（rfc4514_string 对冷门 OID 输出点分数字）。
_OID_LABELS = {
    "1.2.840.113549.1.9.1": "emailAddress",
}

WATERMARK_ACTIONS = ("c2pa.watermarked",)
# 「内容未被篡改」的判定依据：claim 签名校验通过或时间戳在有效期内。
_TRUSTED_CODES = ("claimSignature.validated", "claimSignature.insideValidity")


def available() -> bool:
    """c2pa 解析库与证书解析库是否就绪。"""
    return _C2paReader is not None and _x509 is not None


def _dn_directory_order(name: Any) -> str:
    """把 x509 Name 转成「C=US, O=..., CN=...」的目录序（最显著分量在前）。"""
    text = name.rfc4514_string()
    parts: list[str] = []
    buffer = ""
    for index, char in enumerate(text):
        if char == "," and (index == 0 or text[index - 1] != "\\"):
            parts.append(buffer)
            buffer = ""
        else:
            buffer += char
    parts.append(buffer)
    parts = [
        _OID_LABELS.get(part.split("=", 1)[0], part.split("=", 1)[0]) + "=" + part.split("=", 1)[1]
        if "=" in part
        else part
        for part in reversed(parts)
    ]
    # rfc4514 的转义逗号（值内 "\,"）在展示时还原，markdown 表格不需要转义。
    return ", ".join(parts).replace("\\,", ",")


def _extract_certs(raw: bytes) -> list[Any]:
    """从文件原始字节里扫出 DER 证书（COSE x5chain 连续存放，直接可解析）。"""
    if _x509 is None:
        return []
    certs: list[Any] = []
    cursor = 0
    while True:
        cursor = raw.find(b"\x30\x82", cursor)
        if cursor < 0 or cursor + 4 > len(raw):
            break
        length = int.from_bytes(raw[cursor + 2 : cursor + 4], "big")
        if 200 < length < 4000 and cursor + 4 + length <= len(raw):
            try:
                certs.append(_x509.load_der_x509_certificate(raw[cursor : cursor + 4 + length]))
                cursor += 4 + length
                continue
            except Exception:
                pass
        cursor += 2
    return certs


def _signing_cert(raw: bytes, common_name: str | None) -> Any | None:
    """按 subject CN 匹配签名证书；匹配不到就取链上第一张。"""
    certs = _extract_certs(raw)
    if not certs:
        return None
    if common_name:
        for cert in certs:
            try:
                attrs = cert.subject.get_attributes_for_oid(_x509.NameOID.COMMON_NAME)
                if attrs and attrs[0].value == common_name:
                    return cert
            except Exception:
                continue
    return certs[0]


def _actions_of(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    for assertion in manifest.get("assertions") or []:
        label = str(assertion.get("label") or "")
        if label.startswith("c2pa.actions"):
            data = assertion.get("data") or {}
            return list(data.get("actions") or [])
    return []


def profile_from_manifest(manifest: dict[str, Any], raw: bytes) -> dict[str, Any] | None:
    """把一份 C2PA manifest 解析成报告用的「签名画像」。"""
    active_id = manifest.get("active_manifest")
    active = (manifest.get("manifests") or {}).get(active_id) if isinstance(active_id, str) else None
    if not isinstance(active, dict):
        active = next(iter((manifest.get("manifests") or {}).values()), None)
    if not isinstance(active, dict):
        return None

    generators = active.get("claim_generator_info") or []
    generator = generators[0].get("name") if generators else None

    actions = _actions_of(active)
    software_agent = None
    watermarked = False
    source_type = None
    for action in actions:
        agent = action.get("softwareAgent")
        if isinstance(agent, dict) and agent.get("name") and software_agent is None:
            version = f" · {agent['version']}" if agent.get("version") else ""
            software_agent = f"{agent['name']}{version}"
        if str(action.get("action") or "").startswith(WATERMARK_ACTIONS):
            watermarked = True
        if action.get("digitalSourceType") and source_type is None:
            source_type = str(action["digitalSourceType"]).rsplit("/", 1)[-1]

    signature = active.get("signature_info") or {}
    alg = (signature.get("alg") or "").upper() or None
    common_name = signature.get("common_name")

    subject_dn = issuer_dn = None
    cert = _signing_cert(raw, common_name)
    if cert is not None:
        subject_dn = _dn_directory_order(cert.subject)
        issuer_dn = _dn_directory_order(cert.issuer)

    validation = (manifest.get("validation_results") or {}).get("activeManifest") or {}
    success_codes = sorted({str(item.get("code")) for item in validation.get("success") or []})
    failure_items = list(validation.get("failure") or [])
    failure_codes = sorted({str(item.get("code")) for item in failure_items})
    verified = any(code in _TRUSTED_CODES for code in success_codes)

    return {
        "kind": "c2pa_signature",
        "generator": generator,
        "software_agent": software_agent,
        "signature_alg": alg,
        "subject": subject_dn or common_name,
        "issuer": issuer_dn or signature.get("issuer"),
        "signed_at": signature.get("time"),
        "watermarked": watermarked,
        "source_type": source_type,
        "verified": verified,
        "success_codes": success_codes,
        "failure_codes": failure_codes,
        "failure_explanations": [
            {"code": str(item.get("code")), "explanation": item.get("explanation")}
            for item in failure_items
        ],
    }


def read_c2pa_profile(raw: bytes) -> dict[str, Any] | None:
    """读一段图片字节里的 C2PA 清单并解析成画像；无清单/解析失败返回 None。"""
    if _C2paReader is None:
        return None
    try:
        with _C2paReader(io.BytesIO(raw)) as reader:
            payload = reader.json()
        manifest = json.loads(payload) if isinstance(payload, str) else payload
    except Exception:
        return None
    return profile_from_manifest(manifest, raw)


def _profile_key(profile: dict[str, Any]) -> tuple:
    return (
        profile.get("generator"),
        profile.get("software_agent"),
        profile.get("signature_alg"),
        profile.get("subject"),
        profile.get("issuer"),
        profile.get("watermarked"),
        profile.get("verified"),
    )


def build_image_c2pa(
    rows: Iterable[dict[str, Any]], result_dir: str | Path, max_images: int = C2PA_SCAN_MAX_IMAGES
) -> dict[str, Any]:
    """扫结果目录里已落盘的图片，汇总 C2PA 签名画像。

    C2PA 只能从**原文件**读，所以只在开启图片保留时可用；解析在报告收尾
    阶段做，不占压测请求的热路径。
    """
    result_path = Path(result_dir)
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        for image in row.get("images") or []:
            if not isinstance(image, dict) or image.get("saved") is False:
                continue
            # 压测套件写 `file`，4K 验真写 `path`（均相对结果目录）。
            if not (image.get("file") or image.get("path")):
                continue
            candidates.append((row, image))

    summary: dict[str, Any] = {
        "available": available(),
        "total_images": len(candidates),
        "scanned": 0,
        "signed": 0,
        "unsigned": 0,
        "profiles": [],
        "scan_truncated": False,
    }
    if not summary["available"]:
        summary["reason"] = "c2pa 解析库未安装（pip install c2pa-python）"
        return summary
    if not candidates:
        return summary

    selected = candidates[:max_images]
    summary["scan_truncated"] = len(candidates) > len(selected)

    groups: dict[tuple, dict[str, Any]] = {}
    for row, image in selected:
        relative = str(image.get("file") or image.get("path"))
        path = result_path / relative
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        summary["scanned"] += 1
        profile = read_c2pa_profile(raw)
        if profile is None:
            summary["unsigned"] += 1
            continue
        summary["signed"] += 1
        key = _profile_key(profile)
        group = groups.setdefault(key, {"profile": profile, "members": []})
        if len(group["members"]) < PROFILE_MEMBER_LIMIT:
            group["members"].append(f"#{row.get('index') if row.get('index') is not None else '?'} {relative}")

    ordered = sorted(groups.values(), key=lambda group: -len(group["members"]))
    for group in ordered:
        entry = dict(group["profile"])
        entry["count"] = len(group["members"])
        entry["members"] = group["members"]
        summary["profiles"].append(entry)
    return summary


def _check_line(profile: dict[str, Any]) -> str:
    alg = profile.get("signature_alg") or "未知算法"
    state = "内容签名有效" if profile.get("verified") else "内容签名未通过"
    return f"{alg} · {state}"


def _watermark_line(profile: dict[str, Any]) -> str:
    return "有不可见水印" if profile.get("watermarked") else "—"


def c2pa_lines(summary: dict[str, Any] | None) -> list[str]:
    """渲染报告的「内容签名（C2PA）」小节；拼进图片指纹章节之后。"""
    lines = ["", "### 内容签名（C2PA）", ""]
    if not summary or not summary.get("available"):
        reason = (summary or {}).get("reason") or "c2pa 解析库未安装"
        lines += [f"> C2PA 签名数据不可用：{reason}。", ""]
        return lines
    if not summary.get("scanned"):
        lines += [
            "> 本次没有可扫描的图片原文件（未保存图片或目录为空），无法读取 C2PA 签名。",
            "",
        ]
        return lines

    lines += [
        "| 签名 | 生成器 | 签名主体 | 签发者 | 签名校验 | 水印 | 图片数 |",
        "|---|---|---|---|---|---|---:|",
    ]
    profiles = summary.get("profiles") or []
    if profiles:
        for profile in profiles:
            generator = profile.get("generator") or "—"
            agent = profile.get("software_agent")
            if agent:
                generator = f"{generator}（{agent}）"
            lines.append(
                "| 官方模型签名 | {} | {} | {} | {} | {} | {} |".format(
                    generator,
                    profile.get("subject") or "—",
                    profile.get("issuer") or "—",
                    _check_line(profile),
                    _watermark_line(profile),
                    profile.get("count", 0),
                )
            )
    else:
        lines.append("| 未检出 | — | — | — | — | — | 0 |")
    lines.append("")

    if summary.get("unsigned"):
        lines.append(
            f"> {summary['unsigned']} 张图片未检出 C2PA 清单。无清单 ≠ 无水印："
            "SynthID 一类不可见水印不以 C2PA 记录。"
        )
    if summary.get("scan_truncated"):
        lines.append(
            f"> 图片超过 {C2PA_SCAN_MAX_IMAGES} 张，本节只扫描前 {summary.get('scanned')} 张。"
        )
    if any(profile.get("failure_codes") for profile in profiles):
        lines.append(
            "> `signingCredential.untrusted` 一类失败表示本机信任库未内置该厂商证书链，"
            "不代表签名伪造；内容是否被篡改以「内容签名」一列的校验结果为准。"
        )
    lines.append(
        "> C2PA 数据只作证据、不参与准入判定：它证明「文件由该签名者签发且未被篡改」，"
        "不评价生成质量。"
    )
    lines.append("")
    return lines
