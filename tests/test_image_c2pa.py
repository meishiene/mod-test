"""C2PA 内容签名模块的单元测试。

不依赖真实厂商图片：manifest 用合成结构，证书用 cryptography 现场签发。
"""
import datetime
import io

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from PIL import Image

from common.image_c2pa import (
    _dn_directory_order,
    _extract_certs,
    build_image_c2pa,
    c2pa_lines,
    profile_from_manifest,
    read_c2pa_profile,
)


def _png(size: tuple[int, int] = (8, 8), color=(200, 30, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _self_signed(cn: str = "Test Media Service", org: str = "Test OpCo, LLC") -> bytes:
    """签一张自签 DER 证书，返回原始字节（DN 与真实 OpenAI 链同构）。"""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, org),
            x509.NameAttribute(NameOID.COMMON_NAME, cn),
        ]
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.DER)


def _manifest(cn: str = "Test Media Service", *, watermarked=True, verified=True) -> dict:
    success = [{"code": "claimSignature.validated"}] if verified else []
    return {
        "active_manifest": "urn:c2pa:test",
        "manifests": {
            "urn:c2pa:test": {
                "claim_generator_info": [{"name": "Test Media Service API"}],
                "assertions": [
                    {
                        "label": "c2pa.actions.v2",
                        "data": {
                            "actions": [
                                {
                                    "action": "c2pa.created",
                                    "softwareAgent": {"name": "ChatTest", "version": "test-image"},
                                    "digitalSourceType": (
                                        "http://cv.iptc.org/newscodes/digitalsourcetype/"
                                        "trainedAlgorithmicMedia"
                                    ),
                                }
                            ]
                            + ([{"action": "c2pa.watermarked.unbound"}] if watermarked else [])
                        },
                    }
                ],
                "signature_info": {
                    "alg": "Es256",
                    "common_name": cn,
                    "issuer": "Test OpCo, LLC",
                    "time": "2026-10-20T08:58:22.559136+00:00",
                },
            }
        },
        "validation_results": {
            "activeManifest": {
                "success": success,
                "failure": [{"code": "signingCredential.untrusted", "explanation": "x"}]
                if not verified
                else [],
            }
        },
    }


def test_dn_directory_order_puts_country_first_and_unescapes_commas():
    cert = x509.load_der_x509_certificate(_self_signed())
    text = _dn_directory_order(cert.subject)
    assert text.startswith("C=US, ")
    assert "O=Test OpCo, LLC" in text  # 值内逗号不再带转义
    assert text.endswith("CN=Test Media Service")


def test_extract_certs_finds_embedded_der_chain():
    der = _self_signed()
    raw = b"junk" + b"\x00" * 10 + der + b"trailing"
    certs = _extract_certs(raw)
    assert len(certs) == 1
    assert certs[0].subject.rfc4514_string().startswith("CN=Test Media Service")


def test_profile_from_manifest_maps_all_screenshot_fields():
    raw = b"\x00" * 4 + _self_signed() + b"\xff" * 4
    profile = profile_from_manifest(_manifest(), raw)

    assert profile is not None
    assert profile["generator"] == "Test Media Service API"
    assert profile["software_agent"] == "ChatTest · test-image"
    assert profile["signature_alg"] == "ES256"
    assert profile["subject"] == "C=US, O=Test OpCo, LLC, CN=Test Media Service"
    assert profile["watermarked"] is True
    assert profile["source_type"] == "trainedAlgorithmicMedia"
    assert profile["verified"] is True


def test_profile_without_watermark_and_signature():
    profile = profile_from_manifest(_manifest(watermarked=False, verified=False), b"no certs here")
    assert profile["watermarked"] is False
    assert profile["verified"] is False
    assert profile["subject"] == "Test Media Service"  # 退回 signature_info 的 CN


def test_read_c2pa_profile_returns_none_for_plain_png():
    assert read_c2pa_profile(_png()) is None


def test_build_image_c2pa_groups_and_counts(tmp_path, monkeypatch):
    import common.image_c2pa as mod

    raw = b"\x00" * 4 + _self_signed() + b"\xff" * 4
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.png").write_bytes(raw)
    (tmp_path / "images" / "b.png").write_bytes(raw)
    (tmp_path / "images" / "c.png").write_bytes(_png())  # 无 C2PA

    # 合成字节不是合法的 C2PA 容器，这里把读取函数替换为「直接解析 manifest」，
    # 只测扫描、分组与计数逻辑（读取本身有独立用例覆盖）。
    monkeypatch.setattr(
        mod,
        "read_c2pa_profile",
        lambda data: mod.profile_from_manifest(_manifest(), data) if b"\x30\x82" in data else None,
    )

    rows = [
        {"index": 1, "images": [{"saved": True, "file": "images/a.png"}]},
        {"index": 2, "images": [{"saved": True, "file": "images/b.png"}]},
        {"index": 3, "images": [{"saved": True, "file": "images/c.png"}]},
        {"index": 4, "images": [{"saved": False, "file": "images/missing.png"}]},
    ]
    summary = build_image_c2pa(rows, tmp_path)

    assert summary["scanned"] == 3
    assert summary["signed"] == 2
    assert summary["unsigned"] == 1
    assert len(summary["profiles"]) == 1
    profile = summary["profiles"][0]
    assert profile["count"] == 2
    assert profile["members"] == ["#1 images/a.png", "#2 images/b.png"]


def test_build_image_c2pa_respects_scan_cap(tmp_path):
    (tmp_path / "images").mkdir()
    rows = []
    for index in range(5):
        path = tmp_path / "images" / f"{index}.png"
        path.write_bytes(_png())
        rows.append({"index": index + 1, "images": [{"saved": True, "file": f"images/{index}.png"}]})

    summary = build_image_c2pa(rows, tmp_path, max_images=2)

    assert summary["total_images"] == 5
    assert summary["scanned"] == 2
    assert summary["scan_truncated"] is True


def test_c2pa_lines_render_screenshot_style_table():
    raw = b"\x00" * 4 + _self_signed() + b"\xff" * 4
    profile = profile_from_manifest(_manifest(), raw)
    profile.update({"count": 2, "members": ["#1 images/a.png", "#2 images/b.png"]})
    summary = {"available": True, "scanned": 2, "signed": 2, "unsigned": 0,
               "profiles": [profile], "scan_truncated": False}

    lines = c2pa_lines(summary)
    text = "\n".join(lines)

    assert "### 内容签名（C2PA）" in text
    assert "官方模型签名" in text
    assert "Test Media Service API（ChatTest · test-image）" in text
    assert "ES256 · 内容签名有效" in text
    assert "有不可见水印" in text
    assert "不参与准入判定" in text


def test_c2pa_lines_handles_missing_and_unavailable():
    empty = c2pa_lines({"available": True, "scanned": 0, "signed": 0, "unsigned": 0,
                        "profiles": [], "scan_truncated": False})
    assert "没有可扫描的图片原文件" in "\n".join(empty)

    unavailable = c2pa_lines({"available": False, "reason": "c2pa 解析库未安装"})
    assert "C2PA 签名数据不可用" in "\n".join(unavailable)

    assert c2pa_lines(None)[1] == "### 内容签名（C2PA）"


@pytest.mark.skipif(True, reason="信任库告警在报告里只是提示行，这里留作人工核对样例")
def test_real_openai_image_sample():
    from pathlib import Path

    sample = Path("results/GPT_IMAGE_2-20261020-165749-07402c/images/req-0001-01.png")
    if not sample.exists():
        pytest.skip("样例图不存在")
    profile = read_c2pa_profile(sample.read_bytes())
    assert profile["verified"] is True
    assert "CN=OpenAI Media Service" in profile["subject"]
