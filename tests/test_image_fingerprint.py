"""图片指纹聚合与报告渲染（gpt_image / gemini_image 报告共用）。

重点钉住三件事：字节一致判重、近似重复（同图重编码）分组、以及「指纹只作证据、
不参与准入」在报告尾注里写清楚。
"""
import hashlib
import io

from PIL import Image, ImageDraw

from common.image_fingerprint import (
    DETAIL_LIMIT,
    NEAR_COLOR_DISTANCE,
    NEAR_DUPLICATE_DISTANCE,
    build_image_fingerprints,
    color_distance,
    difference_hash,
    fingerprint_lines,
    hamming_distance,
)
from common.image_store import probe_image


def _png(size=(64, 64), color=(10, 120, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _scene(pattern: str = "bright-dark") -> bytes:
    """构造一张有真实明暗结构的小图（纯渐变会退化成全零 dHash，测不出东西）。"""
    shades = {
        "bright-dark": (240, 20, 200, 30, 250, 40, 210, 25, 235),
        "inverted": (25, 235, 40, 210, 30, 240, 20, 200, 250),
        "mixed": (30, 220, 60, 190, 240, 50, 205, 35, 225),
    }[pattern]
    image = Image.new("RGB", (90, 90), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    for column, shade in enumerate(shades):
        draw.rectangle([column * 10, 0, column * 10 + 9, 89], fill=(shade, shade, shade))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _row(index: int, payloads, model="gpt-image-2"):
    images = []
    for sequence, raw in enumerate(payloads, 1):
        probe = probe_image(raw)
        images.append(
            {
                "sequence": sequence,
                "file": f"images/req-{index:04d}-{sequence:02d}.png",
                "saved": True,
                "format": probe["format"],
                "width": probe["width"],
                "height": probe["height"],
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "dhash": probe["dhash"],
                "chash": probe["chash"],
            }
        )
    return {"index": index, "model": model, "images": images}


def test_hamming_distance_handles_invalid_input():
    assert hamming_distance("0000000000000000", "0000000000000001") == 1
    assert hamming_distance("ffffffffffffffff", "0000000000000000") == 64
    assert hamming_distance(None, "0000000000000000") is None
    assert hamming_distance("not-hex", "0000000000000000") is None


def test_identical_bytes_are_counted_as_duplicates():
    raw = _png()
    stats = build_image_fingerprints([_row(1, [raw, raw]), _row(2, [raw])])

    assert stats["images"] == 3
    assert stats["unique_sha256"] == 1
    assert stats["duplicate_images"] == 2
    assert stats["duplicate_group_count"] == 1
    assert stats["groups"][0]["kind"] == "same_bytes"
    assert [member["request"] for member in stats["groups"][0]["members"]] == [1, 1, 2]
    assert [item["verdict"] for item in stats["detail"]] == ["重复×3"] * 3
    assert all(len(item["sha256"]) == 16 for item in stats["detail"])


def test_reencoded_image_lands_in_a_near_duplicate_group():
    original = _scene()
    # 夹具本身必须有真实明暗结构，否则 dHash 全是 0，测不到东西。
    assert probe_image(original)["dhash"] not in {None, "0000000000000000"}
    buffer = io.BytesIO()
    Image.open(io.BytesIO(original)).convert("RGB").save(buffer, format="JPEG", quality=92)
    reencoded = buffer.getvalue()
    assert original != reencoded

    stats = build_image_fingerprints([_row(1, [original]), _row(2, [reencoded])])

    assert stats["unique_sha256"] == 2
    assert stats["duplicate_images"] == 0
    assert stats["near_duplicate_group_count"] == 1
    assert stats["groups"][0]["kind"] == "near_duplicate"
    assert 0 <= stats["groups"][0]["max_distance"] <= NEAR_DUPLICATE_DISTANCE
    # 同图重编码属于「基本可判为同一张图」的强口径。
    assert stats["same_image_group_count"] == 1
    assert [item["verdict"] for item in stats["detail"]] == ["近似 G1", "近似 G1"]
    assert stats["near_duplicate_distance"] == NEAR_DUPLICATE_DISTANCE


def test_distinct_images_stay_unique():
    left, right = _scene("bright-dark"), _scene("inverted")
    assert (hamming_distance(probe_image(left)["dhash"], probe_image(right)["dhash"]) or 0) > 6
    stats = build_image_fingerprints([_row(1, [left]), _row(2, [right]), _row(3, [_png()])])

    assert stats["unique_sha256"] == 3
    assert stats["duplicate_group_count"] == 0
    assert stats["near_duplicate_group_count"] == 0
    assert stats["groups"] == []
    assert [item["verdict"] for item in stats["detail"]] == ["唯一"] * 3


def test_solid_colors_are_not_near_duplicates():
    # 纯色图的明暗 dHash 全是 0，靠配色指纹挡住误判。
    red = _png((64, 64), (220, 20, 20))
    blue = _png((64, 64), (20, 20, 220))
    assert probe_image(red)["dhash"] == probe_image(blue)["dhash"]

    stats = build_image_fingerprints([_row(1, [red]), _row(2, [blue])])

    assert stats["near_duplicate_group_count"] == 0
    assert [item["verdict"] for item in stats["detail"]] == ["唯一", "唯一"]


def test_same_structure_with_other_palette_is_not_a_duplicate():
    """同一个构图只换配色：明暗差 0 位，但配色差要把它们分开。"""

    def circle(color):
        image = Image.new("RGB", (256, 256), (245, 245, 240))
        draw = ImageDraw.Draw(image)
        draw.ellipse([48, 48, 208, 208], fill=color)
        draw.rectangle([20, 20, 60, 60], fill=(40, 40, 40))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    red, blue = probe_image(circle((200, 30, 30))), probe_image(circle((30, 30, 200)))
    assert hamming_distance(red["dhash"], blue["dhash"]) == 0
    assert color_distance(red["chash"], blue["chash"]) > NEAR_COLOR_DISTANCE

    stats = build_image_fingerprints([_row(1, [circle((200, 30, 30))]), _row(2, [circle((30, 30, 200))])])

    assert stats["near_duplicate_group_count"] == 0


def test_skipped_and_unsaved_images_are_ignored():
    rows = [
        {"index": 1, "model": "m", "images": [{"saved": False, "bytes": 10, "error": "x"}]},
        {"index": 2, "model": "m", "images": []},
        {"index": 3, "model": "m"},
    ]

    stats = build_image_fingerprints(rows)

    assert stats["images"] == 0
    assert stats["detail"] == []
    lines = fingerprint_lines(stats)
    assert "## 图片指纹" in lines
    assert any("没有可用的图片指纹" in line for line in lines)


def test_images_without_saved_flag_are_counted():
    """4K token 验真的请求行没有 saved 字段，只要有 sha256 就要算进指纹。"""
    raw = _png()
    row = _row(1, [raw])
    for image in row["images"]:
        del image["saved"]

    stats = build_image_fingerprints([row])

    assert stats["images"] == 1
    assert stats["unique_sha256"] == 1


def test_detail_is_capped_but_counts_stay_exact():
    raw = _png()
    rows = [_row(index, [raw]) for index in range(1, DETAIL_LIMIT + 5)]

    stats = build_image_fingerprints(rows)

    assert stats["images"] == DETAIL_LIMIT + 4
    assert stats["duplicate_images"] == DETAIL_LIMIT + 3
    assert len(stats["detail"]) == DETAIL_LIMIT
    assert stats["detail_truncated"] is True


def test_difference_hash_tolerates_non_image_input():
    assert difference_hash(Image.new("L", (1, 1))) is not None
    assert difference_hash(None) is None  # type: ignore[arg-type]


def test_report_section_states_evidence_only_semantics():
    raw = _png()
    stats = build_image_fingerprints([_row(1, [raw]), _row(2, [raw])])
    text = "\n".join(fingerprint_lines(stats))

    assert "## 图片指纹" in text
    assert "| 指纹图片数 | 2 |" in text
    assert "| 唯一内容（SHA-256） | 1 |" in text
    assert "### 重复分组" in text
    assert "### 逐图指纹" in text
    assert "SHA-256（前十六位）" in text
    assert "其中基本可判为同一张" in text
    assert "不参与准入判定" in text
    assert "requests.jsonl" in text
