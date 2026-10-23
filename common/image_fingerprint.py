"""图片指纹：给每张落盘的生成图片算内容指纹，并在报告里给出逐图数据与重复情况。

两个图片套件（``gpt_image`` / ``gemini_image``）共用本模块，保证同一套口径：

* **SHA-256**：落盘字节的完整摘要，用来判定「两张图是否完全一致」；
* **dHash**（difference hash）：9×8 灰度差分哈希（64 位，16 位十六进制），
  描述**明暗结构**——同一张图重新编码（PNG↔JPEG）、加壳/加水印之后 SHA-256
  必然变，dHash 通常只差 0～6 位；
* **chash**（配色指纹）：4×4 网格的平均色（48 字节，96 位十六进制）。只靠明暗
  dHash 的话，「同一个构图换了一套配色」「纯色图」都会被判成同一张——网格里的
  绝对色值把它们分开，而且重编码后几乎不变。

为什么不直接用「三通道 dHash」：差分哈希只记相邻像素的强弱方向，换色后每个通道
的梯度结构还是一样（红圆→蓝圆只交换了 R/B 通道），分不出配色；只有取绝对色值
才分得开。

为什么值得进报告：逐图 sha256 一直只躺在 ``requests.jsonl`` 里，报告读者看不到。
中转渠道复用同一张图（缓存、占位图、假生成）是真实存在的风险，把指纹摆到报告
里才能被直接审阅。

判定口径（**重要**）：

* 指纹只作**证据**，不参与准入判定——同一个提示词下的正常生成也可能高度相似，
  重复只说明「值得复核」，单凭指纹不能判假；
* 反过来，指纹全不相同也不能证明上游没有注入或改写（改分辨率、重压缩都会改变
  字节），所以本模块只描述事实，不下结论。
"""
from __future__ import annotations

from typing import Any, Iterable

from PIL import Image

# 9×8 灰度：每行 8 组「相邻像素谁更亮」，共 64 位。
DHASH_WIDTH = 9
DHASH_HEIGHT = 8
# 4×4 网格 → 16 个平均色 → 48 字节。
COLOR_GRID = 4
# 近似重复阈值：明暗汉明距离（位）与配色平均绝对差（0～255，逐通道逐格平均）。
NEAR_DUPLICATE_DISTANCE = 6
NEAR_COLOR_DISTANCE = 12
# 「基本可判为同一张图」的强口径：明暗 ≤ 2 位且配色差 ≤ 4。
SAME_IMAGE_DISTANCE = 2
SAME_COLOR_DISTANCE = 4
# 近似重复是 O(n²) 比对：图片太多时只做 SHA-256 判重，避免收尾阶段卡住。
NEAR_DUPLICATE_MAX_IMAGES = 2000
# 报告与 summary 里的逐图明细上限（完整数据始终在 requests.jsonl）。
DETAIL_LIMIT = 200
GROUP_LIMIT = 20
SHA256_DISPLAY_CHARS = 16


def _resample():
    return getattr(Image, "Resampling", Image).BOX


def _dhash_hex(band: Image.Image) -> str | None:
    """对一张 9×8 的单通道图算出 64 位差分哈希（16 位十六进制）。"""
    try:
        # tobytes() 在 "L" 模式下一个字节一个像素，比 getdata() 快且不触发弃用警告。
        pixels = band.tobytes()
    except Exception:
        return None
    if len(pixels) < DHASH_WIDTH * DHASH_HEIGHT:
        return None
    bits = 0
    for row in range(DHASH_HEIGHT):
        offset = row * DHASH_WIDTH
        for column in range(DHASH_WIDTH - 1):
            left = pixels[offset + column]
            right = pixels[offset + column + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return f"{bits:016x}"


def difference_hash(image: Image.Image) -> str | None:
    """算一张 PIL 图片的明暗 dHash（16 位十六进制）；失败返回 None。

    只取第一帧（动图按首帧算），失败不得影响落盘——指纹是附加证据。
    """
    try:
        return _dhash_hex(image.convert("L").resize((DHASH_WIDTH, DHASH_HEIGHT), _resample()))
    except Exception:  # 解码异常/不支持的色彩模式：留空，不阻断
        return None


def color_hash(image: Image.Image) -> str | None:
    """算配色指纹：4×4 网格平均色的十六进制串（16 格 × RGB = 96 位十六进制）。"""
    try:
        small = image.convert("RGB").resize((COLOR_GRID, COLOR_GRID), _resample())
        return small.tobytes().hex()
    except Exception:
        return None


def image_signature(image: Image.Image) -> dict[str, str | None]:
    """一次解码里取齐两项指纹（明暗 dHash + 配色 chash），落盘时写入请求行。"""
    return {
        "dhash": difference_hash(image),
        "chash": color_hash(image),
    }


def hamming_distance(left: str | None, right: str | None) -> int | None:
    """两个十六进制指纹的汉明距离；任一为空或格式非法时返回 None。"""
    try:
        return bin(int(str(left), 16) ^ int(str(right), 16)).count("1")
    except (TypeError, ValueError):
        return None


def color_distance(left: str | None, right: str | None) -> float | None:
    """配色指纹的平均绝对差（0～255，逐通道逐格平均）；任一侧缺失时返回 None。"""
    try:
        left_bytes = bytes.fromhex(str(left))
        right_bytes = bytes.fromhex(str(right))
    except (TypeError, ValueError):
        return None
    if not left_bytes or len(left_bytes) != len(right_bytes):
        return None
    return round(
        sum(abs(a - b) for a, b in zip(left_bytes, right_bytes)) / len(left_bytes), 2
    )


def _sort_key(entry: dict[str, Any]) -> tuple[int, int, str]:
    request = entry.get("request")
    index = request if isinstance(request, int) else 10**9
    return (0 if isinstance(request, int) else 1, index, str(entry.get("file") or ""))


def _image_entries(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 requests.jsonl 的行摊平成逐图指纹条目（只取真正落盘且有 sha256 的）。"""
    entries: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        images = row.get("images")
        if not isinstance(images, list):
            continue
        for image in images:
            # 压测套件用 saved 标记落盘结果；4K token 验真的行没有该字段（失败图片
            # 根本不进 images 列表），所以只排除显式 saved=False。
            if not isinstance(image, dict) or image.get("saved") is False:
                continue
            sha256 = image.get("sha256")
            if not isinstance(sha256, str) or not sha256:
                continue
            dhash = image.get("dhash")
            chash = image.get("chash")
            entries.append(
                {
                    "request": row.get("index"),
                    "model": row.get("model"),
                    "file": image.get("file"),
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "bytes": image.get("bytes"),
                    "sha256": sha256,
                    "dhash": dhash if isinstance(dhash, str) and dhash else None,
                    "chash": chash if isinstance(chash, str) and chash else None,
                }
            )
    entries.sort(key=_sort_key)
    return entries


def _same_bytes_groups(entries: list[dict[str, Any]]) -> list[list[int]]:
    buckets: dict[str, list[int]] = {}
    for position, entry in enumerate(entries):
        buckets.setdefault(entry["sha256"], []).append(position)
    groups = [members for members in buckets.values() if len(members) > 1]
    groups.sort(key=lambda members: members[0])
    return groups


def _near_groups(entries: list[dict[str, Any]]) -> list[tuple[list[int], int, float]]:
    """按「明暗 dHash + 配色 chash」聚类（并查集），字节重复另行统计。

    返回 ``(成员下标, 组内最大明暗差, 组内最大配色差)``：两个距离给报告分级用——
    重编码后的同一张图通常都差 0～2 位，而「构图相同、配色/内容不同」会散在更大
    的距离上。
    """
    parent = list(range(len(entries)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    distances: dict[tuple[int, int], tuple[int, float]] = {}
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if entries[i]["sha256"] == entries[j]["sha256"]:
                continue
            luma = hamming_distance(entries[i]["dhash"], entries[j]["dhash"])
            if luma is None or luma > NEAR_DUPLICATE_DISTANCE:
                continue
            # 明暗相同但配色差得远 = 不同的图（换个配色的同一构图、纯色图）。
            color = color_distance(entries[i]["chash"], entries[j]["chash"])
            if color is not None and color > NEAR_COLOR_DISTANCE:
                continue
            union(i, j)
            distances[(i, j)] = (luma, color if color is not None else 0.0)
    clusters: dict[int, list[int]] = {}
    for position in range(len(entries)):
        clusters.setdefault(find(position), []).append(position)
    groups: list[tuple[list[int], int, float]] = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        pairs = [
            distances.get((left, right))
            for index, left in enumerate(members)
            for right in members[index + 1 :]
        ]
        known = [value for value in pairs if value is not None]
        luma_max = max((value[0] for value in known), default=0)
        color_max = max((value[1] for value in known), default=0.0)
        groups.append((members, luma_max, color_max))
    groups.sort(key=lambda item: item[0][0])
    return groups


def _member(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "request": entry["request"],
        "file": entry["file"],
        "sha256": str(entry["sha256"])[:SHA256_DISPLAY_CHARS],
        "dhash": entry["dhash"],
    }


def build_image_fingerprints(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """汇总一批请求行的图片指纹：唯一性、重复分组与逐图明细。"""
    entries = _image_entries(rows)
    same_groups = _same_bytes_groups(entries)
    near_skipped = len(entries) > NEAR_DUPLICATE_MAX_IMAGES
    near_groups = [] if near_skipped else _near_groups(entries)

    near_group_of: dict[int, int] = {}
    for number, (members, _luma, _color) in enumerate(near_groups, 1):
        for position in members:
            near_group_of[position] = number
    sha_counts: dict[str, int] = {}
    for entry in entries:
        sha_counts[entry["sha256"]] = sha_counts.get(entry["sha256"], 0) + 1

    detail: list[dict[str, Any]] = []
    for position, entry in enumerate(entries):
        occurrences = sha_counts[entry["sha256"]]
        if occurrences > 1:
            verdict = f"重复×{occurrences}"
        elif position in near_group_of:
            verdict = f"近似 G{near_group_of[position]}"
        else:
            verdict = "唯一"
        detail.append({**_member(entry), "model": entry["model"], "width": entry["width"],
                       "height": entry["height"], "bytes": entry["bytes"], "verdict": verdict})

    groups: list[dict[str, Any]] = [
        {"kind": "same_bytes", "members": [_member(entries[i]) for i in members]}
        for members in same_groups
    ]
    groups += [
        {
            "kind": "near_duplicate",
            "max_distance": luma,
            "max_color_distance": color,
            "members": [_member(entries[i]) for i in members],
        }
        for members, luma, color in near_groups
    ]

    return {
        "images": len(entries),
        "unique_sha256": len(sha_counts),
        "duplicate_images": len(entries) - len(sha_counts),
        "duplicate_group_count": len(same_groups),
        "near_duplicate_group_count": len(near_groups),
        "near_duplicate_distance": NEAR_DUPLICATE_DISTANCE,
        "near_color_distance": NEAR_COLOR_DISTANCE,
        "same_image_distance": SAME_IMAGE_DISTANCE,
        "same_color_distance": SAME_COLOR_DISTANCE,
        "same_image_group_count": sum(
            1
            for _members, luma, color in near_groups
            if luma <= SAME_IMAGE_DISTANCE and color <= SAME_COLOR_DISTANCE
        ),
        "near_duplicate_skipped": near_skipped,
        "near_duplicate_max_images": NEAR_DUPLICATE_MAX_IMAGES,
        "groups": groups[:GROUP_LIMIT],
        "groups_truncated": len(groups) > GROUP_LIMIT,
        "detail": detail[:DETAIL_LIMIT],
        "detail_truncated": len(detail) > DETAIL_LIMIT,
        "detail_limit": DETAIL_LIMIT,
    }


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ") if value is not None else "—"


def fingerprint_lines(stats: dict[str, Any] | None) -> list[str]:
    """把指纹汇总渲染成报告章节（两个图片套件共用同一版式）。"""
    stats = stats or {}
    images = int(stats.get("images") or 0)
    lines = ["", "## 图片指纹", ""]
    if not images:
        lines.append(
            "> 本批没有可用的图片指纹：未保留图片（`--keep-images` 关闭）或图片全部落盘失败；"
            "需要指纹数据请开启图片保留后重跑。"
        )
        return lines

    lines += [
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 指纹图片数 | {images} |",
        f"| 唯一内容（SHA-256） | {stats.get('unique_sha256', 0)} |",
        f"| 重复图片（同一指纹出现多次） | {stats.get('duplicate_images', 0)} |",
    ]
    if stats.get("near_duplicate_skipped"):
        lines.append(
            f"| 近似重复检测 | 图片数超过 {stats.get('near_duplicate_max_images')} 张，"
            "本批只做 SHA-256 判重 |"
        )
    else:
        lines += [
            f"| 近似重复组（明暗差 ≤ {stats.get('near_duplicate_distance')} 位、"
            f"配色差 ≤ {stats.get('near_color_distance')}） "
            f"| {stats.get('near_duplicate_group_count', 0)} |",
            f"| 其中基本可判为同一张（明暗 ≤ {SAME_IMAGE_DISTANCE} 位且配色差 ≤ {SAME_COLOR_DISTANCE}） "
            f"| {stats.get('same_image_group_count', 0)} |",
        ]
    lines.append("")

    groups = stats.get("groups") or []
    if groups:
        lines += ["### 重复分组", "", "| 组 | 类型 | 成员（请求 · 文件 · SHA-256 前十六位 · dHash） |", "|---:|---|---|"]
        for number, group in enumerate(groups, 1):
            if group.get("kind") == "same_bytes":
                label = "字节一致"
            else:
                label = (
                    f"结构近似（明暗差 {_cell(group.get('max_distance'))} 位 / "
                    f"配色差 {_cell(group.get('max_color_distance'))}）"
                )
            members = "；".join(
                f"#{_cell(item.get('request'))} {_cell(item.get('file'))}"
                f"（{_cell(item.get('sha256'))} · {_cell(item.get('dhash'))}）"
                for item in group.get("members") or []
            )
            lines.append(f"| {number} | {label} | {members} |")
        if stats.get("groups_truncated"):
            lines += ["", f"> 分组较多，本表只列前 {len(groups)} 组；完整数据见 `summary.json`。"]
        lines.append("")

    detail = stats.get("detail") or []
    if detail:
        lines += [
            "### 逐图指纹",
            "",
            "| 请求 | 模型 | 文件 | 尺寸 | 字节 | SHA-256（前十六位） | dHash | 判定 |",
            "|---:|---|---|---|---:|---|---|---|",
        ]
        for item in detail:
            size = (
                f"{item.get('width')}×{item.get('height')}"
                if item.get("width") and item.get("height")
                else "—"
            )
            lines.append(
                f"| {_cell(item.get('request'))} | {_cell(item.get('model'))} | {_cell(item.get('file'))} "
                f"| {size} | {_cell(item.get('bytes'))} | {_cell(item.get('sha256'))} "
                f"| {_cell(item.get('dhash'))} | {_cell(item.get('verdict'))} |"
            )
        if stats.get("detail_truncated"):
            lines += [
                "",
                f"> 仅列前 {stats.get('detail_limit')} 张；其余逐图指纹见 `requests.jsonl` 的 `images[]`。",
            ]
        lines.append("")

    lines += [
        "> 指纹口径：SHA-256 是落盘字节的完整摘要，相同即字节完全一致；dHash 是 9×8 灰度差分哈希"
        "（64 位，记明暗结构）；chash 是 4×4 网格平均色（96 位十六进制，记配色）。"
        "同图重编码或轻改后，明暗差通常 0～2 位、配色差 0～4。",
        f"> 判重分级：明暗差 ≤ {SAME_IMAGE_DISTANCE} 位且配色差 ≤ {SAME_COLOR_DISTANCE} → 基本可判为同一张图；"
        f"明暗差 ≤ {stats.get('near_duplicate_distance')} 位且配色差 ≤ {stats.get('near_color_distance')} "
        "→ 只算结构近似（同提示词的不同采样、同风格的不同图也会落进来），需人工复核后才能当作复用证据。",
        "> 指纹只作证据、不参与准入判定：同一提示词下的正常生成也可能高度相似，重复只提示"
        "上游可能复用了同一张图片，需结合提示词与响应 id 复核。",
        "> 完整逐图指纹（含完整 SHA-256 / dHash / chash）见 `requests.jsonl` 的 `images[]`。",
    ]
    return lines
