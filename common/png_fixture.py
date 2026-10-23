"""确定性图片夹具（纯标准库，无 Pillow / 无外网依赖）。

从 dsv4/core.py 抽出的公共实现，供 DeepSeek / GPT-5.6 等套件的图像识别用例共用：
固定"上红下蓝"纯色图案，保证识别用例稳定、可复现（2026-09-24 立项时确立）。

2026-10-20 起追加「内容级」夹具（``png_solid`` / ``png_halves`` / ``png_chessboard`` /
``png_text``），服务 ``forensics`` 套件的图像内容理解用例。这些图案同样完全确定
（点阵字模内嵌、无随机、无外网），只有用例自身会换一个随机数字串来防记忆。
"""
from __future__ import annotations

import base64
import struct
import zlib
from typing import Any, Callable

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# 图片识别用例使用固定纯色，避免依赖外部 URL、联网下载与随机图案。
IMAGE_TOP_RGB = (220, 20, 20)      # 红
IMAGE_BOTTOM_RGB = (20, 20, 220)   # 蓝

# 内容级夹具的调色板（与「上下红蓝」同一套，报告里用文字复述时口径一致）。
RED = (210, 30, 30)
GREEN = (0, 155, 70)
BLUE = (25, 45, 200)
YELLOW = (250, 210, 20)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# 5x7 点阵字模：只收录图像内容理解用例实际要画的字符（0-9 与 GREEN 四个字母）。
# 每行 5 个字符，'1' 表示前景像素。字形固定 ⇒ 渲染结果逐字节确定。
_GLYPHS: dict[str, tuple[str, ...]] = {
    "0": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
}
GLYPH_WIDTH = 5
GLYPH_HEIGHT = 7


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)


def png_bands(top_rgb: tuple[int, int, int] = IMAGE_TOP_RGB, bottom_rgb: tuple[int, int, int] = IMAGE_BOTTOM_RGB,
              width: int = 64, height: int = 64) -> bytes:
    """生成上/下两色的 PNG（纯标准库实现，无 Pillow 依赖）。"""
    raw = b""
    for y in range(height):
        rgb = top_rgb if y < height // 2 else bottom_rgb
        raw += b"\x00" + bytes(rgb) * width
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return PNG_SIGNATURE + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(raw, 9)) + _png_chunk(b"IEND", b"")


def png_canvas(width: int, height: int, pixel: Callable[[int, int], tuple[int, int, int]]) -> bytes:
    """按 ``pixel(x, y) -> (r, g, b)`` 逐像素生成 PNG（8 位真彩、无隔行）。"""
    if width <= 0 or height <= 0:
        raise ValueError("图片尺寸必须大于 0")
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(pixel(x, y))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return PNG_SIGNATURE + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + _png_chunk(b"IEND", b"")


def png_solid(rgb: tuple[int, int, int], width: int = 96, height: int = 96) -> bytes:
    """整幅纯色。用于「这张图是什么颜色」这一级。"""
    return png_canvas(width, height, lambda x, y: rgb)


def png_halves(left_rgb: tuple[int, int, int], right_rgb: tuple[int, int, int],
               width: int = 128, height: int = 96) -> bytes:
    """左右各一色的竖分图。用于「右半部分是什么颜色」（空间关系级）。"""
    return png_canvas(width, height, lambda x, y: left_rgb if x < width // 2 else right_rgb)


def png_chessboard(cells: int = 8, cell: int = 40,
                   dark: tuple[int, int, int] = BLACK,
                   light: tuple[int, int, int] = WHITE) -> bytes:
    """cells × cells 棋盘，左上角为深色。深色格数恒为 cells²/2（cells 为偶数时）。"""
    size = cells * cell

    def pixel(x: int, y: int) -> tuple[int, int, int]:
        return dark if ((x // cell) + (y // cell)) % 2 == 0 else light

    return png_canvas(size, size, pixel)


def text_size(text: str, scale: int = 6, gap: int = 8, margin: int = 12) -> tuple[int, int]:
    """``png_text`` 渲染出的宽高（字数 × (5 × scale + gap) + 页边）。"""
    characters = [character for character in text if character in _GLYPHS]
    if not characters:
        raise ValueError("文本不含可渲染字符（只支持 0-9 与 G/R/E/N）")
    width = margin * 2 + len(characters) * GLYPH_WIDTH * scale + (len(characters) - 1) * gap
    height = margin * 2 + GLYPH_HEIGHT * scale
    return width, height


def png_text(text: str, fg: tuple[int, int, int] = WHITE,
             bg: tuple[int, int, int] = BLACK, scale: int = 6,
             gap: int = 8, margin: int = 12) -> bytes:
    """把 ``text`` 用内嵌 5x7 点阵字模画在纯色底上（只支持 0-9 与 G/R/E/N）。

    用例靠这一张图同时测两件事：读字（OCR 级）与「字面文字与底色冲突时以视觉为准」
    （文本先验压制级）。
    """
    characters = [character for character in text if character in _GLYPHS]
    if not characters:
        raise ValueError("文本不含可渲染字符（只支持 0-9 与 G/R/E/N）")
    width, height = text_size(text, scale=scale, gap=gap, margin=margin)
    stride = GLYPH_WIDTH * scale + gap

    def pixel(x: int, y: int) -> tuple[int, int, int]:
        local_x, local_y = x - margin, y - margin
        if local_x < 0 or local_y < 0 or local_y >= GLYPH_HEIGHT * scale:
            return bg
        index = local_x // stride
        if index >= len(characters):
            return bg
        offset = local_x - index * stride
        if offset >= GLYPH_WIDTH * scale:
            return bg
        row = local_y // scale
        column = offset // scale
        return fg if _GLYPHS[characters[index]][row][column] == "1" else bg

    return png_canvas(width, height, pixel)


def data_url(png: bytes) -> str:
    """把任意 PNG 字节包装成 data URL（OpenAI / Anthropic 的 base64 图像输入都用它）。"""
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def png_data_url(**kwargs: Any) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bands(**kwargs)).decode("ascii")


def image_part(url: str, detail: str | None = None) -> dict[str, Any]:
    """Chat Completions 风格的图片内容块。"""
    block: dict[str, Any] = {"url": url}
    if detail:
        block["detail"] = detail
    return {"type": "image_url", "image_url": block}
