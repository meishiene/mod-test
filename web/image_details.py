"""报告和图库共用的逐图 C2PA 展示字段。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from common import image_c2pa as c2pa


@lru_cache(maxsize=512)
def _cached_profile(path: str, size: int, mtime_ns: int, ctime_ns: int) -> dict | None:
    # 缓存只保存小型签名画像；图片内容不驻留内存。文件变化后自动重新解析。
    raw = Path(path).read_bytes()
    try:
        return c2pa.read_c2pa_profile(raw)
    except Exception:
        # 畸形清单不应中断其余图片的报告与图库。
        return None


def image_fingerprint_fields(path: Path) -> list[dict[str, str]]:
    """仅接收调用方已经检查过目录边界的图片路径；缺少签名不等于没有水印。"""
    profile = None
    state = '未检出 C2PA 签名或解析失败'
    if not c2pa.available():
        state = '签名解析不可用'
    else:
        try:
            stat = path.stat()
            profile = _cached_profile(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        except OSError:
            state = '图片文件无法读取'
    subject = issuer = generator = check = '—'
    if profile:
        subject = profile.get('subject') or '—'
        issuer = profile.get('issuer') or '—'
        state = '内容签名（C2PA）'
        if profile.get('verified') and subject == 'C=US, O=OpenAI OpCo, LLC, CN=OpenAI Media Service':
            state = '官方模型签名（OpenAI）'
        generator = ' · '.join(str(value) for value in (profile.get('generator'), profile.get('software_agent')) if value) or '—'
        check = ' · '.join((profile.get('signature_alg') or '未知算法',
                            '校验通过' if profile.get('verified') else '未通过校验',
                            '有不可见水印' if profile.get('watermarked') else '未检出水印声明'))
    return [{'label': label, 'value': value} for label, value in (
        ('图片指纹', state), ('签名主体', subject), ('签发者', issuer),
        ('指纹生成器', generator), ('签名校验', check),
    )]
