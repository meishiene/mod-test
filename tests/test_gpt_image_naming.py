from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from common.image_store import write_named_image
from gpt_image.naming import image_filename_stem


def test_requested_model_quality_minute_format():
    at = datetime(2026, 10, 22, 18, 24)
    assert image_filename_stem('gpt-image-2', 'medium', at) == 'image2-medium-202610221824'
    assert image_filename_stem('gpt-image-2.5', 'high', at) == 'image2.5-high-202610221824'
    name = image_filename_stem('../../other/model', '../../medium', at)
    assert '/' not in name and not name.startswith('.')


def test_same_minute_parallel_images_are_unique_and_never_overwritten(tmp_path):
    stem = 'image2-medium-202610221824'
    original = tmp_path / f'{stem}.png'
    original.write_bytes(b'original')
    def save(i):
        return write_named_image(tmp_path, stem, 'png', str(i).encode())
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(save, range(20)))
    assert len(set(paths)) == 20
    assert original.read_bytes() == b'original'
    assert {p.read_bytes() for p in paths} == {str(i).encode() for i in range(20)}
    assert (tmp_path / f'{stem}-02.png').exists()


def test_real_file_extension_is_preserved(tmp_path):
    path = write_named_image(tmp_path, 'image2-medium-202610221824', 'webp', b'raw-webp')
    assert path.name == 'image2-medium-202610221824.webp'
