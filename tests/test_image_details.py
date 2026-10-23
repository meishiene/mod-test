"""逐图指纹：读取隔离、缓存与 GPT/Gemini 报告及图库一致性。"""
import json

import pytest

from web import image_details, server


@pytest.fixture
def signed_profile():
    return {
        'subject': 'C=US, O=OpenAI OpCo, LLC, CN=OpenAI Media Service',
        'issuer': 'C=US, O=Trufo Inc., CN=Trufo C2PA Claim Signing CA (2025)',
        'generator': 'OpenAI Media Service API', 'software_agent': 'ChatGPT · gpt-image',
        'signature_alg': 'ES256', 'verified': True, 'watermarked': True,
    }


@pytest.fixture(autouse=True)
def clear_cache():
    image_details._cached_profile.cache_clear()
    yield
    image_details._cached_profile.cache_clear()


def test_per_image_fields_and_cache_invalidation(tmp_path, monkeypatch, signed_profile):
    path = tmp_path / 'image.png'
    path.write_bytes(b'signed')
    calls = []
    monkeypatch.setattr(image_details.c2pa, 'available', lambda: True)
    def read(raw):
        calls.append(raw)
        return signed_profile if raw == b'signed' else None
    monkeypatch.setattr(image_details.c2pa, 'read_c2pa_profile', read)
    fields = {x['label']: x['value'] for x in image_details.image_fingerprint_fields(path)}
    assert fields['图片指纹'] == '官方模型签名（OpenAI）'
    assert fields['签名主体'] == signed_profile['subject']
    assert fields['签发者'] == signed_profile['issuer']
    assert fields['指纹生成器'] == 'OpenAI Media Service API · ChatGPT · gpt-image'
    assert fields['签名校验'] == 'ES256 · 校验通过 · 有不可见水印'
    image_details.image_fingerprint_fields(path)
    assert calls == [b'signed']
    path.write_bytes(b'unsigned-image')
    fields = image_details.image_fingerprint_fields(path)
    assert fields[0]['value'] == '未检出 C2PA 签名或解析失败'
    assert calls == [b'signed', b'unsigned-image']
    assert not any('无水印' in x['value'] for x in fields)


def test_unknown_signature_is_not_openai_and_missing_watermark_is_unknown(tmp_path, monkeypatch, signed_profile):
    path = tmp_path / 'other.png'
    path.write_bytes(b'other')
    profile = dict(signed_profile, subject='CN=Other Service', verified=False, watermarked=False)
    monkeypatch.setattr(image_details.c2pa, 'available', lambda: True)
    monkeypatch.setattr(image_details.c2pa, 'read_c2pa_profile', lambda raw: profile)
    fields = {x['label']: x['value'] for x in image_details.image_fingerprint_fields(path)}
    assert fields['图片指纹'] == '内容签名（C2PA）'
    assert fields['签名校验'] == 'ES256 · 未通过校验 · 未检出水印声明'


def test_unavailable_and_unreadable_images(tmp_path, monkeypatch):
    monkeypatch.setattr(image_details.c2pa, 'available', lambda: False)
    assert image_details.image_fingerprint_fields(tmp_path / 'missing.png')[0]['value'] == '签名解析不可用'
    monkeypatch.setattr(image_details.c2pa, 'available', lambda: True)
    assert image_details.image_fingerprint_fields(tmp_path / 'missing.png')[0]['value'] == '图片文件无法读取'


def test_malformed_signature_does_not_break_image_details(tmp_path, monkeypatch):
    path = tmp_path / 'broken.png'
    path.write_bytes(b'malformed-manifest')
    monkeypatch.setattr(image_details.c2pa, 'available', lambda: True)
    def malformed(raw):
        raise ValueError('malformed manifest')
    monkeypatch.setattr(image_details.c2pa, 'read_c2pa_profile', malformed)
    assert image_details.image_fingerprint_fields(path)[0]['value'] == '未检出 C2PA 签名或解析失败'


@pytest.mark.parametrize('test_kind,case_name,protocol,model', [
    ('gpt-image', 'GPT_IMAGE_2', 'openai_images', 'gpt-image-2'),
    ('gemini-image', 'GEMINI_IMAGE_GEN', 'gemini_images', 'gemini-3-pro-image'),
])
def test_manifest_and_gallery_include_every_image_with_matching_fields(
    tmp_path, monkeypatch, signed_profile, test_kind, case_name, protocol, model,
):
    results = tmp_path / 'results'
    out = results / test_kind
    (out / 'images').mkdir(parents=True)
    (out / 'summary.json').write_text(json.dumps({'test_kind': test_kind, 'images_saved': 10}))
    for index in range(10):
        (out / 'images' / f'image-{index:02d}.png').write_bytes(b'signed' if index % 2 == 0 else b'unsigned')
    monkeypatch.setattr(server, 'ROOT', tmp_path)
    monkeypatch.setattr(server, 'RESULTS', results)
    monkeypatch.setattr(image_details.c2pa, 'available', lambda: True)
    profile = dict(signed_profile, issuer='<script>unsafe & issuer</script>')
    signature_label = '官方模型签名（OpenAI）'
    if test_kind == 'gemini-image':
        # 合成 Google 签名用于验证逐图映射，不以任务模型名推断签名者或水印。
        profile.update(subject='C=US, O=Google LLC, CN=Test Image Signer',
                       generator='Test Google Generator', software_agent='Test Gemini Agent', watermarked=False)
        signature_label = '内容签名（C2PA）'
    monkeypatch.setattr(image_details.c2pa, 'read_c2pa_profile', lambda raw: profile if raw == b'signed' else None)
    row = dict(id='test-run', case_name=case_name, status='completed', passed=1,
               result_dir=f'results/{test_kind}', report_path=None, summary_path=f'results/{test_kind}/summary.json',
               error=None, test_kind=test_kind, protocol=protocol, base_url='https://example.com',
               model=model, created_at='2026-10-22')
    single = server.single_run_manifest(row)
    entries = single['results'][0]['image_fingerprints']
    assert len(entries) == 10
    assert entries[0]['name'] == 'image-00.png'
    assert entries[0]['url'] == '/api/runs/test-run/images/image-00.png'
    assert entries[1]['fields'][0]['value'] == '未检出 C2PA 签名或解析失败'
    batch = {'record_type': 'batch', 'test_kind': test_kind, 'results': [
        dict(record_id='test-run', case=case_name, result_dir=f'results/{test_kind}', summary={}),
    ]}
    server.ensure_image_fingerprints(batch)
    assert batch['results'][0]['image_fingerprints'] == entries
    gallery = server.render_image_gallery(row)
    assert gallery.count('class="image-fingerprint"') == 10
    assert gallery.count(signature_label) == 5
    assert gallery.count('未检出 C2PA 签名或解析失败') == 5
    assert '<script>unsafe' not in gallery
    assert '&lt;script&gt;unsafe &amp; issuer&lt;/script&gt;' in gallery
    assert 'image-09.png' in gallery
    if test_kind == 'gemini-image':
        assert 'OpenAI' not in gallery
        assert gallery.count('C=US, O=Google LLC, CN=Test Image Signer') == 5
        assert gallery.count('未检出水印声明') == 5
        assert '有不可见水印' not in gallery


def test_non_image_manifests_and_path_guards(tmp_path, monkeypatch):
    results = tmp_path / 'results'
    results.mkdir()
    monkeypatch.setattr(server, 'ROOT', tmp_path)
    monkeypatch.setattr(server, 'RESULTS', results)
    payload = {'record_type': 'batch', 'test_kind': 'fable', 'results': [{'summary': {}}]}
    server.ensure_image_fingerprints(payload)
    assert 'image_fingerprints' not in payload['results'][0]
    outside = tmp_path / 'outside'
    (outside / 'images').mkdir(parents=True)
    (outside / 'images' / 'private.png').write_bytes(b'private')
    assert server.generated_image_details('test', str(outside)) == []
