"""ManifestCase 请求体契约。

核心承诺：清单里只描述场景、不足以构成请求的用例，必须判 not_evaluable，
**绝不**发出一个畸形请求——那样会拿到上游 400，把"用例没写完"误报成
"被测模型不达标"（假阴性），或反过来只在 HTTP 200 上判通过（假阳性）。
"""
import asyncio, json, unittest
from pathlib import Path
from common.suite import ManifestCase, execute_manifest_case

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / 'kvv/kimi_k3_temp_cases.json').read_text(encoding='utf-8'))
BY_ID = {str(item['id']): item for item in MANIFEST['cases']}


class RecordingExecutor:
    """记录是否真的发过请求；不可执行用例必须一次都不调用它。"""

    def __init__(self):
        self.calls = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        raise AssertionError('不可执行的用例不应发起请求')


def run_case(case: ManifestCase, executor: RecordingExecutor):
    return asyncio.run(execute_manifest_case(executor, None, case, model='probe-model',
                                             url='http://invalid.invalid/v1/chat/completions',
                                             base_headers={}))


class ManifestPayloadTests(unittest.TestCase):
    def test_never_builds_a_request_without_messages(self):
        offenders = [raw['id'] for raw in MANIFEST['cases']
                     if (payload := ManifestCase(raw).build_payload(model='m')) is not None
                     and 'messages' not in payload]
        self.assertEqual(offenders, [], f'这些用例会发出没有 messages 的请求：{offenders}')

    def test_scenario_only_cases_are_not_evaluable_and_send_nothing(self):
        for cid in ('K3-CACHE-HIT-01', 'K3-CACHE-TTL-01', 'K3-AUTH-001', 'K3-TOOL-002'):
            with self.subTest(case=cid):
                case = ManifestCase(BY_ID[cid])
                self.assertIsNone(case.build_payload(model='m'), f'{cid} 不应构造出请求体')
                executor = RecordingExecutor()
                result = run_case(case, executor)
                self.assertEqual(result['status'], 'not_evaluable')
                self.assertFalse(result['passed'])
                self.assertEqual(executor.calls, [], f'{cid} 竟然发起了请求')
                self.assertIn('messages', ' '.join(result['failure_reasons']))

    def test_cache_scenario_fields_are_recognised_as_non_request(self):
        case = ManifestCase(BY_ID['K3-CACHE-HIT-01'])
        keys = case.blocking_keys()
        for key in ('warm_input_tokens_target', 'phase', 'same_prefix_byte_exact', 'append_tail'):
            self.assertIn(key, keys)

    def test_load_cases_still_get_a_synthesized_prefix(self):
        for cid in ('K3-TTFT-01', 'K3-TPOT-01'):
            with self.subTest(case=cid):
                payload = ManifestCase(BY_ID[cid]).build_payload(model='m')
                self.assertIsNotNone(payload)
                self.assertGreaterEqual(len(payload['messages'][0]['content']), 60000)

    def test_messages_bearing_case_is_untouched(self):
        payload = ManifestCase(BY_ID['K3-TOOL-001']).build_payload(model='probe-model')
        self.assertEqual(payload['model'], 'probe-model')
        self.assertIn('tools', payload)
        self.assertIn('messages', payload)


if __name__ == '__main__':
    unittest.main()
