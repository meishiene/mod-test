import json, tempfile, unittest
from pathlib import Path
from common.artifacts import SCHEMA_VERSION, result_envelope
from common.test_framework import ExecutionRecord, validate_case
from common.validators import status, status_error

class CommonFrameworkTests(unittest.TestCase):
    def test_expected_4xx_is_not_transport_failure(self):
        r=ExecutionRecord('x','rid',{}, {}, 'now', 0, http_status=400,
                          response_body={'error': {'type':'invalid_request_error','message':'bad temperature'}})
        v=validate_case(r,[status(400), status_error(400,error_types=('invalid_request_error',),message_contains=('temperature',))])
        self.assertTrue(v.passed); self.assertEqual(v.status,'passed')
    def test_missing_evidence_fails(self):
        r=ExecutionRecord('x','rid',{}, {}, 'now', 0, http_status=200)
        v=validate_case(r,[status(200),status_error(200)])
        self.assertFalse(v.passed)
    def test_schema(self):
        x=result_envelope(record_id='r',suite='s',model='m',protocol='openai',records=[])
        self.assertEqual(x['schema_version'],SCHEMA_VERSION)
        self.assertEqual(json.loads(json.dumps(x))['suite'],'s')

class ValidatorContractTests(unittest.TestCase):
    def test_expected_error_requires_parseable_evidence(self):
        from common.validators import status_error
        r=ExecutionRecord('x','rid',{}, {}, 'now', 0, http_status=400, response_text='')
        self.assertFalse(validate_case(r,[status_error(400,error_types=('invalid',))]).passed)

    def test_percentile_reports_insufficient_samples(self):
        from common.validators import aggregate_percentile
        r=aggregate_percentile('ttft_p95',[10],p=.95,maximum=100,minimum_samples=2)
        self.assertFalse(r.passed)


class RawChannelMultipartTests(unittest.IsolatedAsyncioTestCase):
    """execute_raw 的 multipart 头必须带 boundary（2026-09-24 Files 上传 400 事故）。"""

    async def _executor_with_capture(self, headers):
        import httpx
        from common.test_framework import HttpExecutor

        seen = {}

        async def handler(request):
            seen["content_type"] = request.headers.get("content-type")
            seen["body"] = request.content
            seen["url"] = str(request.url)
            return httpx.Response(200, json={"id": "file-api-mock1", "deleted": True})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), headers=headers)
        executor = HttpExecutor(client=client)
        await executor.__aenter__()
        return executor, client, seen

    async def test_multipart_gets_boundary_despite_json_client_default(self):
        # 复现事故条件：客户端默认头带 application/json（dsv4 runner 的写法）
        executor, client, seen = await self._executor_with_capture(
            {"Authorization": "Bearer k", "Content-Type": "application/json"})
        try:
            rec = await executor.execute_raw(case_id="DS-V4-41", method="POST", url="https://api.deepseek.com/files",
                                             headers={"X-Request-ID": "rid"},
                                             files={"file": ("a.png", b"\x89PNG", "image/png")},
                                             data={"purpose": "user_data"})
        finally:
            await client.aclose()
        ct = seen["content_type"]
        self.assertTrue(ct.startswith("multipart/form-data; boundary="), ct)
        boundary = ct.split("boundary=", 1)[1].encode()
        self.assertIn(boundary, seen["body"], "body 必须使用与 Content-Type 一致的 boundary")
        self.assertEqual(rec.http_status, 200)
        self.assertEqual(rec.request["_method"], "POST")
        self.assertEqual(rec.request["_url"], "https://api.deepseek.com/files")

    async def test_json_post_still_application_json(self):
        executor, client, seen = await self._executor_with_capture({"Content-Type": "application/json"})
        try:
            await executor.execute_raw(case_id="x", method="POST", url="https://api.deepseek.com/v1/chat/completions",
                                       headers={}, json_body={"model": "deepseek-flash"})
        finally:
            await client.aclose()
        self.assertIn("application/json", seen["content_type"])
