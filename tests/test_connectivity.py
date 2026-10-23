import os
import unittest
from unittest.mock import patch
from volcperf.config import ApiConfig, trust_env_enabled
from volcperf.connectivity import connectivity_endpoint, connectivity_headers, connectivity_payload, check_connectivity

class Response:
    def __init__(self,status=200,data=None,text='',headers=None):
        self.status_code=status;self._data=data if data is not None else {"ok":True};self.text=text;self.headers=headers or {}
    def json(self): return self._data

class ConnectivityTests(unittest.TestCase):
    def test_openai_low_effort(self):
        cfg=ApiConfig('k','https://api.example/v1','m',protocol='volcengine')
        self.assertEqual(connectivity_endpoint(cfg),'https://api.example/v1/chat/completions')
        body=connectivity_payload(cfg);self.assertEqual(body['max_tokens'],2048);self.assertEqual(body['reasoning_effort'],'low')

    def test_gemini_nonstream_endpoint(self):
        cfg=ApiConfig('k','https://google/v1beta','gemini/x',protocol='gemini')
        self.assertEqual(connectivity_endpoint(cfg),'https://google/v1beta/models/gemini%2Fx:generateContent')

    def test_openai_responses_endpoint_and_payload(self):
        cfg=ApiConfig('k','https://mux.la','gpt-x',protocol='openai_responses')
        self.assertEqual(connectivity_endpoint(cfg),'https://mux.la/v1/responses')
        body=connectivity_payload(cfg)
        self.assertEqual(body['input'],'Reply with OK only.')
        self.assertEqual(body['max_output_tokens'],2048)
        self.assertNotIn('reasoning',body)
        self.assertNotIn('stream',body)

    def test_anthropic_k3_compatible_probe(self):
        cfg=ApiConfig('k','https://mux.la','kimi-k3',protocol='anthropic')
        body=connectivity_payload(cfg);request_headers=connectivity_headers(cfg)
        self.assertEqual(body['max_tokens'],2048)
        self.assertNotIn('thinking',body)
        self.assertNotIn('stream',body)
        self.assertEqual(request_headers['Accept'],'application/json')
        self.assertEqual(request_headers['x-api-key'],'k')
        self.assertEqual(request_headers['Authorization'],'Bearer k')

    def test_openai_images_endpoint_and_payload(self):
        cfg=ApiConfig('k','https://mux.la','gpt-image-2',protocol='openai_images')
        self.assertEqual(connectivity_endpoint(cfg),'https://mux.la/v1/images/generations')
        self.assertEqual(connectivity_payload(cfg)['quality'],'low')

    def test_success_does_not_require_visible_content(self):
        cfg=ApiConfig('k','https://api.example/v1','m',protocol='openai')
        fake=type('Client',(),{'__enter__':lambda s:s,'__exit__':lambda *a:None,'post':lambda *a,**k:Response(data={"choices":[{"message":{"content":""}}]})})
        with patch('volcperf.connectivity.httpx.Client',return_value=fake()): result=check_connectivity(cfg)
        self.assertTrue(result['passed'])

    def test_http_failure(self):
        cfg=ApiConfig('k','https://api.example/v1','m',protocol='openai')
        fake=type('Client',(),{'__enter__':lambda s:s,'__exit__':lambda *a:None,'post':lambda *a,**k:Response(401,text='bad key')})
        with patch('volcperf.connectivity.httpx.Client',return_value=fake()): result=check_connectivity(cfg)
        self.assertFalse(result['passed']);self.assertIn('401',result['error'])

    def test_html_upstream_failure_has_protocol_diagnosis(self):
        cfg=ApiConfig('k','https://mux.la','kimi-k3',protocol='openai_responses')
        error='{"error":{"message":"invalid character \'<\' looking for beginning of value"}}'
        fake=type('Client',(),{'__enter__':lambda s:s,'__exit__':lambda *a:None,'post':lambda *a,**k:Response(500,text=error,headers={'x-request-id':'req-1'})})
        with patch('volcperf.connectivity.httpx.Client',return_value=fake()): result=check_connectivity(cfg)
        self.assertFalse(result['passed'])
        self.assertIn('OpenAI Responses',result['diagnosis'])
        self.assertIn('不是本地 Base URL 拼装错误',result['diagnosis'])
        self.assertEqual(result['provider_request_id'],'req-1')
        self.assertEqual(result['request']['endpoint'],'https://mux.la/v1/responses')

    def _raising_client(self,exc):
        class Client:
            def __enter__(self): return self
            def __exit__(self,*a): return None
            def post(self,*a,**k): raise exc
        return Client()

    def test_read_timeout_is_not_reported_as_unreachable(self):
        """ReadTimeout 发生在连接建立之后，不能报成「未建立连接」。"""
        import httpx
        cfg=ApiConfig('k','https://qianfan.baidubce.com','ernie-5.1',protocol='qianfan',trust_env=True)
        with patch('volcperf.connectivity.httpx.Client',return_value=self._raising_client(httpx.ReadTimeout('timed out'))):
            result=check_connectivity(cfg)
        self.assertFalse(result['passed'])
        self.assertEqual(result['error'],'ReadTimeout: timed out')
        self.assertIn('连接已建立',result['diagnosis'])
        self.assertIn('VOLC_CONNECTIVITY_TIMEOUT_SECONDS',result['diagnosis'])
        self.assertNotIn('仍未建立连接',result['diagnosis'])

    def test_connect_timeout_keeps_proxy_hint(self):
        import httpx
        cfg=ApiConfig('k','https://qianfan.baidubce.com','ernie-5.1',protocol='qianfan',trust_env=True)
        with patch('volcperf.connectivity.httpx.Client',return_value=self._raising_client(httpx.ConnectTimeout('timed out'))):
            result=check_connectivity(cfg)
        self.assertIn('仍未建立连接',result['diagnosis'])

    def test_pool_timeout_blames_local_concurrency(self):
        import httpx
        cfg=ApiConfig('k','https://api.example/v1','m',protocol='openai')
        with patch('volcperf.connectivity.httpx.Client',return_value=self._raising_client(httpx.PoolTimeout('pool'))):
            result=check_connectivity(cfg)
        self.assertIn('连接池',result['diagnosis'])
        self.assertNotIn('代理',result['diagnosis'])

    def test_probe_timeout_is_configurable_within_caller_budget(self):
        from volcperf.connectivity import _connectivity_timeout
        cfg=ApiConfig('k','https://api.example/v1','m',protocol='openai')
        bare={k:v for k,v in os.environ.items() if k!='VOLC_CONNECTIVITY_TIMEOUT_SECONDS'}
        with patch.dict(os.environ,bare,clear=True):
            self.assertEqual(_connectivity_timeout(cfg),60.0)
        with patch.dict(os.environ,{'VOLC_CONNECTIVITY_TIMEOUT_SECONDS':'15'}):
            self.assertEqual(_connectivity_timeout(cfg),15.0)
        with patch.dict(os.environ,{'VOLC_CONNECTIVITY_TIMEOUT_SECONDS':'120'}):
            self.assertEqual(_connectivity_timeout(cfg),120.0)
        tight=ApiConfig('k','https://api.example/v1','m',protocol='openai',request_timeout_seconds=10)
        with patch.dict(os.environ,{'VOLC_CONNECTIVITY_TIMEOUT_SECONDS':'120'}):
            self.assertEqual(_connectivity_timeout(tight),10.0)

class TrustEnvFlagTests(unittest.TestCase):
    """VOLC_TRUST_ENV 是 Web 探针与正式压测共用的开关，默认必须是直连。"""

    def test_absent_flag_means_direct(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(trust_env_enabled())

    def test_truthy_values_enable_proxy(self):
        for value in ('true','TRUE','1','yes','on'):
            with patch.dict(os.environ, {'VOLC_TRUST_ENV':value}):
                self.assertTrue(trust_env_enabled(), value)

    def test_falsey_values_keep_direct(self):
        for value in ('false','FALSE','0','no','off',''):
            with patch.dict(os.environ, {'VOLC_TRUST_ENV':value}):
                self.assertFalse(trust_env_enabled(), value)

    def test_from_env_propagates_flag(self):
        env={'VOLC_API_KEY':'k','VOLC_BASE_URL':'https://mux.la','VOLC_MODEL':'kimi-k3','VOLC_TRUST_ENV':'true'}
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(ApiConfig.from_env().trust_env)


if __name__=='__main__':unittest.main()
