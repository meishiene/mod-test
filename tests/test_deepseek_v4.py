import inspect
import json
import unittest

import httpx

from common.model_contracts import model_matches_family
from dsv4.cases import CASES
from dsv4.core import (PRIMARY_MODELS, RETIRED_ALIASES, CaseResult, DeepSeekConfig, DeepSeekV4Runner,
                       _cache_tokens, png_bands, png_data_url, summarize_results)


class DeepSeekV4Tests(unittest.TestCase):
    def test_ui_suites(self):
        self.assertEqual(set(CASES), {"DS_V4_P0", "DS_V4_P1", "DS_V4_P2", "DS_V4_FINGERPRINT"})
        # 2026-09-24 覆盖度评审后：P0 由 18 → 24，P1 由 7 → 18，P2 由 10 → 19
        self.assertEqual(CASES["DS_V4_P0"].request_count,24)
        self.assertEqual(CASES["DS_V4_P1"].request_count,18)
        self.assertEqual(CASES["DS_V4_P2"].request_count,19)

    def test_config_urls(self):
        config=DeepSeekConfig("key", "https://api.deepseek.com/")
        self.assertEqual(config.url("chat/completions"), "https://api.deepseek.com/chat/completions")
        self.assertEqual(config.url("chat/completions", "beta"), "https://api.deepseek.com/beta/chat/completions")
        self.assertEqual(config.url("v1/messages", "anthropic"), "https://api.deepseek.com/anthropic/v1/messages")

    def test_config_urls_strip_api_version_for_beta_and_anthropic(self):
        # 运行时 base_url 已被 api_root() 规范化成 ".../v1"（web/server.py + dsv4/runner.py），
        # 而 beta/anthropic 端点挂在根域下（官方文档：api.deepseek.com/beta、api.deepseek.com/anthropic），
        # 不能再带版本后缀，否则会打到不存在的路径（线上 404）。
        config=DeepSeekConfig("key", "https://api.deepseek.com/v1")
        self.assertEqual(config.url("chat/completions"), "https://api.deepseek.com/v1/chat/completions")
        self.assertEqual(config.url("chat/completions", "beta"), "https://api.deepseek.com/beta/chat/completions")
        self.assertEqual(config.url("completions", "beta"), "https://api.deepseek.com/beta/completions")
        self.assertEqual(config.url("v1/messages", "anthropic"), "https://api.deepseek.com/anthropic/v1/messages")
        self.assertEqual(config.url("responses"), "https://api.deepseek.com/v1/responses")

    def test_explicit_beta_and_anthropic_override_wins(self):
        config=DeepSeekConfig("key", "https://api.deepseek.com/v1",
                              beta_base_url="https://gw.example.com/beta",
                              anthropic_base_url="https://gw.example.com/anthropic")
        self.assertEqual(config.url("chat/completions","beta"), "https://gw.example.com/beta/chat/completions")
        self.assertEqual(config.url("v1/messages","anthropic"), "https://gw.example.com/anthropic/v1/messages")

    def test_cache_usage_shapes(self):
        self.assertEqual(_cache_tokens({"usage":{"prompt_cache_hit_tokens":731}}),731)
        self.assertEqual(_cache_tokens({"usage":{"prompt_tokens_details":{"cached_tokens":123}}}),123)

    def test_summary_has_result_details(self):
        summary=summarize_results([CaseResult("1","a","m",True)])
        self.assertTrue(summary["all_passed"])
        self.assertEqual(summary["results"][0]["case_id"],"1")


class DeepSeekV4MockTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_separates_reasoning_and_content(self):
        events=[{"choices":[{"delta":{"reasoning_content":"思考"}}]},{"choices":[{"delta":{"content":"答案"}}]}]
        body="".join(f"data: {json.dumps(x,ensure_ascii=False)}\n\n" for x in events)+"data: [DONE]\n\n"
        async def handler(request): return httpx.Response(200,text=body,headers={"content-type":"text/event-stream"})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            runner=DeepSeekV4Runner(DeepSeekConfig("key"),client)
            response=await runner.chat("deepseek-v4-pro",[{"role":"user","content":"x"}],stream=True)
        self.assertEqual(response.stream["reasoning_content"],"思考")
        self.assertEqual(response.stream["content"],"答案")
        self.assertTrue(response.stream["done"])


if __name__=="__main__": unittest.main()

class DeepSeekV4OutcomeTests(unittest.IsolatedAsyncioTestCase):
    async def _runner(self, responses):
        queue=list(responses)
        requests=[]
        async def handler(request):
            requests.append(json.loads(request.content))
            status,body=queue.pop(0)
            return httpx.Response(status,json=body)
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return DeepSeekV4Runner(DeepSeekConfig("key"),client),client,requests

    async def test_tool_priming_uses_auto_because_thinking_rejects_required(self):
        first={"choices":[{"message":{"role":"assistant","content":"","reasoning_content":"need tool","tool_calls":[{"id":"call-1","type":"function","function":{"name":"get_number","arguments":"{}"}}]}}]}
        second={"choices":[{"message":{"role":"assistant","content":"数字是 731","reasoning_content":"done"}}]}
        runner,client,requests=await self._runner([(200,first),(200,second)])
        try: outcome=await runner.case_tool_replay("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        # thinking 模式不支持 tool_choice=required（上游 400），前置必须用 auto 诱导
        self.assertEqual(requests[0]["tool_choice"],"auto")
        self.assertEqual(requests[0]["thinking"],{"type":"enabled"})
        self.assertNotIn("required",[r.get("tool_choice") for r in requests])
        self.assertEqual(requests[1]["messages"][1]["reasoning_content"],"need tool")

    async def test_tool_priming_retries_with_stronger_prompt(self):
        no_call={"choices":[{"message":{"role":"assistant","content":"731","reasoning_content":"直接回答"}}]}
        with_call={"choices":[{"message":{"role":"assistant","content":"","reasoning_content":"need tool","tool_calls":[{"id":"call-2","type":"function","function":{"name":"get_number","arguments":"{}"}}]}}]}
        final={"choices":[{"message":{"role":"assistant","content":"数字是 731","reasoning_content":"done"}}]}
        runner,client,requests=await self._runner([(200,no_call),(200,with_call),(200,final)])
        try: outcome=await runner.case_tool_replay("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(len(requests),3)
        self.assertNotEqual(requests[0]["messages"][0]["content"],requests[1]["messages"][0]["content"])
        self.assertEqual(requests[2]["messages"][1]["reasoning_content"],"need tool")

    async def test_tool_precondition_failure_is_blocked_not_failed(self):
        error={"error":{"message":"Thinking mode does not support this tool_choice"}}
        runner,client,_=await self._runner([(400,error)])
        try: outcome=await runner.case_tool_replay("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3],"blocked")
        self.assertEqual(outcome[2]["http_status"],400)
        self.assertIn("HTTP 400",outcome[1])

    async def test_tool_missing_reasoning_400_is_passed(self):
        first={"choices":[{"message":{"role":"assistant","content":"","reasoning_content":"need tool","tool_calls":[{"id":"call-1","type":"function","function":{"name":"get_number","arguments":"{}"}}]}}]}
        second={"error":{"message":"reasoning_content is required when the assistant message carries tool_calls"}}
        runner,client,requests=await self._runner([(200,first),(400,second)])
        try: outcome=await runner.case_tool_missing_reasoning("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(len(outcome),3)   # 400 分支不给显式 status，按 passed 收口
        self.assertEqual(requests[0]["tool_choice"],"auto")
        self.assertNotIn("reasoning_content",requests[1]["messages"][1])

    async def test_tool_missing_reasoning_200_is_warning(self):
        # 线上实际行为：漏传 reasoning_content 不报 400，容忍并正常作答（文档承诺 400）
        first={"choices":[{"message":{"role":"assistant","content":"","reasoning_content":"need tool","tool_calls":[{"id":"call-1","type":"function","function":{"name":"get_number","arguments":"{}"}}]}}]}
        second={"choices":[{"message":{"role":"assistant","content":"数字是 731","reasoning_content":""}}]}
        runner,client,_=await self._runner([(200,first),(200,second)])
        try: outcome=await runner.case_tool_missing_reasoning("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")
        self.assertEqual(outcome[2]["http_status"],200)

    async def test_tool_missing_reasoning_unexpected_status_is_failed(self):
        first={"choices":[{"message":{"role":"assistant","content":"","reasoning_content":"need tool","tool_calls":[{"id":"call-1","type":"function","function":{"name":"get_number","arguments":"{}"}}]}}]}
        runner,client,_=await self._runner([(200,first),(500,{"error":{"message":"boom"}})])
        try: outcome=await runner.case_tool_missing_reasoning("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(len(outcome),3)   # 既非 400 也非 200 → 无显式裁定，由 run() 归为 failed

    async def test_plain_tool_turn_200_is_warning(self):
        first={"choices":[{"message":{"role":"assistant","content":"OK","reasoning_content":"answer directly"}}]}
        second={"choices":[{"message":{"role":"assistant","content":"OK","reasoning_content":"again"}}]}
        runner,client,_=await self._runner([(200,first),(200,second)])
        try: outcome=await runner.case_tools_plain_turn_replay("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")

    async def test_invalid_user_id_accepted_is_warning(self):
        response={"choices":[{"message":{"content":"OK"}}]}
        runner,client,_=await self._runner([(200,response)])
        try: outcome=await runner.case_invalid_user_id("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")

    def test_warning_and_blocked_do_not_fail_admission(self):
        summary=summarize_results([
            CaseResult("1","warning","m",True,"warning"),
            CaseResult("2","blocked","m",False,"blocked"),
            CaseResult("3","passed","m",True,"passed"),
        ])
        self.assertTrue(summary["all_passed"])
        self.assertEqual(summary["warning"],1)
        self.assertEqual(summary["blocked"],1)

    async def test_beta_negative_case_404_is_blocked_not_passed(self):
        # 端点不存在（404）不能被当成"参数被正确拒绝"——原先 400<=status<500 会把 404 误判为通过
        not_found={"error":{"message":"not found"}}
        for handler in ("case_strict_invalid","case_prefix_invalid"):
            runner,client,_=await self._runner([(404,not_found)])
            try: outcome=await getattr(runner,handler)("deepseek-v4-pro")
            finally: await client.aclose()
            self.assertFalse(outcome[0],handler)
            self.assertEqual(outcome[3],"blocked",handler)
            self.assertIn("404",outcome[1],handler)

    async def test_openai_negative_cases_404_is_blocked_not_passed(self):
        # 同类缺陷（DS-V4-22 曾因此在两端点 404 的情况下显示"通过"）：temperature 越界 /
        # 非法 messages / 非法 user_id 原先用宽松的 400<=status<500，404 会被误读为
        # "参数被正确拒绝"。端点不存在一律记 blocked，不能算拒绝。
        not_found={"error":{"message":"not found"}}
        for handler in ("case_temperature_invalid","case_messages_invalid","case_invalid_user_id"):
            runner,client,_=await self._runner([(404,not_found)])
            try: outcome=await getattr(runner,handler)("deepseek-v4-pro")
            finally: await client.aclose()
            self.assertFalse(outcome[0],handler)
            self.assertEqual(outcome[3],"blocked",handler)
            self.assertIn("404",outcome[1],handler)

    async def test_beta_negative_case_400_is_passed_and_200_is_failed(self):
        runner,client,_=await self._runner([(400,{"error":{"message":"unsupported strict schema"}})])
        try: outcome=await runner.case_strict_invalid("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(len(outcome),3)   # 正常拒绝 → 无显式裁定，按 passed 收口

        runner,client,_=await self._runner([(200,{"choices":[{"message":{"content":"ok"}}]})])
        try: outcome=await runner.case_strict_invalid("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(len(outcome),3)   # 未拒绝 → 由 run() 归为 failed

    async def test_prefix_misplacement_4xx_passed_200_warning_5xx_failed(self):
        # 官方《Chat Prefix Completion (Beta)》只要求"调用方自行确保"prefix 位于最后一条
        # assistant 消息，从未承诺对错误放置返回 4xx。线上实测：静默忽略并正常作答（200）。
        # 因此 4xx（严格校验）→ passed；200（宽松忽略）→ warning（不判模型失败、不阻断准入）。
        runner,client,_=await self._runner([(400,{"error":{"message":"prefix must be the last assistant message"}})])
        try: outcome=await runner.case_prefix_invalid("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(len(outcome),3)            # 校验型拒绝 → 无显式裁定，按 passed 收口

        ok={"choices":[{"message":{"role":"assistant","content":"写代码\n\n你好！"}}]}
        runner,client,_=await self._runner([(200,ok)])
        try: outcome=await runner.case_prefix_invalid("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")
        self.assertIn("未承诺拒绝",outcome[1])

        runner,client,_=await self._runner([(503,{"error":{"message":"server overloaded"}})])
        try: outcome=await runner.case_prefix_invalid("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(len(outcome),3)            # 5xx → 由 run() 归为 failed

    async def test_beta_positive_cases_404_are_blocked(self):
        not_found={"error":{"message":"not found"}}
        for handler in ("case_strict_tool","case_prefix","case_fim"):
            runner,client,_=await self._runner([(404,not_found)])
            try: outcome=await getattr(runner,handler)("deepseek-v4-pro")
            finally: await client.aclose()
            self.assertFalse(outcome[0],handler)
            self.assertEqual(outcome[3],"blocked",handler)

    async def test_anthropic_endpoint_404_is_blocked(self):
        runner,client,_=await self._runner([(404,{"error":{"message":"not found"}})])
        try: outcome=await runner.case_anthropic_mapping("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3],"blocked")


class DeepSeekV4SupplementTests(unittest.IsolatedAsyncioTestCase):
    """2026-09-24 覆盖度评审补充用例（DS-V4-36 ~ DS-V4-61）的行为测试。"""

    async def _runner(self, responses, base_url="https://api.deepseek.com/v1"):
        queue=list(responses); seen=[]
        async def handler(request):
            body=None
            if request.content:
                text=request.content.decode("utf-8","replace")
                try: body=json.loads(text)
                except json.JSONDecodeError: body={"_raw":text[:120]}   # multipart 上传内容非 JSON
            seen.append({"method":request.method,"path":request.url.path,"body":body,
                         "auth":request.headers.get("authorization"),"query":dict(request.url.params)})
            item=queue.pop(0)
            if not callable(item): return httpx.Response(item[0],json=item[1])
            result=item(request)
            return await result if inspect.isawaitable(result) else result
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return DeepSeekV4Runner(DeepSeekConfig("key",base_url),client),client,seen

    @staticmethod
    def _chat(content="OK",prompt_tokens=10):
        return {"choices":[{"message":{"role":"assistant","content":content}}],"usage":{"prompt_tokens":prompt_tokens,"completion_tokens":3,"total_tokens":prompt_tokens+3}}

    def test_png_generator_is_valid_and_deterministic(self):
        import struct
        raw=png_bands()
        self.assertTrue(raw.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(struct.unpack(">II",raw[16:24]),(64,64))
        self.assertEqual(raw,png_bands())          # 固定图案，避免随机图导致识别用例不稳定
        self.assertTrue(png_data_url().startswith("data:image/png;base64,"))
        self.assertNotEqual(png_bands(top_rgb=(0,0,0),bottom_rgb=(255,255,255)),raw)

    def test_root_endpoints_stay_on_domain_root(self):
        config=DeepSeekConfig("key","https://api.deepseek.com/v1")
        self.assertEqual(config.url("models","root"),"https://api.deepseek.com/models")
        self.assertEqual(config.url("user/balance","root"),"https://api.deepseek.com/user/balance")
        self.assertEqual(config.url("files","root"),"https://api.deepseek.com/files")
        self.assertEqual(config.url("files","openai"),"https://api.deepseek.com/v1/files")

    async def test_root_request_falls_back_to_versioned_path_on_404(self):
        body={"object":"list","data":[{"id":"deepseek-flash","object":"model","owned_by":"deepseek"}]}
        runner,client,seen=await self._runner([(404,{"error":{"message":"not found"}}),(200,body)])
        try: outcome=await runner.case_models_listing("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual([s["path"] for s in seen],["/models","/v1/models"])

    async def test_models_listing_without_flash_is_warning(self):
        body={"object":"list","data":[{"id":"deepseek-v4-pro","object":"model","owned_by":"deepseek"}]}
        runner,client,_=await self._runner([(200,body)])
        try: outcome=await runner.case_models_listing("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")

    async def test_image_input_requires_content_and_token_growth(self):
        runner,client,seen=await self._runner([(200,self._chat("OK",10)),(200,self._chat("上半部分是红色，下半部分是蓝色",210))])
        try: outcome=await runner.case_image_input("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        # 第二次请求必须真的带图（content 数组 + data URL）
        parts=seen[1]["body"]["messages"][0]["content"]
        self.assertIsInstance(parts,list)
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/png;base64,"))

        runner,client,_=await self._runner([(200,self._chat("OK",10)),(200,self._chat("这是一张图片",210))])
        try: outcome=await runner.case_image_input("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertIn("未正确识别",outcome[1])

    async def test_image_input_token_not_counted_is_failed(self):
        # 200 但 prompt_tokens 与纯文本基线一致 → 图片被丢弃，不能算通过
        runner,client,_=await self._runner([(200,self._chat("OK",10)),(200,self._chat("上半部分红色，下半部分蓝色",10))])
        try: outcome=await runner.case_image_input("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertIn("prompt_tokens",outcome[1])

    async def test_image_wrong_role_lenient_200_is_warning(self):
        runner,client,_=await self._runner([(200,self._chat("看到了")),(200,self._chat("看到了"))])
        try: outcome=await runner.case_image_wrong_role("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")
        runner,client,_=await self._runner([(400,{"error":{"message":"images are only allowed in user messages"}})])
        try: outcome=await runner.case_image_wrong_role("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(len(outcome),3)   # 4xx 拒绝 → 无显式裁定，按 passed 收口

    async def test_effort_full_set_official_values_and_disabled_thinking(self):
        """官方取值域 low/medium/high/xhigh/max 均被接受；关思考用官方形态 thinking=disabled。

        原实现把 minimal/ultra 当"兼容别名"要求必须被接受、并用 reasoning_effort=none 关思考，
        两处都与官方《Thinking Mode》冲突（Chat 侧既无这两个值，也无 effort=none），已按文档改：
        minimal/ultra 降级为观察项，关思考走 thinking={"type":"disabled"}。
        """
        official = ["low", "medium", "high", "xhigh", "max"]
        bodies = [{"choices": [{"message": {"content": "391", "reasoning_content": "思考"}}]} for _ in official]
        bodies.append({"choices": [{"message": {"content": "391", "reasoning_content": ""}}]})          # thinking=disabled
        bodies += [{"choices": [{"message": {"content": "OK", "reasoning_content": "思考"}}]} for _ in ("minimal", "ultra")]
        runner, client, seen = await self._runner([(200, b) for b in bodies])
        try: outcome = await runner.case_effort_full_set("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(len(outcome), 3)   # 通过路径无显式裁定 ⇒ 按 passed 收口
        self.assertEqual([s["body"].get("reasoning_effort") for s in seen[:5]], official)
        self.assertEqual(seen[5]["body"]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", seen[5]["body"])
        self.assertEqual([s["body"]["reasoning_effort"] for s in seen[6:]], ["minimal", "ultra"])

        # thinking=disabled 仍返回 reasoning_content → warning（文档差异，不判失败）
        bodies[5] = {"choices": [{"message": {"content": "391", "reasoning_content": "仍然思考了"}}]}
        runner, client, _ = await self._runner([(200, b) for b in bodies])
        try: outcome = await runner.case_effort_full_set("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    async def test_max_tokens_lenient_bounds_are_warning(self):
        runner,client,seen=await self._runner([(200,self._chat()), (200,self._chat()), (200,self._chat())])
        try: outcome=await runner.case_max_tokens_bounds("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")
        self.assertEqual([s["body"]["max_tokens"] for s in seen],[393216,0,393217])

        runner,client,_=await self._runner([(400,{"error":{"message":"max_tokens must be between 1 and 393216"}})])
        try: outcome=await runner.case_max_tokens_bounds("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertIn("393216",outcome[1])

    async def test_invalid_api_key_contract(self):
        runner,client,seen=await self._runner([(401,{"error":{"message":"Invalid API key provided","type":"invalid_request_error","code":"invalid_api_key"}})])
        try: outcome=await runner.case_invalid_api_key("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[2]["error_type"],"invalid_request_error")
        self.assertTrue(seen[0]["auth"].startswith("Bearer sk-invalid"))

        runner,client,_=await self._runner([(200,self._chat())])
        try: outcome=await runner.case_invalid_api_key("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])   # 无效 Key 返回 200 → 鉴权未生效

    async def test_files_lifecycle_uploads_references_and_deletes(self):
        runner,client,seen=await self._runner([
            (200,{"id":"file-api-abc123","object":"file","bytes":1024,"filename":"dsv4-probe.png","purpose":"user_data"}),
            (200,self._chat("上半部分红色，下半部分蓝色")),
            (200,{"id":"file-api-abc123","object":"file","deleted":True}),
        ])
        try: outcome=await runner.case_files_lifecycle("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual([(s["method"],s["path"]) for s in seen],
                         [("POST","/files"),("POST","/v1/chat/completions"),("DELETE","/files/file-api-abc123")])
        self.assertEqual(seen[1]["body"]["messages"][0]["content"][1]["file_id"],"file-api-abc123")

    async def test_files_lifecycle_warns_when_delete_fails(self):
        runner,client,seen=await self._runner([
            (200,{"id":"file-api-abc123","object":"file"}),
            (200,self._chat("上半部分是红色，下半部分是蓝色")),
            (500,{"error":{"message":"boom"}}),
        ])
        try: outcome=await runner.case_files_lifecycle("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")
        self.assertEqual(seen[-1]["method"],"DELETE")   # 失败也必须尝试清理

    async def test_files_lifecycle_upload_404_is_blocked(self):
        runner,client,_=await self._runner([(404,{"error":{"message":"not found"}}),(404,{"error":{"message":"not found"}})])
        try: outcome=await runner.case_files_lifecycle("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3],"blocked")

    async def test_files_listing_and_retrieve(self):
        listing={"object":"list","data":[{"id":"file-api-1","object":"file","bytes":10,"filename":"a.png","purpose":"user_data"}],"has_more":False}
        detail={"id":"file-api-1","object":"file","bytes":10,"filename":"a.png","purpose":"user_data"}
        runner,client,seen=await self._runner([(200,listing),(200,detail)])
        try: outcome=await runner.case_files_listing("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(seen[0]["query"].get("limit"),"5")
        self.assertEqual(seen[1]["path"],"/files/file-api-1")

        empty={"object":"list","data":[],"has_more":False}
        runner,client,_=await self._runner([(200,empty)])
        try: outcome=await runner.case_files_listing("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])   # 无文件时不校验单文件检索，但结构断言仍生效

    async def test_user_balance_shapes(self):
        runner,client,_=await self._runner([(200,{"is_available":True,"balance_infos":[{"currency":"CNY","total_balance":"110.00","granted_balance":"10.00","topped_up_balance":"100.00"}]})])
        try: outcome=await runner.case_user_balance("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])

        runner,client,_=await self._runner([(200,{"is_available":False,"balance_infos":[{"currency":"CNY","total_balance":"0","granted_balance":"0","topped_up_balance":"0"}]})])
        try: outcome=await runner.case_user_balance("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")

        runner,client,_=await self._runner([(200,{"balance_infos":[{"currency":"CNY"}]})])
        try: outcome=await runner.case_user_balance("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])

    async def test_stream_options_without_stream_contract(self):
        runner,client,seen=await self._runner([(400,{"error":{"message":"stream_options requires stream=true"}})])
        try: outcome=await runner.case_stream_options_without_stream("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertNotIn("stream",seen[0]["body"] or {})

        runner,client,_=await self._runner([(200,self._chat())])
        try: outcome=await runner.case_stream_options_without_stream("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")

    async def test_logprobs_structure(self):
        good={"choices":[{"message":{"content":"OK"},"logprobs":{"content":[{"token":"O","logprob":-0.1,"top_logprobs":[]}]}}]}
        runner,client,_=await self._runner([(200,good),(200,self._chat())])
        try: outcome=await runner.case_logprobs_structure("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])

        runner,client,_=await self._runner([(200,self._chat()), (200,self._chat())])
        try: outcome=await runner.case_logprobs_structure("deepseek-flash")
        finally: await client.aclose()
        # 200 但无 token 级对数概率：官方承诺返回、网关不透传 ⇒ 不影响正常使用，记 warning 不判失败
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")

    async def test_top_p_bounds(self):
        runner,client,seen=await self._runner([(400,{"error":{"message":"top_p must be <= 1"}}),(200,self._chat()),(200,self._chat())])
        try: outcome=await runner.case_top_p_bounds("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual([s["body"]["top_p"] for s in seen],[1.2,0.5,0.5])

        runner,client,_=await self._runner([(200,self._chat())])
        try: outcome=await runner.case_top_p_bounds("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")

    async def test_fim_thinking_conflict(self):
        runner,client,seen=await self._runner([(422,{"error":{"message":"FIM mode with thinking enabled"}})])
        try: outcome=await runner.case_fim_thinking_conflict("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(seen[0]["path"],"/beta/completions")

        runner,client,_=await self._runner([(200,{"choices":[{"text":"ok"}]})])
        try: outcome=await runner.case_fim_thinking_conflict("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")

        runner,client,_=await self._runner([(404,{"error":{"message":"not found"}})])
        try: outcome=await runner.case_fim_thinking_conflict("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3],"blocked")

    async def test_anthropic_system_and_stop_sequences(self):
        ok={"role":"assistant","content":[{"type":"text","text":"ABC"}],"stop_reason":"stop_sequence"}
        runner,client,seen=await self._runner([(200,ok)])
        try: outcome=await runner.case_anthropic_system_stop("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertIn("system",seen[0]["body"])
        self.assertEqual(seen[0]["path"],"/anthropic/v1/messages")

        leaked={"role":"assistant","content":[{"type":"text","text":"ABC<END>XYZ"}]}
        runner,client,_=await self._runner([(200,leaked)])
        try: outcome=await runner.case_anthropic_system_stop("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])

    async def test_anthropic_thinking_block(self):
        with_thinking={"role":"assistant","content":[{"type":"thinking","thinking":"推理"},{"type":"text","text":"837"}]}
        runner,client,seen=await self._runner([(200,with_thinking)])
        try: outcome=await runner.case_anthropic_thinking("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(seen[0]["body"]["thinking"],{"type":"enabled","budget_tokens":1024})
        self.assertEqual(seen[0]["body"]["output_config"],{"effort":"high"})

        runner,client,_=await self._runner([(200,{"role":"assistant","content":[{"type":"text","text":"837"}]})])
        try: outcome=await runner.case_anthropic_thinking("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")

    async def test_responses_stream_semantic_events(self):
        def sse(events, done=False):
            body="".join("data: %s\n\n"%json.dumps(e,ensure_ascii=False) for e in events)
            if done: body+="data: [DONE]\n\n"
            async def handler(request): return httpx.Response(200,text=body,headers={"content-type":"text/event-stream"})
            return handler
        good=[{"type":"response.created","sequence_number":0},
              {"type":"response.output_text.delta","sequence_number":1,"delta":"OK"},
              {"type":"response.completed","sequence_number":2,"response":{"usage":{"output_tokens":2}}}]
        runner,client,_=await self._runner([sse(good)])
        try: outcome=await runner.case_responses_stream("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])

        bad=[{"type":"response.output_text.delta","sequence_number":1,"delta":"OK"},
             {"type":"response.completed","sequence_number":2}]
        runner,client,_=await self._runner([sse(bad)])
        try: outcome=await runner.case_responses_stream("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertIn("response.created",outcome[1])

        runner,client,_=await self._runner([sse(good, done=True)])
        try: outcome=await runner.case_responses_stream("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertIn("[DONE]",outcome[1])

    async def test_responses_ignored_fields_fixed_values(self):
        body={"id":"resp_1","output":[{"type":"message"}],"store":False,"previous_response_id":None,"parallel_tool_calls":True}
        runner,client,seen=await self._runner([(200,body)])
        try: outcome=await runner.case_responses_ignored_fields("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        for key in ("metadata","include","service_tier","stream_options","background","truncation"):
            self.assertIn(key,seen[0]["body"])

        runner,client,_=await self._runner([(200,{"id":"resp_1","output":[],"store":True,"previous_response_id":"resp_x","parallel_tool_calls":True})])
        try: outcome=await runner.case_responses_ignored_fields("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])

    async def test_parallel_tool_calls_single_call_is_warning(self):
        one={"choices":[{"message":{"content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"get_number","arguments":"{}"}}]}}]}
        runner,client,_=await self._runner([(200,one)])
        try: outcome=await runner.case_parallel_tool_calls("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")

        two={"choices":[{"message":{"content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"get_number","arguments":"{}"}},
                                                              {"id":"c2","type":"function","function":{"name":"get_text","arguments":"{}"}}]}}]}
        runner,client,_=await self._runner([(200,two)])
        try: outcome=await runner.case_parallel_tool_calls("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(len(outcome),3)

    async def test_stop_multi_sequences_and_over_limit(self):
        runner,client,_=await self._runner([(200,self._chat("AB")),(400,{"error":{"message":"at most 16 stop sequences"}})])
        try: outcome=await runner.case_stop_multi_sequences("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])

        runner,client,_=await self._runner([(200,self._chat("AB")),(200,self._chat())])
        try: outcome=await runner.case_stop_multi_sequences("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")

        runner,client,_=await self._runner([(200,self._chat("AB<A>Q1"))])
        try: outcome=await runner.case_stop_multi_sequences("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])

    async def test_deprecated_penalties(self):
        runner,client,seen=await self._runner([(200,self._chat()),(200,self._chat())])
        try: outcome=await runner.case_deprecated_penalties("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(seen[1]["body"]["frequency_penalty"],1.8)

        runner,client,_=await self._runner([(200,self._chat()),(400,{"error":{"message":"frequency_penalty is deprecated"}})])
        try: outcome=await runner.case_deprecated_penalties("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")

    async def test_fim_json_conflict_and_prefix_without_flag(self):
        runner,client,_=await self._runner([(400,{"error":{"message":"FIM + JSON mode is not supported"}})])
        try: outcome=await runner.case_fim_json_conflict("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])

        runner,client,seen=await self._runner([(200,self._chat("def add(a, b):\n    return a+b"))])
        try: outcome=await runner.case_prefix_without_flag("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")
        self.assertNotIn("prefix",seen[0]["body"]["messages"][1])

    async def test_anthropic_tool_loop(self):
        first={"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"lookup_order","input":{"order_id":"ORD-731"}}]}
        second={"role":"assistant","content":[{"type":"text","text":"订单 ORD-731 已发货。"}]}
        runner,client,seen=await self._runner([(200,first),(200,second)])
        try: outcome=await runner.case_anthropic_tool_loop("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(seen[0]["body"]["tool_choice"],{"type":"any"})
        self.assertEqual(seen[1]["body"]["messages"][2]["content"][0]["tool_use_id"],"toolu_1")

        runner,client,_=await self._runner([(200,{"role":"assistant","content":[{"type":"text","text":"我直接回答"}]})])
        try: outcome=await runner.case_anthropic_tool_loop("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])

    async def test_responses_tool_matrix(self):
        call={"id":"resp_1","output":[{"type":"function_call","name":"get_number","arguments":"{}"}]}
        msg={"id":"resp_1","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"OK"}]}]}
        bad=(400,{"error":{"message":"Unsupported custom tool: 'not_apply_patch'. Only 'apply_patch' is supported."}})
        ignored=(200,{"id":"resp_2","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"OK"}]}]})
        thinking_400=(400,{"error":{"message":"Thinking mode does not support this tool_choice","type":"invalid_request_error"}})

        # 真实批次口径：默认思考模式下 required/指定工具均 4xx（文档承诺），
        # 关思考（reasoning.effort=none）后 required 触发 function_call → passed
        runner,client,seen=await self._runner([thinking_400,thinking_400,(200,call),bad,ignored])
        try: outcome=await runner.case_responses_tool_matrix("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[2]["triggered_by"],"required+effort=none")
        self.assertEqual(seen[2]["body"]["reasoning"],{"effort":"none"})
        self.assertEqual(seen[2]["body"]["tool_choice"],"required")
        self.assertNotIn("reasoning",seen[0]["body"])            # 负向探针必须用默认（思考）模式
        self.assertEqual(seen[1]["body"]["tool_choice"],{"type":"function","name":"get_number"})
        self.assertNotIn("reasoning",seen[1]["body"])
        self.assertEqual(len(outcome),3)     # 只返回三元组 → 判定为 passed，不附带 warning 状态

        # 上游放宽：默认模式下 required 直接触发 → 能力具备，但记 warning（与文档不一致）
        runner,client,_=await self._runner([(200,call),thinking_400,(200,call),bad,ignored])
        try: outcome=await runner.case_responses_tool_matrix("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")

        # 默认模式下 required 被静默忽略（200 但没触发）→ warning（文档承诺的 4xx 未执行，不升级 failed）
        runner,client,_=await self._runner([(200,msg),thinking_400,(200,call),bad,ignored])
        try: outcome=await runner.case_responses_tool_matrix("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")
        self.assertIn("强制调用被静默忽略",outcome[1])

        # 文档解法失效 → 兜底到「指定工具+关思考」→ warning
        runner,client,seen=await self._runner([thinking_400,thinking_400,(200,msg),(200,call),bad,ignored])
        try: outcome=await runner.case_responses_tool_matrix("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")
        self.assertEqual(outcome[2]["triggered_by"],"指定工具+effort=none")
        self.assertEqual(seen[3]["body"]["reasoning"],{"effort":"none"})

        # 强制形态全不可用 → auto + 两轮强提示兜底 → warning
        runner,client,seen=await self._runner([thinking_400,thinking_400,(200,msg),(200,msg),(200,msg),(200,call),bad,ignored])
        try: outcome=await runner.case_responses_tool_matrix("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")
        self.assertEqual(outcome[2]["triggered_by"],"auto+强提示")
        self.assertNotEqual(seen[4]["body"]["input"],seen[5]["body"]["input"])

        # 任何形态都触发不了 → failed
        runner,client,_=await self._runner([thinking_400,thinking_400,(200,msg),(200,msg),(200,msg),(200,msg),bad,ignored])
        try: outcome=await runner.case_responses_tool_matrix("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertIn("仍未被触发",outcome[1])

        # custom 非 apply_patch 未被拒（文档承诺 400 但线上放行 200）→ warning（宽松放行，不判失败）
        runner,client,_=await self._runner([thinking_400,thinking_400,(200,call),(200,{"id":"resp_3","output":[]}),ignored])
        try: outcome=await runner.case_responses_tool_matrix("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")
        self.assertIn("custom 非 apply_patch 未被拒绝",outcome[1])

        # 网关只认自有工具类型集合（web_search 被 4xx 拒绝）→ warning（网关契约差异，不判失败）
        gateway_ignored=(400,{"error":{"message":"invalid_value: Supported values are: function, mcp, knowledge_search, namespace"}})
        runner,client,_=await self._runner([thinking_400,thinking_400,(200,call),bad,gateway_ignored])
        try: outcome=await runner.case_responses_tool_matrix("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")
        self.assertIn("web_search",outcome[1])

        # 内置工具被执行（出现 web_search_call）→ failed
        runner,client,_=await self._runner([thinking_400,thinking_400,(200,call),bad,(200,{"id":"resp_4","output":[{"type":"web_search_call"}]})])
        try: outcome=await runner.case_responses_tool_matrix("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertIn("web_search",outcome[1])

        # 端点不存在 → blocked，不当作模型行为结论
        runner,client,_=await self._runner([(404,{})])
        try: outcome=await runner.case_responses_tool_matrix("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3],"blocked")

    async def test_model_alias_reports_doc_drift(self):
        runner,client,seen=await self._runner([(200,self._chat()),(200,self._chat())])
        try: outcome=await runner.case_model_alias("deepseek-v4-flash-260813")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertTrue(outcome[2]["configured_model_off_doc"])
        self.assertEqual(seen[1]["body"]["model"],RETIRED_ALIASES[0])
        self.assertEqual(list(PRIMARY_MODELS),["deepseek-v4-pro","deepseek-flash"])

        runner,client,_=await self._runner([(200,self._chat()),(404,{"error":{"message":"model not found"}})])
        try: outcome=await runner.case_model_alias("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")


class ModelGateTests(unittest.TestCase):
    """模型门禁：误配其他家族模型必须在发任何请求前被拦截（2026-09-24 gpt-5.6-sol 事故）。"""

    def test_gate_accepts_deepseek_family_names(self):
        for name in ("deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4", "DeepSeek-V4-Pro"):
            ok, reason = model_matches_family("deepseek", name)
            self.assertTrue(ok, f"{name} 应被放行：{reason}")

    def test_gate_rejects_other_family_models(self):
        for name in ("gpt-5.6-sol", "glm-5.3", "kimi-k3"):
            ok, reason = model_matches_family("deepseek", name)
            self.assertFalse(ok, f"{name!r} 不应被放行")
            self.assertIn("deepseek", reason)
        for name in ("", "  "):
            ok, reason = model_matches_family("deepseek", name)
            self.assertFalse(ok, f"{name!r} 不应被放行")
            self.assertTrue(reason)

    def test_runner_aborts_with_exit_3_and_no_result_dir(self):
        import asyncio
        import os
        import tempfile
        from pathlib import Path

        from dsv4.runner import main_async

        keys = ("DS_API_KEY", "DS_BASE_URL", "DS_MODEL", "VOLC_PROTOCOL")
        saved = {k: os.environ.get(k) for k in keys}
        root = Path(tempfile.mkdtemp())
        os.environ.update({"DS_API_KEY": "k", "DS_BASE_URL": "https://api.deepseek.com/v1",
                           "DS_MODEL": "gpt-5.6-sol", "VOLC_PROTOCOL": "openai"})
        try:
            rc = asyncio.run(main_async(["DS_V4_P0", "--output-root", str(root)]))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertEqual(rc, 3)
        self.assertEqual(list(root.iterdir()), [], "门禁拦截后不应产生任何结果目录")
