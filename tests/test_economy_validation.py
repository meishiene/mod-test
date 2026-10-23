import asyncio
import json

import httpx
import pytest

from common.economy_profiles import ECONOMY_CASES, economy_profile
from common.economy_validation import EconomyRunner
from economy_fixture import request_history, solve_fixture, response_fixture


@pytest.mark.parametrize('kind,model', [
    ('deepseek', 'deepseek-v4-pro'), ('glm', 'glm-5.3-flashx'),
    ('qwen', 'qwen3.8-max'), ('kvv', 'kimi-k3'),
    ('fable', 'claude-sonnet-4-6'), ('gpt', 'gpt-6.1-sol'),
])
def test_all_http_failures_stay_incomplete_and_bounded(kind, model, tmp_path):
    calls = []
    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(429, json={'error': {'message': 'rate limit'}})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            runner = EconomyRunner(economy_profile(kind, model), 'https://example.test/v1', 'secret', tmp_path, client=client)
            return await runner.run()
    report = asyncio.run(run())
    assert report['admission_passed'] is False
    assert not any(r['status'] == 'passed' for r in report['results'])
    assert len(calls) <= 11
    assert all(p.get('max_tokens', p.get('max_output_tokens', 0)) <= 2048 for p in calls)
    assert report['economy_validation']['vendor_full_strength_proven'] is False
    assert 'secret' not in (tmp_path / 'requests.jsonl').read_text()


def test_cross_family_rejected_and_expensive_kimi_card_absent():
    with pytest.raises(ValueError):
        economy_profile('deepseek', 'gpt-6.1-sol')
    assert len(ECONOMY_CASES) == 6
    assert 'KVV_TOOL_SCHEMA' not in ECONOMY_CASES


def test_truncated_answer_is_blocked(tmp_path):
    def respond(request):
        return httpx.Response(200, json={'model': 'glm-5.3', 'choices': [
            {'message': {'role': 'assistant', 'content': 'wrong'}, 'finish_reason': 'length'}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            runner = EconomyRunner(economy_profile('glm', 'glm-5.3'), 'https://example.test/v1', 'secret', tmp_path, client=client)
            return await runner.run()
    report = asyncio.run(run())
    assert all(r['status'] == 'blocked' for r in report['results'])
    assert report['admission_passed'] is False


MODELS = [('deepseek','deepseek-v4-pro'), ('glm','glm-5.3'), ('qwen','qwen3.7-max'),
          ('kvv','kimi-k3'), ('fable','claude-sonnet-4-6'), ('fable','claude-opus-5'),
          ('gpt','gpt-5.6-sol'), ('gpt','gpt-6-sol'), ('gpt','gpt-6-astra'), ('gpt','gpt-6.1-sol')]


def execute_fixture(tmp_path, kind, model, mutate=None, profile=None):
    profile = profile or economy_profile(kind,model)
    requests=[]
    def respond(request):
        payload=json.loads(request.content); requests.append(payload)
        if 'nonexistent-' in payload['model']:
            return httpx.Response(404,json={'error':{'code':'model_not_found','message':'model not found'}})
        text,calls=solve_fixture(request_history(payload,profile.transport))
        body=response_fixture(model,profile.transport,text,calls)
        if mutate: body=mutate(body,payload)
        return httpx.Response(200,json=body)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            runner=EconomyRunner(profile,'https://example.test/v1','secret',tmp_path,client=client)
            return await runner.run()
    return asyncio.run(run()),requests


@pytest.mark.parametrize('kind,model', MODELS)
def test_complete_real_protocol_flow_matches_budget(kind,model,tmp_path):
    report,requests=execute_fixture(tmp_path,kind,model)
    assert report['admission_passed'], [(r['case_id'],r['observations']) for r in report['results'] if not r['passed']]
    assert len(requests)==11
    assert report['economy_validation']['requests_sent']==11
    assert report['economy_validation']['requested_output_tokens']==10*2048+64
    assert report['economy_validation']['serialized_input_chars']<=64000
    assert all('KVV_TOOL_SCHEMA' not in json.dumps(p) for p in requests)


@pytest.mark.parametrize('failure',['wrong_answer','wrong_model','wrong_tools','bad_schema','budget'])
def test_false_positive_counterexamples(failure,tmp_path):
    from dataclasses import replace
    def mutate(body,payload):
        msg=body['choices'][0]['message']
        if failure=='wrong_model': body['model']='glm-5.3-flash'
        if failure=='wrong_answer' and 'arithmetic=' in json.dumps(payload): msg['content']='{}'
        if failure=='bad_schema' and '单价' in json.dumps(payload,ensure_ascii=False):
            parsed=json.loads(msg['content']); parsed['extra']=1; msg['content']=json.dumps(parsed)
        if failure=='wrong_tools' and msg.get('tool_calls'): msg['tool_calls'][0]['function']['arguments']='{}'
        return body
    profile=economy_profile('glm','glm-5.3')
    if failure=='budget': profile=replace(profile,max_requests=2)
    report,requests=execute_fixture(tmp_path,'glm','glm-5.3',mutate,profile)
    assert not report['admission_passed']
    if failure=='budget': assert len(requests)==2
    else: assert any(r['status']=='failed' for r in report['results'])


def test_input_budget_blocks_before_network(tmp_path):
    from dataclasses import replace
    report, requests = execute_fixture(tmp_path, 'glm', 'glm-5.3',
        profile=replace(economy_profile('glm', 'glm-5.3'), max_input_chars=10))
    assert requests == []
    assert report['economy_validation']['serialized_input_chars'] == 0
    assert all(r['status'] == 'blocked' for r in report['results'])


@pytest.mark.parametrize('kind,model,transport', [('gpt','gpt-6.1-sol','responses'),
                                                ('fable','claude-opus-5','messages')])
def test_native_reasoning_and_signatures_survive_tool_replay(kind,model,transport):
    from common.economy_validation import decode_response, encode_request
    body = response_fixture(model,transport,'',[('lookup_order', {'order_id':'O1'})])
    native = ({'type':'reasoning','id':'rs1','summary':[], 'encrypted_content':'opaque-proof'}
              if transport == 'responses' else {'type':'thinking','thinking':'trace','signature':'opaque-proof'})
    key = 'output' if transport == 'responses' else 'content'
    body[key].insert(0, native)
    decoded, gap = decode_response(transport, body)
    assert gap is None
    message = decoded['choices'][0]['message']
    _, payload = encode_request(economy_profile(kind,model),model,
        [{'role':'user','content':'query'}, message,
         {'role':'tool','tool_call_id':'call-0','content':'{"error":"retryable"}'}], {'max_tokens':2048})
    replay = payload['input'] if transport == 'responses' else payload['messages'][1]['content']
    assert native in replay
    assert 'opaque-proof' in json.dumps(payload)


@pytest.mark.parametrize('transport,body,reason', [
    ('chat',{},'invalid_envelope'),
    ('responses',{'status':'incomplete','output':[]},'output_budget_exhausted'),
    ('messages',{'type':'message','stop_reason':'max_tokens','content':[]},'output_budget_exhausted'),
])
def test_protocol_specific_missing_evidence(transport,body,reason):
    from common.economy_validation import decode_response
    assert decode_response(transport,body) == (None,reason)


def test_missing_identity_is_not_a_pass(tmp_path):
    def mutate(body,payload):
        body.pop('model',None)
        return body
    report,requests=execute_fixture(tmp_path,'glm','glm-5.3',mutate)
    assert not report['admission_passed']
    assert [r['status'] for r in report['results'][-2:]] == ['blocked','blocked']
    assert len(requests) == 10


@pytest.mark.parametrize('status,error,expected', [
    (401,{'message':'unknown model, authentication failed'},'blocked'),
    (404,{'message':'model access denied'},'blocked'),
    (400,{'message':'模型不存在，余额不足'},'blocked'),
    (404,{'message':'模型不存在'},'passed'),
    (200,None,'failed'),
])
def test_route_negative_requires_specific_rejection(tmp_path,status,error,expected):
    def respond(request):
        return httpx.Response(status,json={'error':error} if error else response_fixture('glm-5.3','chat','OK',[]))
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            runner=EconomyRunner(economy_profile('glm','glm-5.3'),'https://example.test/v1','secret',tmp_path,client=client)
            runner.responses=[{'case_id':'control','valid_response':True,'has_answer':True,'model':'glm-5.3'}]
            try: return await runner.case_economy_route('glm-5.3')
            finally: runner.audit.close()
    result=asyncio.run(run())
    assert result[3] == expected


def test_finalization_and_batch_do_not_award_legacy_scores(tmp_path):
    from common.suite_runtime import finalize
    from common.scoring import score_batch, score_summary
    from scripts.run_economy_validation import write_report
    summary,_=execute_fixture(tmp_path,'glm','glm-5.3')
    write_report(tmp_path,summary)
    finalized=finalize(tmp_path,summary)
    assert finalized['admission_passed'] is True
    assert finalized['score'] is None
    assert score_summary(summary) is None
    assert score_batch([{'summary':summary}]) == []
    assert '20' in (tmp_path/'report.md').read_text()
    # Partial evidence must remain non-admitted even though legacy finalization
    # normally regards blocked cases as nonblocking.
    summary['results'][0].update(status='blocked',passed=False)
    summary['admission_passed']=False
    assert finalize(tmp_path,summary)['admission_passed'] is False


@pytest.mark.parametrize('kind,model,protocol', [('gpt','gpt-6.1-sol','openai'),
                                              ('fable','claude-sonnet-4-6','anthropic')])
def test_cli_generates_final_artifacts_with_no_live_requests(tmp_path,monkeypatch,kind,model,protocol):
    from scripts import run_economy_validation as cli
    from common.faces import FACES_ENV
    profile=economy_profile(kind,model)
    seen=[]
    def respond(request):
        seen.append(request)
        payload=json.loads(request.content)
        if 'nonexistent-' in payload['model']:
            return httpx.Response(404,json={'error':{'code':'model_not_found'}})
        text,calls=solve_fixture(request_history(payload,profile.transport))
        return httpx.Response(200,json=response_fixture(model,profile.transport,text,calls))
    real_client=httpx.AsyncClient
    monkeypatch.setattr(cli.httpx,'AsyncClient',lambda **kwargs: real_client(transport=httpx.MockTransport(respond),**kwargs))
    for key,value in {'VOLC_BASE_URL':'https://example.test','VOLC_MODEL':model,'VOLC_API_KEY':'cli-secret',
                      'VOLC_PROTOCOL':protocol,FACES_ENV:'model,verify'}.items():
        monkeypatch.setenv(key,value)
    assert cli.main([f'{kind.upper()}_ECONOMY','--output-root',str(tmp_path)]) == 0
    report_path=next(tmp_path.glob('*/summary.json'))
    report=json.loads(report_path.read_text())
    assert report['score'] is None and report['admission_passed']
    assert len(seen)==11
    suffix='/v1/responses' if kind=='gpt' else '/v1/messages'
    assert all(str(r.url).endswith(suffix) for r in seen)
    assert 'cli-secret' not in (report_path.parent/'requests.jsonl').read_text()
    assert (report_path.parent/'validation.json').is_file()


def test_cli_rejects_incompatible_protocol_before_network(tmp_path,monkeypatch):
    from scripts import run_economy_validation as cli
    from common.faces import FACES_ENV
    for key,value in {'VOLC_BASE_URL':'https://example.test','VOLC_MODEL':'gpt-6.1-sol',
                      'VOLC_API_KEY':'secret','VOLC_PROTOCOL':'openai_images',FACES_ENV:'model,verify'}.items():
        monkeypatch.setenv(key,value)
    with pytest.raises(SystemExit) as caught:
        cli.main(['GPT_ECONOMY','--output-root',str(tmp_path)])
    assert caught.value.code == 2
    assert list(tmp_path.iterdir()) == []
