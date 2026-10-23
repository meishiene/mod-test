"""Behavioral counterexamples for shared model validation (no remote calls)."""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from common.model_validation import validation_profile, assess_validation, required_case_ids
from common.probe_assertions import valid_tools, stream_tools, strict_json, identity_match


def test_profiles_do_not_confuse_flash_with_flashx():
    assert validation_profile('provider/glm-5.3-flashx').tier == 'flashx'
    assert validation_profile('GLM-5.3-FlashX').model_id == 'glm-5.3-flashx'
    assert validation_profile('glm5.3flash').tier == 'flash'
    assert validation_profile('qwen3.7-max').image_input is False
    assert validation_profile('qwen3.8-max').image_input is True
    assert validation_profile('qwen3.8-max-fake') is None
    assert not identity_match('glm-5.3', 'glm-5.3-flash')
    assert not identity_match('glm-5.3-flash', 'glm-5.3-flashx')
    assert not identity_match('glm-5.3', None)


def test_missing_or_blocked_evidence_never_proves_full_strength():
    for model in ('glm-5.3', 'glm-5.3-flash', 'glm-5.3-flashx', 'qwen3.8-max', 'qwen3.7-max'):
        rows = [{'case_id': cid, 'status': 'blocked', 'passed': False} for cid in required_case_ids(model)]
        rows.append({'case_id': 'GLM-53-F08', 'status': 'passed', 'passed': True})
        result = assess_validation(rows, model)
        assert not result['admission_passed']
        assert result['full_strength']['state'] == 'insufficient_evidence'
        assert result['authenticity']['proven'] is False


def test_stream_requires_real_complete_tool_calls():
    assert stream_tools([]) == []
    tool = {'type': 'function', 'function': {'name': 'get_weather', 'parameters': {
        'type': 'object', 'properties': {'city': {'type': 'string'}},
        'required': ['city'], 'additionalProperties': False}}}
    call = {'id': 'c1', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city":"北京"}'}}
    assert valid_tools([call], [tool], {'get_weather': {'city': '北京'}})
    assert not valid_tools([{**call, 'function': {'name': 'get_weather', 'arguments': '{}'}}], [tool])
    assert not valid_tools([{**call, 'function': {'name': 'wrong', 'arguments': '{}'}}], [tool])
    assert not strict_json('{"answer":"OK","extra":1}', {
        'type':'object','properties':{'answer':{'const':'OK'}},'required':['answer'],'additionalProperties':False})

from common.probe_program import Program, ProgramRejected
from common.model_validation import PROFILES, PROBE_DEFINITIONS, DEFAULT_SEED, apply_validation_gate
from qwen.core import QwenRunner, QwenConfig
from glm53.core import GLM53Runner, GLMConfig

GOOD_CODE = '''def solve(nums):
    result = []
    for n in nums:
        if n >= 0 and nums.count(n) >= 2 and n not in result:
            result.append(n)
    return sorted(result)
'''


def chat(text='', model='glm-5.3', **message):
    return {'model': model, 'choices': [{'message': {'role':'assistant', 'content':text, **message}}]}


def runner_for(model):
    return QwenRunner(QwenConfig('offline','https://invalid.example')) if model.startswith('qwen') else GLM53Runner(GLMConfig('offline','https://invalid.example'))


def test_generated_code_is_interpreted_without_host_execution():
    assert Program(GOOD_CODE).run([1,2,1,-1,-1,0,0]) == [0,1]
    for source in ('import os\ndef solve(nums): return []',
                   'def solve(nums): return __import__("os").system("false")',
                   'def solve(nums): return nums.__class__',
                   'def solve(nums):\n while True: pass',
                   'def solve(nums): return [1] * 1000000000'):
        with pytest.raises((ProgramRejected, TypeError)):
            Program(source).run([1])


def test_code_probe_runs_hidden_edge_cases_and_rejects_hardcoding():
    async def run():
        for source, expected in ((GOOD_CODE, True), ('def solve(nums): return [1,3]', False)):
            runner=runner_for('glm-5.3')
            runner.request=AsyncMock(return_value=(200,chat(source),0,{}))
            result=await runner.case_quality_code('glm-5.3')
            assert result[0] is expected
            assert len(result[2]['rows'][0]['checks']) == 9
    asyncio.run(run())


@pytest.mark.parametrize('model', list(PROFILES))
def test_profile_context_contains_three_positions_and_actual_workload(model):
    async def run():
        runner=runner_for(model)
        captured=[]
        async def request(m,messages,**kwargs):
            import re
            captured.append((m,messages,kwargs))
            document=messages[0]['content']
            expected=dict(re.findall(r'(HEAD|MIDDLE|TAIL)=([0-9a-f]+)',document))
            return 200,{**chat(json.dumps(expected),model), 'usage':{'prompt_tokens':1234}},0,{}
        runner.request=request
        result=await runner.case_quality_context(model)
        assert result[0]
        obs=result[2]
        assert obs['prompt_tokens']==1234
        assert obs['prompt_chars']>=PROFILES[model].context_probe_chars
        document=captured[0][1][0]['content']
        assert document.index('HEAD=') < len(document)*.1
        assert .4*len(document) < document.index('MIDDLE=') < .6*len(document)
        assert document.index('TAIL=') > len(document)*.9
        assert captured[0][2].get('enable_thinking') is False if model.startswith('qwen') else captured[0][2]['thinking']=={'type':'enabled'}
    asyncio.run(run())


def test_unknown_model_does_not_inherit_flagship_profile():
    async def run():
        runner=runner_for('glm-unknown');runner.request=AsyncMock()
        result=await runner.case_quality_context('glm-unknown')
        assert result[3]=='blocked'
        runner.request.assert_not_called()
    asyncio.run(run())


@pytest.mark.parametrize('model', ['glm-5.3','glm-5.3-flash','glm-5.3-flashx','qwen3.8-max','qwen3.7-max'])
def test_wrong_math_and_done_only_tool_stream_fail(model):
    async def run():
        runner=runner_for(model)
        runner.request=AsyncMock(return_value=(200,chat('2',model,reasoning_content='anything'),0,{}))
        assert not (await runner.case_quality_reasoning(model))[0]
        handler=runner.case_thinking_default if model.startswith('qwen') else runner.case_thinking
        assert not (await handler(model))[0]
        runner.request=AsyncMock(return_value=(200,None,0,{'done':True,'invalid_events':0,'content':''}))
        assert not (await runner.case_tool_stream(model))[0]
    asyncio.run(run())


def test_fragmented_stream_tools_validate_arguments_and_indexes():
    events=[{'choices':[{'index':0,'delta':{'tool_calls':[
        {'index':0,'id':'c','type':'function','function':{'name':'get_weather','arguments':'{"ci'}}]}}]},
        {'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'function':{'arguments':'ty":"北京"}'}}]}}]}]
    calls=stream_tools(events)
    assert valid_tools(calls,runner_for('qwen3.8-max').tools(),{'get_weather':{'city':'北京'}})
    events[1]['choices'][0]['delta']['tool_calls'][0]['index']=1
    assert not valid_tools(stream_tools(events),runner_for('qwen3.8-max').tools())


def test_all_blocked_fingerprints_and_synthetic_summary_cannot_admit():
    from qwen.core import CaseResult as QR, summarize_results as qs
    from glm53.core import CaseResult as GR, summarize_results as gs
    from common.faces import face_verdict
    from glm53.faces import CASE_FACES
    for result_type,summarize,prefix,model,count in ((QR,qs,'QWEN','qwen3.8-max',6),(GR,gs,'GLM-53','glm-5.3',8)):
        rows=[result_type(f'{prefix}-F0{i}','probe',model,False,'blocked') for i in range(1,count)]
        rows.append(result_type(f'{prefix}-F0{count}','summary',model,True,'passed'))
        summary=summarize(rows)
        assessment=summary.get('tier_assessment') or summary['fingerprint_assessment']
        assert assessment['level']=='无法判定'
        verdict=summary.get('relay_verdict') or face_verdict(rows,CASE_FACES)
        assert not verdict['admission_passed']


def test_reference_requires_exact_model_version_seed_profile_and_complete_results():
    model='glm-5.3-flashx'
    rows=[{'case_id':cid,'status':'passed','passed':True,'model':model} for cid in required_case_ids(model)]
    result=assess_validation(rows,model)
    assert result['admission_passed']
    assert result['full_strength']['state']=='profile_passed'
    assert not result['full_strength']['vendor_full_strength_proven']
    reference={'summary':{'model':model,'case':'GLM_53_FULL','admission_passed':True,'model_validation':result,'results':rows},'sha256':'abc'}
    assert assess_validation(rows,model,reference)['full_strength']['state']=='reference_consistent'
    for change in ({'model':'glm-5.3-flash'}, {'admission_passed':False}, {'case':'GLM_53_CORE'}, {'model_validation':{**result,'seed':0}}, {'results':rows[:-1]}):
        invalid={**reference,'summary':{**reference['summary'],**change}}
        verdict=assess_validation(rows,model,invalid)
        assert verdict['reference']['state']=='incompatible'
        assert not verdict['admission_passed']
    degraded=[{**r,'status':'failed','passed':False} if r['case_id'].endswith('Q01') else r for r in rows]
    assert assess_validation(degraded,model,reference)['reference']['state']=='regression'


def test_full_gate_survives_summary_normalization():
    from common.result_adapter import normalize_summary
    summary={'relay_verdict':{'contract':'relay','admission_passed':True},
             'model_validation':assess_validation([], 'qwen3.8-max')}
    apply_validation_gate(summary,True)
    assert not normalize_summary(summary)['admission_passed']


@pytest.mark.parametrize('status,error,expected',[
    (200,None,'failed'), (429,{'message':'model quota'},'blocked'),
    (401,{'message':'model authentication'},'blocked'), (400,{'message':'bad parameter'},'blocked'),
    (400,{'message':'model quota exceeded'},'blocked'),
    (404,{'message':'model not found'},'passed')])
def test_unknown_model_route_requires_positive_control_and_semantic_error(status,error,expected):
    async def run():
        runner=runner_for('glm-5.3-flashx')
        runner.request=AsyncMock(side_effect=[(200,chat('OK','glm-5.3-flashx'),0,{}),
                                             (status,{'error':error},0,{})])
        result=await runner.case_verify_route('glm-5.3-flashx')
        assert result[3]==expected
    asyncio.run(run())


def test_identity_distinguishes_same_family_tiers_and_fresh_challenges():
    async def run():
        runner=runner_for('glm-5.3-flashx');nonces=[]
        async def request(model,messages,**kwargs):
            nonce=messages[0]['content'].split('：')[-1];nonces.append(nonce)
            return 200,chat(nonce,'glm-5.3-flash'),0,{}
        runner.request=request
        result=await runner.case_verify_identity('glm-5.3-flashx')
        assert result[3]=='failed'
        assert len(set(nonces))==3
    asyncio.run(run())


def test_agent_requires_recovery_and_use_of_real_tool_result():
    async def run():
        for correct in (True,False):
            runner=runner_for('qwen3.8-max');seen=[]
            async def request(model,messages,**kwargs):
                import re
                seen.append(messages)
                order=re.search(r'ORD-\d+',messages[0]['content'])[0]
                if len(messages)<4:
                    calls=[{'id':f'call-{len(messages)}','type':'function','function':{
                        'name':'lookup_order','arguments':json.dumps({'order_id':order})}}]
                    return 200,chat(None,model,tool_calls=calls),0,{}
                return 200,chat(messages[-1]['content'] if correct else '{}',model),0,{}
            runner.request=request
            result=await runner.case_quality_agent('qwen3.8-max')
            assert result[0] is correct
            assert len(seen)==3
    asyncio.run(run())


def test_registry_has_each_shared_handler_and_no_unknown_scoring_faces():
    from common.score_registry import inventory
    for kind,model in [('qwen','qwen3.8-max'),('glm','glm-5.3-flashx')]:
        runner=runner_for(model);prefix=validation_profile(model).prefix
        registered=inventory(kind,model)
        for suffix,_,_,handler,face in PROBE_DEFINITIONS:
            assert hasattr(runner,'case_'+handler)
            assert registered[f'{prefix}-{suffix}'].face==face

@pytest.mark.parametrize('model', ['glm-5.3','glm-5.3-flash','glm-5.3-flashx','qwen3.8-max','qwen3.7-max'])
def test_comprehensive_transport_audit_and_assessment(model,tmp_path):
    """Positive end-to-end offline path, including actual SSE delta parsing."""
    import re
    import httpx
    from qwen.cases import CASES as QC
    from glm53.cases import CASES as GC
    from common.faces import annotate_summary, DEFAULT_FACES
    from qwen.faces import CASE_FACES as QF
    from glm53.faces import CASE_FACES as GF
    from common.finalizer import finalize_result_dir
    from qwen.core import summarize_results as qs, write_report as qw
    from glm53.core import summarize_results as gs, write_report as gw
    requests=[]
    runner=None

    def handler(request):
        payload=json.loads(request.content);requests.append(payload)
        assert payload['model']==model or '-nonexistent-' in payload['model']
        if '-nonexistent-' in payload['model']:
            return httpx.Response(404,json={'error':{'message':'model not found'}})
        cid=runner.current_case_id
        suffix=cid.removeprefix('QWEN-').removeprefix('GLM-53-')
        messages=payload['messages']; prompt=messages[-1]['content']; text='OK'; extra={}
        if suffix=='Q01':
            if prompt.startswith('计算 ('):
                a,b,c=map(int,re.findall(r'\d+',prompt));text=str(a*b-c)
            else:
                prices=[int(v) for v in re.findall(r'价格(\d+)元',prompt)]
                units=int(re.search(r'至少买(\d+)件',prompt)[1]);ca,cb=prices
                text=str(min(a*ca+b*cb for a in range(units+1) for b in range(units+1) if 3*a+5*b>=units))
        elif suffix=='Q02': text=GOOD_CODE
        elif suffix=='Q03':
            order=re.search(r'订单(ORD-\d+)',prompt)[1]
            quantity=int(re.search(r'买(\d+)件',prompt)[1]);price=int(re.search(r'单价(\d+)元',prompt)[1])
            text=json.dumps({'order_id':order,'items':[{'sku':'A1','quantity':quantity,'unit_price':price}],
                             'total':quantity*price,'paid':True,'note':None})
        elif suffix in ('Q04','08','F07') and model.startswith('glm') or suffix in ('Q04','06') and model.startswith('qwen'):
            text=json.dumps(dict(re.findall(r'(HEAD|MIDDLE|TAIL)=([0-9a-f]+)',messages[0]['content'])))
        elif suffix=='R01': text=prompt.split('：')[-1]
        elif suffix=='R03': text=re.search(r'只能输出字符串([0-9a-f]+)',messages[0]['content'])[1]
        elif suffix in ('Q05','Q06'):
            if suffix=='Q05' and len(messages)==5: text=messages[-1]['content']
            else:
                order=re.search(r'ORD-\d+',messages[0]['content'])[0]
                calls=[{'id':'c1','type':'function','function':{'name':'lookup_order','arguments':json.dumps({'order_id':order})}}]
                if suffix=='Q06':
                    sku=re.search(r'SKU-\d+',prompt)[0]
                    calls.append({'id':'c2','type':'function','function':{'name':'get_stock','arguments':json.dumps({'sku':sku})}})
                text=None;extra['tool_calls']=calls
        elif payload.get('tools'):
            if messages[-1]['role']=='tool': text='北京晴，25 度'
            else:
                extra['tool_calls']=[{'id':'c1','type':'function','function':{'name':'get_weather','arguments':'{"city":"北京"}'}}]
                if suffix=='A05': extra['tool_calls'].append({'id':'c2','type':'function','function':{'name':'get_time','arguments':'{"zone":"Asia/Shanghai"}'}})
        elif suffix in ('T01','T02','T04') or suffix=='04' and model.startswith('glm'):
            text=str(918273*817263)
            if suffix!='T02': extra['reasoning_content']='拆解乘法'
        elif suffix in ('F02',) and model.startswith('qwen'): text=model
        elif suffix=='F02': extra['reasoning_content']='思考'
        elif suffix=='F03' and model.startswith('qwen'): text='262144' if len(messages)==1 else 'CTX-QWEN-38'
        elif payload.get('response_format'): text='{"answer":"OK"}'
        elif suffix=='05': text='ORBIT-38'
        elif suffix=='07' and model.startswith('glm'): text='ALPHA-53'
        elif suffix=='V01':
            rng=runner.probe_rng('vision');digits=str(rng.randrange(1000,10000));right_yellow=bool(rng.getrandbits(1))
            question=prompt[0]['text']
            text=(digits if '四位数字' in question else ('黄' if right_yellow else '蓝') if '右半部分' in question
                  else '32' if '格子' in question else '黄')
        usage={'prompt_tokens':100,'completion_tokens':30,'total_tokens':130}
        if 'reasoning_content' in extra: usage['completion_tokens_details']={'reasoning_tokens':12}
        if payload.get('stream'):
            delta={'content':text,**extra}
            for call in delta.get('tool_calls',[]): call['index']=0
            events=[{'model':model,'choices':[{'index':0,'delta':delta,'finish_reason':'tool_calls' if extra.get('tool_calls') else 'stop'}]},
                    {'choices':[],'usage':usage}]
            stream='\n\n'.join('data: '+json.dumps(e) for e in events)+'\n\ndata: [DONE]\n\n'
            return httpx.Response(200,text=stream,headers={'content-type':'text/event-stream'})
        return httpx.Response(200,json={**chat(text,model,**extra),'usage':usage})

    async def run():
        nonlocal runner
        is_qwen=model.startswith('qwen');cards=QC if is_qwen else GC;card='QWEN_FULL' if is_qwen else 'GLM_53_FULL'
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        runner=QwenRunner(QwenConfig('mock','https://invalid.example'),tmp_path,client) if is_qwen else GLM53Runner(GLMConfig('mock','https://invalid.example'),tmp_path,client)
        async with runner:
            results=await runner.run(model,cards[card].priorities,faces=DEFAULT_FACES)
        await client.aclose()
        summary=qs(results,model) if is_qwen else gs(results)
        summary.update(model=model,test_kind='qwen' if is_qwen else 'glm',case=card)
        annotate_summary(summary,results,QF if is_qwen else GF,DEFAULT_FACES,with_verdict=not is_qwen)
        apply_validation_gate(summary,True)
        assert summary['admission_passed'],[(r.case_id,r.status,r.details) for r in results if r.status!='passed']
        (qw if is_qwen else gw)(tmp_path,summary)
        final=finalize_result_dir(tmp_path)
        assert final['admission_passed']
        assert final['model_validation']['full_strength']['state']=='profile_passed'
        assert '综合能力证据' in (tmp_path/'report.md').read_text()
        assert (tmp_path/'requests.jsonl').stat().st_size>0
        assert all('mock' not in json.loads(line).get('request_headers',{}).values() for line in (tmp_path/'requests.jsonl').read_text().splitlines())
    asyncio.run(run())


def test_other_model_rows_cannot_supply_missing_evidence():
    model='glm-5.3-flashx'
    rows=[{'case_id':cid,'status':'passed','passed':True,'model':'glm-5.3-flash'} for cid in required_case_ids(model)]
    result=assess_validation(rows,model)
    assert result['coverage']['passed']==0
    assert not result['admission_passed']


def test_forensics_accepts_qwen_protocol_without_network():
    from unittest.mock import patch
    from forensics.runner import config_from_env
    import os
    with patch.dict(os.environ,{'FORENSICS_API_KEY':'mock','FORENSICS_MODEL':'qwen3.8-max',
        'FORENSICS_BASE_URL':'https://dashscope.aliyuncs.com','FORENSICS_PROTOCOL':'qwen'}):
        config=config_from_env()
    assert config.base_url=='https://dashscope.aliyuncs.com/compatible-mode/v1'


def test_strict_usage_rejects_negative_boolean_and_inconsistent_counts():
    from common.probe_assertions import valid_usage
    for values in ((-1,2,1),(True,2,3),(1,2,4),(1.0,2,3)):
        assert not valid_usage({'usage':dict(zip(('prompt_tokens','completion_tokens','total_tokens'),values))})


def test_vision_numeric_substrings_are_not_answers():
    from common.probe_assertions import exact_answer
    assert exact_answer('32',('32',))
    assert not exact_answer('132',('32',))
    assert not exact_answer('不是黄色，是蓝色',('黄','yellow'))


@pytest.mark.parametrize('family,model,card', [('glm','glm-5.3-flashx','GLM_53_FULL'),('qwen','qwen3.7-max','QWEN_FULL')])
def test_cli_all_remote_blocked_cannot_pass_and_artifacts_are_unique(family,model,card,tmp_path,monkeypatch,capsys):
    import httpx
    import importlib
    entry=importlib.import_module('qwen.runner' if family=='qwen' else 'glm53.runner')
    cls=QwenRunner if family=='qwen' else GLM53Runner
    def factory(config,audit_dir):
        instance=cls(config,audit_dir)
        instance.client=httpx.AsyncClient(transport=httpx.MockTransport(lambda req:httpx.Response(403,json={'error':{'message':'denied'}})))
        return instance
    monkeypatch.setattr(entry,'QwenRunner' if family=='qwen' else 'GLM53Runner',factory)
    for key,value in [('API_KEY','offline'),('BASE_URL','https://invalid.example/v1'),('MODEL',model)]:
        monkeypatch.setenv(f'{family.upper()}_{key}',value)
    monkeypatch.delenv('QWEN_CASE_IDS',raising=False)
    monkeypatch.delenv('VOLC_FACES',raising=False)
    for _ in range(2):
        assert entry.main([card,'--output-root',str(tmp_path),'--seed','123'])==2
    summaries=list(tmp_path.glob('*/summary.json'))
    assert len(summaries)==2
    for path in summaries:
        summary=json.loads(path.read_text())
        assert not summary['admission_passed']
        assert summary['model_validation']['seed']==123
        assert summary['model_validation']['coverage']['passed']==0
        assert summary['model_validation']['full_strength']['state']=='insufficient_evidence'
    capsys.readouterr()


def test_cli_partial_capability_selection_is_not_whole_card_pass(tmp_path,monkeypatch,capsys):
    import qwen.runner as entry
    from qwen.core import CaseResult
    class Partial:
        def __init__(self,config,audit_dir):
            (audit_dir/'requests.jsonl').write_text('{"fixture":"offline"}\n')
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def run(self,*args):
            return [CaseResult('QWEN-Q01','math','qwen3.8-max',True,'passed')]
    monkeypatch.setattr(entry,'QwenRunner',Partial)
    for key,value in [('API_KEY','offline'),('BASE_URL','https://invalid.example/v1'),('MODEL','qwen3.8-max'),('CASE_IDS','QWEN-Q01')]:
        monkeypatch.setenv(f'QWEN_{key}',value)
    assert entry.main(['QWEN_CAPABILITY','--output-root',str(tmp_path)])==2
    summary=json.loads(next(tmp_path.glob('*/summary.json')).read_text())
    assert not summary['admission_passed']
    capsys.readouterr()


def test_impossible_cache_counters_do_not_pass():
    async def run():
        runner=runner_for('glm-5.3')
        body={**chat('OK'),'usage':{'prompt_tokens':100,'prompt_tokens_details':{'cached_tokens':200}}}
        runner.request=AsyncMock(return_value=(200,body,0,{}))
        result=await runner.case_cache('glm-5.3')
        assert result[3]=='failed'
    asyncio.run(run())


def test_empty_glm_response_and_strict_schema_cannot_pass():
    from glm53.core import _schema_valid
    assert not _schema_valid('{"answer":"OK","extra":1}')
    async def run():
        runner=runner_for('glm-5.3')
        runner.request=AsyncMock(return_value=(200,chat(None),0,{}))
        result=await runner.case_basic('glm-5.3')
        assert result[0] is False and result[3]=='failed'
    asyncio.run(run())


def test_cyclic_model_program_result_is_rejected_and_remains_json_serializable():
    async def run():
        runner=runner_for('glm-5.3')
        runner.request=AsyncMock(return_value=(200,chat('def solve(nums):\n result=[]\n result.append(result)\n return result'),0,{}))
        result=await runner.case_quality_code('glm-5.3')
        assert not result[0]
        json.dumps(result)
    asyncio.run(run())


def test_duplicate_case_ids_cannot_overwrite_failure_with_pass():
    model='glm-5.3'
    rows=[{'case_id':cid,'model':model,'status':'passed','passed':True} for cid in required_case_ids(model)]
    first={**rows[0],'status':'failed','passed':False}
    result=assess_validation([first,*rows],model)
    assert not result['admission_passed']
    assert result['coverage']['duplicate_case_ids']==[first['case_id']]
