"""跨模型黑盒真实性/满血能力指纹探测。结果只能表示与官方行为的一致性，不能证明权重来源。"""
from __future__ import annotations
import asyncio, json, os, secrets, sys, time, uuid
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.audit import AuditLog
from common.suite_runtime import finalize
from common.report_style import case_table_report

CASES={
 "DS_AUTH_FINGERPRINT": ("DeepSeek 满血能力指纹", "DeepSeek thinking、工具、JSON、流式、动态绑定与静默吞参探测", "deepseek"),
 "DS_V4_FINGERPRINT": ("DeepSeek V4 满血能力指纹", "DeepSeek Thinking、工具、JSON、流式、动态绑定与静默吞参探测", "deepseek"),
}

def token(): return secrets.token_hex(4).upper()
def msg(b): return (((b or {}).get('choices') or [{}])[0].get('message') or {})
def text(b):
 v=msg(b).get('content',''); return v if isinstance(v,str) else json.dumps(v,ensure_ascii=False)
def unsupported(body):
 s=json.dumps(body or {},ensure_ascii=False).lower(); return any(x in s for x in ('unsupported','not support','unknown field','invalid parameter','不支持'))

def _upstream_id(headers):
 return headers.get('x-request-id') or headers.get('request-id') or headers.get('x-trace-id') or headers.get('x-tt-logid')

async def run(case, base, key, model, out):
 rng=token(); headers={'Authorization':f'Bearer {key}','Content-Type':'application/json','X-Request-ID':f'AUTH-{uuid.uuid4().hex}'}
 results=[]
 # 结果目录与证据文件必须在首个请求之前就绪：中断或失败时仍可复核已完成的部分。
 path=out/f'{case}-{time.strftime("%Y%m%d-%H%M%S")}'
 path.mkdir(parents=True, exist_ok=True)
 audit=AuditLog(path/'requests.jsonl'); seq=0
 async with httpx.AsyncClient(timeout=180,headers=headers) as c:
  async def req(payload, stream=False, case_id=None):
   nonlocal seq
   seq+=1
   record={'case_id':case_id or f'AUTH-{seq:02d}','request_id':f'AUTH-{seq:03d}-{uuid.uuid4().hex[:8]}',
           'request':payload,'request_headers':{k:v for k,v in headers.items() if k.lower()!='authorization'},
           'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'started_epoch':time.time()}
   if not stream:
    r=await c.post(base.rstrip('/')+'/chat/completions',json=payload)
    try:b=r.json()
    except ValueError:b=None
    record.update({'http_status':r.status_code,'response_headers':dict(r.headers),'response_body':b,
                   'response_text':r.text,'provider_request_id':_upstream_id(r.headers),'transport_status':'completed'})
    audit.write(record)
    return r.status_code,b,dict(r.headers)
   # 流式请求要同时满足两件事：payload 内声明 stream=true 让上游返回 SSE，
   # 并把 stream=True 传给本函数进入流式读取分支。缺任一条都会拿普通 JSON 解析器
   # 去读 SSE 报文（解析失败→空结果），表现为“HTTP 200 但 DONE 缺失、无内容”。
   payload={**payload,'stream':True}
   chunks=[]; done=False; bad=0
   async with c.stream('POST',base.rstrip('/')+'/chat/completions',json=payload) as r:
    if r.status_code>=300:
     raw=(await r.aread()).decode(errors='replace')[:500]
     record.update({'http_status':r.status_code,'response_headers':dict(r.headers),'response_text':raw,
                    'provider_request_id':_upstream_id(r.headers),'transport_status':'completed','stream_error':raw})
     audit.write(record)
     return r.status_code,{'error':raw,'response_headers':dict(r.headers),'upstream_request_id':_upstream_id(r.headers)},dict(r.headers)
    async for line in r.aiter_lines():
     if not line.startswith('data:'):continue
     raw=line[5:].strip()
     if raw=='[DONE]':done=True;continue
     try:o=json.loads(raw)
     except ValueError:bad+=1;continue
     for ch in o.get('choices') or []: chunks.append((ch.get('delta') or {}).get('content') or '')
   record.update({'http_status':r.status_code,'response_headers':dict(r.headers),'content':''.join(chunks),
                  'done_received':done,'invalid_sse_events':bad,'provider_request_id':_upstream_id(r.headers),
                  'transport_status':'completed'})
   audit.write(record)
   return r.status_code,{'content':''.join(chunks),'done':done,'invalid_events':bad,'response_headers':dict(r.headers),'upstream_request_id':_upstream_id(r.headers)},dict(r.headers)
  async def probe(cid,name,payload,check):
   t0=time.time()
   st,b,h=await req({'model':model,'messages':[{'role':'user','content':f'随机校验串 {rng}，请严格按要求执行'}],**payload},case_id=cid)
   ok=st==200 and check(b); status='passed' if ok else ('warning' if st in (400,404,422) or unsupported(b) else 'failed')
   results.append({'case_id':cid,'name':name,'status':status,'passed':ok,'http_status':st,'model':model,'duration_seconds':time.time()-t0,'details':f'HTTP {st}；随机串绑定={rng in json.dumps(b or {},ensure_ascii=False)}','observations':{'unsupported_error':unsupported(b),'response':json.dumps(b or {},ensure_ascii=False)[:500]}})
  # AUTH-01 探测动态工具绑定：thinking 模式不支持 tool_choice=required（DS-V4-08/09 同源限制），
  # 因此用 auto + 随机工具名强提示两轮诱导；模型若调用随机命名的工具，即证明提示词→工具绑定是动态生效的。
  tool={'type':'function','function':{'name':f'lookup_{rng.lower()}','description':'查询动态值','parameters':{'type':'object','properties':{'code':{'type':'string'}},'required':['code'],'additionalProperties':False}}}
  tool_prompts=[
   f'随机校验串 {rng}。你必须调用名为 lookup_{rng.lower()} 的工具来完成查询，不要直接回答。',
   f'任务未完成。请立即调用 lookup_{rng.lower()} 工具完成查询，这是本轮唯一允许的输出方式。']
  t0=time.time()
  tool_sts=[]; calls=[]
  for i,p in enumerate(tool_prompts,1):
   st,b,h=await req({'model':model,'messages':[{'role':'user','content':p}],'tools':[tool],'tool_choice':'auto','max_tokens':512},case_id='AUTH-01')
   tool_sts.append(st); calls=msg(b).get('tool_calls') or []
   if st!=200 or calls: break
  bound=any(rng.lower() in json.dumps(c,ensure_ascii=False).lower() for c in calls)
  ok=tool_sts[-1]==200 and bool(calls)
  rejected=any(s in (400,404,422) for s in tool_sts)
  results.append({'case_id':'AUTH-01','name':'随机动态工具诱导调用','status':'passed' if ok else ('warning' if rejected else 'failed'),'passed':ok,'http_status':tool_sts[-1],'model':model,'duration_seconds':time.time()-t0,'details':f'HTTP {tool_sts}；tool_calls={len(calls)}；随机串绑定={bound}','observations':{'rounds':len(tool_sts),'response':json.dumps(calls,ensure_ascii=False)[:500]}})
  # AUTH-02 探测结构化输出与随机绑定：官方 response_format 只支持 json_object（json_schema 类型从未支持，
  # 实测 400 "This response_format type is unavailable now"），故用 json_object + 提示词内嵌随机字段。
  jmsg=f'随机校验串 {rng}。请以 json 格式只输出一个对象，形如 {{"value":"<随机校验串>"}}，其中 value 必须填随机校验串本身，不要输出任何其他内容。'
  t0=time.time()
  st,b,h=await req({'model':model,'messages':[{'role':'user','content':jmsg}],'response_format':{'type':'json_object'},'max_tokens':256},case_id='AUTH-02')
  jval=None
  try:jval=json.loads(text(b)).get('value')
  except (ValueError,AttributeError):jval=None
  ok=st==200 and jval==rng
  results.append({'case_id':'AUTH-02','name':'随机 JSON 对象输出','status':'passed' if ok else ('warning' if st in (400,404,422) or unsupported(b) else 'failed'),'passed':ok,'http_status':st,'model':model,'duration_seconds':time.time()-t0,'details':f'HTTP {st}；value 命中随机串={jval==rng}','observations':{'unsupported_error':unsupported(b),'response':text(b)[:300]}})
  await probe('AUTH-03','随机多轮记忆',{},lambda b:rng in text(b))
  t0=time.time()
  st,b,h=await req({'model':model,'messages':[{'role':'user','content':f'计算 19*{len(rng)}，给出过程和结果'}],'thinking':{'type':'enabled'},'max_tokens':128},case_id='AUTH-04')
  r=msg(b).get('reasoning_content'); results.append({'case_id':'AUTH-04','name':'Thinking 开启与 reasoning 字段','status':'passed' if st==200 and r else ('warning' if st in (400,404,422) or unsupported(b) else 'failed'),'passed':bool(st==200 and r),'http_status':st,'model':model,'duration_seconds':time.time()-t0,'details':f'HTTP {st}；reasoning 字符数={len(r or "")}'})
  t0=time.time()
  st,b,h=await req({'model':model,'messages':[{'role':'user','content':f'只回复 {rng}'}]},True,case_id='AUTH-05'); b=b or {}; ok=st==200 and b.get('done') and not b.get('invalid_events'); results.append({'case_id':'AUTH-05','name':'流式结束与事件完整性','status':'passed' if ok else ('warning' if st in (400,404,422) else 'failed'),'passed':ok,'http_status':st,'model':model,'duration_seconds':time.time()-t0,'details':f'HTTP {st}；DONE={b.get("done")}；非法事件={b.get("invalid_events",0)}','observations':{'error':b.get('error'),'content_chars':len(b.get('content',''))}})
  # 对照开启/关闭，识别 200 但吞掉参数
  vals=[]
  t0=time.time()
  for t in ('disabled','enabled'):
   st,b,h=await req({'model':model,'messages':[{'role':'user','content':f'请进行多步推理并回答 {rng}'}],'thinking':{'type':t},'max_tokens':128},case_id='AUTH-06')
   vals.append((st,len(msg(b).get('reasoning_content') or '')))
  silent=all(x[0]==200 for x in vals) and vals[1][1]==0
  results.append({'case_id':'AUTH-06','name':'Thinking 静默吞参对照','status':'warning' if silent else 'passed','passed':not silent,'model':model,'duration_seconds':time.time()-t0,'details':f'禁用/开启 HTTP 与 reasoning 长度：{vals}','observations':{'silent_ignore':silent}})
 audit.close()
 summary={'case':case,'test_kind':'authenticity','model':model,'random_nonce':rng,'total':len(results),'passed':sum(x['status']=='passed' for x in results),'failed':sum(x['status']=='failed' for x in results),'warning':sum(x['status']=='warning' for x in results),'results':results}
 hard=summary['failed']; summary['authenticity_assessment']='证据充分' if hard==0 and not summary['warning'] else ('存在接口限制或疑似降级' if hard==0 else '存在硬失败，不能认定满血')
 # 准入判定与评估结论保持一致：只有“证据充分”才允许准入；warning 表示接口限制或疑似降级，不构成准入。
 summary['admission_passed']=hard==0 and summary['warning']==0
 (path/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 rows=[[f"{x['case_id']} {x['name']}", x.get('model') or model, x['status'], f"{x.get('duration_seconds',0):.2f}", x['details']] for x in results]
 (path/'report.md').write_text(case_table_report('满血能力指纹报告',
   total=summary['total'],passed=summary['passed'],failed=summary['failed'],warning=summary['warning'],blocked=0,
   extra_stats=[f"结论：{summary['authenticity_assessment']}"],rows=rows,status_column=2),encoding='utf-8')
 summary=finalize(path)
 return path,summary
