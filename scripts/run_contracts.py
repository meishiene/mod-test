#!/usr/bin/env python3
"""运行 KVV/DeepSeek/GLM 共用契约探针；模型差异由 common.model_contracts.PROFILES 控制。"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time
from pathlib import Path
import httpx
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.model_contracts import PROFILES, load_profile_data, model_matches_family, validate_identity, validate_usage, text, message
from common.suite_runtime import finalize
from common.report_style import case_table_report
from common.test_framework import HttpExecutor
from common.probe_assertions import valid_tools, strict_json, integer_answer, identity_match
from common.audit import AuditLog
from volcperf.protocols import api_root

async def main():
    ap=argparse.ArgumentParser(description="跨模型公共契约探针")
    ap.add_argument("family", choices=PROFILES); ap.add_argument("--model"); ap.add_argument("--base-url"); ap.add_argument("--api-key"); ap.add_argument("--output-root",type=Path,default=Path("results"))
    a=ap.parse_args(); env={"kvv":("VOLC_BASE_URL","VOLC_API_KEY","VOLC_MODEL"),"deepseek":("DS_BASE_URL","DS_API_KEY","DS_MODEL"),"glm":("GLM_BASE_URL","GLM_API_KEY","GLM_MODEL")}[a.family]
    base=a.base_url or os.getenv(env[0]); key=a.api_key or os.getenv(env[1]); model=a.model or os.getenv(env[2])
    if not base or not key or not model: raise SystemExit(f"缺少 {env[0]}/{env[1]}/{env[2]}")
    _gok,_greason=model_matches_family(a.family,model)
    if not _gok: raise SystemExit(_greason)  # 模型门禁：防误配模型产生假通过
    base=api_root(os.getenv("VOLC_PROTOCOL","volcengine"),base)
    profile=PROFILES[a.family]; data=load_profile_data(a.family); results=[]
    out=a.output_root/f"{os.getenv('CONTRACT_CASE') or f'CONTRACT-{a.family.upper()}'}-{time.strftime('%Y%m%d-%H%M%S')}"; out.mkdir(parents=True,exist_ok=True)
    audit_path=out/'requests.jsonl'; request_seq=0
    audit_log=AuditLog(audit_path)
    def audit(x):
      audit_log.write(x)
    expected_tokens=tuple(data.get("model_tokens", profile.expected_model_tokens))
    marker=data.get("smoke_marker", profile.answer_marker)
    thinking_on=data.get("thinking_on", profile.thinking_on or {})
    thinking_off=data.get("thinking_off", profile.thinking_off or {})
    tool_name=data.get("tool_name", profile.tool_name)
    supports_reasoning=bool(data.get("supports_reasoning", profile.supports_reasoning))
    async with httpx.AsyncClient(timeout=180,headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"}) as c:
      executor=HttpExecutor(timeout=180, client=c, trust_env=False)
      await executor.__aenter__()
      async def req(payload, stream=False):
        nonlocal request_seq
        request_seq += 1; rid=f"CONTRACT-{request_seq:03d}-{os.urandom(4).hex()}"
        record=await executor.execute(case_id=f"CONTRACT-{request_seq:02d}", url=base.rstrip('/')+'/chat/completions', payload=payload, headers={"Authorization":f"Bearer {key}","Content-Type":"application/json","X-Request-ID":rid}, stream=stream)
        audit(record)
        if not stream:
          return record.http_status or 0, record.response_body if isinstance(record.response_body,dict) else None, {}
        return record.http_status or 0, None, {"content":record.content,"reasoning_content":record.reasoning,"done":record.done_received,"invalid_events":record.invalid_sse_events,"finish_reasons":record.finish_reasons,"raw_sse_lines":record.raw_sse_lines,"raw_sse_events":record.raw_sse_events}
      def payload(messages, **kw): return {"model":model,"messages":messages,"max_tokens":2048,**kw}
      t0=time.time()
      st,b,o=await req(payload([{"role":"user","content":f"只回复 {marker}"}],**(thinking_off or {}))); ok,obs=validate_identity(st,b,expected_tokens); ok=ok and (a.family!="glm" or identity_match(model,(b or {}).get("model"))); positive_control=ok; results.append({"id":"CONTRACT-01","name":"身份与路由","status":"passed" if ok else "failed","passed":ok,"duration_seconds":time.time()-t0,"details":f"HTTP {st}","observations":obs})
      t0=time.time()
      st,b,o=await req(payload([{"role":"user","content":"只回复 OK"}],**(thinking_off or {}),temperature=2.1)); error_text=json.dumps((b or {}).get("error") or {},ensure_ascii=False).lower(); ok=positive_control and st in (400,422) and "temperature" in error_text; results.append({"id":"CONTRACT-02","name":"参数错误契约","status":"passed" if ok else "failed","passed":ok,"duration_seconds":time.time()-t0,"details":f"非法 temperature HTTP {st}","observations":{}})
      t0=time.time()
      st,b,o=await req(payload([{"role":"user","content":"计算 37*83，并简要说明。"}],**(thinking_on or {}),stream=True),True); ok=st==200 and o.get('done') and o.get('invalid_events')==0 and bool(o.get('content')); results.append({"id":"CONTRACT-03","name":"流式协议完整性","status":"passed" if ok else "failed","passed":ok,"duration_seconds":time.time()-t0,"details":f"HTTP {st}，DONE={o.get('done')}","observations":o})
      t0=time.time()
      tool={"type":"function","function":{"name":tool_name,"description":"查询订单","parameters":{"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"],"additionalProperties":False}}}
      messages=[{"role":"user","content":"查询订单 ORD-000731，必须调用工具。最终只输出工具返回的 JSON，不要添加字段。"}]
      st,b,o=await req(payload(messages,tools=[tool],tool_choice="required",**(thinking_off or {})))
      calls=message(b).get('tool_calls') or []
      ok=st==200 and valid_tools(calls,[tool],{tool_name:{"order_id":"ORD-000731"}})
      observations={"tool_calls":calls,"first_status":st}
      if ok:
        tool_result={"order_id":"ORD-000731","status":"paid","nonce":os.urandom(8).hex()}
        messages += [{**message(b),"role":"assistant"},{"role":"tool","tool_call_id":calls[0]["id"],"content":json.dumps(tool_result)}]
        st2,b2,o2=await req(payload(messages,tools=[tool],**(thinking_off or {})))
        ok=st2==200 and strict_json(text(b2),{"const":tool_result})
        observations.update({"replay_status":st2,"expected":tool_result,"answer":text(b2)})
      results.append({"id":"CONTRACT-04","name":"工具调用多轮闭环","status":"passed" if ok else "failed","passed":ok,"duration_seconds":time.time()-t0,"details":f"首轮 HTTP {st}，结果利用={ok}","observations":observations})
      if a.family in {"deepseek", "kvv", "glm"}:
        t0=time.time()
        efforts=data.get("reasoning_efforts", list(profile.reasoning_efforts)); statuses={}; chars={}; answers={}
        for effort in efforts:
          stx,bx,ox=await req(payload([{"role":"user","content":"计算 918273*817263，思考后最终正文只输出整数。"}],**{**thinking_on,"reasoning_effort":effort})); raw=message(bx).get('reasoning_content') or message(bx).get('reasoning') or ''; statuses[effort]=stx; chars[effort]=len(raw); answers[effort]=text(bx)
        ok=all(v==200 for v in statuses.values()) and all(v>0 for v in chars.values()) and all(integer_answer(v,918273*817263) for v in answers.values()); results.append({"id":"CONTRACT-05","name":"Thinking 与 reasoning_effort","status":"passed" if ok else "failed","passed":ok,"duration_seconds":time.time()-t0,"details":f"档位={efforts}，HTTP={statuses}，reasoning 字符={chars}","observations":{"statuses":statuses,"reasoning_chars":chars,"answers":answers}})

    audit_log.close()
    summary={"family":a.family,"model":model,"total":len(results),"passed":sum(x['status']=='passed' for x in results),"failed":sum(x['status']=='failed' for x in results),"warning":sum(x['status']=='warning' for x in results),"admission_passed":not any(x['status']=='failed' for x in results),"results":results}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    case_title=os.getenv('CONTRACT_CASE') or f"CONTRACT-{a.family.upper()}"
    rows=[[f"{x['id']} {x['name']}",model,x['status'],f"{x.get('duration_seconds',0):.2f}",x['details']] for x in results]
    (out/'report.md').write_text(case_table_report("公共契约探针测试报告",
        total=summary['total'],passed=summary['passed'],failed=summary['failed'],warning=summary['warning'],blocked=0,
        extra_stats=[f"模型：`{model}`；Family：`{a.family}`"],
        rows=rows,status_column=2),encoding='utf-8')
    finalize(out);print(json.dumps(summary,ensure_ascii=False,indent=2));print(f"结果：{out.resolve()}");return 0 if not summary['failed'] else 2
if __name__=='__main__': raise SystemExit(asyncio.run(main()))
