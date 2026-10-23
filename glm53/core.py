from __future__ import annotations
import asyncio, json, os, time, uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import httpx
from common.test_framework import HttpExecutor, ResultLog
from common.report_style import case_table_report
from common.model_probes import ModelProbeMixin
from common.model_validation import DEFAULT_SEED, PROBE_DEFINITIONS, assess_validation, validation_report_stats
from common.probe_assertions import valid_usage, valid_tools, stream_tools, strict_json, identity_match, integer_answer

@dataclass(frozen=True)
class GLMConfig:
    api_key: str
    base_url: str
    timeout: float = 180.0
    verify_ssl: bool = True
    trust_env: bool = False
    seed: int = DEFAULT_SEED

@dataclass
class CaseResult:
    case_id: str; name: str; model: str; passed: bool; status: str = "passed"
    details: str = ""; duration_seconds: float = 0.0; observations: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class CaseSpec:
    case_id: str; name: str; priority: str; handler: str

CASE_SPECS = (
    CaseSpec("GLM-53-01", "基础对话与模型标识", "P0", "basic"),
    CaseSpec("GLM-53-02", "非流式 Usage 完整性", "P0", "usage"),
    CaseSpec("GLM-53-03", "流式输出与结束事件", "P0", "stream"),
    CaseSpec("GLM-53-04", "深度思考与 reasoning_effort", "P0", "thinking"),
    CaseSpec("GLM-53-06", "结构化 JSON 输出", "P0", "json_output"),
    CaseSpec("GLM-53-07", "多轮上下文保持", "P0", "multi_turn"),
    CaseSpec("GLM-53-08", "长上下文稳定响应", "P0", "long_context"),
    CaseSpec("GLM-53-09", "工具调用基础链路", "P1", "tool_call"),
    CaseSpec("GLM-53-10", "tool_choice=required", "P1", "tool_required"),
    CaseSpec("GLM-53-11", "工具结果多轮回传", "P1", "tool_replay"),
    CaseSpec("GLM-53-12", "工具流式输出", "P1", "tool_stream"),
    CaseSpec("GLM-53-C01", "Context Cache 命中专项", "PCACHE", "cache"),
    CaseSpec("GLM-53-C02", "同身份多轮 Session 连续性与缓存命中", "PCACHE", "session_continuity_cache"),
    CaseSpec("GLM-53-F01", "模型身份一致性", "P2", "fingerprint_identity"),
    CaseSpec("GLM-53-F02", "Thinking 字段与错误指纹", "P2", "fingerprint_thinking"),
    CaseSpec("GLM-53-F03", "Tool Choice 强制调用指纹", "P2", "fingerprint_tool_choice"),
    CaseSpec("GLM-53-F04", "JSON Schema 能力指纹", "P2", "fingerprint_schema"),
    CaseSpec("GLM-53-F05", "流式协议完整性指纹", "P2", "fingerprint_stream"),
    CaseSpec("GLM-53-F06", "参数静默忽略检测", "P2", "fingerprint_silent_ignore"),
    CaseSpec("GLM-53-F07", "上下文长度边界指纹", "P2", "fingerprint_context"),
    CaseSpec("GLM-53-F08", "满血证据汇总", "P2", "fingerprint_summary"),
)

CASE_SPECS += tuple(CaseSpec(f"GLM-53-{suffix}", name, priority, handler)
                    for suffix, name, priority, handler, _ in PROBE_DEFINITIONS)

def msg(body):
    return (((body or {}).get("choices") or [{}])[0].get("message") or {})
def content(body):
    value = msg(body).get("content", "")
    return value if isinstance(value, str) else ""
def usage_ok(body):
    return valid_usage(body)

def probe_status(http_status, ok):
    return "passed" if http_status==200 and ok else "failed" if http_status==200 else "blocked"

def cache_usage(body):
    u=(body or {}).get("usage") or {}; details=u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    total=int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
    hit=int(u.get("prompt_cache_hit_tokens") or u.get("cache_read_input_tokens") or details.get("cached_tokens") or details.get("cache_read_input_tokens") or 0)
    created=int(u.get("cache_creation_input_tokens") or details.get("cache_creation_input_tokens") or 0)
    return total,hit,created

def _schema_valid(raw):
    return strict_json(raw, {"type":"object", "properties":{"answer":{"type":"string"}},
                             "required":["answer"], "additionalProperties":False})

class GLM53Runner(ModelProbeMixin):
    def __init__(self, config: GLMConfig, audit_dir: Path | None = None, client: httpx.AsyncClient | None = None):
        self.config=config; self.client=client; self._external_client=client; self.executor=None; self.log=None; self.audit_dir=audit_dir; self.audit_path=None; self.request_seq=0; self.current_case_id=None
        if audit_dir:
            audit_dir.mkdir(parents=True, exist_ok=True)
            self.audit_path=audit_dir / "requests.jsonl"
    async def __aenter__(self):
        if self.client is None: self.client=httpx.AsyncClient(timeout=self.config.timeout, verify=self.config.verify_ssl, trust_env=self.config.trust_env, headers={"Authorization":f"Bearer {self.config.api_key}","Content-Type":"application/json"})
        self.executor=HttpExecutor(timeout=self.config.timeout, verify_ssl=self.config.verify_ssl, trust_env=self.config.trust_env, client=self.client)
        await self.executor.__aenter__()
        if self.audit_dir: self.log=ResultLog(self.audit_dir / "requests.jsonl")
        return self
    async def __aexit__(self,*args):
        if self.log: self.log.close()
        if self.executor: await self.executor.__aexit__(*args)
        if self.client and self._external_client is None: await self.client.aclose()
    def _audit(self, record: dict[str, Any]):
        if not self.audit_path: return
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    async def request(self, model, messages, *, stream=False, **kwargs):
        if self.executor is None:
            await self.__aenter__()
        payload={"model":model,"messages":messages,"max_tokens":kwargs.pop("max_tokens",2048),"stream":stream,**kwargs}
        request_id=f"GLM53-{uuid.uuid4().hex[:12]}"
        record=await self.executor.execute(case_id=self.current_case_id or "GLM53-request", url=self.config.base_url.rstrip("/")+"/chat/completions", payload=payload, headers={"Authorization":f"Bearer {self.config.api_key}","Content-Type":"application/json","X-Request-ID":request_id}, stream=stream)
        if self.log: await self.log.write(record)
        if not stream:
            return record.http_status or 0, record.response_body if isinstance(record.response_body,dict) else None, record.timing.get("e2e_ms",0)/1000, {}
        parsed={"content":record.content,"reasoning_content":record.reasoning,"done":record.done_received,"invalid_events":record.invalid_sse_events,"finish_reasons":record.finish_reasons,"raw_sse_lines":record.raw_sse_lines,"raw_sse_events":record.raw_sse_events,"usage":record.usage or {},"tool_calls":stream_tools(record.parsed_sse_events)}
        return record.http_status or 0, None, record.timing.get("e2e_ms",0)/1000, parsed

    def tools(self): return [{"type":"function","function":{"name":"get_weather","description":"查询天气","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":False}}}]
    async def run(self, model, priorities, faces=None):
        # 面过滤（默认 model+verify）：通道契约面（tool_choice 严格语义、参数是否被忽略、
        # 错误信封）在中转链路上直接 failed，默认不跑。
        from common.faces import filter_specs, selected_faces
        from .faces import CASE_FACES
        active = selected_faces(faces)
        out=[]
        for s in filter_specs([x for x in CASE_SPECS if x.priority in priorities], CASE_FACES, active):
            self.current_case_id=s.case_id
            start=time.perf_counter()
            try:
                passed,detail,obs,status=await getattr(self,"case_"+s.handler)(model)
            except Exception as e:
                passed,detail,obs,status=False,f"{type(e).__name__}: {e}",{},"error"
            out.append(CaseResult(s.case_id,s.name,model,passed,status,detail,time.perf_counter()-start,obs))
        self.current_case_id=None
        return out
    async def _basic(self,model,**kw):
        st,b,_,_=await self.request(model,[{"role":"user","content":"只回复 OK"}],**kw)
        ok=st==200 and content(b).strip()=="OK"
        return ok,f"HTTP {st}",{"model":(b or {}).get("model")},probe_status(st,ok)
    async def case_basic(self,m): return await self._basic(m)
    async def case_usage(self,m):
        st,b,_,_=await self.request(m,[{"role":"user","content":"只回复 OK"}]); ok=st==200 and usage_ok(b); return ok,f"HTTP {st}，Usage {'完整' if ok else '不完整'}",{"usage":(b or {}).get("usage")},probe_status(st,ok)
    async def case_stream(self,m):
        st,_,_,o=await self.request(m,[{"role":"user","content":"只回复 OK"}],stream=True); ok=st==200 and o.get("done") and o.get("content", "").strip()=="OK" and not o.get("invalid_events") and "stop" in o.get("finish_reasons",[]); return bool(ok),f"HTTP {st}，收到结束事件={o.get('done')}",o,probe_status(st,ok)
    async def case_thinking(self,m):
        statuses={}; reasoning_chars={}; reasoning_tokens={}; answers={}
        for effort in ("low", "medium", "high"):
            st,b,_,_=await self.request(m,[{"role":"user","content":"计算 918273*817263，思考后最终正文只输出整数。"}],thinking={"type":"enabled"},reasoning_effort=effort,max_tokens=2048)
            mm=msg(b); raw=mm.get("reasoning_content") or mm.get("reasoning") or ""; usage=(b or {}).get("usage") or {}; details=usage.get("completion_tokens_details") or {}
            statuses[effort]=st; reasoning_chars[effort]=len(raw); reasoning_tokens[effort]=details.get("reasoning_tokens"); answers[effort]=content(b)
        ok=all(st==200 for st in statuses.values()) and all(n>0 for n in reasoning_chars.values()) and all(integer_answer(v,918273*817263) for v in answers.values())
        return ok,f"reasoning_effort {statuses}，reasoning 字符={reasoning_chars}",{"statuses":statuses,"reasoning_chars":reasoning_chars,"reasoning_tokens":reasoning_tokens,"efforts":["low","medium","high"],"answers":answers},("passed" if ok else "failed" if any(statuses[e]==200 and (reasoning_chars[e]==0 or not integer_answer(answers[e],918273*817263)) for e in statuses) else "blocked")
    async def case_json_output(self,m):
        st,b,_,_=await self.request(m,[{"role":"user","content":"返回 JSON，字段 answer，值为 OK"}],thinking={"type":"enabled"},response_format={"type":"json_object"}); raw=content(b)
        try: obj=json.loads(raw); ok=st==200 and obj.get("answer")=="OK"
        except (ValueError, AttributeError): ok=False
        return ok,f"HTTP {st}，JSON 可解析={ok}",{"content":raw[:500]},probe_status(st,ok)
    async def case_multi_turn(self,m):
        st,b,_,_=await self.request(m,[{"role":"user","content":"记住代号是 ALPHA-53"},{"role":"assistant","content":"好的"},{"role":"user","content":"代号是什么？只回复代号"}]); ok=st==200 and "ALPHA-53" in content(b); return ok,f"HTTP {st}",{},probe_status(st,ok)
    async def case_long_context(self,m): return await self.case_quality_context(m)
    async def case_tool_call(self,m):
        st,b,_,_=await self.request(m,[{"role":"user","content":"请查询北京天气"}],tools=self.tools(),thinking={"type":"enabled"}); calls=msg(b).get("tool_calls") or []; ok=st==200 and valid_tools(calls,self.tools(),{"get_weather":{"city":"北京"}}); return ok,f"HTTP {st}，工具调用数={len(calls)}",{"tool_calls":calls[:1]},probe_status(st,ok)
    async def case_tool_required(self,m):
        st,b,_,_=await self.request(m,[{"role":"user","content":"请查询天气"}],tools=self.tools(),tool_choice="required",thinking={"type":"enabled"}); ok=st==200 and valid_tools(msg(b).get("tool_calls"),self.tools()); return ok,f"HTTP {st}",{},probe_status(st,ok)
    async def case_tool_replay(self,m):
        tool=self.tools()[0]; st,b,_,_=await self.request(m,[{"role":"user","content":"查询北京天气"}],tools=[tool],tool_choice="required",thinking={"type":"enabled"}); calls=msg(b).get("tool_calls") or []
        if st!=200 or not valid_tools(calls,[tool],{"get_weather":{"city":"北京"}}): return False,f"首轮 HTTP {st} 未产生工具调用",{},probe_status(st,False)
        call=calls[0]; msgs=[{"role":"user","content":"查询北京天气"},{**msg(b),"role":"assistant","tool_calls":calls},{"role":"tool","tool_call_id":call.get("id","call_1"),"content":"晴，25度"}]; st2,b2,_,_=await self.request(m,msgs,tools=[tool],thinking={"type":"enabled"}); ok=st2==200 and "晴" in content(b2) and "25" in content(b2); return ok,f"首轮/回传 HTTP {st}/{st2}",{},probe_status(st2,ok)
    async def case_tool_stream(self,m):
        st,_,_,o=await self.request(m,[{"role":"user","content":"查询北京天气，调用工具"}],tools=self.tools(),stream=True); ok=st==200 and o.get("done") and not o.get("invalid_events") and "tool_calls" in o.get("finish_reasons",[]) and valid_tools(o.get("tool_calls"),self.tools()); return bool(ok),f"HTTP {st}，结束事件={o.get('done')}",o,probe_status(st,ok)


    async def case_session_continuity_cache(self,m):
        """同一 user/session 身份追加完整历史，分别验证连续性和缓存 usage。"""
        session_id = f"glm-session-{uuid.uuid4().hex[:10]}"
        prefix = "SESSION-CONTINUITY-GLM53-" + ("固定会话背景。" * 700)
        turns = [
            [{"role":"system","content":prefix},{"role":"user","content":"记住会话暗号：ORBIT-731。只回复已记住。"}],
            [{"role":"system","content":prefix},{"role":"user","content":"记住会话暗号：ORBIT-731。只回复已记住。"},{"role":"assistant","content":"已记住。"},{"role":"user","content":"会话暗号是什么？只回复暗号。"}],
            [{"role":"system","content":prefix},{"role":"user","content":"记住会话暗号：ORBIT-731。只回复已记住。"},{"role":"assistant","content":"已记住。"},{"role":"user","content":"会话暗号是什么？只回复暗号。"},{"role":"assistant","content":"ORBIT-731"},{"role":"user","content":"再次确认暗号，只回复暗号。"}],
        ]
        rows=[]
        for i,messages in enumerate(turns):
            st,b,elapsed,_=await self.request(m,messages,user=session_id,thinking={"type":"enabled"},max_tokens=2048)
            total,hit,created=cache_usage(b); text=content(b)
            rows.append({"turn":i+1,"http_status":st,"content":text[:100],"prompt_tokens":total,"cached_tokens":hit,"cache_creation_tokens":created,"cache_ratio":hit/total if total else None,"e2e_seconds":elapsed})
            if i < len(turns)-1: await asyncio.sleep(2)
        continuity=all(r["http_status"]==200 for r in rows) and all("ORBIT-731" in rows[i]["content"] for i in (1,2))
        cache_observed=any(0<r["cached_tokens"]<=r["prompt_tokens"] for r in rows[1:])
        continuity=continuity and all(0<=r["cached_tokens"]<=r["prompt_tokens"] for r in rows)
        ok=continuity and cache_observed
        status="passed" if ok else ("blocked" if continuity and not cache_observed else "failed")
        return ok,f"Session 连续性={'通过' if continuity else '失败'}，后续轮缓存命中={[r['cached_tokens'] for r in rows[1:]]}",{"session_id":session_id,"turns":rows,"continuity":continuity,"cache_observed":cache_observed},status

    async def case_cache(self,m):
        datasets=[("math","计算 37*83，只回复结果。"),("code","写一个 Python 回文判断函数，只回复代码。"),("business","将订单已支付改写成正式通知。")]
        observations={}; probe_ratios=[]; group_ratios=[]; usage_complete=True; requests_ok=True
        for name,question in datasets:
            prefix=f"GLM53-CACHE-{name}-"+("固定公共上下文。"*700)
            group=[]
            for label in ("seed","probe1","probe2"):
                st,b,elapsed,_=await self.request(m,[{"role":"system","content":prefix},{"role":"user","content":question}],thinking={"type":"enabled"},max_tokens=256)
                total,hit,created=cache_usage(b); ratio=hit/total if total else None
                group.append({"label":label,"http_status":st,"prompt_tokens":total,"cached_tokens":hit,"cache_creation_tokens":created,"cache_ratio":ratio,"e2e_seconds":elapsed})
                requests_ok=requests_ok and st==200 and 0<=hit<=total
                if label!="seed":
                    usage_complete=usage_complete and total>0 and ("prompt_tokens_details" in ((b or {}).get("usage") or {}) or "prompt_cache_hit_tokens" in ((b or {}).get("usage") or {}) or "cache_read_input_tokens" in ((b or {}).get("usage") or {}))
                    if ratio is not None: probe_ratios.append(ratio)
            ratios=[x["cache_ratio"] for x in group[1:] if x["cache_ratio"] is not None]
            group_ratio=sum(ratios)/len(ratios) if ratios else 0; group_ratios.append(group_ratio); observations[name]=group
        mean=sum(probe_ratios)/len(probe_ratios) if probe_ratios else 0
        if not usage_complete:
            return False,"接口未完整返回缓存 usage，无法验证",{"target":0.80,"tolerance":0.10,"mean":mean,"groups":group_ratios,"observations":observations},"blocked"
        ok=requests_ok and len(probe_ratios)==6 and mean>=0.70 and all(x>=0.50 for x in group_ratios)
        return ok,f"缓存读取率均值 {mean:.1%}（目标80%，允许±10个百分点）",{"target":0.80,"tolerance":0.10,"mean":mean,"groups":group_ratios,"observations":observations},("passed" if ok else "failed" if all(r["http_status"]==200 for group in observations.values() for r in group) else "blocked")

    async def case_fingerprint_identity(self,m):
        st,b,_,_=await self.request(m,[{"role":"user","content":"只回复 OK"}])
        returned=(b or {}).get("model"); ok=st==200 and identity_match(m,returned)
        status="passed" if ok else "blocked" if st!=200 or not returned else "failed"
        return ok,f"HTTP {st}，请求={m}，返回={returned}",{"requested_model":m,"returned_model":returned},status
    async def _capability_probe(self,m,payload,label,validator):
        st,b,_,_=await self.request(m,[{"role":"user","content":"执行测试并只返回结果"}],**payload)
        raw=json.dumps(b or {},ensure_ascii=False)[:500]; text=raw.lower(); unsupported=any(x in text for x in ("unsupported","not support","unknown field","invalid parameter","不支持"))
        ok=st==200 and validator(b)
        status=probe_status(st,ok)
        return ok,f"{label}：HTTP {st}，{'支持' if ok else ('接口拒绝/疑似未开放' if unsupported else '未满足')}",{"http_status":st,"unsupported_error":unsupported,"response":raw,"probe":label},status
    async def case_fingerprint_thinking(self,m):
        return await self._capability_probe(m,{"thinking":{"type":"enabled"},"reasoning_effort":"high","max_tokens":1024},"thinking+reasoning_effort",lambda b:bool(msg(b).get("reasoning_content") or msg(b).get("reasoning")))
    async def case_fingerprint_tool_choice(self,m):
        return await self._capability_probe(m,{"tools":self.tools(),"tool_choice":"required"},"tool_choice=required",lambda b:bool(msg(b).get("tool_calls")))
    async def case_fingerprint_schema(self,m):
        schema={"type":"json_schema","json_schema":{"name":"answer_schema","strict":True,"schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":False}}}
        return await self._capability_probe(m,{"response_format":schema},"json_schema",lambda b:_schema_valid(content(b)))
    async def case_fingerprint_stream(self,m):
        st,_,_,o=await self.request(m,[{"role":"user","content":"只回复 OK"}],stream=True); ok=st==200 and bool(o.get("done")) and not o.get("invalid_events") and o.get("content", "").strip()=="OK" and "stop" in o.get("finish_reasons",[])
        return ok,f"stream：HTTP {st}，DONE={o.get('done')}，非法事件={o.get('invalid_events',0)}",{"http_status":st,**o,"probe":"stream"},probe_status(st,ok)
    async def case_fingerprint_silent_ignore(self,m):
        # GLM3/GLM5.3 必须开启深度思考：不再发送 disabled 对照，改为验证 enabled 的稳定返回。
        statuses=[]; lengths=[]
        for prompt in ("计算 17*23，并说明步骤。", "解释为什么 2+2=4，并说明步骤。"):
            st,b,_,_=await self.request(m,[{"role":"user","content":prompt}],thinking={"type":"enabled"},max_tokens=2048)
            r=msg(b).get("reasoning_content") or msg(b).get("reasoning") or ""; statuses.append(st); lengths.append(len(r))
        ok=all(st==200 for st in statuses) and all(n>0 for n in lengths)
        return ok,f"Thinking 必须开启：HTTP {statuses}，reasoning 字符数={lengths}",{"statuses":statuses,"reasoning_chars":lengths,"probe":"thinking_enabled_stability"},("passed" if ok else "failed" if any(code==200 and length==0 for code,length in zip(statuses,lengths)) else "blocked")
    async def case_fingerprint_context(self,m):
        return await self.case_quality_context(m)
    async def case_fingerprint_summary(self,m):
        return False,"本地汇总不构成远端证据",{"synthetic":True},"blocked"


def summarize_results(results):
    counts={x:sum(r.status==x for r in results) for x in ("passed","failed","warning","blocked","error")}; hard=counts["failed"]+counts["error"]; evaluated=counts["passed"]+hard
    fingerprint=[r for r in results if r.case_id.startswith("GLM-53-F") and r.case_id != "GLM-53-F08"]
    warnings=[r for r in fingerprint if r.status=="warning"]
    hard_fingerprint=[r for r in fingerprint if r.status in {"failed","error"}]
    if hard_fingerprint: fingerprint_level="证据不足/存在硬失败"
    elif warnings: fingerprint_level="部分能力或接口层限制，不能认定满血"
    elif {r.case_id for r in fingerprint if r.status=="passed" and r.passed}=={f"GLM-53-F0{i}" for i in range(1,8)}: fingerprint_level="已测接口指纹相容（不证明同款或满血）"
    else: fingerprint_level="无法判定"
    diagnosis=[{"case_id":r.case_id,"name":r.name,"status":r.status,"details":r.details,"observations":r.observations} for r in fingerprint if r.status!="passed"]
    return {"generated_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"total":len(results),"passed":counts["passed"],"failed":hard,"warning":counts["warning"],"blocked":counts["blocked"],"error":counts["error"],"pass_rate":counts["passed"]/evaluated if evaluated else 0,"all_passed":hard==0 and any(r.status in {"passed","warning"} and r.case_id!="GLM-53-F08" for r in results),"results":[asdict(r) for r in results],"model_validation":assess_validation(results,results[0].model if results else ""),"fingerprint_assessment":{"level":fingerprint_level,"confidence":"medium" if any(r.status=="passed" and r.passed for r in fingerprint) else "low","tested":len(fingerprint),"warnings":len(warnings),"hard_failures":len(hard_fingerprint),"suspected_degradations":diagnosis}}

def write_report(out,summary):
    out.mkdir(parents=True,exist_ok=True); (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    fp=summary.get("fingerprint_assessment",{})
    rows=[]
    for r in summary["results"]:
        rows.append([f"{r['case_id']} {r['name']}",r['model'],r['status'],f"{r.get('duration_seconds',0):.2f}",r['details']])
    # 只有真跑过指纹用例才显示满血结论；单独跑核心卡片时这行是噪声。
    extra_stats=validation_report_stats(summary)
    if fp.get("tested"):
        extra_stats += [f"满血能力指纹结论：{fp.get('level','未执行')}",
                        f"指纹置信度：{fp.get('confidence','—')}；已测试：{fp.get('tested',0)} 项；"
                        f"疑似限制/异常：{fp.get('warnings',0)+fp.get('hard_failures',0)} 项"]
    report=case_table_report("GLM 5.3 专项测试报告",
        total=summary['total'],passed=summary['passed'],failed=summary['failed'],
        warning=summary.get('warning',0),blocked=summary.get('blocked',0),pass_rate=summary.get('pass_rate'),
        extra_stats=extra_stats,
        rows=rows,status_column=2)
    (out/"report.md").write_text(report,encoding="utf-8")
