#!/usr/bin/env python3
"""Run the open MathArena/arxivmath final-answer benchmark."""
from __future__ import annotations
import argparse, csv, json, os, re, time
from pathlib import Path
import httpx
from volcperf.protocols import OPENAI_COMPATIBLE_PROTOCOLS, endpoint, headers, payload
from common.suite_runtime import finalize
from common.audit import AuditLog
from common.report_style import stats_lines

BOXED = re.compile(r"\\(?:boxed|fbox)\s*\{((?:[^{}]|\{[^{}]*\})*)\}", re.S)
def answer_text(text: str) -> str:
    matches = BOXED.findall(text or "")
    return (matches[-1] if matches else (text or "").strip()).strip()
def stream_piece(protocol: str, data: dict) -> str:
    if protocol in OPENAI_COMPATIBLE_PROTOCOLS:
        return "".join((c.get("delta") or {}).get("content") or "" for c in data.get("choices", []))
    if protocol == "anthropic":
        return ((data.get("delta") or {}).get("text") or "") if data.get("type") == "content_block_delta" else ""
    if protocol == "gemini":
        return "".join(p.get("text") or "" for c in data.get("candidates", []) for p in ((c.get("content") or {}).get("parts") or []))
    return ""

def canonical(value: str) -> str:
    value = answer_text(str(value)).lower().strip()
    value = re.sub(r"\\(?:left|right)\b", "", value)
    value = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\s+", "", value).replace("\\,", "").replace("$", "")
    return value

def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="ARXIVMATH", choices=["ARXIVMATH"])
    ap.add_argument("--output-root", type=Path, default=Path("results"))
    ap.add_argument("--max-problems", type=int, default=None)
    args=ap.parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("缺少 datasets 依赖，请安装 requirements.txt") from exc
    dataset_id=os.getenv("ARXIVMATH_DATASET","MathArena/arxivmath")
    split=os.getenv("ARXIVMATH_SPLIT","train")
    kwargs={}; token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token: kwargs["token"]=token
    rows=list(load_dataset(dataset_id, split=split, **kwargs))
    if args.max_problems: rows=rows[:args.max_problems]
    if not rows: raise SystemExit("ArXivMath 数据集为空")
    key=os.getenv("VOLC_API_KEY"); base=os.getenv("VOLC_BASE_URL"); model=os.getenv("VOLC_MODEL"); protocol=os.getenv("VOLC_PROTOCOL","volcengine")
    if not all((key,base,model)): raise SystemExit("缺少 VOLC_API_KEY/VOLC_BASE_URL/VOLC_MODEL")
    out=args.output_root/f"ARXIVMATH-{time.strftime('%Y%m%d-%H%M%S')}"; out.mkdir(parents=True,exist_ok=True)
    results=[]
    audit=AuditLog(out/'requests.jsonl')
    with httpx.Client(timeout=httpx.Timeout(600,connect=30), verify=os.getenv('VOLC_VERIFY_SSL','true').lower() not in {'0','false'}) as client:
      for i,row in enumerate(rows):
        question=str(row.get("question") or "").strip(); gold=str(row.get("answer") or "").strip(); started=time.perf_counter()
        rec={"index":i,"paper_id":row.get("paper_id"),"title":row.get("title"),"question":question,"gold_answer":gold}
        try:
          prompt="You are given a difficult question. Solve it. Put the final answer within \\boxed{}\n\n"+question
          body=payload(protocol,model,prompt,16384,True); body["stream"]=True
          text_parts=[]; raw=[]
          with client.stream("POST",endpoint(protocol,base,model),headers=headers(protocol,key,f"{os.getenv('VOLC_RECORD_ID','ARXIVMATH')}-{i}"),json=body) as resp:
            if resp.status_code>=400: raise RuntimeError(f"HTTP {resp.status_code}: {resp.read()[:500]}")
            for line in resp.iter_lines():
              line=line if isinstance(line,str) else line.decode("utf8","ignore"); raw.append(line)
              if not line.startswith("data: "): continue
              data=line[6:]
              if data=="[DONE]": continue
              try:
                piece=stream_piece(protocol, json.loads(data))
                if piece: text_parts.append(piece)
              except json.JSONDecodeError: pass
          response="".join(text_parts); predicted=answer_text(response)
          rec.update({"status":"ok","response_text":response,"predicted_answer":predicted,"correct":canonical(predicted)==canonical(gold),"latency_ms":round((time.perf_counter()-started)*1000,2),"raw_sse":"\n".join(raw)})
        except Exception as exc: rec.update({"status":"error","error":str(exc),"correct":False,"latency_ms":round((time.perf_counter()-started)*1000,2)})
        results.append(rec)
        audit.write(rec)
    audit.close()
    evaluated=[x for x in results if x["status"]=="ok"]; correct=sum(x.get("correct",False) for x in evaluated)
    summary={"schema_version":"arxivmath.result.v1","record_id":os.getenv("VOLC_RECORD_ID"),"test_kind":"arxivmath","case":"ARXIVMATH","dataset":dataset_id,"split":split,"total_problems":len(results),"evaluated":len(evaluated),"correct":correct,"accuracy":correct/len(evaluated) if evaluated else 0,"failed_requests":len(results)-len(evaluated),"admission_passed":len(evaluated)==len(results) and len(evaluated)>0,"answer_policy":"官方 answer 字段；最终答案抽取与规范化仅用于评分，不修改数据集题目/答案","status":"completed"}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    report=['# ArXivMath 测试报告','',*stats_lines(len(results),correct,len(evaluated)-correct,0,len(results)-len(evaluated),summary['accuracy'],
        [f'数据集：`{dataset_id}`（Hugging Face 官方开源数据集）',f'有效回答：{len(evaluated)}',
         '说明：本测试只评估 final answer，不代表完整证明能力。'])]
    (out/"report.md").write_text('\n'.join(report)+'\n',encoding="utf8")
    finalize(out)
    print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
