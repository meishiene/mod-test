#!/usr/bin/env python3
"""Run exactly 10 T0-profile requests without replenishing failures."""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import httpx
from k3perf.cases import CASES
from k3perf.client import stream_chat
from k3perf.config import ApiConfig
from k3perf.data import load_prompts
from k3perf.metrics import summarize
from k3perf.reporting import write_artifacts
from k3perf.runner import ResultWriter

async def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--requests',type=int,default=10); args=ap.parse_args()
    case=CASES['T0']; cfg=ApiConfig.from_env(); prompts=load_prompts(case.data_file)[:args.requests]
    out=Path('results')/f'T0-FIXED10-{time.strftime("%Y%m%d-%H%M%S")}'; out.mkdir(parents=True)
    meta={'case':'T0-FIXED10','profile':'T0','fixed_requests':args.requests,'concurrency':2,'max_tokens':case.max_tokens,'temperature':1,'endpoint':cfg.endpoint,'model':cfg.model,'started_at_epoch':time.time(),'no_replenishment':True,'environment':{'python':sys.version,'pid':os.getpid()}}
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    writer=ResultWriter(out/'requests.jsonl'); rows=[]; sem=asyncio.Semaphore(2)
    timeout=httpx.Timeout(cfg.request_timeout_seconds,connect=cfg.connect_timeout_seconds)
    limits=httpx.Limits(max_connections=2,max_keepalive_connections=2)
    async with httpx.AsyncClient(timeout=timeout,limits=limits,verify=cfg.verify_ssl,trust_env=cfg.trust_env,headers={'Accept':'text/event-stream'}) as session:
        async def one(i):
            async with sem:
                r=await stream_chat(session,cfg,f'T0-FIXED10-{i:02d}-{uuid.uuid4().hex[:8]}',prompts[i].content,case.max_tokens,'T0-FIXED10',prompts[i].id,'measured',True)
            rows.append(r);writer.write(r);print(f"{i+1}/{args.requests} status={r.get('http_status')} success={r.get('success')} ttft={r.get('ttft_seconds')} error={r.get('error_type')}")
        await asyncio.gather(*(one(i) for i in range(args.requests)))
    writer.close()
    summary=summarize(case,rows,None);summary['case']='T0-FIXED10';summary['diagnostic_only']=True;summary['fixed_requests']=args.requests;summary['no_replenishment']=True
    # For this fixed diagnostic, record whether all 10 yielded usable content separately.
    summary['all_10_usable']=sum(bool(r.get('success')) for r in rows)==args.requests
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');write_artifacts(out,rows,summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2));print('Results:',out.resolve())
    return 0 if summary['all_10_usable'] else 2
if __name__=='__main__':raise SystemExit(asyncio.run(main()))
