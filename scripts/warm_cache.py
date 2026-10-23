#!/usr/bin/env python3
"""Warm one cache-aware dataset with a single request, then verify cached_tokens with probes."""
from __future__ import annotations
import argparse, asyncio, json, sys, time, uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import httpx
from k3perf.cases import CACHE_CASES
from k3perf.client import stream_chat
from k3perf.config import ApiConfig
from k3perf.data import load_prompts

async def main():
 p=argparse.ArgumentParser();p.add_argument('case',choices=CACHE_CASES);p.add_argument('--probes',type=int,default=3);p.add_argument('--output-root',type=Path,default=Path('results/cache-warmup'));a=p.parse_args()
 case=CACHE_CASES[a.case]; cfg=ApiConfig.from_env(); records=load_prompts(case.data_file); a.output_root.mkdir(parents=True,exist_ok=True)
 timeout=httpx.Timeout(cfg.request_timeout_seconds,connect=cfg.connect_timeout_seconds)
 rows=[]
 async with httpx.AsyncClient(timeout=timeout,verify=cfg.verify_ssl,trust_env=cfg.trust_env,headers={'Accept':'text/event-stream'}) as session:
  for i in range(a.probes+1):
   r=await stream_chat(session,cfg,f'CACHE-WARM-{case.name}-{i}-{uuid.uuid4().hex[:8]}',records[i].content,16,'CACHE_WARMUP',records[i].id,'seed' if i==0 else 'probe',False)
   rows.append(r);print(i,'seed' if i==0 else 'probe','prompt',r.get('prompt_tokens'),'cached',r.get('cached_tokens'),'ratio',r.get('cache_ratio'),'error',r.get('error_type'))
 out=a.output_root/f'{case.name}-{time.strftime("%Y%m%d-%H%M%S")}.json';out.write_text(json.dumps({'case':case.name,'target':[case.cache_ratio_min,case.cache_ratio_max],'rows':rows},ensure_ascii=False,indent=2)+'\n')
 probes=rows[1:]; valid=[r.get('cache_ratio') for r in probes if r.get('cache_ratio') is not None]
 ok=len(valid)==len(probes) and all(case.cache_ratio_min <= x <= case.cache_ratio_max for x in valid)
 print('cache verification', 'PASSED' if ok else 'FAILED', out.resolve());return 0 if ok else 2
if __name__=='__main__':raise SystemExit(asyncio.run(main()))
