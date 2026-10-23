#!/usr/bin/env python3
"""Run the official DataCurve DeepSWE corpus through Pier; no tasks are reimplemented."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, time
from pathlib import Path
from common.suite_runtime import finalize
from common.report_style import stats_lines
from volcperf.protocols import OPENAI_COMPATIBLE_PROTOCOLS, api_root

ROOT=Path(__file__).resolve().parent.parent
REPO_URL="https://github.com/datacurve-ai/deep-swe.git"
CORPUS=ROOT/"data_cache"/"deepswe-official"

def main()->int:
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--case',default='DEEPSWE'); ap.add_argument('--output-root',type=Path,default=ROOT/'results'); ap.add_argument('--n-tasks',type=int,default=int(os.getenv('DEEPSWE_N_TASKS','0'))); ap.add_argument('--sample-seed',type=int,default=int(os.getenv('DEEPSWE_SAMPLE_SEED','0'))); args=ap.parse_args()
 if not (CORPUS/'tasks').is_dir():
  CORPUS.parent.mkdir(parents=True,exist_ok=True); subprocess.run(['git','clone','--depth','1',REPO_URL,str(CORPUS)],check=True)
 if shutil.which('pier') is None: raise SystemExit('缺少 Pier：请安装官方运行器 datacurve-pier（DeepSWE 官方推荐）')
 key=os.getenv('VOLC_API_KEY'); base=os.getenv('VOLC_BASE_URL'); model=os.getenv('VOLC_MODEL'); protocol=os.getenv('VOLC_PROTOCOL','volcengine')
 if not all((key,base,model)): raise SystemExit('缺少 VOLC_API_KEY/VOLC_BASE_URL/VOLC_MODEL')
 base=api_root(protocol,base)
 provider='openai' if protocol in OPENAI_COMPATIBLE_PROTOCOLS else protocol
 full_model=model if '/' in model else f'{provider}/{model}'
 env=os.environ.copy(); env['OPENAI_API_KEY']=key; env['OPENAI_BASE_URL']=base.rstrip('/')
 if protocol=='anthropic': env['ANTHROPIC_API_KEY']=key
 out=args.output_root/f'DEEPSWE-{time.strftime("%Y%m%d-%H%M%S")}'; out.mkdir(parents=True,exist_ok=True)
 cmd=['pier','run','-p',str(CORPUS/'tasks'),'--agent',os.getenv('DEEPSWE_AGENT','mini-swe-agent'),'--model',full_model]
 if args.n_tasks>0: cmd += ['--n-tasks',str(args.n_tasks),'--sample-seed',str(args.sample_seed)]
 started=time.time()
 proc=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True)
 (out/'run.log').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf8')
 rewards=list(ROOT.rglob('reward.json'))
 # Only collect artifacts generated after this run, avoiding stale results.
 rewards=[p for p in rewards if p.stat().st_mtime>=started-2]
 rows=[]
 for p in rewards:
  try: rows.append(json.loads(p.read_text(encoding='utf8'))|{'artifact':str(p.relative_to(ROOT))})
  except (OSError,json.JSONDecodeError): pass
 (out/'requests.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf8')
 passed=sum(1 for x in rows if float(x.get('reward',x.get('pass_rate',0)) or 0)>=1); total=len(rows)
 summary={'schema_version':'deepswe.result.v1','record_id':os.getenv('VOLC_RECORD_ID'),'test_kind':'deepswe','case':'DEEPSWE','dataset':'datacurve-ai/deep-swe/tasks','runner':'Pier','agent':os.getenv('DEEPSWE_AGENT','mini-swe-agent'),'model':full_model,'requested_tasks':args.n_tasks or 'all','scored_tasks':total,'passed_tasks':passed,'pass_rate':passed/total if total else None,'exit_code':proc.returncode,'admission_passed':proc.returncode==0 and total>0,'status':'completed' if proc.returncode==0 else 'failed'}
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 report=['# DeepSWE 测试报告','',*stats_lines(total,passed,total-passed,0,0,summary['pass_rate'],
    ["数据集：DataCurve 官方开源 `deep-swe/tasks`",f"运行器：Pier + {summary['agent']}",f"模型：`{full_model}`",
     '说明：任务、环境和 verifier 均来自官方开源项目，本系统不重新实现测试集。'])]
 (out/'report.md').write_text('\n'.join(report)+'\n',encoding='utf8')
 finalize(out)
 print(json.dumps(summary,ensure_ascii=False,indent=2)); return proc.returncode
if __name__=='__main__': raise SystemExit(main())
