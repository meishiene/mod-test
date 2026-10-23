#!/usr/bin/env python3
"""Run Harvey's official Legal Agent Benchmark harness and evaluator."""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, time
from pathlib import Path
from common.suite_runtime import finalize
from common.report_style import stats_lines
from volcperf.protocols import OPENAI_COMPATIBLE_PROTOCOLS, api_root
ROOT=Path(__file__).resolve().parent.parent
REPO=ROOT/'data_cache'/'harvey-labs-official'
URL='https://github.com/harveyai/harvey-labs.git'
REF='v1.0'

def main()->int:
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--case',default='LEGAL_AGENT_BENCHMARK'); ap.add_argument('--output-root',type=Path,default=ROOT/'results'); ap.add_argument('--n-tasks',type=int,default=int(os.getenv('LEGAL_AGENT_BENCHMARK_N_TASKS','1'))); ap.add_argument('--sample-seed',type=int,default=int(os.getenv('LEGAL_AGENT_BENCHMARK_SAMPLE_SEED','0'))); a=ap.parse_args()
 if not (REPO/'harness').is_dir(): REPO.parent.mkdir(parents=True,exist_ok=True); subprocess.run(['git','clone','--depth','1','--branch',REF,URL,str(REPO)],check=True)
 if shutil.which('uv') is None: raise SystemExit('缺少 uv；请按 Harvey Labs 官方 scripts/setup.sh 安装运行环境')
 key=os.getenv('VOLC_API_KEY'); base=os.getenv('VOLC_BASE_URL'); model=os.getenv('VOLC_MODEL'); protocol=os.getenv('VOLC_PROTOCOL','volcengine')
 if not all((key,base,model)): raise SystemExit('缺少 VOLC_API_KEY/VOLC_BASE_URL/VOLC_MODEL')
 if protocol not in OPENAI_COMPATIBLE_PROTOCOLS: raise SystemExit('当前集成仅支持 OpenAI/火山云/百度千帆兼容端点')
 base=api_root(protocol,base)
 env=os.environ.copy(); env.update({'OPENAI_API_KEY':key,'OPENAI_BASE_URL':base.rstrip('/'),'OPENAI_API_BASE':base.rstrip('/')})
 task_files=sorted((REPO/'tasks').rglob('task.json')); tasks=[str(p.parent.relative_to(REPO/'tasks')) for p in task_files]
 import random; random.Random(a.sample_seed).shuffle(tasks); tasks=tasks[:max(1,a.n_tasks)]
 out=a.output_root/f'LEGAL-AGENT-BENCHMARK-{time.strftime("%Y%m%d-%H%M%S")}'; out.mkdir(parents=True,exist_ok=True); records=[]; logs=[]
 agent_model=model if '/' in model else f'openai/{model}'; judge_model=agent_model
 for task in tasks:
  cmd=['uv','run','python','-m','harness.run','--model',agent_model,'--task',task,'--max-turns',os.getenv('LEGAL_AGENT_BENCHMARK_MAX_TURNS','200')]
  run=subprocess.run(cmd,cwd=REPO,env=env,text=True,capture_output=True); text=run.stdout+'\n'+run.stderr; logs.append(f'## RUN {task}\n{text}')
  m=re.search(r'Run complete:\s*(.+)',text); run_id=m.group(1).strip() if m else None
  rec={'task':task,'run_id':run_id,'run_exit_code':run.returncode,'evaluated':False}
  if run_id:
   ev=subprocess.run(['uv','run','python','-m','evaluation.run_eval','--run-id',run_id,'--task',task,'--judges',judge_model],cwd=REPO,env=env,text=True,capture_output=True); et=ev.stdout+'\n'+ev.stderr; logs.append(f'## EVAL {task}\n{et}'); rec.update({'eval_exit_code':ev.returncode,'evaluated':ev.returncode==0})
   run_dir=REPO/'results'/run_id
   score_files=list(run_dir.glob('scores*.json')) if run_dir.is_dir() else []
   for sf in score_files:
    try:
     score=json.loads(sf.read_text(encoding='utf8')); rec['score_artifact']=str(sf.relative_to(REPO)); rec['score']=score.get('score',score.get('overall_score')); rec['criterion_pass_rate']=score.get('criterion_pass_rate',score.get('pass_rate')); break
    except (OSError,json.JSONDecodeError): pass
  records.append(rec)
 (out/'run.log').write_text('\n\n'.join(logs),encoding='utf8'); (out/'requests.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records),encoding='utf8')
 passed=sum(x.get('score')==1 or x.get('score')==1.0 for x in records); evaluated=sum(x['evaluated'] for x in records)
 summary={'schema_version':'legal-agent-benchmark.result.v1','record_id':os.getenv('VOLC_RECORD_ID'),'test_kind':'legal-agent-benchmark','case':'LEGAL_AGENT_BENCHMARK','dataset':'harveyai/harvey-labs@v1.0:tasks','runner':'official Harvey Labs harness + evaluator','model':agent_model,'judge_model':judge_model,'requested_tasks':len(tasks),'evaluated_tasks':evaluated,'all_pass_tasks':passed,'all_pass_rate':passed/evaluated if evaluated else None,'admission_passed':evaluated==len(tasks) and evaluated>0,'status':'completed' if evaluated==len(tasks) else 'failed','results':records}
 report=['# Legal Agent Benchmark 测试报告','',*stats_lines(len(tasks),passed,evaluated-passed,0,len(tasks)-evaluated,summary['all_pass_rate'],
    ['官方项目：`harveyai/harvey-labs@v1.0`（官方 harness/evaluator）',f"模型：`{agent_model}`",f"已评分：{evaluated}",
     '说明：任务材料、rubric、harness 和评分器均直接来自官方开源项目；当前使用被测模型作为单一 LLM judge。'])]
 (out/'report.md').write_text('\n'.join(report)+'\n',encoding='utf8')
 finalize(out)
 print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0 if summary['admission_passed'] else 2
if __name__=='__main__': raise SystemExit(main())
