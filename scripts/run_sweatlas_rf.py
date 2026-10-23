#!/usr/bin/env python3
"""Run the official SWE-Atlas Refactoring corpus with Harbor."""
from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path
from common.suite_runtime import finalize
from common.report_style import stats_lines
from volcperf.protocols import OPENAI_COMPATIBLE_PROTOCOLS, api_root

ROOT=Path(__file__).resolve().parent.parent
REPO_URL="https://github.com/scaleapi/SWE-Atlas.git"
REPO=ROOT/"data_cache"/"swe-atlas-official"

def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',default='SWE_ATLAS_RF'); ap.add_argument('--output-root',type=Path,default=ROOT/'results')
    ap.add_argument('--n-tasks',type=int,default=int(os.getenv('SWE_ATLAS_RF_N_TASKS','0')))
    ap.add_argument('--sample-seed',type=int,default=int(os.getenv('SWE_ATLAS_RF_SAMPLE_SEED','0')))
    args=ap.parse_args()
    if not (REPO/'data'/'rf').is_dir():
        REPO.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--depth','1',REPO_URL,str(REPO)],cwd=ROOT,check=True)
    harbor=os.getenv('HARBOR_BIN','harbor')
    if subprocess.run(['which',harbor],capture_output=True).returncode!=0:
        raise SystemExit('缺少 Harbor：请按 SWE-Atlas 官方 README 安装 harbor')
    key=os.getenv('VOLC_API_KEY'); base=os.getenv('VOLC_BASE_URL'); model=os.getenv('VOLC_MODEL'); protocol=os.getenv('VOLC_PROTOCOL','volcengine')
    if not all((key,base,model)): raise SystemExit('缺少 VOLC_API_KEY/VOLC_BASE_URL/VOLC_MODEL')
    base=api_root(protocol,base)
    provider='openai' if protocol in OPENAI_COMPATIBLE_PROTOCOLS else protocol
    harbor_model=model if '/' in model else f'{provider}/{model}'
    env=os.environ.copy(); env['OPENAI_API_KEY']=key; env['OPENAI_BASE_URL']=base.rstrip('/')
    if protocol=='anthropic': env['ANTHROPIC_API_KEY']=key
    allowed_host=os.getenv('HARBOR_AGENT_ALLOWED_HOST') or base.split('/')[2]
    out=args.output_root/f'SWE-ATLAS-RF-{time.strftime("%Y%m%d-%H%M%S")}'; out.mkdir(parents=True,exist_ok=True)
    config=REPO/'run_config'/'rf'/'mswea_rf_config.yaml'
    cmd=[harbor,'run','-p',str(REPO/'data'/'rf'),'-a',os.getenv('SWE_ATLAS_RF_AGENT','mini-swe-agent'),'-m',harbor_model,'-e',os.getenv('SWE_ATLAS_RF_ENV','modal'),'-k',os.getenv('SWE_ATLAS_RF_ATTEMPTS','1'),'-n',str(args.n_tasks or 66),'--ak',f'config_file={config}','--ak',f'reasoning_effort={os.getenv("SWE_ATLAS_RF_REASONING","high")}', '--allow-agent-host',allowed_host,'-o',str(out)]
    if args.n_tasks>0: cmd += ['--sample-seed',str(args.sample_seed)]
    proc=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True)
    (out/'run.log').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf8')
    rewards=[]
    for p in out.rglob('reward.json'):
        try: rewards.append(json.loads(p.read_text(encoding='utf8'))|{'artifact':str(p.relative_to(out))})
        except (OSError,json.JSONDecodeError): pass
    (out/'requests.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rewards),encoding='utf8')
    def reward(x):
        for k in ('reward','pass_rate','score'):
            if isinstance(x.get(k),(int,float)): return float(x[k])
        return 0.0
    passed=sum(reward(x)>=1 for x in rewards)
    summary={'schema_version':'swe-atlas-rf.result.v1','record_id':os.getenv('VOLC_RECORD_ID'),'test_kind':'swe-atlas-rf','case':'SWE_ATLAS_RF','dataset':'scaleapi/SWE-Atlas:data/rf','runner':'Harbor','agent':os.getenv('SWE_ATLAS_RF_AGENT','mini-swe-agent'),'model':harbor_model,'requested_tasks':args.n_tasks or 'official default sample','scored_tasks':len(rewards),'passed_tasks':passed,'pass_rate':passed/len(rewards) if rewards else None,'exit_code':proc.returncode,'admission_passed':proc.returncode==0 and bool(rewards),'status':'completed' if proc.returncode==0 else 'failed'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    report=['# SWE-Atlas RF 测试报告','',*stats_lines(len(rewards),passed,len(rewards)-passed,0,0,summary['pass_rate'],
        ['数据集：Scale AI 官方开源 `SWE-Atlas/data/rf`',f"运行器：Harbor + {summary['agent']}",f"模型：`{harbor_model}`",
         '说明：任务、环境、提示配置和 verifier 均来自官方开源项目，本系统不重写测试集。'])]
    (out/'report.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    finalize(out)
    print(json.dumps(summary,ensure_ascii=False,indent=2)); return proc.returncode
if __name__=='__main__': raise SystemExit(main())
