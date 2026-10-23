import argparse,asyncio,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from authenticity import CASES,run
from common.model_contracts import model_matches_family
from volcperf.protocols import api_root
p=argparse.ArgumentParser();p.add_argument('case',choices=CASES);p.add_argument('--output-root',type=Path,default=Path('results'));a=p.parse_args()
key=os.getenv('VOLC_API_KEY');base=os.getenv('VOLC_BASE_URL');model=os.getenv('VOLC_MODEL')
if not key or not base or not model: raise SystemExit('缺少 VOLC_API_KEY/VOLC_BASE_URL/VOLC_MODEL')
_gok,_greason=model_matches_family('deepseek',model)
if not _gok: raise SystemExit(_greason)  # 模型门禁：防误配模型产生假通过
base=api_root(os.getenv('VOLC_PROTOCOL','volcengine'),base)
path,summary=asyncio.run(run(a.case,base,key,model,a.output_root)); print(f'结果目录: {path.resolve()}'); raise SystemExit(0 if summary.get('admission_passed') else 2)
