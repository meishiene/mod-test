from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).parent
SETS=['general_common','general_full','virtual_partner_common','virtual_partner_full','sillytavern_style','sillytavern_smoke']
def load(name):
 p=ROOT/name/'cases.jsonl'
 if not p.exists(): raise SystemExit(f'找不到子项目: {name}')
 return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]
def main():
 ap=argparse.ArgumentParser(description='内容审核边界测试案例离线检查器')
 ap.add_argument('project',choices=SETS+['all']); ap.add_argument('--show',type=int,default=0,help='显示前 N 条用例')
 ap.add_argument('--audit-invariance',action='store_true',help='检查上下文不变性套件的分组完整性')
 a=ap.parse_args(); names=SETS if a.project=='all' else [a.project]
 total=0
 for name in names:
  rows=load(name); total+=len(rows)
  if a.audit_invariance and name in {'general_common','general_full','virtual_partner_common','virtual_partner_full'}:
   groups={}
   for row in rows: groups.setdefault(row.get('invariance_group'), []).append(row)
   expected_per_group=5
   bad=[(gid,len(items)) for gid,items in groups.items() if len(items) != expected_per_group or len({x.get('expected_decision') for x in items}) != 1]
   missing=[r['id'] for r in rows if not all(k in r.get('pass_criteria',{}) for k in ('decision','category','decision_rule','executed_user_input','followed_context_override','system_prompt_leaked','final_output_complete'))]
   if bad or missing: raise SystemExit(f'审核套件校验错误: groups={bad[:5]} missing_pass_criteria={missing[:5]}')
   print(f"  invariance_groups={len(groups)} expected_per_group={expected_per_group} pass_criteria=complete status=ok")
  print(f'[{name}] cases={len(rows)} categories={len(set(r["category"] for r in rows))} variants={len(set(r["variant"] for r in rows))}')
  if a.show:
   for r in rows[:a.show]:
    view={k:r.get(k) for k in ('id','text','category','variant','expected_action','expected_decision','context_variant')}
    print(json.dumps(view,ensure_ascii=False))
 print(f'TOTAL cases={total}')
if __name__=='__main__': main()
