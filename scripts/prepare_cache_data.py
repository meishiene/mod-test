#!/usr/bin/env python3
from __future__ import annotations
import gzip, hashlib, json, random
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'data_cache'
# target, records; 85% of generated payload words are byte-identical common prefix.
SPECS={'CI1':(7500,500),'CI2':(12000,500),'CI3':(24000,500),'CI4':(48000,500),'CC1':(10000,7300)}
VOCAB=22
TARGET_RATIO=0.85
OVERHEAD=120

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 manifest={'generator_version':1,'format':'cache-prefix-recipe-v1','target_common_prefix_ratio':TARGET_RATIO,'datasets':{}}
 for name,(target,count) in SPECS.items():
  payload=max(1,target-OVERHEAD); common=int(payload*TARGET_RATIO); unique=payload-common
  path=OUT/f'{name}.jsonl.gz'
  with gzip.open(path,'wt',encoding='utf-8',compresslevel=9) as f:
   for i in range(count):
    seed=f'k3-cache-v1:{name}:{i}'
    row={'id':f'{name}-{i:05d}','dataset':name,'target_prompt_tokens':target,'cache_mode':True,'marker':hashlib.sha256(seed.encode()).hexdigest(),'common_word_count':common,'unique_word_count':unique,'word_offset':random.Random(seed).randrange(VOCAB),'word_count':0}
    f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
  manifest['datasets'][name]={'path':path.name,'records':count,'target_prompt_tokens':target,'common_word_count':common,'unique_word_count':unique,'designed_common_ratio':common/(common+unique),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
  print(name,count,'common',common,'unique',unique)
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
