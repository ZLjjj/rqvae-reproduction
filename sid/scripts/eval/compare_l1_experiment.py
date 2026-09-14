#!/usr/bin/env python3
"""Compare L1 codebook usage and full SID collision for two index JSONL files."""
import argparse, json
from pathlib import Path
import numpy as np

def load(path):
    codes=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            r=json.loads(line); v=r.get('indices') or r.get('tokens') or r.get('sid')
            row=[]
            for x in v:
                row.append(int(x) if isinstance(x,int) else int(str(x).rsplit('_',1)[1][:-1]))
            codes.append(row)
    return np.asarray(codes,dtype=np.int64)

def one(path, sizes):
    c=load(path); n=len(c)
    out={'path':str(path),'N':n,'layers':[]}
    for l,size in enumerate(sizes):
        used=np.unique(c[:,l]); out['layers'].append({'level':l+1,'active_codes':int(len(used)),'utilization':float(len(used)/size),'max_code_count':int(np.bincount(c[:,l],minlength=size).max())})
    unique=len({tuple(x) for x in c.tolist()}); out['unique_paths']=unique; out['icr']=unique/n; out['collision_rate']=1-unique/n
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--baseline',required=True); p.add_argument('--candidate',required=True); p.add_argument('--output',required=True); p.add_argument('--baseline-sizes',type=int,nargs=5,default=[1024,1024,1024,1024,512]); p.add_argument('--candidate-sizes',type=int,nargs=5,default=[1024,1024,1024,1024,512]); a=p.parse_args()
    out={'baseline':one(a.baseline,a.baseline_sizes),'candidate':one(a.candidate,a.candidate_sizes)}
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(a.output)
if __name__=='__main__': main()
