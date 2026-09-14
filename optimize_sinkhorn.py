import argparse, json, numpy as np
from pathlib import Path

def sinkhorn(C, reg=0.1, maxiter=1000):
    N,K=C.shape; kernel=np.exp(-C/reg); a,b=np.ones(N)/N,np.ones(K)/K; u,v=np.ones(N),np.ones(K)
    for i in range(maxiter):
        u0=u.copy(); u=a/(kernel@v); v=b/(kernel.T@u)
        if np.max(np.abs(u-u0))<1e-9: print(f"  Converged at {i+1}"); break
    return u[:,None]*kernel*v[None,:]

def reassign(inp, reg=0.1, maxiter=1000):
    data=[json.loads(l) for l in open(inp, encoding="utf8")]
    N=len(data); l1=np.array([d["indices"][0] for d in data]); K=int(l1.max())+1
    cnt=np.bincount(l1,minlength=K); print(f"L1: {np.sum(cnt>0)}/{K}={np.sum(cnt>0)/K:.2%}")
    cost=1.0/(cnt+1e-6); C=cost[l1][:,None]*np.ones((N,K))
    for k in range(K): C[:,k]+=np.abs(cnt[k]-N/K)/N
    P=sinkhorn(C,reg,maxiter); new=np.argmax(P,axis=1); new_cnt=np.bincount(new,minlength=K)
    print(f"New: {np.sum(new_cnt>0)}/{K}={np.sum(new_cnt>0)/K:.2%}, max={new_cnt.max()}")
    for i,d in enumerate(data):
        d["indices"][0]=int(new[i]); t=[f"<l{j+1}_{c}>" for j,c in enumerate(d["indices"])]; d["tokens"]=d["sid"]=t
    return data

def metrics(data):
    idx=[tuple(d["indices"]) for d in data]; N,L=len(idx),len(data[0]["indices"]); u=len(set(idx)); icr=u/N
    bs=[]
    for lv in range(L):
        c=[d["indices"][lv] for d in data]; K=max(c)+1; cnt=np.bincount(c,minlength=K); a=int(np.sum(cnt>0)); nz=cnt[cnt>0]
        bs.append({"level":lv+1,"bucket_count":a,"mean":float(np.mean(nz)),"max":int(cnt.max())})
    return {"N":N,"L":L,"collision_rate":1-icr,"unique_paths":u,"icr":icr,"bucket_size":bs}

p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output_dir",required=True)
p.add_argument("--reg",type=float,default=0.1); p.add_argument("--max_iter",type=int,default=1000); a=p.parse_args()
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); data=reassign(a.input,a.reg,a.max_iter)
m=metrics(data); print(f"CR={m['collision_rate']:.6f}, ICR={m['icr']:.6f}")
for b in m["bucket_size"]: print(f"  L{b['level']}: {b['bucket_count']}, max={b['max']}")
with (out/"indices.jsonl").open("w",encoding="utf8") as f:
    for d in data: f.write(json.dumps(d,ensure_ascii=False)+"\n")
with (out/"metrics.json").open("w",encoding="utf8") as f: json.dump(m,f,ensure_ascii=False,indent=2)
print("Done!")

