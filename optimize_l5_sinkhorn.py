import argparse, json, numpy as np
from pathlib import Path

def sinkhorn(C, reg=0.5, maxiter=1000):
    N,K=C.shape; kernel=np.exp(-C/reg); a,b=np.ones(N)/N,np.ones(K)/K; u,v=np.ones(N),np.ones(K)
    for i in range(maxiter):
        u0=u.copy(); u=a/(kernel@v); v=b/(kernel.T@u)
        if np.max(np.abs(u-u0))<1e-9: print(f"  Converged at {i+1}"); break
    return u[:,None]*kernel*v[None,:]

def reassign_l5(inp, reg=0.5, maxiter=1000):
    data=[json.loads(l) for l in open(inp, encoding="utf8")]
    N=len(data); l5=np.array([d["indices"][4] for d in data]); K=int(l5.max())+1
    cnt=np.bincount(l5,minlength=K); active=np.sum(cnt>0)
    print(f"L5 before: {active}/{K}={active/K:.2%}, max={cnt.max()}, min={(cnt[cnt>0].min() if active>0 else 0)}")
    
    # 代价矩阵：让过度使用的 code 代价高
    cost=np.zeros((N,K))
    target=N/K
    for i in range(N):
        code=l5[i]
        # 当前 code 的使用频率
        freq=cnt[code]
        # 如果过度使用，增加代价
        if freq>target*1.5:
            cost[i,code]+=10.0*(freq-target)/target
        # 对未使用或使用不足的 code 降低代价
        for k in range(K):
            if cnt[k]<target*0.5:
                cost[i,k]-=5.0
    
    print(f"Applying Sinkhorn with reg={reg}...")
    P=sinkhorn(cost,reg,maxiter); new=np.argmax(P,axis=1); new_cnt=np.bincount(new,minlength=K)
    new_active=np.sum(new_cnt>0)
    print(f"L5 after: {new_active}/{K}={new_active/K:.2%}, max={new_cnt.max()}, min={(new_cnt[new_cnt>0].min() if new_active>0 else 0)}")
    
    for i,d in enumerate(data):
        d["indices"][4]=int(new[i]); t=[f"<l{j+1}_{c}>" for j,c in enumerate(d["indices"])]; d["tokens"]=d["sid"]=t
    return data

def metrics(data):
    idx=[tuple(d["indices"]) for d in data]; N,L=len(idx),len(data[0]["indices"]); u=len(set(idx)); icr=u/N
    bs=[]
    for lv in range(L):
        c=[d["indices"][lv] for d in data]; K=max(c)+1; cnt=np.bincount(c,minlength=K); a=int(np.sum(cnt>0)); nz=cnt[cnt>0]
        bs.append({"level":lv+1,"bucket_count":a,"mean":float(np.mean(nz)),"max":int(cnt.max())})
    return {"N":N,"L":L,"collision_rate":1-icr,"unique_paths":u,"icr":icr,"bucket_size":bs}

p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output_dir",required=True)
p.add_argument("--reg",type=float,default=0.5); p.add_argument("--max_iter",type=int,default=1000); a=p.parse_args()
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); data=reassign_l5(a.input,a.reg,a.max_iter)
m=metrics(data); print(f"\nFinal: CR={m['collision_rate']:.6f}, ICR={m['icr']:.6f}")
for b in m["bucket_size"]: print(f"  L{b['level']}: {b['bucket_count']}, max={b['max']}")
with (out/"indices.jsonl").open("w",encoding="utf8") as f:
    for d in data: f.write(json.dumps(d,ensure_ascii=False)+"\n")
with (out/"metrics.json").open("w",encoding="utf8") as f: json.dump(m,f,ensure_ascii=False,indent=2)
print("Done!")

