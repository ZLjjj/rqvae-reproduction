#!/usr/bin/env python3
"""子空间独立 RQ with SimVQ - 在第3层使用 SimVQ 来提升码本利用率"""
import argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

class SimVQLayer(nn.Module):
    def __init__(self, dim, codebook_size, basis_size=None):
        super().__init__()
        self.dim, self.K = dim, codebook_size
        self.M = basis_size or max(codebook_size // 4, 64)
        self.basis = nn.Parameter(torch.randn(self.M, dim) * 0.01)
        self.transform = nn.Linear(self.M, self.K, bias=False)
        with torch.no_grad():
            nn.init.orthogonal_(self.basis)
            self.transform.weight.zero_()
            for i in range(min(self.M, self.K)): self.transform.weight[i,i] = 1.0
    
    def get_codebook(self): return torch.mm(self.transform.weight, self.basis)  # [K,M]@[M,dim]=[K,dim]
    
    def forward(self, x):
        codebook = self.get_codebook()
        distances = torch.cdist(x, codebook)
        codes = distances.argmin(dim=1)
        return codes, codebook[codes]

class TraditionalVQLayer(nn.Module):
    def __init__(self, dim, codebook_size):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim) * 0.01)
        with torch.no_grad(): nn.init.xavier_uniform_(self.codebook)
    
    def forward(self, x):
        distances = torch.cdist(x, self.codebook)
        codes = distances.argmin(dim=1)
        return codes, self.codebook[codes]

class SimVQResidualQuantizer(nn.Module):
    def __init__(self, dim, codebook_size=1024, use_simvq_last=True):
        super().__init__()
        self.layer1 = SimVQLayer(dim, codebook_size, codebook_size//4)
        self.layer2 = SimVQLayer(dim, codebook_size, codebook_size//4)
        self.layer3 = SimVQLayer(dim, codebook_size, codebook_size//4) if use_simvq_last else TraditionalVQLayer(dim, codebook_size)
    
    def forward(self, x, return_loss=True):
        residual, codes_list, recon_loss = x, [], 0
        for layer in [self.layer1, self.layer2, self.layer3]:
            codes, quant = layer(residual)
            if return_loss: recon_loss += F.mse_loss(residual, quant)
            quant = residual + (quant - residual).detach()
            codes_list.append(codes)
            residual = residual - quant
        return (codes_list, recon_loss) if return_loss else codes_list

def train_subspace_simvq(sub_data, codebook_size=1024, max_train=50000, epochs=50, device='cuda'):
    N, dim = sub_data.shape
    print(f"  Training: N={N}, dim={dim}")
    idx = np.random.choice(N, min(N, max_train), replace=False)
    train_data = torch.from_numpy(sub_data[idx]).float()
    print(f"  Train samples: {len(train_data)}")
    
    model = SimVQResidualQuantizer(dim, codebook_size, use_simvq_last=True)
    if torch.cuda.is_available() and device=='cuda':
        model, train_data = model.cuda(), train_data.cuda()
        print("  Using CUDA")
    else:
        device = 'cpu'
        print("  Using CPU")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    batch_size = 256
    model.train()
    
    for epoch in range(epochs):
        perm = torch.randperm(len(train_data))
        epoch_loss, n_batches = 0, 0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[perm[i:i+batch_size]]
            codes_list, loss = model(batch, return_loss=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if (epoch+1)%10==0 or epoch==0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/n_batches:.6f}")
    
    print(f"  Encoding all {N} samples...")
    model.eval()
    full_data = torch.from_numpy(sub_data).float()
    all_codes = []
    with torch.no_grad():
        for i in range(0, len(full_data), batch_size):
            batch = full_data[i:i+batch_size]
            if device=='cuda': batch = batch.cuda()
            codes_list = model(batch, return_loss=False)
            all_codes.append(torch.stack(codes_list, dim=1).cpu())
    all_codes = torch.cat(all_codes, dim=0).numpy()
    print(f"  Done: {all_codes.shape}")
    return all_codes, model

def compute_metrics(codes):
    N, L = codes.shape
    unique = len(set(tuple(row) for row in codes))
    icr, cr = unique/N, 1-unique/N
    bucket_size, cap, sizes = [], [], []
    for level in range(L):
        lc, K = codes[:,level], int(np.max(codes[:,level]))+1
        counts, active = np.bincount(lc, minlength=K), int(np.sum(np.bincount(lc, minlength=K)>0))
        nz = counts[counts>0]
        bucket_size.append({"level":level+1, "bucket_count":active, "mean":float(np.mean(nz)), "median":float(np.median(nz)), "p25":float(np.percentile(nz,25)), "p75":float(np.percentile(nz,75)), "max":int(np.max(counts))})
        cap.append(active/K)
        sizes.append(K)
    return {"N":N, "L":L, "collision_rate":cr, "unique_paths":unique, "icr":icr, "capacity_cur_list":cap, "codebook_sizes":sizes, "bucket_size":bucket_size}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_npy", required=True)
    p.add_argument("--meta_jsonl", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--codebook_size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", default='cuda')
    a = p.parse_args()
    
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"Loading {a.data_npy}", flush=True)
    data = np.load(a.data_npy).astype(np.float32)
    N, d = data.shape
    print(f"Data: {data.shape}", flush=True)
    
    d_sub = d//2
    sub1, sub2 = data[:,:d_sub], data[:,d_sub:2*d_sub]
    
    print(f"\nSubspace 1/2:", flush=True)
    codes1, model1 = train_subspace_simvq(sub1, a.codebook_size, epochs=a.epochs, device=a.device)
    print(f"\nSubspace 2/2:", flush=True)
    codes2, model2 = train_subspace_simvq(sub2, a.codebook_size, epochs=a.epochs, device=a.device)
    
    full = np.concatenate([codes1, codes2], axis=1)
    print(f"\nFull: {full.shape}", flush=True)
    
    sub1_m, sub2_m, full_m = compute_metrics(codes1), compute_metrics(codes2), compute_metrics(full)
    print(f"\nSubspace1: CR={sub1_m['collision_rate']:.4f}, L3 util={sub1_m['capacity_cur_list'][2]:.2%}")
    print(f"Subspace2: CR={sub2_m['collision_rate']:.4f}, L3 util={sub2_m['capacity_cur_list'][2]:.2%}")
    print(f"Full: CR={full_m['collision_rate']:.4f}, ICR={full_m['icr']:.4f}", flush=True)
    
    print("\nGenerating indices.jsonl...", flush=True)
    meta = []
    if a.meta_jsonl:
        with open(a.meta_jsonl, encoding="utf8") as f: meta = [json.loads(l) for l in f]
    
    with (out/"indices.jsonl").open("w", encoding="utf8") as f:
        for idx, row in enumerate(full):
            s1, s2 = row[:3].tolist(), row[3:].tolist()
            t1 = [f"<a{i+1}_{c}>" for i,c in enumerate(s1)]
            t2 = [f"<b{i+1}_{c}>" for i,c in enumerate(s2)]
            rec = {"indices":row.tolist(), "tokens":t1+t2, "sid":t1+t2, "sid_subspace1":t1, "sid_subspace2":t2}
            if meta: rec.update(meta[idx])
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    
    with (out/"metrics.json").open("w", encoding="utf8") as f:
        json.dump({"subspace1":sub1_m, "subspace2":sub2_m, "full":full_m}, f, ensure_ascii=False, indent=2)
    print("Done!", flush=True)

if __name__ == "__main__": main()

