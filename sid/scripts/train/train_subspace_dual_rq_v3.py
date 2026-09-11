#!/usr/bin/env python3
"""子空间独立 RQ (采样训练版)"""
import argparse
import json
from pathlib import Path
import faiss
import numpy as np


def train_rq_on_subspace(sub_data, rq_levels=3, codebook_size=1024, max_train=50000):
    d_sub = sub_data.shape[1]
    N = sub_data.shape[0]
    nbits = int(np.log2(codebook_size))
    print(f"  RQ: levels={rq_levels}, d_sub={d_sub}, nbits={nbits}", flush=True)
    if N > max_train:
        idx = np.random.choice(N, max_train, replace=False)
        train_data = np.ascontiguousarray(sub_data[idx])
        print(f"  Sampling {max_train}/{N} for training", flush=True)
    else:
        train_data = sub_data
    rq = faiss.ResidualQuantizer(d_sub, rq_levels, nbits)
    rq.train(train_data)
    print(f"  Encoding all {N} samples...", flush=True)
    codes = rq.compute_codes(sub_data)
    return rq, codes


def unpack_codes(codes, rq_levels, nbits):
    N = codes.shape[0]
    out = np.zeros((N, rq_levels), dtype=np.int64)
    nbytes = codes.shape[1]
    for r in range(N):
        bits = 0
        for b in codes[r]:
            bits = (bits << 8) | int(b)
        shift = nbytes * 8
        for l in range(rq_levels):
            shift -= nbits
            out[r, l] = (bits >> shift) & ((1 << nbits) - 1)
    return out


def compute_metrics(codes):
    N, L = codes.shape
    paths = [tuple(row) for row in codes]
    unique = len(set(paths))
    icr = unique / N
    cr = 1 - icr
    bucket_size = []
    cap = []
    sizes = []
    for level in range(L):
        lc = codes[:, level]
        K = int(np.max(lc)) + 1
        counts = np.bincount(lc, minlength=K)
        active = int(np.sum(counts > 0))
        nz = counts[counts > 0]
        bucket_size.append({"level": level+1, "bucket_count": active, "mean": float(np.mean(nz)), "median": float(np.median(nz)), "p25": float(np.percentile(nz,25)), "p75": float(np.percentile(nz,75)), "max": int(np.max(counts))})
        cap.append(active / K)
        sizes.append(K)
    return {"N": N, "L": L, "collision_rate": cr, "unique_paths": unique, "icr": icr, "capacity_cur_list": cap, "codebook_sizes": sizes, "bucket_size": bucket_size}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_npy", required=True)
    p.add_argument("--meta_jsonl", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_subspaces", type=int, default=2)
    p.add_argument("--rq_levels", type=int, default=3)
    p.add_argument("--codebook_size", type=int, default=1024)
    a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"Loading {a.data_npy}", flush=True)
    data = np.ascontiguousarray(np.load(a.data_npy).astype(np.float32))
    N, d = data.shape
    print(f"Data: {data.shape}", flush=True)
    m = a.num_subspaces
    d_sub = d // m
    nbits = int(np.log2(a.codebook_size))
    codes_list = []
    for i in range(m):
        sub = np.ascontiguousarray(data[:, i*d_sub:(i+1)*d_sub])
        print(f"Subspace {i+1}/{m}", flush=True)
        rq, packed = train_rq_on_subspace(sub, a.rq_levels, a.codebook_size)
        codes = unpack_codes(packed, a.rq_levels, nbits)
        codes_list.append(codes)
    full = np.concatenate(codes_list, axis=1)
    print(f"Full codes: {full.shape}", flush=True)
    sub_metrics = [compute_metrics(c) for c in codes_list]
    full_metrics = compute_metrics(full)
    print(f"Full CR={full_metrics['collision_rate']:.4f} ICR={full_metrics['icr']:.4f}", flush=True)
    meta = []
    if a.meta_jsonl:
        with open(a.meta_jsonl, encoding="utf8") as f:
            meta = [json.loads(l) for l in f]
    with (out/"indices.jsonl").open("w", encoding="utf8") as f:
        for idx, row in enumerate(full):
            s1 = row[:a.rq_levels].tolist()
            s2 = row[a.rq_levels:].tolist()
            t1 = [f"<a{i+1}_{c}>" for i,c in enumerate(s1)]
            t2 = [f"<b{i+1}_{c}>" for i,c in enumerate(s2)]
            rec = {"indices": row.tolist(), "tokens": t1+t2, "sid": t1+t2, "sid_subspace1": t1, "sid_subspace2": t2}
            if meta:
                rec.update(meta[idx])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (out/"metrics.json").open("w", encoding="utf8") as f:
        json.dump({"subspace1": sub_metrics[0], "subspace2": sub_metrics[1] if len(sub_metrics)>1 else None, "full": full_metrics}, f, ensure_ascii=False, indent=2)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()

