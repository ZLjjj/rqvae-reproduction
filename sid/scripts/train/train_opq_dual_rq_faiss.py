#!/usr/bin/env python3
"""
OPQ + Dual Independent RQ

1. 用 OPQ 旋转输入到 m 个子空间
2. 在每个子空间内独立训练 RQ (多层残差量化)
3. 输出 m 套独立的 SID，每套包含 rq_levels 层
"""
import argparse
import json
from pathlib import Path

import faiss
import numpy as np


def train_opq(data, m=2):
    """训练 OPQ 旋转矩阵"""
    d = data.shape[1]
    print(f"Training OPQ with {m} subspaces, d={d}")
    opq = faiss.OPQMatrix(d, m)
    opq.train(data)
    print("OPQ training done")
    return opq


def train_rq_on_subspace(sub_data, rq_levels=3, codebook_size=1024):
    """在单个子空间上训练 RQ"""
    d_sub = sub_data.shape[1]
    nbits = int(np.log2(codebook_size))
    print(f"  Training RQ: levels={rq_levels}, d_sub={d_sub}, nbits={nbits}")
    
    rq = faiss.ResidualQuantizer(d_sub, rq_levels, nbits)
    rq.train(sub_data)
    
    codes = rq.compute_codes(sub_data)
    print(f"  RQ training done, codes shape={codes.shape}")
    return rq, codes


def compute_metrics(codes, emb=None):
    """计算 CR/ICR/bucket 指标"""
    N, L = codes.shape
    
    # 转成 tuple 用于 unique 统计
    paths = [tuple(row) for row in codes]
    unique_paths = len(set(paths))
    
    icr = unique_paths / N
    cr = 1 - icr
    
    # 逐层统计
    bucket_size = []
    capacity_cur_list = []
    codebook_sizes = []
    
    for level in range(L):
        layer_codes = codes[:, level]
        unique_codes = np.unique(layer_codes)
        K = int(np.max(layer_codes)) + 1  # 推断 codebook 大小
        
        counts = np.bincount(layer_codes, minlength=K)
        active = np.sum(counts > 0)
        
        bucket_size.append({
            "level": level + 1,
            "bucket_count": int(active),
            "mean": float(np.mean(counts[counts > 0])) if active > 0 else 0,
            "median": float(np.median(counts[counts > 0])) if active > 0 else 0,
            "p25": float(np.percentile(counts[counts > 0], 25)) if active > 0 else 0,
            "p75": float(np.percentile(counts[counts > 0], 75)) if active > 0 else 0,
            "max": int(np.max(counts))
        })
        
        capacity_cur_list.append(active / K)
        codebook_sizes.append(K)
    
    metrics = {
        "N": N,
        "L": L,
        "collision_rate": cr,
        "unique_paths": unique_paths,
        "icr": icr,
        "total_cur": icr,
        "capacity_cur_list": capacity_cur_list,
        "codebook_sizes": codebook_sizes,
        "bucket_size": bucket_size
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="OPQ + Dual Independent RQ")
    parser.add_argument("--data_npy", required=True)
    parser.add_argument("--meta_jsonl", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_subspaces", type=int, default=2)
    parser.add_argument("--rq_levels", type=int, default=3, help="RQ levels per subspace")
    parser.add_argument("--codebook_size", type=int, default=1024)
    
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"Loading {args.data_npy}")
    data = np.load(args.data_npy)
    print(f"Data shape: {data.shape}")
    
    N, d = data.shape
    m = args.num_subspaces
    
    # Step 1: 训练 OPQ
    opq = train_opq(data, m=m)
    
    # Step 2: 旋转数据
    print("Applying OPQ rotation")
    data_rotated = opq.apply_py(data)
    
    # Step 3: 分成 m 个子空间
    d_sub = d // m
    subspaces = [data_rotated[:, i*d_sub:(i+1)*d_sub] for i in range(m)]
    
    print(f"Split into {m} subspaces, each d_sub={d_sub}")
    
    # Step 4: 每个子空间独立训练 RQ
    rq_list = []
    codes_list = []
    
    for i, sub_data in enumerate(subspaces):
        print(f"\nSubspace {i+1}/{m}:")
        rq, codes = train_rq_on_subspace(sub_data, args.rq_levels, args.codebook_size)
        rq_list.append(rq)
        codes_list.append(codes)
    
    # Step 5: 合并成完整 codes
    full_codes = np.concatenate(codes_list, axis=1)  # [N, m * rq_levels]
    print(f"\nFull codes shape: {full_codes.shape}")
    
    # Step 6: 计算指标
    print("\nComputing metrics...")
    
    # 每个子空间的独立指标
    subspace_metrics = []
    for i, codes in enumerate(codes_list):
        print(f"  Subspace {i+1} metrics:")
        metrics = compute_metrics(codes)
        print(f"    CR={metrics['collision_rate']:.4f}, ICR={metrics['icr']:.4f}")
        subspace_metrics.append(metrics)
    
    # 完整 6 层的指标
    print("  Full 6-layer metrics:")
    full_metrics = compute_metrics(full_codes)
    print(f"    CR={full_metrics['collision_rate']:.4f}, ICR={full_metrics['icr']:.4f}")
    
    # Step 7: 生成 indices.jsonl
    print("\nGenerating indices.jsonl...")
    
    # 加载 meta 如果提供
    meta_list = []
    if args.meta_jsonl:
        with open(args.meta_jsonl, encoding='utf8') as f:
            for line in f:
                meta_list.append(json.loads(line))
    
    with (out_dir / "indices.jsonl").open("w", encoding="utf8") as f:
        for idx, row in enumerate(full_codes):
            # 分成两个子空间的 SID
            sub1_codes = row[:args.rq_levels].tolist()
            sub2_codes = row[args.rq_levels:].tolist()
            
            # 生成 token
            sub1_tokens = [f"<a{i+1}_{c}>" for i, c in enumerate(sub1_codes)]
            sub2_tokens = [f"<b{i+1}_{c}>" for i, c in enumerate(sub2_codes)]
            all_tokens = sub1_tokens + sub2_tokens
            
            record = {
                "indices": row.tolist(),
                "tokens": all_tokens,
                "sid": all_tokens,
                "sid_subspace1": sub1_tokens,
                "sid_subspace2": sub2_tokens
            }
            
            # 合并 meta
            if meta_list:
                record.update(meta_list[idx])
            
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Saved {N} records to {out_dir / 'indices.jsonl'}")
    
    # Step 8: 保存 metrics
    all_metrics = {
        "subspace1": subspace_metrics[0],
        "subspace2": subspace_metrics[1] if len(subspace_metrics) > 1 else None,
        "full": full_metrics
    }
    
    with (out_dir / "metrics.json").open("w", encoding="utf8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    
    print(f"Saved metrics to {out_dir / 'metrics.json'}")
    print("\nDone!")


if __name__ == "__main__":
    main()
