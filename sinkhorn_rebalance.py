#!/usr/bin/env python3
"""
Sinkhorn 算法对 SID 进行重分配优化

用途：对已有的 indices.jsonl 应用 Sinkhorn 算法，优化某一层的码本利用率和均衡性
适用场景：当某层利用率低或分布不均衡时使用

示例：
  python sinkhorn_rebalance.py \\
    --input results/indices.jsonl \\
    --output_dir results_optimized \\
    --layer 0 \\
    --reg 0.1 \\
    --max_iter 1000
"""
import argparse
import json
import numpy as np
from pathlib import Path


def sinkhorn_knopp(cost_matrix, reg=0.1, max_iter=1000, tol=1e-9):
    """
    Sinkhorn-Knopp 算法求解最优传输
    
    参数：
        cost_matrix: [N, K] 代价矩阵
        reg: entropy regularization 参数，越小越接近精确解（但可能不收敛）
        max_iter: 最大迭代次数
        tol: 收敛容忍度
    
    返回：
        P: [N, K] 传输计划矩阵，P[i,j] 表示样本 i 分配到 code j 的概率
    """
    N, K = cost_matrix.shape
    
    # Gibbs kernel: K = exp(-C / reg)
    kernel = np.exp(-cost_matrix / reg)
    
    # 边际分布（均匀分布）
    a = np.ones(N) / N  # 源分布
    b = np.ones(K) / K  # 目标分布
    
    # Sinkhorn 迭代
    u = np.ones(N)
    v = np.ones(K)
    
    for iteration in range(max_iter):
        u_old = u.copy()
        
        # 更新 u 和 v
        u = a / (kernel @ v)
        v = b / (kernel.T @ u)
        
        # 检查收敛
        if np.max(np.abs(u - u_old)) < tol:
            print(f"  Sinkhorn converged at iteration {iteration+1}")
            break
    
    # 传输计划：P = diag(u) @ K @ diag(v)
    P = u[:, None] * kernel * v[None, :]
    
    return P


def reassign_layer(data, layer_idx, reg=0.1, max_iter=1000):
    """
    对指定层应用 Sinkhorn 重分配
    
    参数：
        data: 列表，每个元素是一个 dict，包含 'indices' 字段
        layer_idx: 要优化的层索引（0-based）
        reg: Sinkhorn 正则化参数
        max_iter: 最大迭代次数
    
    返回：
        data: 更新后的数据
    """
    N = len(data)
    layer_codes = np.array([d['indices'][layer_idx] for d in data])
    K = int(layer_codes.max()) + 1
    
    # 统计当前分布
    counts = np.bincount(layer_codes, minlength=K)
    active = np.sum(counts > 0)
    
    print(f"Layer {layer_idx+1} before Sinkhorn:")
    print(f"  Utilization: {active}/{K} = {active/K:.2%}")
    print(f"  Max bucket: {counts.max()}")
    print(f"  Min bucket (active): {counts[counts>0].min() if active>0 else 0}")
    
    # 构造代价矩阵
    # 策略：让过度使用的 code 代价高，未使用的 code 代价低
    target_count = N / K  # 理想均匀分布
    
    cost_matrix = np.zeros((N, K))
    
    for i in range(N):
        current_code = layer_codes[i]
        current_freq = counts[current_code]
        
        for k in range(K):
            # 基础代价：与当前 code 的频率成正比
            cost_matrix[i, k] = current_freq / target_count
            
            # 调整：鼓励使用未充分利用的 code
            code_freq = counts[k]
            if code_freq < target_count * 0.5:
                # 未充分使用，降低代价
                cost_matrix[i, k] -= 5.0
            elif code_freq > target_count * 1.5:
                # 过度使用，增加代价
                cost_matrix[i, k] += 5.0
    
    # 应用 Sinkhorn
    print(f"Applying Sinkhorn with reg={reg}, max_iter={max_iter}...")
    P = sinkhorn_knopp(cost_matrix, reg=reg, max_iter=max_iter)
    
    # 重新分配：选择概率最大的 code
    new_codes = np.argmax(P, axis=1)
    
    # 统计新分布
    new_counts = np.bincount(new_codes, minlength=K)
    new_active = np.sum(new_counts > 0)
    
    print(f"Layer {layer_idx+1} after Sinkhorn:")
    print(f"  Utilization: {new_active}/{K} = {new_active/K:.2%}")
    print(f"  Max bucket: {new_counts.max()}")
    print(f"  Min bucket (active): {new_counts[new_counts>0].min() if new_active>0 else 0}")
    
    # 更新数据
    for i, d in enumerate(data):
        d['indices'][layer_idx] = int(new_codes[i])
        # 重新生成 tokens
        tokens = [f"<l{j+1}_{c}>" for j, c in enumerate(d['indices'])]
        d['tokens'] = tokens
        d['sid'] = tokens
    
    return data


def compute_metrics(data):
    """计算完整的 metrics"""
    indices_list = [tuple(d['indices']) for d in data]
    N = len(indices_list)
    L = len(data[0]['indices'])
    
    unique_paths = len(set(indices_list))
    icr = unique_paths / N
    cr = 1 - icr
    
    bucket_size = []
    for level in range(L):
        codes = [d['indices'][level] for d in data]
        K = max(codes) + 1
        counts = np.bincount(codes, minlength=K)
        active = int(np.sum(counts > 0))
        nz = counts[counts > 0]
        
        bucket_size.append({
            "level": level + 1,
            "bucket_count": active,
            "mean": float(np.mean(nz)),
            "median": float(np.median(nz)),
            "max": int(np.max(counts))
        })
    
    return {
        "N": N,
        "L": L,
        "collision_rate": cr,
        "unique_paths": unique_paths,
        "icr": icr,
        "bucket_size": bucket_size
    }


def main():
    parser = argparse.ArgumentParser(description="Sinkhorn rebalancing for SID")
    parser.add_argument("--input", required=True, help="Input indices.jsonl")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to optimize (0-based)")
    parser.add_argument("--reg", type=float, default=0.1, help="Sinkhorn regularization parameter")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max Sinkhorn iterations")
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"Loading {args.input}...")
    data = []
    with open(args.input, 'r', encoding='utf8') as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} samples")
    
    # 应用 Sinkhorn
    data = reassign_layer(data, args.layer, args.reg, args.max_iter)
    
    # 计算 metrics
    print("\nComputing final metrics...")
    metrics = compute_metrics(data)
    
    print(f"\nFinal results:")
    print(f"  CR: {metrics['collision_rate']:.6f}")
    print(f"  ICR: {metrics['icr']:.6f}")
    print(f"  Unique paths: {metrics['unique_paths']:,} / {metrics['N']:,}")
    
    print("\nPer-layer stats:")
    for b in metrics['bucket_size']:
        print(f"  L{b['level']}: active={b['bucket_count']}, max={b['max']}")
    
    # 保存结果
    print(f"\nSaving to {output_dir}...")
    with (output_dir / "indices.jsonl").open("w", encoding="utf8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    
    with (output_dir / "metrics.json").open("w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main()
