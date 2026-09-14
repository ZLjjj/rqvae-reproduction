import json
import numpy as np

print("Loading indices.jsonl...")
data = []
with open('results_three_way/exp_task3_l1_sinkhorn/indices/indices.jsonl', 'r', encoding='utf8') as f:
    for line in f:
        data.append(json.loads(line))

N = len(data)
print(f"Loaded {N} samples")

# 提取 indices
indices_list = [tuple(d['indices']) for d in data]
L = len(data[0]['indices'])
print(f"Layers: {L}")

# 计算唯一路径
unique_paths = len(set(indices_list))
icr = unique_paths / N
cr = 1 - icr

print(f"\nOverall:")
print(f"  Unique paths: {unique_paths:,} / {N:,}")
print(f"  CR: {cr:.6f}")
print(f"  ICR: {icr:.6f}")

# 逐层统计
print(f"\nPer-layer utilization:")
for level in range(L):
    codes = [d['indices'][level] for d in data]
    K = max(codes) + 1
    counts = np.bincount(codes, minlength=K)
    active = int(np.sum(counts > 0))
    util = active / K
    max_bucket = int(counts.max())
    min_bucket = int(counts[counts > 0].min()) if active > 0 else 0
    mean_bucket = float(np.mean(counts[counts > 0])) if active > 0 else 0
    
    print(f"  L{level+1}: {active}/{K} = {util:.2%}, max={max_bucket}, min={min_bucket}, mean={mean_bucket:.1f}")

