import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def compute_weighted_cosine(embedding_file, indices_file, num_layers=4, max_bucket_sample=500):
    """计算 L1-L4 的 weighted cosine 相似度（大bucket采样）"""
    print(f"处理: {indices_file.parent.parent.name}/{indices_file.parent.name}", flush=True)
    
    embeddings = np.load(embedding_file).astype(np.float32)
    N = len(embeddings)
    
    # 预归一化所有 embedding
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    with open(indices_file, 'r', encoding='utf-8') as f:
        all_codes = []
        for line in f:
            item = json.loads(line)
            codes = item.get('indices') or item.get('codes_5layer') or item.get('codes') or []
            all_codes.append(codes)
    
    results = {}
    rng = np.random.default_rng(42)
    
    for layer in range(1, num_layers + 1):
        prefix_groups = defaultdict(list)
        for idx, codes in enumerate(all_codes):
            if len(codes) >= layer:
                prefix = tuple(codes[:layer])
                prefix_groups[prefix].append(idx)
        
        total_weighted = 0.0
        total_weight = 0
        valid_buckets = 0
        
        for prefix, sample_indices in prefix_groups.items():
            bsize = len(sample_indices)
            if bsize < 2:
                continue
            
            # 大bucket采样
            if bsize > max_bucket_sample:
                sampled = rng.choice(sample_indices, max_bucket_sample, replace=False)
            else:
                sampled = sample_indices
            
            bucket_embeds = embeddings[sampled]  # 已归一化
            sim_matrix = bucket_embeds @ bucket_embeds.T
            
            # 上三角平均
            n = len(sampled)
            iu = np.triu_indices(n, k=1)
            bucket_sim = sim_matrix[iu].mean()
            
            # 用原始 bucket 大小加权
            total_weighted += bucket_sim * bsize
            total_weight += bsize
            valid_buckets += 1
        
        if total_weight == 0:
            results[f'L{layer}'] = 0.0
            print(f"    L{layer}: 无有效bucket", flush=True)
        else:
            weighted_sim = total_weighted / total_weight
            results[f'L{layer}'] = float(weighted_sim)
            print(f"    L{layer}: {weighted_sim:.6f} ({valid_buckets} valid buckets)", flush=True)
    
    return results

experiments = {
    '任务3+L1 Sinkhorn': ('results_three_way/exp_task3_l1_sinkhorn/indices/indices.jsonl', '3_text_format_embedding.npy'),
    '任务2+L1 Sinkhorn': ('results_three_way/l1_screen_task2_eps003/indices/indices.jsonl', '2_meta_full_video.npy'),
    'RQ-OPQ 任务3': ('results_three_way/rqopq_task3/indices.jsonl', '3_text_format_embedding.npy'),
    'RQ-OPQ 任务2': ('results_three_way/rqopq_task2/indices.jsonl', '2_meta_full_video.npy'),
    'RQ-KMeans 任务3': ('results_three_way/rqkmeans_task3/indices.jsonl', '3_text_format_embedding.npy'),
    'RQ-KMeans 任务2': ('results_three_way/rqkmeans_task2/indices.jsonl', '2_meta_full_video.npy'),
    '任务3原始': ('results_three_way/video_v3/indices/indices.jsonl', '3_text_format_embedding.npy'),
    '任务2原始': ('results_three_way/same_source_video/indices/indices.jsonl', '2_meta_full_video.npy'),
}

base_dir = Path('C:/Users/dszlj/Desktop/gensearchrec-main-0901')
all_results = {}

for exp_name, (idx_path, emb_path) in experiments.items():
    indices_file = base_dir / idx_path
    embedding_file = base_dir / emb_path
    if not indices_file.exists() or not embedding_file.exists():
        print(f"跳过 {exp_name}: 文件不存在", flush=True)
        continue
    try:
        all_results[exp_name] = compute_weighted_cosine(embedding_file, indices_file)
    except Exception as e:
        print(f"错误 {exp_name}: {e}", flush=True)

print("\n" + "="*80, flush=True)
print("所有实验 Weighted Cosine 汇总", flush=True)
print("="*80, flush=True)
for exp_name, r in all_results.items():
    print(f"{exp_name}: L1={r['L1']:.4f} L2={r['L2']:.4f} L3={r['L3']:.4f} L4={r['L4']:.4f}", flush=True)

with open(base_dir / 'weighted_cosine_all_experiments.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print("\n已保存", flush=True)

