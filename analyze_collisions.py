import json
from collections import defaultdict, Counter
import os

# 定义要分析的实验
experiments = {
    'task2_original': 'results_three_way/same_source_video/indices/indices.jsonl',
    'task3_original': 'results_three_way/video_v3/indices/indices.jsonl',
    'task2_l1_sinkhorn': 'results_three_way/l1_screen_task2_eps003/indices/indices.jsonl',
    'task3_l1_sinkhorn': 'results_three_way/exp_task3_l1_sinkhorn/indices/indices.jsonl',
    'rqopq_task3': 'results_three_way/rqopq_task3/indices.jsonl',
    'rqkmeans_task3': 'results_three_way/rqkmeans_task3/indices.jsonl',
}

def analyze_collisions(filepath):
    """分析单个实验的碰撞情况"""
    if not os.path.exists(filepath):
        return None
    
    sid_to_samples = defaultdict(list)
    total = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            idx = data.get('idx', total)
            tokens = data.get('tokens', [])
            sid = ''.join(tokens)
            
            sid_to_samples[sid].append({
                'idx': idx,
                'indices': data.get('indices', []),
                'meta': data.get('meta_main_name', 'Unknown'),
                'saletype': data.get('saletype', 'Unknown')
            })
            total += 1
    
    # 找出碰撞
    collisions = {sid: samples for sid, samples in sid_to_samples.items() if len(samples) > 1}
    
    return {
        'total': total,
        'unique_sids': len(sid_to_samples),
        'collisions': len(collisions),
        'collision_samples': sum(len(samples) for samples in collisions.values()),
        'collision_details': collisions
    }

# 分析所有实验
print("开始分析各实验的碰撞数据...\n")
results = {}

for name, path in experiments.items():
    print(f"正在分析: {name}")
    result = analyze_collisions(path)
    if result:
        results[name] = result
        icr = result['unique_sids'] / result['total'] * 100
        cr = result['collision_samples'] / result['total'] * 100
        print(f"  总样本数: {result['total']}")
        print(f"  唯一SID: {result['unique_sids']}")
        print(f"  碰撞SID数: {result['collisions']}")
        print(f"  碰撞样本数: {result['collision_samples']}")
        print(f"  ICR: {icr:.2f}%")
        print(f"  CR: {cr:.2f}%")
        print()
    else:
        print(f"  文件不存在: {path}\n")

# 保存汇总结果（不包含详细碰撞数据，文件太大）
summary = {
    name: {
        'total': data['total'],
        'unique_sids': data['unique_sids'],
        'collisions': data['collisions'],
        'collision_samples': data['collision_samples'],
        'ICR': data['unique_sids'] / data['total'] * 100,
        'CR': data['collision_samples'] / data['total'] * 100
    }
    for name, data in results.items()
}

output_summary = 'collision_analysis_summary.json'
with open(output_summary, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"汇总结果已保存到: {output_summary}")

# 为每个实验采样碰撞样本（前10个碰撞）
print("\n正在采样碰撞数据...")
sampled_collisions = {}

for name, data in results.items():
    collision_details = data['collision_details']
    
    # 按碰撞样本数排序，取前10个
    sorted_collisions = sorted(
        collision_details.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:10]
    
    sampled_collisions[name] = [
        {
            'sid': sid,
            'collision_count': len(samples),
            'samples': samples
        }
        for sid, samples in sorted_collisions
    ]

output_samples = 'collision_samples.json'
with open(output_samples, 'w', encoding='utf-8') as f:
    json.dump(sampled_collisions, f, ensure_ascii=False, indent=2)

print(f"采样碰撞数据已保存到: {output_samples}")
print("\n分析完成！")
