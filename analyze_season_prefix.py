import json
from collections import defaultdict
import pandas as pd

# 读取indices.jsonl文件
indices_path = "results_rqvae_l1_sinkhorn/task3_correct/indices/indices.jsonl"

print("=== 加载数据 ===")
data = []
with open(indices_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

print(f"总数据条数: {len(data)}")

# 按meta_main_name分组，找出有多季的资源
print("\n=== 查找有多季的资源 ===")
resource_seasons = defaultdict(list)

for item in data:
    main_name = item.get('meta_main_name', '')
    season = item.get('season', '0')
    
    # 只统计有明确季数信息且不是0的
    if main_name and season and season != '0':
        resource_seasons[main_name].append({
            'season': season,
            'indices': item['indices'],
            'sid': item['sid'],
            'id': item['id'],
            'publish_year': item.get('publish_year', ''),
            'video_type': item.get('video_type', '')
        })

# 筛选出有多季的资源
multi_season_resources = {k: v for k, v in resource_seasons.items() if len(v) > 1}

print(f"找到 {len(multi_season_resources)} 个有多季的资源")

# 分析prefix相似度
print("\n=== 分析同一资源不同季的SID Prefix ===")

results = []
for resource_name, seasons in sorted(multi_season_resources.items()):
    seasons_sorted = sorted(seasons, key=lambda x: x['season'])
    
    # 比较每一层的indices
    prefix_analysis = {
        'resource': resource_name,
        'total_seasons': len(seasons_sorted),
        'seasons_info': []
    }
    
    for season_info in seasons_sorted:
        prefix_analysis['seasons_info'].append({
            'season': season_info['season'],
            'year': season_info['publish_year'],
            'indices': season_info['indices'],
            'sid': ' '.join(season_info['sid'])
        })
    
    # 计算各层的一致性
    layer_consistency = []
    for layer_idx in range(5):  # 5层
        values = [s['indices'][layer_idx] for s in seasons_sorted]
        unique_values = len(set(values))
        is_same = (unique_values == 1)
        layer_consistency.append({
            'layer': f'L{layer_idx+1}',
            'all_same': is_same,
            'unique_count': unique_values,
            'values': values
        })
    
    prefix_analysis['layer_consistency'] = layer_consistency
    results.append(prefix_analysis)

# 输出前10个案例详细分析
print("\n" + "="*80)
print("详细案例分析（前10个多季资源）")
print("="*80)

for i, result in enumerate(results[:10], 1):
    print(f"\n【案例 {i}】资源: {result['resource']}")
    print(f"季数: {result['total_seasons']}")
    
    for season_info in result['seasons_info']:
        print(f"  第{season_info['season']}季 ({season_info['year']}年): {season_info['sid']}")
    
    print("\n  各层一致性分析:")
    for layer in result['layer_consistency']:
        status = "✅ 相同" if layer['all_same'] else f"❌ 不同({layer['unique_count']}个值)"
        print(f"    {layer['layer']}: {status} - 值: {layer['values']}")

# 统计总体一致性
print("\n" + "="*80)
print("总体统计")
print("="*80)

layer_stats = defaultdict(lambda: {'same': 0, 'different': 0})

for result in results:
    for layer in result['layer_consistency']:
        if layer['all_same']:
            layer_stats[layer['layer']]['same'] += 1
        else:
            layer_stats[layer['layer']]['different'] += 1

print(f"\n多季资源总数: {len(results)}")
print("\n各层前缀一致性统计:")
for layer in ['L1', 'L2', 'L3', 'L4', 'L5']:
    stats = layer_stats[layer]
    total = stats['same'] + stats['different']
    same_pct = (stats['same'] / total * 100) if total > 0 else 0
    print(f"  {layer}: 相同={stats['same']} ({same_pct:.1f}%), 不同={stats['different']} ({100-same_pct:.1f}%)")

# 找出完全一致和完全不一致的案例
print("\n" + "="*80)
print("特殊案例")
print("="*80)

all_same = [r for r in results if all(l['all_same'] for l in r['layer_consistency'])]
all_different = [r for r in results if all(not l['all_same'] for l in r['layer_consistency'])]

print(f"\n✅ 所有层都相同的资源: {len(all_same)}")
if len(all_same) > 0:
    print("示例:")
    for r in all_same[:3]:
        print(f"  - {r['resource']} ({r['total_seasons']}季)")

print(f"\n❌ 所有层都不同的资源: {len(all_different)}")
if len(all_different) > 0:
    print("示例:")
    for r in all_different[:3]:
        print(f"  - {r['resource']} ({r['total_seasons']}季)")

# 分析前4层的一致性（排除L5年份层）
prefix_l1_l4_same = [r for r in results if all(l['all_same'] for l in r['layer_consistency'][:4])]
print(f"\n🎯 前4层(L1-L4)完全相同的资源: {len(prefix_l1_l4_same)} ({len(prefix_l1_l4_same)/len(results)*100:.1f}%)")

