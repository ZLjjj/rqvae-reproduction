import json
from collections import defaultdict

print("=== 任务3+L1 Sinkhorn：同一资源不同季的Prefix一致性分析 ===\n")

# 读取数据
indices_path = "results_rqvae_l1_sinkhorn/task3_correct/indices/indices.jsonl"

data = []
with open(indices_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

print(f"总数据量: {len(data)} 条\n")

# 按资源名+季数分组
resource_seasons = defaultdict(list)

for item in data:
    name = item.get('meta_main_name', '')
    season = item.get('season', '0')
    
    if name and season and season != '0':
        try:
            season_num = int(season)
            resource_seasons[name].append({
                'season': season_num,
                'year': item.get('publish_year', ''),
                'indices': item['indices'],
                'sid': item['sid']
            })
        except:
            pass

# 筛选真正的多季作品（有连续季数的）
multi_season = {}
for name, seasons in resource_seasons.items():
    if len(seasons) < 2:
        continue
    
    # 去重：同一季只保留一个
    unique_seasons = {}
    for s in seasons:
        if s['season'] not in unique_seasons:
            unique_seasons[s['season']] = s
    
    seasons_list = list(unique_seasons.values())
    if len(seasons_list) < 2:
        continue
    
    # 检查是否有连续季数
    season_nums = sorted([s['season'] for s in seasons_list])
    has_sequential = False
    for i in range(len(season_nums) - 1):
        if season_nums[i+1] - season_nums[i] == 1:
            has_sequential = True
            break
    
    if has_sequential:
        multi_season[name] = sorted(seasons_list, key=lambda x: x['season'])

print(f"找到真正的多季续集: {len(multi_season)} 个\n")

# 统计各层一致性
stats = {
    'L1': {'same': 0, 'diff': 0},
    'L2': {'same': 0, 'diff': 0},
    'L3': {'same': 0, 'diff': 0},
    'L4': {'same': 0, 'diff': 0},
    'L5': {'same': 0, 'diff': 0}
}

l1_l4_all_same = 0
all_layers_same = 0
all_layers_diff = 0

print("=" * 80)
print("前10个案例详细分析")
print("=" * 80)

for idx, (name, seasons) in enumerate(list(multi_season.items())[:10], 1):
    print(f"\n【案例 {idx}】{name}")
    print(f"季数: {len(seasons)} ({', '.join([str(s['season']) for s in seasons])}季)")
    
    for s in seasons:
        sid_str = ' '.join(s['sid'])
        print(f"  第{s['season']}季 ({s['year']}年): {sid_str}")
    
    print("\n  各层一致性:")
    layer_results = []
    for layer_idx in range(5):
        values = [s['indices'][layer_idx] for s in seasons]
        is_same = len(set(values)) == 1
        layer_results.append(is_same)
        
        status = "✅ 相同" if is_same else f"❌ 不同({len(set(values))}个值)"
        print(f"    L{layer_idx+1}: {status} - 值: {values}")

# 统计所有资源
for name, seasons in multi_season.items():
    layer_results = []
    for layer_idx in range(5):
        values = [s['indices'][layer_idx] for s in seasons]
        is_same = len(set(values)) == 1
        layer_results.append(is_same)
        
        layer_name = f'L{layer_idx+1}'
        if is_same:
            stats[layer_name]['same'] += 1
        else:
            stats[layer_name]['diff'] += 1
    
    # 前4层
    if all(layer_results[:4]):
        l1_l4_all_same += 1
    
    # 所有层
    if all(layer_results):
        all_layers_same += 1
    elif not any(layer_results):
        all_layers_diff += 1

print("\n" + "=" * 80)
print("总体统计")
print("=" * 80)
print(f"\n多季资源总数: {len(multi_season)}\n")

print("各层Prefix一致性:")
for layer in ['L1', 'L2', 'L3', 'L4', 'L5']:
    same = stats[layer]['same']
    diff = stats[layer]['diff']
    total = same + diff
    pct = (same / total * 100) if total > 0 else 0
    print(f"  {layer}: 相同={same} ({pct:.1f}%), 不同={diff} ({100-pct:.1f}%)")

print(f"\n特殊情况:")
pct1 = (l1_l4_all_same / len(multi_season) * 100) if len(multi_season) > 0 else 0
print(f"  🎯 前4层(L1-L4)完全相同: {l1_l4_all_same} ({pct1:.1f}%)")

pct2 = (all_layers_same / len(multi_season) * 100) if len(multi_season) > 0 else 0
print(f"  ✅ 所有5层都相同: {all_layers_same} ({pct2:.1f}%)")

pct3 = (all_layers_diff / len(multi_season) * 100) if len(multi_season) > 0 else 0
print(f"  ❌ 所有5层都不同: {all_layers_diff} ({pct3:.1f}%)")
