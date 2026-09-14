import re

# 读取原始文件
with open('table_with_cosine.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 更新数据映射（方法名 -> 新的 weighted cosine 值）
updates = {
    '任务3+L1+L4 Sinkhorn': 'L1: 0.664 / L2: 0.755 / L3: 0.872 / L4: 0.920',
    '任务2+L1+L4 Sinkhorn': 'L1: 0.596 / L2: 0.695 / L3: 0.866 / L4: 0.940',
    'RQ-OPQ': 'L1: 0.692 / L2: 0.786 / L3: 0.885 / L4: 0.937',
    'RQ-KMeans': 'L1: 0.692 / L2: 0.786 / L3: 0.885 / L4: 0.934',
    '任务3原始': 'L1: 0.621 / L2: 0.727 / L3: 0.838 / L4: 0.916',
    '任务2原始': 'L1: 0.545 / L2: 0.660 / L3: 0.822 / L4: 0.926'
}

# 处理每一行
lines = content.split('\n')
updated_lines = []
update_count = 0

for line in lines:
    updated = False
    for method, new_value in updates.items():
        if method in line and '|' in line:
            # 这是表格行，需要更新 weighted cosine 列（第8列，索引7）
            parts = line.split('|')
            if len(parts) >= 11:  # 确保有足够的列
                # 找到 weighted cosine 列并更新
                # 列顺序: 0(空) 1(方法) 2(参数) 3(数据) 4(ICR) 5(CR) 6(L1利用率) 7(L1 max_bucket) 8(weighted cosine) ...
                parts[8] = f' {new_value} '
                line = '|'.join(parts)
                updated = True
                update_count += 1
                print(f'✓ 更新: {method}')
                break
    
    updated_lines.append(line)

# 保存更新后的文件
updated_content = '\n'.join(updated_lines)
with open('table_with_cosine_updated.md', 'w', encoding='utf-8') as f:
    f.write(updated_content)

print(f'\n总共更新了 {update_count} 行')
print(f'已保存到: table_with_cosine_updated.md')
