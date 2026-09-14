import json

k2 = json.load(open('results_three_way/rqkmeans_task2/metrics.json'))
k3 = json.load(open('results_three_way/rqkmeans_task3/metrics.json'))
o2 = json.load(open('results_three_way/rqopq_task2/metrics.json'))
o3 = json.load(open('results_three_way/rqopq_task3/metrics.json'))

print("K2 (任务2 RQ-KMeans):")
print(f"  4-layer CR: {k2['before_balance']['metrics_4layer']['collision_rate']:.4f}")
print(f"  5-layer CR: {k2['before_balance']['metrics_5layer']['collision_rate']:.4f}")
print(f"  L1 utilization: {k2['before_balance']['metrics_4layer']['capacity_cur_list'][0]:.4f}")
print(f"  L1 max bucket: {k2['before_balance']['metrics_4layer']['bucket_size'][0]['max']}")
print()

print("K3 (任务3 RQ-KMeans):")
print(f"  4-layer CR: {k3['before_balance']['metrics_4layer']['collision_rate']:.4f}")
print(f"  5-layer CR: {k3['before_balance']['metrics_5layer']['collision_rate']:.4f}")
print(f"  L1 utilization: {k3['before_balance']['metrics_4layer']['capacity_cur_list'][0]:.4f}")
print(f"  L1 max bucket: {k3['before_balance']['metrics_4layer']['bucket_size'][0]['max']}")
print()

print("O2 (任务2 RQ-OPQ):")
print(f"  5-layer CR: {o2['before_balance']['collision_rate']:.4f}")
print(f"  L1 utilization: {o2['before_balance']['capacity_cur_list'][0]:.4f}")
print(f"  L1 max bucket: {o2['before_balance']['bucket_size'][0]['max']}")
print()

print("O3 (任务3 RQ-OPQ):")
print(f"  5-layer CR: {o3['before_balance']['collision_rate']:.4f}")
print(f"  L1 utilization: {o3['before_balance']['capacity_cur_list'][0]:.4f}")
print(f"  L1 max bucket: {o3['before_balance']['bucket_size'][0]['max']}")
