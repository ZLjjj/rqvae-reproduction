# Sinkhorn 重分配脚本使用说明

## 脚本功能

`sinkhorn_rebalance.py` 使用 Sinkhorn-Knopp 算法对 SID（Semantic ID）的某一层进行重分配，以提升码本利用率和分布均衡性。

## 适用场景

- 某一层的码本利用率低（例如只用了 50%）
- 某一层分布不均衡（max_bucket 过大）
- 已有 RQVAE/RQ 训练好的 indices.jsonl，想进一步优化

## 使用方法

### 基本用法

```bash
python sinkhorn_rebalance.py \
  --input results/indices.jsonl \
  --output_dir results_optimized \
  --layer 0 \
  --reg 0.1 \
  --max_iter 1000
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `--input` | str | 必填 | 输入的 indices.jsonl 文件路径 |
| `--output_dir` | str | 必填 | 输出目录 |
| `--layer` | int | 0 | 要优化的层索引（0-based，0表示第1层） |
| `--reg` | float | 0.1 | Sinkhorn 正则化参数 |
| `--max_iter` | int | 1000 | Sinkhorn 最大迭代次数 |

### `reg` 参数调优指南

`reg` 是 entropy regularization 参数，控制最优传输的"精确度"：

| reg 值 | 特点 | 适用场景 | 风险 |
|---|---|---|---|
| **0.5-1.0** | 宽松约束 | 初步优化，保守调整 | 效果有限 |
| **0.1-0.5** | 中等约束 | 一般优化 | **推荐** |
| **0.01-0.1** | 严格约束 | 追求极致均衡 | 容易坍缩 |
| **<0.01** | 极严格 | 理论研究 | **不推荐** |

**建议：**
- 首次优化：`reg=0.1`
- 如果效果不明显：逐步降低到 `0.05`
- 如果出现坍缩：提高到 `0.5`

## 工作原理

### Sinkhorn-Knopp 算法

Sinkhorn 算法通过迭代求解最优传输问题：

```
min_{P} <C, P> + reg * KL(P || ab^T)
s.t. P @ 1_K = a, P^T @ 1_N = b
```

其中：
- `C`: 代价矩阵（[N, K]）
- `P`: 传输计划（[N, K]）
- `a`: 源分布（均匀）
- `b`: 目标分布（均匀）
- `reg`: 熵正则化参数

### 代价矩阵设计

脚本使用以下策略构造代价矩阵：

1. **基础代价**：与当前 code 的使用频率成正比
2. **鼓励未充分使用的 code**：频率 < 50% 目标 → 代价 -5.0
3. **惩罚过度使用的 code**：频率 > 150% 目标 → 代价 +5.0

最终通过 Sinkhorn 迭代找到平衡源分布和目标分布的最优传输计划。

## 使用案例

### 案例1：优化 L5 层利用率

假设 L5 层利用率只有 47.84%，max_bucket=18422：

```bash
python sinkhorn_rebalance.py \
  --input results_three_way/exp_task3_l1_sinkhorn/indices/indices.jsonl \
  --output_dir results_three_way/exp_task3_l5_optimized \
  --layer 4 \
  --reg 0.1 \
  --max_iter 2000
```

### 案例2：优化 L2 层均衡性

假设 L2 层 max_bucket=1178 过大：

```bash
python sinkhorn_rebalance.py \
  --input results/indices.jsonl \
  --output_dir results_l2_balanced \
  --layer 1 \
  --reg 0.05 \
  --max_iter 1000
```

## 输出

脚本会在 `output_dir` 生成：

1. `indices.jsonl`：重分配后的 SID
2. `metrics.json`：完整的统计指标

## 注意事项

### ⚠️ 什么时候不应该用 Sinkhorn？

1. **已经100%利用率且均衡**：如 L1-L4 全部100%，max_bucket<300
2. **硬编码的业务层**：如 L5 包含 saletype、年份等业务字段，重分配会破坏语义
3. **训练效果已经很好**：CR<0.1%，继续优化风险大于收益

### ⚠️ 潜在风险

1. **坍缩风险**：`reg` 太小可能导致所有样本映射到少数 code
2. **语义破坏**：重分配可能打散原有的语义聚类
3. **重构质量下降**：强行均衡可能增大 quantization error

### ✅ 最佳实践

1. **先分析再优化**：用 `analyze_current.py` 查看各层利用率
2. **逐层优化**：一次只优化一层，观察效果
3. **保守参数**：首次尝试用 `reg=0.1`
4. **保留备份**：优化前备份原始 indices.jsonl
5. **验证结果**：检查 CR/ICR 是否改善

## 实验总结

根据任务3的优化尝试：

| 层 | 优化前 | 优化后 | 建议 |
|---|---|---|---|
| L1 | 100%利用率 | 坍缩 | ❌ 不优化 |
| L2-L4 | 100%利用率 | - | ✅ 保持现状 |
| L5 | 47.84%利用率 | - | ⚠️ 硬编码层，不应优化 |

**结论：RQVAE + L1 Sinkhorn (CR=0.09%) 已是最优，无需进一步优化。**

## 脚本位置

- 主脚本：`C:\Users\dszlj\Desktop\gensearchrec-main-0901\sinkhorn_rebalance.py`
- 分析脚本：`C:\Users\dszlj\Desktop\gensearchrec-main-0901\analyze_current.py`
- 优化总结：`C:\Users\dszlj\Desktop\gensearchrec-main-0901\results_three_way\sinkhorn_optimization_summary.md`

