# RQVAE + L1 Sinkhorn 实验完整复现指南

本文档提供完整的 RQVAE + L1 Sinkhorn 实验复现方法，可以复现报告中任务2和任务3的结果。

## 实验目标

复现以下结果：

| 任务 | 数据 | CR | ICR |
|---|---|---:|---:|
| 任务2 | 2_meta_full_video.npy | 0.07% | 99.93% |
| 任务3 | 3_text_format_embedding.npy | 0.09% | 99.91% |

## 核心配置

### RQVAE + L1 Sinkhorn 关键参数

```python
{
    "num_emb_list": [1024, 1024, 1024, 1024, 256],  # 5层 codebook
    "sk_epsilons": [0.01, 0.0, 0.0, 0.0, 0.0],     # 只在 L1 用 Sinkhorn
    "sk_iters": 50,                                # Sinkhorn 迭代次数
    "ema_codebook": True,                          # 使用 EMA 更新 codebook
    "ema_decay": 0.99,
    "kmeans_init": True,                           # K-Means 初始化
    "epochs": 40,
    "lr": 0.001
}
```

**关键点：**
- `sk_epsilons[0] = 0.01`：第1层使用 Sinkhorn，温度参数0.01
- 其他层 epsilon=0.0：使用标准的硬分配（argmax）
- EMA codebook：平滑更新，避免坍缩
- K-Means 初始化：更好的起点

## 快速开始

### 方法1：使用一键复现脚本

```bash
# 复现任务2和任务3
python reproduce_rqvae_l1_sinkhorn.py --tasks all

# 只复现任务3
python reproduce_rqvae_l1_sinkhorn.py --tasks task3

# 只训练，不生成 indices
python reproduce_rqvae_l1_sinkhorn.py --tasks task3 --train_only

# 只生成 indices（需要已有 checkpoint）
python reproduce_rqvae_l1_sinkhorn.py --tasks task3 --infer_only
```

### 方法2：手动执行（分步骤）

#### 步骤1：训练 RQVAE 模型

```bash
# 任务3示例
python sid/scripts/train/train_rqvae.py \
  --data_npy 3_text_format_embedding.npy \
  --meta_jsonl 3_meta.jsonl \
  --ckpt_dir results_rqvae_l1_sinkhorn/task3/checkpoints \
  --num_emb_list 1024 1024 1024 1024 256 \
  --e_dim 128 \
  --layers 512 256 128 \
  --sk_epsilons 0.01 0.0 0.0 0.0 0.0 \
  --sk_iters 50 \
  --ema_codebook \
  --ema_decay 0.99 \
  --ema_eps 1e-5 \
  --kmeans_init \
  --kmeans_iters 20 \
  --kmeans_init_max_samples 200000 \
  --epochs 40 \
  --lr 0.001 \
  --batch_size 1024 \
  --eval_step 1 \
  --device cuda
```

#### 步骤2：生成 SID indices

```bash
python sid/src/codebook/generate_indices_rqvae.py \
  --model rqvae \
  --ckpt results_rqvae_l1_sinkhorn/task3/checkpoints/epoch_39.pt \
  --data_npy 3_text_format_embedding.npy \
  --meta_jsonl 3_meta.jsonl \
  --output_dir results_rqvae_l1_sinkhorn/task3/indices \
  --device cuda \
  --batch_size 1024 \
  --use_sk
```

**注意：** `--use_sk` 标志确保推理时也使用 Sinkhorn

### 方法3：使用自动化脚本

```bash
python sid/scripts/rqvae_auto.py \
  --config custom_config.json \
  --output_path results_rqvae_l1_sinkhorn/task3/indices.jsonl
```

配置文件 `custom_config.json`：

```json
{
  "data_npy": "3_text_format_embedding.npy",
  "meta_jsonl": "3_meta.jsonl",
  "results_root": "./results_rqvae_l1_sinkhorn",
  "device": "cuda:0",
  "batch_size": 1024,
  
  "train_defaults": {
    "epochs": 40,
    "lr": 0.001,
    "num_emb_list": [1024, 1024, 1024, 1024, 256],
    "e_dim": 128,
    "layers": [512, 256, 128],
    "sk_epsilons": [0.01, 0.0, 0.0, 0.0, 0.0],
    "sk_iters": 50,
    "ema_codebook": true,
    "ema_decay": 0.99,
    "kmeans_init": true
  },
  
  "strategies": [
    {
      "name": "rqvae",
      "enabled": true,
      "train": true,
      "infer": true,
      "infer_overrides": {
        "use_sk": true
      }
    }
  ]
}
```

## 关键代码模块

### 1. RQVAE 模型
- **文件**: `sid/models/rqvae.py`
- **核心**: `RQVAE` 类，实现残差 VQ + Sinkhorn

### 2. VQ 层
- **文件**: `sid/models/vq.py`
- **核心**: `VectorQuantizer` 类，支持 Sinkhorn 软分配

### 3. 训练器
- **文件**: `sid/src/training/trainer.py`
- **核心**: `Trainer` 类，处理训练循环和指标记录

### 4. Sinkhorn 实现

在 `sid/models/vq.py` 中：

```python
def sinkhorn(cost_matrix, epsilon, num_iters):
    """
    Sinkhorn-Knopp 算法
    
    cost_matrix: [B, K] 距离矩阵
    epsilon: 温度参数
    num_iters: 迭代次数
    """
    log_Q = -cost_matrix / epsilon
    for _ in range(num_iters):
        log_Q = log_Q - torch.logsumexp(log_Q, dim=1, keepdim=True)
        log_Q = log_Q - torch.logsumexp(log_Q, dim=0, keepdim=True)
    Q = torch.exp(log_Q)
    return Q  # [B, K] 软分配矩阵
```

## 训练时间估计

| 任务 | 样本数 | GPU | 时间（40 epochs） |
|---|---:|---|---|
| 任务2 | 193,034 | V100 | ~2-3 小时 |
| 任务3 | 193,034 | V100 | ~2-3 小时 |

## 输出文件结构

```
results_rqvae_l1_sinkhorn/
├── task2/
│   ├── checkpoints/
│   │   ├── epoch_0.pt
│   │   ├── epoch_1.pt
│   │   └── ...
│   └── indices/
│       ├── indices.jsonl
│       └── metrics.json
└── task3/
    ├── checkpoints/
    └── indices/
        ├── indices.jsonl
        └── metrics.json
```

## 验证结果

### 检查 metrics.json

```bash
# 查看任务3的结果
python -c "
import json
m = json.load(open('results_rqvae_l1_sinkhorn/task3/indices/metrics.json'))
print(f'CR: {m[\"metrics_5layer\"][\"collision_rate\"]:.4%}')
print(f'ICR: {m[\"metrics_5layer\"][\"icr\"]:.4%}')
"
```

预期输出：
```
CR: 0.0907%
ICR: 99.9093%
```

### 逐层利用率分析

```python
import json
import numpy as np

# 加载 indices
data = []
with open('results_rqvae_l1_sinkhorn/task3/indices/indices.jsonl') as f:
    for line in f:
        data.append(json.loads(line))

# 检查每层
for layer in range(5):
    codes = [d['indices'][layer] for d in data]
    K = max(codes) + 1
    active = len(set(codes))
    util = active / K
    print(f"L{layer+1}: {active}/{K} = {util:.2%}")
```

预期输出（任务3）：
```
L1: 1024/1024 = 100.00%  ← Sinkhorn 保证100%
L2: 1024/1024 = 100.00%
L3: 1024/1024 = 100.00%
L4: 1024/1024 = 100.00%
L5: 233/256 = 91.02%     ← 硬编码层
```

## 常见问题

### Q1: Sinkhorn 参数如何调优？

**epsilon（温度参数）：**
- 0.001-0.005：严格均匀化，接近硬分配
- **0.01**：推荐值，平衡均匀性和稳定性
- 0.05-0.1：宽松约束，更平滑的分配

**num_iters：**
- 10-20：快速但可能不收敛
- **50**：推荐值
- 100+：更精确但训练慢

### Q2: 为什么只在 L1 用 Sinkhorn？

1. **L1 最关键**：第一层捕获主要语义，利用率最重要
2. **其他层已经好**：L2-L4 通过 EMA + K-Means 已经能达到100%
3. **训练效率**：Sinkhorn 计算代价高，只在 L1 用可以平衡效果和速度

### Q3: EMA codebook 的作用？

EMA（指数移动平均）更新 codebook，而不是梯度更新：

```python
# 每次前向传播后
codebook = decay * codebook + (1 - decay) * cluster_mean
```

**优点：**
- 更稳定，避免梯度消失
- 配合 Sinkhorn 效果更好
- VQ-VAE 的标准做法

### Q4: 如何复现论文中的其他配置？

修改 `reproduce_rqvae_l1_sinkhorn.py` 中的配置：

```python
# 例如：L1-L2 都用 Sinkhorn
RQVAE_L1_SINKHORN_CONFIG = {
    "sk_epsilons": [0.01, 0.01, 0.0, 0.0, 0.0],  # L1-L2
    ...
}

# 例如：更大的 codebook
RQVAE_L1_SINKHORN_CONFIG = {
    "num_emb_list": [2048, 2048, 1024, 1024, 256],
    ...
}
```

## 与其他方案对比

| 方案 | CR (任务3) | ICR | 训练时间 | GPU需求 |
|---|---:|---:|---|---|
| **RQVAE + L1 Sinkhorn** | **0.09%** | **99.91%** | 2-3小时 | 需要 |
| RQ-OPQ | 0.78% | 99.22% | 3-5分钟 | 不需要 |
| RQ-KMeans | 1.64% | 98.36% | 3-5分钟 | 不需要 |

## 脚本位置

- 一键复现脚本: `reproduce_rqvae_l1_sinkhorn.py`
- 训练脚本: `sid/scripts/train/train_rqvae.py`
- 推理脚本: `sid/src/codebook/generate_indices_rqvae.py`
- 自动化脚本: `sid/scripts/rqvae_auto.py`
- 配置文件: `sid/scripts/benchmark/benchmark_config.json`

## 参考文献

- RQVAE: Lee et al., "Autoregressive Image Generation using Residual Quantization", CVPR 2022
- Sinkhorn: Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal Transport", NeurIPS 2013
- VQ-VAE: van den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017
