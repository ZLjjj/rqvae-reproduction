# RQVAE + L1+L4 Sinkhorn 完整复现指南

本仓库包含完整复现 RQVAE + L1+L4 Sinkhorn 实验的代码和结果。

## 实验结果

**任务3（3_text_format_embedding.npy）完美复现：**

| 指标 | 目标 | 复现结果 | 状态 |
|---|---:|---:|---|
| **CR** | **0.0907%** | **0.0907%** | ✅ 完全一致 |
| **ICR** | 99.91% | 99.91% | ✅ 完全一致 |
| 唯一路径 | 192,859 | 192,859 | ✅ 完全一致 |
| 碰撞数 | 175 | 175 | ✅ 完全一致 |
| L1 max_bucket | 216 | 216 | ✅ 完全一致 |

## 核心配置（关键！）

```python
{
    "num_emb_list": [1024, 1024, 1024, 1024, 256],  # 5层 codebook
    "sk_epsilons": [0.003, 0.0, 0.0, 0.003, 0.0],   # L1 和 L4 都用 Sinkhorn！
    "sk_iters": 100,                                # Sinkhorn 迭代次数
    "ema_codebook": True,                           # EMA 更新
    "ema_decay": 0.99,
    "kmeans_init": True,                            # K-Means 初始化
    "kmeans_iters": 100,
    "epochs": 10  # 实际 epoch 8 最佳
}
```

**关键点：**
- 不是"L1 Sinkhorn"，而是 **"L1+L4 Sinkhorn"**
- epsilon = 0.003（不是 0.01）
- sk_iters = 100（不是 50）
- kmeans_iters = 100（不是 20）

## 快速复现

### 方法1：一键运行

```bash
python reproduce_rqvae_correct.py --task task3
```

### 方法2：手动运行

#### 训练（8-10 epochs）

```bash
python sid/scripts/train/train_rqvae.py \
  --data_npy 3_text_format_embedding.npy \
  --meta_jsonl 3_meta.jsonl \
  --ckpt_dir results_reproduction/checkpoints \
  --num_emb_list 1024 1024 1024 1024 256 \
  --e_dim 128 \
  --layers 512 256 128 \
  --sk_epsilons 0.003 0.0 0.0 0.003 0.0 \
  --sk_iters 100 \
  --ema_codebook \
  --ema_decay 0.99 \
  --ema_eps 1e-05 \
  --kmeans_init \
  --kmeans_iters 100 \
  --kmeans_init_max_samples 200000 \
  --epochs 10 \
  --lr 0.001 \
  --batch_size 1024 \
  --eval_step 1 \
  --device cpu
```

#### 推理（生成 SID）

```bash
python sid/src/codebook/generate_indices_rqvae.py \
  --model rqvae \
  --ckpt results_reproduction/checkpoints/best_collision_model.pth \
  --data_npy 3_text_format_embedding.npy \
  --meta_jsonl 3_meta.jsonl \
  --output_dir results_reproduction/indices \
  --device cpu \
  --batch_size 1024 \
  --use_sk
```

**重要：推理时必须加 `--use_sk` 标志！**

## 训练时间

- **CPU**: 40-50分钟（8 epochs）
- **GPU**: 预计 5-10分钟（8 epochs）
- 每个 epoch: 约5-6分钟（CPU）

## 目录结构

```
gensearchrec-main-0901/
├── sid/
│   ├── models/
│   │   ├── rqvae.py          # RQVAE 模型
│   │   ├── vq.py             # VQ 层（含 Sinkhorn）
│   │   └── rq.py             # RQ 层
│   ├── scripts/
│   │   └── train/
│   │       └── train_rqvae.py  # 训练脚本
│   └── src/
│       ├── codebook/
│       │   └── generate_indices_rqvae.py  # 推理脚本
│       └── training/
│           └── trainer.py     # 训练器
├── reproduce_rqvae_correct.py  # 一键复现脚本
├── results_rqvae_l1_sinkhorn/
│   └── task3_correct/
│       ├── checkpoints/       # 训练的模型
│       └── indices/           # 生成的 SID
│           ├── indices.jsonl  # CR=0.0907%
│           └── metrics.json
└── REPRODUCTION_GUIDE.md      # 本文档
```

## 验证结果

```bash
python -c "
import json
m = json.load(open('results_rqvae_l1_sinkhorn/task3_correct/indices/metrics.json'))
print(f'CR: {m[\"metrics_5layer\"][\"collision_rate\"]*100:.4f}%')
print(f'ICR: {m[\"metrics_5layer\"][\"icr\"]*100:.4f}%')
print(f'L1 max_bucket: {m[\"metrics_5layer\"][\"bucket_size\"][0][\"max\"]}')
"
```

预期输出：
```
CR: 0.0907%
ICR: 99.9093%
L1 max_bucket: 216
```

## 常见错误

### ❌ 错误配置1：只在 L1 用 Sinkhorn

```python
"sk_epsilons": [0.01, 0.0, 0.0, 0.0, 0.0]  # 错误！
```

**结果：** CR ≈ 0.22%（比目标差2.4倍）

### ❌ 错误配置2：epsilon 太大

```python
"sk_epsilons": [0.01, 0.0, 0.0, 0.01, 0.0]  # 错误！
```

**结果：** 分布过于平滑，CR 升高

### ❌ 错误配置3：迭代次数不足

```python
"sk_iters": 50,        # 错误！应该 100
"kmeans_iters": 20     # 错误！应该 100
```

**结果：** 收敛不充分，CR 升高

### ✅ 正确配置

```python
{
    "sk_epsilons": [0.003, 0.0, 0.0, 0.003, 0.0],  # L1+L4
    "sk_iters": 100,
    "kmeans_iters": 100
}
```

**结果：** CR = 0.0907%（完美复现）

## Sinkhorn 算法实现

在 `sid/models/vq.py` 中：

```python
def sinkhorn(cost_matrix, epsilon, num_iters):
    """Sinkhorn-Knopp 算法"""
    log_Q = -cost_matrix / epsilon
    for _ in range(num_iters):
        # 行归一化
        log_Q = log_Q - torch.logsumexp(log_Q, dim=1, keepdim=True)
        # 列归一化
        log_Q = log_Q - torch.logsumexp(log_Q, dim=0, keepdim=True)
    Q = torch.exp(log_Q)
    return Q  # [B, K] 软分配矩阵
```

**作用：**
- 将硬分配（argmax）软化为概率分布
- 强制每个 code 被均匀使用
- epsilon 控制软化程度（0.003 是最优值）

## 与其他方案对比

| 方案 | CR (任务3) | ICR | 训练时间 | GPU需求 |
|---|---:|---:|---|---|
| **RQVAE + L1+L4 Sinkhorn** | **0.09%** | **99.91%** | 40-50分钟 | 可选 |
| RQ-OPQ | 0.78% | 99.22% | 3-5分钟 | 不需要 |
| RQ-KMeans | 1.64% | 98.36% | 3-5分钟 | 不需要 |

**RQVAE + L1+L4 Sinkhorn 是 CR 最低的方案！**

## 引用

如果使用本代码，请引用：

- RQVAE: Lee et al., "Autoregressive Image Generation using Residual Quantization", CVPR 2022
- Sinkhorn: Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal Transport", NeurIPS 2013
- VQ-VAE: van den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017

## 许可证

MIT License

## 联系

如有问题，请提 Issue。
