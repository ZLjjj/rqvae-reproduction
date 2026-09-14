# 当前 RQVAE 方法说明

本文说明任务二和任务三使用的 RQVAE 方法。两者模型结构与训练参数相同，区别只在输入 embedding：任务二使用完整视频 embedding，任务三使用清洗后的 `text_format` embedding。

## 1. 整体结构

```text
2560 维 embedding
        ↓
MLP Encoder：2560 → 512 → 256 → 128
        ↓
4 层 Residual Vector Quantization
        ↓
Hard-code 第五层
        ↓
MLP Decoder：128 → 256 → 512 → 2560
        ↓
重构 embedding
```

当前配置：

| 模块 | 配置 |
|---|---|
| 输入 | 2560 维 float32 embedding |
| latent | 128 维 |
| RQ 层数 | 4 层 |
| 前四层 codebook | 每层 1024 个 code |
| 第五层 | hard-code，9-bit，容量 512 |
| 重构损失 | MSE |
| 优化器 | AdamW，learning rate=1e-3，weight decay=0 |
| 训练 | 40 epochs，batch size=1024 |

## 2. Residual Vector Quantization

Encoder 输出 latent `z` 后，四层量化器逐层处理残差：

```text
q1 = VQ1(z)
r1 = z - q1
q2 = VQ2(r1)
r2 = r1 - q2
q3 = VQ3(r2)
r3 = r2 - q3
q4 = VQ4(r3)
q  = q1 + q2 + q3 + q4
```

前四层对应 SID：

```text
<a_x><b_x><c_x><d_x>
```

第一层通常表达粗粒度方向，后续层表达前一层未解释的细节残差。

## 3. KMeans 初始化

当前开启：

```text
kmeans_init = true
kmeans_init_dedup = true
kmeans_init_max_samples = 200000
kmeans_init_seed = 0
```

训练开始时，从 latent 样本中抽取最多 200,000 个样本，去除重复样本后运行 KMeans，得到每层初始 codebook 中心。之后再由训练过程中的 EMA 或梯度继续更新。

KMeans 只负责初始化，不等于 RQ-Kmeans 模型；当前主体仍是 MLP 自编码器加 Residual VQ。

## 4. Sinkhorn 分配

### 4.1 使用位置

当前配置：

```python
sk_epsilons = [0.0, 0.0, 0.0, 0.003, 0.0]
```

实际生效：

```text
L1：关闭
L2：关闭
L3：关闭
L4：开启，epsilon=0.003
L5：不经过 RQ，属于 hard-code
```

### 4.2 普通分配

对 latent `h_i` 和 codebook 向量 `e_k`，计算距离：

```text
d_ik = ||h_i - e_k||²
```

关闭 Sinkhorn 时，直接选择最近的 code：

```text
c_i = argmin_k d_ik
```

这种方式只考虑单个样本，可能让大量样本集中到少数 code。

### 4.3 Sinkhorn 的计算

开启 Sinkhorn 时，先将距离中心化，然后计算软分配矩阵：

```text
Q_ik = exp(-d_ik / epsilon)
```

随后反复执行行归一化和列归一化，使一个 batch 内的 code 分配更接近均衡约束。当前 `sk_iters=100`。最后取每个样本在 `Q` 中概率最大的 code。

流程是：

1. 计算 latent 到所有 code 的距离矩阵。
2. 对距离做中心化。
3. 用 `exp(-distance / epsilon)` 得到软分配。
4. 进行 100 次行列归一化。
5. 取每个样本最大分配概率对应的 code。

### 4.4 epsilon 的作用

| epsilon | 行为 |
|---:|---|
| 0 | 关闭 Sinkhorn，直接最近邻分配 |
| 较小，如 0.001 | 接近最近邻，只有轻微均衡效果 |
| 0.003 | 当前配置，进行中等平滑和均衡 |
| 更大，如 0.01 | 均衡更强，但可能牺牲局部距离最优性 |

`epsilon` 越小，距离差异越敏感；越大，分配越平滑。Sinkhorn 是 batch 级约束，不保证全数据集的 code 使用完全均匀。

### 4.5 与利用率的关系

L4 开启 Sinkhorn 后，任务二和任务三的 L4 利用率达到 100%。但这个结果还同时受 video-only 数据、KMeans 初始化、EMA、batch size 和残差分布影响，不能全部归因于 Sinkhorn。

## 5. EMA 码本更新

### 5.1 作用

普通 VQ 可以通过梯度直接更新 codebook，但 codebook 可能随 batch 剧烈移动。当前使用 VQ-VAE 风格的 EMA：codebook 根据每个 code 被分配到的 latent 统计量平滑更新。

配置为：

```text
ema_codebook = true
ema_decay = 0.99
ema_eps = 1e-5
```

### 5.2 Batch 统计

设第 `k` 个 code 当前 batch 被分配到的样本集合为 `A_k`：

```text
n_k = |A_k|
s_k = Σ h_i，i ∈ A_k
```

代码用 `scatter_add_` 计算每个 code 的样本数和 latent 和，避免创建巨大的 one-hot 矩阵。

### 5.3 指数移动平均

设旧统计量为 `N_k`、`S_k`，当前 batch 统计量为 `n_k`、`s_k`，则：

```text
N_k ← gamma · N_k + (1 - gamma) · n_k
S_k ← gamma · S_k + (1 - gamma) · s_k
```

其中当前 `gamma=0.99`。然后更新 codebook：

```text
e_k ← S_k / N_k
```

实现中还使用 `ema_eps=1e-5` 做平滑，避免某个 code 的统计量接近零时除零或产生极端向量。

### 5.4 decay 的含义

| 参数 | 含义 |
|---|---|
| `ema_decay=0.99` | 保留约 99% 历史统计，吸收约 1% 当前 batch 统计 |
| 更大 decay | 更新更平滑，但适应数据变化更慢 |
| 更小 decay | 适应更快，但 codebook 更容易抖动 |
| `ema_eps=1e-5` | 防止 cluster size 接近零导致数值不稳定 |

EMA 能提高训练稳定性，但不会强制所有 code 都被使用；低利用率仍可能来自数据分布、初始化或分配策略。

## 6. Hard-code 第五层

第五层不是普通可学习 codebook，而是由业务字段直接计算。主要字段为：

```text
publish_year
sale_type / saletype
```

年份先转换为 bucket，再与付费状态打包：

```text
year_bucket = 2035 - publish_year
index = (year_bucket << 2) | sale_type
```

第五层对应 `<e_x>`，表达年份和付费状态等业务属性，不属于纯语义 embedding 量化。评测时按 9-bit 容量 512 统计。

## 7. 损失与优化

总损失为：

```text
L = L_recon + lambda_q · L_quant
```

当前：

```text
L_recon = MSE(reconstructed_embedding, input_embedding)
lambda_q = 1.0
beta = 0.25
```

训练使用 AdamW，并进行 `max_norm=1.0` 的梯度裁剪。

## 8. 当前实际使用的方法清单

```text
MLP Autoencoder
Residual Vector Quantization
KMeans codebook 初始化
第4层 Sinkhorn 分配
EMA codebook 更新
年份与付费状态 hard-code
MSE reconstruction loss
quantization / commitment loss
AdamW
gradient clipping
```

当前没有使用：

```text
OPQ / PQ
RQ-OPQ
FAISS RQ-Kmeans
L1 专门 balance loss
全层 Sinkhorn
learned 第五层 codebook
```

## 9. 训练流程

1. 读取 embedding 和对齐的元数据。
2. Encoder 得到 128 维 latent。
3. 依次执行 L1 到 L4 残差量化。
4. L4 使用 Sinkhorn，其他 RQ 层使用最近邻分配。
5. 训练阶段用 EMA 统计更新 codebook。
6. 拼接年份和付费状态 hard-code。
7. Decoder 重构原始 embedding。
8. 计算 MSE 和量化损失并用 AdamW 更新。
9. 每轮计算 collision rate，保存最佳 checkpoint。
10. 用最佳 checkpoint 对全量数据生成五层 SID。

## 10. 任务二与任务三

任务二和任务三使用完全相同的 RQVAE 方法。任务二输入完整视频 metadata embedding，任务三输入 cleaned text_format embedding。

任务三的 L1 利用率为 7.71%，高于任务二的 4.79%，且最大 L1 bucket 更小；两者 L2-L4 利用率均达到 100%。这说明 cleaned text_format 主要改善了粗粒度语义分区，而不是改变 RQVAE 的量化结构。
