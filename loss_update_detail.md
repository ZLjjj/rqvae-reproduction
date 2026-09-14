# RQVAE 损失函数与参数更新说明

## 1 前向计算

输入 embedding 记为 `x`，Encoder 输出 latent：

```text
z = Encoder(x)
```

四层残差量化得到：

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

拼接年份和付费状态 hard-code 后，Decoder 生成重构向量：

```text
x_hat = Decoder(q)
```

## 2 重构损失

当前使用 MSE：

```text
L_recon = mean((x_hat - x)^2)
```

它衡量重构 embedding 与原始 embedding 的距离，推动 Encoder、Decoder 和量化结果共同保留输入语义。

## 3 量化损失

对每一层，设输入 latent 为 `h`，量化向量为 `h_q`，当前 commitment 系数为 `beta=0.25`：

```text
L_commit = MSE(stop_gradient(h_q), h)
L_quant  = beta * L_commit
```

`stop_gradient` 表示这项损失不通过 `h_q` 更新 codebook，主要推动 Encoder 输出靠近当前 code。

四层 RQ 的量化损失取平均：

```text
L_quant = mean(L_quant_1, L_quant_2, L_quant_3, L_quant_4)
```

## 4 EMA 模式下的总损失

当前启用 `ema_codebook=true`，因此 codebook loss 被置为 0：

```text
L_total = L_recon + quant_loss_weight * L_quant
L_total = L_recon + 1.0 * L_quant
```

EMA 模式下：

```text
Encoder：通过 L_total 反向传播更新
Decoder：通过 L_recon 反向传播更新
Codebook：不通过梯度更新，而通过 EMA 更新
Hard-code：按年份和付费状态规则计算，不训练
```

量化层使用 straight-through estimator：

```text
h_st = h + stop_gradient(h_q - h)
```

前向使用 `h_q`，反向时梯度近似直接传给 Encoder 输出 `h`。

## 5 EMA 更新

对第 `k` 个 code，当前 batch 的分配样本集合为 `A_k`：

```text
n_k = |A_k|
s_k = sum(h_i), i in A_k
```

当前参数 `decay=0.99`：

```text
N_k <- decay * N_k + (1 - decay) * n_k
S_k <- decay * S_k + (1 - decay) * s_k
```

随后更新码本向量：

```text
e_k <- S_k / N_k
```

实现使用 `ema_eps=1e-5` 做平滑，避免 `N_k` 接近零时数值不稳定。

## 6 非 EMA 模式对比

关闭 EMA 后，codebook 会通过普通 codebook loss 获得梯度：

```text
L_codebook = MSE(h_q, stop_gradient(h))
L_quant = L_codebook + beta * L_commit
```

两种模式的区别：

| 模式 | Encoder | Decoder | Codebook |
|---|---|---|---|
| 当前 EMA 模式 | 梯度更新 | 梯度更新 | EMA 更新 |
| 普通 VQ 模式 | 梯度更新 | 梯度更新 | 梯度更新 |

## 7 优化器更新

计算 `L_total` 后执行：

```text
optimizer.zero_grad()
L_total.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

当前优化器是 AdamW：

```text
learning_rate = 1e-3
weight_decay = 0
max_grad_norm = 1.0
```
