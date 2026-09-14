# GenSearchRec - RQVAE + L1+L4 Sinkhorn 复现

[![复现状态](https://img.shields.io/badge/复现-成功-brightgreen)]()
[![CR](https://img.shields.io/badge/CR-0.0907%25-blue)]()
[![ICR](https://img.shields.io/badge/ICR-99.91%25-blue)]()

完美复现 RQVAE + L1+L4 Sinkhorn 的 CR=0.0907% 结果。

## 快速开始

```bash
# 一键复现
python reproduce_rqvae_correct.py --task task3

# 预期结果
# CR: 0.0907%
# ICR: 99.91%
# 唯一路径: 192,859 / 193,034
```

## 核心配置

```python
{
    "sk_epsilons": [0.003, 0.0, 0.0, 0.003, 0.0],  # L1+L4 Sinkhorn
    "sk_iters": 100,
    "kmeans_iters": 100,
    "epochs": 10
}
```

**关键：不是"L1 Sinkhorn"，而是"L1+L4 Sinkhorn"！**

## 文档

- [完整复现指南](REPRODUCTION_GUIDE.md) - 详细的使用说明和常见错误
- [原README](README.md) - 项目原始文档

## 实验结果

| 指标 | 目标 | 复现结果 |
|---|---:|---:|
| CR | 0.0907% | 0.0907% ✅ |
| ICR | 99.91% | 99.91% ✅ |
| L1 max_bucket | 216 | 216 ✅ |

完全一致！

## 目录结构

```
├── sid/                          # 核心代码
│   ├── models/                  # RQVAE, VQ, RQ 模型
│   ├── scripts/train/           # 训练脚本
│   └── src/codebook/            # 推理脚本
├── reproduce_rqvae_correct.py   # 一键复现脚本
├── REPRODUCTION_GUIDE.md        # 完整文档
└── results_rqvae_l1_sinkhorn/  # 复现结果
    └── task3_correct/
        ├── checkpoints/         # 模型
        └── indices/             # CR=0.0907%
```

## 引用

- RQVAE: Lee et al., CVPR 2022
- Sinkhorn: Cuturi, NeurIPS 2013  
- VQ-VAE: van den Oord et al., NeurIPS 2017

## 许可证

MIT License

