## 0. 所有实验方法与参数汇总表

| SID生成方法 | 关键参数设置 | 数据来源 | ICR | CR | L1利用率 | L1 max_bucket | L2-L4利用率 | 训练时间 | 推荐度 |
|---|---|---|---:|---:|---:|---:|---|---|---|
| **任务3+L1+L4 Sinkhorn** | 码本: [1024,1024,1024,1024,256] / Sinkhorn: L1+L4, ε=0.003, iter=100 / kmeans_iters=100 / EMA: decay=0.99, eps=1e-5 / epochs=10 / optimizer=AdamW, lr=0.001, batch=1024 | 3_text_format (清洗后) | **99.91%** | **0.09%** | **100%** | **216** | L2-L4=100% | 40-50分钟 (epoch 8) | ⭐⭐⭐⭐⭐ |
| 任务2+L1+L4 Sinkhorn | 码本: [1024,1024,1024,1024,256] / Sinkhorn: L1+L4, ε=0.003, iter=100 / kmeans_iters=100 / EMA: decay=0.99, eps=1e-5 / epochs=10 / optimizer=AdamW, lr=0.001, batch=1024 | 2_meta_full_video | 99.93% | 0.07% | 100% | 210 | L2=91%, L3=80% | 40-50分钟 (epoch 8) | ⭐⭐⭐⭐ |
| RQ-OPQ | 码本: [1024,1024,1024,1024,512] / 方法: FAISS RQ-OPQ / 旋转优化: OPQ / 训练: K-Means, max_iter=25, CPU only | 3_text_format | 99.22% | 0.78% | 100% | - | L2-L5=100% | **3-5分钟** | ⭐⭐⭐⭐ |
| RQ-KMeans | 码本: [1024,1024,1024,1024,512] / 硬编码: L5(saletype) / 方法: FAISS RQ-KMeans / 训练: K-Means, max_iter=25, CPU only | 3_text_format | 98.36% | 1.64% | 100% | - | L2-L4=100%, L5=48% | **3-5分钟** | ⭐⭐⭐ |
| 任务3原始 | 码本: [1024,1024,1024,1024,256] / Sinkhorn: L4 only, ε=0.003, iter=100 / EMA: decay=0.99 / epochs=40 / optimizer=AdamW, lr=0.001, batch=1024 | 3_text_format | 99.51% | 0.49% | 7.71% | 6,962 | L2-L4=100% | 40分钟 | ⭐⭐ |
| 任务2原始 | 码本: [1024,1024,1024,1024,256] / Sinkhorn: L4 only, ε=0.003, iter=100 / EMA: decay=0.99 / epochs=40 / optimizer=AdamW, lr=0.001, batch=1024 | 2_meta_full_video | 99.51% | 0.49% | 4.79% | 9,511 | L2-L4=100% | 40分钟 | ⭐⭐ |
| Baseline混合域 | 码本: [1024,1024,1024,1024,256] / Sinkhorn: L4 only, ε=0.003, iter=100 / epochs=40 / optimizer=AdamW, lr=0.001, batch=1024, weight_decay=0 | Station + Video 混合 | 96.13% | 3.87% | 41.02% | 3,065 | - | 40分钟 | ⭐ |
| 子空间独立RQ | 码本: 6层 / 子空间: 2×1280维 / 每个子空间3层RQ / epochs=40 / optimizer=AdamW, lr=0.001, batch=1024 | 3_text_format | 98.01% | 1.99% | - | - | L3/L6=26% | 40分钟 | ❌ |
| SimVQ RQ | 码本: 6层 / L3用SimVQ, basis=[M,dim] / L1-L2传统VQ / epochs=50 / batch=256 / 采样训练: 50K samples | 3_text_format | 10.03% | 89.97% | 0.93% | - | L2=0.58% | - | ❌ |

---

