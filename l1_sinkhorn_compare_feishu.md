---

## 任务2与任务3 L1 Sinkhorn 对比实验

### 实验配置

两组实验均只在 L1 增加 Sinkhorn，保留原有 RQVAE 结构、EMA、KMeans 初始化、L4 Sinkhorn 和优化器配置。

| 项目 | 任务2 L1 Sinkhorn | 任务3 L1 Sinkhorn |
|---|---|---|
| 输入 embedding | `2_meta_full_video.npy` | `3_text_format_embedding.npy` |
| 数据规模 | 193,034 条 × 2,560 维 | 193,034 条 × 2,560 维 |
| `num_emb_list` | `[1024,1024,1024,1024,256]` | `[1024,1024,1024,1024,256]` |
| `sk_epsilons` | `[0.003,0,0,0.003,0]` | `[0.003,0,0,0.003,0]` |
| EMA | decay=0.99，eps=1e-5 | decay=0.99，eps=1e-5 |
| 筛选训练 | 10 epoch计划，实际第8轮暂停 | 10 epoch完成 |

### 与各自原始任务对比

| 指标 | 任务2原始 | 任务2 L1 Sinkhorn | 任务3原始 | 任务3 L1 Sinkhorn |
|---|---:|---:|---:|---:|
| L1 利用率 | 4.79% | **100.00%** | 7.71% | **100.00%** |
| L1 active code | 49 | **1024** | 79 | **1024** |
| L1 最大 code 频次 | 9,511 | **210** | 6,962 | **216** |
| L2 利用率 | 100.00% | 90.72% | 100.00% | **100.00%** |
| L3 利用率 | 100.00% | 79.88% | 100.00% | **100.00%** |
| L4 利用率 | 100.00% | 100.00% | 100.00% | 100.00% |
| L5 利用率 | 45.51% | 45.51% | 45.51% | 45.51% |
| 完整 SID ICR | 0.995125 | **0.999254** | 0.995110 | **0.999093** |
| 完整 SID CR | 0.004875 | **0.000746** | 0.004890 | **0.000907** |

### 两个候选方案直接对比

| 指标 | 任务2 L1 Sinkhorn | 任务3 L1 Sinkhorn |
|---|---:|---:|
| L1 利用率 | 100.00% | 100.00% |
| L1 最大 code 频次 | 210 | 216 |
| L2 利用率 | 90.72% | **100.00%** |
| L3 利用率 | 79.88% | **100.00%** |
| L4 利用率 | 100.00% | 100.00% |
| 完整 SID ICR | **0.999254** | 0.999093 |
| 完整 SID CR | **0.000746** | 0.000907 |

### 初步结论

两组 L1 Sinkhorn 都将 L1 利用率提升到 100%，并把最大 L1 code 频次从数千降低到约 210，显著改善了第一层的负载均衡。任务2候选的完整 SID ICR 略高、CR 略低，但 L2/L3 利用率下降到 90.72%/79.88%，说明部分容量分工转移到了 L1。任务3候选在 L1 利用率、L2-L4 利用率和完整 SID 唯一率之间更均衡：L2-L4 均保持 100%，ICR 达到 0.999093，CR 为 0.000907。

当前建议优先继续验证任务3 L1 Sinkhorn；任务2候选保留为完整 ICR 更高的对照方案。两组结果目前都属于筛选实验，正式替换前还需要独立随机种子复现，并补充全量内容 purity、标签 Jaccard 和 code 频次熵。

### 实验产物

| 实验 | checkpoint | SID | 数值对比 | 内容分析 |
|---|---|---|---|---|
| 任务2 L1 Sinkhorn | `results_three_way/l1_screen_task2_eps003/checkpoints/best_collision_model.pth` | `results_three_way/l1_screen_task2_eps003/indices/indices.jsonl` | `results_three_way/l1_screen_task2_eps003/comparison.json` | `results_three_way/l1_screen_task2_eps003/content_analysis.md` |
| 任务3 L1 Sinkhorn | `results_three_way/exp_task3_l1_sinkhorn/checkpoints/best_collision_model.pth` | `results_three_way/exp_task3_l1_sinkhorn/indices/indices.jsonl` | `results_three_way/exp_task3_l1_sinkhorn/comparison.json` | `results_three_way/exp_task3_l1_sinkhorn/content_analysis.md` |
