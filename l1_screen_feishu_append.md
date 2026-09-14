---

## L1 Sinkhorn 增量筛选实验

### 实验设置

本实验在任务2同源 video-only RQVAE 基础上，仅修改第一层 Sinkhorn 配置，模型结构、数据、codebook 大小、EMA 和优化器保持不变。

| 项目 | 原任务2 | L1 Sinkhorn 候选 |
|---|---|---|
| `sk_epsilons` | `[0, 0, 0, 0.003, 0]` | `[0.003, 0, 0, 0.003, 0]` |
| 改动 | L4 使用 Sinkhorn | L1、L4 使用 Sinkhorn |
| 数据 | `2_meta_full_video.npy` | `2_meta_full_video.npy` |
| 训练计划 | 40 epochs正式模型 | 10 epochs筛选实验 |
| 实际进度 | 已完成 | 第8/10轮暂停 |

### 结果对比

| 指标 | 原任务2 | L1 Sinkhorn 候选 |
|---|---:|---:|
| L1 利用率 | 4.79% | **100.00%** |
| L1 active code | 49 | **1024** |
| L1 最大 code 频次 | 9,511 | **210** |
| L2 利用率 | 100.00% | 90.72% |
| L3 利用率 | 100.00% | 79.88% |
| L4 利用率 | 100.00% | 100.00% |
| L5 利用率 | 45.51% | 45.51% |
| 完整 SID ICR | 0.995125 | **0.999254** |
| 完整 SID CR | 0.004875 | **0.000746** |

### 初步判断

L1 Sinkhorn 将第一层利用率从 4.79% 提升到 100%，并将最大 L1 code 频次从 9,511 降至 210，说明第一层分配明显更加均衡；完整 SID 的 CR 也从 0.4875% 降至 0.0746%。同时，L2 和 L3 利用率分别下降到 90.72% 和 79.88%，说明均衡约束改变了残差层之间的容量分工。

该结果是筛选实验，不应只依据 L1 利用率直接替换正式模型。下一步需要在任务3上复现同一配置，并补充全量内容字段 purity、标签 Jaccard、各层 code 频次熵和重复实验，确认 L1 均衡是否带来可解释的内容分组。

### 实验产物

- checkpoint：`results_three_way/l1_screen_task2_eps003/checkpoints/best_collision_model.pth`
- SID：`results_three_way/l1_screen_task2_eps003/indices/indices.jsonl`
- 数值对比：`results_three_way/l1_screen_task2_eps003/comparison.json`
- 内容样例：`results_three_way/l1_screen_task2_eps003/content_analysis.md`

### 实验状态

- 状态：筛选完成至第8轮，未完成10轮计划
- 复现状态：未复现
- 是否推荐直接替换：暂不推荐
