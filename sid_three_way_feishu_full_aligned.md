# SID 三方视频量化对比实验报告

<callout emoji="bulb" background-color="light-blue" border-color="blue">
**核心结论**：将混合域 SID 改为 video-only 训练后，完整 SID 的 ICR 从 **0.961333** 提升到 **0.995125**，CR 从 **3.8667%** 降至 **0.4875%**。清洗后的 `text_format` 在唯一率上与同源 video-only 基本持平，同时改善早期 prefix 的语义聚合和 bucket 均衡性。
</callout>

## 1. 实验目标

本实验回答两个问题：

1. 只使用视频数据训练 codebook，是否优于现有 station + video 混合训练的 baseline。
2. 对 `text_format` 做字段清洗、去重和重组后，是否能进一步改善 SID 的前缀结构。

三方方案使用同一批 193,034 条视频资源进行比较，统一计算完整 SID 唯一率、碰撞率、prefix bucket 分布、CUR 和 bucket 内 cosine 相似度。

## 2. 三方方案

<grid cols="3">
<column>
<callout emoji="memo" background-color="pale-gray">
**baseline 视频**

现有 station + video 混合 SID 中的视频连续尾段。用于衡量当前线上/既有方案在视频域上的表现。
</callout>
</column>
<column>
<callout emoji="white_check_mark" background-color="light-green" border-color="green">
**任务2 同源 video-only**

使用 `2_meta_full_video.npy`，只用视频数据重新训练 RQVAE，隔离训练域范围的影响。
</callout>
</column>
<column>
<callout emoji="rocket" background-color="light-blue" border-color="blue">
**任务3 cleaned text_format**

使用 `3_text_format_embedding.npy`，在 video-only 基础上验证文本字段清洗和重组的影响。
</callout>
</column>
</grid>

## 3. 数据与配置

| 项目 | 配置 |
|---|---|
| 任务2数据 | `2_meta_full_video.jsonl` + `2_meta_full_video.npy` |
| 任务3数据 | `3_meta.jsonl` + `3_text_format_embedding.npy` |
| 数据规模 | 193,034 条 × 2,560 维，float32 |
| 模型结构 | MLP Encoder → 4 层 Residual VQ → hard-code 第 5 层 → MLP Decoder |
| codebook | `[1024, 1024, 1024, 1024, 256]` |
| latent dimension | `e_dim = 128` |
| MLP layers | `[512, 256, 128]` |
| Sinkhorn | `[0, 0, 0, 0.003, 0]` |
| EMA codebook | 开启，decay `0.99`，eps `1e-5` |
| 训练 | 40 epochs，AdamW，batch size 1024，CPU |
| checkpoint | `best_collision_model.pth` |

第五层是业务 hard-code 层，评测按 9-bit 容量兼容统计；cosine 指标只计算 L1-L4，不把 hard-code 层当作语义 embedding 层。

## 4. 核心结果

| 基础Embedding模型 | SID生成方法 | 关键参数设置 | 梯度更新方式 | 冲突率 | 独立编码率 | 总体内聚性 | 视频内聚性 | 电台内聚性 |
|---|---|---|---|---:|---:|---|---|---|
| Qwen3-Embedding-4B | RQ-VAE + 硬编码（baseline 视频子集） | 码本：4层1024维；硬编码层：1层；混合 station + video 训练后取 video | 指数移动平均更新 EMA（decay=0.99） | 0.038667 | 0.961333 | 共享一层：0.575612；共享两层：0.667208；共享三层：0.755617 | 共享一层：0.575612；共享两层：0.667208；共享三层：0.755617 | 不适用 |
| Qwen3-Embedding-4B | RQ-VAE + 硬编码（任务2 同源 video-only） | 码本：4层1024维；硬编码层：1层；e_dim=128 | 指数移动平均更新 EMA（decay=0.99） | 0.004875 | 0.995125 | 共享一层：0.547222；共享两层：0.695476；共享三层：0.839722 | 共享一层：0.547222；共享两层：0.695476；共享三层：0.839722 | 不适用 |
| Qwen3-Embedding-4B cleaned text_format | RQ-VAE + 硬编码（任务3 cleaned video-only） | 码本：4层1024维；硬编码层：1层；e_dim=128；清洗重组 text_format | 指数移动平均更新 EMA（decay=0.99） | 0.004890 | 0.995110 | 共享一层：0.620066；共享两层：0.755967；共享三层：0.850521 | 共享一层：0.620066；共享两层：0.755967；共享三层：0.850521 | 不适用 |

<callout emoji="✅" background-color="light-green" border-color="green">
**video-only 的收益**：任务2相较 baseline 的 CR 从 3.8667% 降到 0.4875%，下降约 87.4%；这说明训练域从混合 station + video 收敛到 video-only 是最主要的收益来源。
</callout>

## 5. Prefix bucket 分布

| 层级 | 方案 | bucket 数 | 均值 | 中位数 | P25 | P75 | 最大 bucket |
|---|---|---:|---:|---:|---:|---:|---:|
| L1 | baseline | 420 | 459.605 | 10 | 2 | 895 | 3,065 |
| L1 | 任务2 | 49 | 3,939.469 | 3,393 | 2,416 | 4,755 | 9,511 |
| L1 | 任务3 | 79 | 2,443.468 | 2,130 | 1,345.5 | 3,245 | **6,962** |
| L2 | baseline | 33,029 | 5.844 | 2 | 1 | 5 | 421 |
| L2 | 任务2 | 31,961 | 6.040 | 3 | 1 | 6 | 234 |
| L2 | 任务3 | 40,889 | 4.721 | 2 | 1 | 4 | 456 |
| L3 | baseline | 130,461 | 1.480 | 1 | 1 | 1 | 112 |
| L3 | 任务2 | 175,338 | 1.101 | 1 | 1 | 1 | 19 |
| L3 | 任务3 | 177,064 | 1.090 | 1 | 1 | 1 | 51 |
| L4 | baseline | 176,548 | 1.093 | 1 | 1 | 1 | 77 |
| L4 | 任务2 | 191,745 | 1.007 | 1 | 1 | 1 | 9 |
| L4 | 任务3 | 191,993 | 1.005 | 1 | 1 | 1 | **8** |
| L5 | baseline | 185,570 | 1.040 | 1 | 1 | 1 | 47 |
| L5 | 任务2 | 192,093 | 1.005 | 1 | 1 | 1 | 9 |
| L5 | 任务3 | 192,090 | 1.005 | 1 | 1 | 1 | **8** |

任务3相较任务2将 L1 最大 bucket 从 9,511 降到 6,962，降幅约 26.8%；L4/L5 最大 bucket 也进一步降到 8，说明清洗后的输入在粗粒度和细粒度分桶上都更均衡。

## 6. Prefix CUR

| 层级 | baseline | 任务2 | 任务3 |
|---|---:|---:|---:|
| L1 | 0.410156 | 0.047852 | 0.077148 |
| L2 | 0.031499 | 0.030480 | 0.038995 |
| L3 | 0.000122 | 0.000163 | 0.000165 |
| L4 | 1.606e-7 | 1.744e-7 | 1.746e-7 |
| L5 | 3.296e-10 | 3.412e-10 | 3.412e-10 |

CUR 是已使用 prefix 数量与理论 codebook 容量的比值。三方容量相同，CUR 主要反映 prefix 使用的稀疏程度，需结合 bucket 分布和 cosine 一起判断，不能单独作为优劣结论。

## 7. Bucket cosine 相似度

以下为 `weighted_sim / unweighted_sim`，第五层不参与 cosine 计算。

| 层级 | baseline | 任务2 | 任务3 |
|---|---:|---:|---:|
| L1 | 0.575612 / 0.645326 | 0.547222 / 0.542767 | **0.620066 / 0.617029** |
| L2 | 0.667208 / 0.676441 | 0.695476 / 0.641968 | **0.755967 / 0.713348** |
| L3 | 0.755617 / 0.760105 | 0.839722 / 0.815486 | **0.850521 / 0.833719** |
| L4 | 0.821035 / 0.839862 | **0.926319 / 0.926358** | 0.914693 / 0.915704 |

任务3在 L1-L3 的 weighted cosine 全部高于任务2，说明共享早期 prefix 的资源语义更一致；L4 略低于任务2，但最终 ICR 几乎相同，且最大 bucket 更小。

## 8. 结果分析

### 8.1 混合域训练的影响

baseline 的 L1 只有 420 个 prefix bucket，L1 最大 bucket 为 3,065；任务2和任务3重新训练后，后续层的区分度显著增强。任务2的 L4/L5 最大 bucket 为 9/9，任务3为 8/8，而 baseline 为 77/47。video-only 训练让后层更接近“一条 SID 路径对应一条资源”。

### 8.2 text_format 清洗的影响

任务3不是简单换数据，而是对视频文本做标题规范化、标签合并去重、字段顺序统一，并保留 IP、付费状态、出版时间、导演、演员、角色等关键字段。结果显示，清洗主要改善早期 prefix 的语义聚合与 bucket 均衡，而不是继续显著提高最终唯一率。

### 8.3 方案建议

<grid cols="2">
<column>
<callout emoji="white_check_mark" background-color="light-green" border-color="green">
**推荐默认：任务3**

ICR 与任务2等价，L1-L3 weighted cosine 更高，L1/L4/L5 bucket 更均衡，适合作为后续检索与推荐链路的默认候选。
</callout>
</column>
<column>
<callout emoji="memo" background-color="pale-gray">
**保留对照：任务2**

任务2取得略高的最终 ICR，可作为极限去重率和 codebook 结构的对照方案。
</callout>
</column>
</grid>

## 9. 产物与复现

训练和评测产物：

- `results_three_way/same_source_video/checkpoints/best_collision_model.pth`
- `results_three_way/same_source_video/indices/indices.jsonl`
- `results_three_way/video_v3/checkpoints/best_collision_model.pth`
- `results_three_way/video_v3/indices/indices.jsonl`
- `results_three_way/evaluation/three_way_summary.json`

统一评测入口：`sid/scripts/eval/run_sid_three_way.py`

输入校验入口：`sid/scripts/eval/verify_three_way_inputs.py`

## 10. 限制与后续工作

- 本报告评估的是离散 codebook 结构指标，尚未覆盖 LLM SFT 后的 Recall@K、NDCG、端到端命中率和线上延迟。
- 需要在同一批查询、同一候选集上比较任务2和任务3的 SID 生成约束推理效果。
- 当前结果在 CPU 上完成，建议后续在 CUDA 环境复跑一次，确认训练速度和数值稳定性。
