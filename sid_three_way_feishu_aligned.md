# SID 三方视频实验对比结果

<callout emoji="bulb" background-color="light-blue" border-color="blue">
**阅读说明**：本表沿用原实验表的列结构。三方实验均针对同一批 193,034 条视频资源；“总体内聚性”在本实验中等同于“视频内聚性”，数值按共享 L1 / L2 / L3 prefix 的 weighted cosine 填写。
</callout>

## 实验一 三方视频 SID 总表

| 基础Embedding模型 | SID生成方法 | 关键参数设置 | 梯度更新方式 | 冲突率 | 独立编码率 | 总体内聚性 | 视频内聚性 | 电台内聚性 |
|---|---|---|---|---:|---:|---|---|---|
| Qwen3-Embedding-4B | RQ-VAE + 硬编码（baseline 视频子集） | 4层1024维；hard-code 1层；混合 station + video 训练后取 video | EMA，decay=0.99 | 0.038667 | 0.961333 | 0.575612 / 0.667208 / 0.755617 | 0.575612 / 0.667208 / 0.755617 | 不适用 |
| Qwen3-Embedding-4B | RQ-VAE + 硬编码（任务2 同源 video-only） | 4层1024维；hard-code 1层；video-only；e_dim=128 | EMA，decay=0.99 | **0.004875** | **0.995125** | 0.547222 / 0.695476 / 0.839722 | 0.547222 / 0.695476 / 0.839722 | 不适用 |
| Qwen3-Embedding-4B cleaned text_format | RQ-VAE + 硬编码（任务3 cleaned video-only） | 4层1024维；hard-code 1层；清洗重组 text_format；e_dim=128 | EMA，decay=0.99 | 0.004890 | 0.995110 | **0.620066 / 0.755967 / 0.850521** | **0.620066 / 0.755967 / 0.850521** | 不适用 |

### 指标定义

| 指标 | 定义 | 趋势 |
|---|---|---|
| 冲突率 | 1 - 独立编码率；完整 5 层 SID 中发生碰撞的比例 | 越低越好 |
| 独立编码率 | unique full SID paths / N | 越高越好 |
| 总体内聚性 | 共享 prefix bucket 内 embedding 两两 cosine 的 weighted mean | 越高越好 |
| 视频内聚性 | 仅视频资源的 prefix bucket 内聚性；本实验与总体内聚性相同 | 越高越好 |
| 电台内聚性 | 纯视频实验没有电台样本 | 不适用 |

---

## 实验二 Prefix bucket 分层指标

| 层级 | 方案 | bucket 数 | 均值 | 中位数 | P25 | P75 | 最大 bucket |
|---|---|---:|---:|---:|---:|---:|---:|
| L1 | baseline 视频 | 420 | 459.605 | 10 | 2 | 895 | 3,065 |
| L1 | 任务2 同源 video-only | 49 | 3,939.469 | 3,393 | 2,416 | 4,755 | 9,511 |
| L1 | 任务3 cleaned text_format | 79 | 2,443.468 | 2,130 | 1,345.5 | 3,245 | **6,962** |
| L2 | baseline 视频 | 33,029 | 5.844 | 2 | 1 | 5 | 421 |
| L2 | 任务2 同源 video-only | 31,961 | 6.040 | 3 | 1 | 6 | 234 |
| L2 | 任务3 cleaned text_format | 40,889 | 4.721 | 2 | 1 | 4 | 456 |
| L3 | baseline 视频 | 130,461 | 1.480 | 1 | 1 | 1 | 112 |
| L3 | 任务2 同源 video-only | 175,338 | 1.101 | 1 | 1 | 1 | 19 |
| L3 | 任务3 cleaned text_format | 177,064 | 1.090 | 1 | 1 | 1 | 51 |
| L4 | baseline 视频 | 176,548 | 1.093 | 1 | 1 | 1 | 77 |
| L4 | 任务2 同源 video-only | 191,745 | 1.007 | 1 | 1 | 1 | 9 |
| L4 | 任务3 cleaned text_format | 191,993 | 1.005 | 1 | 1 | 1 | **8** |
| L5 | baseline 视频 | 185,570 | 1.040 | 1 | 1 | 1 | 47 |
| L5 | 任务2 同源 video-only | 192,093 | 1.005 | 1 | 1 | 1 | 9 |
| L5 | 任务3 cleaned text_format | 192,090 | 1.005 | 1 | 1 | 1 | **8** |

## 实验三 Prefix 内聚性对比

| 层级 | baseline 视频 | 任务2 同源 video-only | 任务3 cleaned text_format |
|---|---:|---:|---:|
| L1 weighted cosine | 0.575612 | 0.547222 | **0.620066** |
| L2 weighted cosine | 0.667208 | 0.695476 | **0.755967** |
| L3 weighted cosine | 0.755617 | 0.839722 | **0.850521** |
| L4 weighted cosine | 0.821035 | **0.926319** | 0.914693 |

## 实验四 结果结论

<grid cols="2">
<column>
<callout emoji="white_check_mark" background-color="light-green" border-color="green">
**任务2结论**

同源 video-only 训练将 CR 从 3.8667% 降至 0.4875%，ICR 提升到 0.995125，说明训练范围从混合域收敛到视频域是最大收益来源。
</callout>
</column>
<column>
<callout emoji="rocket" background-color="light-blue" border-color="blue">
**任务3结论**

清洗后的 text_format 在 ICR 上与任务2基本持平，但 L1-L3 内聚性更高，L1 最大 bucket 从 9,511 降至 6,962，更适合前缀召回和分桶。
</callout>
</column>
</grid>

## 复现信息

| 项目 | 路径 |
|---|---|
| baseline 索引 | `1_vs_Qwen4B_rqvae_hc_ema.index.meta.jsonl` |
| 任务2数据 | `2_meta_full_video.jsonl` + `2_meta_full_video.npy` |
| 任务3数据 | `3_meta.jsonl` + `3_text_format_embedding.npy` |
| 任务2 SID | `results_three_way/same_source_video/indices/indices.jsonl` |
| 任务3 SID | `results_three_way/video_v3/indices/indices.jsonl` |
| 三方汇总 | `results_three_way/evaluation/three_way_summary.json` |
