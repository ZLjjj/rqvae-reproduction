# SID 三方视频对比实验运行手册

当前目录已经包含任务2和任务3的数据。先在 CUDA 训练机上执行输入校验：

```bash
python sid/scripts/eval/verify_three_way_inputs.py \
  --meta 2_meta_full_video.jsonl --npy 2_meta_full_video.npy
python sid/scripts/eval/verify_three_way_inputs.py \
  --meta 3_meta.jsonl --npy 3_text_format_embedding.npy
```

然后运行两套 video-only RQVAE，并自动评测 baseline、任务2和任务3：

```bash
bash sid/scripts/run_video_three_way.sh
```

若数据不在仓库根目录，可以覆盖变量：

```bash
DATA_ROOT=/path/to/data WORK=/path/to/results DEVICE=cuda:0 \
  bash sid/scripts/run_video_three_way.sh
```

统一结果输出到：

```text
results_three_way/evaluation/three_way_summary.json
```

三方定义如下：

1. `baseline_video`：现有混合 station/video SID 中的视频尾段，使用任务2的原始视频 embedding 对齐评测。
2. `same_source_video`：使用 `2_meta_full_video.npy` 训练 video-only RQVAE。
3. `video_v3`：使用 `3_text_format_embedding.npy` 训练清洗后的 text_format RQVAE。

三套训练统一使用 `[1024,1024,1024,1024,256]`、`e_dim=128`、Sinkhorn 第四层、EMA codebook；评测使用固定的 codebook size `[1024,1024,1024,1024,512]`，第五层按 9-bit hard-code 处理，cosine 指标排除最后一层。
