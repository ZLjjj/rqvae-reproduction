#!/bin/bash
# Step 0: 生成 text_format embedding 及 hardcode 字段（video_data_v2 管线）
# 用 --rebuild_text_format 开启去垂域前缀重建 + 标题去重 + hardcode 归一化
# 适用于视频/电台混合数据或纯视频数据
# 依赖：sentence-transformers, Qwen3-Embedding-4B 模型

set -e

INPUT=${INPUT:-"/mnt/yangjunli1/sid-gen/data/meta_full_video.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"/mnt/yangjunli1/sid-gen/video-data-v2"}
MODEL_PATH=${MODEL_PATH:-"/mnt/lizhaoxuan1/models/Qwen3-Embedding-4B"}
DEVICE=${DEVICE:-"cuda:0"}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_GPUS=${NUM_GPUS:-1}
GPU_ID=${GPU_ID:-0}

# REPO_ROOT 回到 gensearchrec 根目录（dirname/../..，避免双重 sid/sid）
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "Input:      $INPUT"
echo "Output dir: $OUTPUT_DIR"
echo "Model:      $MODEL_PATH"
echo "Device:     $DEVICE"
echo "rebuild_text_format: ON (去垂域+标题去重+hardcode归一化)"

PYTHONPATH="$REPO_ROOT/sid/src" python "$REPO_ROOT/sid/scripts/embedding/generate_embeddings.py" \
  --input "$INPUT" \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$MODEL_PATH" \
  --embedding_fields text_format \
  --hardcode_fields pay_type,publish_year,years_ago \
  --rebuild_text_format \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --num_gpus "$NUM_GPUS" \
  --gpu_id "$GPU_ID"

echo "Done. Output: $OUTPUT_DIR"
echo "产物：text_format_embedding_gpu*.npy + meta_gpu*.jsonl（merge 后合成）"
