#!/bin/bash
# 仅视频 embedding 生成：从 merged meta.jsonl 提取视频行 → 生成 text_format embedding
# 用法: bash sid/scripts/run_video_embedding.sh
# 可覆盖环境变量: INPUT / OUTPUT_DIR / MODEL_PATH / DEVICE / BATCH_SIZE / NUM_GPUS / GPU_ID
# 日志: 全量输出 tee 到 $OUTPUT_DIR/run_<时间>.log

set -eo pipefail

INPUT=${INPUT:-/mnt/zhanggehang1/wulinyang1/raw_data/sidflow/meta.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/yangjunli1/sid-gen/video-data-v3}
MODEL_PATH=${MODEL_PATH:-/mnt/zhanggehang1/wulinyang1/models/Qwen3-Embedding-4B}
DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_GPUS=${NUM_GPUS:-1}
GPU_ID=${GPU_ID:-0}

# REPO_ROOT 用 git 顶层目录，脚本放仓库任意位置都能正确定位
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$REPO_ROOT" ]; then
    REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi
VIDEO_ONLY="${OUTPUT_DIR}/video_only.jsonl"
LOG="${OUTPUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUTPUT_DIR"

{
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始生成仅视频 embedding"
echo "============================================================"
echo "Input:       $INPUT"
echo "Output dir:  $OUTPUT_DIR"
echo "Model:       $MODEL_PATH"
echo "Device:      $DEVICE"
echo "Batch size:  $BATCH_SIZE"
echo "Num GPUs:    $NUM_GPUS  GPU_ID: $GPU_ID"
echo "Video only:  $VIDEO_ONLY"
echo "Log:         $LOG"
echo "============================================================"
} | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Step 1: 提取视频行 ===" | tee -a "$LOG"
PYTHONPATH="$REPO_ROOT/sid/src" python "$REPO_ROOT/sid/scripts/embedding/extract_video_only.py" \
  --input "$INPUT" --output "$VIDEO_ONLY" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Step 2: 生成 text_format embedding ===" | tee -a "$LOG"
INPUT="$VIDEO_ONLY" \
OUTPUT_DIR="$OUTPUT_DIR" \
MODEL_PATH="$MODEL_PATH" \
DEVICE="$DEVICE" \
BATCH_SIZE="$BATCH_SIZE" \
NUM_GPUS="$NUM_GPUS" \
GPU_ID="$GPU_ID" \
bash "$REPO_ROOT/sid/scripts/step0-generate-embedding.sh" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done." | tee -a "$LOG"
echo "产物:" | tee -a "$LOG"
echo "  $OUTPUT_DIR/text_format_embedding.npy  (embedding)" | tee -a "$LOG"
echo "  $OUTPUT_DIR/meta.jsonl                   (原始全部字段 + 仅 text_format 替换为新值)" | tee -a "$LOG"
echo "  $LOG  (日志)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
