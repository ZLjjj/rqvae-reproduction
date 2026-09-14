#!/usr/bin/env bash
set -euo pipefail

# Run the two new video-only RQVAE experiments and then evaluate all three
# systems with one metric protocol. Execute on the CUDA training host.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SID="$ROOT/sid"
DATA_ROOT="${DATA_ROOT:-$ROOT}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
WORK="${WORK:-$ROOT/results_three_way}"

BASELINE_INDEX="${BASELINE_INDEX:-$DATA_ROOT/1_vs_Qwen4B_rqvae_hc_ema.index.meta.jsonl}"
BASELINE_EMB="${BASELINE_EMB:-$DATA_ROOT/2_meta_full_video.npy}"
SAME_META="${SAME_META:-$DATA_ROOT/2_meta_full_video.jsonl}"
SAME_EMB="${SAME_EMB:-$DATA_ROOT/2_meta_full_video.npy}"
V3_META="${V3_META:-$DATA_ROOT/3_meta.jsonl}"
V3_EMB="${V3_EMB:-$DATA_ROOT/3_text_format_embedding.npy}"

train_and_infer() {
  local name="$1" meta="$2" emb="$3"
  local out="$WORK/$name"
  mkdir -p "$out/checkpoints"
  python "$SID/scripts/train/train_rqvae.py" \
    --data_npy "$emb" --meta_jsonl "$meta" --ckpt_dir "$out/checkpoints" \
    --device "$DEVICE" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
    --eval_step 1 --learner AdamW --lr_scheduler_type constant \
    --kmeans_init --kmeans_init_dedup --ema_codebook --ema_decay 0.99 --ema_eps 1e-5 \
    --num_emb_list 1024 1024 1024 1024 256 --e_dim 128 \
    --layers 512 256 128 --sk_epsilons 0 0 0 0.003 0
  python "$SID/src/codebook/generate_indices_rqvae.py" \
    --ckpt "$out/checkpoints/best_collision_model.pth" --data_npy "$emb" \
    --meta_jsonl "$meta" --model rqvae --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" --output_dir "$out/indices"
  cp "$out/indices/indices.jsonl" "$out/$name.index.jsonl"
}

train_and_infer "same_source_video" "$SAME_META" "$SAME_EMB"
train_and_infer "video_v3" "$V3_META" "$V3_EMB"

python "$SID/scripts/eval/run_sid_three_way.py" \
  --baseline-index "$BASELINE_INDEX" --baseline-emb "$BASELINE_EMB" \
  --same-source-index "$WORK/same_source_video/same_source_video.index.jsonl" \
  --same-source-emb "$SAME_EMB" \
  --v3-index "$WORK/video_v3/video_v3.index.jsonl" --v3-emb "$V3_EMB" \
  --output-dir "$WORK/evaluation"
