#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for embedding generation (codebook workflows can call this first)
# Override via env vars before running.
INPUT=${INPUT:-"/path/to/input.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings"}
MODEL=${MODEL:-"/mnt/wulinyang/models/Qwen3-Embedding-4B"}
DOMAIN=${DOMAIN:-"station"}          # station or video
NUM_GPUS=${NUM_GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-512}
EMB_FIELDS=${EMB_FIELDS:-""}          # leave empty to use domain defaults
HARD_FIELDS=${HARD_FIELDS:-""}        # optional extra meta fields
DEVICE_PREFIX=${DEVICE_PREFIX:-"cuda"} # e.g., "cuda" or "cuda:"; set empty to let auto-detect
DELETE_PARTIALS=${DELETE_PARTIALS:-1}  # 1 to delete shard files after merge
LOG_LEVEL=${LOG_LEVEL:-INFO}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

run_shard() {
  local gid=$1
  local device_arg=""
  if [[ -n "${DEVICE_PREFIX}" ]]; then
    device_arg="--device ${DEVICE_PREFIX%:}:$gid"
  fi
  local emb_arg=""
  if [[ -n "${EMB_FIELDS}" ]]; then
    emb_arg="--embedding_fields ${EMB_FIELDS}"
  fi
  local hard_arg=""
  if [[ -n "${HARD_FIELDS}" ]]; then
    hard_arg="--hardcode_fields ${HARD_FIELDS}"
  fi

  echo "[shard ${gid}] starting..."
  python sid/scripts/embedding/generate_embeddings.py \
    --input "${INPUT}" \
    --output_dir "${OUTPUT_DIR}" \
    --domain "${DOMAIN}" \
    --model_name_or_path "${MODEL}" \
    ${device_arg} \
    --batch_size "${BATCH_SIZE}" \
    --max_seq_length "${MAX_SEQ_LEN}" \
    --num_gpus "${NUM_GPUS}" \
    --gpu_id "${gid}" \
    ${emb_arg} \
    ${hard_arg} \
    --log_level "${LOG_LEVEL}" &
}

# Launch shards
for gid in $(seq 0 $((NUM_GPUS - 1))); do
  run_shard "${gid}"
done
wait

delete_flag=""
if [[ "${DELETE_PARTIALS}" == "1" ]]; then
  delete_flag="--delete_partials"
fi

emb_arg=""
if [[ -n "${EMB_FIELDS}" ]]; then
  emb_arg="--embedding_fields ${EMB_FIELDS}"
fi

# Merge shards
python sid/scripts/embedding/generate_embeddings.py \
  --output_dir "${OUTPUT_DIR}" \
  --domain "${DOMAIN}" \
  --num_gpus "${NUM_GPUS}" \
  --merge ${delete_flag} \
  ${emb_arg} \
  --log_level "${LOG_LEVEL}"

echo "All done. Outputs in ${OUTPUT_DIR}"
