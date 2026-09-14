#!/usr/bin/env bash
set -euo pipefail

# One-click script: embed only text_format for video 20260208, auto-detect GPU count, and merge outputs.
# Usage: bash /mnt/wulinyang/to_push/gensearchrec/sid/scripts/embedding/run_video_textonly.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

MODEL="/mnt/wulinyang/models/Qwen3-Embedding-4B"
OUT_ROOT="/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings"

VIDEO_INPUT="/mnt/wulinyang/raw_data/video_20260208/video_20260208.jsonl"
VIDEO_OUT="${OUT_ROOT}/video_20260208_text"

# Auto-detect GPU count (override by exporting NUM_GPUS manually)
if [[ -z "${NUM_GPUS:-}" ]]; then
  NUM_GPUS=$(python - <<'PY'
try:
    import torch
    n = torch.cuda.device_count()
    print(n if n and n > 0 else 1)
except Exception:
    try:
        import subprocess, shlex
        out = subprocess.check_output(shlex.split('nvidia-smi -L'), stderr=subprocess.DEVNULL).decode()
        n = sum(1 for line in out.splitlines() if line.strip().startswith('GPU '))
        print(n if n > 0 else 1)
    except Exception:
        print(1)
PY
  )
fi

echo "Using NUM_GPUS=${NUM_GPUS}"

RUN_GEN_SH="${SCRIPT_DIR}/run_generate_embeddings.sh"

# video text_format only (sharded if NUM_GPUS>1)
INPUT="${VIDEO_INPUT}" OUTPUT_DIR="${VIDEO_OUT}" MODEL="${MODEL}" DOMAIN=video EMB_FIELDS=text_format NUM_GPUS="${NUM_GPUS}" \
  bash "${RUN_GEN_SH}"

echo "Done. Outputs in ${VIDEO_OUT}"
