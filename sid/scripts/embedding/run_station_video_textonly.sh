#!/usr/bin/env bash
set -euo pipefail

# One-click script: embed only text_format for station+video 20260208, auto-detect GPU count, and merge outputs.
# Can be run from anywhere: bash /mnt/wulinyang/to_push/gensearchrec/sid/scripts/embedding/run_station_video_textonly.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

MODEL="/mnt/zhanggehang1/wulinyang1/models/Qwen3-Embedding-4B-station-video-sft"
OUT_ROOT="/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings"

STATION_INPUT="/mnt/zhanggehang1/wulinyang1/raw_data/station_20260208/station_20260208.jsonl"
VIDEO_INPUT="/mnt/zhanggehang1/wulinyang1/raw_data/video_20260208/video_20260208.jsonl"
STATION_OUT="${OUT_ROOT}/station_20260208_text_sft"
VIDEO_OUT="${OUT_ROOT}/video_20260208_text_sft"
MERGED_OUT="${OUT_ROOT}/station_video_20260208_text_sft"

# Auto-detect GPU count (can override by exporting NUM_GPUS manually)
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

# 1) station text_format only (sharded if NUM_GPUS>1)
INPUT="${STATION_INPUT}" OUTPUT_DIR="${STATION_OUT}" MODEL="${MODEL}" DOMAIN=station EMB_FIELDS=text_format NUM_GPUS="${NUM_GPUS}" \
  bash "${RUN_GEN_SH}"

# 2) video text_format only
INPUT="${VIDEO_INPUT}" OUTPUT_DIR="${VIDEO_OUT}" MODEL="${MODEL}" DOMAIN=video EMB_FIELDS=text_format NUM_GPUS="${NUM_GPUS}" \
  bash "${RUN_GEN_SH}"

# 3) merge station+video into a single text_format_embedding.npy and meta.jsonl
mkdir -p "${MERGED_OUT}"
python - <<'PY'
import json
import numpy as np
from pathlib import Path

station_out = Path("${STATION_OUT}")
video_out = Path("${VIDEO_OUT}")
merged_out = Path("${MERGED_OUT}")

st_emb = np.load(station_out / "text_format_embedding.npy")
vd_emb = np.load(video_out / "text_format_embedding.npy")
merged = np.concatenate([st_emb, vd_emb], axis=0)
np.save(merged_out / "text_format_embedding.npy", merged)

meta_path = merged_out / "meta.jsonl"
with meta_path.open("w", encoding="utf-8") as fout:
    for path in [station_out / "meta.jsonl", video_out / "meta.jsonl"]:
        with Path(path).open("r", encoding="utf-8") as fin:
            for line in fin:
                obj = json.loads(line)
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Merged embedding shape: {merged.shape}")
print(f"Merged meta: {meta_path}")
PY

echo "Done. Merged outputs in ${MERGED_OUT}"
