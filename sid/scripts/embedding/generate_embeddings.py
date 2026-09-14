#!/usr/bin/env python3
"""
Convenience launcher for embedding generation.

Examples:
  # station
  python sid/scripts/embedding/generate_embeddings.py \
    --input /path/to/station.jsonl \
    --domain station \
    --output_dir /mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings \
    --num_gpus 1 --gpu_id 0

  # video
  python sid/scripts/embedding/generate_embeddings.py \
    --input /path/to/video.jsonl \
    --domain video \
    --output_dir /mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings \
    --num_gpus 1 --gpu_id 0
"""
import sys
from pathlib import Path

# Add sid/src to path
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[2]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from embedding.processor import main_cli

if __name__ == "__main__":
    main_cli()
