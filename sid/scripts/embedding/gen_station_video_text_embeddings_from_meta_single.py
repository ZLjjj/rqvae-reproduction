#!/usr/bin/env python3
"""Single-GPU convenience runner.

Writes outputs into timestamped subdirectories to avoid clobbering multi-GPU shard outputs.

It reuses `gen_station_video_text_embeddings_from_meta.py` by invoking it with:
- --num_gpus 1 --gpu_id 0
- custom output dirs under:
  station_20260208_text/single_run_<ts>/
  video_20260208_text/single_run_<ts>/
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run single-GPU embedding generation into a non-conflicting output folder")
    p.add_argument("--device", default="cuda:0", help="Device for the single run (e.g. cuda:0 or cpu)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_rows", type=int, default=None, help="Debug: limit rows")
    p.add_argument("--run_name", default=None, help="Subfolder name (default: timestamp)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base = Path(__file__).resolve().parent
    runner = base / "gen_station_video_text_embeddings_from_meta.py"

    run_name = args.run_name or datetime.now().strftime("single_run_%Y%m%d_%H%M%S")

    station_base = Path("/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings/station_20260208_text")
    video_base = Path("/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings/video_20260208_text")

    station_out = station_base / run_name
    video_out = video_base / run_name

    cmd = [
        sys.executable,
        str(runner),
        "--num_gpus",
        "1",
        "--gpu_id",
        "0",
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--station_output_dir",
        str(station_out),
        "--video_output_dir",
        str(video_out),
    ]

    if args.max_rows is not None:
        cmd += ["--max_rows", str(args.max_rows)]

    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
