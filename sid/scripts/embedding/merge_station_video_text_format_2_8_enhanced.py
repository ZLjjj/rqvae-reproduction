#!/usr/bin/env python3
"""Merge station/video embeddings with a 2:8 enhancement.

Station enhanced embedding:
  enhanced = 0.2 * mean(name, subname, tags) + 0.8 * text_format

Then concatenate with video text_format embedding by rows:
  merged = [station_enhanced; video_text_format]

Writes a single output file:
- /mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings/station_video_20260208_text_2:8enhanced.npy

Meta is NOT written; reuse existing merged meta.jsonl externally.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np


DEFAULT_OUT_NPY = "/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings/station_video_20260208_text_2:8enhanced.npy"

DEFAULT_STATION_NAME = (
    "/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings/station_20260208_text/"
    "single_run_20260226_203847/name_embedding_gpu0.npy"
)
DEFAULT_STATION_SUBNAME = (
    "/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings/station_20260208_text/"
    "single_run_20260226_203847/subname_embedding_gpu0.npy"
)
DEFAULT_STATION_TAGS = (
    "/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings/station_20260208_text/"
    "single_run_20260226_203847/tags_embedding_gpu0.npy"
)
DEFAULT_STATION_TEXT_FORMAT = "/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings/station_20260208_text/text_format_embedding.npy"
DEFAULT_VIDEO_TEXT_FORMAT = "/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings/video_20260208_text/text_format_embedding.npy"

DEFAULT_STATION_META = "/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings/station_20260208_text/meta.jsonl"
DEFAULT_VIDEO_META = "/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings/video_20260208_text/meta.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge station/video embeddings with 2:8 enhancement")

    p.add_argument("--station_name", default=DEFAULT_STATION_NAME)
    p.add_argument("--station_subname", default=DEFAULT_STATION_SUBNAME)
    p.add_argument("--station_tags", default=DEFAULT_STATION_TAGS)
    p.add_argument("--station_text_format", default=DEFAULT_STATION_TEXT_FORMAT)
    p.add_argument("--video_text_format", default=DEFAULT_VIDEO_TEXT_FORMAT)

    p.add_argument("--alpha", type=float, default=0.2, help="Weight for mean(name,subname,tags)")
    p.add_argument("--beta", type=float, default=0.8, help="Weight for station text_format")

    p.add_argument("--output_npy", default=DEFAULT_OUT_NPY)
    p.add_argument("--overwrite", action="store_true", help="Overwrite output_npy if exists")
    p.add_argument("--chunk_rows", type=int, default=8192, help="Rows per chunk for streaming compute")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_path = Path(args.output_npy).resolve()
    if out_path.exists() and args.overwrite:
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    name = np.load(args.station_name, mmap_mode="r")
    subname = np.load(args.station_subname, mmap_mode="r")
    tags = np.load(args.station_tags, mmap_mode="r")
    st_tf = np.load(args.station_text_format, mmap_mode="r")
    v_tf = np.load(args.video_text_format, mmap_mode="r")

    if name.shape != subname.shape or name.shape != tags.shape or name.shape != st_tf.shape:
        raise ValueError(
            f"station shapes mismatch: name={name.shape}, subname={subname.shape}, tags={tags.shape}, text_format={st_tf.shape}"
        )
    if name.dtype != np.float32 or subname.dtype != np.float32 or tags.dtype != np.float32 or st_tf.dtype != np.float32:
        raise ValueError("Expected station embeddings dtype=float32")
    if v_tf.dtype != np.float32:
        raise ValueError("Expected video embeddings dtype=float32")

    n_station, dim = name.shape
    n_video, dim2 = v_tf.shape
    if dim2 != dim:
        raise ValueError(f"Dim mismatch: station dim={dim}, video dim={dim2}")

    alpha = float(args.alpha)
    beta = float(args.beta)
    if not np.isclose(alpha + beta, 1.0):
        # allow non-normalized but warn by raising explicit error to avoid silent mistakes
        raise ValueError(f"alpha+beta must be 1.0, got {alpha}+{beta}={alpha+beta}")

    merged_path = out_path

    # pre-create output file via open_memmap
    merged = np.lib.format.open_memmap(
        merged_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_station + n_video, dim),
    )

    chunk = int(args.chunk_rows)
    for s in range(0, n_station, chunk):
        e = min(n_station, s + chunk)
        avg = (name[s:e] + subname[s:e] + tags[s:e]) / 3.0
        merged[s:e] = alpha * avg + beta * st_tf[s:e]

    for s in range(0, n_video, chunk):
        e = min(n_video, s + chunk)
        merged[n_station + s : n_station + e] = v_tf[s:e]

    merged.flush()

    print(str(out_path))


if __name__ == "__main__":
    main()
