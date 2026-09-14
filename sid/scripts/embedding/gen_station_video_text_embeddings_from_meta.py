#!/usr/bin/env python3
"""Generate selected text embeddings from existing meta.jsonl files.

This script is a convenience wrapper around `embedding.processor` logic, but uses the
already-exported `meta.jsonl` as the input source.

Outputs:
- station: name_embedding.npy, subname_embedding.npy, tags_embedding.npy
- video:   main_name_embedding.npy

Default paths are hard-coded for the 20260208 dataset; CLI flags can override.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Make sid/src importable
_THIS_DIR = Path(__file__).resolve().parent
_SID_DIR = _THIS_DIR.parents[1]
_SRC_DIR = _SID_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from embedding.processor import compute_shard, extract_fields, resolve_main_name, load_model


def _iter_jsonl(path: Path, max_rows: Optional[int] = None, *, start: int = 0, end: Optional[int] = None):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if end is not None and i >= end:
                break
            if max_rows is not None and (i - start) >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _embed_field(model, texts: list[Optional[str]], *, embed_dim: int, batch_size: int, desc: str) -> np.ndarray:
    cleaned = [t if (isinstance(t, str) and t.strip()) else None for t in texts]
    to_encode_idx = [i for i, t in enumerate(cleaned) if t is not None]

    if not to_encode_idx:
        return np.zeros((len(texts), embed_dim), dtype=np.float32)

    to_encode = [cleaned[i] for i in to_encode_idx]
    embeds = model.encode(
        to_encode,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
        desc=desc,
    ).astype(np.float32, copy=False)

    out = np.zeros((len(texts), embeds.shape[1]), dtype=np.float32)
    for pos, row_idx in enumerate(to_encode_idx):
        out[row_idx] = embeds[pos]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate station/video text embeddings from existing meta.jsonl")

    p.add_argument(
        "--model_name_or_path",
        default="/mnt/wulinyang/models/Qwen3-Embedding-4B",
        help="SentenceTransformer model path/name",
    )
    p.add_argument("--device", default=None, help="Device, e.g. cuda:0 (default: auto)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_rows", type=int, default=None, help="Debug: limit rows per GPU shard")

    p.add_argument("--num_gpus", type=int, default=1, help="Total number of GPUs (for sharding)")
    p.add_argument("--gpu_id", type=int, default=0, help="This process GPU id [0..num_gpus-1]")
    p.add_argument("--merge", action="store_true", help="Only merge shard outputs and exit")
    p.add_argument("--spawn", action="store_true", help="Spawn one subprocess per GPU (single command multi-GPU run)")

    p.add_argument(
        "--station_meta_jsonl",
        default="/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings/station_20260208_text/meta.jsonl",
    )
    p.add_argument(
        "--video_meta_jsonl",
        default="/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings/video_20260208_text/meta.jsonl",
    )

    p.add_argument(
        "--station_output_dir",
        default="/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings/station_20260208_text",
    )
    p.add_argument(
        "--video_output_dir",
        default="/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings/video_20260208_text",
    )

    return p.parse_args()


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _merge_embeddings(out_dir: Path, *, fields: list[str], num_gpus: int) -> None:
    # embeddings
    for field in fields:
        parts = [np.load(out_dir / f"{field}_embedding_gpu{gid}.npy") for gid in range(num_gpus)]
        merged = np.concatenate(parts, axis=0)
        np.save(out_dir / f"{field}_embedding.npy", merged)


def _merge_meta(out_dir: Path, *, num_gpus: int) -> None:
    merged = out_dir / "meta.jsonl"
    with merged.open("w", encoding="utf-8") as fout:
        for gid in range(num_gpus):
            shard = out_dir / f"meta_gpu{gid}.jsonl"
            with shard.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)


def _run_shard(args: argparse.Namespace) -> None:
    model, embed_dim, _ = load_model(args.model_name_or_path, device=args.device)

    station_path = Path(args.station_meta_jsonl).resolve()
    video_path = Path(args.video_meta_jsonl).resolve()

    station_out = Path(args.station_output_dir).resolve()
    video_out = Path(args.video_output_dir).resolve()
    station_out.mkdir(parents=True, exist_ok=True)
    video_out.mkdir(parents=True, exist_ok=True)

    # shard ranges are computed independently for station/video to preserve per-file alignment
    station_total = _count_lines(station_path)
    video_total = _count_lines(video_path)
    station_shard = compute_shard(station_total, args.num_gpus, args.gpu_id)
    video_shard = compute_shard(video_total, args.num_gpus, args.gpu_id)

    # ----- station shard -----
    station_name: list[Optional[str]] = []
    station_subname: list[Optional[str]] = []
    station_tags: list[Optional[str]] = []
    station_meta_lines: list[str] = []

    for obj in _iter_jsonl(station_path, max_rows=args.max_rows, start=station_shard.start, end=station_shard.end):
        meta = obj.get("meta") or {}
        station_name.append(meta.get("name") or meta.get("mainname") or meta.get("rawname"))
        station_subname.append(meta.get("subname") or "")
        extracted = extract_fields(obj)
        station_tags.append(extracted.get("tags"))
        station_meta_lines.append(json.dumps(obj, ensure_ascii=False))

    np.save(
        station_out / f"name_embedding_gpu{args.gpu_id}.npy",
        _embed_field(model, station_name, embed_dim=embed_dim, batch_size=args.batch_size, desc=f"station:name:gpu{args.gpu_id}"),
    )
    np.save(
        station_out / f"subname_embedding_gpu{args.gpu_id}.npy",
        _embed_field(model, station_subname, embed_dim=embed_dim, batch_size=args.batch_size, desc=f"station:subname:gpu{args.gpu_id}"),
    )
    np.save(
        station_out / f"tags_embedding_gpu{args.gpu_id}.npy",
        _embed_field(model, station_tags, embed_dim=embed_dim, batch_size=args.batch_size, desc=f"station:tags:gpu{args.gpu_id}"),
    )

    with (station_out / f"meta_gpu{args.gpu_id}.jsonl").open("w", encoding="utf-8") as f:
        for line in station_meta_lines:
            f.write(line + "\n")

    # ----- video shard -----
    video_main_name: list[Optional[str]] = []
    video_meta_lines: list[str] = []

    for obj in _iter_jsonl(video_path, max_rows=args.max_rows, start=video_shard.start, end=video_shard.end):
        meta = obj.get("meta") or {}
        obj2 = dict(obj)
        obj2["meta_main_name"] = meta.get("meta_main_name")
        obj2["meta_origin_name"] = meta.get("meta_origin_name")
        obj2["name"] = meta.get("meta_main_name") or meta.get("meta_origin_name")
        extracted = extract_fields(obj2)
        video_main_name.append(resolve_main_name(obj2, extracted))
        video_meta_lines.append(json.dumps(obj, ensure_ascii=False))

    np.save(
        video_out / f"main_name_embedding_gpu{args.gpu_id}.npy",
        _embed_field(model, video_main_name, embed_dim=embed_dim, batch_size=args.batch_size, desc=f"video:main_name:gpu{args.gpu_id}"),
    )

    with (video_out / f"meta_gpu{args.gpu_id}.jsonl").open("w", encoding="utf-8") as f:
        for line in video_meta_lines:
            f.write(line + "\n")


def main() -> None:
    args = parse_args()

    if args.spawn:
        # Fire one process per GPU and then merge.
        procs = []
        for gid in range(args.num_gpus):
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--num_gpus",
                str(args.num_gpus),
                "--gpu_id",
                str(gid),
                "--model_name_or_path",
                args.model_name_or_path,
                "--device",
                f"cuda:{gid}",
                "--batch_size",
                str(args.batch_size),
            ]
            if args.max_rows is not None:
                cmd += ["--max_rows", str(args.max_rows)]
            cmd += [
                "--station_meta_jsonl",
                args.station_meta_jsonl,
                "--video_meta_jsonl",
                args.video_meta_jsonl,
                "--station_output_dir",
                args.station_output_dir,
                "--video_output_dir",
                args.video_output_dir,
            ]
            procs.append(subprocess.Popen(cmd))

        for p in procs:
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"Shard process failed with code {rc}")

        args.merge = True

    if args.merge:
        station_out = Path(args.station_output_dir).resolve()
        video_out = Path(args.video_output_dir).resolve()
        _merge_embeddings(station_out, fields=["name", "subname", "tags"], num_gpus=args.num_gpus)
        _merge_meta(station_out, num_gpus=args.num_gpus)
        _merge_embeddings(video_out, fields=["main_name"], num_gpus=args.num_gpus)
        _merge_meta(video_out, num_gpus=args.num_gpus)
        print(str(station_out))
        print(str(video_out))
        return

    _run_shard(args)


if __name__ == "__main__":
    main()
