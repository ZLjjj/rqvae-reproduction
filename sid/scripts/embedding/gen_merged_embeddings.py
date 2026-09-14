#!/usr/bin/env python3
"""Generate embeddings from merged station+video meta jsonl file.

Input: merged jsonl file path (station and video data already combined)
Output: npy embedding file in same directory as input, same name, .npy suffix

Default behavior: auto-detect all available *visible* GPUs and use all of them
(one process per GPU). Disable with --no_spawn.

Notes:
- text field fixed to "text_format"
- batch_size fixed to 64
"""

import argparse
import json
import subprocess
import sys
import torch
from pathlib import Path
from typing import Optional, NamedTuple

import numpy as np

# Make sid/src importable
_THIS_DIR = Path(__file__).resolve().parent
_SID_DIR = _THIS_DIR.parents[1]
_SRC_DIR = _SID_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from embedding.processor import load_model, compute_shard


class ShardRange(NamedTuple):
    start: int
    end: int


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


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


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
    p = argparse.ArgumentParser(description="Generate embeddings from merged station+video jsonl")
    p.add_argument(
        "--input",
        default="/mnt/zhanggehang1/wulinyang1/raw_data/sidflow/meta.jsonl",
        help="Path to merged station+video jsonl file",
    )
    p.add_argument(
        "--model_path",
        default="/mnt/zhanggehang1/wulinyang1/models/Qwen3-Embedding-4B",
        help="Embedding model path",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output .npy path (default: same directory/name as --input, with .npy suffix)",
    )
    p.add_argument("--max_rows", type=int, default=None, help="Debug: limit rows to process")
    p.add_argument("--domain", default=None, choices=["video", "station"], help="只生成指定 domain 的 embedding（默认全部）")

    # By default, use all visible GPUs (one process per GPU). Disable with --no_spawn.
    p.add_argument(
        "--no_spawn",
        action="store_true",
        help="Disable multi-GPU parallel processing (run a single process on GPU 0)",
    )

    # Internal shard params (not for end user)
    p.add_argument("--num_gpus", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--gpu_id", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--merge", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--spawn", action="store_true", help=argparse.SUPPRESS)

    return p.parse_args()


def _run_shard(args: argparse.Namespace, input_path: Path, output_dir: Path) -> None:
    BATCH_SIZE = 16
    TEXT_FIELD = "text_format"

    model, embed_dim, _ = load_model(args.model_path, device=f"cuda:{args.gpu_id}")
    total = _count_lines(input_path)
    shard = compute_shard(total, args.num_gpus, args.gpu_id)
    
    # Read text data for this shard
    texts: list[Optional[str]] = []
    for obj in _iter_jsonl(input_path, max_rows=args.max_rows, start=shard.start, end=shard.end):
        if getattr(args, "domain", None) and obj.get("domain") != args.domain:
            continue
        text = obj.get(TEXT_FIELD)
        texts.append(text)

    # Generate embeddings
    embeddings = _embed_field(
        model,
        texts,
        embed_dim=embed_dim,
        batch_size=BATCH_SIZE,
        desc=f"GPU {args.gpu_id}: Embedding shard",
    )

    # Save shard output
    np.save(output_dir / f"embedding_gpu{args.gpu_id}.npy", embeddings)


def main() -> None:
    args = parse_args()

    # Resolve paths
    raw_input = args.input
    if not isinstance(raw_input, str) or not raw_input.strip() or raw_input.strip() == "/":
        raise ValueError(f"Invalid --input: {raw_input!r} (expected a non-empty file path)")

    input_path = Path(raw_input).resolve()
    if input_path == Path("/"):
        raise ValueError(f"Invalid --input resolves to root: {raw_input!r}")

    if args.output is None:
        output_path = input_path.with_suffix(".npy")
    else:
        output_path = Path(args.output).resolve()
        if output_path == Path("/"):
            raise ValueError(f"Invalid --output resolves to root: {args.output!r}")

    temp_dir = input_path.parent / f".embedding_tmp_{input_path.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Auto detect GPUs (default: enabled). Use all *visible* GPUs.
    args.spawn = args.spawn or (not args.no_spawn)
    actual_num_gpus = args.num_gpus

    if args.spawn:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("Warning: No GPUs detected, falling back to CPU processing", file=sys.stderr)
            num_gpus = 1

        print(f"Detected {num_gpus} GPUs, starting parallel processing...")

        # Spawn one process per GPU
        procs = []
        for gid in range(num_gpus):
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--input",
                str(input_path),
                "--model_path",
                str(args.model_path),
                "--output",
                str(output_path),
                "--num_gpus",
                str(num_gpus),
                "--gpu_id",
                str(gid),
                "--no_spawn",
            ]
            if args.max_rows is not None:
                cmd += ["--max_rows", str(args.max_rows)]
            if args.domain:
                cmd += ["--domain", str(args.domain)]
            procs.append(subprocess.Popen(cmd))

        # Wait for all shards to complete
        for idx, p in enumerate(procs):
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"Shard process {idx} failed with code {rc}")

        args.merge = True
        # 保存实际使用的GPU数量用于合并
        actual_num_gpus = num_gpus
    if args.merge:
        # Merge all shards
        num_gpus = actual_num_gpus if args.spawn else (args.num_gpus if args.num_gpus > 0 else torch.cuda.device_count())
        shards = []
        for gid in range(num_gpus):
            shard_path = temp_dir / f"embedding_gpu{gid}.npy"
            shards.append(np.load(shard_path))
        
        merged = np.concatenate(shards, axis=0)
        np.save(output_path, merged)
        
        # Clean up temp dir
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"✅ Success! Embeddings saved to: {output_path}")
        print(f"📊 Shape: {merged.shape}")
        return

    # Run single shard (called by spawned processes)
    _run_shard(args, input_path, temp_dir)


if __name__ == "__main__":
    main()
