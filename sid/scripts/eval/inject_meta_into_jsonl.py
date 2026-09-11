#!/usr/bin/env python3
"""Inject per-line meta into an existing JSONL.

Reads two JSONL files with the same number/order of lines:
- src_jsonl: each line is a JSON object (dict)
- meta_jsonl: each line is a JSON object used as the `meta` field

Writes a new JSONL where each output line is:
  obj = src_obj; obj["meta"] = meta_obj

Note: This overwrites any existing `meta` field in src_jsonl.
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inject meta_jsonl into src_jsonl as per-line obj['meta']")
    p.add_argument(
        "--src_jsonl",
        default=(
            "/mnt/wulinyang/gensearchrec2/gensearchrec/benchmark_results_hc/20260205_204902/"
            "merged_emb.npy_rqkmeans_plus/rqkmeans_plus/merged_emb.npy_rqkmeans_plus/"
            "merged_emb.npy_rqkmeans_plus.faiss-rq.index.meta.jsonl"
        ),
    )
    p.add_argument(
        "--meta_jsonl",
        default="/mnt/wulinyang/to_push/gensearchrec/sid/data/meta/station_video_4B_text_format.jsonl",
    )
    p.add_argument(
        "--output_jsonl",
        default=(
            "/mnt/wulinyang/gensearchrec2/gensearchrec/benchmark_results_hc/20260205_204902/"
            "merged_emb.npy_rqkmeans_plus/rqkmeans_plus/merged_emb.npy_rqkmeans_plus/"
            "merged_emb.npy_rqkmeans_plus.faiss-rq.index.meta.jsonl"
        ),
    )
    p.add_argument(
        "--error_on_length_mismatch",
        action="store_true",
        help="If set, raise when line counts mismatch; otherwise stop at shortest file.",
    )
    return p.parse_args()


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def main() -> None:
    args = parse_args()

    src_path = Path(args.src_jsonl).resolve()
    meta_path = Path(args.meta_jsonl).resolve()
    out_path = Path(args.output_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.error_on_length_mismatch:
        n1 = _count_lines(src_path)
        n2 = _count_lines(meta_path)
        if n1 != n2:
            raise ValueError(f"Line count mismatch: src={n1}, meta={n2}")

    with src_path.open("r", encoding="utf-8") as f_src, meta_path.open("r", encoding="utf-8") as f_meta, out_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        for i, (l_src, l_meta) in enumerate(zip(f_src, f_meta), start=1):
            l_src = l_src.strip()
            l_meta = l_meta.strip()
            if not l_src or not l_meta:
                raise ValueError(f"Empty line at {i}")

            obj = json.loads(l_src)
            obj_meta = json.loads(l_meta)
            obj["meta"] = obj_meta
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(str(out_path))


if __name__ == "__main__":
    main()
