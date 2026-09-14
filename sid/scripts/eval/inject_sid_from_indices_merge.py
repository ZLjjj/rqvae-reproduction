#!/usr/bin/env python3
"""Inject global idx + sid(=indices) into station/video raw JSONL and write a single merged JSONL.

Paths are intentionally hard-coded for one-off data preparation.

Assumptions:
- index_jsonl order: all station rows first, then all video rows.
- idx is 0-based, and video continues numbering after station.
- sid is exactly the `indices` list from index_jsonl.

Hard checks:
- station_lines + video_lines == index_lines

Output:
- /mnt/wulinyang/data/20260204_kmeans_8:2enhanced.4layers_new.jsonl
"""

import json
from pathlib import Path

STATION_RAW = Path("/mnt/wulinyang/raw_data/station_20260124.jsonl")
VIDEO_RAW = Path("/mnt/wulinyang/raw_data/video_20260126.jsonl")
INDEX_JSONL = Path(
    "/mnt/wulinyang/gensearchrec2/gensearchrec/benchmark_results_hc/20260205_204902/"
    "merged_emb.npy_rqkmeans_plus/rqkmeans_plus/merged_emb.npy_rqkmeans_plus/"
    "merged_emb.npy_rqkmeans_plus.faiss-rq.index.jsonl"
)
OUTPUT_JSONL = Path("/mnt/wulinyang/data/20260204_kmeans_8:2enhanced.4layers_new.jsonl")


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _iter_indices(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            indices = obj.get("indices")
            if not isinstance(indices, list):
                raise ValueError(f"INDEX_JSONL line {i}: missing/invalid indices")
            yield indices


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON at {path}:{i}") from e


def main() -> None:
    n_station = _count_lines(STATION_RAW)
    n_video = _count_lines(VIDEO_RAW)
    n_index = _count_lines(INDEX_JSONL)
    if n_station + n_video != n_index:
        raise ValueError(f"line mismatch: station({n_station}) + video({n_video}) != index({n_index})")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    idx = 0
    indices_it = _iter_indices(INDEX_JSONL)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as fout:
        for obj in _iter_jsonl(STATION_RAW):
            obj["idx"] = idx
            obj["sid"] = next(indices_it)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            idx += 1

        for obj in _iter_jsonl(VIDEO_RAW):
            obj["idx"] = idx
            obj["sid"] = next(indices_it)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            idx += 1

    try:
        next(indices_it)
        raise ValueError("INDEX_JSONL has extra non-empty lines beyond station+video")
    except StopIteration:
        pass

    print(str(OUTPUT_JSONL))


if __name__ == "__main__":
    main()
