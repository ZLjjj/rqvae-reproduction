#!/usr/bin/env python3
"""Build 5-layer merged station+video JSONL from raw sources + enhanced sid/meta_json reference.

Goal:
- Base fields come from raw source files:
  - /mnt/wulinyang/raw_data/station_20260124.jsonl
  - /mnt/wulinyang/raw_data/video_20260126.jsonl
- idx is 0-based and video continues numbering.
- sid (5 layers) and meta_json are taken line-by-line from:
  - /mnt/wulinyang/data/20260204_kmeans_8:2enhanced.jsonl
    (uses top-level meta_json and indices)

Output:
- /mnt/wulinyang/data/20260204_kmeans_8:2enhanced.5layers_new.with_meta_json.jsonl

Hard checks:
- station_lines + video_lines == ref_lines
- each ref line must contain indices (len==5) and meta_json (top-level)
"""

import json
from pathlib import Path

STATION_RAW = Path("/mnt/wulinyang/raw_data/station_20260124.jsonl")
VIDEO_RAW = Path("/mnt/wulinyang/raw_data/video_20260126.jsonl")
REF_JSONL = Path("/mnt/wulinyang/data/20260204_kmeans_8:2enhanced.jsonl")
OUT_JSONL = Path("/mnt/wulinyang/data/20260204_kmeans_8:2enhanced.5layers_new.meta_string.with_meta_json.jsonl")


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


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
    n_ref = _count_lines(REF_JSONL)
    if n_station + n_video != n_ref:
        raise ValueError(f"line mismatch: station({n_station}) + video({n_video}) != ref({n_ref})")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    ref_it = _iter_jsonl(REF_JSONL)
    idx = 0

    with OUT_JSONL.open("w", encoding="utf-8") as fout:
        for base in _iter_jsonl(STATION_RAW):
            ref = next(ref_it)
            indices = ref.get("indices")
            if not isinstance(indices, list) or len(indices) != 5:
                raise ValueError(f"REF line {idx+1}: expected indices list len=5")
            meta_json = ref.get("meta_json")
            if meta_json is None:
                raise ValueError(f"REF line {idx+1}: missing top-level meta_json")

            meta_json = dict(meta_json)
            meta_json.setdefault("publish_year", "未知")

            out_obj = {
                "idx": str(idx),
                "id": base.get("id"),
                "domain": "station",
                "meta": json.dumps(base, ensure_ascii=False),
                "meta_json": meta_json,
                "text_format": base.get("text_format"),
                "json_format": base.get("json_format"),
                "sid": indices,
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            idx += 1

        for base in _iter_jsonl(VIDEO_RAW):
            ref = next(ref_it)
            indices = ref.get("indices")
            if not isinstance(indices, list) or len(indices) != 5:
                raise ValueError(f"REF line {idx+1}: expected indices list len=5")
            meta_json = ref.get("meta_json")
            if meta_json is None:
                raise ValueError(f"REF line {idx+1}: missing top-level meta_json")

            meta_json = dict(meta_json)
            meta_json.setdefault("publish_year", "未知")

            out_obj = {
                "idx": str(idx),
                "id": base.get("id"),
                "domain": "video",
                "meta": json.dumps(base, ensure_ascii=False),
                "meta_json": meta_json,
                "text_format": base.get("text_format"),
                "json_format": base.get("json_format"),
                "sid": indices,
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            idx += 1

    try:
        next(ref_it)
        raise ValueError("REF_JSONL has extra non-empty lines beyond station+video")
    except StopIteration:
        pass

    print(str(OUT_JSONL))


if __name__ == "__main__":
    main()
