#!/usr/bin/env python3
"""Convert 5-layer indices JSONL to sid JSONL (minimal change).

Input:
- /mnt/wulinyang/data/20260204_kmeans_8:2enhanced.jsonl
  Each line is a JSON object containing at least: idx, indices, meta_json (top-level), ...

Output:
- /mnt/wulinyang/data/20260204_kmeans_8:2enhanced.5layers_new.with_meta_json.jsonl

Transform per line:
- Add sid = indices
- Remove indices
- Optionally coerce idx string -> int
- Keep all other fields untouched

Also normalizes meta_json:
- Prefer top-level meta_json
- If missing, fallback to meta.meta_json when meta is a dict
"""

import json
from pathlib import Path

IN_PATH = Path("/mnt/wulinyang/data/20260204_kmeans_8:2enhanced.jsonl")
OUT_PATH = Path("/mnt/wulinyang/data/20260204_kmeans_8:2enhanced.5layers_new.with_meta_json.jsonl")
COERCE_IDX_TO_INT = True


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            indices = obj.get("indices")
            if not isinstance(indices, list):
                raise ValueError(f"Line {line_no}: missing/invalid indices")

            obj["sid"] = indices
            obj.pop("indices", None)

            if COERCE_IDX_TO_INT and "idx" in obj and isinstance(obj["idx"], str):
                try:
                    obj["idx"] = int(obj["idx"])
                except Exception as e:
                    raise ValueError(f"Line {line_no}: idx is not int-like: {obj['idx']!r}") from e

            if obj.get("meta_json") is None:
                meta = obj.get("meta")
                if isinstance(meta, dict) and meta.get("meta_json") is not None:
                    obj["meta_json"] = meta.get("meta_json")

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(str(OUT_PATH))


if __name__ == "__main__":
    main()
