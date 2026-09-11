#!/usr/bin/env python3
"""Trim 5-layer sid to 4 layers for meta-string JSONL.

Input:
- /mnt/wulinyang/data/20260204_kmeans_8:2enhanced.5layers_new.meta_string.with_meta_json.jsonl

Output:
- /mnt/wulinyang/data/20260204_kmeans_8:2enhanced.4layers_new.meta_string.with_meta_json.jsonl

Per line:
- sid = sid[:4]
- keep all other fields unchanged
"""

import json
from pathlib import Path

IN_PATH = Path("/mnt/wulinyang/data/20260204_kmeans_8:2enhanced.5layers_new.meta_string.with_meta_json.jsonl")
OUT_PATH = Path("/mnt/wulinyang/data/20260204_kmeans_8:2enhanced.4layers_new.meta_string.with_meta_json.jsonl")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = obj.get("sid")
            if not isinstance(sid, list):
                raise ValueError(f"Line {line_no}: missing/invalid sid")
            if len(sid) < 4:
                raise ValueError(f"Line {line_no}: sid length < 4: {len(sid)}")
            obj["sid"] = sid[:4]
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(str(OUT_PATH))


if __name__ == "__main__":
    main()
