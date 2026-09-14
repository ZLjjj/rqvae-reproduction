#!/usr/bin/env python3
"""Inject meta_json from one JSONL into another JSONL line-by-line.

- Reads src_jsonl (base objects to keep)
- Reads meta_src_jsonl (provides meta_json per line)
- Writes output_jsonl where each line is src_obj with src_obj['meta_json'] replaced/added

Hard checks:
- line counts must match (non-empty lines are assumed 1:1; both files should be "dense" JSONL)

Paths are hard-coded for one-off usage.
"""

import json
from pathlib import Path

BASE_JSONL = Path("/mnt/wulinyang/data/20260204_kmeans_8:2enhanced.4layers_new.jsonl")
META_JSONL = Path("/mnt/wulinyang/data/20260204_kmeans_8:2enhanced.4layers.jsonl")
OUTPUT_JSONL = Path("/mnt/wulinyang/data/20260204_kmeans_8:2enhanced.4layers_new.with_meta_json.jsonl")


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def main() -> None:
    n_base = _count_lines(BASE_JSONL)
    n_meta = _count_lines(META_JSONL)
    if n_base != n_meta:
        raise ValueError(f"Line mismatch: base={n_base} meta={n_meta}")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with BASE_JSONL.open("r", encoding="utf-8") as f_base, META_JSONL.open("r", encoding="utf-8") as f_meta, OUTPUT_JSONL.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line_no, (l_base, l_meta) in enumerate(zip(f_base, f_meta), start=1):
            l_base = l_base.strip()
            l_meta = l_meta.strip()
            if not l_base or not l_meta:
                raise ValueError(f"Empty line at {line_no}")

            base_obj = json.loads(l_base)
            meta_obj = json.loads(l_meta)

            meta_json = meta_obj.get("meta_json")
            if meta_json is None:
                meta_field = meta_obj.get("meta")
                if isinstance(meta_field, dict):
                    meta_json = meta_field.get("meta_json")

            base_obj["meta_json"] = meta_json
            f_out.write(json.dumps(base_obj, ensure_ascii=False) + "\n")

    print(str(OUTPUT_JSONL))


if __name__ == "__main__":
    main()
