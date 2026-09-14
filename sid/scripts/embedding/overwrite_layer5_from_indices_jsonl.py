#!/usr/bin/env python3
"""Overwrite 5th-layer (level-5) index in a JSONL using another JSONL as source.

Use case:
  - target: rqvae-ema.jsonl
  - source: sid/results/.../rqkmeans/indices.jsonl

Both files typically share stable keys: (idx, id, domain). We join on these keys.

For each target record:
  - find matching source record
  - set target["indices"][4] = source["indices"][4]
  - update target["tokens"][4] to f"<e_{new}>" if tokens exist

Outputs a NEW file; does not overwrite input.

Example:
  python3 overwrite_layer5_from_indices_jsonl.py \
    --target /path/to/rqvae-ema.jsonl \
    --source /path/to/indices.jsonl \
    --output /path/to/rqvae-ema.layer5fromrqkmeans.jsonl

Notes:
  - This script loads the full mapping from source into memory. For ~1-2M lines this is usually OK.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Tuple


Key = Tuple[int, int, str]


def _as_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _as_str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    return str(v)


def _key(rec: Dict[str, Any]) -> Key | None:
    idx = _as_int(rec.get("idx"), default=-1)
    rid = _as_int(rec.get("id"), default=-1)
    domain = _as_str(rec.get("domain"))
    if idx < 0 or rid < 0 or domain is None:
        return None
    return (idx, rid, domain)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    mapping: Dict[Key, int] = {}
    src_total = 0
    src_bad = 0

    with open(args.source, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src_total += 1
            rec = json.loads(line)
            k = _key(rec)
            inds = rec.get("indices")
            if k is None or not isinstance(inds, list) or len(inds) < 5:
                src_bad += 1
                continue
            mapping[k] = _as_int(inds[4], default=0)

    out_total = 0
    out_changed = 0
    out_missing = 0
    out_bad = 0

    with open(args.target, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out_total += 1

            k = _key(rec)
            inds = rec.get("indices")
            if k is None or not isinstance(inds, list) or len(inds) < 5:
                out_bad += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            new_e = mapping.get(k)
            if new_e is None:
                out_missing += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            old_e = inds[4]
            if old_e != new_e:
                inds[4] = int(new_e)
                out_changed += 1

            toks = rec.get("tokens")
            if isinstance(toks, list) and len(toks) >= 5:
                toks[4] = f"<e_{int(inds[4])}>"

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "source_total": src_total,
                "source_bad": src_bad,
                "mapping_size": len(mapping),
                "target_total": out_total,
                "target_bad": out_bad,
                "target_missing": out_missing,
                "target_changed": out_changed,
                "output": args.output,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
