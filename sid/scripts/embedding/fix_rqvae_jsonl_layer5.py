#!/usr/bin/env python3
"""Fix 5th-layer hardcode indices in rqvae jsonl outputs.

The correct rule matches sid/scripts/train/train_rqkmeans_faiss.py::pack_hardcode:
  e = ((saletype & 0x3) << 6) | (publish_year & 0x3F)
where:
  saletype is clipped to [0, 3]
  publish_year is clipped to [0, 63]

This script reads an input JSONL and writes a new JSONL with:
  - indices[4] updated
  - tokens[4] updated to "<e_{indices[4]}>" if tokens exist

Where to read saletype/publish_year from each record:
  - record["saletype"] / record["publish_year"], or
  - record["meta"]["saletype"] / record["meta"]["publish_year"], or
  - record["meta_json"][...] if meta_json is a dict

If missing, defaults to 0.

Usage:
  python3 fix_rqvae_jsonl_layer5.py \
    --input /path/to/rqvae-ema.jsonl \
    --output /path/to/rqvae-ema.fixed.jsonl
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def _clip_int(x: Any, lo: int, hi: int, default: int = 0) -> int:
    try:
        v = int(x)
    except Exception:
        return default
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _get_field(rec: dict[str, Any], key: str) -> Any:
    if key in rec:
        return rec.get(key)

    meta = rec.get("meta")
    if isinstance(meta, dict) and key in meta:
        return meta.get(key)

    meta_json = rec.get("meta_json")
    if isinstance(meta_json, dict) and key in meta_json:
        return meta_json.get(key)

    return None


def pack_hardcode_from_record(rec: dict[str, Any]) -> int:
    saletype = _clip_int(_get_field(rec, "saletype"), 0, 3, default=0)
    publish_year = _clip_int(_get_field(rec, "publish_year"), 0, 63, default=0)
    return ((saletype & 0x3) << 6) | (publish_year & 0x3F)


def fix_record(rec: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    indices = rec.get("indices")
    if not isinstance(indices, list) or len(indices) < 5:
        return rec, False

    new_e = int(pack_hardcode_from_record(rec))
    changed = False

    old_e = indices[4]
    if old_e != new_e:
        indices[4] = new_e
        changed = True

    tokens = rec.get("tokens")
    if isinstance(tokens, list) and len(tokens) >= 5:
        new_tok = f"<e_{new_e}>"
        if tokens[4] != new_tok:
            tokens[4] = new_tok
            changed = True

    return rec, changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Optional: only process first N lines")
    args = ap.parse_args()

    n = 0
    changed = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if args.limit is not None and n >= args.limit:
                break

            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            rec2, did_change = fix_record(rec)
            if did_change:
                changed += 1

            fout.write(json.dumps(rec2, ensure_ascii=False) + "\n")
            n += 1

    print(json.dumps({"processed": n, "changed": changed, "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
