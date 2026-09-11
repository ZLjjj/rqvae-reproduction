#!/usr/bin/env python3
"""Line-by-line overwrite selected fields from a source JSONL into a target JSONL.

This is intentionally *positional* (line N matches line N). Use only when you
trust both files are aligned.

Overwrites:
  - sid
  - indices (entire list)
  - tokens (entire list)

Usage:
  python3 line_overwrite_sid_tokens_indices.py \
    --target /path/to/rqvae-ema.jsonl \
    --source /path/to/indices.jsonl \
    --output /path/to/rqvae-ema.lineoverwrite.jsonl
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    n = 0
    changed_sid = 0
    changed_indices = 0
    changed_tokens = 0
    missing_source = 0

    with open(args.target, "r", encoding="utf-8") as ft, open(args.source, "r", encoding="utf-8") as fs, open(
        args.output, "w", encoding="utf-8"
    ) as fo:
        while True:
            lt = ft.readline()
            ls = fs.readline()
            if not lt and not ls:
                break
            if not lt or not ls:
                # line count mismatch
                raise RuntimeError("target/source line count mismatch")

            lt = lt.strip()
            ls = ls.strip()
            if not lt and not ls:
                continue
            if not lt or not ls:
                raise RuntimeError("blank-line alignment mismatch")

            tgt: Dict[str, Any] = json.loads(lt)
            src: Dict[str, Any] = json.loads(ls)

            if "sid" in src:
                if tgt.get("sid") != src.get("sid"):
                    changed_sid += 1
                tgt["sid"] = src.get("sid")
            else:
                missing_source += 1

            if "indices" in src:
                if tgt.get("indices") != src.get("indices"):
                    changed_indices += 1
                tgt["indices"] = src.get("indices")

            if "tokens" in src:
                if tgt.get("tokens") != src.get("tokens"):
                    changed_tokens += 1
                tgt["tokens"] = src.get("tokens")

            fo.write(json.dumps(tgt, ensure_ascii=False) + "\n")
            n += 1

    print(
        json.dumps(
            {
                "processed": n,
                "changed_sid": changed_sid,
                "changed_indices": changed_indices,
                "changed_tokens": changed_tokens,
                "source_missing_sid": missing_source,
                "output": args.output,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
