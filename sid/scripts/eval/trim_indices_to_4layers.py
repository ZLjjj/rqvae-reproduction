#!/usr/bin/env python3
"""Trim 5-layer indices to 4 layers in a JSONL file.

For each line (a JSON object), if it contains a list field `indices`,
replace it with `indices[:4]`.

Writes to a new file by default.
"""

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = "/mnt/wulinyang/data/20260204_kmeans_8:2enhanced.jsonl"
DEFAULT_OUTPUT = "/mnt/wulinyang/data/20260204_kmeans_8:2enhanced.4layers.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trim indices to first 4 layers")
    p.add_argument("--input_jsonl", default=DEFAULT_INPUT)
    p.add_argument("--output_jsonl", default=DEFAULT_OUTPUT)
    p.add_argument("--strict", action="store_true", help="Error if indices missing or not a list, or len<4")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.input_jsonl).resolve()
    out_path = Path(args.output_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for n, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idxs = obj.get("indices")
            if not isinstance(idxs, list):
                if args.strict:
                    raise ValueError(f"Line {n}: indices is missing or not a list")
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            if len(idxs) < 4 and args.strict:
                raise ValueError(f"Line {n}: indices length < 4: {len(idxs)}")

            obj["indices"] = idxs[:4]
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(str(out_path))


if __name__ == "__main__":
    main()
