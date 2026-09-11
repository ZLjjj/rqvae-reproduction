#!/usr/bin/env python3
"""Validate JSONL/NPY row alignment before launching multi-hour training."""
import argparse
import json
from pathlib import Path
import numpy as np

def ids(path, positions):
    wanted = set(positions)
    got = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in wanted:
                got[i] = json.loads(line).get("id")
            if i > max(wanted):
                break
    return got

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True)
    p.add_argument("--npy", required=True)
    p.add_argument("--other-meta")
    args = p.parse_args()
    arr = np.load(args.npy, mmap_mode="r")
    pos = [0, max(0, arr.shape[0] // 2), arr.shape[0] - 1]
    samples = ids(args.meta, pos)
    if len(samples) != len(pos):
        raise ValueError("meta JSONL has fewer rows than embedding")
    if args.other_meta:
        other = ids(args.other_meta, pos)
        if samples != other:
            raise ValueError(f"ID mismatch: {samples} != {other}")
    print(json.dumps({"meta": str(Path(args.meta).resolve()), "npy": str(Path(args.npy).resolve()), "shape": list(arr.shape), "sample_ids": samples}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
