#!/usr/bin/env python3
"""Evaluate one SID JSONL file with experiment-plan metrics."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "src"))
from codebook.metrics import compute_core_metrics, compute_prefix_bucket_stats, compute_prefix_bucket_cosine

def parse_codes(record):
    vals = record.get("indices") or record.get("tokens") or record.get("sid")
    if not isinstance(vals, list):
        raise ValueError("record has no indices/tokens/sid list")
    out = []
    for value in vals:
        if isinstance(value, (int, np.integer)):
            out.append(int(value))
        else:
            text = str(value)
            out.append(int(text.rsplit("_", 1)[1][:-1]))
    return out

def domain_of(record, key):
    value = record.get(key)
    if value is None and isinstance(record.get("meta"), dict):
        value = record["meta"].get(key)
    if value is None and isinstance(record.get("json_format"), str):
        try:
            jf = json.loads(record["json_format"])
            value = jf.get(key) or jf.get("垂域")
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    if value in ("视频", "video"):
        return "video"
    if value in ("电台", "station"):
        return "station"
    return str(value) if value is not None else "__missing__"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--emb-npy")
    p.add_argument("--emb-offset", type=int, default=0, help="row offset in embedding matrix for selected records")
    p.add_argument("--domain")
    p.add_argument("--domain-key", default="domain")
    p.add_argument("--max-lines", type=int)
    p.add_argument("--exclude-last-level-cosine", action="store_true")
    p.add_argument("--codebook-sizes", type=int, nargs="+", default=[1024, 1024, 1024, 1024, 512])
    args = p.parse_args()
    records = []
    with open(args.input, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if args.max_lines is not None and line_no >= args.max_lines:
                break
            if line.strip():
                r = json.loads(line)
                if args.domain is None or domain_of(r, args.domain_key) == args.domain:
                    records.append(r)
    codes = np.asarray([parse_codes(r) for r in records], dtype=np.int64)
    result = {"input": str(Path(args.input).resolve()), "N": len(records)}
    result.update(compute_core_metrics(codes, codebook_sizes=args.codebook_sizes))
    result.update(compute_prefix_bucket_stats(codes))
    if args.emb_npy:
        emb = np.load(args.emb_npy, mmap_mode="r")
        start = args.emb_offset
        if start < 0 or emb.shape[0] < start + len(records):
            raise ValueError("embedding rows are fewer than selected records")
        result.update(compute_prefix_bucket_cosine(np.asarray(emb[start:start + len(records)]), codes, exclude_last_level=args.exclude_last_level_cosine))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)

if __name__ == "__main__":
    main()
