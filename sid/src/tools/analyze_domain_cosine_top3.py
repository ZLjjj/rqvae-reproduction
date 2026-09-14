#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

# allow running as a standalone tool
import os
import sys

CUR_DIR = Path(__file__).resolve().parent
SRC_DIR = CUR_DIR.parents[0]  # .../sid/src/tools -> want .../sid/src
SRC_DIR = CUR_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from codebook.metrics import compute_prefix_bucket_cosine
except ModuleNotFoundError:
    # When invoked without PYTHONPATH, running from some cwd may not resolve imports.
    # Fallback: add sid/src explicitly.
    _SRC_FALLBACK = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_SRC_FALLBACK))
    from codebook.metrics import compute_prefix_bucket_cosine


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute per-domain prefix bucket cosine similarity for first 3 layers (station vs video)."
    )
    p.add_argument("--indices_jsonl", required=True, help="Path to indices.jsonl (must include 'domain' and 'indices')")
    p.add_argument("--emb_npy", required=True, help="Path to embeddings .npy aligned with indices order")
    p.add_argument(
        "--domains",
        default="station,video",
        help="Comma-separated domains to analyze (default: station,video)",
    )
    p.add_argument("--max_level", type=int, default=3, help="Max prefix level to compute (default: 3)")
    p.add_argument(
        "--max_items_per_bucket",
        type=int,
        default=512,
        help="Cap items per bucket to control cost (default: 512)",
    )
    p.add_argument(
        "--domain_key",
        default="domain",
        help="Field name for domain in indices.jsonl (default: domain)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    emb = np.load(args.emb_npy, mmap_mode="r")

    idxs_by_domain: dict[str, list[int]] = {d: [] for d in domains}
    missing_domain = 0
    missing_indices = 0
    total = 0

    with open(args.indices_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue

            dom = rec.get(args.domain_key)
            if dom is None:
                missing_domain += 1
                continue
            if dom not in idxs_by_domain:
                continue

            codes = rec.get("indices")
            if not isinstance(codes, list) or len(codes) < args.max_level:
                missing_indices += 1
                continue

            idxs_by_domain[dom].append(i)

    print(f"indices_jsonl: {args.indices_jsonl}")
    print(f"emb_npy: {args.emb_npy}")
    print(f"total_lines: {total}")
    print(f"missing_domain: {missing_domain}")
    print(f"missing_indices: {missing_indices}")

    for dom in domains:
        idxs = idxs_by_domain.get(dom, [])
        if not idxs:
            print(f"--- domain={dom} (no rows) ---")
            continue

        codes = []
        for i in idxs:
            # re-read indices only for selected rows is expensive; instead we stored just indices in the first pass.
            # To keep memory low, do a second streaming pass collecting codes for that domain.
            pass

        # Second pass: collect codes for this domain
        codes = np.zeros((len(idxs), args.max_level), dtype=np.int64)
        take = set(idxs)
        j = 0
        with open(args.indices_jsonl, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i not in take:
                    continue
                rec = json.loads(line)
                row = rec["indices"]
                codes[j, :] = np.asarray(row[: args.max_level], dtype=np.int64)
                j += 1
                if j >= len(idxs):
                    break

        emb_dom = np.asarray(emb[idxs])

        stats = compute_prefix_bucket_cosine(
            emb_dom,
            codes,
            max_level=args.max_level,
            exclude_last_level=False,
            max_items_per_bucket=args.max_items_per_bucket,
        )

        print(f"--- domain={dom} rows={len(idxs)} ---")
        for item in stats.get("bucket_cosine", []):
            lvl = item["level"]
            ws = item["weighted_sim"]
            us = item["unweighted_sim"]
            print(f"level={lvl}\tweighted_sim={ws:.6f}\tunweighted_sim={us:.6f}")


if __name__ == "__main__":
    main()
