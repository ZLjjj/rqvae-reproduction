#!/usr/bin/env python3
import argparse
import json
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser(description="Count publish_year distribution from indices.jsonl")
    p.add_argument("--jsonl", required=True, help="Path to indices.jsonl (or any jsonl with publish_year field)")
    p.add_argument("--top", type=int, default=127, help="Top-K to print")
    p.add_argument(
        "--path",
        default="publish_year",
        help="Field path to read, e.g. 'publish_year' or 'meta.publish_year'",
    )
    p.add_argument("--key", default=None, help="Deprecated; use --path")
    return p.parse_args()


def main():
    args = parse_args()

    cnt = Counter()
    total = 0
    missing = 0
    bad_json = 0

    path = args.path
    if args.key is not None:
        path = args.key

    keys = [k for k in path.split(".") if k]

    def _get_path(o: dict):
        cur = o
        for k in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        return cur

    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad_json += 1
                continue

            total += 1
            v = _get_path(obj)
            if v is None:
                missing += 1
                continue

            # normalize: keep raw strings, ints, etc. as dict keys
            if isinstance(v, str):
                v = v.strip()
            cnt[v] += 1

    print(f"file: {args.jsonl}")
    print(f"total_rows: {total}")
    print(f"missing_{path}: {missing}")
    print(f"bad_json: {bad_json}")
    print(f"unique_{path}: {len(cnt)}")
    print("--- top ---")

    denom = max(total - bad_json, 1)
    for k, c in cnt.most_common(args.top):
        ratio = c / denom
        print(f"{k}\t{c}\t{ratio:.6f}")


if __name__ == "__main__":
    main()
