#!/usr/bin/env python3
import argparse
import json
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser(description="Count sale_type/pay_type value distribution from indices.jsonl")
    p.add_argument("--jsonl", required=True)
    p.add_argument("--top", type=int, default=127)
    p.add_argument(
        "--paths",
        default="meta.saletype,meta.sale_type,meta.pay_type,meta.paytype,saletype,sale_type,pay_type,paytype,meta_json.saletype",
        help="Comma-separated field paths to try in order; first non-null is used",
    )
    return p.parse_args()


def _get_path(o, path: str):
    cur = o
    for k in [x for x in path.split(".") if x]:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def main():
    args = parse_args()
    paths = [p.strip() for p in args.paths.split(",") if p.strip()]

    total = 0
    bad_json = 0
    missing = 0
    picked_from = Counter()
    values = Counter()

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
            v = None
            src = None
            for p in paths:
                vv = _get_path(obj, p)
                if vv is not None:
                    v = vv
                    src = p
                    break

            if v is None:
                missing += 1
                continue

            picked_from[src] += 1
            if isinstance(v, str):
                v = v.strip()
                vn = v.upper()
                if vn in {"FREE", "PAY"}:
                    v = vn
                elif v.isdigit():
                    v = int(v)
            values[v] += 1

    denom = max(total - bad_json, 1)
    print(f"file: {args.jsonl}")
    print(f"total_rows: {total}")
    print(f"bad_json: {bad_json}")
    print(f"missing_all_paths: {missing}")
    print(f"unique_values: {len(values)}")
    print("--- picked_from ---")
    for k, c in picked_from.most_common():
        print(f"{k}\t{c}\t{c/denom:.6f}")

    print("--- top_values ---")
    for k, c in values.most_common(args.top):
        print(f"{k}\t{c}\t{c/denom:.6f}")


if __name__ == "__main__":
    main()
