import argparse
import json
from pathlib import Path


def _get_meta_field(obj: dict, k: str, default=0):
    if k in obj:
        return obj.get(k, default)
    inner = obj.get("meta")
    if isinstance(inner, dict):
        return inner.get(k, default)
    return default


def pack_hardcode(saletype: int, publish_year: int) -> int:
    # Must match torch models: (publish_year<<2) | saletype
    s = max(0, min(int(saletype or 0), 2))
    y = max(0, min(int(publish_year or 0), 63))
    return ((y & 0x3F) << 2) | (s & 0x3)


def main():
    p = argparse.ArgumentParser(description="Self-check hardcode extraction/packing for sid meta.jsonl")
    p.add_argument("meta_jsonl", help="Path to meta.jsonl (either flat or embedding processor layout)")
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()

    path = Path(args.meta_jsonl)
    n = 0
    uniq = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if args.limit is not None and n >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            s = _get_meta_field(obj, "saletype", 0)
            y = _get_meta_field(obj, "publish_year", 0)
            idx = pack_hardcode(s, y)
            uniq.add(idx)
            print(json.dumps({"i": n, "saletype": s, "publish_year": y, "hardcode_idx": idx}, ensure_ascii=False))
            n += 1

    print(json.dumps({"rows": n, "unique_hardcode": len(uniq)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
