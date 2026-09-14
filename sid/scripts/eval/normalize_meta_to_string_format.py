#!/usr/bin/env python3
"""Normalize JSONL rows to the "meta-as-JSON-string" format.

Target format (top-level keys):
- idx, id, domain
- meta: JSON string (structured fields)
- meta_json (dict)
- text_format (str)
- json_format (str)
- sid (optional)
- indices (list)

Input can contain either:
1) meta as a JSON string (already target style)
2) meta as a dict (wrapper style): may include nested keys like
   {idx,id,domain, meta:<text>, meta_json, text_format, json_format, ...}

Rule for case (2):
- Promote text_format/json_format/meta_json to top-level (prefer existing top-level values if present)
- Build meta string from the dict after dropping wrapper keys:
  drop: idx, id, domain, meta, meta_json, text_format, json_format
  (i.e. do NOT include the nested 'meta' text)

This script writes a new JSONL file.
"""

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "/mnt/wulinyang/data/20260204_kmeans_8:2enhanced.4layers.jsonl"
DEFAULT_OUTPUT = "/mnt/wulinyang/data/20260204_kmeans_8:2enhanced.4layers.meta_string.jsonl"

_DROP_KEYS = {"idx", "id", "domain", "meta", "meta_json", "text_format", "json_format"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize meta field to JSON string format")
    p.add_argument("--input_jsonl", default=DEFAULT_INPUT)
    p.add_argument("--output_jsonl", default=DEFAULT_OUTPUT)
    p.add_argument("--strict", action="store_true", help="Error on unexpected/missing fields")
    return p.parse_args()


def _coalesce(obj: dict[str, Any], key: str) -> Any:
    v = obj.get(key)
    if v is not None:
        return v
    meta = obj.get("meta")
    if isinstance(meta, dict):
        return meta.get(key)
    return None


def main() -> None:
    args = parse_args()

    in_path = Path(args.input_jsonl).resolve()
    out_path = Path(args.output_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            idx = obj.get("idx")
            _id = obj.get("id")
            domain = obj.get("domain")
            indices = obj.get("indices")

            if args.strict:
                if idx is None or _id is None or domain is None:
                    raise ValueError(f"Line {line_no}: missing idx/id/domain")
                if not isinstance(indices, list):
                    raise ValueError(f"Line {line_no}: indices missing or not list")

            meta_val = obj.get("meta")
            if isinstance(meta_val, str):
                meta_str = meta_val
            elif isinstance(meta_val, dict):
                meta_payload = {k: v for k, v in meta_val.items() if k not in _DROP_KEYS}
                meta_str = json.dumps(meta_payload, ensure_ascii=False)
            else:
                if args.strict:
                    raise ValueError(f"Line {line_no}: meta is neither str nor dict")
                meta_str = json.dumps({}, ensure_ascii=False)

            out_obj: dict[str, Any] = {
                "idx": idx,
                "id": _id,
                "domain": domain,
                "meta": meta_str,
                "meta_json": _coalesce(obj, "meta_json"),
                "text_format": _coalesce(obj, "text_format"),
                "json_format": _coalesce(obj, "json_format"),
                "indices": indices,
            }

            # keep sid if present
            if "sid" in obj:
                out_obj["sid"] = obj.get("sid")

            if args.strict:
                if out_obj["meta_json"] is None:
                    raise ValueError(f"Line {line_no}: meta_json missing")
                if out_obj["text_format"] is None:
                    raise ValueError(f"Line {line_no}: text_format missing")
                if out_obj["json_format"] is None:
                    raise ValueError(f"Line {line_no}: json_format missing")

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(str(out_path))


if __name__ == "__main__":
    main()
