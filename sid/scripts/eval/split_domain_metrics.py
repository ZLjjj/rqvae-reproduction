#!/usr/bin/env python3

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Allow running as a standalone script without requiring callers to set PYTHONPATH.
# Repo layout: sid/scripts/eval/this_file.py -> sid/src is two levels up from scripts/.
_THIS_DIR = Path(__file__).resolve().parent
_SID_DIR = _THIS_DIR.parents[1]
_SRC_DIR = _SID_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from codebook.metrics import compute_core_metrics, compute_prefix_bucket_cosine, compute_prefix_bucket_stats


_INDEX_RE = re.compile(r"^<([a-zA-Z]+)_(-?\d+)>$")


@dataclass(frozen=True)
class ParsedIndex:
    layer_tag: str
    code: int


def _parse_index_token(tok: str) -> ParsedIndex:
    m = _INDEX_RE.match(tok)
    if not m:
        raise ValueError(f"Bad index token: {tok!r}")
    return ParsedIndex(layer_tag=m.group(1), code=int(m.group(2)))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {path} at line {ln}: {e}") from e


def _codes_from_records(records: list[dict[str, Any]]) -> np.ndarray:
    if not records:
        return np.zeros((0, 0), dtype=np.int64)

    first = records[0]
    indices = first.get("indices") or first.get("tokens") or first.get("sid")
    if not isinstance(indices, list) or not indices:
        raise ValueError("Record missing non-empty 'indices' list")

    L = len(indices)
    out = np.full((len(records), L), -1, dtype=np.int64)

    for i, r in enumerate(records):
        idxs = r.get("indices") or r.get("tokens") or r.get("sid")
        if not isinstance(idxs, list) or len(idxs) != L:
            raise ValueError(f"Inconsistent 'indices' length at row {i}: expected {L}, got {None if not isinstance(idxs, list) else len(idxs)}")
        for l, tok in enumerate(idxs):
            if isinstance(tok, (int, np.integer)):
                out[i, l] = int(tok)
            else:
                p = _parse_index_token(str(tok))
                out[i, l] = p.code

    return out


def _group_by_domain(records: Iterable[dict[str, Any]], *, domain_key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        d = r.get(domain_key)
        if d is None and isinstance(r.get("meta"), dict):
            d = r["meta"].get(domain_key)
        if d is None and isinstance(r.get("json_format"), str):
            try:
                jf = json.loads(r["json_format"])
                d = jf.get(domain_key) or jf.get("垂域")
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        if d in ("视频", "video"):
            d = "video"
        elif d in ("电台", "station"):
            d = "station"
        d = str(d) if d is not None else "__missing__"
        groups[d].append(r)
    return dict(groups)


def _summarize_one(
    records: list[dict[str, Any]],
    *,
    emb: np.ndarray | None,
    max_level: int | None,
    exclude_last_level_cosine: bool,
    max_items_per_bucket: int | None,
) -> dict[str, Any]:
    codes = _codes_from_records(records)

    out: dict[str, Any] = {}
    out.update(compute_core_metrics(codes))
    out.update(compute_prefix_bucket_stats(codes))

    if emb is not None:
        if emb.shape[0] != codes.shape[0]:
            raise ValueError(f"emb rows ({emb.shape[0]}) != codes rows ({codes.shape[0]})")
        out.update(
            compute_prefix_bucket_cosine(
                emb,
                codes,
                max_level=max_level,
                exclude_last_level=exclude_last_level_cosine,
                max_items_per_bucket=max_items_per_bucket,
            )
        )

    return out


def _summarize_for_groups(
    groups: dict[str, list[dict[str, Any]]],
    *,
    emb_by_domain: dict[str, np.ndarray] | None,
    max_level: int | None,
    exclude_last_level_cosine: bool,
    max_items_per_bucket: int | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # overall
    all_records: list[dict[str, Any]] = []
    all_emb: list[np.ndarray] = []
    for d, v in groups.items():
        all_records.extend(v)
        if emb_by_domain is not None:
            all_emb.append(emb_by_domain[d])

    emb_overall = np.concatenate(all_emb, axis=0) if all_emb else None
    out["overall"] = _summarize_one(
        all_records,
        emb=emb_overall,
        max_level=max_level,
        exclude_last_level_cosine=exclude_last_level_cosine,
        max_items_per_bucket=max_items_per_bucket,
    )

    # per domain
    per_domain: dict[str, Any] = {}
    for domain, rs in sorted(groups.items(), key=lambda x: x[0]):
        per_domain[domain] = _summarize_one(
            rs,
            emb=emb_by_domain.get(domain) if emb_by_domain is not None else None,
            max_level=max_level,
            exclude_last_level_cosine=exclude_last_level_cosine,
            max_items_per_bucket=max_items_per_bucket,
        )
    out["per_domain"] = per_domain

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split merged station/video codebooks by domain and compute code metrics")

    # Hard-coded defaults for quick local runs; CLI flags can override.
    p.add_argument(
        "--input_dir",
        default="/mnt/wulinyang/data/20260208_station_video_text_format_4B",
        help="Directory containing rqvae.jsonl/rqvae-opq.jsonl/rqkmeans.jsonl/rqkmeans-opq.jsonl",
    )
    p.add_argument(
        "--emb_npy",
        default="/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings/station_video_20260208_text/text_format_embedding.npy",
        help="Embedding .npy aligned by row with the JSONL files",
    )
    p.add_argument("--output_root", default="/mnt/wulinyang/to_push/gensearchrec/sid/results", help="Root results dir")
    p.add_argument("--run_name", default=None, help="Run folder name under results_root (default: timestamp)")
    p.add_argument("--domain_key", default="domain", help="Field name used to split domains (default: domain)")
    p.add_argument("--max_lines", type=int, default=None, help="Debug: only read first N lines from each jsonl")

    # cosine options
    p.add_argument("--cosine_max_level", type=int, default=None, help="Compute cosine only up to this prefix level")
    p.add_argument("--cosine_exclude_last_level", action="store_true", help="Exclude last prefix level when computing cosine")
    p.add_argument(
        "--cosine_max_items_per_bucket",
        type=int,
        default=512,
        help="Subsample max items per bucket when computing cosine (default: 512; set 0 to disable limit)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sid_root = repo_root / "sid"

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    output_root = Path(args.output_root).resolve() if args.output_root else (sid_root / "results").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (output_root / run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "rqvae": input_dir / "rqvae.jsonl",
        "rqvae_opq": input_dir / "rqvae-opq.jsonl",
        "rqkmeans": input_dir / "rqkmeans.jsonl",
        "rqkmeans_opq": input_dir / "rqkmeans-opq.jsonl",
    }

    results: dict[str, Any] = {
        "input_dir": str(input_dir),
        "run_name": run_name,
        "domain_key": args.domain_key,
        "strategies": {},
    }

    emb_all: np.ndarray | None = None
    if args.emb_npy:
        emb_path = Path(args.emb_npy).resolve()
        emb_all = np.load(str(emb_path), mmap_mode="r")

    cosine_max_items_per_bucket = args.cosine_max_items_per_bucket
    if cosine_max_items_per_bucket == 0:
        cosine_max_items_per_bucket = None

    for name, fp in files.items():
        if not fp.exists():
            results["strategies"][name] = {"error": f"missing file: {fp}"}
            continue

        recs: list[dict[str, Any]] = []
        for i, r in enumerate(_iter_jsonl(fp)):
            recs.append(r)
            if args.max_lines is not None and i + 1 >= args.max_lines:
                break

        # embeddings are aligned by row with jsonl
        emb_slice = None
        if emb_all is not None:
            n = len(recs)
            if emb_all.shape[0] < n:
                raise ValueError(f"emb_npy rows ({emb_all.shape[0]}) < lines read ({n})")
            emb_slice = np.asarray(emb_all[:n])

        groups = _group_by_domain(recs, domain_key=args.domain_key)

        emb_by_domain = None
        if emb_slice is not None:
            # build per-domain embedding slices in the same order as records
            idxs_by_domain: dict[str, list[int]] = defaultdict(list)
            for i, r in enumerate(recs):
                d = r.get(args.domain_key)
                d = str(d) if d is not None else "__missing__"
                idxs_by_domain[d].append(i)
            emb_by_domain = {d: emb_slice[idxs] for d, idxs in idxs_by_domain.items()}

        results["strategies"][name] = {
            "file": str(fp),
            "emb_npy": str(Path(args.emb_npy).resolve()) if args.emb_npy else None,
            "domain_counts": {k: len(v) for k, v in sorted(groups.items(), key=lambda x: x[0])},
            **_summarize_for_groups(
                groups,
                emb_by_domain=emb_by_domain,
                max_level=args.cosine_max_level,
                exclude_last_level_cosine=args.cosine_exclude_last_level,
                max_items_per_bucket=cosine_max_items_per_bucket,
            ),
        }

    out_path = out_dir / "domain_metrics.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
