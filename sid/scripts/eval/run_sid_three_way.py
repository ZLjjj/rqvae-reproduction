#!/usr/bin/env python3
"""Run the three-way video SID evaluation with one consistent metric protocol.

The baseline file is mixed station+video, with video rows contiguous at the end;
its embedding therefore uses ``--baseline-emb-offset`` (1129176 by default).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-index", required=True)
    p.add_argument("--baseline-emb", required=True)
    p.add_argument("--same-source-index", required=True)
    p.add_argument("--same-source-emb", required=True)
    p.add_argument("--v3-index", required=True)
    p.add_argument("--v3-emb", required=True)
    p.add_argument("--baseline-emb-offset", type=int, default=0,
                   help="offset only when baseline embedding is the original mixed matrix")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    script = Path(__file__).resolve().with_name("evaluate_sid.py")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("baseline_video", args.baseline_index, args.baseline_emb, args.baseline_emb_offset),
        ("same_source_video", args.same_source_index, args.same_source_emb, 0),
        ("video_v3", args.v3_index, args.v3_emb, 0),
    ]
    summary = {}
    for name, index_path, emb_path, offset in jobs:
        out = out_dir / f"{name}.json"
        cmd = [sys.executable, str(script), "--input", index_path, "--output", str(out),
               "--emb-npy", emb_path, "--emb-offset", str(offset),
               "--exclude-last-level-cosine"]
        if name == "baseline_video":
            cmd.extend(["--domain", "video"])
        subprocess.run(cmd, check=True)
        summary[name] = json.loads(out.read_text(encoding="utf-8"))

    (out_dir / "three_way_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out_dir / "three_way_summary.json")


if __name__ == "__main__":
    main()
