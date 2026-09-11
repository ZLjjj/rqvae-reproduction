#!/usr/bin/env python3
"""从 merged station+video jsonl 里提取仅视频行。

判定规则（满足任一即视为 video）：
- 有 meta_origin_name / video_type / pay_type / directors / genre_types 等 video 专属字段
- 或没有 station 专属字段（cpId / f_tags）

用法：
  python extract_video_only.py --input merged.jsonl --output video_only.jsonl
  python extract_video_only.py --input merged.jsonl --output video_only.jsonl --meta_npy merged.npy --output_npy video_only.npy
"""
import argparse
import json
from pathlib import Path

VIDEO_KEYS = {"meta_origin_name", "meta_main_name", "video_type", "pay_type",
              "directors", "main_actors", "genre_types", "characters", "actors"}
STATION_KEYS = {"cpId", "f_tags", "s_tags", "k_tags"}


def is_video(obj: dict) -> bool:
    """优先按 json_format 里的 垂域 字段判定（video/视频 vs station/电台）。
    json_format 是 JSON 字符串，含 {"垂域": "视频"/"电台", ...}。
    缺失时 fallback 到字段存在性规则。
    """
    jf = obj.get("json_format")
    if jf is not None:
        try:
            jfd = json.loads(jf) if isinstance(jf, str) else jf
            dom = jfd.get("垂域")
            if dom is not None:
                return str(dom) in ("视频", "video")
        except (json.JSONDecodeError, AttributeError):
            pass
    # fallback: 字段存在性
    keys = set(obj.keys())
    if keys & VIDEO_KEYS:
        return True
    if keys & STATION_KEYS:
        return False
    return True  # 默认归 video


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def main():
    p = argparse.ArgumentParser(description="Extract video-only rows from merged jsonl")
    p.add_argument("--input", required=True, help="merged station+video jsonl")
    p.add_argument("--output", required=True, help="output video-only jsonl")
    p.add_argument("--meta_npy", default=None, help="可选: merged embedding npy，按行同步切出 video-only npy")
    p.add_argument("--output_npy", default=None, help="输出 video-only npy 路径（需配合 --meta_npy）")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    video_idx = []  # 行号（用于切 npy）
    n_total = 0
    n_video = 0
    n_station = 0

    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            n_total += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if is_video(obj):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                video_idx.append(i)
                n_video += 1
            else:
                n_station += 1

    print(f"总行数: {n_total}")
    print(f"video: {n_video}  station: {n_station}")
    print(f"video-only jsonl -> {out}")

    # 同步切 npy
    if args.meta_npy and args.output_npy:
        import numpy as np
        arr = np.load(args.meta_npy, mmap_mode="r")
        print(f"merged npy shape: {arr.shape}")
        if len(video_idx) > 0:
            sub = arr[video_idx]
            Path(args.output_npy).parent.mkdir(parents=True, exist_ok=True)
            np.save(args.output_npy, sub)
            print(f"video-only npy -> {args.output_npy}  shape={sub.shape}")


if __name__ == "__main__":
    main()
