#!/usr/bin/env python3
"""
Standalone RQVAE auto runner (extracted from benchmark)
Only runs RQVAE strategy, no other benchmarks
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CUR_DIR = Path(__file__).resolve().parent
# This file lives in <repo>/sid/scripts; parents[1] is the sid directory.
SID_DIR = CUR_DIR.parents[1]
REPO_ROOT = SID_DIR.parent

CONFIG_DEFAULT = REPO_ROOT / "sid" / "scripts" / "benchmark" / "benchmark_config.json"

STRATEGY_INFER_CMD = {
    "rqvae": f"python {SID_DIR / 'src' / 'codebook' / 'generate_indices_rqvae.py'} --model rqvae",
}

STRATEGY_TRAIN_CMD = {
    "rqvae": f"python {SID_DIR / 'scripts' / 'train' / 'train_rqvae.py'}",
}


def run_cmd(cmd: str, desc: str):
    print(f"\n[RUN] {desc}\n{cmd}\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SID_DIR / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    ret = subprocess.run(cmd, shell=True, env=env)
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed: {desc}")


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _merge_dict(a: dict, b: dict):
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Standalone RQVAE runner")
    p.add_argument("--meta", default="/mnt/zhanggehang1/wulinyang1/raw_data/sidflow/meta.jsonl", help="Path to original meta jsonl file")
    p.add_argument("--npy", default="/mnt/zhanggehang1/wulinyang1/raw_data/sidflow/meta.npy", help="Path to embedding npy file")
    p.add_argument("--output", default="/mnt/zhanggehang1/wulinyang1/raw_data/sidflow/sid.jsonl", help="Output path to save indices.jsonl")
    p.add_argument("--config", default=None, help="Optional custom config file path")
    return p.parse_args()


def main():
    args = parse_args()
    
    cfg_path = Path(args.config or CONFIG_DEFAULT).resolve()
    cfg = load_config(cfg_path)

    data_npy = str(Path(args.npy).resolve())
    meta_jsonl = str(Path(args.meta).resolve())
    
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    balanced_init = cfg.get("balanced_kmeans_init", {}) or {}
    balanced_init_enabled = bool(balanced_init.get("enabled", False))

    # Resolve results path (unchanged from benchmark)
    results_root_cfg = Path(cfg.get("results_root", "./results"))
    sid_root = Path(__file__).resolve().parents[1]
    results_root = (sid_root / results_root_cfg).resolve() if not results_root_cfg.is_absolute() else results_root_cfg
    results_root.mkdir(parents=True, exist_ok=True)

    run_name = cfg.get("run_name")
    if not run_name:
        from datetime import datetime
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_root = results_root / run_name / "rqvae"
    run_root.mkdir(parents=True, exist_ok=True)

    device = cfg.get("device", "cuda:0")
    batch_size = cfg.get("batch_size", 2048)
    num_workers = cfg.get("num_workers", 4)
    seed = cfg.get("seed", 2024)

    train_defaults = cfg.get("train_defaults", {})
    infer_defaults = cfg.get("infer_defaults", {})

    # Find RQVAE config from strategies
    strategies = cfg.get("strategies", [])
    rqvae_strat = None
    for s in strategies:
        if s.get("name") == "rqvae":
            rqvae_strat = s
            break
    
    if not rqvae_strat:
        # No rqvae config found, use default
        rqvae_strat = {
            "name": "rqvae",
            "enabled": True,
            "train": True,
            "infer": True,
            "train_overrides": {},
            "infer_overrides": {}
        }

    if rqvae_strat.get("enabled") is False:
        raise ValueError("RQVAE strategy is disabled in config")

    name = "rqvae"

    # ---- train stage ----
    ckpt = rqvae_strat.get("ckpt")
    ckpt = str(Path(ckpt).resolve()) if ckpt else None
    if rqvae_strat.get("train"):
        (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
        train_out = run_root / "checkpoints"

        train_params = _merge_dict(train_defaults, rqvae_strat.get("train_overrides", {}))

        cmd = (
            f"{STRATEGY_TRAIN_CMD[name]} "
            f"--data_npy {data_npy} "
            f"--ckpt_dir {train_out} "
            f"--device {device} "
            f"--seed {seed} "
            f"--epochs {train_params.get('epochs', 40)} "
            f"--lr {train_params.get('lr', 0.001)} "
            f"--batch_size {batch_size} "
            f"--num_workers {num_workers} "
            f"--eval_step {train_params.get('eval_step', 1)} "
            f"--learner {train_params.get('learner', 'AdamW')} "
            f"--lr_scheduler_type {train_params.get('lr_scheduler_type', 'constant')} "
            f"--warmup_epochs {train_params.get('warmup_epochs', 0)} "
            f"--weight_decay {train_params.get('weight_decay', 0.0)} "
            f"--save_limit {train_params.get('save_limit', 5)} "
            f"--dropout_prob {train_params.get('dropout_prob', 0.0)} "
            f"--loss_type {train_params.get('loss_type', 'mse')} "
            f"--quant_loss_weight {train_params.get('quant_loss_weight', 1.0)} "
            f"--beta {train_params.get('beta', 0.25)} "
            f"--kmeans_iters {train_params.get('kmeans_iters', 20)} "
            f"--sk_iters {train_params.get('sk_iters', 50)} "
        )

        if balanced_init_enabled:
            if not meta_jsonl:
                raise ValueError("balanced_kmeans_init.enabled=true requires meta_jsonl in config")
            cmd += f"--meta_jsonl {meta_jsonl} "

        if train_params.get("bn"):
            cmd += "--bn "
        if train_params.get("kmeans_init"):
            cmd += "--kmeans_init "
        if train_params.get("kmeans_init_max_samples") is not None:
            cmd += f"--kmeans_init_max_samples {train_params.get('kmeans_init_max_samples')} "
        cmd += f"--kmeans_init_seed {train_params.get('kmeans_init_seed', 0)} "
        if train_params.get("kmeans_init_dedup", True):
            cmd += "--kmeans_init_dedup "

        # RQVAE specific params
        rqvae_cfg = train_params.get("rqvae", {})
        cmd += f"--e_dim {rqvae_cfg.get('e_dim', 128)} "
        cmd += "--num_emb_list " + " ".join(str(x) for x in rqvae_cfg.get("num_emb_list", [1024, 1024, 1024, 1024, 256])) + " "
        cmd += "--layers " + " ".join(str(x) for x in rqvae_cfg.get("layers", [512, 256, 128])) + " "
        cmd += "--sk_epsilons " + " ".join(str(x) for x in rqvae_cfg.get("sk_epsilons", [0.0, 0.0, 0.0, 0.003, 0.0])) + " "

        if rqvae_cfg.get("ema_codebook"):
            cmd += "--ema_codebook "
            cmd += f"--ema_decay {rqvae_cfg.get('ema_decay', 0.99)} "
            cmd += f"--ema_eps {rqvae_cfg.get('ema_eps', 1e-5)} "

        run_cmd(cmd, desc=f"train RQVAE")

        # Load trained checkpoint
        ckpt_path = (train_out / "best_collision_model.pth").resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Expected checkpoint not found: {ckpt_path}")
        ckpt = str(ckpt_path)

    if not ckpt:
        raise ValueError("RQVAE missing ckpt (either provide ckpt in config or set train=true)")

    # ---- infer stage ----
    if rqvae_strat.get("infer") is not False:
        infer_params = _merge_dict(infer_defaults, rqvae_strat.get("infer_overrides", {}))

        out_dir = run_root
        out_dir.mkdir(parents=True, exist_ok=True)

        if not Path(ckpt).exists():
            raise FileNotFoundError(f"Missing ckpt for infer: {ckpt}")

        cmd = (
            f"{STRATEGY_INFER_CMD[name]} "
            f"--ckpt {ckpt} "
            f"--data_npy {data_npy} "
            f"--output_dir {out_dir} "
            f"--device {device} "
            f"--batch_size {batch_size} "
            f"--num_workers {num_workers} "
        )

        if meta_jsonl:
            cmd += f"--meta_jsonl {meta_jsonl} "

        if infer_params.get("use_sk") is False:
            cmd += "--no_use_sk "

        run_cmd(cmd, desc=f"infer RQVAE")

    indices_path = run_root / "indices.jsonl"
    if indices_path.exists():
        shutil.copy(indices_path, output_path)
        print(f"\n✅ RQVAE run finished successfully!")
        print(f"📂 Indices saved to: {output_path}")
        print(f"📂 Full results saved to: {run_root}")
    else:
        raise FileNotFoundError(f"Expected indices.jsonl not found at {indices_path}")


if __name__ == "__main__":
    main()
