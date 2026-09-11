#!/usr/bin/env python3
"""
Simplified auto-benchmark for 4 fixed strategies (RQVAE, RQKMeans, RQOPQVAE, RQOPQKMeans).
Assumptions:
- Codebook size fixed (1024), model architectures fixed.
- Hardcode packing already handled in models (saletype + publish_year).
- Inputs: merged embeddings .npy (+ optional meta.jsonl) and checkpoint paths.
- Outputs: results/{run_name}/{strategy}/...
"""

import json
import os
import subprocess
import sys
from pathlib import Path

CUR_DIR = Path(__file__).resolve().parent
REPO_ROOT = CUR_DIR.parents[2]
SID_DIR = REPO_ROOT / "sid"

CONFIG_DEFAULT = REPO_ROOT / "sid" / "scripts" / "benchmark" / "benchmark_config.json"

STRATEGY_INFER_CMD = {
    "rqvae": f"python {SID_DIR / 'src' / 'codebook' / 'generate_indices_rqvae.py'} --model rqvae",
    "rqkmeans": f"python {SID_DIR / 'src' / 'codebook' / 'generate_indices_rqvae.py'} --model rqkmeans",
    "rqvae_opq": f"python {SID_DIR / 'src' / 'codebook' / 'generate_indices_rqopq.py'} --model rqvae_opq",
    "rqkmeans_opq": f"python {SID_DIR / 'src' / 'codebook' / 'generate_indices_rqopq.py'} --model rqkmeans_opq",
}

STRATEGY_TRAIN_CMD = {
    "rqvae": f"python {SID_DIR / 'scripts' / 'train' / 'train_rqvae.py'}",
    "rqvae_opq": f"python {SID_DIR / 'scripts' / 'train' / 'train_rqvae_opq.py'}",
    "rqkmeans": f"python {SID_DIR / 'scripts' / 'train' / 'train_rqkmeans_faiss.py'}",
    "rqkmeans_opq": f"python {SID_DIR / 'scripts' / 'train' / 'train_rqkmeans_opq_faiss.py'}",
}


def run_cmd(cmd: str, desc: str):
    print(f"\n[RUN] {desc}\n{cmd}\n")
    env = os.environ.copy()
    # Ensure `sid/src` is importable for scripts that use `from data...` / `from models...`.
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


def main(cfg_path: str = None):
    cfg_path = Path(cfg_path or CONFIG_DEFAULT).resolve()
    cfg = load_config(cfg_path)

    data_npy = str(Path(cfg["data_npy"]).resolve())
    meta_jsonl = cfg.get("meta_jsonl")
    meta_jsonl = str(Path(meta_jsonl).resolve()) if meta_jsonl else None

    balanced_init = cfg.get("balanced_kmeans_init", {}) or {}
    balanced_init_enabled = bool(balanced_init.get("enabled", False))

    # results_root is resolved relative to sid/ (project root), not the config file directory.
    results_root_cfg = Path(cfg.get("results_root", "./results"))
    sid_root = Path(__file__).resolve().parents[2]
    results_root = (sid_root / results_root_cfg).resolve() if not results_root_cfg.is_absolute() else results_root_cfg
    results_root.mkdir(parents=True, exist_ok=True)

    run_name = cfg.get("run_name")
    if not run_name:
        from datetime import datetime

        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_root = results_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    device = cfg.get("device", "cuda:0")
    batch_size = cfg.get("batch_size", 2048)
    num_workers = cfg.get("num_workers", 4)
    seed = cfg.get("seed", 2024)

    train_defaults = cfg.get("train_defaults", {})
    infer_defaults = cfg.get("infer_defaults", {})

    strategies = cfg.get("strategies", [])
    if not strategies:
        raise ValueError("No strategies configured")

    # enforce requested order if present: rqvae, rqvae_opq, rqkmeans, rqkmeans_opq
    order = {"rqvae": 0, "rqvae_opq": 1, "rqkmeans": 2, "rqkmeans_opq": 3}
    strategies = sorted(strategies, key=lambda s: order.get(s.get("name"), 999))

    for strat in strategies:
        if strat.get("enabled") is False:
            continue

        name = strat["name"]

        # ---- train (optional) ----
        ckpt = strat.get("ckpt")
        ckpt = str(Path(ckpt).resolve()) if ckpt else None
        if strat.get("train"):
            if name not in STRATEGY_TRAIN_CMD:
                raise ValueError(f"Strategy {name} does not support train stage in this simplified sid")

            strat_dir = run_root / name
            strat_dir.mkdir(parents=True, exist_ok=True)

            if name in ["rqkmeans", "rqkmeans_opq"]:
                # faiss trainers write indices/metrics directly under strat_dir; no checkpoints needed
                train_out = strat_dir
            else:
                (strat_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
                train_out = strat_dir / "checkpoints"

            train_params = _merge_dict(train_defaults, strat.get("train_overrides", {}))

            if name == "rqkmeans":
                # kmeans train directly outputs indices+codebooks (no ckpt)
                cmd = (
                    f"{STRATEGY_TRAIN_CMD[name]} "
                    f"--data_npy {data_npy} "
                    f"--output_dir {strat_dir} "
                )
                if meta_jsonl:
                    cmd += f"--meta_jsonl {meta_jsonl} "
                rqk_cfg = train_params.get("rqkmeans", {})
                cmd += f"--num_levels {rqk_cfg.get('num_levels', 4)} "
                cmd += "--codebook_size " + " ".join(str(x) for x in rqk_cfg.get("codebook_size", [1024])) + " "
                if rqk_cfg.get("uniform"):
                    cmd += "--uniform "
                cmd += f"--sinkhorn_iters {rqk_cfg.get('sinkhorn_iters', 30)} "
                if rqk_cfg.get("sinkhorn_tau") is not None:
                    cmd += f"--sinkhorn_tau {rqk_cfg.get('sinkhorn_tau')} "
                cmd += f"--sinkhorn_topk {rqk_cfg.get('sinkhorn_topk', 32)} "
                cmd += f"--sinkhorn_seed {rqk_cfg.get('sinkhorn_seed', 42)} "
            elif name == "rqkmeans_opq":
                rqkopq_cfg = train_params.get("rqkmeans_opq", {})
                cmd = (
                    f"{STRATEGY_TRAIN_CMD[name]} "
                    f"--data_npy {data_npy} "
                    f"--output_dir {strat_dir} "
                    f"--rq_levels {rqkopq_cfg.get('rq_levels', 3)} "
                    f"--rq_codebook_size {rqkopq_cfg.get('rq_codebook_size', 1024)} "
                )
                if rqkopq_cfg.get("uniform"):
                    cmd += "--uniform "
                cmd += f"--sinkhorn_iters {rqkopq_cfg.get('sinkhorn_iters', 30)} "
                if rqkopq_cfg.get("sinkhorn_tau") is not None:
                    cmd += f"--sinkhorn_tau {rqkopq_cfg.get('sinkhorn_tau')} "
                cmd += f"--sinkhorn_topk {rqkopq_cfg.get('sinkhorn_topk', 32)} "
                cmd += f"--sinkhorn_seed {rqkopq_cfg.get('sinkhorn_seed', 42)} "
                cmd += (
                    f"--opq_subquantizers {rqkopq_cfg.get('opq_subquantizers', 2)} "
                    f"--opq_codebook_size {rqkopq_cfg.get('opq_codebook_size', 1024)} "
                    f"--opq_niter {rqkopq_cfg.get('opq_niter', 5)} "
                    f"--pq_niter {rqkopq_cfg.get('pq_niter', 10)} "
                )
            else:
                cmd = (
                    f"{STRATEGY_TRAIN_CMD[name]} "
                    f"--data_npy {data_npy} "
                    f"--ckpt_dir {train_out} "
                    f"--device {device} "
                    f"--seed {seed} "
                    f"--epochs {train_params.get('epochs', 200)} "
                    f"--lr {train_params.get('lr', 0.001)} "
                    f"--batch_size {batch_size} "
                    f"--num_workers {num_workers} "
                    f"--eval_step {train_params.get('eval_step', 10)} "
                    f"--learner {train_params.get('learner', 'AdamW')} "
                    f"--lr_scheduler_type {train_params.get('lr_scheduler_type', 'constant')} "
                    f"--warmup_epochs {train_params.get('warmup_epochs', 0)} "
                    f"--weight_decay {train_params.get('weight_decay', 0.0)} "
                    f"--save_limit {train_params.get('save_limit', 5)} "
                    f"--dropout_prob {train_params.get('dropout_prob', 0.0)} "
                    f"--loss_type {train_params.get('loss_type', 'mse')} "
                    f"--quant_loss_weight {train_params.get('quant_loss_weight', 1.0)} "
                    f"--beta {train_params.get('beta', 0.25)} "
                    f"--kmeans_iters {train_params.get('kmeans_iters', 100)} "
                    f"--sk_iters {train_params.get('sk_iters', 100)} "
                )

                # Optional: balanced 1:1 init only applies to torch-based (rqvae/rqvae_opq) trainers.
                if balanced_init_enabled:
                    if not meta_jsonl:
                        raise ValueError("balanced_kmeans_init.enabled=true requires meta_jsonl in config")
                    cmd += f"--meta_jsonl {meta_jsonl} "

            if name not in ["rqkmeans", "rqkmeans_opq"]:
                if train_params.get("bn"):
                    cmd += "--bn "
                if train_params.get("kmeans_init"):
                    cmd += "--kmeans_init "
                if train_params.get("kmeans_init_max_samples") is not None:
                    cmd += f"--kmeans_init_max_samples {train_params.get('kmeans_init_max_samples')} "
                cmd += f"--kmeans_init_seed {train_params.get('kmeans_init_seed', 0)} "
                if train_params.get("kmeans_init_dedup", True):
                    cmd += "--kmeans_init_dedup "

            # model-specific pieces
            if name == "rqvae":
                rqvae_cfg = train_params.get("rqvae", {})
                cmd += f"--e_dim {rqvae_cfg.get('e_dim', 64)} "
                cmd += "--num_emb_list " + " ".join(str(x) for x in rqvae_cfg.get("num_emb_list", [1024, 1024, 1024, 1024, 256])) + " "
                cmd += "--layers " + " ".join(str(x) for x in rqvae_cfg.get("layers", [512, 256, 128])) + " "
                cmd += "--sk_epsilons " + " ".join(str(x) for x in rqvae_cfg.get("sk_epsilons", [0.0, 0.0, 0.0, 0.003, 0.0])) + " "

                if rqvae_cfg.get("ema_codebook"):
                    cmd += "--ema_codebook "
                    cmd += f"--ema_decay {rqvae_cfg.get('ema_decay', 0.99)} "
                    cmd += f"--ema_eps {rqvae_cfg.get('ema_eps', 1e-5)} "

            if name == "rqvae_opq":
                rqopq_cfg = train_params.get("rqvae_opq", {})
                cmd += f"--e_dim {rqopq_cfg.get('e_dim', 64)} "
                cmd += "--num_emb_list " + " ".join(str(x) for x in rqopq_cfg.get("num_emb_list", [1024, 1024, 1024, 1024])) + " "
                cmd += "--layers " + " ".join(str(x) for x in rqopq_cfg.get("layers", [512, 256, 128])) + " "

                if rqopq_cfg.get("ema_codebook"):
                    cmd += "--ema_codebook "
                    cmd += f"--ema_decay {rqopq_cfg.get('ema_decay', 0.99)} "
                    cmd += f"--ema_eps {rqopq_cfg.get('ema_eps', 1e-5)} "

            run_cmd(cmd, desc=f"train {name}")

            if name not in ["rqkmeans", "rqkmeans_opq"]:
                # trainer writes best_collision_model.pth
                ckpt_path = (train_out / "best_collision_model.pth").resolve()
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Expected checkpoint not found: {ckpt_path}")
                ckpt = str(ckpt_path)

        if name not in ["rqkmeans", "rqkmeans_opq"] and not ckpt:
            raise ValueError(f"Strategy {name} missing ckpt (either provide ckpt or set train=true)")

        # ---- infer (optional) ----
        if strat.get("infer") is False:
            continue

        # kmeans/faiss train already produced indices/metrics in-place
        if name in ["rqkmeans", "rqkmeans_opq"] and strat.get("train"):
            continue

        if name in ["rqkmeans", "rqkmeans_opq"] and strat.get("infer") and not ckpt:
            raise ValueError(f"Strategy {name} infer=true requires an explicit ckpt path, or set train=true")

        if name not in STRATEGY_INFER_CMD:
            raise ValueError(f"Unknown strategy {name}")

        infer_params = _merge_dict(infer_defaults, strat.get("infer_overrides", {}))

        out_dir = run_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        if name not in ["rqkmeans", "rqkmeans_opq"]:
            if not ckpt or not Path(ckpt).exists():
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

        if name in ["rqvae", "rqkmeans", "rqvae_opq", "rqkmeans_opq"]:
            if meta_jsonl:
                cmd += f"--meta_jsonl {meta_jsonl} "

        # inference scripts default to use_sk=True; only pass flag when disabling
        if infer_params.get("use_sk") is False:
            cmd += "--no_use_sk "

        run_cmd(cmd, desc=f"infer {name}")

    print("\nAll benchmarks finished.")


if __name__ == "__main__":
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg_arg)
