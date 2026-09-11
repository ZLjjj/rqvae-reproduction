#!/usr/bin/env python3
"""
RQVAE + L1+L4 Sinkhorn 一键复现脚本

完美复现任务3的 CR=0.0907% 结果
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = REPO_ROOT / "sid" / "scripts" / "train" / "train_rqvae.py"
INFER_SCRIPT = REPO_ROOT / "sid" / "src" / "codebook" / "generate_indices_rqvae.py"

# 正确的配置（完美复现 CR=0.0907%）
CORRECT_CONFIG = {
    "num_emb_list": [1024, 1024, 1024, 1024, 256],
    "e_dim": 128,
    "layers": [512, 256, 128],
    "sk_epsilons": [0.003, 0.0, 0.0, 0.003, 0.0],  # L1 和 L4 都用 Sinkhorn
    "sk_iters": 100,
    "ema_codebook": True,
    "ema_decay": 0.99,
    "ema_eps": 1e-5,
    "kmeans_init": True,
    "kmeans_iters": 100,
    "kmeans_init_max_samples": 200000,
    "epochs": 10,
    "lr": 0.001,
    "batch_size": 1024,
    "eval_step": 1,
}

TASKS = {
    "task3": {
        "data_npy": "3_text_format_embedding.npy",
        "meta_jsonl": "3_meta.jsonl",
        "output_dir": "results_reproduction/task3",
    },
}


def run_cmd(cmd, desc):
    """执行命令"""
    print(f"\n{'='*60}")
    print(f"[RUN] {desc}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    result = subprocess.run(cmd, shell=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        print(f"[FAIL] Failed: {desc}")
        sys.exit(1)
    print(f"[OK] Completed: {desc}\n")


def train_rqvae(task_name, config):
    """训练 RQVAE 模型"""
    task = TASKS[task_name]
    ckpt_dir = Path(task["output_dir"]) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = (
        f"python {TRAIN_SCRIPT} "
        f"--data_npy {task['data_npy']} "
        f"--ckpt_dir {ckpt_dir} "
        f"--num_emb_list {' '.join(map(str, config['num_emb_list']))} "
        f"--e_dim {config['e_dim']} "
        f"--layers {' '.join(map(str, config['layers']))} "
        f"--sk_epsilons {' '.join(map(str, config['sk_epsilons']))} "
        f"--sk_iters {config['sk_iters']} "
        f"--ema_codebook "
        f"--ema_decay {config['ema_decay']} "
        f"--ema_eps {config['ema_eps']} "
        f"--kmeans_init "
        f"--kmeans_iters {config['kmeans_iters']} "
        f"--kmeans_init_max_samples {config['kmeans_init_max_samples']} "
        f"--epochs {config['epochs']} "
        f"--lr {config['lr']} "
        f"--batch_size {config['batch_size']} "
        f"--eval_step {config['eval_step']} "
        f"--device cpu "
    )
    
    run_cmd(cmd, f"Train RQVAE + L1+L4 Sinkhorn for {task_name}")
    
    return ckpt_dir


def generate_indices(task_name, ckpt_dir):
    """生成 indices.jsonl"""
    task = TASKS[task_name]
    output_dir = Path(task["output_dir"]) / "indices"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_ckpt = ckpt_dir / "best_collision_model.pth"
    if not best_ckpt.exists():
        print(f"[FAIL] No checkpoint found at {best_ckpt}")
        sys.exit(1)
    
    cmd = (
        f"python {INFER_SCRIPT} "
        f"--model rqvae "
        f"--ckpt {best_ckpt} "
        f"--data_npy {task['data_npy']} "
        f"--meta_jsonl {task['meta_jsonl']} "
        f"--output_dir {output_dir} "
        f"--device cpu "
        f"--batch_size 1024 "
        f"--use_sk "
    )
    
    run_cmd(cmd, f"Generate indices for {task_name}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="复现 RQVAE + L1+L4 Sinkhorn")
    parser.add_argument("--task", choices=["task3"], default="task3")
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--infer_only", action="store_true")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("RQVAE + L1+L4 Sinkhorn")
    print("="*60)
    print(f"Task: {args.task}")
    print(f"Target: CR=0.0907%, ICR=99.91%")
    print("="*60 + "\n")
    
    task_name = args.task
    ckpt_dir = None
    
    if not args.infer_only:
        ckpt_dir = train_rqvae(task_name, CORRECT_CONFIG)
    
    if not args.train_only:
        if ckpt_dir is None:
            ckpt_dir = Path(TASKS[task_name]["output_dir"]) / "checkpoints"
        output_dir = generate_indices(task_name, ckpt_dir)
        print(f"\n[OK] Results: {output_dir}")
    
    print("\n[SUCCESS] Done!")


if __name__ == "__main__":
    main()

