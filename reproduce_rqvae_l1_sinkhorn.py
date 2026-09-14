#!/usr/bin/env python3
"""
完整复现 RQVAE + L1 Sinkhorn 实验

对任务2和任务3分别训练 RQVAE 模型，第1层使用 Sinkhorn，其他层不用
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = REPO_ROOT / "sid" / "scripts" / "train" / "train_rqvae.py"
INFER_SCRIPT = REPO_ROOT / "sid" / "src" / "codebook" / "generate_indices_rqvae.py"

# RQVAE + L1 Sinkhorn 配置
RQVAE_L1_SINKHORN_CONFIG = {
    "num_emb_list": [1024, 1024, 1024, 1024, 256],  # 5层，最后一层256
    "e_dim": 128,
    "layers": [512, 256, 128],  # MLP encoder/decoder 结构
    "sk_epsilons": [0.01, 0.0, 0.0, 0.0, 0.0],  # 只在 L1 使用 Sinkhorn
    "sk_iters": 50,
    "ema_codebook": True,
    "ema_decay": 0.99,
    "ema_eps": 1e-5,
    "kmeans_init": True,
    "kmeans_iters": 20,
    "kmeans_init_max_samples": 200000,
    "epochs": 40,
    "lr": 0.001,
    "batch_size": 1024,
    "eval_step": 1,
}

TASKS = {
    "task2": {
        "data_npy": "2_meta_full_video.npy",
        "meta_jsonl": "2_meta.jsonl",
        "output_dir": "results_rqvae_l1_sinkhorn/task2",
    },
    "task3": {
        "data_npy": "3_text_format_embedding.npy",
        "meta_jsonl": "3_meta.jsonl",
        "output_dir": "results_rqvae_l1_sinkhorn/task3",
    },
}


def run_cmd(cmd, desc):
    """执行命令"""
    print(f"\n{'='*60}")
    print(f"[RUN] {desc}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    result = subprocess.run(cmd, shell=True, text=True, encoding="utf-8")
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
        f"--meta_jsonl {task['meta_jsonl']} "
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
        f"--device cuda "
    )
    
    run_cmd(cmd, f"Train RQVAE + L1 Sinkhorn for {task_name}")
    
    return ckpt_dir


def generate_indices(task_name, ckpt_dir):
    """生成 indices.jsonl"""
    task = TASKS[task_name]
    output_dir = Path(task["output_dir"]) / "indices"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 找到最佳 checkpoint
    ckpt_files = sorted(ckpt_dir.glob("*.pt"))
    if not ckpt_files:
        print(f"[FAIL] No checkpoint found in {ckpt_dir}")
        sys.exit(1)
    
    best_ckpt = ckpt_files[-1]  # 使用最后一个 epoch 的 checkpoint
    
    cmd = (
        f"python {INFER_SCRIPT} "
        f"--model rqvae "
        f"--ckpt {best_ckpt} "
        f"--data_npy {task['data_npy']} "
        f"--meta_jsonl {task['meta_jsonl']} "
        f"--output_dir {output_dir} "
        f"--device cuda "
        f"--batch_size 1024 "
        f"--use_sk "  # 推理时使用 Sinkhorn
    )
    
    run_cmd(cmd, f"Generate indices for {task_name}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="复现 RQVAE + L1 Sinkhorn 实验")
    parser.add_argument("--tasks", nargs="+", choices=["task2", "task3", "all"], 
                       default=["all"], help="要运行的任务")
    parser.add_argument("--train_only", action="store_true", 
                       help="只训练，不生成 indices")
    parser.add_argument("--infer_only", action="store_true", 
                       help="只生成 indices（需要已有 checkpoint）")
    args = parser.parse_args()
    
    tasks_to_run = []
    if "all" in args.tasks:
        tasks_to_run = ["task2", "task3"]
    else:
        tasks_to_run = args.tasks
    
    print("\n" + "="*60)
    print("RQVAE + L1 Sinkhorn 实验复现")
    print("="*60)
    print(f"任务: {', '.join(tasks_to_run)}")
    print(f"配置: L1 Sinkhorn epsilon={RQVAE_L1_SINKHORN_CONFIG['sk_epsilons'][0]}")
    print("="*60 + "\n")
    
    for task_name in tasks_to_run:
        print(f"\n{'#'*60}")
        print(f"# 开始处理: {task_name}")
        print(f"{'#'*60}\n")
        
        ckpt_dir = None
        
        if not args.infer_only:
            # 训练
            ckpt_dir = train_rqvae(task_name, RQVAE_L1_SINKHORN_CONFIG)
        
        if not args.train_only:
            # 生成 indices
            if ckpt_dir is None:
                # infer_only 模式，需要找到现有的 checkpoint
                ckpt_dir = Path(TASKS[task_name]["output_dir"]) / "checkpoints"
                if not ckpt_dir.exists():
                    print(f"[FAIL] Checkpoint directory not found: {ckpt_dir}")
                    sys.exit(1)
            
            output_dir = generate_indices(task_name, ckpt_dir)
            
            print(f"\n[OK] {task_name} 完成！")
            print(f"[DIR] 结果保存在: {output_dir}")
    
    print("\n" + "="*60)
    print("[SUCCESS] 所有任务完成！")
    print("="*60)
    print("\n结果目录:")
    for task_name in tasks_to_run:
        task_dir = Path(TASKS[task_name]["output_dir"])
        print(f"  {task_name}: {task_dir}")
    print()


if __name__ == "__main__":
    main()
