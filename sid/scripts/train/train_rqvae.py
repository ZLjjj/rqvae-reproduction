#!/usr/bin/env python3
import argparse
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# allow `python sid/scripts/train/train_rqvae.py` from repo root
CUR_DIR = Path(__file__).resolve().parent
REPO_ROOT = CUR_DIR.parents[2]
SRC_DIR = REPO_ROOT / "sid" / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

from data.emb_dataset import EmbDataset
from models.rqvae import RQVAE
from training.trainer import Trainer


@dataclass
class Args:
    # training
    lr: float
    epochs: int
    batch_size: int
    num_workers: int
    eval_step: int
    learner: str
    lr_scheduler_type: str
    warmup_epochs: int
    weight_decay: float
    save_limit: int
    ckpt_dir: str
    device: str
    max_batches: int | None

    # model
    in_dim: int | None
    num_emb_list: list[int]
    e_dim: int
    layers: list[int]
    dropout_prob: float
    bn: bool
    loss_type: str
    quant_loss_weight: float
    beta: float
    kmeans_init: bool
    kmeans_iters: int
    kmeans_init_max_samples: int | None
    kmeans_init_seed: int
    kmeans_init_dedup: bool
    sk_epsilons: list[float]
    sk_iters: int

    # ema codebook (optional)
    ema_codebook: bool
    ema_decay: float
    ema_eps: float


def parse_args():
    p = argparse.ArgumentParser(description="Train RQVAE (4+1) in sid")

    p.add_argument("--data_npy", required=True)
    p.add_argument("--meta_jsonl", default=None, help="Optional meta.jsonl aligned with data_npy (for balanced kmeans init)")
    p.add_argument("--ckpt_dir", required=True)

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--max_batches", type=int, default=None, help="Debug limit per train/eval epoch")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--eval_step", type=int, default=10)
    p.add_argument("--learner", type=str, default="AdamW")
    p.add_argument("--lr_scheduler_type", type=str, default="constant")
    p.add_argument("--warmup_epochs", type=int, default=0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--save_limit", type=int, default=5)

    p.add_argument("--dropout_prob", type=float, default=0.0)
    p.add_argument("--bn", action="store_true")
    p.add_argument("--loss_type", type=str, default="mse")

    p.add_argument("--kmeans_init", action="store_true")
    p.add_argument("--kmeans_iters", type=int, default=100)
    p.add_argument("--kmeans_init_max_samples", type=int, default=200_000)
    p.add_argument("--kmeans_init_seed", type=int, default=0)
    p.add_argument("--kmeans_init_dedup", action="store_true")
    p.add_argument("--sk_epsilons", type=float, nargs="+", default=[0.0, 0.0, 0.0, 0.003, 0.0])
    p.add_argument("--sk_iters", type=int, default=100)

    p.add_argument("--ema_codebook", action="store_true", help="Enable post-step EMA smoothing for the first 4 VQ codebooks")
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--ema_eps", type=float, default=1e-5)

    p.add_argument("--num_emb_list", type=int, nargs="+", default=[1024, 1024, 1024, 1024, 256])
    p.add_argument("--e_dim", type=int, default=64)
    p.add_argument("--quant_loss_weight", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--layers", type=int, nargs="+", default=[512, 256, 128])

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    cli = parse_args()
    set_seed(cli.seed)

    logging.basicConfig(level=logging.INFO)

    ds = EmbDataset(cli.data_npy)
    dl = DataLoader(
        ds,
        batch_size=cli.batch_size,
        shuffle=True,
        num_workers=cli.num_workers,
        pin_memory=True,
    )

    args = Args(
        lr=cli.lr,
        epochs=cli.epochs,
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        eval_step=cli.eval_step,
        learner=cli.learner,
        lr_scheduler_type=cli.lr_scheduler_type,
        warmup_epochs=cli.warmup_epochs,
        weight_decay=cli.weight_decay,
        save_limit=cli.save_limit,
        ckpt_dir=cli.ckpt_dir,
        device=cli.device,
        max_batches=cli.max_batches,
        in_dim=None,
        num_emb_list=list(cli.num_emb_list),
        e_dim=cli.e_dim,
        layers=list(cli.layers),
        dropout_prob=cli.dropout_prob,
        bn=bool(cli.bn),
        loss_type=cli.loss_type,
        quant_loss_weight=cli.quant_loss_weight,
        beta=cli.beta,
        kmeans_init=bool(cli.kmeans_init),
        kmeans_iters=cli.kmeans_iters,
        kmeans_init_max_samples=cli.kmeans_init_max_samples,
        kmeans_init_seed=cli.kmeans_init_seed,
        kmeans_init_dedup=bool(cli.kmeans_init_dedup),
        sk_epsilons=list(cli.sk_epsilons),
        sk_iters=cli.sk_iters,
        ema_codebook=bool(cli.ema_codebook),
        ema_decay=float(cli.ema_decay),
        ema_eps=float(cli.ema_eps),
    )

    model = RQVAE(
        in_dim=ds.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        kmeans_init_max_samples=args.kmeans_init_max_samples,
        kmeans_init_seed=args.kmeans_init_seed,
        kmeans_init_dedup=args.kmeans_init_dedup,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )

    if args.ema_codebook:
        from models.vq_mix import VectorQuantizerMix

        rq = model.rq
        for i, vq in enumerate(rq.vq_layers):
            if i < 4:
                vmix = VectorQuantizerMix(
                    vq.n_e,
                    vq.e_dim,
                    beta=vq.beta,
                    kmeans_init=vq.kmeans_init,
                    kmeans_iters=vq.kmeans_iters,
                    kmeans_init_max_samples=getattr(vq, "kmeans_init_max_samples", 200_000),
                    kmeans_init_seed=getattr(vq, "kmeans_init_seed", 0),
                    kmeans_init_dedup=getattr(vq, "kmeans_init_dedup", True),
                    sk_epsilon=vq.sk_epsilon,
                    sk_iters=vq.sk_iters,
                    ema_decay=args.ema_decay,
                    ema_eps=args.ema_eps,
                )
                vmix.embedding.weight.data.copy_(vq.embedding.weight.data)
                rq.vq_layers[i] = vmix

    # Optional: balanced KMeans init to avoid domain-order bias.
    if cli.meta_jsonl and args.kmeans_init:
        from tools.balanced_kmeans_init import maybe_balanced_init_all_vq

        init_stats = maybe_balanced_init_all_vq(
            model=model,
            emb=ds.data,
            meta_jsonl=cli.meta_jsonl,
            kmeans_init=args.kmeans_init,
            kmeans_init_seed=args.kmeans_init_seed,
            kmeans_init_max_samples=args.kmeans_init_max_samples,
            device=torch.device(args.device),
        )
        logging.info("balanced_kmeans_init: %s", init_stats)

    trainer = Trainer(args, model, steps_per_epoch=len(dl))
    best_loss, best_collision = trainer.fit(dl)

    print("Best Loss", best_loss)
    print("Best Collision Rate", best_collision)


if __name__ == "__main__":
    main()
