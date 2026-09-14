import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ensure sid/src on path
CUR_DIR = Path(__file__).resolve().parent
SRC_DIR = CUR_DIR.parents[1]  # .../sid/src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.emb_dataset import EmbDataset
from models.rqvae_opq import RQOPQVAE
from models.rqkmeans_opq import RQOPQKMeans
from codebook.metrics import summarize


def parse_args():
    p = argparse.ArgumentParser(description="Generate indices for RQOPQ (VAE/KMeans)")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pth)")
    p.add_argument("--data_npy", required=True, help="Embedding matrix .npy")
    p.add_argument("--meta_jsonl", default=None, help="Optional meta.jsonl to be written to outputs")
    p.add_argument("--model", choices=["rqvae_opq", "rqkmeans_opq"], default="rqvae_opq")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="./outputs/indices_rqopq")
    p.add_argument(
        "--use_sk",
        action="store_true",
        default=True,
        help="Use Sinkhorn assignment when generating indices (default: enabled)",
    )
    p.add_argument("--no_use_sk", action="store_false", dest="use_sk", help="Disable Sinkhorn assignment")
    p.add_argument("--max_batches", type=int, default=None, help="Debug: limit batches")
    return p.parse_args()


def load_model(ckpt_path: str, model_type: str, device: str, *, in_dim_fallback: int | None = None):
    # PyTorch 2.6 defaults weights_only=True; OPQ checkpoints need full state
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    args = ckpt.get("args")
    args_dict = ckpt.get("args_dict")
    if args_dict is None and args is not None:
        args_dict = getattr(args, "__dict__", None)
    if args_dict is None:
        args_dict = {}

    state_dict = ckpt["state_dict"]

    def _get(k, default=None):
        return args_dict.get(k, getattr(args, k, default) if args is not None else default)

    common_kwargs = dict(
        num_emb_list=_get("num_emb_list"),
        e_dim=_get("e_dim"),
        beta=_get("beta", 0.25),
        kmeans_init=_get("kmeans_init", False),
        kmeans_iters=_get("kmeans_iters", 100),
        kmeans_init_max_samples=_get("kmeans_init_max_samples", 200_000),
        kmeans_init_seed=_get("kmeans_init_seed", 0),
        kmeans_init_dedup=_get("kmeans_init_dedup", True),
        sk_iters=_get("sk_iters", 100),
    )

    if model_type == "rqvae_opq":
        in_dim = _get("in_dim") or _get("input_dim") or in_dim_fallback
        if in_dim is None:
            raise ValueError("Missing in_dim in checkpoint args; pass in_dim_fallback from data_npy shape")

        model = RQOPQVAE(
            in_dim=in_dim,
            layers=_get("layers"),
            dropout_prob=_get("dropout_prob", 0.0),
            bn=_get("bn", False),
            loss_type=_get("loss_type", "mse"),
            quant_loss_weight=_get("quant_loss_weight", 1.0),
            **common_kwargs,
        )
    else:
        model = RQOPQKMeans(**common_kwargs)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, args_dict


def _iter_meta(meta_jsonl: str):
    with open(meta_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _read_meta_batch(meta_iter, bsz: int):
    batch = []
    for _ in range(bsz):
        try:
            batch.append(next(meta_iter))
        except StopIteration:
            raise ValueError("meta_jsonl rows < embeddings rows")
    return batch


def main():
    args = parse_args()

    device = torch.device(args.device)
    ds = EmbDataset(args.data_npy)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    meta_iter = _iter_meta(args.meta_jsonl) if args.meta_jsonl else None

    model, ckpt_args = load_model(args.ckpt, args.model, device, in_dim_fallback=int(ds.data.shape[1]))

    indices_all = []
    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(dl, desc="infer")):
            if args.max_batches is not None and bidx >= args.max_batches:
                break
            bsz = batch.size(0)
            meta_batch = None
            if meta_iter is not None:
                meta_batch = _read_meta_batch(meta_iter, bsz)

            batch = batch.to(device)
            idx = model.get_indices(batch, use_sk=args.use_sk)
            idx = idx.view(idx.size(0), -1).cpu().numpy()
            indices_all.append(idx)

    if meta_iter is not None:
        try:
            next(meta_iter)
            raise ValueError("meta_jsonl rows > embeddings rows")
        except StopIteration:
            pass

    codes = np.concatenate(indices_all, axis=0)

    # metrics should be computed on the same subset as codes (max_batches may truncate)
    emb = np.asarray(ds.data[: codes.shape[0]])
    # OPQ: no hardcode layer, include cosine metrics for all 5 codes
    metrics = summarize(codes, emb=emb, exclude_last_level_cosine=False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # benchmark layout wants checkpoints for VAE; do not create codebooks/ by default
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "indices.jsonl"
    out_metrics = out_dir / "metrics.json"

    meta_out_iter = _iter_meta(args.meta_jsonl) if args.meta_jsonl else None
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]
    with out_jsonl.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(codes):
            sid = [prefix[i].format(int(v)) for i, v in enumerate(row)]
            rec = {"idx": idx, "indices": row.tolist(), "tokens": sid, "sid": sid}
            if meta_out_iter is not None:
                try:
                    meta_row = next(meta_out_iter)
                except StopIteration:
                    raise ValueError("meta_jsonl rows < embeddings rows when writing outputs")

                # Format-B: keep original meta row fields at top-level, and inject sid.
                if isinstance(meta_row, dict):
                    meta_row = dict(meta_row)
                    meta_row.update(rec)
                    rec = meta_row
                else:
                    rec["meta"] = meta_row

                # Ensure meta_json exists and contains saletype/publish_year as strings.
                def _pick(obj: dict, key: str):
                    if not isinstance(obj, dict):
                        return None
                    v = obj.get(key)
                    return v

                if "meta_json" not in rec or not isinstance(rec.get("meta_json"), dict):
                    rec["meta_json"] = {}

                # saletype: prefer existing meta_json, then top-level, then meta dict
                s = _pick(rec.get("meta_json"), "saletype")
                if s is None:
                    s = _pick(rec, "saletype")
                if s is None:
                    s = _pick(rec.get("meta"), "saletype")
                rec["meta_json"]["saletype"] = str(0 if s is None else s)

                # publish_year: prefer existing meta_json, then top-level, then meta dict
                y = _pick(rec.get("meta_json"), "publish_year")
                if y is None:
                    y = _pick(rec, "publish_year")
                if y is None:
                    y = _pick(rec.get("meta"), "publish_year")
                rec["meta_json"]["publish_year"] = "未知" if y is None else str(y)

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if meta_out_iter is not None:
        try:
            next(meta_out_iter)
            raise ValueError("meta_jsonl rows > embeddings rows")
        except StopIteration:
            pass

    with out_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved indices -> {out_jsonl}")
    print(f"Saved metrics -> {out_metrics}")
    print(metrics)


if __name__ == "__main__":
    main()
