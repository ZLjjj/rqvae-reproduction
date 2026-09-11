import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ensure sid/src on path
CUR_DIR = Path(__file__).resolve().parent
SRC_DIR = CUR_DIR.parents[1] / "src"  # .../sid/src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.emb_dataset import EmbDataset
from models.rqvae import RQVAE
from models.rqkmeans import RQKMeans
from codebook.metrics import summarize


def parse_args():
    p = argparse.ArgumentParser(description="Generate indices for RQVAE / RQKMeans")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pth)")
    p.add_argument("--data_npy", required=True, help="Embedding matrix .npy")
    p.add_argument("--meta_jsonl", default=None, help="meta.jsonl for hardcode (saletype/publish_year)")
    p.add_argument("--model", choices=["rqvae", "rqkmeans"], default="rqvae")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="./outputs/indices_rqvae")
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
    # PyTorch 2.6 defaults weights_only=True; rqvae checkpoints need full state
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Backward compatible:
    # - new checkpoints store args_dict (plain dict)
    # - old checkpoints stored args as a dataclass defined in __main__ (not reliably loadable)
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
        sk_epsilons=_get("sk_epsilons", [0.0, 0.0, 0.0, 0.003]),
        sk_iters=_get("sk_iters", 100),
    )

    if model_type == "rqvae":
        in_dim = _get("in_dim") or _get("input_dim") or in_dim_fallback
        if in_dim is None:
            raise ValueError("Missing in_dim in checkpoint args; pass in_dim_fallback from data_npy shape")

        model = RQVAE(
            in_dim=in_dim,
            layers=_get("layers"),
            dropout_prob=_get("dropout_prob", 0.0),
            bn=_get("bn", False),
            loss_type=_get("loss_type", "mse"),
            quant_loss_weight=_get("quant_loss_weight", 1.0),
            **common_kwargs,
        )
    else:
        # rqkmeans checkpoints were historically trained with a 256-sized hardcode last level.
        # If HardCodeMapper is expanded (e.g. 9 bits -> 512), ensure inference uses a compatible
        # num_emb_list[4] even when the checkpoint args didn't record it.
        num_emb_list = common_kwargs.get("num_emb_list")
        if not num_emb_list:
            num_emb_list = [1024, 1024, 1024, 1024, 512]
        elif len(num_emb_list) >= 5 and num_emb_list[4] == 256:
            num_emb_list = list(num_emb_list)
            num_emb_list[4] = 512
        common_kwargs["num_emb_list"] = num_emb_list
        model = RQKMeans(**common_kwargs)

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
    base = 0
    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(dl, desc="infer")):
            if args.max_batches is not None and bidx >= args.max_batches:
                break

            bsz = batch.size(0)
            meta_batch = None
            hard_fields = None
            if meta_iter is not None:
                meta_batch = _read_meta_batch(meta_iter, bsz)

                def _get_first_present(m: dict, keys: list[str], default=0):
                    """Support both flat and nested meta fields with alias list."""
                    for k in keys:
                        if k in m:
                            v = m.get(k, default)
                            return default if v is None else v
                    inner = m.get("meta")
                    if isinstance(inner, dict):
                        for k in keys:
                            if k in inner:
                                v = inner.get(k, default)
                                return default if v is None else v
                    return default

                hard_fields = {
                    "saletype": [_get_first_present(m, ["saletype", "sale_type", "pay_type", "paytype"], 0) for m in meta_batch],
                    "publish_year": [_get_first_present(m, ["publish_year"], 0) for m in meta_batch],
                }

            batch = batch.to(device)
            idx = model.get_indices(batch, use_sk=args.use_sk, hard_fields=hard_fields)
            idx = idx.view(idx.size(0), -1).cpu().numpy()
            indices_all.append(idx)
            base += bsz

    if meta_iter is not None:
        try:
            next(meta_iter)
            raise ValueError("meta_jsonl rows > embeddings rows")
        except StopIteration:
            pass

    codes = np.concatenate(indices_all, axis=0)
    # metrics should be computed on the same subset as codes (max_batches may truncate)
    emb = np.asarray(ds.data[: codes.shape[0]])

    # metrics: output both 4-layer and 5-layer views
    metrics = {
        "metrics_4layer": summarize(codes[:, :4], emb=emb, exclude_last_level_cosine=False),
        "metrics_5layer": summarize(codes, emb=emb, exclude_last_level_cosine=True),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # benchmark layout wants checkpoints for VAE; do not create codebooks/ by default
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "indices.jsonl"
    out_metrics = out_dir / "metrics.json"

    meta_out_iter = _iter_meta(args.meta_jsonl) if args.meta_jsonl else None
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]  # 5th layer may exceed 255 (hardcode uses 9 bits)
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
                    return obj.get(key)

                def _top_first(obj: dict, keys: list[str]):
                    for k in keys:
                        if isinstance(obj, dict) and k in obj:
                            return obj.get(k)
                    return None

                def _normalize_sale(v):
                    if v is None:
                        return 0
                    if isinstance(v, str):
                        t = v.strip().upper()
                        if t == "FREE":
                            return 1
                        if t == "PAY":
                            return 2
                        if t.isdigit():
                            try:
                                return int(t)
                            except Exception:
                                return 0
                        return 0
                    try:
                        iv = int(v)
                        if iv in (0, 1, 2):
                            return iv
                        return 0
                    except Exception:
                        return 0

                # 优先从 meta 中取，其次顶层字段；不再使用/写入 meta_json
                s = _top_first(rec.get("meta", {}), ["saletype", "pay_type", "paytype", "sale_type"])
                if s is None:
                    s = _top_first(rec, ["saletype", "pay_type", "paytype", "sale_type"])
                rec["saletype"] = str(_normalize_sale(s))

                y = _top_first(rec.get("meta", {}), ["publish_year"])
                if y is None:
                    y = _top_first(rec, ["publish_year"])
                rec["publish_year"] = "未知" if y is None else str(y)
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
