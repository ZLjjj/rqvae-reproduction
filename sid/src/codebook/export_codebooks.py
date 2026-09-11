import os
from pathlib import Path

import numpy as np
import torch


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_rq_codebooks(model, out_dir: str | os.PathLike, *, include_hardcode: bool):
    """Export codebooks as .npy.

    Creates:
    - codebooks_4layer.npy : (4, K, D)
    - codebooks_5layer.npy : (5, K, D) when include_hardcode=True

    Note:
    - hardcode layer has no vectors; we write a zero placeholder to keep shape stable.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not hasattr(model, "rq") or not hasattr(model.rq, "get_codebook"):
        raise ValueError("model.rq.get_codebook() is required to export codebooks")

    cb = model.rq.get_codebook()  # (n_layers, K, D)
    cb = _as_numpy(cb)

    cb4 = cb[:4]
    np.save(out / "codebooks_4layer.npy", cb4)

    if include_hardcode:
        # placeholder for hardcode
        hard = np.zeros((1, cb4.shape[1], cb4.shape[2]), dtype=cb4.dtype)
        cb5 = np.concatenate([cb4, hard], axis=0)
        np.save(out / "codebooks_5layer.npy", cb5)


def save_opq_codebooks(model, out_dir: str | os.PathLike):
    """Export OPQ strategy codebooks.

    We only export a 5-codebook placeholder array to match 3+2 layout.
    Vectors are taken from RQ layers when available; OPQ doesn't expose a simple codebook
    in this simplified implementation, so we write zeros for OPQ parts.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Try to fetch RQ layer codebooks if present
    cb_rq = None
    if hasattr(model, "rqopq") and hasattr(model.rqopq, "layers"):
        # layers: [PlainRQ, PlainRQ, PlainRQ, OPQ]
        rq_layers = []
        for layer in model.rqopq.layers[:3]:
            if hasattr(layer, "vq") and hasattr(layer.vq, "get_codebook"):
                rq_layers.append(_as_numpy(layer.vq.get_codebook()))
            elif hasattr(layer, "get_codebook"):
                rq_layers.append(_as_numpy(layer.get_codebook()))
        if rq_layers:
            cb_rq = np.stack(rq_layers, axis=0)  # (3, K, D)

    if cb_rq is None:
        # fallback: (3, K, D) zeros
        K = getattr(model, "num_emb_list", [1024])[0]
        D = getattr(model, "e_dim", 64)
        cb_rq = np.zeros((3, K, D), dtype=np.float32)

    K = cb_rq.shape[1]
    D = cb_rq.shape[2]

    # OPQ has 2 sub-quantizers; placeholders
    opq1 = np.zeros((1, K, D), dtype=cb_rq.dtype)
    opq2 = np.zeros((1, K, D), dtype=cb_rq.dtype)

    cb5 = np.concatenate([cb_rq, opq1, opq2], axis=0)
    np.save(out / "codebooks_5layer.npy", cb5)
