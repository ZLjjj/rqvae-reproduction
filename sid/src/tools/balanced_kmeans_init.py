#!/usr/bin/env python3
"""Balanced KMeans init helper for VQ codebooks.

Goal: avoid domain-order bias in the very first KMeans codebook initialization.

This module builds a fixed-size 1:1 sample index list from meta.jsonl (aligned with embedding.npy rows),
then provides a convenience function to run init_emb() for all VQ layers.

It is intentionally lightweight and dependency-free beyond numpy/torch.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class BalancedInitConfig:
    meta_jsonl: str
    domains: tuple[str, str] = ("station", "video")
    per_domain: int = 100_000
    seed: int = 0


def sample_balanced_indices(cfg: BalancedInitConfig) -> np.ndarray:
    """Return shuffled indices with exact 1:1 domain ratio (or less if insufficient rows).

    - indices correspond to line numbers in meta.jsonl (0-based)
    - output is shuffled so downstream sees mixed domains
    """

    buckets: dict[str, list[int]] = defaultdict(list)

    with open(cfg.meta_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            dom = rec.get("domain")
            if dom in cfg.domains:
                buckets[dom].append(i)

    a, b = cfg.domains
    na = len(buckets.get(a, []))
    nb = len(buckets.get(b, []))
    take = min(cfg.per_domain, na, nb)
    if take <= 0:
        raise ValueError(f"No enough rows for balanced init: {a}={na}, {b}={nb}")

    rng = np.random.default_rng(cfg.seed)

    idx_a = np.asarray(buckets[a], dtype=np.int64)
    idx_b = np.asarray(buckets[b], dtype=np.int64)

    # sample without replacement
    choose_a = rng.choice(idx_a, size=take, replace=False)
    choose_b = rng.choice(idx_b, size=take, replace=False)

    out = np.concatenate([choose_a, choose_b], axis=0)
    rng.shuffle(out)
    return out


def maybe_balanced_init_all_vq(
    *,
    model,
    emb: np.ndarray,
    meta_jsonl: str | None,
    kmeans_init: bool,
    kmeans_init_seed: int,
    kmeans_init_max_samples: int | None,
    device: torch.device,
    domains: tuple[str, str] = ("station", "video"),
    per_domain: int | None = None,
) -> dict | None:
    """If meta_jsonl is provided and kmeans_init=True, do balanced init for all VQ layers.

    This runs before training starts to prevent each VQ from initializing on the first batch only.

    Returns a small stats dict when executed; otherwise None.
    """

    if not kmeans_init or not meta_jsonl:
        return None

    if per_domain is None:
        if kmeans_init_max_samples is None:
            # fall back to a safe default
            per_domain = 100_000
        else:
            per_domain = max(int(kmeans_init_max_samples) // 2, 1)

    cfg = BalancedInitConfig(meta_jsonl=meta_jsonl, domains=domains, per_domain=int(per_domain), seed=int(kmeans_init_seed))
    idx = sample_balanced_indices(cfg)

    # Pull the sample to torch once; init_emb() will do its own max_samples subsampling if configured.
    # Keep it on device to avoid repeated H2D copies.
    x0 = torch.from_numpy(np.asarray(emb[idx], dtype=np.float32)).to(device)

    initted = 0

    # RQVAE path: model.rq.vq_layers (first 4 learned)
    rq = getattr(model, "rq", None)
    if rq is not None and hasattr(rq, "vq_layers"):
        for vq in rq.vq_layers:
            if getattr(vq, "kmeans_init", False) and not getattr(vq, "initted", True):
                vq.init_emb(x0.view(-1, vq.e_dim))
                initted += 1

    # RQOPQVAE path: model.rqopq.layers = [PlainRQ, PlainRQ, PlainRQ, OPQ]
    rqopq = getattr(model, "rqopq", None)
    if rqopq is not None and hasattr(rqopq, "layers"):
        for layer in rqopq.layers:
            # PlainRQ has .quantizer; OPQ has .quantizers (list)
            q = getattr(layer, "quantizer", None)
            if q is not None:
                if getattr(q, "kmeans_init", False) and not getattr(q, "initted", True):
                    q.init_emb(x0.view(-1, q.e_dim))
                    initted += 1
                continue

            qs = getattr(layer, "quantizers", None)
            if qs is not None:
                for q in qs:
                    if getattr(q, "kmeans_init", False) and not getattr(q, "initted", True):
                        q.init_emb(x0.view(-1, q.e_dim))
                        initted += 1

    return {
        "meta_jsonl": meta_jsonl,
        "domains": cfg.domains,
        "per_domain": cfg.per_domain,
        "total_samples": int(idx.shape[0]),
        "seed": cfg.seed,
        "vq_initted": initted,
    }
