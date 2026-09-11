#!/usr/bin/env python3
"""Train RQKMeans+OPQ via FAISS and directly export indices/codebooks.

Layout (within output_dir):
- indices.jsonl
- metrics.json
- codebooks/codebooks_5layer.npy

This matches the simplified 3+2 strategy:
- 3 RQ levels (RQ prefix)
- 2 OPQ subquantizers (PQ codes)

Note: we do NOT export a real OPQ codebook tensor here (FAISS OPQ/PQ doesn't map cleanly
into (K,D) like our torch RQ). We export placeholders to keep file contract stable.
"""

import argparse
import json
import os
from pathlib import Path

import faiss
import numpy as np
import ot

CUR_DIR = Path(__file__).resolve().parent
REPO_ROOT = CUR_DIR.parents[2]
SRC_DIR = REPO_ROOT / "sid" / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))


def train_rq_prefix(data: np.ndarray, levels: int, codebook_size: int):
    nbits = int(np.log2(int(codebook_size)))
    rq = faiss.ResidualQuantizer(data.shape[1], levels, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.max_beam_size = 1
    rq.train(np.ascontiguousarray(data.astype(np.float32)))
    return rq


def unpack_rq_codes(codes_packed: np.ndarray, nbits: int, num_levels: int):
    N = codes_packed.shape[0]
    ints = np.zeros(N, dtype=np.int64)
    for i in range(codes_packed.shape[1]):
        ints |= codes_packed[:, i].astype(np.int64) << (8 * i)

    mask = (1 << nbits) - 1
    out = np.zeros((N, num_levels), dtype=np.int32)
    for i in range(num_levels):
        out[:, i] = (ints >> (i * nbits)) & mask
    return out


def unpack_pq_codes(codes_packed: np.ndarray, nbits: int, M: int):
    N = codes_packed.shape[0]
    mask = (1 << nbits) - 1
    out = np.zeros((N, M), dtype=np.int32)
    for row_idx, row in enumerate(codes_packed):
        val = int.from_bytes(row.tobytes(), byteorder="little", signed=False)
        for m in range(M):
            out[row_idx, m] = (val >> (m * nbits)) & mask
    return out


def train_opq_pq(residuals: np.ndarray, *, subquantizers: int, codebook_size: int, opq_niter: int, pq_niter: int):
    d = residuals.shape[1]
    nbits = int(np.log2(int(codebook_size)))

    opq = faiss.OPQMatrix(d, subquantizers)
    opq.niter = int(opq_niter)
    opq.train(residuals)

    residuals_rot = opq.apply_py(residuals)

    pq = faiss.ProductQuantizer(d, subquantizers, nbits)
    pq.niter = int(pq_niter)
    pq.train(residuals_rot)

    return opq, pq


def encode_opq(opq, pq, residuals: np.ndarray):
    residuals_rot = opq.apply_py(residuals)
    codes_packed = pq.compute_codes(residuals_rot)
    nbits = int(np.log2(pq.ksub))
    return unpack_pq_codes(codes_packed, nbits, pq.M)


def get_rq_codebooks(rq):
    M, d = rq.M, rq.d
    nbits_vec = faiss.vector_to_array(rq.nbits)
    if not np.all(nbits_vec == nbits_vec[0]):
        return None
    K = 1 << int(nbits_vec[0])
    cb_flat = faiss.vector_to_array(rq.codebooks).astype(np.float32)
    return cb_flat.reshape(M, K, d)


def compute_metrics(codes: np.ndarray, emb: np.ndarray):
    from codebook.metrics import summarize

    # OPQ has no hardcode layer, include cosine for all 5 codes
    return summarize(codes, emb=emb, exclude_last_level_cosine=False)


def pairwise_sq_dists_batch(X, C, C_norm2=None):
    if C_norm2 is None:
        C_norm2 = np.sum(C * C, axis=1)
    X_norm2 = np.sum(X * X, axis=1, keepdims=True)
    dots = X @ C.T
    return X_norm2 + C_norm2[None, :] - 2.0 * dots


def estimate_tau(residuals, centroids, sample_size=4000, percentile=90, min_tau=1e-6):
    N = residuals.shape[0]
    idx = np.random.choice(N, size=min(sample_size, N), replace=False)
    X = residuals[idx]
    Cn2 = np.sum(centroids * centroids, axis=1)
    D = pairwise_sq_dists_batch(X, centroids, Cn2)
    spread = np.percentile(D - D.min(axis=1, keepdims=True), percentile, axis=1)
    tau = float(np.median(spread) * 0.1)
    return max(tau, min_tau)


def sinkhorn_balance_level(residuals, centroids, *, iters=30, tau=None, topk=32, seed=42, verbose=True):
    rng = np.random.RandomState(seed)
    N = residuals.shape[0]
    K = centroids.shape[0]

    capacities = np.full(K, N // K, dtype=np.int64)
    capacities[: (N % K)] += 1

    if tau is None:
        tau = estimate_tau(residuals, centroids)

    if verbose:
        print(f"  Sinkhorn(OPQ): N={N} K={K} tau={tau:.5g} iters={iters}")

    a = np.ones(N) / N
    b = capacities / float(N)
    Cn2 = np.sum(centroids * centroids, axis=1)

    D_full = pairwise_sq_dists_batch(residuals, centroids, Cn2).astype(np.float64)
    P = ot.sinkhorn(a, b, D_full, tau, numItermax=int(iters))

    remaining = capacities.copy()
    assign = np.empty(N, dtype=np.int32)
    order = np.arange(N)
    rng.shuffle(order)

    for i in order:
        probs = P[i]
        if topk and topk < K:
            cand = np.argpartition(-probs, topk - 1)[:topk]
            cand = cand[np.argsort(-probs[cand])]
        else:
            cand = np.argsort(-probs)

        chosen = -1
        for c in cand:
            if remaining[c] > 0:
                chosen = int(c)
                break
        if chosen < 0:
            c = int(np.argmax(probs))
            if remaining[c] == 0:
                c = int(np.argmin(remaining))
            chosen = c

        remaining[chosen] -= 1
        assign[i] = chosen

    return assign


def main():
    p = argparse.ArgumentParser(description="FAISS RQKMeans+OPQ (3+2) -> indices + metrics")
    p.add_argument("--data_npy", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--rq_levels", type=int, default=3)
    p.add_argument("--rq_codebook_size", type=int, default=1024)

    p.add_argument("--uniform", action="store_true", help="Enable Sinkhorn uniform mapping on last RQ level (before OPQ/PQ)")
    p.add_argument("--sinkhorn_iters", type=int, default=30)
    p.add_argument("--sinkhorn_tau", type=float, default=None)
    p.add_argument("--sinkhorn_topk", type=int, default=32)
    p.add_argument("--sinkhorn_seed", type=int, default=42)

    p.add_argument("--opq_subquantizers", type=int, default=2)
    p.add_argument("--opq_codebook_size", type=int, default=1024)
    p.add_argument("--opq_niter", type=int, default=5)
    p.add_argument("--pq_niter", type=int, default=10)

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data_npy)
    data = np.ascontiguousarray(data.astype(np.float32))

    rq = train_rq_prefix(data, args.rq_levels, args.rq_codebook_size)

    rq_codes_packed = rq.compute_codes(data)
    rq_nbits = int(np.log2(int(args.rq_codebook_size)))
    rq_codes_before = unpack_rq_codes(rq_codes_packed, rq_nbits, args.rq_levels)

    rq_codes_after = rq_codes_before
    if args.uniform:
        # balance the last RQ level (level rq_levels)
        codebooks = get_rq_codebooks(rq)
        if not isinstance(codebooks, np.ndarray):
            raise ValueError("sinkhorn uniform mapping for rqkmeans_opq requires uniform-bit RQ codebooks")

        approx_before = rq.decode(rq_codes_packed)
        residuals_before = data - approx_before

        last_level = args.rq_levels - 1
        residuals_upto = residuals_before.copy()
        for l in range(last_level):
            residuals_upto -= codebooks[l][rq_codes_before[:, l]]

        centroids_last = codebooks[last_level]
        new_ids = sinkhorn_balance_level(
            residuals_upto,
            centroids_last,
            iters=args.sinkhorn_iters,
            tau=args.sinkhorn_tau,
            topk=args.sinkhorn_topk,
            seed=args.sinkhorn_seed + last_level,
            verbose=True,
        )
        rq_codes_after = rq_codes_before.copy()
        rq_codes_after[:, last_level] = new_ids

    # residuals for OPQ stage should be computed from final rq_codes
    approx_after = rq.decode(rq_codes_packed)
    residuals = data - approx_after
    if args.uniform:
        # recompute approx using adjusted last-level ids
        approx_after = codebooks[0][rq_codes_after[:, 0]]
        for l in range(1, args.rq_levels):
            approx_after += codebooks[l][rq_codes_after[:, l]]
        residuals = data - approx_after

    opq, pq = train_opq_pq(
        residuals,
        subquantizers=args.opq_subquantizers,
        codebook_size=args.opq_codebook_size,
        opq_niter=args.opq_niter,
        pq_niter=args.pq_niter,
    )

    opq_codes = encode_opq(opq, pq, residuals).astype(np.int32)

    codes5_before = np.hstack([rq_codes_before.astype(np.int32), opq_codes])
    codes5_after = np.hstack([rq_codes_after.astype(np.int32), opq_codes])
    if codes5_after.shape[1] != 5:
        raise ValueError(f"Expected 5 codes (3+2), got {codes5_after.shape[1]}")

    tpl = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]
    with (out_dir / "indices.jsonl").open("w", encoding="utf-8") as f:
        for row in codes5_after:
            sid = [tpl[i].format(int(row[i])) for i in range(5)]
            f.write(json.dumps({"indices": row.tolist(), "tokens": sid, "sid": sid}, ensure_ascii=False) + "\n")

    metrics = {
        "before_balance": compute_metrics(codes5_before, emb=data),
        "after_balance": compute_metrics(codes5_after, emb=data),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # codebooks are not exported by default (indices are the canonical artifact)


if __name__ == "__main__":
    main()
