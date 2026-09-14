#!/usr/bin/env python3
"""Train RQKMeans via FAISS ResidualQuantizer and directly export indices.

This matches the old `rqkmeans_faiss.py` behavior: no torch ckpt; outputs indices + metrics.

Output layout (within output_dir):
- indices.jsonl
- metrics.json
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


def train_faiss_rq(data: np.ndarray, num_levels: int, codebook_size: int | list[int]):
    if isinstance(codebook_size, list):
        if len(codebook_size) == 1:
            codebook_size = int(codebook_size[0])

    if isinstance(codebook_size, int):
        nbits = int(np.log2(codebook_size))
        rq = faiss.ResidualQuantizer(data.shape[1], num_levels, nbits)
    else:
        nbits_list = [int(np.log2(cs)) for cs in codebook_size]
        rq = faiss.ResidualQuantizer(data.shape[1], nbits_list)

    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.max_beam_size = 1

    rq.train(np.ascontiguousarray(data.astype(np.float32)))
    return rq


def encode_with_rq(rq, data: np.ndarray, codebook_size: int | list[int]):
    data = np.ascontiguousarray(data.astype(np.float32))

    if isinstance(codebook_size, list):
        if len(codebook_size) == 1:
            nbits_list = [int(np.log2(int(codebook_size[0])))] * rq.M
        else:
            nbits_list = [int(np.log2(int(cs))) for cs in codebook_size]
    else:
        nbits_list = [int(np.log2(int(codebook_size)))] * rq.M

    codes_packed = rq.compute_codes(data)

    all_same = all(nb == nbits_list[0] for nb in nbits_list)
    if all_same and (nbits_list[0] % 8 == 0):
        codes = codes_packed.astype(np.int32)
        if codes.ndim == 1:
            codes = codes.reshape(-1, 1)
        return codes

    # unpack bit-packed codes into (N,M)
    N = codes_packed.shape[0]
    packed_ints = np.zeros(N, dtype=np.int64)
    for i in range(codes_packed.shape[1]):
        packed_ints |= codes_packed[:, i].astype(np.int64) << (8 * i)

    codes = np.zeros((N, rq.M), dtype=np.int32)
    shift = 0
    for i in range(rq.M):
        nbits = int(nbits_list[i])
        mask = (1 << nbits) - 1
        codes[:, i] = (packed_ints >> shift) & mask
        shift += nbits

    return codes.astype(np.int32)


def get_rq_codebooks(rq):
    M, d = rq.M, rq.d
    nbits_vec = faiss.vector_to_array(rq.nbits)

    if np.all(nbits_vec == nbits_vec[0]):
        K = 1 << int(nbits_vec[0])
        cb_flat = faiss.vector_to_array(rq.codebooks).astype(np.float32)
        return cb_flat.reshape(M, K, d)

    cb_flat = faiss.vector_to_array(rq.codebooks).astype(np.float32)
    codebooks = []
    start = 0
    for i in range(M):
        K_i = 1 << int(nbits_vec[i])
        end = start + K_i * d
        codebooks.append(cb_flat[start:end].reshape(K_i, d))
        start = end
    return codebooks


def compute_metrics(codes: np.ndarray, emb: np.ndarray):
    # hardcode strategy: export both 4-layer and 5-layer metrics
    from codebook.metrics import summarize

    return {
        "metrics_4layer": summarize(codes[:, :4], emb=emb, exclude_last_level_cosine=False),
        "metrics_5layer": summarize(codes, emb=emb, exclude_last_level_cosine=True),
    }


def pairwise_sq_dists_batch(X, C, C_norm2=None):
    if C_norm2 is None:
        C_norm2 = np.sum(C * C, axis=1)
    X_norm2 = np.sum(X * X, axis=1, keepdims=True)
    dots = X @ C.T
    return X_norm2 + C_norm2[None, :] - 2.0 * dots


def compute_residuals_upto_level(rq, data, codes, upto_level, codebooks=None):
    if codebooks is None:
        codebooks = get_rq_codebooks(rq)

    residuals = np.ascontiguousarray(data.astype(np.float32)).copy()
    is_uniform = isinstance(codebooks, np.ndarray)

    for l in range(upto_level):
        if is_uniform:
            residuals -= codebooks[l][codes[:, l]]
        else:
            residuals -= codebooks[l][codes[:, l]]

    return residuals


def estimate_tau(residuals, centroids, sample_size=4000, percentile=90, min_tau=1e-6):
    N = residuals.shape[0]
    idx = np.random.choice(N, size=min(sample_size, N), replace=False)
    X = residuals[idx]
    Cn2 = np.sum(centroids * centroids, axis=1)
    D = pairwise_sq_dists_batch(X, centroids, Cn2)
    spread = np.percentile(D - D.min(axis=1, keepdims=True), percentile, axis=1)
    tau = float(np.median(spread) * 0.1)
    return max(tau, min_tau)


def sinkhorn_balance_level(
    residuals,
    centroids,
    capacities=None,
    *,
    iters=30,
    tau=None,
    verbose=True,
    topk=32,
    seed=42,
):
    rng = np.random.RandomState(seed)
    N = residuals.shape[0]
    K = centroids.shape[0]

    if capacities is None:
        capacities = np.full(K, N // K, dtype=np.int64)
        capacities[: (N % K)] += 1
    capacities = capacities.astype(np.int64)

    if tau is None:
        tau = estimate_tau(residuals, centroids)

    if verbose:
        print(f"  Sinkhorn: N={N} K={K} tau={tau:.5g} iters={iters}")

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

    if verbose:
        used = capacities - remaining
        print(f"    balanced: min={used.min()} max={used.max()}")

    return assign


def sinkhorn_uniform_mapping_last_level(
    rq,
    data,
    codes,
    *,
    iters=30,
    tau=None,
    verbose=True,
    topk=32,
    seed=42,
):
    codebooks = get_rq_codebooks(rq)
    N, M = codes.shape

    codes_bal = codes.copy()
    last_level = M - 1

    if verbose:
        print(f"\n=== Sinkhorn uniform mapping level {last_level + 1}/{M} (last level only) ===")

    residuals = compute_residuals_upto_level(rq, data, codes_bal, upto_level=last_level, codebooks=codebooks)

    current_cb = codebooks[last_level] if isinstance(codebooks, np.ndarray) else codebooks[last_level]
    current_K = current_cb.shape[0]

    capacities = np.full(current_K, N // current_K, dtype=np.int64)
    capacities[: (N % current_K)] += 1

    new_ids = sinkhorn_balance_level(
        residuals,
        current_cb,
        capacities=capacities,
        iters=iters,
        tau=tau,
        verbose=verbose,
        topk=topk,
        seed=seed + last_level,
    )

    codes_bal[:, last_level] = new_ids
    return codes_bal


def pack_hardcode(meta: dict[str, list], base: int, bsz: int):
    # Keep consistent with torch models (src/models/*):
    # - publish_year_bucket = 2035 - year, bucket in [0,125] else 0
    # - paid flag is 1 bit; all non-paid/unknown values map to 0
    # - index = (publish_year_bucket & 0x7F) << 1 | (paid & 0x1)
    s = np.asarray(meta["saletype"][base : base + bsz], dtype=np.int64)
    year = np.asarray(meta["publish_year"][base : base + bsz], dtype=np.int64)

    s = (s == 1).astype(np.int64)

    bucket = 2035 - year
    bucket = np.where((bucket >= 0) & (bucket <= 125), bucket, 0)

    return ((bucket.astype(np.int64) & 0x7F) << 1) | (s & 0x1)


def _normalize_sale(v):
    if v is None:
        return 0
    if isinstance(v, str):
        t = v.strip().upper()
        if t in {"PAY", "PAID", "CHARGE", "CHARGED", "付费", "收费", "TRUE", "YES", "Y"}:
            return 1
        if t.isdigit():
            try:
                return 1 if int(t) == 1 else 0
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


def _normalize_year(v):
    if v is None:
        return 0
    try:
        return int(v)
    except Exception:
        return 0


def _get_meta_field(obj: dict, keys: list[str], default=0, *, kind: str = "sale"):
    # Support both layouts:
    # 1) flat meta rows: {"saletype": ..., "publish_year": ...}
    # 2) embedding processor rows: {"meta": {...}}
    # Try top-level keys first, then meta dict.
    if not isinstance(obj, dict):
        return default
    for k in keys:
        if k in obj:
            v = obj.get(k)
            break
    else:
        inner = obj.get("meta")
        v = None
        if isinstance(inner, dict):
            for k in keys:
                if k in inner:
                    v = inner.get(k)
                    break
    if v is None:
        return default
    if kind == "sale":
        return _normalize_sale(v)
    if kind == "year":
        return _normalize_year(v)
    return v if v is not None else default


def load_meta(meta_jsonl: str):
    meta = {"saletype": [], "publish_year": []}
    with open(meta_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            meta["saletype"].append(_get_meta_field(obj, ["saletype", "pay_type", "paytype", "sale_type"], 0, kind="sale"))
            meta["publish_year"].append(_get_meta_field(obj, ["publish_year"], 0, kind="year"))
    return meta


def main():
    p = argparse.ArgumentParser(description="FAISS RQKMeans (4+1) -> indices + metrics")
    p.add_argument("--data_npy", required=True)
    p.add_argument("--meta_jsonl", default=None)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--num_levels", type=int, default=4)
    p.add_argument("--codebook_size", type=int, nargs="+", default=[1024])

    p.add_argument("--uniform", action="store_true", help="Enable Sinkhorn uniform mapping (last level only)")
    p.add_argument("--sinkhorn_iters", type=int, default=30)
    p.add_argument("--sinkhorn_tau", type=float, default=None)
    p.add_argument("--sinkhorn_topk", type=int, default=32)
    p.add_argument("--sinkhorn_seed", type=int, default=42)

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data_npy)
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    rq = train_faiss_rq(data, num_levels=args.num_levels, codebook_size=args.codebook_size)
    codes4_raw = encode_with_rq(rq, data, codebook_size=args.codebook_size)

    if codes4_raw.shape[1] != args.num_levels:
        raise ValueError(f"Expected {args.num_levels} levels, got {codes4_raw.shape[1]}")

    codes4_final = codes4_raw
    if args.uniform:
        codes4_final = sinkhorn_uniform_mapping_last_level(
            rq,
            data,
            codes4_raw,
            iters=args.sinkhorn_iters,
            tau=args.sinkhorn_tau,
            topk=args.sinkhorn_topk,
            seed=args.sinkhorn_seed,
            verbose=True,
        )

    meta = None
    if args.meta_jsonl:
        meta = load_meta(args.meta_jsonl)
        if len(meta["saletype"]) != data.shape[0]:
            raise ValueError("meta_jsonl rows != embeddings rows")

    # append 5th hardcode
    if meta is None:
        hard = np.zeros((codes4_raw.shape[0],), dtype=np.int32)
    else:
        hard = pack_hardcode(meta, 0, codes4_raw.shape[0]).astype(np.int32)

    codes5_before = np.concatenate([codes4_raw.astype(np.int32), hard.reshape(-1, 1)], axis=1)
    codes5_after = np.concatenate([codes4_final.astype(np.int32), hard.reshape(-1, 1)], axis=1)

    tpl = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]
    meta_rows = None
    if args.meta_jsonl:
        # Re-read to keep full original rows for Format-B output.
        meta_rows = []
        with open(args.meta_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                meta_rows.append(json.loads(line))
        if len(meta_rows) != codes5_after.shape[0]:
            raise ValueError("meta_jsonl rows != embeddings rows when writing outputs")

    def _get_first_present(m: dict, keys: list[str]):
        if not isinstance(m, dict):
            return None
        for k in keys:
            if k in m:
                return m.get(k)
        inner = m.get("meta")
        if isinstance(inner, dict):
            for k in keys:
                if k in inner:
                    return inner.get(k)
        return None

    with (out_dir / "indices.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(codes5_after):
            sid = [tpl[j].format(int(row[j])) for j in range(5)]
            rec = {"indices": row.tolist(), "tokens": sid, "sid": sid}
            if meta_rows is not None:
                meta_row = meta_rows[i]
                if isinstance(meta_row, dict):
                    meta_row = dict(meta_row)
                    meta_row.update(rec)
                    rec = meta_row

            # Normalize and write top-level saletype / publish_year (do not synthesize meta_json)
            s = _get_first_present(rec, ["saletype", "sale_type", "pay_type", "paytype"])
            if s is not None:
                rec["saletype"] = str(_normalize_sale(s))
            y = _get_first_present(rec, ["publish_year"])
            rec["publish_year"] = "未知" if y is None else str(y)

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics = {
        "before_balance": compute_metrics(codes5_before, emb=data),
        "after_balance": compute_metrics(codes5_after, emb=data),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # codebooks are not exported by default (indices are the canonical artifact)


if __name__ == "__main__":
    main()
