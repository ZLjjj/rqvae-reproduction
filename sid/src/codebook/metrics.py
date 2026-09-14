import collections
from typing import Any

import numpy as np


def _as_int_matrix(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes)
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2D (N,L), got shape={codes.shape}")
    if not np.issubdtype(codes.dtype, np.integer):
        codes = codes.astype(np.int64)
    return codes


def _default_codebook_sizes(codes: np.ndarray, *, min_first_layers: int = 4, default_first_size: int = 1024) -> list[int]:
    # Mirrors old calc_metrics_diff.py heuristic: at least 1024 for first layers.
    N, L = codes.shape
    sizes: list[int] = []
    for l in range(L):
        col = codes[:, l]
        valid = col[col >= 0]
        m = int(valid.max()) if valid.size else 0
        base = m + 1
        if l < min_first_layers:
            base = max(base, default_first_size)
        sizes.append(int(base))
    return sizes


def _extend_codebook_sizes(codebook_sizes: list[int], codes: np.ndarray) -> list[int]:
    sizes = list(codebook_sizes)
    N, L = codes.shape
    if len(sizes) >= L:
        return sizes[:L]
    for l in range(len(sizes), L):
        col = codes[:, l]
        valid = col[col >= 0]
        m = int(valid.max()) if valid.size else 0
        sizes.append(int(m + 1))
    return sizes


def compute_core_metrics(codes: np.ndarray, *, codebook_sizes: list[int] | None = None) -> dict[str, Any]:
    """Compute collision_rate, ICR and CUR exactly like old calc_metrics_diff.py.

    - collision_rate = (N - unique_full_paths) / N
    - icr = unique_full_paths / N
    - prefix_cur_list[l] = used_unique_prefixes_len(l+1) / N (prefix coverage)
    - capacity_cur_list[l] = used_unique_prefixes_len(l+1) / product(codebook_sizes[:l+1])
    """

    codes = _as_int_matrix(codes)
    N, L = codes.shape

    if codebook_sizes is None:
        codebook_sizes = _default_codebook_sizes(codes)
    codebook_sizes = _extend_codebook_sizes(codebook_sizes, codes)

    rows = [tuple(r) for r in codes.tolist()]
    unique_paths = len(set(rows))
    collision_rate = (N - unique_paths) / N if N > 0 else 0.0
    icr = unique_paths / N if N > 0 else 0.0

    prefix_cur_list: list[float] = []
    capacity_cur_list: list[float] = []
    prefix_capacity = 1
    for l in range(L):
        prefix_capacity *= int(codebook_sizes[l])
        prefix_rows = [tuple(r[: l + 1]) for r in rows]
        used_prefix = len(set(prefix_rows))
        coverage = used_prefix / N if N > 0 else 0.0
        capacity_cur = used_prefix / prefix_capacity if prefix_capacity > 0 else 0.0
        prefix_cur_list.append(float(coverage))
        capacity_cur_list.append(float(capacity_cur))

    total_cur = float(prefix_cur_list[-1]) if prefix_cur_list else 0.0

    return {
        "N": int(N),
        "L": int(L),
        "collision_rate": float(collision_rate),
        "unique_paths": int(unique_paths),
        "icr": float(icr),
        "total_cur": float(total_cur),
        "prefix_cur_list": prefix_cur_list,
        "capacity_cur_list": capacity_cur_list,
        "codebook_sizes": [int(x) for x in codebook_sizes],
    }


def compute_prefix_bucket_stats(codes: np.ndarray) -> dict[str, Any]:
    """Bucket size stats per prefix length.

    For each level l (1..L): treat unique prefixes of length l as buckets.
    Stats are computed over bucket sizes.
    """

    codes = _as_int_matrix(codes)
    N, L = codes.shape

    rows = [tuple(r) for r in codes.tolist()]

    per_level: list[dict[str, Any]] = []
    for l in range(1, L + 1):
        buckets = collections.Counter(tuple(r[:l]) for r in rows)
        sizes = np.fromiter(buckets.values(), dtype=np.int64)
        if sizes.size == 0:
            stats = {
                "level": int(l),
                "bucket_count": 0,
                "mean": 0.0,
                "median": 0.0,
                "p25": 0.0,
                "p75": 0.0,
                "max": 0,
            }
        else:
            stats = {
                "level": int(l),
                "bucket_count": int(len(buckets)),
                "mean": float(np.mean(sizes)),
                "median": float(np.median(sizes)),
                "p25": float(np.percentile(sizes, 25)),
                "p75": float(np.percentile(sizes, 75)),
                "max": int(np.max(sizes)),
            }
        per_level.append(stats)

    return {"bucket_size": per_level}


def _cosine_matrix(X: np.ndarray) -> np.ndarray:
    # X: (n,d)
    X = X.astype(np.float32, copy=False)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    Xn = X / nrm
    return Xn @ Xn.T


def compute_prefix_bucket_cosine(
    emb: np.ndarray,
    codes: np.ndarray,
    *,
    max_level: int | None = None,
    exclude_last_level: bool = False,
    max_items_per_bucket: int | None = 512,
) -> dict[str, Any]:
    """Within-bucket cosine similarity stats per prefix level.

    - weighted_sim: weighted by number of pairs in each bucket (n*(n-1)/2)
    - unweighted_sim: average of per-bucket mean similarities

    Notes:
    - For very large buckets, optionally subsample up to max_items_per_bucket to control cost.
    - If exclude_last_level=True, we compute up to L-1 (used for hardcode last layer).
    """

    emb = np.asarray(emb)
    codes = _as_int_matrix(codes)
    N, L = codes.shape

    if emb.shape[0] != N:
        raise ValueError(f"emb rows ({emb.shape[0]}) != codes rows ({N})")

    last = L
    if exclude_last_level:
        last = max(0, L - 1)

    if max_level is not None:
        last = min(last, int(max_level))

    rows = [tuple(r) for r in codes.tolist()]

    out: list[dict[str, Any]] = []
    for l in range(1, last + 1):
        # group indices by prefix
        groups: dict[tuple, list[int]] = {}
        for i, r in enumerate(rows):
            p = tuple(r[:l])
            groups.setdefault(p, []).append(i)

        per_bucket_means: list[float] = []
        weighted_sum = 0.0
        weighted_pairs = 0

        for idxs in groups.values():
            n = len(idxs)
            if n < 2:
                continue

            if max_items_per_bucket is not None and n > max_items_per_bucket:
                # deterministic-ish slice (avoid rng dependency for now)
                idxs = idxs[:max_items_per_bucket]
                n = len(idxs)
                if n < 2:
                    continue

            X = emb[idxs]
            S = _cosine_matrix(X)
            # take upper triangle (excluding diagonal)
            triu = np.triu_indices(n, k=1)
            vals = S[triu]
            mean_sim = float(np.mean(vals)) if vals.size else 0.0

            per_bucket_means.append(mean_sim)
            pairs = n * (n - 1) // 2
            weighted_sum += mean_sim * pairs
            weighted_pairs += pairs

        if weighted_pairs > 0:
            weighted_sim = float(weighted_sum / weighted_pairs)
        else:
            weighted_sim = 0.0

        if per_bucket_means:
            unweighted_sim = float(np.mean(per_bucket_means))
        else:
            unweighted_sim = 0.0

        out.append({"level": int(l), "weighted_sim": weighted_sim, "unweighted_sim": unweighted_sim})

    return {"bucket_cosine": out}


def summarize(
    codes: np.ndarray,
    *,
    codebook_sizes: list[int] | None = None,
    emb: np.ndarray | None = None,
    exclude_last_level_cosine: bool = False,
) -> dict[str, Any]:
    core = compute_core_metrics(codes, codebook_sizes=codebook_sizes)
    buckets = compute_prefix_bucket_stats(codes)

    out: dict[str, Any] = {}
    out.update(core)
    out.update(buckets)

    if emb is not None:
        out.update(
            compute_prefix_bucket_cosine(
                emb,
                codes,
                exclude_last_level=exclude_last_level_cosine,
            )
        )

    return out
