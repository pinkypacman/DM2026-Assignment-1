"""ARM enhancements beyond the A/B/C ablation.

D — Rule-similarity spectral clustering
    Cluster on a precomputed Jaccard-similarity matrix between transactions'
    itemset memberships, sidestepping the Euclidean-distance assumption.

E — Iterative pattern-cluster refinement (EM-style)
    Self-supervised: vanilla K-means -> mine *contrast* itemsets per cluster
    -> derive feature weights from those contrast lifts -> re-run K-means
    initialized from the previous centroids.  Repeat until stable.

F — ARM-guided feature selection
    Drop original features that do not appear in any retained high-lift
    itemset before running K-means.  Reduces dimensionality + noise.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler

from .components import build_indicator_matrix
from .data import BINARY_NATIVE, K_DEFAULT, SEEDS, build_transactions
from .methods import (
    ARMConfig,
    evaluate,
    hungarian_align,
    run_kmeans_once,
)
from .mining import mine_itemsets, MinedItemsets


# ---- Method D: rule-similarity spectral clustering ---------------------


def jaccard_similarity_matrix(B: np.ndarray) -> np.ndarray:
    """Pairwise Jaccard similarity between rows of a binary indicator matrix.

    For rows that match no itemset (B_i = 0), Jaccard with anything is 0/0;
    we define those as 0 here so SpectralClustering can still run, and add a
    tiny diagonal so the affinity is connected.
    """
    Bb = B.astype(bool)
    intersection = Bb @ Bb.T.astype(int)            # (N, N)
    row_sums = Bb.sum(axis=1, keepdims=True)        # (N, 1)
    union = row_sums + row_sums.T - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, intersection / union, 0.0)
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float64)


def run_method_D(
    X: pd.DataFrame, y: np.ndarray,
    seeds: Iterable[int] = SEEDS, cfg: ARMConfig = ARMConfig(),
) -> pd.DataFrame:
    transactions = build_transactions(
        X, include_binary_native=cfg.include_binary_native,
        discretize_mode=cfg.discretize_mode,
    )
    mined = mine_itemsets(
        transactions, min_support=cfg.min_support, min_size=cfg.min_size,
        top_m=cfg.top_m, rank_by=cfg.rank_by, min_lift=cfg.min_lift,
    )
    if not mined.itemsets:
        # fallback: no itemsets to compute similarity, run vanilla
        return _vanilla_fallback(X, y, seeds, label="D_kernel")

    B = build_indicator_matrix(transactions, mined.itemsets)
    S = jaccard_similarity_matrix(B)

    rows = []
    for seed in seeds:
        spec = SpectralClustering(
            n_clusters=K_DEFAULT, affinity="precomputed",
            random_state=seed, assign_labels="kmeans", n_init=10,
        )
        clusters = spec.fit_predict(S)
        aligned = hungarian_align(y, clusters)
        rows.append({
            "method": "D_kernel", "seed": seed,
            **evaluate(y, aligned), "n_itemsets": len(mined.itemsets),
        })
    return pd.DataFrame(rows)


def _vanilla_fallback(
    X: pd.DataFrame, y: np.ndarray, seeds: Iterable[int], *, label: str,
) -> pd.DataFrame:
    Z = StandardScaler().fit_transform(X)
    rows = []
    for seed in seeds:
        clusters = run_kmeans_once(Z, seed)
        aligned = hungarian_align(y, clusters)
        rows.append({"method": label, "seed": seed, **evaluate(y, aligned)})
    return pd.DataFrame(rows)


# ---- Method E: iterative pattern-cluster refinement --------------------


def _contrast_pattern_weights(
    X: pd.DataFrame,
    clusters: np.ndarray,
    cfg: ARMConfig,
    binary_native: Sequence[str] = BINARY_NATIVE,
    *,
    min_global_support: float = 0.03,
    contrast_min: float = 1.5,
    top_per_cluster: int = 20,
) -> np.ndarray:
    """For each cluster, mine itemsets whose within-cluster support is much
    higher than global support (lift_in_cluster := sup_k(I) / sup(I)).
    Aggregate per-feature: w_f = max over clusters and over contrast itemsets
    containing items of f, of the cluster-lift.  Normalize to mean 1.
    """
    transactions_all = build_transactions(
        X, include_binary_native=cfg.include_binary_native,
        discretize_mode=cfg.discretize_mode,
    )

    # Global supports of single items (fast: count from transactions).
    n_total = len(transactions_all)
    item_to_global_count: dict[str, int] = {}
    for t in transactions_all:
        for it in set(t):
            item_to_global_count[it] = item_to_global_count.get(it, 0) + 1
    item_global_supp = {k: v / n_total for k, v in item_to_global_count.items()}

    cols = list(X.columns)
    col_to_items: dict[str, set[str]] = {c: set() for c in cols}
    for c in cols:
        if c in binary_native:
            col_to_items[c].add(f"{c}=1")
        else:
            for label in ["low", "medium", "high"]:
                col_to_items[c].add(f"{c}_{label}")

    weights = np.ones(len(cols), dtype=np.float64)
    cluster_ids = np.unique(clusters)
    for k in cluster_ids:
        mask = clusters == k
        n_k = int(mask.sum())
        if n_k < 10:
            continue
        # Tighter min_support relative to cluster size.
        cluster_transactions = [transactions_all[i] for i in np.where(mask)[0]]
        mined_k = mine_itemsets(
            cluster_transactions,
            min_support=max(0.10, min_global_support / max(n_k / n_total, 0.05)),
            min_size=cfg.min_size,
            top_m=top_per_cluster,
            rank_by="lift",
            min_lift=1.0,
        )
        for itemset, sup_k in zip(mined_k.itemsets, mined_k.supports):
            # Compute global support of this itemset by fast subset checks.
            sup_global = float(np.mean([itemset.issubset(set(t)) for t in transactions_all]))
            if sup_global < min_global_support:
                continue
            contrast = sup_k / sup_global
            if contrast < contrast_min:
                continue
            for j, c in enumerate(cols):
                if itemset & col_to_items[c]:
                    weights[j] = max(weights[j], contrast)

    weights /= weights.mean()
    return weights


def run_method_E(
    X: pd.DataFrame, y: np.ndarray,
    seeds: Iterable[int] = SEEDS, cfg: ARMConfig = ARMConfig(),
    *, max_iter: int = 4,
) -> pd.DataFrame:
    Z_base = StandardScaler().fit_transform(X)

    rows = []
    for seed in seeds:
        weights = np.ones(Z_base.shape[1], dtype=np.float64)
        clusters = None
        for it in range(max_iter):
            Z = Z_base * np.sqrt(weights)
            if clusters is None:
                km = KMeans(
                    n_clusters=K_DEFAULT, init="k-means++",
                    n_init=10, random_state=seed,
                )
            else:
                # Reuse previous centroids as init in the new weighted space.
                centroids = np.array(
                    [Z[clusters == k].mean(axis=0) for k in range(K_DEFAULT)
                     if (clusters == k).any()]
                )
                if centroids.shape[0] < K_DEFAULT:
                    km = KMeans(
                        n_clusters=K_DEFAULT, init="k-means++",
                        n_init=10, random_state=seed,
                    )
                else:
                    km = KMeans(
                        n_clusters=K_DEFAULT, init=centroids,
                        n_init=1, random_state=seed,
                    )
            new_clusters = km.fit_predict(Z)
            if clusters is not None and np.all(new_clusters == clusters):
                break
            clusters = new_clusters
            weights = _contrast_pattern_weights(X, clusters, cfg)

        aligned = hungarian_align(y, clusters)
        rows.append({"method": "E_iterative", "seed": seed, **evaluate(y, aligned)})
    return pd.DataFrame(rows)


# ---- Method F: ARM-guided feature selection ----------------------------


def features_in_itemsets(
    X: pd.DataFrame, mined: MinedItemsets,
    binary_native: Sequence[str] = BINARY_NATIVE,
) -> np.ndarray:
    """Boolean mask: True for features whose items appear in at least one
    retained itemset.
    """
    cols = list(X.columns)
    mask = np.zeros(len(cols), dtype=bool)
    if not mined.itemsets:
        return ~mask  # keep all if nothing mined
    items_used: set[str] = set().union(*mined.itemsets)
    for j, c in enumerate(cols):
        if c in binary_native:
            if f"{c}=1" in items_used:
                mask[j] = True
        else:
            for label in ["low", "medium", "high"]:
                if f"{c}_{label}" in items_used:
                    mask[j] = True
                    break
    return mask


def run_method_F(
    X: pd.DataFrame, y: np.ndarray,
    seeds: Iterable[int] = SEEDS, cfg: ARMConfig = ARMConfig(),
    *, also_weight: bool = False,
) -> pd.DataFrame:
    """Drop features absent from all mined itemsets.  Optionally apply
    component-C weighting on the kept features (`F+C`).
    """
    transactions = build_transactions(
        X, include_binary_native=cfg.include_binary_native,
        discretize_mode=cfg.discretize_mode,
    )
    mined = mine_itemsets(
        transactions, min_support=cfg.min_support, min_size=cfg.min_size,
        top_m=cfg.top_m, rank_by=cfg.rank_by, min_lift=cfg.min_lift,
    )
    keep_mask = features_in_itemsets(X, mined)
    n_kept = int(keep_mask.sum())
    if n_kept == 0:
        return _vanilla_fallback(X, y, seeds, label="F_selected")

    X_sel = X.iloc[:, keep_mask]
    Z_sel = StandardScaler().fit_transform(X_sel)

    if also_weight and len(mined.itemsets) > 0:
        from .components import feature_weights_from_itemsets
        w = feature_weights_from_itemsets(X_sel, mined)
        Z_sel = Z_sel * np.sqrt(w)
        label = "F+C"
    else:
        label = "F_selected"

    rows = []
    for seed in seeds:
        clusters = run_kmeans_once(Z_sel, seed)
        aligned = hungarian_align(y, clusters)
        rows.append({
            "method": label, "seed": seed, **evaluate(y, aligned),
            "n_kept": n_kept, "n_dropped": int(len(X.columns) - n_kept),
        })
    return pd.DataFrame(rows)
