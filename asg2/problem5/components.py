"""Components A (augmented features), B (seeded init), C (feature weighting)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .data import BINARY_NATIVE
from .mining import MinedItemsets, supporting_indices


# ---- Component A: ARM-augmented features ------------------------------


def build_indicator_matrix(
    transactions: list[list[str]], itemsets: list[frozenset]
) -> np.ndarray:
    n, m = len(transactions), len(itemsets)
    B = np.zeros((n, m), dtype=np.float64)
    rows = [set(t) for t in transactions]
    for j, itemset in enumerate(itemsets):
        for i, rowset in enumerate(rows):
            if itemset.issubset(rowset):
                B[i, j] = 1.0
    return B


def build_augmented_space(
    Z: np.ndarray,
    transactions: list[list[str]],
    itemsets: list[frozenset],
    *,
    alpha: float = 0.3,
) -> np.ndarray:
    if not itemsets:
        return Z
    B = build_indicator_matrix(transactions, itemsets)
    mu = B.mean(axis=0, keepdims=True)
    sigma = B.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    B_std = (B - mu) / sigma
    return np.concatenate([Z, alpha * B_std], axis=1)


# ---- Component B: ARM-seeded init -------------------------------------


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def select_seed_itemsets(
    mined: MinedItemsets, K: int, tau: float = 0.5
) -> list[np.ndarray]:
    transactions = mined.transactions
    candidates = list(zip(mined.itemsets, mined.supports))
    candidates.sort(key=lambda x: -x[1])

    for cur_tau in [tau, 0.7, 0.9]:
        chosen_supports: list[set] = []
        chosen_arrays: list[np.ndarray] = []
        for itemset, _ in candidates:
            S = supporting_indices(transactions, itemset)
            if len(S) == 0:
                continue
            S_set = set(S.tolist())
            if all(_jaccard(S_set, prev) <= cur_tau for prev in chosen_supports):
                chosen_supports.append(S_set)
                chosen_arrays.append(S)
                if len(chosen_arrays) == K:
                    return chosen_arrays
        if len(chosen_arrays) == K:
            return chosen_arrays
    return chosen_arrays


def seeded_centroids(
    Z_aug: np.ndarray, seed_index_sets: list[np.ndarray], K: int, rng_seed: int,
) -> np.ndarray:
    centroids = [Z_aug[S].mean(axis=0) for S in seed_index_sets]
    if len(centroids) < K:
        rng = np.random.default_rng(rng_seed)
        all_idx = np.arange(Z_aug.shape[0])
        used = set(np.concatenate(seed_index_sets).tolist()) if seed_index_sets else set()
        avail = np.array([i for i in all_idx if i not in used])
        rng.shuffle(avail)
        for i in avail:
            centroids.append(Z_aug[i])
            if len(centroids) == K:
                break
    return np.array(centroids[:K])


# ---- Component C: ARM-derived feature weighting -----------------------


def feature_weights_from_itemsets(
    X: pd.DataFrame, mined: MinedItemsets,
    binary_native: Sequence[str] = BINARY_NATIVE,
) -> np.ndarray:
    """For each column of X, score = max(lift) over itemsets containing any of
    its items.  Normalize so mean(weights) = 1.
    """
    cols = list(X.columns)
    weights = np.ones(len(cols), dtype=np.float64)
    if len(mined.itemsets) == 0:
        return weights

    col_to_items: dict[str, set[str]] = {c: set() for c in cols}
    for c in cols:
        if c in binary_native:
            col_to_items[c].add(f"{c}=1")
        else:
            for label in ["low", "medium", "high"]:
                col_to_items[c].add(f"{c}_{label}")

    for j, c in enumerate(cols):
        items = col_to_items[c]
        max_lift = 1.0
        for itemset, lift in zip(mined.itemsets, mined.lifts):
            if itemset & items:
                max_lift = max(max_lift, float(lift))
        weights[j] = max_lift

    weights /= weights.mean()
    return weights
