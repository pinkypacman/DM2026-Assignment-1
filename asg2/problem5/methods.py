"""Configurable methods that compose components A, B, C plus standard runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from .components import (
    build_augmented_space,
    feature_weights_from_itemsets,
    seeded_centroids,
    select_seed_itemsets,
)
from .data import K_DEFAULT, SEEDS, build_transactions
from .mining import mine_itemsets


METHODS = ["vanilla", "A_only", "B_only", "A+B", "C_only", "B+C", "A+B+C"]
PROPOSED_METHOD = "F+C"  # strongest method overall (in `run_method_F` with also_weight=True)


@dataclass
class ARMConfig:
    min_support: float = 0.05
    min_size: int = 2
    top_m: int | None = 30
    alpha: float = 0.3
    tau: float = 0.5
    include_binary_native: bool = True
    discretize_mode: str = "width_3_4_3"
    rank_by: str = "lift"
    min_lift: float = 1.2


@dataclass
class PreparedSpace:
    Z: np.ndarray
    Z_aug: np.ndarray
    seed_sets: list[np.ndarray]
    weights: np.ndarray
    n_itemsets: int
    n_aug_dims: int


def prepare_space(X: pd.DataFrame, cfg: ARMConfig, *,
                  augment: bool, seed_init: bool, weight_features: bool = False
                  ) -> PreparedSpace:
    Z = StandardScaler().fit_transform(X)
    transactions = build_transactions(
        X,
        include_binary_native=cfg.include_binary_native,
        discretize_mode=cfg.discretize_mode,
    )
    mined = mine_itemsets(
        transactions,
        min_support=cfg.min_support,
        min_size=cfg.min_size,
        top_m=cfg.top_m,
        rank_by=cfg.rank_by,
        min_lift=cfg.min_lift,
    )

    weights = np.ones(Z.shape[1], dtype=np.float64)
    if weight_features and len(mined.itemsets) > 0:
        weights = feature_weights_from_itemsets(X, mined)
        Z = Z * np.sqrt(weights)

    if augment and len(mined.itemsets) > 0:
        Z_aug = build_augmented_space(Z, transactions, mined.itemsets, alpha=cfg.alpha)
    else:
        Z_aug = Z

    seed_sets = []
    if seed_init and len(mined.itemsets) > 0:
        seed_sets = select_seed_itemsets(mined, K=K_DEFAULT, tau=cfg.tau)

    return PreparedSpace(
        Z=Z, Z_aug=Z_aug, seed_sets=seed_sets, weights=weights,
        n_itemsets=len(mined.itemsets), n_aug_dims=Z_aug.shape[1],
    )


def hungarian_align(y_true: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    classes = np.unique(y_true)
    cluster_ids = np.unique(clusters)
    cm = np.zeros((len(cluster_ids), len(classes)), dtype=int)
    for i, k in enumerate(cluster_ids):
        for j, c in enumerate(classes):
            cm[i, j] = int(np.sum((clusters == k) & (y_true == c)))
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {cluster_ids[r]: classes[c] for r, c in zip(row_ind, col_ind)}
    return np.array([mapping.get(c, c) for c in clusters])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def run_kmeans_once(
    Z: np.ndarray, seed: int, *, K: int = K_DEFAULT,
    init: str | np.ndarray = "k-means++", n_init: int = 10,
) -> np.ndarray:
    if isinstance(init, np.ndarray):
        n_init = 1
    km = KMeans(n_clusters=K, init=init, n_init=n_init, random_state=seed)
    return km.fit_predict(Z)


def run_method(
    X: pd.DataFrame, y: np.ndarray, method: str,
    seeds: Iterable[int] = SEEDS, cfg: ARMConfig = ARMConfig(),
) -> pd.DataFrame:
    augment = method in {"A_only", "A+B", "A+B+C"}
    seed_init = method in {"B_only", "A+B", "B+C", "A+B+C"}
    weight_features = method in {"C_only", "B+C", "A+B+C"}
    space = prepare_space(
        X, cfg,
        augment=augment, seed_init=seed_init, weight_features=weight_features,
    )

    rows = []
    for seed in seeds:
        if seed_init:
            init = seeded_centroids(space.Z_aug, space.seed_sets, K_DEFAULT, seed)
        else:
            init = "k-means++"
        clusters = run_kmeans_once(space.Z_aug, seed, init=init)
        aligned = hungarian_align(y, clusters)
        rows.append({"method": method, "seed": seed, **evaluate(y, aligned)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, group="method") -> pd.DataFrame:
    metrics = ["accuracy", "precision", "recall", "f1"]
    agg = df.groupby(group, sort=False)[metrics].agg(["mean", "std"])
    out = pd.DataFrame(index=agg.index)
    for m in metrics:
        out[m] = (
            agg[(m, "mean")].map("{:.4f}".format)
            + " ± "
            + agg[(m, "std")].map("{:.4f}".format)
        )
    return out
