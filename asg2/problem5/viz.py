"""Visualizations and qualitative analysis for Problem 5."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from .components import (
    build_augmented_space,
    feature_weights_from_itemsets,
    seeded_centroids,
    select_seed_itemsets,
)
from .data import K_DEFAULT, build_transactions
from .methods import ARMConfig, hungarian_align, run_kmeans_once
from .mining import mine_itemsets, supporting_indices


def pca_triptych(X, y, cfg=ARMConfig(), seed=42, save_path: Path | None = None):
    """Three-panel scatter: ground-truth, vanilla clusters, proposed (C_only) clusters."""
    Z = StandardScaler().fit_transform(X)
    pca2 = PCA(n_components=2, random_state=seed).fit_transform(Z)

    transactions = build_transactions(X, include_binary_native=cfg.include_binary_native)
    mined = mine_itemsets(
        transactions, min_support=cfg.min_support, min_size=cfg.min_size,
        top_m=cfg.top_m, rank_by=cfg.rank_by, min_lift=cfg.min_lift,
    )
    weights = feature_weights_from_itemsets(X, mined)
    Z_w = Z * np.sqrt(weights)

    vanilla = run_kmeans_once(Z, seed)
    proposed = run_kmeans_once(Z_w, seed)
    vanilla_aligned = hungarian_align(y, vanilla)
    proposed_aligned = hungarian_align(y, proposed)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharex=True, sharey=True)
    panels = [
        (axes[0], y, "Ground truth (price_range)"),
        (axes[1], vanilla_aligned, f"Vanilla K-means (seed={seed})"),
        (axes[2], proposed_aligned, f"Proposed C_only K-means (seed={seed})"),
    ]
    cmap = plt.get_cmap("tab10")
    for ax, labels, title in panels:
        for i, lab in enumerate(sorted(np.unique(labels))):
            mask = labels == lab
            ax.scatter(pca2[mask, 0], pca2[mask, 1], s=10, color=cmap(i),
                       alpha=0.7, label=f"class {int(lab)}")
        ax.set_title(title)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8, frameon=True)

    fig.suptitle("Q5 PCA-2D triptych: ground truth vs vanilla vs proposed")
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def confusion_pair(X, y, cfg=ARMConfig(), seed=42, save_path: Path | None = None):
    Z = StandardScaler().fit_transform(X)
    transactions = build_transactions(X, include_binary_native=cfg.include_binary_native)
    mined = mine_itemsets(
        transactions, min_support=cfg.min_support, min_size=cfg.min_size,
        top_m=cfg.top_m, rank_by=cfg.rank_by, min_lift=cfg.min_lift,
    )
    weights = feature_weights_from_itemsets(X, mined)
    Z_w = Z * np.sqrt(weights)

    vanilla = hungarian_align(y, run_kmeans_once(Z, seed))
    proposed = hungarian_align(y, run_kmeans_once(Z_w, seed))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, pred, title in [
        (axes[0], vanilla, "Vanilla K-means"),
        (axes[1], proposed, "Proposed C_only"),
    ]:
        cm = confusion_matrix(y, pred, labels=sorted(np.unique(y)))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(cm.shape[1])); ax.set_yticks(range(cm.shape[0]))
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{title} (seed={seed})")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Q5 confusion matrices (Hungarian-aligned)")
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def top_itemset_purity(X, y, cfg=ARMConfig(), top: int = 10) -> pd.DataFrame:
    transactions = build_transactions(X, include_binary_native=cfg.include_binary_native)
    mined = mine_itemsets(
        transactions, min_support=cfg.min_support, min_size=cfg.min_size,
        top_m=cfg.top_m, rank_by=cfg.rank_by, min_lift=cfg.min_lift,
    )
    rows = []
    for itemset, support, lift in zip(
        mined.itemsets[:top], mined.supports[:top], mined.lifts[:top]
    ):
        idx = supporting_indices(transactions, itemset)
        if len(idx) == 0:
            continue
        labels = y[idx]
        dist = pd.Series(labels).value_counts(normalize=True).sort_index()
        rows.append({
            "itemset": ", ".join(sorted(itemset)),
            "support": float(support),
            "lift": float(lift),
            "n": len(idx),
            **{f"class_{int(c)}": float(dist.get(c, 0.0))
               for c in sorted(np.unique(y))},
            "purity": float(dist.max()),
            "majority_class": int(dist.idxmax()),
        })
    return pd.DataFrame(rows)


def feature_weight_bar(X, cfg=ARMConfig(), save_path: Path | None = None):
    """Bar chart of Component-C feature weights — qualitative inspection of which
    features participate in high-lift patterns and get amplified."""
    transactions = build_transactions(X, include_binary_native=cfg.include_binary_native)
    mined = mine_itemsets(
        transactions, min_support=cfg.min_support, min_size=cfg.min_size,
        top_m=cfg.top_m, rank_by=cfg.rank_by, min_lift=cfg.min_lift,
    )
    w = feature_weights_from_itemsets(X, mined)
    order = np.argsort(-w)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(range(len(w)), w[order], color="#1f77b4")
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.6, label="mean = 1")
    ax.set_xticks(range(len(w)))
    ax.set_xticklabels([X.columns[i] for i in order], rotation=70, ha="right")
    ax.set_ylabel("Component-C weight (max lift over containing itemsets)")
    ax.set_title("Per-feature ARM-derived weights (sorted)")
    ax.legend()
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
