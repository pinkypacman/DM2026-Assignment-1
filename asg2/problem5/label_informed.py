"""Label-informed K-means via Class-Conditional Rule Evidence (CCRE).

This module implements a label-informed enhancement of K-means: association
rules are mined with `price_range` as the consequent, and each sample is
turned into a 4-dim soft class-membership feature derived from the rules
that fire on it. K-means then runs on the concatenation of those features
with the standardized originals.

Differentiations from the obvious recipe (rules with class consequent +
weighted-sum scoring + concatenate):

  1. Discretization is **4-bin quantile** (`pd.qcut`, q=4), not width-based
     3:4:3 — bins are dataset-driven rather than range-driven.
  2. Per-rule contribution is `log(1 + lift)` instead of `confidence × lift`
     — bounded, dampens extreme-lift outliers.
  3. The class-rule pool is **top-K per class** (balanced) rather than a
     global top-N — minority classes can't be drowned out.
  4. The 4 raw evidence sums go through a **softmax then standardize**
     pipeline — softmax bounds them into per-row class probabilities, and
     the subsequent z-score restores unit variance so K-means actually
     sees them alongside the 20 standardized originals (the reference's
     direct-standardize-of-weighted-sum collapses both steps into one).

`run_label_informed` returns the same per-seed metric DataFrame shape used
elsewhere in `problem5/` so it slots into `summarize` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from .data import K_DEFAULT, SEEDS


def hungarian_align(y_true: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Map cluster ids to true class labels via the Hungarian algorithm."""
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


@dataclass
class CCREConfig:
    n_bins: int = 4                  # quantile bins per continuous feature
    min_support: float = 0.05
    min_confidence: float = 0.4
    min_lift: float = 1.0
    top_per_class: int = 60          # top-K class-consequent rules per class
    softmax_temperature: float = 1.0


def _discretize_quantile(series: pd.Series, n_bins: int) -> pd.Series:
    """4-bin quantile discretization. Binary cols stay binary."""
    if series.nunique() <= 2:
        return series.astype(int).astype(str)
    labels = [f"q{i+1}" for i in range(n_bins)]
    return pd.qcut(
        series, q=n_bins, labels=labels, duplicates="drop"
    ).astype("object")


def _build_class_transactions(
    X: pd.DataFrame, y: np.ndarray, n_bins: int
) -> tuple[list[list[str]], list[list[str]], list[str]]:
    """Build transactions that include a `price_range_X` item.

    Returns (transactions_with_class, transactions_without_class, feature_cols).
    The "without" form is used at scoring time so rule firing checks only
    consider feature antecedents, never the label.
    """
    feat_cols = list(X.columns)
    disc = pd.DataFrame(
        {c: _discretize_quantile(X[c], n_bins) for c in feat_cols},
        index=X.index,
    )
    with_class: list[list[str]] = []
    without_class: list[list[str]] = []
    for i, (_, row) in enumerate(disc.iterrows()):
        items = [f"{c}_{row[c]}" for c in feat_cols]
        without_class.append(items)
        with_class.append(items + [f"price_range_{int(y[i])}"])
    return with_class, without_class, feat_cols


def _mine_class_rules(
    transactions: list[list[str]], cfg: CCREConfig, n_classes: int
) -> pd.DataFrame:
    """Mine rules with `price_range_X` as a single-item consequent.

    Keeps `top_per_class` rules per class, ranked by lift then confidence
    (so each class gets balanced representation in the evidence pool).
    """
    te = TransactionEncoder()
    onehot = pd.DataFrame(
        te.fit(transactions).transform(transactions), columns=te.columns_
    )
    freq = fpgrowth(onehot, min_support=cfg.min_support, use_colnames=True)
    if freq.empty:
        return freq

    rules = association_rules(
        freq, metric="confidence", min_threshold=cfg.min_confidence
    )
    if rules.empty:
        return rules

    class_items = {f"price_range_{c}" for c in range(n_classes)}

    def _is_single_class_consequent(cons: frozenset) -> bool:
        return len(cons) == 1 and next(iter(cons)) in class_items

    def _antecedent_label_free(ante: frozenset) -> bool:
        return not any(it in class_items for it in ante)

    rules = rules[
        rules["consequents"].apply(_is_single_class_consequent)
        & rules["antecedents"].apply(_antecedent_label_free)
        & (rules["lift"] >= cfg.min_lift)
    ].copy()

    rules["class_label"] = rules["consequents"].apply(
        lambda s: int(next(iter(s)).rsplit("_", 1)[-1])
    )

    parts = []
    for c in range(n_classes):
        sub = rules[rules["class_label"] == c]
        sub = sub.sort_values(
            ["lift", "confidence", "support"], ascending=False
        ).head(cfg.top_per_class)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else rules


def _evidence_raw(
    transactions_no_class: list[list[str]],
    rules: pd.DataFrame,
    n_classes: int,
) -> np.ndarray:
    """Per-sample, per-class raw evidence: Σ log(1 + lift_r) over firing rules."""
    raw = np.zeros((len(transactions_no_class), n_classes), dtype=np.float64)
    item_sets = [set(t) for t in transactions_no_class]
    if not rules.empty:
        for ante, cls, lift in zip(
            rules["antecedents"], rules["class_label"], rules["lift"]
        ):
            ante_set = set(ante)
            score = float(np.log1p(float(lift)))
            for i, items in enumerate(item_sets):
                if ante_set.issubset(items):
                    raw[i, int(cls)] += score
    return raw


def _softmax(raw: np.ndarray, temperature: float) -> np.ndarray:
    scaled = raw / max(temperature, 1e-12)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    denom = exp.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return exp / denom


def _evidence_features(
    transactions_no_class: list[list[str]],
    rules: pd.DataFrame,
    n_classes: int,
    temperature: float,
) -> np.ndarray:
    """Convenience wrapper used by the main CCRE path: raw → softmax."""
    return _softmax(
        _evidence_raw(transactions_no_class, rules, n_classes), temperature
    )


def _build_feature_matrix(
    X: pd.DataFrame, y: np.ndarray, cfg: CCREConfig
) -> tuple[np.ndarray, pd.DataFrame]:
    """Z_std (20-dim) ⊕ standardized softmax evidence (4-dim) → 24-dim feature matrix.

    The softmax probabilities have variance much smaller than 1 (their range
    is bounded by [0, 1]), so they need to be re-standardized before being
    concatenated with the 20 unit-variance feature columns; otherwise
    K-means' Euclidean distance is dominated by the original features and
    the evidence channel goes ignored.
    """
    n_classes = int(np.unique(y).size)
    with_cls, without_cls, _ = _build_class_transactions(X, y, cfg.n_bins)
    rules = _mine_class_rules(with_cls, cfg, n_classes)
    P = _evidence_features(without_cls, rules, n_classes, cfg.softmax_temperature)
    P_std = StandardScaler().fit_transform(P)
    Z = StandardScaler().fit_transform(X)
    return np.hstack([Z, P_std]), rules


def run_label_informed(
    X: pd.DataFrame,
    y: np.ndarray,
    seeds: Iterable[int] = SEEDS,
    cfg: CCREConfig = CCREConfig(),
    K: int = K_DEFAULT,
) -> pd.DataFrame:
    """Run vanilla K-means and CCRE-augmented K-means across seeds."""
    Z = StandardScaler().fit_transform(X)
    Z_aug, _ = _build_feature_matrix(X, y, cfg)

    rows = []
    for seed in seeds:
        for method, mat in [("vanilla", Z), ("CCRE (proposed)", Z_aug)]:
            km = KMeans(n_clusters=K, random_state=seed, n_init=10)
            clusters = km.fit_predict(mat)
            aligned = hungarian_align(y, clusters)
            rows.append(
                {"method": method, "seed": seed, **evaluate(y, aligned)}
            )
    return pd.DataFrame(rows)


def summarize_label_informed(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "precision", "recall", "f1"]
    agg = df.groupby("method", sort=False)[metrics].agg(["mean", "std"])
    out = pd.DataFrame(index=agg.index)
    for m in metrics:
        out[m] = (
            agg[(m, "mean")].map("{:.4f}".format)
            + " ± "
            + agg[(m, "std")].map("{:.4f}".format)
        )
    return out


def top_class_rules_table(
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: CCREConfig = CCREConfig(),
    top_per_class: int = 5,
) -> pd.DataFrame:
    """Show the highest-lift class-consequent rules per class for interpretation."""
    n_classes = int(np.unique(y).size)
    with_cls, _, _ = _build_class_transactions(X, y, cfg.n_bins)
    rules = _mine_class_rules(with_cls, cfg, n_classes)
    if rules.empty:
        return pd.DataFrame()
    parts = []
    for c in range(n_classes):
        sub = rules[rules["class_label"] == c].sort_values(
            ["lift", "confidence", "support"], ascending=False
        ).head(top_per_class)
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    out = out[["class_label", "antecedents", "support", "confidence", "lift"]]
    out["antecedents"] = out["antecedents"].apply(lambda s: sorted(s))
    return out.reset_index(drop=True)


def pca_triptych_label_informed(
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: CCREConfig = CCREConfig(),
    seed: int = 42,
    save_path: Path | None = None,
):
    """Three-panel PCA-2D scatter: ground truth, vanilla, CCRE-augmented."""
    from sklearn.decomposition import PCA

    Z = StandardScaler().fit_transform(X)
    Z_aug, _ = _build_feature_matrix(X, y, cfg)
    pca2 = PCA(n_components=2, random_state=seed).fit_transform(Z)

    vanilla = hungarian_align(
        y, KMeans(n_clusters=K_DEFAULT, random_state=seed, n_init=10).fit_predict(Z)
    )
    proposed = hungarian_align(
        y, KMeans(n_clusters=K_DEFAULT, random_state=seed, n_init=10).fit_predict(Z_aug)
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharex=True, sharey=True)
    panels = [
        (axes[0], y, "Ground truth (price_range)"),
        (axes[1], vanilla, f"Vanilla K-means (seed={seed})"),
        (axes[2], proposed, f"CCRE-augmented K-means (seed={seed})"),
    ]
    cmap = plt.get_cmap("tab10")
    for ax, labels, title in panels:
        for i, lab in enumerate(sorted(np.unique(labels))):
            mask = labels == lab
            ax.scatter(
                pca2[mask, 0], pca2[mask, 1], s=10, color=cmap(i),
                alpha=0.7, label=f"class {int(lab)}",
            )
        ax.set_title(title)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle("Q5 PCA-2D triptych: ground truth vs vanilla vs CCRE")
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def confusion_pair_label_informed(
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: CCREConfig = CCREConfig(),
    seed: int = 42,
    save_path: Path | None = None,
):
    """Confusion matrices: vanilla vs CCRE, both Hungarian-aligned."""
    Z = StandardScaler().fit_transform(X)
    Z_aug, _ = _build_feature_matrix(X, y, cfg)
    vanilla = hungarian_align(
        y, KMeans(n_clusters=K_DEFAULT, random_state=seed, n_init=10).fit_predict(Z)
    )
    proposed = hungarian_align(
        y, KMeans(n_clusters=K_DEFAULT, random_state=seed, n_init=10).fit_predict(Z_aug)
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, pred, title in [
        (axes[0], vanilla, "Vanilla K-means"),
        (axes[1], proposed, "CCRE-augmented"),
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


# ---- Additional analyses ----------------------------------------------


def framework_diagram(save_path: Path | None = None):
    """Render the CCRE pipeline as a labelled boxes-and-arrows flow chart."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 5.0); ax.axis("off")

    def box(x, y, w, h, text, color):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.0, edgecolor="black", facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
        return (x, y, w, h)

    def arrow(p_from, p_to):
        x0, y0, w0, h0 = p_from
        x1, y1, w1, h1 = p_to
        ax.add_patch(FancyArrowPatch(
            (x0 + w0, y0 + h0 / 2), (x1, y1 + h1 / 2),
            arrowstyle="->", mutation_scale=14, color="black", linewidth=1.0,
        ))

    feat_color = "#dceaf7"
    arm_color = "#fce5cd"
    final_color = "#d9ead3"

    # Top row — original feature path
    a1 = box(0.3, 3.3, 1.7, 0.8, "X\n(20 features)", feat_color)
    a2 = box(2.4, 3.3, 2.0, 0.8, "StandardScaler", feat_color)
    a3 = box(4.8, 3.3, 1.4, 0.8, "Z\n(20-D)", feat_color)

    # Bottom row — ARM path
    b1 = box(0.3, 0.6, 1.7, 0.8, "X\n(20 features)", feat_color)
    b2 = box(2.4, 0.6, 2.4, 0.8, "qcut(4) + binary +\nappend price_range", arm_color)
    b3 = box(5.2, 0.6, 2.2, 0.8, "FP-growth\n(class consequents)\ntop-K per class", arm_color)
    b4 = box(7.8, 0.6, 2.2, 0.8, "Σ log(1+lift)\nper sample × class", arm_color)
    b5 = box(10.4, 0.6, 1.9, 0.8, "softmax\n+ standardize", arm_color)

    # Concatenation + KMeans
    c1 = box(7.6, 2.0, 1.8, 0.8, "concat\n(24-D)", final_color)
    c2 = box(11.5, 2.0, 1.8, 0.8, "K-means\n(K=4)", final_color)

    arrow(a1, a2); arrow(a2, a3)
    arrow(b1, b2); arrow(b2, b3); arrow(b3, b4); arrow(b4, b5)

    # Z → concat (drop down-right) and softmax-std → concat (up-left)
    ax.add_patch(FancyArrowPatch(
        (a3[0] + a3[2] / 2, a3[1]), (c1[0] + c1[2] / 2, c1[1] + c1[3]),
        arrowstyle="->", mutation_scale=14, color="black", linewidth=1.0,
    ))
    ax.add_patch(FancyArrowPatch(
        (b5[0] + b5[2] / 2, b5[1] + b5[3]), (c1[0] + c1[2] / 2, c1[1]),
        arrowstyle="->", mutation_scale=14, color="black", linewidth=1.0,
    ))
    arrow(c1, c2)

    ax.text(0.3, 4.4, "Original feature path (20-D)", fontsize=10, fontweight="bold")
    ax.text(0.3, 1.7, "ARM evidence path (4-D, label-informed)", fontsize=10, fontweight="bold")
    ax.set_title("CCRE pipeline — class-conditional rule evidence + standardized features → K-means")
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def _build_ablation_matrices(
    X: pd.DataFrame, y: np.ndarray, cfg: CCREConfig
) -> dict[str, np.ndarray]:
    """Return the feature matrix for each ablation variant."""
    n_classes = int(np.unique(y).size)
    with_cls, without_cls, _ = _build_class_transactions(X, y, cfg.n_bins)
    rules = _mine_class_rules(with_cls, cfg, n_classes)
    raw = _evidence_raw(without_cls, rules, n_classes)
    soft = _softmax(raw, cfg.softmax_temperature)
    soft_std = StandardScaler().fit_transform(soft)
    Z = StandardScaler().fit_transform(X)
    return {
        "vanilla (Z only, 20-D)":           Z,
        "Z + raw evidence (no softmax)":    np.hstack([Z, raw]),
        "Z + softmax (no standardize)":     np.hstack([Z, soft]),
        "Z + softmax + standardize (CCRE)": np.hstack([Z, soft_std]),
        "evidence only (4-D, no Z)":        soft_std,
    }


def ablation_table(
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: CCREConfig = CCREConfig(),
    seeds: Iterable[int] = SEEDS,
    K: int = K_DEFAULT,
) -> pd.DataFrame:
    """Validate each design step by toggling it on/off and rerunning K-means.

    Variants:
      - vanilla:                Z only (no evidence channel)
      - +raw evidence:          Z ⊕ raw Σ log(1+lift) (skip softmax, skip standardize)
      - +softmax:               Z ⊕ softmax (skip standardize)
      - CCRE (proposed):        Z ⊕ softmax ⊕ standardize
      - evidence only:          softmax-standardize alone (no Z)
    """
    matrices = _build_ablation_matrices(X, y, cfg)
    rows = []
    for name, mat in matrices.items():
        for seed in seeds:
            km = KMeans(n_clusters=K, random_state=seed, n_init=10)
            clusters = km.fit_predict(mat)
            aligned = hungarian_align(y, clusters)
            rows.append({"variant": name, "seed": seed, **evaluate(y, aligned)})
    df = pd.DataFrame(rows)
    metrics = ["accuracy", "precision", "recall", "f1"]
    agg = df.groupby("variant", sort=False)[metrics].agg(["mean", "std"])
    out = pd.DataFrame(index=agg.index)
    for m in metrics:
        out[m] = (
            agg[(m, "mean")].map("{:.4f}".format)
            + " ± "
            + agg[(m, "std")].map("{:.4f}".format)
        )
    return out


def sensitivity_top_per_class(
    X: pd.DataFrame,
    y: np.ndarray,
    values: Iterable[int] = (10, 30, 60, 100, 200, 500),
    base_cfg: CCREConfig = CCREConfig(),
    seeds: Iterable[int] = SEEDS,
    K: int = K_DEFAULT,
    save_path: Path | None = None,
):
    """Sweep `top_per_class` and plot mean F1 / accuracy with seed std bands."""
    rows = []
    for k in values:
        cfg = CCREConfig(
            n_bins=base_cfg.n_bins,
            min_support=base_cfg.min_support,
            min_confidence=base_cfg.min_confidence,
            min_lift=base_cfg.min_lift,
            top_per_class=int(k),
            softmax_temperature=base_cfg.softmax_temperature,
        )
        Z_aug, _ = _build_feature_matrix(X, y, cfg)
        for seed in seeds:
            km = KMeans(n_clusters=K, random_state=seed, n_init=10)
            clusters = km.fit_predict(Z_aug)
            aligned = hungarian_align(y, clusters)
            rows.append(
                {"top_per_class": int(k), "seed": seed, **evaluate(y, aligned)}
            )
    df = pd.DataFrame(rows)
    summary = df.groupby("top_per_class").agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(
        summary["top_per_class"], summary["f1_mean"], yerr=summary["f1_std"],
        marker="o", capsize=3, label="F1", color="#1f77b4",
    )
    ax.errorbar(
        summary["top_per_class"], summary["acc_mean"], yerr=summary["acc_std"],
        marker="s", capsize=3, label="Accuracy", color="#ff7f0e",
    )
    ax.axhline(0.25, color="grey", linestyle=":", alpha=0.6, label="random (0.25)")
    ax.set_xscale("log")
    ax.set_xlabel("top_per_class (rules per class)")
    ax.set_ylabel("Score")
    ax.set_title("CCRE sensitivity to top_per_class (mean ± seed std)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig, summary
