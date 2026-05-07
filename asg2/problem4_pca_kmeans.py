"""Problem 4 — PCA visualization and K-means clustering on mobile_price.csv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mobile_price.csv"
RANDOM_STATE = 42
N_CLUSTERS = 4
CLASS_NAMES = {0: "low cost", 1: "medium cost", 2: "high cost", 3: "very high cost"}


@dataclass
class Q4Data:
    X: np.ndarray            # raw features
    Z: np.ndarray            # z-scored features
    pca2: np.ndarray         # 2-D PCA projection
    pca: PCA
    y: np.ndarray            # price_range
    feature_names: list[str] # original feature column names


def load_q4_data() -> Q4Data:
    df = pd.read_csv(DATA_PATH)
    y = df["price_range"].to_numpy()
    feat_df = df.drop(columns=["price_range"])
    X = feat_df.to_numpy()
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(Z)
    pca2 = pca.transform(Z)
    return Q4Data(
        X=X, Z=Z, pca2=pca2, pca=pca, y=y,
        feature_names=list(feat_df.columns),
    )


def _scatter(coords: np.ndarray, labels: np.ndarray, title: str,
             label_names: dict | None = None,
             ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5))
    label_names = label_names or {int(v): str(int(v)) for v in np.unique(labels)}
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(sorted(np.unique(labels))):
        m = labels == lab
        ax.scatter(
            coords[m, 0], coords[m, 1],
            s=14, color=cmap(i), alpha=0.7,
            label=label_names.get(int(lab), str(lab)),
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.3)
    return ax


def run_q4b(data: Q4Data, save_path: Path | None = None) -> plt.Figure:
    """Scatter the first two PCs colored by class label."""
    var = data.pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(7, 5.2))
    _scatter(
        data.pca2, data.y,
        title=(
            "Q4(b) PCA-2D scatter, colored by price_range "
            f"(var: PC1={var[0]:.2%}, PC2={var[1]:.2%})"
        ),
        label_names=CLASS_NAMES, ax=ax,
    )
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def run_q4c(data: Q4Data, save_path: Path | None = None
            ) -> tuple[plt.Figure, float]:
    """K-means on all 20 standardized features; visualize on PCA-2D coords."""
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    cluster = km.fit_predict(data.Z)
    ari = adjusted_rand_score(data.y, cluster)

    fig, ax = plt.subplots(figsize=(7, 5.2))
    _scatter(
        data.pca2, cluster,
        title=(
            f"Q4(c) K-means (K=4) clusters on all 20 features\n"
            f"plotted on PCA-2D, ARI={ari:.4f}"
        ),
        label_names={i: f"cluster {i}" for i in range(N_CLUSTERS)},
        ax=ax,
    )
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig, ari


def run_q4d(data: Q4Data, save_path: Path | None = None
            ) -> tuple[plt.Figure, float]:
    """K-means on the 2-D PCA features; visualize and report ARI."""
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    cluster = km.fit_predict(data.pca2)
    ari = adjusted_rand_score(data.y, cluster)

    fig, ax = plt.subplots(figsize=(7, 5.2))
    _scatter(
        data.pca2, cluster,
        title=f"Q4(d) K-means (K=4) on PCA-2D features, ARI={ari:.4f}",
        label_names={i: f"cluster {i}" for i in range(N_CLUSTERS)},
        ax=ax,
    )
    # Overlay centroids in PCA-2D (km is already in PCA-2D space).
    ax.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        marker="X", s=180, c="black", edgecolors="white",
        linewidths=1.5, label="centroids",
    )
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig, ari


def run_q4e(data: Q4Data, save_path: Path | None = None
            ) -> tuple[plt.Figure, pd.DataFrame]:
    """Per-feature importance: ANOVA F vs price_range alongside |PC1|/|PC2|.

    Reveals the gap between PCA's unsupervised loading distribution and the
    actually class-discriminative features — explains why both PCA-2D and
    20-D K-means miss `price_range`.
    """
    f_stat, _ = f_classif(data.Z, data.y)
    pc1 = np.abs(data.pca.components_[0])
    pc2 = np.abs(data.pca.components_[1])

    importance = pd.DataFrame({
        "feature": data.feature_names,
        "f_stat": f_stat,
        "abs_pc1_loading": pc1,
        "abs_pc2_loading": pc2,
    }).sort_values("f_stat", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    y_pos = np.arange(len(importance))

    axes[0].barh(y_pos, importance["f_stat"], color="#377eb8")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(importance["feature"])
    axes[0].set_xlabel("ANOVA F-statistic vs price_range")
    axes[0].set_title("Class-discriminative power (supervised)")
    axes[0].grid(axis="x", alpha=0.3)

    width = 0.4
    axes[1].barh(y_pos - width / 2, importance["abs_pc1_loading"],
                 height=width, color="#e41a1c", label="|PC1 loading|")
    axes[1].barh(y_pos + width / 2, importance["abs_pc2_loading"],
                 height=width, color="#4daf4a", label="|PC2 loading|")
    axes[1].set_xlabel("Absolute PCA loading")
    axes[1].set_title("PCA contribution (unsupervised)")
    axes[1].legend(loc="lower right", frameon=True)
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Q4(e) Feature importance — supervised F-stat vs PCA loadings "
        "(features sorted by F-stat)",
        fontsize=12,
    )
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig, importance.iloc[::-1].reset_index(drop=True)


def main():
    out_dir = Path(__file__).parent / "figures"
    data = load_q4_data()

    print("=== Q4(a): standardization summary ===")
    print(
        pd.DataFrame(data.Z).describe().T[["mean", "std"]].round(4).head(),
    )

    print("\n=== Q4(b): PCA-2D scatter (saved) ===")
    run_q4b(data, save_path=out_dir / "q4b_pca_classes.png")
    print(
        "Explained variance: PC1 = "
        f"{data.pca.explained_variance_ratio_[0]:.4%}, PC2 = "
        f"{data.pca.explained_variance_ratio_[1]:.4%}"
    )

    print("\n=== Q4(c): K-means on all 20 features ===")
    _, ari_full = run_q4c(data, save_path=out_dir / "q4c_kmeans_full.png")
    print(f"ARI (full features) = {ari_full:.4f}")

    print("\n=== Q4(d): K-means on PCA-2D features ===")
    _, ari_pca = run_q4d(data, save_path=out_dir / "q4d_kmeans_pca2.png")
    print(f"ARI (PCA-2D features) = {ari_pca:.4f}")

    print("\n=== Q4(e): per-feature importance (F-stat vs PCA loadings) ===")
    _, importance = run_q4e(data, save_path=out_dir / "q4e_feature_importance.png")
    print("Top 5 by ANOVA F-statistic against price_range:")
    print(importance.head(5).round(4).to_string(index=False))

    print("\nFigures saved to asg2/figures/.")


if __name__ == "__main__":
    main()
