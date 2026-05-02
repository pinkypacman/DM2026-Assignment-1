"""Problem 4 — PCA visualization and K-means clustering on mobile_price.csv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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


def load_q4_data() -> Q4Data:
    df = pd.read_csv(DATA_PATH)
    y = df["price_range"].to_numpy()
    X = df.drop(columns=["price_range"]).to_numpy()
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(Z)
    pca2 = pca.transform(Z)
    return Q4Data(X=X, Z=Z, pca2=pca2, pca=pca, y=y)


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

    print("\nFigures saved to asg2/figures/.")


if __name__ == "__main__":
    main()
