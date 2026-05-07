"""Problem 2 — SVM on mobile_price.csv.

The target `price_range` is a 4-class label, so we report macro-averaged F1
(unweighted mean over the 4 classes) alongside accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mobile_price.csv"
RANDOM_STATE = 42
# (b) explore different values of the regularization parameter C
C_GRID = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]


@dataclass
class Split:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def load_split(seed: int = RANDOM_STATE) -> Split:
    """Load mobile_price.csv and split 60/20/20 with the given seed.

    Features are standardized with the scaler fit only on the training
    portion — SVC with the default RBF kernel is scale-sensitive, and the
    raw features span very different ranges (e.g. ram ~ thousands vs binary
    flags), so unscaled inputs let one or two large-variance features
    dominate the kernel.
    """
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["price_range"]).to_numpy()
    y = df["price_range"].to_numpy()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, shuffle=True, stratify=y
    )
    # 20:60 -> 1:3 val:train split -> test size = 0.25 for trainval
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=seed, shuffle=True, stratify=y_trainval
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return Split(X_train, X_val, X_test, y_train, y_val, y_test)


def _score(model, X, y) -> tuple[float, float]:
    pred = model.predict(X)
    return accuracy_score(y, pred), f1_score(y, pred, average="macro")


def run_q2a(split: Split, C: float = 1.0) -> pd.DataFrame:
    """Train SVC(C=1.0) and return acc/macro-F1 on train/val/test."""
    model = SVC(C=C)
    model.fit(split.X_train, split.y_train)
    rows = []
    for name, X, y in [
        ("train", split.X_train, split.y_train),
        ("val", split.X_val, split.y_val),
        ("test", split.X_test, split.y_test),
    ]:
        acc, f1 = _score(model, X, y)
        rows.append({"split": name, "accuracy": acc, "macro_f1": f1})
    return pd.DataFrame(rows)

def run_q2b(split: Split, C_grid=C_GRID) -> pd.DataFrame:
    """Sweep C; return long-form DataFrame with cols [C, split, accuracy, macro_f1]."""
    rows = []
    for C in C_grid:
        model = SVC(C=C)
        model.fit(split.X_train, split.y_train)
        for name, X, y in [
            ("train", split.X_train, split.y_train),
            ("val", split.X_val, split.y_val),
            ("test", split.X_test, split.y_test),
        ]:
            acc, f1 = _score(model, X, y)
            C_out = int(C) if C >= 1 else C  # for nicer printing: int if whole number, else float
            rows.append({"C": C_out, "split": name, "accuracy": acc, "macro_f1": f1})
    return pd.DataFrame(rows)


def plot_q2b(results: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Two-panel line plot showing how acc and macro-F1 change across C values."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1-score")]
    colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}

    val_argmax_C = pick_best_C(results)

    for ax, (col, title) in zip(axes, metrics):
        for split_name in ["train", "val", "test"]:
            sub = results[results["split"] == split_name].sort_values("C")
            ax.plot(
                sub["C"],
                sub[col],
                marker="o",
                label=split_name,
                color=colors[split_name],
            )
        ax.axvline(
            val_argmax_C,
            color="grey",
            linestyle="--",
            alpha=0.6,
            label=f"best C (by val) = {val_argmax_C:g}",
        )
        ax.set_xscale("log")
        ax.set_xlabel("C (log scale)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs C")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle("SVM performance vs regularization parameter C")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def pick_best_C(results: pd.DataFrame) -> float:
    """Return the C maximizing validation macro F1 (tie-break by accuracy)."""
    val = results[results["split"] == "val"].copy()
    val = val.sort_values(["macro_f1", "accuracy"], ascending=False)
    return float(val.iloc[0]["C"])


def pivot_results(results: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    """Wide-form pivot: rows = C, cols = split, vals = metric."""
    return results.pivot(index="C", columns="split", values=metric)[
        ["train", "val", "test"]
    ]


def main():
    split = load_split()
    print("=== Q2(a): SVC(C=1.0) ===")
    print(run_q2a(split).round(4))

    print("\n=== Q2(b): C sweep ===")
    results = run_q2b(split)
    print("Macro F1 per C:")
    print(pivot_results(results, "macro_f1").round(4))
    print("\nAccuracy per C:")
    print(pivot_results(results, "accuracy").round(4))

    best_C = pick_best_C(results)
    print(f"\nBest C by validation macro F1: {best_C:g}")
    plot_q2b(results, save_path=Path(__file__).parent / "figures" / "q2b_c_sweep.png")
    print("Saved figure to asg2/figures/q2b_c_sweep.png")


if __name__ == "__main__":
    main()
