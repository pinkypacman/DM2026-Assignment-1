"""Data loading, discretization, and transaction encoding for Problem 5."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "mobile_price.csv"
SEEDS = [0, 10, 42, 100, 999]
K_DEFAULT = 4

BINARY_NATIVE = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]


def load_xy() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATA_PATH)
    y = df["price_range"].to_numpy()
    X = df.drop(columns=["price_range"])
    return X, y


def discretize_3_4_3(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    rng = hi - lo
    return pd.cut(
        series,
        bins=[lo - 1e-9, lo + 0.3 * rng, lo + 0.7 * rng, hi + 1e-9],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype("object")


def discretize_quantile(series: pd.Series) -> pd.Series:
    return pd.qcut(
        series, q=3, labels=["low", "medium", "high"], duplicates="drop"
    ).astype("object")


def build_transactions(
    X: pd.DataFrame,
    *,
    binary_native: Sequence[str] = BINARY_NATIVE,
    include_binary_native: bool = True,
    discretize_mode: str = "width_3_4_3",
) -> list[list[str]]:
    binary_set = set(binary_native) if include_binary_native else set()
    cont_cols = [c for c in X.columns if c not in binary_set]

    if discretize_mode == "width_3_4_3":
        disc_fn = discretize_3_4_3
    elif discretize_mode == "quantile":
        disc_fn = discretize_quantile
    else:
        raise ValueError(f"unknown discretize_mode: {discretize_mode}")

    disc = pd.DataFrame({c: disc_fn(X[c]) for c in cont_cols}, index=X.index)

    transactions: list[list[str]] = []
    for idx, row in disc.iterrows():
        items = [f"{c}_{row[c]}" for c in cont_cols]
        if include_binary_native:
            for b in binary_native:
                if int(X.loc[idx, b]) == 1:
                    items.append(f"{b}=1")
        transactions.append(items)
    return transactions


def transactions_to_onehot(transactions: list[list[str]]) -> pd.DataFrame:
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    return pd.DataFrame(arr, columns=te.columns_)
