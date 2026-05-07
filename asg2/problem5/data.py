"""Data loading constants for Problem 5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "mobile_price.csv"
SEEDS = [0, 10, 42, 100, 999]
K_DEFAULT = 4


def load_xy() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATA_PATH)
    y = df["price_range"].to_numpy()
    X = df.drop(columns=["price_range"])
    return X, y
