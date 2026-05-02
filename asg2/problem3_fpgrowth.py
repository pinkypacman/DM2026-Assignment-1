"""Problem 3 — FP-growth association rule mining on mobile_price.csv.

Filter rows where price_range == 1, take {ram, int_memory, px_width,
battery_power}, discretize each into low / medium / high using a 3:4:3
*width* ratio, build transactions, and mine frequent patterns and rules.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mobile_price.csv"

TARGET_FEATURES = ["ram", "int_memory", "px_width", "battery_power"]
LABELS = ["low", "medium", "high"]

MIN_SUPPORT = 0.3
MIN_CONFIDENCE = 0.4
MIN_LIFT = 0.8


def discretize_3_4_3(series: pd.Series) -> pd.Series:
    """Split a numeric series into low/medium/high using a 3:4:3 width ratio.

    Cuts at min + 0.3*range and min + 0.7*range. Returns categorical labels.
    """
    lo, hi = series.min(), series.max()
    rng = hi - lo
    cut1 = lo + 0.3 * rng
    cut2 = lo + 0.7 * rng
    # `right=True` (default for pd.cut) is fine; `include_lowest=True` makes the
    # smallest value land in the first bin.
    return pd.cut(
        series,
        bins=[lo - 1e-9, cut1, cut2, hi + 1e-9],
        labels=LABELS,
        include_lowest=True,
    ).astype("object")


def build_transactions(df: pd.DataFrame, features=TARGET_FEATURES) -> list[list[str]]:
    """For each row, return [feature_label, ...] using 3:4:3 discretization."""
    disc = pd.DataFrame(
        {f: discretize_3_4_3(df[f]) for f in features}, index=df.index
    )
    transactions = []
    for _, row in disc.iterrows():
        transactions.append([f"{f}_{row[f]}" for f in features])
    return transactions


def transactions_to_onehot(transactions: list[list[str]]) -> pd.DataFrame:
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    return pd.DataFrame(arr, columns=te.columns_)


def load_filtered_df(price_range: int = 1) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df[df["price_range"] == price_range].reset_index(drop=True)


def run_q3a(min_support: float = MIN_SUPPORT) -> pd.DataFrame:
    """Frequent itemsets with support >= min_support."""
    df = load_filtered_df(1)
    transactions = build_transactions(df)
    onehot = transactions_to_onehot(transactions)
    freq = fpgrowth(onehot, min_support=min_support, use_colnames=True)
    freq = freq.sort_values("support", ascending=False).reset_index(drop=True)
    freq["itemsets"] = freq["itemsets"].apply(lambda s: sorted(s))
    return freq


def run_q3b(
    min_support: float = MIN_SUPPORT,
    min_confidence: float = MIN_CONFIDENCE,
    min_lift: float = MIN_LIFT,
) -> pd.DataFrame:
    """Association rules with support>=, confidence>=, lift>= thresholds."""
    df = load_filtered_df(1)
    transactions = build_transactions(df)
    onehot = transactions_to_onehot(transactions)
    freq = fpgrowth(onehot, min_support=min_support, use_colnames=True)
    rules = association_rules(
        freq, metric="confidence", min_threshold=min_confidence
    )
    rules = rules[(rules["support"] >= min_support) & (rules["lift"] >= min_lift)]
    rules = rules.sort_values(
        ["lift", "confidence", "support"], ascending=False
    ).reset_index(drop=True)
    rules["antecedents"] = rules["antecedents"].apply(lambda s: sorted(s))
    rules["consequents"] = rules["consequents"].apply(lambda s: sorted(s))
    return rules


def main():
    print(f"=== Q3(a): frequent patterns with support >= {MIN_SUPPORT} ===")
    freq = run_q3a()
    print(f"{len(freq)} frequent itemsets found.")
    with pd.option_context("display.max_rows", None, "display.max_colwidth", None):
        print(freq.round(4))

    print(
        f"\n=== Q3(b): rules with support >= {MIN_SUPPORT}, "
        f"confidence >= {MIN_CONFIDENCE}, lift >= {MIN_LIFT} ==="
    )
    rules = run_q3b()
    print(f"{len(rules)} rules found.")
    with pd.option_context("display.max_rows", None, "display.max_colwidth", None):
        cols = [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift",
        ]
        print(rules[cols].round(4))


if __name__ == "__main__":
    main()
