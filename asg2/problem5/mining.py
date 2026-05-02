"""FP-growth itemset mining + lift computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mlxtend.frequent_patterns import fpgrowth

from .data import transactions_to_onehot


@dataclass
class MinedItemsets:
    itemsets: list[frozenset]
    supports: np.ndarray
    lifts: np.ndarray
    transactions: list[list[str]]


def _itemset_lift(itemset: frozenset, support: float,
                  item_supports: dict[str, float]) -> float:
    if len(itemset) <= 1:
        return 1.0
    denom = 1.0
    for it in itemset:
        denom *= item_supports.get(it, support)
    return support / denom if denom > 0 else 1.0


def mine_itemsets(
    transactions: list[list[str]],
    *,
    min_support: float = 0.05,
    min_size: int = 2,
    top_m: int | None = 30,
    rank_by: str = "lift",
    min_lift: float = 1.2,
) -> MinedItemsets:
    """Mine frequent itemsets and rank them by `support` or `lift`.

    Lift surfaces interesting (non-trivial) co-occurrences that are more
    discriminative for clustering than the most-frequent itemsets, which are
    dominated by trivial common features.
    """
    onehot = transactions_to_onehot(transactions)
    freq = fpgrowth(onehot, min_support=min_support, use_colnames=True)
    if freq.empty:
        return MinedItemsets([], np.array([]), np.array([]), transactions)

    item_supports: dict[str, float] = {}
    for _, row in freq.iterrows():
        if len(row["itemsets"]) == 1:
            (item,) = tuple(row["itemsets"])
            item_supports[item] = float(row["support"])

    freq = freq[freq["itemsets"].apply(len) >= min_size].copy()
    if freq.empty:
        return MinedItemsets([], np.array([]), np.array([]), transactions)

    freq["lift"] = [
        _itemset_lift(frozenset(its), float(sup), item_supports)
        for its, sup in zip(freq["itemsets"], freq["support"])
    ]
    freq = freq[freq["lift"] >= min_lift]

    if rank_by == "lift":
        freq = freq.sort_values(
            ["lift", "support"], ascending=[False, False]
        ).reset_index(drop=True)
    elif rank_by == "support":
        freq = freq.sort_values("support", ascending=False).reset_index(drop=True)
    else:
        raise ValueError(f"unknown rank_by: {rank_by}")

    if top_m is not None:
        freq = freq.head(top_m).reset_index(drop=True)

    itemsets = [frozenset(s) for s in freq["itemsets"].tolist()]
    return MinedItemsets(
        itemsets=itemsets,
        supports=freq["support"].to_numpy(),
        lifts=freq["lift"].to_numpy(),
        transactions=transactions,
    )


def supporting_indices(
    transactions: list[list[str]], itemset: frozenset
) -> np.ndarray:
    rows = [set(t) for t in transactions]
    return np.array(
        [i for i, rs in enumerate(rows) if itemset.issubset(rs)], dtype=int
    )
