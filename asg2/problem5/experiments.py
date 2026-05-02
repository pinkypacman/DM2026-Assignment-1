"""Experiment functions producing tidy DataFrames for reporting."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data import K_DEFAULT, SEEDS, build_transactions
from .enhancements import run_method_D, run_method_E, run_method_F
from .methods import (
    ARMConfig,
    METHODS,
    PROPOSED_METHOD,
    evaluate,
    hungarian_align,
    prepare_space,
    run_kmeans_once,
    run_method,
    seeded_centroids,
    summarize,
)
from .mining import mine_itemsets
from .components import select_seed_itemsets


def experiment_vanilla_vs_proposed(
    X, y, seeds=SEEDS, cfg=ARMConfig(), proposed: str = "F+C",
):
    """Headline comparison: vanilla vs the strongest method (F+C) by default.

    `proposed` may be any name in METHODS or one of the enhancement labels
    {"D_kernel", "E_iterative", "F_selected", "F+C"}.
    """
    enhancement_runners = {
        "D_kernel":   lambda: run_method_D(X, y, seeds=seeds, cfg=cfg),
        "E_iterative": lambda: run_method_E(X, y, seeds=seeds, cfg=cfg),
        "F_selected": lambda: run_method_F(X, y, seeds=seeds, cfg=cfg, also_weight=False),
        "F+C":        lambda: run_method_F(X, y, seeds=seeds, cfg=cfg, also_weight=True),
    }
    rows = [run_method(X, y, "vanilla", seeds=seeds, cfg=cfg)]
    if proposed in enhancement_runners:
        rows.append(enhancement_runners[proposed]())
    else:
        rows.append(run_method(X, y, proposed, seeds=seeds, cfg=cfg))
    raw = pd.concat(rows, ignore_index=True)
    return raw, summarize(raw)


def experiment_ablation(X, y, seeds=SEEDS, cfg=ARMConfig()):
    rows = [run_method(X, y, m, seeds=seeds, cfg=cfg) for m in METHODS]
    raw = pd.concat(rows, ignore_index=True)
    return raw, summarize(raw)


def experiment_enhancements(X, y, seeds=SEEDS, cfg=ARMConfig()):
    """Methods D, E, F (and F+C); also include vanilla and C_only for context."""
    rows = [
        run_method(X, y, "vanilla", seeds=seeds, cfg=cfg),
        run_method(X, y, "C_only", seeds=seeds, cfg=cfg),
        run_method_D(X, y, seeds=seeds, cfg=cfg),
        run_method_E(X, y, seeds=seeds, cfg=cfg),
        run_method_F(X, y, seeds=seeds, cfg=cfg, also_weight=False),
        run_method_F(X, y, seeds=seeds, cfg=cfg, also_weight=True),
    ]
    raw = pd.concat(rows, ignore_index=True)
    return raw, summarize(raw)


def experiment_min_support_sweep(
    X, y, seeds=SEEDS,
    supports: Sequence[float] = (0.05, 0.08, 0.10, 0.15, 0.20, 0.30),
):
    rows = []
    for s in supports:
        cfg = ARMConfig(min_support=s)
        df = run_method(X, y, "C_only", seeds=seeds, cfg=cfg)
        df["min_support"] = s
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def experiment_K_sweep(
    X, y, seeds=SEEDS, Ks: Sequence[int] = (3, 4, 5, 6),
):
    cfg = ARMConfig()
    rows = []
    for K in Ks:
        Z_base = StandardScaler().fit_transform(X)
        for seed in seeds:
            clusters = run_kmeans_once(Z_base, seed, K=K, n_init=10)
            aligned = hungarian_align(y, clusters)
            rows.append({"method": "vanilla", "K": K, "seed": seed,
                         **evaluate(y, aligned)})

        space = prepare_space(X, cfg, augment=True, seed_init=False)
        transactions = build_transactions(
            X, include_binary_native=cfg.include_binary_native
        )
        mined = mine_itemsets(
            transactions, min_support=cfg.min_support, min_size=cfg.min_size,
            top_m=cfg.top_m, rank_by=cfg.rank_by, min_lift=cfg.min_lift,
        )
        seed_sets_K = select_seed_itemsets(mined, K=K, tau=cfg.tau)
        for seed in seeds:
            centroids = seeded_centroids(space.Z_aug, seed_sets_K, K, seed)
            clusters = run_kmeans_once(space.Z_aug, seed, K=K, init=centroids)
            aligned = hungarian_align(y, clusters)
            rows.append({"method": "A+B", "K": K, "seed": seed,
                         **evaluate(y, aligned)})
    return pd.DataFrame(rows)


def experiment_binary_native_toggle(X, y, seeds=SEEDS):
    rows = []
    for include in [True, False]:
        cfg = ARMConfig(include_binary_native=include)
        df = run_method(X, y, "C_only", seeds=seeds, cfg=cfg)
        df["include_binary_native"] = include
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def experiment_alpha_sweep(
    X, y, seeds=SEEDS, alphas: Sequence[float] = (0.5, 1.0, 2.0),
):
    rows = []
    for a in alphas:
        cfg = ARMConfig(alpha=a)
        df = run_method(X, y, "A+B", seeds=seeds, cfg=cfg)
        df["alpha"] = a
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def experiment_compute_cost(X, y, seeds=SEEDS, cfg=ARMConfig()):
    rows = []
    method_calls = [
        ("vanilla", lambda: run_method(X, y, "vanilla", seeds=[s], cfg=cfg)),
        ("C_only",  lambda: run_method(X, y, "C_only",  seeds=[s], cfg=cfg)),
        ("D_kernel", lambda: run_method_D(X, y, seeds=[s], cfg=cfg)),
        ("E_iterative", lambda: run_method_E(X, y, seeds=[s], cfg=cfg)),
        ("F+C", lambda: run_method_F(X, y, seeds=[s], cfg=cfg, also_weight=True)),
    ]
    for label, fn in method_calls:
        for s in seeds:
            t0 = time.perf_counter()
            fn()
            rows.append({"method": label, "seed": s,
                         "wall_seconds": time.perf_counter() - t0})
    return pd.DataFrame(rows)
