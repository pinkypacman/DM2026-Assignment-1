"""CLI entry point: `python -m problem5` (when asg2/ is on sys.path)."""

from pathlib import Path

from .data import load_xy
from .experiments import (
    experiment_ablation,
    experiment_alpha_sweep,
    experiment_binary_native_toggle,
    experiment_compute_cost,
    experiment_enhancements,
    experiment_K_sweep,
    experiment_min_support_sweep,
    experiment_vanilla_vs_proposed,
)
from .methods import PROPOSED_METHOD, summarize
from .viz import confusion_pair, feature_weight_bar, pca_triptych, top_itemset_purity


def main():
    out_dir = Path(__file__).resolve().parent.parent / "figures"
    X, y = load_xy()

    print(f"=== Q5 main results: vanilla vs proposed ({PROPOSED_METHOD}) ===")
    _, summary_main = experiment_vanilla_vs_proposed(X, y)
    print(summary_main)

    print("\n=== Q5 ablation: A/B/C and combinations ===")
    _, summary_abl = experiment_ablation(X, y)
    print(summary_abl)

    print("\n=== Q5 enhancements: D (kernel), E (iterative), F (selection) ===")
    _, summary_enh = experiment_enhancements(X, y)
    print(summary_enh)

    print("\n=== Q5 sensitivity: min_support sweep (C_only) ===")
    print(summarize(experiment_min_support_sweep(X, y), group="min_support"))

    print("\n=== Q5 sensitivity: K sweep ===")
    print(summarize(experiment_K_sweep(X, y), group=["method", "K"]))

    print("\n=== Q5 binary-native toggle (C_only) ===")
    print(summarize(experiment_binary_native_toggle(X, y),
                    group="include_binary_native"))

    print("\n=== Q5 alpha sweep (A+B) ===")
    print(summarize(experiment_alpha_sweep(X, y), group="alpha"))

    print("\n=== Q5 compute cost ===")
    cost = experiment_compute_cost(X, y)
    print(cost.groupby("method")["wall_seconds"].agg(["mean", "std"]).round(4))

    print("\n=== Q5 top-itemset purity ===")
    print(top_itemset_purity(X, y, top=10).round(4))

    pca_triptych(X, y, save_path=out_dir / "q5_triptych.png")
    confusion_pair(X, y, save_path=out_dir / "q5_confusion.png")
    feature_weight_bar(X, save_path=out_dir / "q5_feature_weights.png")
    print(f"\nFigures saved under {out_dir}")


if __name__ == "__main__":
    main()
