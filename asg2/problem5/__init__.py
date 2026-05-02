"""Problem 5 — Enhancing K-means with Association Rule Mining.

Public API (imported via `import problem5 as q5`):

  Data:        load_xy, build_transactions, BINARY_NATIVE, SEEDS, K_DEFAULT
  Mining:      mine_itemsets, MinedItemsets, supporting_indices
  Components:  build_augmented_space, select_seed_itemsets, seeded_centroids,
               feature_weights_from_itemsets
  Methods:     ARMConfig, METHODS, PROPOSED_METHOD, run_method, summarize,
               hungarian_align, evaluate
  Enhancements: run_method_D, run_method_E, run_method_F  (D=kernel/spectral
               on rule similarity, E=iterative pattern-cluster refinement,
               F=ARM-guided feature selection)
  Experiments: experiment_vanilla_vs_proposed, experiment_ablation,
               experiment_enhancements, experiment_min_support_sweep,
               experiment_K_sweep, experiment_binary_native_toggle,
               experiment_alpha_sweep, experiment_compute_cost
  Viz:         pca_triptych, confusion_pair, top_itemset_purity,
               feature_weight_bar
"""

from .data import (
    BINARY_NATIVE, K_DEFAULT, SEEDS,
    build_transactions, discretize_3_4_3, discretize_quantile, load_xy,
)
from .mining import MinedItemsets, mine_itemsets, supporting_indices
from .components import (
    build_augmented_space, build_indicator_matrix,
    feature_weights_from_itemsets, seeded_centroids, select_seed_itemsets,
)
from .methods import (
    ARMConfig, METHODS, PROPOSED_METHOD, PreparedSpace,
    evaluate, hungarian_align, prepare_space, run_kmeans_once, run_method,
    summarize,
)
from .enhancements import run_method_D, run_method_E, run_method_F
from .experiments import (
    experiment_alpha_sweep, experiment_ablation, experiment_binary_native_toggle,
    experiment_compute_cost, experiment_enhancements, experiment_K_sweep,
    experiment_min_support_sweep, experiment_vanilla_vs_proposed,
)
from .viz import (
    confusion_pair, feature_weight_bar, pca_triptych, top_itemset_purity,
)

__all__ = [
    # data
    "BINARY_NATIVE", "K_DEFAULT", "SEEDS",
    "build_transactions", "discretize_3_4_3", "discretize_quantile", "load_xy",
    # mining
    "MinedItemsets", "mine_itemsets", "supporting_indices",
    # components
    "build_augmented_space", "build_indicator_matrix",
    "feature_weights_from_itemsets", "seeded_centroids", "select_seed_itemsets",
    # methods
    "ARMConfig", "METHODS", "PROPOSED_METHOD", "PreparedSpace",
    "evaluate", "hungarian_align", "prepare_space", "run_kmeans_once",
    "run_method", "summarize",
    # enhancements
    "run_method_D", "run_method_E", "run_method_F",
    # experiments
    "experiment_alpha_sweep", "experiment_ablation",
    "experiment_binary_native_toggle", "experiment_compute_cost",
    "experiment_enhancements", "experiment_K_sweep",
    "experiment_min_support_sweep", "experiment_vanilla_vs_proposed",
    # viz
    "confusion_pair", "feature_weight_bar", "pca_triptych",
    "top_itemset_purity",
]
