"""Problem 5 — Enhancing K-means with Association Rule Mining.

Public API (imported via `import problem5 as q5`):

  Data:           load_xy, K_DEFAULT, SEEDS
  Label-informed: CCREConfig, run_label_informed, summarize_label_informed,
                  top_class_rules_table, pca_triptych_label_informed,
                  confusion_pair_label_informed
"""

from .data import K_DEFAULT, SEEDS, load_xy
from .label_informed import (
    CCREConfig,
    ablation_table,
    confusion_pair_label_informed,
    framework_diagram,
    pca_triptych_label_informed,
    run_label_informed,
    sensitivity_top_per_class,
    summarize_label_informed,
    top_class_rules_table,
)

__all__ = [
    # data
    "K_DEFAULT", "SEEDS", "load_xy",
    # label-informed (CCRE)
    "CCREConfig",
    "ablation_table",
    "confusion_pair_label_informed",
    "framework_diagram",
    "pca_triptych_label_informed",
    "run_label_informed",
    "sensitivity_top_per_class",
    "summarize_label_informed",
    "top_class_rules_table",
]
