"""
AfroXLMR / XLM-R Mechanistic Interpretability Toolkit

Tools for analysing and comparing internal representations
across multilingual transformer models, with a focus on
localisation and false localisation in African language models.
"""

from .pipeline import (
    DatasetSchemaError,
    RelativeDepthAligner,
    linear_cka,
    pooled_cosine_similarity,
    compute_false_localisation_index,
    summarise_late_layer_share,
    summarise_early_mid_divergence,
    run_full_comparison,
)

__all__ = [
    "DatasetSchemaError",
    "RelativeDepthAligner",
    "linear_cka",
    "pooled_cosine_similarity",
    "compute_false_localisation_index",
    "summarise_late_layer_share",
    "summarise_early_mid_divergence",
    "run_full_comparison",
]
