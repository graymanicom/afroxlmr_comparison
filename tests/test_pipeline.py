import numpy as np
import pandas as pd
import torch

from afroxlmr_mechint.pipeline import (
    DatasetSchemaError,
    RelativeDepthAligner,
    compute_false_localisation_index,
    linear_cka,
    masked_mean_pool,
    pooled_cosine_similarity,
    summarise_early_mid_divergence,
    summarise_late_layer_share,
    validate_dataset_schema,
)


def test_relative_depth_alignment():
    aligner = RelativeDepthAligner((0.0, 0.25, 0.5, 0.75, 1.0))
    assert aligner.layer_indices(12) == [0, 3, 6, 8, 11]
    assert aligner.layer_indices(24) == [0, 6, 12, 17, 23]


def test_masked_mean_pool():
    hs = torch.tensor([
        [[1.0, 0.0], [3.0, 0.0], [100.0, 100.0]],
        [[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]],
    ])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    pooled = masked_mean_pool(hs, mask)
    assert torch.allclose(pooled[0], torch.tensor([2.0, 0.0]))
    assert torch.allclose(pooled[1], torch.tensor([4.0, 4.0]))


def test_linear_cka_identity():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 6))
    assert abs(linear_cka(x, x) - 1.0) < 1e-6


def test_pooled_cosine_similarity_orthogonal():
    x = np.array([[1.0, 0.0], [1.0, 0.0]])
    y = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert abs(pooled_cosine_similarity(x, y)) < 1e-8


def test_validate_dataset_schema_accepts_good_pairs():
    df = pd.DataFrame({
        "id": ["a", "b"],
        "text": ["x", "y"],
        "label": [0, 1],
        "language": ["eng", "eng"],
        "is_local_task": [1, 1],
        "pair_id": ["p1", "p1"],
        "pair_role": ["base", "local"],
        "split": ["test", "test"],
    })
    out = validate_dataset_schema(df)
    assert len(out) == 2


def test_validate_dataset_schema_rejects_incomplete_pair():
    df = pd.DataFrame({
        "id": ["a"],
        "text": ["x"],
        "label": [0],
        "language": ["eng"],
        "is_local_task": [1],
        "pair_id": ["p1"],
        "pair_role": ["base"],
        "split": ["test"],
    })
    try:
        validate_dataset_schema(df)
        assert False
    except DatasetSchemaError:
        assert True


def test_false_localisation_index_bounds():
    score = compute_false_localisation_index(0.9, 0.1)
    assert 0.0 <= score <= 1.0


def test_summary_helpers():
    ablation_df = pd.DataFrame({
        "layer": ["baseline", 0, 1, 2, 3, 4, 5],
        "local_accuracy": [1.0, 0.95, 0.95, 0.94, 0.80, 0.70, 0.60],
        "global_accuracy": [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94],
    })
    share = summarise_late_layer_share(ablation_df)
    assert 0.0 <= share <= 1.0
    cka_df = pd.DataFrame({
        "aligned_depth": ["0%", "25%", "50%", "75%", "100%"],
        "linear_cka": [0.95, 0.90, 0.85, 0.60, 0.55],
    })
    div = summarise_early_mid_divergence(cka_df)
    assert 0.0 <= div <= 1.0
