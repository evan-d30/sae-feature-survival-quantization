"""Small reusable metrics for QDM-style feature-survival analysis."""

from __future__ import annotations

import numpy as np


def pearson_corr_per_feature(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute per-column Pearson correlation between two activation matrices.

    Args:
        x: FP16 feature activations with shape [tokens, features].
        y: compressed feature activations with shape [tokens, features].
        eps: numerical stabilizer for near-zero variance features.

    Returns:
        Correlation vector with shape [features].
    """
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    num = (x * y).sum(axis=0)
    den = np.sqrt((x * x).sum(axis=0) * (y * y).sum(axis=0)) + eps
    return num / den


def survival_bands(corr: np.ndarray, survival_threshold: float = 0.9, damage_threshold: float = 0.5) -> dict[str, float]:
    """Return survived/degraded/damaged percentages for a correlation vector."""
    corr = np.asarray(corr)
    n = corr.size
    if n == 0:
        return {"survived": np.nan, "degraded": np.nan, "damaged": np.nan}
    survived = (corr > survival_threshold).mean() * 100
    damaged = (corr < damage_threshold).mean() * 100
    degraded = 100 - survived - damaged
    return {"survived": survived, "degraded": degraded, "damaged": damaged}


def jaccard_non_survived(corr_a: np.ndarray, corr_b: np.ndarray, threshold: float = 0.9) -> float:
    """Jaccard overlap between non-survived feature sets under two conditions."""
    a = np.asarray(corr_a) <= threshold
    b = np.asarray(corr_b) <= threshold
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)
