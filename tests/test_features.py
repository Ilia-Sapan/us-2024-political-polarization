"""Tests for feature matrix generation."""

from __future__ import annotations

import pandas as pd

from polarization.cleaning import build_preprocessing_pipeline
from polarization.features import fit_transform_features, transform_features


def test_fit_transform_returns_feature_names() -> None:
    """Fitting feature pipeline should return matrix and consistent names."""

    frame = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "gender": ["M", "F", "F"],
        }
    )
    preprocessor = build_preprocessing_pipeline(["age"], ["gender"])
    matrix, fitted, names = fit_transform_features(frame, preprocessor)
    assert matrix.shape[0] == 3
    assert len(names) == matrix.shape[1]
    assert any("gender" in name for name in names)
    assert fitted is preprocessor


def test_transform_handles_unseen_categories() -> None:
    """Unknown categorical levels should not crash transformation."""

    train = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "gender": ["M", "F", "F"],
        }
    )
    test = pd.DataFrame(
        {
            "age": [25, 35],
            "gender": ["M", "X"],  # unseen category X
        }
    )

    preprocessor = build_preprocessing_pipeline(["age"], ["gender"])
    fit_transform_features(train, preprocessor)
    transformed = transform_features(test, preprocessor)
    assert transformed.shape[0] == 2

