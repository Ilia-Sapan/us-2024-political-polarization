"""Tests for cleaning and preprocessing helpers."""

from __future__ import annotations

import pandas as pd

from polarization.cleaning import (
    build_preprocessing_pipeline,
    clean_dataframe,
    infer_feature_types,
)


def test_clean_dataframe_drops_duplicate_ids_and_normalizes_missing_tokens() -> None:
    """Duplicate ID rows should be dropped and string missing tokens set to NaN."""

    frame = pd.DataFrame(
        {
            "caseid": [1, 1, 2],
            "age": [30, 30, 45],
            "gender": ["Male", "Male", "NA"],
        }
    )
    cleaned = clean_dataframe(frame, id_column="caseid")
    assert cleaned.shape[0] == 2
    assert cleaned["gender"].isna().sum() == 1


def test_infer_feature_types_respects_exclusions() -> None:
    """Type inference should split numeric and categorical columns correctly."""

    frame = pd.DataFrame(
        {
            "caseid": [1, 2, 3],
            "age": [20, 30, 40],
            "income": [50000, 60000, 70000],
            "gender": ["M", "F", "F"],
            "weight": [0.8, 1.1, 0.9],
        }
    )
    numeric, categorical = infer_feature_types(
        frame, id_column="caseid", exclude_columns=["weight"]
    )
    assert numeric == ["age", "income"]
    assert categorical == ["gender"]


def test_build_preprocessing_pipeline_transforms_dataframe() -> None:
    """ColumnTransformer should fit and transform mixed feature frames."""

    frame = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "pid7": [1, 2, 3],
            "gender": ["M", "F", "F"],
        }
    )
    preprocessor = build_preprocessing_pipeline(
        numeric_columns=["age", "pid7"],
        categorical_columns=["gender"],
    )
    matrix = preprocessor.fit_transform(frame)
    assert matrix.shape[0] == 3
    assert matrix.shape[1] >= 4

