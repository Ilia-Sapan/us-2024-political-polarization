"""Cleaning and preprocessing helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MISSING_STRING_TOKENS = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "missing",
    "refused",
    "dk",
    "don't know",
    "dont know",
}


def clean_dataframe(dataframe: pd.DataFrame, id_column: str | None = None) -> pd.DataFrame:
    """Apply conservative cleaning steps to raw survey data."""

    cleaned = dataframe.copy()

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        as_text = cleaned[column].astype("string").str.strip()
        lowered = as_text.str.lower()
        cleaned[column] = as_text.mask(lowered.isin(MISSING_STRING_TOKENS), np.nan)

    if id_column and id_column in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=[id_column], keep="first")
    else:
        cleaned = cleaned.drop_duplicates(keep="first")

    # sklearn imputers handle np.nan consistently across object/string columns.
    cleaned = cleaned.replace({pd.NA: np.nan})

    return cleaned


def infer_feature_types(
    dataframe: pd.DataFrame,
    id_column: str | None = None,
    exclude_columns: Sequence[str] | None = None,
    max_categorical_levels: int = 120,
    max_categorical_ratio: float = 0.2,
) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature columns for preprocessing."""

    excluded = set(exclude_columns or [])
    if id_column:
        excluded.add(id_column)

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    n_rows = max(len(dataframe), 1)

    for column in dataframe.columns:
        if column in excluded:
            continue
        if dataframe[column].isna().all():
            continue

        if pd.api.types.is_numeric_dtype(dataframe[column]):
            numeric_cols.append(column)
        elif (
            pd.api.types.is_object_dtype(dataframe[column])
            or pd.api.types.is_categorical_dtype(dataframe[column])
            or pd.api.types.is_bool_dtype(dataframe[column])
            or pd.api.types.is_string_dtype(dataframe[column])
        ):
            unique_count = int(dataframe[column].nunique(dropna=True))
            unique_ratio = unique_count / n_rows
            ratio_ok = True if n_rows < 100 else unique_ratio <= max_categorical_ratio
            if unique_count <= max_categorical_levels and ratio_ok:
                categorical_cols.append(column)

    return numeric_cols, categorical_cols


def build_preprocessing_pipeline(
    numeric_columns: Sequence[str], categorical_columns: Sequence[str]
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for mixed-type survey features."""

    if not numeric_columns and not categorical_columns:
        raise ValueError(
            "No usable features found for preprocessing. "
            "Check config.data.exclude_columns and input schema."
        )

    transformers: list[tuple[str, Pipeline, Sequence[str]]] = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, list(numeric_columns)))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, list(categorical_columns)))

    return ColumnTransformer(transformers=transformers, remainder="drop")
