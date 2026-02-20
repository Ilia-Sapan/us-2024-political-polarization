"""Feature matrix generation utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted


def fit_transform_features(
    dataframe: pd.DataFrame, preprocessor: ColumnTransformer
) -> tuple[Any, ColumnTransformer, list[str]]:
    """Fit the preprocessing pipeline and transform a dataframe."""

    matrix = preprocessor.fit_transform(dataframe)
    feature_names = get_feature_names(preprocessor)
    return matrix, preprocessor, feature_names


def transform_features(dataframe: pd.DataFrame, preprocessor: ColumnTransformer) -> Any:
    """Transform a dataframe using an already-fitted preprocessor."""

    check_is_fitted(preprocessor)
    return preprocessor.transform(dataframe)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract transformed feature names from a fitted ColumnTransformer."""

    check_is_fitted(preprocessor)
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError as exc:
        raise RuntimeError(
            "Unable to extract feature names from preprocessor."
        ) from exc

