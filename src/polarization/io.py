"""Data loading and saving utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def detect_raw_data_file(raw_dir: Path) -> Path:
    """Find the first supported raw dataset file in ``raw_dir``."""

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory does not exist: {raw_dir}. "
            "Create it and place a CSV or DTA file inside."
        )

    candidates = sorted(
        [path for path in raw_dir.iterdir() if path.is_file() and path.suffix.lower() in {".csv", ".dta"}]
    )
    if not candidates:
        raise FileNotFoundError(
            f"No CSV or DTA files found in {raw_dir}. "
            "Place your CCES dataset in data/raw/ or set paths.raw_data_file in config."
        )
    return candidates[0]


def validate_required_columns(dataframe: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """Validate required columns and raise a helpful error when missing."""

    if not required_columns:
        return

    missing = sorted(set(required_columns) - set(dataframe.columns))
    if missing:
        sample_cols = list(dataframe.columns[:25])
        raise ValueError(
            "Dataset schema validation failed. Missing required columns: "
            f"{missing}. Available columns (first 25): {sample_cols}"
        )


def load_survey_data(file_path: Path, required_columns: Sequence[str]) -> pd.DataFrame:
    """Load survey data from CSV or Stata and validate schema."""

    if not file_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {file_path}")

    suffix = file_path.suffix.lower()
    LOGGER.info("Loading dataset from %s", file_path)
    if suffix == ".csv":
        dataframe = pd.read_csv(file_path, low_memory=False)
    elif suffix == ".dta":
        dataframe = pd.read_stata(file_path, convert_categoricals=False)
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. Supported formats are .csv and .dta."
        )

    validate_required_columns(dataframe, required_columns)
    LOGGER.info("Loaded dataframe shape: %s", dataframe.shape)
    return dataframe


def save_dataframe_parquet(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save a dataframe to parquet using snappy compression."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False, compression="snappy")
    LOGGER.info("Saved parquet file: %s", output_path)


def read_parquet(input_path: Path) -> pd.DataFrame:
    """Read a parquet dataset from disk."""

    if not input_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {input_path}")
    dataframe = pd.read_parquet(input_path)
    LOGGER.info("Loaded parquet file %s with shape %s", input_path, dataframe.shape)
    return dataframe

