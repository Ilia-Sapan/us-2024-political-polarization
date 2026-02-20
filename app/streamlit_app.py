"""Interactive dashboard for political polarization analysis outputs."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# Allow running app before package is installed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polarization.config import load_config  # noqa: E402
from polarization.utils import load_json  # noqa: E402


@st.cache_resource(show_spinner=False)
def get_config() -> Any:
    """Load project config once."""

    return load_config()


@st.cache_data(show_spinner=False)
def load_analysis_data(path: Path) -> pd.DataFrame:
    """Load model output dataframe used in dashboard visualizations."""

    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_cluster_profiles(path: Path) -> pd.DataFrame:
    """Load cluster profile table."""

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_feature_contrasts(path: Path) -> pd.DataFrame:
    """Load top distinguishing feature table."""

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_metrics(path: Path) -> dict[str, Any]:
    """Load metrics artifact."""

    if not path.exists():
        return {}
    return load_json(path)


@st.cache_resource(show_spinner=False)
def load_models(config: Any) -> dict[str, Any]:
    """Load fitted preprocessing and clustering artifacts."""

    models: dict[str, Any] = {}
    if config.paths.preprocessor_file.exists():
        models["preprocessor"] = joblib.load(config.paths.preprocessor_file)
    if config.paths.kmeans_model_file.exists():
        models["kmeans"] = joblib.load(config.paths.kmeans_model_file)
    if config.paths.reducer_model_file.exists():
        models["reducer"] = joblib.load(config.paths.reducer_model_file)
    return models


def _extract_selected_indices(event: Any) -> list[int]:
    """Normalize Streamlit plot selection event into point indices."""

    if event is None:
        return []

    points: list[Any] = []
    if isinstance(event, dict):
        points = event.get("selection", {}).get("points", [])
    else:
        selection = getattr(event, "selection", None)
        if selection is not None:
            points = getattr(selection, "points", [])

    indices: list[int] = []
    for point in points:
        if isinstance(point, dict):
            value = point.get("point_index")
        else:
            value = getattr(point, "point_index", None)
        if value is not None:
            indices.append(int(value))
    return indices


def apply_sidebar_filters(dataframe: pd.DataFrame, filter_columns: list[str]) -> pd.DataFrame:
    """Apply demographic filters from sidebar controls."""

    filtered = dataframe.copy()
    st.sidebar.header("Filters")

    for column in filter_columns:
        if column not in filtered.columns:
            continue

        if pd.api.types.is_numeric_dtype(filtered[column]):
            numeric_series = filtered[column].dropna()
            if numeric_series.empty:
                continue
            min_val = float(numeric_series.min())
            max_val = float(numeric_series.max())
            selected_min, selected_max = st.sidebar.slider(
                label=column,
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
            )
            filtered = filtered[
                filtered[column].between(selected_min, selected_max, inclusive="both")
            ]
        else:
            options = sorted(filtered[column].dropna().astype(str).unique().tolist())
            if not options:
                continue
            selected = st.sidebar.multiselect(
                label=column,
                options=options,
                default=options,
            )
            if selected:
                filtered = filtered[filtered[column].astype(str).isin(selected)]

    return filtered


def main() -> None:
    """Streamlit app entrypoint."""

    st.set_page_config(page_title="US 2024 Political Polarization", layout="wide")
    st.title("US 2024 Political Polarization Dashboard")

    config = get_config()
    models = load_models(config)
    analysis_path = config.paths.analysis_data_file
    if not analysis_path.exists():
        st.error(
            f"Analysis data missing at `{analysis_path}`. "
            "Run `python -m polarization train` first."
        )
        return

    analysis_df = load_analysis_data(analysis_path)
    metrics = load_metrics(config.paths.metrics_file)
    profiles_df = load_cluster_profiles(config.paths.cluster_profiles_file)
    contrasts_df = load_feature_contrasts(config.paths.feature_contrasts_file)

    st.caption(
        f"Rows: {len(analysis_df):,} | Embedding: {metrics.get('embedding_method_used', 'n/a')} | "
        f"K: {metrics.get('selected_k', 'n/a')} | Models loaded: {', '.join(models.keys()) or 'none'}"
    )

    filtered = apply_sidebar_filters(analysis_df, config.data.demographic_columns)
    st.subheader("Embedding Scatter")
    if {"emb_1", "emb_2", "cluster"}.issubset(filtered.columns):
        hover_cols = [c for c in [config.data.id_column, "row_index"] if c in filtered.columns]
        fig = px.scatter(
            filtered,
            x="emb_1",
            y="emb_2",
            color=filtered["cluster"].astype(str),
            opacity=0.75,
            height=620,
            hover_data=hover_cols,
            labels={"color": "cluster"},
            title="Respondents in Embedding Space",
        )
        selection_event = st.plotly_chart(
            fig,
            use_container_width=True,
            key="embedding_plot",
            on_select="rerun",
        )
    else:
        selection_event = None
        st.warning("Embedding columns `emb_1`, `emb_2`, or `cluster` are missing.")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Cluster Sizes (Filtered)")
        size_frame = filtered["cluster"].value_counts().sort_index().rename("n").to_frame()
        st.bar_chart(size_frame)
        st.dataframe(size_frame, use_container_width=True)

    with right:
        st.subheader("Cluster Profiles")
        if profiles_df.empty:
            st.info("No cluster profile table found yet.")
        else:
            st.dataframe(profiles_df, use_container_width=True, hide_index=True)

    st.subheader("Distinguishing Features by Cluster")
    if contrasts_df.empty:
        st.info("No feature contrast table found yet.")
    else:
        available_clusters = sorted(contrasts_df["cluster"].unique().tolist())
        selected_cluster = st.selectbox("Cluster", options=available_clusters)
        display_frame = contrasts_df[contrasts_df["cluster"] == selected_cluster].copy()
        st.dataframe(
            display_frame.sort_values("std_diff", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Selected Respondent Details")
    selected_indices = _extract_selected_indices(selection_event)
    if selected_indices:
        selected_rows = filtered.iloc[selected_indices]
        st.dataframe(selected_rows, use_container_width=True)
    else:
        if "row_index" not in filtered.columns or filtered.empty:
            st.info("No selectable row index available in the filtered data.")
        else:
            min_index = int(filtered["row_index"].min())
            max_index = int(filtered["row_index"].max())
            picked_index = st.number_input(
                "Inspect row_index",
                min_value=min_index,
                max_value=max_index,
                value=min_index,
                step=1,
            )
            row = filtered[filtered["row_index"] == int(picked_index)]
            if row.empty:
                st.info("Selected row index is outside current filter selection.")
            else:
                st.dataframe(row, use_container_width=True)


if __name__ == "__main__":
    main()

