"""CLI for the US 2024 political polarization project."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

from .cleaning import build_preprocessing_pipeline, clean_dataframe, infer_feature_types
from .config import AppConfig, load_config
from .features import fit_transform_features
from .io import detect_raw_data_file, load_survey_data, read_parquet, save_dataframe_parquet
from .modeling import (
    compute_silhouette,
    fit_kmeans,
    fit_optional_hdbscan,
    reduce_dimensions,
    select_best_kmeans_k,
)
from .profiling import build_cluster_profile_table, compute_feature_contrasts
from .utils import configure_logging, dump_json, ensure_directories, load_json, write_markdown
from .viz import plot_embedding, plot_k_selection

LOGGER = logging.getLogger(__name__)


def _ensure_project_directories(config: AppConfig) -> None:
    """Ensure required output directories exist."""

    ensure_directories(
        [
            config.paths.raw_dir,
            config.paths.processed_dir,
            config.paths.artifacts_dir,
            config.paths.reports_figures_dir,
            config.paths.reports_tables_dir,
        ]
    )


def _resolve_input_file(config: AppConfig, cli_input: str | None) -> Path:
    """Resolve the raw input dataset path from CLI or config defaults."""

    if cli_input:
        return Path(cli_input).resolve()
    if config.paths.raw_data_file is not None:
        return config.paths.raw_data_file
    return detect_raw_data_file(config.paths.raw_dir)


def run_preprocess(config: AppConfig, cli_input: str | None = None) -> None:
    """Execute data loading, validation, and cleaning."""

    _ensure_project_directories(config)
    input_file = _resolve_input_file(config, cli_input)
    raw_df = load_survey_data(input_file, required_columns=config.data.required_columns)
    cleaned_df = clean_dataframe(raw_df, id_column=config.data.id_column)
    save_dataframe_parquet(cleaned_df, config.paths.cleaned_data_file)

    LOGGER.info(
        "Preprocess complete. Input rows=%d, cleaned rows=%d, columns=%d",
        len(raw_df),
        len(cleaned_df),
        cleaned_df.shape[1],
    )


def run_train(config: AppConfig) -> None:
    """Train clustering pipeline and generate analysis artifacts."""

    _ensure_project_directories(config)
    dataframe = read_parquet(config.paths.cleaned_data_file)

    numeric_columns, categorical_columns = infer_feature_types(
        dataframe=dataframe,
        id_column=config.data.id_column,
        exclude_columns=config.data.exclude_columns,
    )
    LOGGER.info(
        "Detected %d numeric and %d categorical features.",
        len(numeric_columns),
        len(categorical_columns),
    )

    preprocessor = build_preprocessing_pipeline(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    matrix, fitted_preprocessor, feature_names = fit_transform_features(
        dataframe=dataframe,
        preprocessor=preprocessor,
    )
    LOGGER.info("Feature matrix shape after preprocessing: %s", matrix.shape)

    selection = select_best_kmeans_k(
        matrix=matrix,
        k_min=config.modeling.k_min,
        k_max=config.modeling.k_max,
        random_state=config.modeling.random_state,
        sample_size=config.modeling.silhouette_sample_size,
    )
    kmeans_model, cluster_labels = fit_kmeans(
        matrix=matrix,
        n_clusters=selection.best_k,
        random_state=config.modeling.random_state,
    )
    silhouette = compute_silhouette(
        matrix=matrix,
        labels=cluster_labels,
        sample_size=config.modeling.silhouette_sample_size,
        random_state=config.modeling.random_state,
    )

    embedding = reduce_dimensions(
        matrix=matrix,
        n_components=config.modeling.embedding_components,
        method=config.modeling.embedding_method,
        random_state=config.modeling.random_state,
    )

    hdbscan_labels = None
    if config.modeling.enable_hdbscan:
        hdbscan_labels, _ = fit_optional_hdbscan(
            embedding=embedding.coordinates,
            min_cluster_size=config.modeling.hdbscan_min_cluster_size,
        )
        if hdbscan_labels is None:
            LOGGER.info("HDBSCAN dependency unavailable. Skipping alternative clustering.")
        else:
            LOGGER.info("HDBSCAN completed on embedding coordinates.")

    analysis_data = pd.DataFrame({"row_index": np.arange(len(dataframe)), "cluster": cluster_labels})
    if config.data.id_column in dataframe.columns:
        analysis_data[config.data.id_column] = dataframe[config.data.id_column].values

    for component_idx in range(embedding.coordinates.shape[1]):
        analysis_data[f"emb_{component_idx + 1}"] = embedding.coordinates[:, component_idx]

    for demographic_column in config.data.demographic_columns:
        if demographic_column in dataframe.columns:
            analysis_data[demographic_column] = dataframe[demographic_column].values

    if hdbscan_labels is not None:
        analysis_data["hdbscan_cluster"] = hdbscan_labels

    save_dataframe_parquet(analysis_data, config.paths.analysis_data_file)

    feature_contrasts = compute_feature_contrasts(
        matrix=matrix,
        labels=cluster_labels,
        feature_names=feature_names,
        top_n=config.modeling.top_features_per_cluster,
    )
    feature_contrasts.to_csv(config.paths.feature_contrasts_file, index=False)

    cluster_profiles = build_cluster_profile_table(
        dataframe=dataframe,
        labels=cluster_labels,
        demographic_columns=config.data.demographic_columns,
    )
    cluster_profiles.to_csv(config.paths.cluster_profiles_file, index=False)

    plot_embedding(analysis_data, output_path=config.paths.embedding_figure)
    plot_k_selection(selection.scores, output_path=config.paths.k_selection_figure)

    joblib.dump(fitted_preprocessor, config.paths.preprocessor_file)
    joblib.dump(kmeans_model, config.paths.kmeans_model_file)
    joblib.dump(embedding.reducer, config.paths.reducer_model_file)
    dump_json({"feature_names": feature_names}, config.paths.feature_names_file)

    metrics = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_rows": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_count": len(categorical_columns),
        "selected_k": int(selection.best_k),
        "silhouette_score": silhouette,
        "k_selection_scores": {str(k): v for k, v in selection.scores.items()},
        "embedding_method_requested": config.modeling.embedding_method,
        "embedding_method_used": embedding.method,
        "embedding_components": int(embedding.coordinates.shape[1]),
        "hdbscan_ran": hdbscan_labels is not None,
    }
    dump_json(metrics, config.paths.metrics_file)

    LOGGER.info(
        "Training complete. Selected K=%d, silhouette=%s, artifacts saved to %s",
        selection.best_k,
        silhouette,
        config.paths.artifacts_dir,
    )


def _markdown_cluster_table(cluster_profiles: pd.DataFrame) -> str:
    """Create a markdown table for cluster sizes."""

    if cluster_profiles.empty:
        return "_No cluster profile data available._"

    headers = ["cluster", "n", "pct"]
    lines = ["| cluster | n | pct |", "|---:|---:|---:|"]
    for _, row in cluster_profiles[headers].iterrows():
        lines.append(
            f"| {int(row['cluster'])} | {int(row['n'])} | {float(row['pct']):.4f} |"
        )
    return "\n".join(lines)


def run_report(config: AppConfig) -> None:
    """Generate a markdown summary report from saved artifacts."""

    if not config.paths.metrics_file.exists():
        raise FileNotFoundError(
            f"Metrics file missing: {config.paths.metrics_file}. Run train before report."
        )

    metrics = load_json(config.paths.metrics_file)
    cluster_profiles = (
        pd.read_csv(config.paths.cluster_profiles_file)
        if config.paths.cluster_profiles_file.exists()
        else pd.DataFrame(columns=["cluster", "n", "pct"])
    )

    report = [
        "# US 2024 Political Polarization Summary Report",
        "",
        f"- Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"- Rows analyzed: {metrics.get('n_rows', 'n/a')}",
        f"- Feature space width: {metrics.get('n_features', 'n/a')}",
        f"- Selected K (KMeans): {metrics.get('selected_k', 'n/a')}",
        f"- Silhouette score: {metrics.get('silhouette_score', 'n/a')}",
        f"- Embedding method used: {metrics.get('embedding_method_used', 'n/a')}",
        "",
        "## Cluster Sizes",
        "",
        _markdown_cluster_table(cluster_profiles),
        "",
        "## Output Artifacts",
        "",
        f"- Embedding figure: `{config.paths.embedding_figure}`",
        f"- K selection figure: `{config.paths.k_selection_figure}`",
        f"- Cluster profiles CSV: `{config.paths.cluster_profiles_file}`",
        f"- Feature contrasts CSV: `{config.paths.feature_contrasts_file}`",
        f"- Analysis parquet: `{config.paths.analysis_data_file}`",
    ]
    write_markdown(config.paths.summary_report_file, "\n".join(report))
    LOGGER.info("Report generated at %s", config.paths.summary_report_file)


def run_app(config: AppConfig) -> None:
    """Run the Streamlit app."""

    app_file = config.project_root / "app" / "streamlit_app.py"
    if not app_file.exists():
        raise FileNotFoundError(f"Streamlit app file not found: {app_file}")

    command = [sys.executable, "-m", "streamlit", "run", str(app_file)]
    LOGGER.info("Launching Streamlit: %s", " ".join(command))
    subprocess.run(command, cwd=str(config.project_root), check=False)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        prog="polarization",
        description="US 2024 political polarization pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config TOML file (default: config/settings.toml).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Load, validate, and clean raw data.")
    preprocess.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional path to raw CSV/DTA dataset.",
    )

    subparsers.add_parser("train", help="Fit preprocessing, embedding, and clustering models.")
    subparsers.add_parser("report", help="Generate markdown summary report.")
    subparsers.add_parser("run-app", help="Launch Streamlit dashboard.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""

    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    LOGGER.info("Using config file: %s", config.config_file)

    if args.command == "preprocess":
        run_preprocess(config=config, cli_input=args.input)
    elif args.command == "train":
        run_train(config=config)
    elif args.command == "report":
        run_report(config=config)
    elif args.command == "run-app":
        run_app(config=config)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

