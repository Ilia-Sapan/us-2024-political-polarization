"""Configuration management for the polarization project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths used across the project."""

    raw_dir: Path
    processed_dir: Path
    reports_figures_dir: Path
    reports_tables_dir: Path
    artifacts_dir: Path
    raw_data_file: Path | None
    cleaned_data_file: Path
    analysis_data_file: Path
    preprocessor_file: Path
    kmeans_model_file: Path
    reducer_model_file: Path
    metrics_file: Path
    feature_names_file: Path
    cluster_profiles_file: Path
    feature_contrasts_file: Path
    embedding_figure: Path
    k_selection_figure: Path
    summary_report_file: Path


@dataclass(frozen=True)
class DataConfig:
    """Data schema and feature selection settings."""

    id_column: str
    required_columns: list[str]
    exclude_columns: list[str]
    demographic_columns: list[str]


@dataclass(frozen=True)
class ModelingConfig:
    """Modeling hyperparameters and algorithm choices."""

    embedding_method: str
    embedding_components: int
    random_state: int
    k_min: int
    k_max: int
    silhouette_sample_size: int
    top_features_per_cluster: int
    enable_hdbscan: bool
    hdbscan_min_cluster_size: int


@dataclass(frozen=True)
class AppConfig:
    """Root config object."""

    project_root: Path
    config_file: Path
    paths: PathsConfig
    data: DataConfig
    modeling: ModelingConfig


def _as_list(value: Any) -> list[str]:
    """Convert TOML values into a safe string list."""

    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"Expected list, got {type(value)!r}")
    return [str(item) for item in value]


def _resolve_path(project_root: Path, value: str | None) -> Path | None:
    """Resolve an absolute or project-relative path."""

    if value is None or str(value).strip() == "":
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


def load_config(config_file: str | Path | None = None) -> AppConfig:
    """Load project configuration from TOML.

    Parameters
    ----------
    config_file:
        Optional custom TOML path. Defaults to ``config/settings.toml``.
    """

    project_root = Path(__file__).resolve().parents[2]
    resolved_config = (
        Path(config_file).resolve()
        if config_file
        else (project_root / "config" / "settings.toml").resolve()
    )

    config_data: dict[str, Any] = {}
    if resolved_config.exists():
        with resolved_config.open("rb") as handle:
            config_data = tomllib.load(handle)

    path_section = config_data.get("paths", {})
    data_section = config_data.get("data", {})
    modeling_section = config_data.get("modeling", {})

    raw_dir = _resolve_path(project_root, path_section.get("raw_dir", "data/raw"))
    processed_dir = _resolve_path(
        project_root, path_section.get("processed_dir", "data/processed")
    )
    reports_figures_dir = _resolve_path(
        project_root, path_section.get("reports_figures_dir", "reports/figures")
    )
    reports_tables_dir = _resolve_path(
        project_root, path_section.get("reports_tables_dir", "reports/tables")
    )
    raw_data_file = _resolve_path(project_root, path_section.get("raw_data_file", ""))

    assert raw_dir is not None
    assert processed_dir is not None
    assert reports_figures_dir is not None
    assert reports_tables_dir is not None

    artifacts_dir = processed_dir / "artifacts"

    paths = PathsConfig(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        reports_figures_dir=reports_figures_dir,
        reports_tables_dir=reports_tables_dir,
        artifacts_dir=artifacts_dir,
        raw_data_file=raw_data_file,
        cleaned_data_file=processed_dir / "cleaned.parquet",
        analysis_data_file=processed_dir / "analysis.parquet",
        preprocessor_file=artifacts_dir / "preprocessor.joblib",
        kmeans_model_file=artifacts_dir / "kmeans_model.joblib",
        reducer_model_file=artifacts_dir / "reducer.joblib",
        metrics_file=artifacts_dir / "metrics.json",
        feature_names_file=artifacts_dir / "feature_names.json",
        cluster_profiles_file=reports_tables_dir / "cluster_profiles.csv",
        feature_contrasts_file=reports_tables_dir / "cluster_feature_contrasts.csv",
        embedding_figure=reports_figures_dir / "embedding_clusters.png",
        k_selection_figure=reports_figures_dir / "k_selection_silhouette.png",
        summary_report_file=project_root / "reports" / "summary_report.md",
    )

    id_column = str(data_section.get("id_column", "caseid"))
    required_columns = _as_list(data_section.get("required_columns", [id_column]))
    exclude_columns = _as_list(data_section.get("exclude_columns", []))
    demographic_columns = _as_list(
        data_section.get("demographic_columns", ["age", "gender", "educ", "pid7"])
    )

    data = DataConfig(
        id_column=id_column,
        required_columns=required_columns,
        exclude_columns=exclude_columns,
        demographic_columns=demographic_columns,
    )

    modeling = ModelingConfig(
        embedding_method=str(modeling_section.get("embedding_method", "umap")),
        embedding_components=int(modeling_section.get("embedding_components", 2)),
        random_state=int(modeling_section.get("random_state", 42)),
        k_min=int(modeling_section.get("k_min", 2)),
        k_max=int(modeling_section.get("k_max", 12)),
        silhouette_sample_size=int(
            modeling_section.get("silhouette_sample_size", 10_000)
        ),
        top_features_per_cluster=int(
            modeling_section.get("top_features_per_cluster", 10)
        ),
        enable_hdbscan=bool(modeling_section.get("enable_hdbscan", True)),
        hdbscan_min_cluster_size=int(
            modeling_section.get("hdbscan_min_cluster_size", 200)
        ),
    )

    return AppConfig(
        project_root=project_root,
        config_file=resolved_config,
        paths=paths,
        data=data,
        modeling=modeling,
    )

