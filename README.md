# us-2024-political-polarization

Production-quality, reproducible Python project for analyzing political polarization in the U.S. 2024 electorate using CCES survey responses. The pipeline supports local raw data ingestion, cleaning, feature preprocessing, dimensionality reduction, clustering, interpretable cluster profiling, static report artifacts, and an interactive Streamlit dashboard.

## Dataset Citation

- Source: Cooperative Election Study (CCES) 2024 release.
- This project does not download data automatically.

## Quickstart

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -e .
# Optional extras:
# pip install -e .[dev,umap,hdbscan]
```

3. Place your dataset locally in `data/raw/` (CSV or Stata `.dta`), or set `paths.raw_data_file` in `config/settings.toml`.

4. Run the full pipeline:

```bash
python -m polarization preprocess
python -m polarization train
python -m polarization report
```

5. Run the Streamlit app:

```bash
python -m polarization run-app
# or: streamlit run app/streamlit_app.py
```

## CLI Commands

- `python -m polarization preprocess [--input path/to/file.csv]`
- `python -m polarization train`
- `python -m polarization report`
- `python -m polarization run-app`

## Configuration

Project settings live in `config/settings.toml`, including:

- Data locations and output paths
- Required schema columns
- ID/demographic columns
- KMeans search range
- Embedding method (`umap` preferred, `pca` fallback)
- Optional HDBSCAN settings

## Folder Structure

```text
us-2024-political-polarization/
  README.md
  pyproject.toml
  .gitignore
  LICENSE
  config/
    settings.toml
  data/
    raw/        (user-provided files, not tracked)
    processed/  (generated artifacts)
  notebooks/
    01_eda.ipynb
    02_modeling.ipynb
  src/
    polarization/
      __init__.py
      __main__.py
      config.py
      io.py
      cleaning.py
      features.py
      modeling.py
      profiling.py
      viz.py
      utils.py
  app/
    streamlit_app.py
  reports/
    figures/    (generated)
    tables/     (generated)
  tests/
    test_cleaning.py
    test_features.py
    test_modeling.py
```

## Reproducible Outputs

Running `train` + `report` generates:

- `data/processed/cleaned.parquet`
- `data/processed/analysis.parquet`
- `data/processed/artifacts/*.joblib` and `metrics.json`
- `reports/figures/embedding_clusters.png`
- `reports/figures/k_selection_silhouette.png`
- `reports/tables/cluster_profiles.csv`
- `reports/tables/cluster_feature_contrasts.csv`
- `reports/summary_report.md`
- `reports/insights.md`

## Screenshots Placeholders

- Embedding clusters: `reports/figures/embedding_clusters.png`
- K selection curve: `reports/figures/k_selection_silhouette.png`
- Dashboard screenshot: save as `reports/figures/dashboard_overview.png`

## Testing

```bash
pytest -q
```

## Data Ethics

- Privacy: Treat individual-level survey responses as sensitive; never publish personally identifying records.
- Aggregation: Prefer reporting cluster-level aggregates and anonymized summaries.
- Bias: Survey nonresponse, weighting, and measurement choices can bias inferred polarization structure.
- Interpretability limits: Clusters are statistical segments, not fixed political identities.
- Reproducibility: Keep config and preprocessing assumptions explicit; document any manual feature exclusions.
