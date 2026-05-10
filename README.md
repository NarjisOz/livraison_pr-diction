# Logistics AI Prediction Platform

Professional master's thesis project upgraded into an intelligent logistics analytics and decision-support platform with Machine Learning, Streamlit, explainable preprocessing, external-data validation, and operational dashboards.

Recommended GitHub repository name: `logistics-ai-prediction-platform`.

## Objective

The application transforms raw logistics datasets into an intelligent decision-support workflow. It profiles the dataset first, scores data quality and logistics compatibility, checks governance and leakage risks, then chooses preprocessing and modeling strategies from the observed data characteristics.

- upload CSV, Excel, or Parquet datasets once and reuse the active session dataset across every page,
- automatically inspect data quality and column roles,
- generate explainable EDA storytelling for missingness, distributions, anomalies, operational hotspots, and logistics behavior,
- analyze semantic column types, cardinality, distributions, missingness, correlations, date relationships, suspicious columns, leakage risk, governance risk, and deployment compatibility,
- clean data and derive delivery-delay targets when possible,
- detect numerical, categorical, boolean, datetime, identifier-like, and target columns dynamically,
- choose and explain cleaning, imputation, encoding, outlier handling, and feature engineering strategies adaptively,
- engineer datetime, duration, seasonality, route, volume, cost, risk, and logistics-specific derived features,
- train a dual prediction portfolio: classification for delayed/not-delayed risk and regression for delivery duration or delay days,
- compare compatible prediction systems dynamically instead of forcing one modeling style onto every dataset,
- evaluate the external Nigeria cities weather dataset by city joins, latitude/longitude relationships, weather influence, route distance, and validation improvement before recommending integration,
- validate train/test contamination and target leakage risk,
- save the best model as one reusable artifact,
- generate SHAP/importance explanations, operational recommendations, and dashboards for non-technical users,
- download cleaned datasets, predictions, dashboard exports, model reports, preprocessing explanations, and insight summaries,
- switch between fast Business Mode and detailed Advanced/Data Science Mode,
- deploy for free on Streamlit Community Cloud.

## Architecture

1. `src/delivery_delay/` is the production Python package. The legacy `modules/` folder is retained only for notebook traceability; the Streamlit app and CLI use the package.
2. The backend uses a standard `src` layout with `pyproject.toml` package metadata. Streamlit Cloud installs it from `requirements.txt` via `-e .`, so imports such as `from delivery_delay.training import ...` work without `sys.path` hacks.
3. Data loading is centralized in `data_io.py` so Streamlit uploads and command-line scripts behave consistently.
4. Schema detection lives in `schema.py`; column roles are inferred from dtypes, names, parseability, cardinality, and identifier patterns instead of fixed column lists.
5. Target preparation is isolated in `targeting.py`. It can use a selected target or derive `is_delayed`, `delay_days`, and `is_delayed_from_dates` from delivery status and expected/actual dates.
6. Leakage prevention happens before training. Target source columns, status columns, actual delivery columns, and derived target siblings are removed from model features.
7. Feature engineering is implemented as sklearn transformers in `features.py`, so the same adaptive transformations run during training and prediction.
8. `dual_prediction.py` trains classification and regression systems when compatible, stores them in a model portfolio, and selects a recommended default.
9. `external_intelligence.py` tests weather, geography, and route-distance features before recommending integration.
10. Business Mode is the default UX path. Advanced/Data Science Mode unlocks diagnostics, leaderboards, feature importance, and SHAP on demand.

## Project Structure

```text
.
|-- app.py
|-- pages/
|   |-- 1_Data_Explorer.py
|   |-- 2_Preprocessing_Intelligence.py
|   |-- 3_Model_Training.py
|   |-- 4_Predictions.py
|   `-- 5_Explainability.py
|-- src/
|   `-- delivery_delay/
|       |-- __init__.py
|       |-- cleaning.py
|       |-- data_io.py
|       |-- dual_prediction.py
|       |-- eda.py
|       |-- evaluation.py
|       |-- exceptions.py
|       |-- explainability.py
|       |-- external_intelligence.py
|       |-- features.py
|       |-- insights.py
|       |-- modeling.py
|       |-- performance.py
|       |-- prediction.py
|       |-- prediction_dashboard.py
|       |-- preprocessing.py
|       |-- recommendations.py
|       |-- reporting.py
|       |-- schema.py
|       |-- streamlit_app.py
|       |-- targeting.py
|       |-- training.py
|       `-- understanding.py
|-- scripts/
|   |-- train.py
|   `-- predict_cli.py
|-- sample_data/
|   `-- delivery_sample.csv
|-- data/raw/
|   |-- nigeria_cities_weather_data.csv
|   `-- test_sample.csv
|-- .streamlit/config.toml
|-- .gitignore
|-- pyproject.toml
|-- requirements.txt
`-- runtime.txt
```

Generated artifacts, local models, raw parquet data, cache folders, virtual environments, and reports are intentionally excluded from Git.

## Local Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Command-Line Training

```bash
python scripts/train.py sample_data/delivery_sample.csv --target is_delayed --problem-type classification
```

If no target is provided, the system attempts to derive one from `delivery_status`, `expected_delivery_date`, and `actual_delivery_date`.

## Command-Line Prediction

```bash
python scripts/predict_cli.py sample_data/delivery_sample.csv --model artifacts/best_model.joblib --output data/predictions.csv
```

## Streamlit Community Cloud Deployment

1. Create a GitHub repository named `logistics-ai-prediction-platform`.
2. Push the project root to GitHub.
3. Confirm these files are committed: `app.py`, `pages/`, `src/`, `scripts/`, `pyproject.toml`, `requirements.txt`, `runtime.txt`, `.streamlit/config.toml`, `sample_data/`, and `data/raw/nigeria_cities_weather_data.csv`.
4. Confirm these files are not committed: `.venv/`, `.venv311/`, `artifacts/`, `reports/`, `models/*.pkl`, cache folders, generated predictions, and the large raw parquet dataset.
5. In Streamlit Community Cloud, create a new app from the GitHub repository.
6. Set the main file path to `app.py`.
7. Use Python 3.11 to match `runtime.txt`.
8. Deploy. Streamlit will run `pip install -r requirements.txt`, install the local `delivery_delay` package through `-e .`, and then launch `app.py`.

## Deployment Readiness

- `src/delivery_delay` is a real Python package with `__init__.py`.
- `pyproject.toml` declares the `src` package layout with setuptools.
- `requirements.txt` installs packaging tools, the local package, and runtime dependencies.
- `app.py`, pages, and scripts import `delivery_delay` normally without `sys.path` edits.
- Paths are project-relative through `delivery_delay.config`; no local Windows paths are required.
- CLI training, CLI prediction, pytest, Ruff, and package import smoke checks pass.

## Notes for the Thesis

The project explicitly separates data ingestion, schema detection, target engineering, leakage control, preprocessing, training, evaluation, explainability, prediction, and UI concerns. This makes the implementation easier to test, explain, and extend than a notebook-centered workflow.
