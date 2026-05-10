"""Streamlit caching and lightweight execution helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from .dual_prediction import discover_prediction_tasks
from .eda import build_eda_report
from .explainability import model_feature_importance, shap_importance
from .external_intelligence import evaluate_external_weather_intelligence
from .insights import build_eda_story
from .prediction import load_model_artifact, predict_dataframe
from .schema import detect_columns
from .understanding import analyze_dataset, preprocessing_explanations


def dataframe_cache_key(df, sample_rows=600):
    """Return a stable, low-cost cache key for large uploaded dataframes."""

    if df is None:
        return "none"
    head = df.head(sample_rows)
    tail = df.tail(min(sample_rows, len(df)))
    payload = [
        "shape={}".format(df.shape),
        "columns={}".format(tuple(map(str, df.columns))),
        "dtypes={}".format(tuple(map(str, df.dtypes))),
    ]
    try:
        sample = pd.concat([head, tail], axis=0).drop_duplicates()
        hashed = pd.util.hash_pandas_object(sample.astype(str), index=True).sum()
        payload.append("sample_hash={}".format(int(hashed)))
    except Exception:
        payload.append("sample_hash=unavailable")
    return "|".join(payload)


def artifact_cache_key(artifact):
    """Return a compact cache key for model artifacts and portfolios."""

    if not isinstance(artifact, dict):
        return str(id(artifact))
    if artifact.get("artifact_type") == "dual_prediction_portfolio":
        model_keys = tuple(sorted(artifact.get("models", {}).keys()))
        return "portfolio|{}|{}|{}".format(
            artifact.get("trained_at"),
            artifact.get("recommended_model_key"),
            model_keys,
        )
    return "model|{}|{}|{}|{}".format(
        artifact.get("trained_at"),
        artifact.get("best_model_name"),
        artifact.get("target_column"),
        artifact.get("problem_type"),
    )


HASH_FUNCS = {
    pd.DataFrame: dataframe_cache_key,
    dict: artifact_cache_key,
    Pipeline: lambda pipeline: str(id(pipeline)),
}


@st.cache_data(show_spinner=False, max_entries=12, hash_funcs=HASH_FUNCS)
def cached_detect_columns(df):
    return detect_columns(df)


@st.cache_data(show_spinner=False, max_entries=12, hash_funcs=HASH_FUNCS)
def cached_analyze_dataset(df, target_column=None, problem_type="auto", known_leakage_columns=()):
    return analyze_dataset(
        df,
        target_column=target_column,
        problem_type=problem_type,
        known_leakage_columns=list(known_leakage_columns or ()),
    )


@st.cache_data(show_spinner=False, max_entries=8, hash_funcs=HASH_FUNCS)
def cached_preprocessing_explanations(understanding):
    return preprocessing_explanations(understanding)


@st.cache_data(show_spinner=False, max_entries=8, hash_funcs=HASH_FUNCS)
def cached_build_eda_story(df, target_column=None):
    return build_eda_story(df, target_column=target_column)


@st.cache_data(show_spinner=False, max_entries=6, hash_funcs=HASH_FUNCS)
def cached_build_eda_report(df, target_column=None):
    return build_eda_report(df, target_column=target_column)


@st.cache_data(show_spinner=False, max_entries=8, hash_funcs=HASH_FUNCS)
def cached_discover_prediction_tasks(df):
    return discover_prediction_tasks(df)


@st.cache_data(show_spinner=True, max_entries=3, hash_funcs=HASH_FUNCS)
def cached_external_weather_intelligence(
    df,
    target_column=None,
    problem_type="auto",
    weather_path=None,
    max_rows=30000,
    run_predictive_test=True,
):
    kwargs = {
        "target_column": target_column,
        "problem_type": problem_type,
        "max_rows": max_rows,
        "run_predictive_test": run_predictive_test,
    }
    if weather_path:
        kwargs["weather_path"] = weather_path
    return evaluate_external_weather_intelligence(df, **kwargs)


@st.cache_resource(show_spinner=False, max_entries=4)
def cached_load_model_artifact(path, modified_ns=None):
    return load_model_artifact(Path(path))


def model_file_signature(path):
    """Return a file modification signature for cache invalidation."""

    path = Path(path)
    if not path.exists():
        return None
    stat = path.stat()
    return "{}:{}:{}".format(path.resolve(), stat.st_size, stat.st_mtime_ns)


@st.cache_data(show_spinner=False, max_entries=8, hash_funcs=HASH_FUNCS)
def cached_predict_dataframe(artifact, df, model_key=None):
    return predict_dataframe(artifact, df, model_key=model_key)


@st.cache_data(show_spinner=False, max_entries=12, hash_funcs=HASH_FUNCS)
def cached_model_feature_importance(pipeline, top_n=30):
    return model_feature_importance(pipeline, top_n=top_n)


@st.cache_data(show_spinner=True, max_entries=4, hash_funcs=HASH_FUNCS)
def cached_shap_importance(pipeline, df, max_rows=80, top_n=30):
    return shap_importance(pipeline, df, max_rows=max_rows, top_n=top_n)


def lightweight_table(df, max_rows=300, max_columns=40):
    """Return a bounded dataframe slice for responsive rendering."""

    if df is None:
        return df
    columns = list(df.columns[:max_columns])
    return df.loc[:, columns].head(max_rows)


def estimate_memory_mb(df):
    """Estimate dataframe memory footprint in MB."""

    if df is None:
        return 0.0
    try:
        return float(df.memory_usage(deep=True).sum() / (1024 * 1024))
    except Exception:
        return 0.0


def streamlit_cloud_ready():
    """Best-effort signal for deployment-friendly behavior."""

    return bool(os.environ.get("STREAMLIT_SERVER_PORT") or os.environ.get("STREAMLIT_CLOUD"))
