"""Data cleaning routines shared by EDA, training, and prediction."""

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from .schema import detect_columns


def clean_dataframe(df, drop_duplicates=True):
    """Apply conservative, model-safe cleaning without domain assumptions."""

    if df is None:
        raise ValueError("A DataFrame is required.")

    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    cleaned = cleaned.replace(r"^\s*$", np.nan, regex=True)

    text_columns = [
        column
        for column in cleaned.columns
        if is_object_dtype(cleaned[column])
        or is_string_dtype(cleaned[column])
        or isinstance(cleaned[column].dtype, pd.CategoricalDtype)
    ]
    for column in text_columns:
        cleaned[column] = cleaned[column].map(lambda value: value.strip() if isinstance(value, str) else value)

    schema = detect_columns(cleaned)
    for column in schema.datetime:
        cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned


def basic_quality_report(df):
    """Return high-level quality indicators for non-technical dashboards."""

    if df is None or df.empty:
        return {
            "rows": 0,
            "columns": 0,
            "missing_cells": 0,
            "missing_rate": 0.0,
            "duplicate_rows": 0,
        }

    total_cells = max(df.shape[0] * df.shape[1], 1)
    missing_cells = int(df.isna().sum().sum())
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": missing_cells,
        "missing_rate": float(missing_cells / total_cells),
        "duplicate_rows": int(df.duplicated().sum()),
    }
