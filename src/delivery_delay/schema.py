"""Dynamic schema and column-role detection."""

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


DATE_KEYWORDS = (
    "date",
    "time",
    "timestamp",
    "created",
    "updated",
    "shipped",
    "delivered",
    "arrival",
    "expected",
    "actual",
    "deadline",
    "due",
)

TARGET_KEYWORDS = (
    "target",
    "delay",
    "delayed",
    "late",
    "status",
    "delivery_time",
    "lead_time",
    "duration",
    "is_delayed",
)

ID_KEYWORDS = ("id", "uuid", "key", "reference", "ref", "number", "code")


@dataclass
class ColumnSchema:
    """Detected semantic roles for the columns of a DataFrame."""

    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    datetime: List[str] = field(default_factory=list)
    boolean: List[str] = field(default_factory=list)
    id_like: List[str] = field(default_factory=list)
    high_cardinality: List[str] = field(default_factory=list)
    target_candidates: List[str] = field(default_factory=list)
    parseable_datetime: Dict[str, float] = field(default_factory=dict)


def normalize_column_name(name):
    """Normalize a column name for keyword matching."""

    value = str(name).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def _name_contains(name, keywords):
    normalized = normalize_column_name(name)
    return any(keyword in normalized for keyword in keywords)


def _datetime_parse_ratio(series, sample_size=200):
    values = series.dropna().astype(str).head(sample_size)
    if values.empty:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(values, errors="coerce")
    return float(parsed.notna().mean())


def looks_like_datetime(series, column_name, min_parse_ratio=0.85):
    """Detect datetime columns from dtype, name, and parseability."""

    if is_datetime64_any_dtype(series):
        return True, 1.0

    if is_numeric_dtype(series) or is_bool_dtype(series):
        return False, 0.0

    ratio = _datetime_parse_ratio(series)
    has_date_name = _name_contains(column_name, DATE_KEYWORDS)
    if has_date_name and ratio >= 0.5:
        return True, ratio
    if ratio >= min_parse_ratio:
        return True, ratio
    return False, ratio


def looks_like_id(series, column_name):
    """Identify identifier-like columns that usually hurt generalization."""

    normalized = normalize_column_name(column_name)
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_ratio = float(non_null.nunique(dropna=True)) / float(len(non_null))
    name_hint = any(
        normalized == keyword or normalized.endswith("_" + keyword) or keyword + "_" in normalized
        for keyword in ID_KEYWORDS
    )
    return bool(name_hint and unique_ratio > 0.6) or bool(unique_ratio > 0.98 and len(non_null) > 50)


def detect_columns(df):
    """Detect numerical, categorical, datetime, boolean, id, and target columns."""

    schema = ColumnSchema()
    if df is None or df.empty:
        return schema

    for column in df.columns:
        series = df[column]
        is_dt, parse_ratio = looks_like_datetime(series, column)
        if is_dt:
            schema.datetime.append(column)
            schema.parseable_datetime[column] = parse_ratio
        elif is_bool_dtype(series):
            schema.boolean.append(column)
        elif is_numeric_dtype(series):
            schema.numeric.append(column)
        else:
            schema.categorical.append(column)

        if looks_like_id(series, column):
            schema.id_like.append(column)

        non_null = series.dropna()
        if not non_null.empty:
            unique_count = non_null.nunique(dropna=True)
            if unique_count > 50 and unique_count / max(len(non_null), 1) > 0.25:
                schema.high_cardinality.append(column)

        if _name_contains(column, TARGET_KEYWORDS):
            schema.target_candidates.append(column)

    schema.target_candidates = _rank_target_candidates(df, schema.target_candidates)
    return schema


def _rank_target_candidates(df, candidates):
    """Put likely delivery-delay targets first."""

    def score(column):
        normalized = normalize_column_name(column)
        value = 0
        if normalized in ("is_delayed", "delayed", "delay_flag"):
            value += 100
        if "delay" in normalized or "late" in normalized:
            value += 50
        if "status" in normalized:
            value += 25
        if is_numeric_dtype(df[column]) and df[column].nunique(dropna=True) <= 20:
            value += 10
        return value

    unique_candidates = list(dict.fromkeys(candidates))
    return sorted(unique_candidates, key=score, reverse=True)


def infer_problem_type(y, requested="auto"):
    """Infer classification or regression from the target values."""

    if requested in ("classification", "regression"):
        return requested

    clean_y = pd.Series(y).dropna()
    if clean_y.empty:
        return "classification"

    unique_count = clean_y.nunique(dropna=True)
    if not is_numeric_dtype(clean_y):
        return "classification"
    if unique_count <= 20 or unique_count / max(len(clean_y), 1) < 0.02:
        return "classification"
    return "regression"


def safe_column_list(columns, df):
    """Return only columns that exist in df, preserving order."""

    existing = set(df.columns)
    return [column for column in columns if column in existing]


def summarize_schema(schema):
    """Convert a ColumnSchema into a compact table for dashboards."""

    return pd.DataFrame(
        [
            {"role": "Numerical", "count": len(schema.numeric), "columns": ", ".join(schema.numeric[:12])},
            {"role": "Categorical", "count": len(schema.categorical), "columns": ", ".join(schema.categorical[:12])},
            {"role": "Datetime", "count": len(schema.datetime), "columns": ", ".join(schema.datetime[:12])},
            {"role": "Boolean", "count": len(schema.boolean), "columns": ", ".join(schema.boolean[:12])},
            {"role": "Identifier-like", "count": len(schema.id_like), "columns": ", ".join(schema.id_like[:12])},
            {
                "role": "Target candidates",
                "count": len(schema.target_candidates),
                "columns": ", ".join(schema.target_candidates[:12]),
            },
        ]
    )
