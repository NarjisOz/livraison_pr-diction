"""Target creation and leakage control."""

import re
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .exceptions import TargetDetectionError
from .schema import detect_columns, infer_problem_type, normalize_column_name


POSITIVE_STATUS_PATTERN = re.compile(r"delay|delayed|late|retard", flags=re.IGNORECASE)
NEGATIVE_STATUS_PATTERN = re.compile(
    r"on.?time|delivered|early|success|complete|completed|a.?temps|ponctuel",
    flags=re.IGNORECASE,
)


@dataclass
class PreparedTarget:
    X: pd.DataFrame
    y: pd.Series
    target_column: str
    problem_type: str
    derived_targets: List[str] = field(default_factory=list)
    source_columns: Dict[str, List[str]] = field(default_factory=dict)
    leakage_columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def add_derived_delivery_targets(df):
    """Create reusable delivery-delay targets when source columns exist."""

    data = df.copy()
    source_columns = {}

    status_column = _find_status_column(data)
    if status_column:
        mapped = _map_status_to_delay(data[status_column])
        if mapped.notna().sum() > 0:
            data["is_delayed"] = mapped
            source_columns["is_delayed"] = [status_column]

    expected_column, actual_column = _find_expected_actual_pair(data)
    if expected_column and actual_column:
        expected = pd.to_datetime(data[expected_column], errors="coerce")
        actual = pd.to_datetime(data[actual_column], errors="coerce")
        data["delay_days"] = (actual - expected).dt.total_seconds() / 86400.0
        data["is_delayed_from_dates"] = (data["delay_days"] > 0).astype(float)
        data.loc[data["delay_days"].isna(), "is_delayed_from_dates"] = np.nan
        source_columns["delay_days"] = [expected_column, actual_column]
        source_columns["is_delayed_from_dates"] = [expected_column, actual_column]

    return data, source_columns


def prepare_target(df, target_column=None, problem_type="auto", exclude_columns=None):
    """Split a cleaned DataFrame into X/y and protect against target leakage."""

    data, derived_sources = add_derived_delivery_targets(df)
    target_column = target_column or _choose_default_target(data)
    if not target_column or target_column not in data.columns:
        raise TargetDetectionError(
            "No usable target found. Select a target column or include delivery status / expected and actual dates."
        )

    y = data[target_column]
    warnings = []

    if target_column not in derived_sources and _is_status_like(target_column):
        mapped = _map_status_to_delay(y)
        if mapped.notna().sum() > 0 and mapped.nunique(dropna=True) <= 2:
            y = mapped
            warnings.append("Status labels were converted to a binary delayed/not-delayed target.")

    inferred_problem_type = infer_problem_type(y, requested=problem_type)
    valid_mask = y.notna()
    y = y.loc[valid_mask]
    X = data.loc[valid_mask].drop(columns=[target_column], errors="ignore")

    leakage_columns = _build_leakage_columns(data, target_column, derived_sources)
    if exclude_columns:
        leakage_columns.extend(exclude_columns)
    leakage_columns = list(dict.fromkeys([column for column in leakage_columns if column in X.columns]))
    X = X.drop(columns=leakage_columns, errors="ignore")

    if y.empty:
        raise TargetDetectionError("The selected target contains no non-missing values.")

    return PreparedTarget(
        X=X,
        y=y,
        target_column=target_column,
        problem_type=inferred_problem_type,
        derived_targets=list(derived_sources.keys()),
        source_columns=derived_sources,
        leakage_columns=leakage_columns,
        warnings=warnings,
    )


def available_targets(df):
    """Return an ordered list of explicit and derived target candidates."""

    data, _ = add_derived_delivery_targets(df)
    schema = detect_columns(data)
    preferred = [column for column in ("is_delayed", "is_delayed_from_dates", "delay_days") if column in data.columns]
    combined = preferred + schema.target_candidates
    return list(dict.fromkeys([column for column in combined if column in data.columns]))


def _choose_default_target(df):
    for column in ("is_delayed", "is_delayed_from_dates", "delay_days"):
        if column in df.columns and df[column].notna().sum() > 0:
            return column
    candidates = detect_columns(df).target_candidates
    return candidates[0] if candidates else None


def _find_status_column(df):
    for column in df.columns:
        normalized = normalize_column_name(column)
        if "status" in normalized or normalized in ("state", "delivery_state"):
            sample = df[column].dropna().astype(str).head(500)
            if sample.str.contains(POSITIVE_STATUS_PATTERN).any():
                return column
    return None


def _is_status_like(column):
    normalized = normalize_column_name(column)
    return "status" in normalized or normalized in ("state", "delivery_state")


def _map_status_to_delay(series):
    values = series.astype(str)
    mapped = pd.Series(np.nan, index=series.index, dtype="float")
    mapped.loc[values.str.contains(POSITIVE_STATUS_PATTERN, na=False)] = 1.0
    mapped.loc[values.str.contains(NEGATIVE_STATUS_PATTERN, na=False)] = 0.0
    return mapped


def _find_expected_actual_pair(df):
    schema = detect_columns(df)
    expected_candidates = []
    actual_candidates = []
    for column in schema.datetime:
        normalized = normalize_column_name(column)
        if any(token in normalized for token in ("expected", "planned", "promise", "due", "estimated")):
            expected_candidates.append(column)
        if any(token in normalized for token in ("actual", "delivered", "arrival", "completed", "received")):
            actual_candidates.append(column)

    if expected_candidates and actual_candidates:
        return expected_candidates[0], actual_candidates[0]
    return None, None


def _build_leakage_columns(df, target_column, derived_sources):
    leakage = set()
    leakage.add(target_column)

    for derived_target, sources in derived_sources.items():
        if derived_target == target_column:
            leakage.update(_leaky_source_columns(sources))
        elif derived_target in df.columns:
            leakage.add(derived_target)

    target_name = normalize_column_name(target_column)
    for column in df.columns:
        normalized = normalize_column_name(column)
        if column == target_column:
            continue
        if normalized == target_name or target_name in normalized:
            leakage.add(column)
        if "actual" in normalized and ("delivery" in normalized or "delivered" in normalized):
            leakage.add(column)
        if target_name.startswith("is_delayed") and "status" in normalized:
            leakage.add(column)

    return list(leakage)


def _leaky_source_columns(sources):
    leaky_tokens = ("actual", "delivered", "arrival", "completed", "received", "status", "state")
    leaky = []
    for column in sources:
        normalized = normalize_column_name(column)
        if any(token in normalized for token in leaky_tokens):
            leaky.append(column)
    return leaky
