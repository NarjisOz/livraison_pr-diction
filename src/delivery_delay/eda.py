"""Automated EDA tables and Plotly visualizations."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .cleaning import basic_quality_report
from .schema import detect_columns


def dataset_overview(df):
    """Return one-row dataset overview."""

    report = basic_quality_report(df)
    return pd.DataFrame([report])


def missing_values_table(df, top_n=30):
    """Rank columns by missing-value rate."""

    if df is None or df.empty:
        return pd.DataFrame(columns=["column", "missing_count", "missing_rate"])

    missing = df.isna().sum().sort_values(ascending=False)
    table = pd.DataFrame(
        {
            "column": missing.index,
            "missing_count": missing.values,
            "missing_rate": missing.values / max(len(df), 1),
        }
    )
    return table.head(top_n)


def numeric_profile(df):
    """Profile numerical columns."""

    schema = detect_columns(df)
    if not schema.numeric:
        return pd.DataFrame()
    profile = df[schema.numeric].describe().T.reset_index().rename(columns={"index": "column"})
    profile["missing_rate"] = df[schema.numeric].isna().mean().values
    profile["skewness"] = df[schema.numeric].skew(numeric_only=True).values
    return profile


def categorical_profile(df, top_n=30):
    """Profile categorical columns with cardinality and dominant values."""

    schema = detect_columns(df)
    rows = []
    for column in schema.categorical + schema.boolean:
        counts = df[column].value_counts(dropna=False).head(3)
        rows.append(
            {
                "column": column,
                "unique_values": int(df[column].nunique(dropna=True)),
                "missing_rate": float(df[column].isna().mean()),
                "top_values": ", ".join(["{} ({})".format(index, value) for index, value in counts.items()]),
            }
        )
    return pd.DataFrame(rows).sort_values("unique_values", ascending=False).head(top_n)


def datetime_profile(df):
    """Profile datetime columns and parseable date-like fields."""

    schema = detect_columns(df)
    rows = []
    for column in schema.datetime:
        values = pd.to_datetime(df[column], errors="coerce")
        rows.append(
            {
                "column": column,
                "valid_dates": int(values.notna().sum()),
                "missing_rate": float(values.isna().mean()),
                "min": values.min(),
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def outlier_report(df):
    """Detect outliers using the IQR rule for numerical columns."""

    schema = detect_columns(df)
    rows = []
    for column in schema.numeric:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((values < lower) | (values > upper)).sum())
        rows.append(
            {
                "column": column,
                "outlier_count": count,
                "outlier_rate": count / max(len(values), 1),
                "lower_bound": lower,
                "upper_bound": upper,
            }
        )
    return pd.DataFrame(rows).sort_values("outlier_rate", ascending=False)


def correlation_table(df, target_column=None):
    """Return correlations for numerical columns, optionally with target first."""

    schema = detect_columns(df)
    numeric = [column for column in schema.numeric if column in df.columns]
    if len(numeric) < 2:
        return pd.DataFrame()

    corr = df[numeric].corr(numeric_only=True)
    if target_column in corr.columns:
        return corr[[target_column]].sort_values(target_column, ascending=False).reset_index()
    return corr


def plot_missing_values(df):
    table = missing_values_table(df)
    table = table[table["missing_count"] > 0]
    if table.empty:
        return go.Figure()
    return px.bar(table, x="column", y="missing_rate", title="Missing values by column")


def plot_target_distribution(df, target_column):
    if target_column is None or target_column not in df.columns:
        return go.Figure()
    counts = df[target_column].value_counts(dropna=False).reset_index()
    counts.columns = [target_column, "count"]
    return px.bar(counts, x=target_column, y="count", title="Target distribution")


def plot_numeric_distribution(df, column):
    if column is None or column not in df.columns:
        return go.Figure()
    return px.histogram(df, x=column, nbins=40, marginal="box", title="Distribution: {}".format(column))


def plot_categorical_counts(df, column, top_n=20):
    if column is None or column not in df.columns:
        return go.Figure()
    counts = df[column].value_counts(dropna=False).head(top_n).reset_index()
    counts.columns = [column, "count"]
    return px.bar(counts, x=column, y="count", title="Top categories: {}".format(column))


def plot_correlation_heatmap(df):
    schema = detect_columns(df)
    numeric = [column for column in schema.numeric if column in df.columns]
    if len(numeric) < 2:
        return go.Figure()
    corr = df[numeric].corr(numeric_only=True)
    return px.imshow(corr, text_auto=False, aspect="auto", title="Numerical correlation matrix")


def plot_model_comparison(leaderboard):
    if leaderboard is None or leaderboard.empty or "selection_score" not in leaderboard.columns:
        return go.Figure()
    data = leaderboard[leaderboard["status"] == "ok"].copy()
    if data.empty:
        return go.Figure()
    return px.bar(data, x="model", y="selection_score", title="Model comparison")


def build_eda_report(df, target_column=None):
    """Collect all EDA tables in a dictionary."""

    return {
        "overview": dataset_overview(df),
        "missing_values": missing_values_table(df),
        "numeric_profile": numeric_profile(df),
        "categorical_profile": categorical_profile(df),
        "datetime_profile": datetime_profile(df),
        "outliers": outlier_report(df),
        "correlation": correlation_table(df, target_column=target_column),
    }
