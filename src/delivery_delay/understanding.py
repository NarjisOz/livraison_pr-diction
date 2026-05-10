"""Dataset understanding, governance, compatibility, and preprocessing planning."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .schema import detect_columns, normalize_column_name


LOGISTICS_KEYWORDS = (
    "delivery",
    "deliver",
    "shipment",
    "ship",
    "order",
    "route",
    "origin",
    "destination",
    "supplier",
    "carrier",
    "logistics",
    "warehouse",
    "dispatch",
    "transport",
    "freight",
    "delay",
    "late",
    "expected",
    "actual",
    "arrival",
    "city",
    "region",
)

SENSITIVE_KEYWORDS = (
    "email",
    "phone",
    "mobile",
    "address",
    "passport",
    "national_id",
    "cin",
    "ssn",
    "credit_card",
    "card_number",
)


@dataclass
class ColumnUnderstanding:
    column: str
    semantic_type: str
    missing_rate: float
    unique_count: int
    unique_ratio: float
    suspicious_reasons: List[str] = field(default_factory=list)
    recommended_action: str = "use"
    imputation_strategy: Optional[str] = None
    encoding_strategy: Optional[str] = None
    outlier_strategy: Optional[str] = None
    skewness: Optional[float] = None
    outlier_rate: Optional[float] = None
    correlation_with_target: Optional[float] = None


@dataclass
class DatasetUnderstanding:
    rows: int
    columns: int
    target_column: Optional[str]
    problem_type: str
    quality_score: float
    compatibility_score: float
    governance_score: float
    semantic_coverage: float
    missing_score: float
    duplicate_rate: float
    suspicious_columns: List[Dict[str, Any]]
    leakage_warnings: List[Dict[str, Any]]
    governance_warnings: List[Dict[str, Any]]
    date_relationships: List[Dict[str, Any]]
    preprocessing_plan: Dict[str, Any]
    columns_report: List[ColumnUnderstanding]
    recommendations: List[str]


def analyze_dataset(
    df,
    target_column=None,
    problem_type="auto",
    known_leakage_columns=None,
    high_missing_threshold=0.85,
    low_cardinality_threshold=25,
    high_cardinality_threshold=50,
):
    """Deeply inspect a logistics dataset before preprocessing decisions are made."""

    known_leakage_columns = known_leakage_columns or []
    schema = detect_columns(df)
    rows, columns = df.shape
    target_series = df[target_column] if target_column in df.columns else None
    y_numeric = _safe_numeric_target(target_series)
    duplicate_rate = float(df.duplicated().mean()) if rows else 0.0

    date_relationships = _date_relationships(df, schema.datetime)
    leakage_warnings = _leakage_warnings(df, target_column, known_leakage_columns, y_numeric)
    governance_warnings = _governance_warnings(df)

    column_reports = []
    for column in df.columns:
        report = _analyze_column(
            df,
            column,
            schema,
            y_numeric,
            high_missing_threshold=high_missing_threshold,
            low_cardinality_threshold=low_cardinality_threshold,
            high_cardinality_threshold=high_cardinality_threshold,
        )
        if column == target_column:
            report.recommended_action = "target"
        column_reports.append(report)

    preprocessing_plan = _build_preprocessing_plan(
        column_reports,
        target_column=target_column,
        known_leakage_columns=known_leakage_columns,
    )
    suspicious_columns = _collect_suspicious_columns(column_reports)
    quality_score, missing_score = _quality_score(df, column_reports, duplicate_rate)
    compatibility_score, semantic_coverage = _compatibility_score(df, column_reports, schema, target_column)
    governance_score = _governance_score(governance_warnings, rows)
    recommendations = _dataset_recommendations(
        quality_score=quality_score,
        compatibility_score=compatibility_score,
        governance_score=governance_score,
        duplicate_rate=duplicate_rate,
        suspicious_columns=suspicious_columns,
        leakage_warnings=leakage_warnings,
        date_relationships=date_relationships,
    )

    return DatasetUnderstanding(
        rows=rows,
        columns=columns,
        target_column=target_column,
        problem_type=problem_type,
        quality_score=quality_score,
        compatibility_score=compatibility_score,
        governance_score=governance_score,
        semantic_coverage=semantic_coverage,
        missing_score=missing_score,
        duplicate_rate=duplicate_rate,
        suspicious_columns=suspicious_columns,
        leakage_warnings=leakage_warnings,
        governance_warnings=governance_warnings,
        date_relationships=date_relationships,
        preprocessing_plan=preprocessing_plan,
        columns_report=column_reports,
        recommendations=recommendations,
    )


def understanding_to_dict(understanding):
    """Return a JSON-serializable understanding report."""

    data = asdict(understanding)
    for column in data["columns_report"]:
        for key in ("skewness", "outlier_rate", "correlation_with_target"):
            if column.get(key) is not None and not np.isfinite(column[key]):
                column[key] = None
    return data


def columns_report_table(understanding):
    """Create a compact table for Streamlit and report downloads."""

    rows = []
    for report in understanding.columns_report:
        rows.append(
            {
                "column": report.column,
                "semantic_type": report.semantic_type,
                "missing_rate": report.missing_rate,
                "unique_count": report.unique_count,
                "unique_ratio": report.unique_ratio,
                "skewness": report.skewness,
                "outlier_rate": report.outlier_rate,
                "correlation_with_target": report.correlation_with_target,
                "recommended_action": report.recommended_action,
                "imputation_strategy": report.imputation_strategy,
                "encoding_strategy": report.encoding_strategy,
                "outlier_strategy": report.outlier_strategy,
                "suspicious_reasons": "; ".join(report.suspicious_reasons),
            }
        )
    return pd.DataFrame(rows)


def plan_summary_table(preprocessing_plan):
    """Convert preprocessing decisions into a readable table."""

    rows = []
    for action, columns in preprocessing_plan.get("drop_columns", {}).items():
        for column in columns:
            rows.append({"column": column, "decision": "drop", "reason": action})

    for strategy, columns in preprocessing_plan.get("numeric_imputers", {}).items():
        for column in columns:
            rows.append({"column": column, "decision": "numeric imputation", "reason": strategy})

    for strategy, columns in preprocessing_plan.get("categorical_encoders", {}).items():
        for column in columns:
            rows.append({"column": column, "decision": "categorical encoding", "reason": strategy})

    for strategy, columns in preprocessing_plan.get("outlier_handlers", {}).items():
        for column in columns:
            rows.append({"column": column, "decision": "outlier handling", "reason": strategy})

    return pd.DataFrame(rows)


def preprocessing_explanations(understanding):
    """Explain every major preprocessing decision in plain language."""

    rows = []
    target_column = understanding.target_column
    leakage_columns = {
        item.get("column")
        for item in understanding.leakage_warnings
        if isinstance(item, dict) and item.get("column")
    }

    for report in understanding.columns_report:
        column = report.column
        if report.recommended_action == "target" or column == target_column:
            rows.append(
                {
                    "column": column,
                    "decision": "Use as target",
                    "what_was_done": "The column is excluded from model inputs and used as the outcome to predict.",
                    "why_selected": "It was selected as the active target for {}.".format(
                        understanding.problem_type
                    ),
                    "selected_by": "target_selector + targeting.prepare_target",
                    "user_friendly_note": "This prevents the model from seeing the answer during training.",
                }
            )
            continue

        if column in leakage_columns:
            rows.append(
                {
                    "column": column,
                    "decision": "Remove leakage risk",
                    "what_was_done": "The column is treated as unsafe for training inputs.",
                    "why_selected": "Its name or statistics suggest that it may contain post-event outcome information.",
                    "selected_by": "understanding._leakage_warnings",
                    "user_friendly_note": "Leakage can make validation look excellent while failing in real deployment.",
                }
            )

        if report.recommended_action == "drop":
            rows.append(_drop_explanation(report))
            continue

        if report.semantic_type == "datetime":
            rows.append(
                {
                    "column": column,
                    "decision": "Engineer date features",
                    "what_was_done": "The original date is converted into year, month, weekday, quarter, weekend, season, and elapsed-day signals.",
                    "why_selected": "The schema detector found a parseable datetime field.",
                    "selected_by": "schema.looks_like_datetime + DateFeatureEngineer",
                    "user_friendly_note": "This lets the model learn weekly and seasonal delivery behavior without memorizing raw dates.",
                }
            )
            continue

        if report.semantic_type == "numerical":
            rows.append(_numeric_imputation_explanation(report))
            rows.append(
                {
                    "column": column,
                    "decision": "Robust scaling",
                    "what_was_done": "Values are scaled with RobustScaler after imputation.",
                    "why_selected": "Logistics numeric fields often include extreme costs, distances, weights, or durations.",
                    "selected_by": "preprocessing._numeric_transformers",
                    "user_friendly_note": "Robust scaling reduces the influence of unusually large values.",
                }
            )
            if report.outlier_strategy == "iqr_capping":
                rows.append(
                    {
                        "column": column,
                        "decision": "IQR outlier capping",
                        "what_was_done": "Extreme values are clipped to the training-set IQR bounds.",
                        "why_selected": "The estimated outlier rate is {:.1%}, above the 3% rule threshold.".format(
                            report.outlier_rate or 0.0
                        ),
                        "selected_by": "understanding._iqr_outlier_rate + IQRCapper",
                        "user_friendly_note": "The model still sees high values, but one extreme record cannot dominate the learning process.",
                    }
                )
            continue

        if report.semantic_type in ("categorical", "boolean"):
            rows.append(_categorical_explanation(report))

    return pd.DataFrame(rows)


def preprocessing_explanations_markdown(understanding):
    """Build a concise downloadable preprocessing explanation report."""

    table = preprocessing_explanations(understanding)
    lines = [
        "# Explainable Preprocessing Report",
        "",
        "- Target column: {}".format(understanding.target_column or "not selected"),
        "- Problem type: {}".format(understanding.problem_type),
        "- Quality score: {}/100".format(understanding.quality_score),
        "- Compatibility score: {}/100".format(understanding.compatibility_score),
        "",
    ]
    for _, row in table.iterrows():
        lines.extend(
            [
                "## {}".format(row["column"]),
                "",
                "- Decision: {}".format(row["decision"]),
                "- What was done: {}".format(row["what_was_done"]),
                "- Why selected: {}".format(row["why_selected"]),
                "- Algorithm or rule: {}".format(row["selected_by"]),
                "- Note: {}".format(row["user_friendly_note"]),
                "",
            ]
        )
    return "\n".join(lines)


def _drop_explanation(report):
    reasons = set(report.suspicious_reasons or [])
    if "high_missing_rate" in reasons:
        why = "The missing rate is {:.1%}, meeting the high-missing removal rule.".format(
            report.missing_rate
        )
        rule = "missing_rate >= 85%"
        note = "A mostly empty field usually adds noise and false confidence."
    elif "constant_or_single_value" in reasons:
        why = "The column has {} unique non-missing value(s), so it cannot separate outcomes.".format(
            report.unique_count
        )
        rule = "unique_count <= 1"
        note = "A constant field gives the model no useful distinction."
    elif "identifier_like" in reasons:
        why = "The column behaves like an identifier with high uniqueness."
        rule = "schema.looks_like_id"
        note = "Identifiers often cause memorization instead of general delivery intelligence."
    else:
        why = "The column was marked as unsuitable for model inputs."
        rule = "DatasetUnderstanding column action"
        note = "The platform removes fields that are unlikely to generalize safely."
    return {
        "column": report.column,
        "decision": "Drop column",
        "what_was_done": "The field is removed from the modeling feature set.",
        "why_selected": why,
        "selected_by": rule,
        "user_friendly_note": note,
    }


def _numeric_imputation_explanation(report):
    strategy = report.imputation_strategy or "median"
    if strategy == "mean":
        why = (
            "Mean imputation was selected because the distribution is close to symmetric "
            "and the estimated outlier rate is below 3%."
        )
    else:
        why = (
            "Median imputation was selected because the field is skewed, outlier-prone, "
            "or missing enough that a robust center is safer."
        )
    return {
        "column": report.column,
        "decision": "{} numeric imputation".format(strategy.title()),
        "what_was_done": "Missing numeric values are replaced with the training-set {}.".format(strategy),
        "why_selected": "{} Skewness={}, outlier_rate={}.".format(
            why,
            _format_optional_float(report.skewness),
            _format_optional_percent(report.outlier_rate),
        ),
        "selected_by": "understanding._is_normalish + SimpleImputer(strategy='{}')".format(strategy),
        "user_friendly_note": "The imputation rule is learned on training data only and reused consistently for prediction.",
    }


def _categorical_explanation(report):
    encoding = report.encoding_strategy or "one_hot"
    if encoding == "target_encoding":
        why = (
            "Target encoding was selected because the column has high cardinality "
            "or a high unique-value ratio."
        )
        note = "This keeps the feature space compact while fitting the encoding only on the training fold."
        algorithm = "TargetMeanEncoder with smoothing"
    else:
        why = "One-hot encoding was selected because the column is boolean or has manageable cardinality."
        note = "Each common category becomes an explicit signal that is easy to inspect."
        algorithm = "OneHotEncoder(handle_unknown='infrequent_if_exist' when available)"
    imputation = report.imputation_strategy or "most_frequent"
    return {
        "column": report.column,
        "decision": "{} categorical encoding".format(encoding.replace("_", "-").title()),
        "what_was_done": "Missing categories use '{}' handling, then categories are encoded for the model.".format(
            imputation
        ),
        "why_selected": "{} unique_count={}, unique_ratio={:.1%}.".format(
            why,
            report.unique_count,
            report.unique_ratio,
        ),
        "selected_by": "cardinality thresholds + {}".format(algorithm),
        "user_friendly_note": note,
    }


def _format_optional_float(value):
    if value is None:
        return "n/a"
    return "{:.3f}".format(value)


def _format_optional_percent(value):
    if value is None:
        return "n/a"
    return "{:.1%}".format(value)


def _analyze_column(
    df,
    column,
    schema,
    y_numeric,
    high_missing_threshold,
    low_cardinality_threshold,
    high_cardinality_threshold,
):
    series = df[column]
    missing_rate = float(series.isna().mean())
    unique_count = int(series.nunique(dropna=True))
    unique_ratio = float(unique_count / max(series.notna().sum(), 1))
    semantic_type = _semantic_type(column, schema)
    suspicious = []
    action = "use"
    imputation = None
    encoding = None
    outlier_strategy = None
    skewness = None
    outlier_rate = None
    correlation = None

    if missing_rate >= high_missing_threshold:
        action = "drop"
        suspicious.append("high_missing_rate")
    elif unique_count <= 1:
        action = "drop"
        suspicious.append("constant_or_single_value")
    elif column in schema.id_like:
        action = "drop"
        suspicious.append("identifier_like")
    elif column in schema.datetime:
        action = "feature_engineer"
    elif semantic_type == "numerical":
        values = pd.to_numeric(series, errors="coerce")
        skewness = _safe_float(values.skew())
        outlier_rate = _iqr_outlier_rate(values)
        imputation = "mean" if _is_normalish(skewness, outlier_rate) and missing_rate < 0.2 else "median"
        if outlier_rate >= 0.03:
            outlier_strategy = "iqr_capping"
        if y_numeric is not None and values.notna().sum() > 3:
            correlation = _safe_abs_corr(values, y_numeric)
            if correlation is not None and correlation >= 0.98:
                suspicious.append("near_perfect_target_correlation")
    elif semantic_type in ("categorical", "boolean"):
        imputation = "most_frequent" if missing_rate < 0.2 else "missing_label"
        if semantic_type == "boolean" or unique_count <= low_cardinality_threshold:
            encoding = "one_hot"
        elif unique_count >= high_cardinality_threshold or unique_ratio >= 0.2:
            encoding = "target_encoding"
            suspicious.append("high_cardinality")
        else:
            encoding = "one_hot"

    if missing_rate > 0.4 and action != "drop":
        suspicious.append("substantial_missing_rate")

    return ColumnUnderstanding(
        column=column,
        semantic_type=semantic_type,
        missing_rate=missing_rate,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        suspicious_reasons=suspicious,
        recommended_action=action,
        imputation_strategy=imputation,
        encoding_strategy=encoding,
        outlier_strategy=outlier_strategy,
        skewness=skewness,
        outlier_rate=outlier_rate,
        correlation_with_target=correlation,
    )


def _semantic_type(column, schema):
    if column in schema.datetime:
        return "datetime"
    if column in schema.boolean:
        return "boolean"
    if column in schema.numeric:
        return "numerical"
    if column in schema.categorical:
        return "categorical"
    return "unknown"


def _build_preprocessing_plan(column_reports, target_column, known_leakage_columns):
    plan = {
        "drop_columns": {
            "high_missing_or_constant_or_id": [],
            "target_or_leakage": [],
        },
        "numeric_imputers": {"mean": [], "median": []},
        "categorical_encoders": {"one_hot": [], "target_encoding": []},
        "outlier_handlers": {"iqr_capping": []},
        "datetime_columns": [],
        "feature_engineering": {
            "date_parts": [],
            "duration_pairs": [],
            "seasonality": [],
            "risk_indicators": [],
        },
    }
    leakage_set = set(known_leakage_columns or [])
    if target_column:
        leakage_set.add(target_column)

    for report in column_reports:
        column = report.column
        if column in leakage_set or report.recommended_action == "target":
            plan["drop_columns"]["target_or_leakage"].append(column)
            continue
        if report.recommended_action == "drop":
            plan["drop_columns"]["high_missing_or_constant_or_id"].append(column)
            continue
        if report.semantic_type == "datetime":
            plan["datetime_columns"].append(column)
            plan["feature_engineering"]["date_parts"].append(column)
            plan["feature_engineering"]["seasonality"].append(column)
            continue
        if report.semantic_type == "numerical":
            plan["numeric_imputers"][report.imputation_strategy or "median"].append(column)
            if report.outlier_strategy:
                plan["outlier_handlers"][report.outlier_strategy].append(column)
        elif report.semantic_type in ("categorical", "boolean"):
            plan["categorical_encoders"][report.encoding_strategy or "one_hot"].append(column)

    plan["feature_engineering"]["duration_pairs"] = _duration_pairs(plan["datetime_columns"])
    plan["feature_engineering"]["risk_indicators"] = _risk_indicator_columns(column_reports)
    return plan


def _date_relationships(df, datetime_columns):
    relationships = []
    parsed = {column: pd.to_datetime(df[column], errors="coerce") for column in datetime_columns}
    for idx, start_col in enumerate(datetime_columns):
        for end_col in datetime_columns[idx + 1 :]:
            start = parsed[start_col]
            end = parsed[end_col]
            duration = (end - start).dt.total_seconds() / 86400.0
            valid = duration.dropna()
            if valid.empty:
                continue
            relationships.append(
                {
                    "start_column": start_col,
                    "end_column": end_col,
                    "valid_pairs": int(valid.shape[0]),
                    "median_days": float(valid.median()),
                    "negative_rate": float((valid < 0).mean()),
                    "p95_days": float(valid.quantile(0.95)),
                }
            )
    return sorted(relationships, key=lambda item: abs(item["median_days"]))[:12]


def _duration_pairs(datetime_columns):
    pairs = []
    for start in datetime_columns:
        start_name = normalize_column_name(start)
        for end in datetime_columns:
            if start == end:
                continue
            end_name = normalize_column_name(end)
            if _looks_like_start(start_name) and _looks_like_end(end_name):
                pairs.append({"start": start, "end": end})
    return pairs[:8]


def _risk_indicator_columns(column_reports):
    indicators = []
    for report in column_reports:
        normalized = normalize_column_name(report.column)
        if any(token in normalized for token in ("cost", "distance", "quantity", "weight", "volume", "priority")):
            indicators.append(report.column)
    return indicators[:12]


def _leakage_warnings(df, target_column, known_leakage_columns, y_numeric):
    warnings = []
    leakage_set = set(known_leakage_columns or [])
    for column in df.columns:
        normalized = normalize_column_name(column)
        reasons = []
        if column in leakage_set:
            reasons.append("known_target_source_or_outcome_column")
        if target_column and column != target_column and normalize_column_name(target_column) in normalized:
            reasons.append("name_contains_target")
        if any(token in normalized for token in ("actual_delivery", "delivered_at", "completed_at", "delivery_status")):
            reasons.append("possible_post_event_information")
        if y_numeric is not None and column != target_column and is_numeric_dtype(df[column]):
            corr = _safe_abs_corr(pd.to_numeric(df[column], errors="coerce"), y_numeric)
            if corr is not None and corr >= 0.98:
                reasons.append("near_perfect_target_correlation")
        if reasons:
            warnings.append({"column": column, "reasons": reasons})
    return warnings


def _governance_warnings(df):
    warnings = []
    for column in df.columns:
        normalized = normalize_column_name(column)
        reasons = [keyword for keyword in SENSITIVE_KEYWORDS if keyword in normalized]
        if reasons:
            warnings.append(
                {
                    "column": column,
                    "risk": "possible_sensitive_or_personal_data",
                    "matched_terms": reasons,
                }
            )
    return warnings


def _collect_suspicious_columns(column_reports):
    return [
        {"column": report.column, "reasons": report.suspicious_reasons}
        for report in column_reports
        if report.suspicious_reasons
    ]


def _quality_score(df, column_reports, duplicate_rate):
    if df.empty:
        return 0.0, 0.0
    mean_missing = float(np.mean([report.missing_rate for report in column_reports])) if column_reports else 1.0
    high_missing_share = float(np.mean([report.missing_rate >= 0.5 for report in column_reports])) if column_reports else 1.0
    constant_share = float(
        np.mean(["constant_or_single_value" in report.suspicious_reasons for report in column_reports])
    ) if column_reports else 1.0
    missing_score = max(0.0, 1.0 - mean_missing)
    score = 100.0 * (
        0.45 * missing_score
        + 0.25 * max(0.0, 1.0 - duplicate_rate)
        + 0.20 * max(0.0, 1.0 - high_missing_share)
        + 0.10 * max(0.0, 1.0 - constant_share)
    )
    return round(float(np.clip(score, 0, 100)), 2), round(float(np.clip(missing_score * 100, 0, 100)), 2)


def _compatibility_score(df, column_reports, schema, target_column):
    if df.empty or not column_reports:
        return 0.0, 0.0
    semantic_hits = sum(_is_logistics_column(report.column) for report in column_reports)
    semantic_coverage = semantic_hits / max(len(column_reports), 1)
    has_target = 1.0 if target_column else 0.0
    has_date = 1.0 if schema.datetime else 0.0
    has_category = 1.0 if schema.categorical else 0.0
    has_numeric = 1.0 if schema.numeric else 0.0
    row_score = min(len(df) / 1000.0, 1.0)
    score = 100.0 * (
        0.30 * semantic_coverage
        + 0.25 * has_target
        + 0.15 * has_date
        + 0.10 * has_category
        + 0.10 * has_numeric
        + 0.10 * row_score
    )
    return round(float(np.clip(score, 0, 100)), 2), round(float(semantic_coverage * 100), 2)


def _governance_score(governance_warnings, rows):
    if not governance_warnings:
        return 100.0
    penalty = min(len(governance_warnings) * 12, 70)
    if rows < 30:
        penalty += 10
    return round(float(max(0.0, 100.0 - penalty)), 2)


def _dataset_recommendations(
    quality_score,
    compatibility_score,
    governance_score,
    duplicate_rate,
    suspicious_columns,
    leakage_warnings,
    date_relationships,
):
    messages = []
    if compatibility_score < 50:
        messages.append("Compatibility is low: confirm that the uploaded file is a logistics or delivery dataset.")
    elif compatibility_score < 75:
        messages.append("Compatibility is moderate: the model can proceed, but review target and date columns carefully.")
    else:
        messages.append("Compatibility is strong: the file contains useful logistics signals for prediction.")

    if quality_score < 60:
        messages.append("Data quality needs attention before trusting model results.")
    if governance_score < 100:
        messages.append("Potential sensitive columns were detected; remove them before public deployment or sharing.")
    if duplicate_rate > 0.02:
        messages.append("Duplicate rows are present and should be removed during training.")
    if leakage_warnings:
        messages.append("Potential leakage columns were detected and should not be used as model inputs.")
    if suspicious_columns:
        messages.append("Some columns look suspicious because of missingness, identifiers, cardinality, or target correlation.")
    if any(item["negative_rate"] > 0.05 for item in date_relationships):
        messages.append("Some date relationships contain negative durations; check date quality and business chronology.")
    return messages


def _safe_numeric_target(series):
    if series is None:
        return None
    if is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    codes, _ = pd.factorize(series)
    encoded = pd.Series(codes, index=series.index, dtype="float")
    encoded[series.isna()] = np.nan
    return encoded


def _safe_abs_corr(values, y_numeric):
    aligned = pd.concat([values, y_numeric], axis=1).dropna()
    if aligned.shape[0] < 5 or aligned.iloc[:, 0].nunique() <= 1 or aligned.iloc[:, 1].nunique() <= 1:
        return None
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return _safe_float(abs(corr))


def _iqr_outlier_rate(values):
    clean = values.dropna()
    if clean.empty:
        return None
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return float(((clean < lower) | (clean > upper)).mean())


def _is_normalish(skewness, outlier_rate):
    if skewness is None:
        return False
    return abs(skewness) < 0.75 and (outlier_rate or 0.0) < 0.03


def _safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value


def _is_logistics_column(column):
    normalized = normalize_column_name(column)
    return any(keyword in normalized for keyword in LOGISTICS_KEYWORDS)


def _looks_like_start(name):
    return any(token in name for token in ("order", "ship", "start", "created", "pickup", "dispatch", "sent"))


def _looks_like_end(name):
    return any(token in name for token in ("expected", "actual", "deliver", "arrival", "due", "end", "received"))
