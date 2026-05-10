"""External weather and geography intelligence for logistics prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .cleaning import clean_dataframe
from .config import RANDOM_STATE, WEATHER_DATA_PATH
from .evaluation import evaluate_classifier, evaluate_regressor
from .explainability import model_feature_importance
from .features import DateFeatureEngineer, LogisticsFeatureEngineer
from .preprocessing import build_preprocessor
from .schema import detect_columns, infer_problem_type, normalize_column_name
from .targeting import add_derived_delivery_targets, prepare_target


WEATHER_NUMERIC_COLUMNS = (
    "latitude",
    "longitude",
    "temp",
    "temp_min",
    "temp_max",
    "pressure",
    "humidity",
    "sea_level",
    "ground_level",
    "wind_speed",
    "wind_degree",
    "timezone",
    "cloud",
    "population",
)

WEATHER_CATEGORICAL_COLUMNS = ("country", "region", "description")


@dataclass
class ExternalIntelligenceResult:
    """Result of deciding whether weather/geography should join the model."""

    available: bool
    recommended: bool
    decision: str
    reasons: List[str]
    weather_path: str
    city_columns: List[str] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)
    match_rate: float = 0.0
    matched_rows: int = 0
    total_rows: int = 0
    weather_summary: Dict[str, Any] = field(default_factory=dict)
    geo_summary: Dict[str, Any] = field(default_factory=dict)
    weather_influence: List[Dict[str, Any]] = field(default_factory=list)
    predictive_test: Dict[str, Any] = field(default_factory=dict)
    enriched_data: Optional[pd.DataFrame] = None


def evaluate_external_weather_intelligence(
    df,
    target_column=None,
    problem_type="auto",
    weather_path=WEATHER_DATA_PATH,
    max_rows=30000,
    run_predictive_test=True,
):
    """Analyze the Nigeria weather dataset and decide whether to use it."""

    total_rows = int(len(df)) if df is not None else 0
    path = Path(weather_path)
    if df is None or total_rows == 0:
        return ExternalIntelligenceResult(
            available=False,
            recommended=False,
            decision="not_evaluated",
            reasons=["No active logistics dataset is available."],
            weather_path=str(path),
            total_rows=total_rows,
        )

    if not path.exists():
        return ExternalIntelligenceResult(
            available=False,
            recommended=False,
            decision="not_available",
            reasons=["External weather dataset was not found at '{}'.".format(path)],
            weather_path=str(path),
            total_rows=total_rows,
        )

    weather = _load_weather(path)
    if weather.empty or "city" not in weather.columns:
        return ExternalIntelligenceResult(
            available=False,
            recommended=False,
            decision="not_available",
            reasons=["External weather data is empty or has no city column."],
            weather_path=str(path),
            total_rows=total_rows,
        )

    cleaned = clean_dataframe(df, drop_duplicates=False)
    enriched, join_report = enrich_with_weather_features(cleaned, weather)
    weather_summary = analyze_weather_dataset(weather)
    geo_summary = analyze_geography(cleaned, weather, join_report, enriched)
    active_target = _resolve_target(enriched, target_column)
    weather_influence = analyze_weather_influence(enriched, active_target, join_report["feature_columns"])

    predictive_test = {}
    if run_predictive_test and active_target:
        predictive_test = test_weather_predictive_usefulness(
            cleaned,
            enriched,
            target_column=active_target,
            problem_type=problem_type,
            max_rows=max_rows,
        )

    recommended, decision, reasons = _integration_decision(
        join_report=join_report,
        predictive_test=predictive_test,
        weather_influence=weather_influence,
    )

    return ExternalIntelligenceResult(
        available=True,
        recommended=recommended,
        decision=decision,
        reasons=reasons,
        weather_path=str(path),
        city_columns=join_report["city_columns"],
        feature_columns=join_report["feature_columns"],
        match_rate=join_report["match_rate"],
        matched_rows=join_report["matched_rows"],
        total_rows=join_report["total_rows"],
        weather_summary=weather_summary,
        geo_summary=geo_summary,
        weather_influence=weather_influence,
        predictive_test=predictive_test,
        enriched_data=enriched,
    )


def external_result_to_dict(result):
    """Serialize an external intelligence result without embedding the dataframe."""

    data = asdict(result)
    data.pop("enriched_data", None)
    return data


def external_result_markdown(result):
    """Create a downloadable weather/geography intelligence report."""

    lines = [
        "# External Weather and Geography Intelligence",
        "",
        "- Decision: {}".format(result.decision),
        "- Recommended for modeling: {}".format("yes" if result.recommended else "no"),
        "- Match rate: {:.1%}".format(result.match_rate),
        "- Matched rows: {:,} of {:,}".format(result.matched_rows, result.total_rows),
        "- City columns evaluated: {}".format(", ".join(result.city_columns) or "none"),
        "",
        "## Why",
        "",
    ]
    for reason in result.reasons:
        lines.append("- {}".format(reason))

    if result.predictive_test:
        lines.extend(["", "## Predictive Test", ""])
        for key, value in result.predictive_test.items():
            lines.append("- {}: {}".format(key, value))
        if result.predictive_test.get("weather_feature_importance"):
            lines.extend(["", "## Weather Feature Importance", ""])
            for item in result.predictive_test["weather_feature_importance"][:8]:
                lines.append("- {}: {}".format(item["feature"], item["importance"]))

    if result.weather_influence:
        lines.extend(["", "## Weather Influence", ""])
        for item in result.weather_influence[:8]:
            lines.append(
                "- {}: correlation={} ({})".format(
                    item.get("feature"),
                    item.get("correlation"),
                    item.get("interpretation"),
                )
            )

    lines.extend(["", "## Geography", ""])
    for key, value in result.geo_summary.items():
        lines.append("- {}: {}".format(key, value))
    return "\n".join(lines)


def enrich_with_weather_features(df, weather):
    """Join aggregated Nigeria weather and coordinate features by detected city fields."""

    lookup = _weather_lookup(weather)
    data = df.copy()
    city_columns = _detect_city_columns(data)
    feature_columns = []
    match_columns = []
    prefix_by_role = {}

    for city_column in city_columns[:3]:
        prefix = "{}_weather".format(normalize_column_name(city_column))
        role = _city_role(city_column)
        if role and role not in prefix_by_role:
            prefix_by_role[role] = prefix

        key_column = "__{}_city_key".format(prefix)
        data[key_column] = data[city_column].map(_normalize_city_value)

        right = lookup.copy()
        rename = {
            column: "{}_{}".format(prefix, column)
            for column in right.columns
            if column != "__weather_city_key"
        }
        right = right.rename(columns=rename)
        right_key = "__{}_join_key".format(prefix)
        right = right.rename(columns={"__weather_city_key": right_key})

        data = data.merge(right, how="left", left_on=key_column, right_on=right_key)
        data = data.drop(columns=[key_column, right_key], errors="ignore")

        prefixed_features = list(rename.values())
        feature_columns.extend(prefixed_features)
        match_column = "{}_match".format(prefix)
        weather_records_column = "{}_weather_records".format(prefix)
        data[match_column] = data[weather_records_column].notna() if weather_records_column in data.columns else False
        feature_columns.append(match_column)
        match_columns.append(match_column)

    if "origin" in prefix_by_role and "destination" in prefix_by_role:
        distance_col = "weather_route_distance_km"
        origin_prefix = prefix_by_role["origin"]
        destination_prefix = prefix_by_role["destination"]
        distance = _haversine_series(
            data.get("{}_latitude".format(origin_prefix)),
            data.get("{}_longitude".format(origin_prefix)),
            data.get("{}_latitude".format(destination_prefix)),
            data.get("{}_longitude".format(destination_prefix)),
        )
        if distance is not None:
            data[distance_col] = distance
            feature_columns.append(distance_col)

    if match_columns:
        matched = data[match_columns].any(axis=1)
        matched_rows = int(matched.sum())
        match_rate = float(matched.mean())
    else:
        matched_rows = 0
        match_rate = 0.0

    report = {
        "city_columns": city_columns,
        "feature_columns": list(dict.fromkeys(feature_columns)),
        "match_columns": match_columns,
        "match_rate": match_rate,
        "matched_rows": matched_rows,
        "total_rows": int(len(data)),
    }
    return data, report


def analyze_weather_dataset(weather):
    """Summarize weather, location, and coverage relationships."""

    summary = {
        "rows": int(len(weather)),
        "cities": int(weather["city"].nunique(dropna=True)) if "city" in weather.columns else 0,
        "regions": int(weather["region"].nunique(dropna=True)) if "region" in weather.columns else 0,
    }
    for column in ("latitude", "longitude", "temp", "humidity", "wind_speed", "cloud", "population"):
        if column in weather.columns:
            values = pd.to_numeric(weather[column], errors="coerce")
            summary["{}_mean".format(column)] = _safe_float(values.mean())
            summary["{}_min".format(column)] = _safe_float(values.min())
            summary["{}_max".format(column)] = _safe_float(values.max())

    lat = pd.to_numeric(weather.get("latitude"), errors="coerce") if "latitude" in weather.columns else None
    lon = pd.to_numeric(weather.get("longitude"), errors="coerce") if "longitude" in weather.columns else None
    if lat is not None and lon is not None:
        summary["latitude_longitude_correlation"] = _safe_float(lat.corr(lon))
    return summary


def analyze_geography(df, weather, join_report, enriched):
    """Explain geographic coverage and distance contribution potential."""

    schema = detect_columns(df)
    geo_columns = [
        column
        for column in schema.numeric + schema.categorical
        if any(token in normalize_column_name(column) for token in ("lat", "lon", "latitude", "longitude", "city", "region"))
    ]
    summary = {
        "logistics_geo_columns": geo_columns[:12],
        "weather_city_match_rate": round(float(join_report.get("match_rate", 0.0)), 4),
        "weather_feature_count": len(join_report.get("feature_columns", [])),
    }

    if "weather_route_distance_km" in enriched.columns:
        distance = pd.to_numeric(enriched["weather_route_distance_km"], errors="coerce")
        summary.update(
            {
                "route_distance_available": True,
                "route_distance_median_km": _safe_float(distance.median()),
                "route_distance_p90_km": _safe_float(distance.quantile(0.90)),
            }
        )
    else:
        summary.update(
            {
                "route_distance_available": False,
                "route_distance_note": (
                    "Origin and destination city columns were not both matched, so route distance could not be computed."
                ),
            }
        )

    if "latitude" in weather.columns and "longitude" in weather.columns:
        summary["weather_latitude_span"] = _safe_float(
            pd.to_numeric(weather["latitude"], errors="coerce").max()
            - pd.to_numeric(weather["latitude"], errors="coerce").min()
        )
        summary["weather_longitude_span"] = _safe_float(
            pd.to_numeric(weather["longitude"], errors="coerce").max()
            - pd.to_numeric(weather["longitude"], errors="coerce").min()
        )
    return summary


def analyze_weather_influence(enriched, target_column, feature_columns, top_n=10):
    """Estimate simple target relationships for added weather/geography features."""

    if not target_column or target_column not in enriched.columns:
        return []

    y = _numeric_target(enriched[target_column])
    if y is None or y.notna().sum() < 10:
        return []

    rows = []
    for column in feature_columns:
        if column not in enriched.columns:
            continue
        values = pd.to_numeric(enriched[column], errors="coerce")
        aligned = pd.concat([values, y], axis=1).dropna()
        if aligned.shape[0] < 10 or aligned.iloc[:, 0].nunique() <= 1:
            continue
        corr = _safe_float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        if corr is None:
            continue
        rows.append(
            {
                "feature": column,
                "correlation": round(corr, 4),
                "absolute_correlation": round(abs(corr), 4),
                "interpretation": _correlation_interpretation(corr),
            }
        )
    return sorted(rows, key=lambda item: item["absolute_correlation"], reverse=True)[:top_n]


def test_weather_predictive_usefulness(
    baseline_df,
    enriched_df,
    target_column,
    problem_type="auto",
    max_rows=30000,
):
    """Compare quick baseline and weather-enhanced model scores."""

    try:
        baseline_score = _quick_prediction_score(
            baseline_df,
            target_column=target_column,
            problem_type=problem_type,
            max_rows=max_rows,
        )
        enriched_score = _quick_prediction_score(
            enriched_df,
            target_column=target_column,
            problem_type=problem_type,
            max_rows=max_rows,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "reason": str(exc),
            "target_column": target_column,
        }

    improvement = enriched_score["selection_score"] - baseline_score["selection_score"]
    relative_improvement = _relative_improvement(baseline_score["selection_score"], improvement)
    return {
        "status": "ok",
        "target_column": target_column,
        "problem_type": baseline_score["problem_type"],
        "metric": baseline_score["metric"],
        "baseline_score": round(float(baseline_score["selection_score"]), 5),
        "weather_score": round(float(enriched_score["selection_score"]), 5),
        "score_delta": round(float(improvement), 5),
        "relative_improvement": relative_improvement,
        "baseline_metrics": baseline_score["metrics"],
        "weather_metrics": enriched_score["metrics"],
        "weather_feature_importance": enriched_score.get("weather_feature_importance", []),
        "weather_importance_share": enriched_score.get("weather_importance_share", 0.0),
    }


def _quick_prediction_score(df, target_column, problem_type, max_rows):
    prepared = prepare_target(
        clean_dataframe(df, drop_duplicates=True),
        target_column=target_column,
        problem_type=problem_type,
    )
    X, y = prepared.X, prepared.y
    if len(X) > max_rows:
        sampled = X.assign(__target__=y).sample(n=max_rows, random_state=RANDOM_STATE)
        X = sampled.drop(columns=["__target__"])
        y = sampled["__target__"]

    inferred = prepared.problem_type
    if inferred == "classification" and pd.Series(y).nunique(dropna=True) < 2:
        raise ValueError("Classification target has fewer than two classes.")
    if len(X) < 30:
        raise ValueError("At least 30 labeled rows are required for weather usefulness testing.")

    stratify = _safe_stratify(y, inferred)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    probe_engineer = DateFeatureEngineer()
    probe_logistics = LogisticsFeatureEngineer()
    probe_base = probe_engineer.fit(X_train).transform(X_train.head(min(len(X_train), 500)))
    probe_X = probe_logistics.fit(probe_base).transform(probe_base)
    preprocessor = build_preprocessor(probe_X, y=pd.Series(y_train).head(len(probe_X)), problem_type=inferred)
    model = (
        RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        if inferred == "classification"
        else RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )
    pipeline = Pipeline(
        steps=[
            ("features", DateFeatureEngineer()),
            ("logistics_features", LogisticsFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    if inferred == "classification":
        metrics, _, _ = evaluate_classifier(pipeline, X_test, y_test)
        metric = "roc_auc" if "roc_auc" in metrics else "f1_weighted"
    else:
        metrics, _, _ = evaluate_regressor(pipeline, X_test, y_test)
        metric = "r2"
    return {
        "problem_type": inferred,
        "metric": metric,
        "selection_score": metrics["selection_score"],
        "metrics": {key: _safe_float(value) for key, value in metrics.items()},
        "weather_feature_importance": _weather_feature_importance(pipeline),
        "weather_importance_share": _weather_importance_share(pipeline),
    }


def _load_weather(path):
    weather = pd.read_csv(path)
    weather.columns = [str(column).strip() for column in weather.columns]
    return weather


def _weather_lookup(weather):
    data = weather.copy()
    data["__weather_city_key"] = data["city"].map(_normalize_city_value)
    numeric = [column for column in WEATHER_NUMERIC_COLUMNS if column in data.columns]
    categorical = [column for column in WEATHER_CATEGORICAL_COLUMNS if column in data.columns]

    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    grouped = data.groupby("__weather_city_key", dropna=True)
    parts = []
    if numeric:
        parts.append(grouped[numeric].mean())
    if categorical:
        parts.append(grouped[categorical].agg(_mode_or_missing))
    lookup = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=grouped.size().index)
    lookup["weather_records"] = grouped.size()
    return lookup.reset_index()


def _detect_city_columns(df):
    schema = detect_columns(df)
    candidates = []
    for column in schema.categorical:
        normalized = normalize_column_name(column)
        if any(token in normalized for token in ("city", "town", "location", "origin", "destination", "dest")):
            sample = df[column].dropna().astype(str)
            if sample.empty:
                continue
            unique_count = sample.nunique(dropna=True)
            if unique_count >= 1:
                priority = _city_column_priority(normalized, unique_count)
                candidates.append((priority, unique_count, column))
    return [column for _, _, column in sorted(candidates)]


def _city_column_priority(normalized, unique_count):
    if "destination" in normalized or normalized.startswith("dest"):
        base = 0
    elif "origin" in normalized:
        base = 1
    elif "city" in normalized:
        base = 2
    else:
        base = 3
    return base + min(unique_count / 100000.0, 0.1)


def _city_role(column):
    normalized = normalize_column_name(column)
    if "origin" in normalized:
        return "origin"
    if "destination" in normalized or normalized.startswith("dest"):
        return "destination"
    return None


def _resolve_target(df, target_column):
    data, _ = add_derived_delivery_targets(df)
    if target_column in data.columns:
        return target_column
    for column in ("is_delayed", "is_delayed_from_dates", "delay_days"):
        if column in data.columns:
            return column
    candidates = detect_columns(data).target_candidates
    return candidates[0] if candidates else None


def _integration_decision(join_report, predictive_test, weather_influence):
    reasons = []
    match_rate = join_report.get("match_rate", 0.0)
    matched_rows = join_report.get("matched_rows", 0)

    if not join_report.get("city_columns"):
        return False, "reject_no_join_key", [
            "No city, origin, or destination column was found, so weather data cannot be joined safely."
        ]

    if match_rate < 0.20 or matched_rows < 30:
        return False, "reject_low_match_rate", [
            "Only {:.1%} of records matched Nigeria weather cities.".format(match_rate),
            "The platform avoids adding sparse external features because they can increase noise.",
        ]

    if predictive_test.get("status") == "ok":
        delta = float(predictive_test.get("score_delta", 0.0))
        threshold = 0.015 if predictive_test.get("problem_type") == "classification" else 0.02
        if delta >= threshold:
            relative = predictive_test.get("relative_improvement")
            improvement_text = (
                " ({:.1%} relative improvement)".format(relative)
                if relative is not None
                else ""
            )
            reasons.append(
                "The weather-enhanced quick test improved {} by {:.4f}{}.".format(
                    predictive_test.get("metric", "selection score"),
                    delta,
                    improvement_text,
                )
            )
            if predictive_test.get("weather_feature_importance"):
                top_importance = predictive_test["weather_feature_importance"][0]
                reasons.append(
                    "The most useful external feature was '{}' with importance {:.4f}.".format(
                        top_importance["feature"],
                        top_importance["importance"],
                    )
                )
            if weather_influence:
                top = weather_influence[0]
                reasons.append(
                    "The strongest weather/geography relationship was '{}' with correlation {}.".format(
                        top["feature"], top["correlation"]
                    )
                )
            reasons.append("The join coverage is acceptable at {:.1%} of records.".format(match_rate))
            return True, "recommend_integration", reasons

        reasons.append(
            "The weather-enhanced quick test changed the validation score by only {:.4f}.".format(delta)
        )
        if predictive_test.get("weather_importance_share") is not None:
            reasons.append(
                "Weather/geography features represented {:.1%} of quick-model importance.".format(
                    predictive_test.get("weather_importance_share", 0.0)
                )
            )
        reasons.append("That is below the automatic integration threshold, so external features remain optional.")
        if weather_influence:
            reasons.append("Simple correlations exist, but they were not enough to improve validation quality.")
        return False, "reject_no_predictive_gain", reasons

    if weather_influence and weather_influence[0]["absolute_correlation"] >= 0.08:
        reasons.append(
            "Weather/geography features show a measurable relationship with the target, but model testing was unavailable."
        )
        reasons.append("Use the enriched dataset experimentally and validate before deployment.")
        return False, "review_manually", reasons

    reason = predictive_test.get("reason") if predictive_test else None
    if reason:
        reasons.append("Predictive usefulness test could not run: {}.".format(reason))
    reasons.append("No strong target relationship was detected, so the external data is not forced into training.")
    return False, "reject_unproven_value", reasons


def _normalize_city_value(value):
    if pd.isna(value):
        return np.nan
    text = normalize_column_name(value)
    return text.replace("_", " ")


def _mode_or_missing(series):
    mode = series.dropna().mode()
    return mode.iloc[0] if not mode.empty else np.nan


def _numeric_target(series):
    if series is None:
        return None
    clean = pd.Series(series)
    if pd.api.types.is_numeric_dtype(clean):
        return pd.to_numeric(clean, errors="coerce")
    inferred = infer_problem_type(clean, requested="auto")
    if inferred == "classification":
        codes, _ = pd.factorize(clean)
        encoded = pd.Series(codes, index=clean.index, dtype="float")
        encoded[clean.isna()] = np.nan
        return encoded
    return pd.to_numeric(clean, errors="coerce")


def _safe_stratify(y, problem_type):
    if problem_type != "classification":
        return None
    counts = pd.Series(y).value_counts(dropna=False)
    if len(counts) < 2 or counts.min() < 2:
        return None
    return y


def _haversine_series(lat1, lon1, lat2, lon2):
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None
    lat1 = pd.to_numeric(lat1, errors="coerce")
    lon1 = pd.to_numeric(lon1, errors="coerce")
    lat2 = pd.to_numeric(lat2, errors="coerce")
    lon2 = pd.to_numeric(lon2, errors="coerce")
    radius_km = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2.0) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


def _correlation_interpretation(corr):
    direction = "higher" if corr > 0 else "lower"
    strength = "strong" if abs(corr) >= 0.3 else "moderate" if abs(corr) >= 0.12 else "weak"
    return "{} positive-risk relationship: {} values tend to align with higher target values.".format(
        strength,
        direction,
    )


def _weather_feature_importance(pipeline, top_n=10):
    importance = model_feature_importance(pipeline, top_n=80)
    if importance.empty:
        return []
    rows = []
    for _, row in importance.iterrows():
        feature = str(row["feature"])
        if _is_weather_feature(feature):
            rows.append({"feature": feature, "importance": round(float(row["importance"]), 6)})
    return rows[:top_n]


def _weather_importance_share(pipeline):
    importance = model_feature_importance(pipeline, top_n=1000)
    if importance.empty or "importance" not in importance.columns:
        return 0.0
    total = float(importance["importance"].sum())
    if total <= 0:
        return 0.0
    weather_total = float(
        importance.loc[importance["feature"].astype(str).map(_is_weather_feature), "importance"].sum()
    )
    return round(weather_total / total, 6)


def _is_weather_feature(feature):
    normalized = normalize_column_name(feature)
    return "weather" in normalized or "humidity" in normalized or "latitude" in normalized or "longitude" in normalized


def _relative_improvement(baseline_score, improvement):
    try:
        baseline_score = float(baseline_score)
        improvement = float(improvement)
    except Exception:
        return None
    if not np.isfinite(baseline_score) or abs(baseline_score) < 1e-9:
        return None
    return round(improvement / abs(baseline_score), 6)


def _safe_float(value):
    try:
        value = float(value)
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value
