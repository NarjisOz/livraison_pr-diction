"""Business-friendly dashboard helpers for prediction results."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .cleaning import basic_quality_report
from .schema import detect_columns, normalize_column_name


PREDICTION_COLUMNS = {
    "predicted_delay_class",
    "delay_probability",
    "predicted_delay_value",
    "estimated_delivery_days",
    "estimated_delivery_date",
    "risk_score",
    "prediction_confidence",
    "confidence_level",
    "prediction_system",
    "prediction_target",
}


def enrich_prediction_segments(predictions, low_threshold=0.35, high_threshold=0.65):
    """Add readable risk segments to prediction output."""

    data = predictions.copy()
    if "delay_probability" in data.columns:
        probability = pd.to_numeric(data["delay_probability"], errors="coerce")
        data["risk_segment"] = pd.cut(
            probability,
            bins=[-0.001, low_threshold, high_threshold, 1.001],
            labels=["Low risk", "Medium risk", "High risk"],
        ).astype("object")
        data.loc[probability.isna(), "risk_segment"] = "Unknown"
        data["risk_percent"] = probability * 100
        data["risk_score"] = probability * 100
    elif "predicted_delay_value" in data.columns:
        values = pd.to_numeric(data["predicted_delay_value"], errors="coerce")
        low = values.quantile(0.33)
        high = values.quantile(0.66)
        data["risk_segment"] = pd.cut(
            values,
            bins=[float("-inf"), low, high, float("inf")],
            labels=["Shorter delay", "Typical delay", "Longer delay"],
        ).astype("object")
        data.loc[values.isna(), "risk_segment"] = "Unknown"
        if "risk_score" not in data.columns:
            data["risk_score"] = values.rank(pct=True) * 100.0
    return data


def prediction_kpis(predictions):
    """Return dashboard KPI values for classification or regression predictions."""

    if predictions is None or predictions.empty:
        return {}

    quality = basic_quality_report(predictions)
    kpis = {
        "rows": quality["rows"],
        "columns": quality["columns"],
        "missing_rate": quality["missing_rate"],
    }

    if "delay_probability" in predictions.columns:
        probability = pd.to_numeric(predictions["delay_probability"], errors="coerce")
        segments = predictions.get("risk_segment", pd.Series(index=predictions.index, dtype="object"))
        kpis.update(
            {
                "average_risk": float(probability.mean()),
                "median_risk": float(probability.median()),
                "high_risk_count": int((segments == "High risk").sum()),
                "medium_risk_count": int((segments == "Medium risk").sum()),
                "low_risk_count": int((segments == "Low risk").sum()),
                "max_risk": float(probability.max()),
                "average_confidence": float(
                    pd.to_numeric(predictions.get("prediction_confidence"), errors="coerce").mean()
                )
                if "prediction_confidence" in predictions.columns
                else None,
            }
        )

    if "predicted_delay_value" in predictions.columns or "estimated_delivery_days" in predictions.columns:
        base_values = predictions.get("predicted_delay_value", predictions.get("estimated_delivery_days"))
        values = pd.to_numeric(base_values, errors="coerce")
        estimated_days = pd.to_numeric(predictions.get("estimated_delivery_days", values), errors="coerce")
        kpis.update(
            {
                "average_prediction": float(values.mean()),
                "median_prediction": float(values.median()),
                "max_prediction": float(values.max()),
                "min_prediction": float(values.min()),
                "average_estimated_days": float(estimated_days.mean()),
                "average_confidence": float(
                    pd.to_numeric(predictions.get("prediction_confidence"), errors="coerce").mean()
                )
                if "prediction_confidence" in predictions.columns
                else None,
            }
        )
        if "estimated_delivery_date" in predictions.columns:
            estimated_dates = pd.to_datetime(predictions["estimated_delivery_date"], errors="coerce")
            kpis["estimated_date_coverage"] = float(estimated_dates.notna().mean())
            if estimated_dates.notna().any():
                kpis["earliest_estimated_date"] = estimated_dates.min()
                kpis["latest_estimated_date"] = estimated_dates.max()

    return kpis


def prediction_reliability_indicators(predictions, artifact=None):
    """Build transparent reliability indicators for business and analyst users."""

    if predictions is None or predictions.empty:
        return pd.DataFrame(columns=["indicator", "value", "status", "interpretation"])

    kpis = prediction_kpis(predictions)
    rows = []
    missing_rate = float(kpis.get("missing_rate", 0.0))
    rows.append(
        _indicator_row(
            "Input completeness",
            1.0 - missing_rate,
            "Higher is better. Missing cells are imputed, but complete rows produce more trustworthy decisions.",
        )
    )

    output_columns = [
        column
        for column in ("delay_probability", "predicted_delay_value", "estimated_delivery_days", "risk_score")
        if column in predictions.columns
    ]
    if output_columns:
        coverage = predictions[output_columns].notna().any(axis=1).mean()
        rows.append(
            _indicator_row(
                "Prediction coverage",
                float(coverage),
                "Share of rows with at least one usable model output.",
            )
        )

    if "prediction_confidence" in predictions.columns:
        confidence = pd.to_numeric(predictions["prediction_confidence"], errors="coerce")
        rows.append(
            _indicator_row(
                "Average confidence",
                float(confidence.mean()),
                "Read low-confidence rows as planning signals that need human context.",
            )
        )

    if "estimated_delivery_date" in predictions.columns:
        date_coverage = pd.to_datetime(predictions["estimated_delivery_date"], errors="coerce").notna().mean()
        rows.append(
            _indicator_row(
                "Estimated date coverage",
                float(date_coverage),
                "Rows with enough date context to produce an estimated delivery date.",
            )
        )

    validation_score, validation_label = _validation_signal(artifact)
    if validation_score is not None:
        rows.append(
            _indicator_row(
                "Validation signal",
                validation_score,
                validation_label,
            )
        )

    return pd.DataFrame(rows, columns=["indicator", "value", "status", "interpretation"])


def anomaly_alerts(predictions, group_column=None):
    """Identify operational anomalies that should be surfaced before raw tables."""

    if predictions is None or predictions.empty:
        return []

    alerts = []
    kpis = prediction_kpis(predictions)
    rows = max(int(kpis.get("rows", len(predictions))), 1)
    missing_rate = float(kpis.get("missing_rate", 0.0))

    if "delay_probability" in predictions.columns:
        high_count = int(kpis.get("high_risk_count", 0))
        high_rate = high_count / rows
        if high_count:
            alerts.append(
                {
                    "severity": "high" if high_rate >= 0.25 else "medium",
                    "alert": "High delay-risk concentration",
                    "evidence": "{:,} records ({:.1%}) are in the high-risk segment.".format(
                        high_count, high_rate
                    ),
                    "action": "Prioritize these rows for carrier, supplier, route, and promised-date review.",
                }
            )

    if "prediction_confidence" in predictions.columns:
        confidence = pd.to_numeric(predictions["prediction_confidence"], errors="coerce")
        low_confidence = int((confidence < 0.60).sum())
        if low_confidence:
            alerts.append(
                {
                    "severity": "medium" if low_confidence / rows < 0.25 else "high",
                    "alert": "Low-confidence predictions",
                    "evidence": "{:,} records have confidence below 60%.".format(low_confidence),
                    "action": "Treat these as review candidates and compare them with operational context before acting.",
                }
            )

    if missing_rate >= 0.10:
        alerts.append(
            {
                "severity": "medium" if missing_rate < 0.25 else "high",
                "alert": "Incomplete prediction inputs",
                "evidence": "{:.1%} of cells are missing in the prediction output/input context.".format(
                    missing_rate
                ),
                "action": "Improve source-system capture for dates, routes, carriers, suppliers, and delivery status.",
            }
        )

    if "estimated_delivery_days" in predictions.columns:
        days = pd.to_numeric(predictions["estimated_delivery_days"], errors="coerce")
        if days.notna().sum() >= 8:
            q1 = days.quantile(0.25)
            q3 = days.quantile(0.75)
            iqr = q3 - q1
            threshold = q3 + 1.5 * iqr if pd.notna(iqr) and iqr > 0 else days.quantile(0.95)
            extreme_count = int((days > threshold).sum())
            if extreme_count:
                alerts.append(
                    {
                        "severity": "medium",
                        "alert": "Extreme delivery-duration estimates",
                        "evidence": "{:,} records are above the expected duration range.".format(
                            extreme_count
                        ),
                        "action": "Check whether these shipments involve long routes, special handling, or overloaded partners.",
                    }
                )

    if group_column:
        grouped = group_risk_table(predictions, group_column, top_n=1)
        if not grouped.empty and "average_delay_risk" in grouped.columns:
            top = grouped.iloc[0]
            if top["average_delay_risk"] >= 0.65:
                alerts.append(
                    {
                        "severity": "high",
                        "alert": "Risk hotspot detected",
                        "evidence": "{} '{}' averages {:.1%} delay risk.".format(
                            group_column,
                            top[group_column],
                            top["average_delay_risk"],
                        ),
                        "action": "Investigate this route, city, carrier, supplier, or customer segment before lower-risk work.",
                    }
                )

    if not alerts:
        alerts.append(
            {
                "severity": "low",
                "alert": "No severe anomaly detected",
                "evidence": "The prediction output does not show a dominant reliability or risk anomaly.",
                "action": "Continue monitoring grouped risk and low-confidence rows as new data arrives.",
            }
        )
    return alerts


def executive_ai_summary(predictions, artifact=None, group_column=None, recommendations=None):
    """Generate a concise logistics-consultant narrative for the dashboard."""

    if predictions is None or predictions.empty:
        return ["No prediction rows are available yet."]

    data = enrich_prediction_segments(predictions)
    kpis = prediction_kpis(data)
    model_name = (
        artifact.get("best_model_name", "selected model") if isinstance(artifact, dict) else "selected model"
    )
    target = (
        artifact.get("target_column", "the selected target")
        if isinstance(artifact, dict)
        else "the selected target"
    )
    messages = [
        "{} is active for {}, and {:,} records were scored.".format(
            model_name,
            target,
            int(kpis.get("rows", 0)),
        )
    ]

    if "delay_probability" in data.columns:
        messages.append(
            "Average delay risk is {:.1%}, with {:,} high-risk records requiring first attention.".format(
                kpis.get("average_risk", 0.0),
                int(kpis.get("high_risk_count", 0)),
            )
        )

    if "estimated_delivery_days" in data.columns:
        messages.append(
            "Average estimated delivery duration is {:.2f} days.".format(
                kpis.get("average_estimated_days", 0.0)
            )
        )

    if "estimated_delivery_date" in data.columns:
        dates = pd.to_datetime(data["estimated_delivery_date"], errors="coerce")
        if dates.notna().any():
            messages.append(
                "Estimated delivery dates span {} to {} where date inputs are available.".format(
                    dates.min().date(),
                    dates.max().date(),
                )
            )

    if kpis.get("average_confidence") is not None:
        messages.append(
            "Average prediction confidence is {:.1%}; low-confidence rows are clearly marked for review.".format(
                kpis.get("average_confidence", 0.0)
            )
        )

    if group_column:
        grouped = group_risk_table(data, group_column, top_n=1)
        if not grouped.empty:
            top_group = grouped.iloc[0]
            if "average_delay_risk" in grouped.columns:
                messages.append(
                    "The most exposed {} is '{}', averaging {:.1%} delay risk.".format(
                        group_column,
                        top_group[group_column],
                        top_group["average_delay_risk"],
                    )
                )
            elif "average_predicted_delay" in grouped.columns:
                messages.append(
                    "The slowest {} is '{}', averaging {:.2f} estimated days.".format(
                        group_column,
                        top_group[group_column],
                        top_group["average_predicted_delay"],
                    )
                )

    if recommendations:
        first = recommendations[0]
        messages.append("Recommended next action: {}".format(first.get("action", "review priority rows.")))

    return messages


def recommended_group_columns(predictions, max_unique=30):
    """Find readable categorical columns that explain where risk is concentrated."""

    schema = detect_columns(predictions)
    candidates = []
    preferred_keywords = (
        "city",
        "region",
        "company",
        "carrier",
        "supplier",
        "customer",
        "product",
        "route",
        "warehouse",
        "origin",
        "destination",
    )

    for column in schema.categorical + schema.boolean:
        if column in PREDICTION_COLUMNS or column.startswith("probability_"):
            continue
        unique_count = predictions[column].nunique(dropna=True)
        if 1 < unique_count <= max_unique:
            normalized = normalize_column_name(column)
            priority = 0 if any(keyword in normalized for keyword in preferred_keywords) else 1
            candidates.append((priority, unique_count, column))

    return [column for _, _, column in sorted(candidates)]


def group_risk_table(predictions, group_column, top_n=15):
    """Aggregate prediction risk by a user-selected business dimension."""

    if group_column not in predictions.columns:
        return pd.DataFrame()

    if "delay_probability" in predictions.columns:
        data = predictions.copy()
        data["delay_probability"] = pd.to_numeric(data["delay_probability"], errors="coerce")
        data["is_high_risk"] = data.get("risk_segment", "") == "High risk"
        grouped = (
            data.groupby(group_column, dropna=False)
            .agg(
                shipments=(group_column, "size"),
                average_delay_risk=("delay_probability", "mean"),
                highest_delay_risk=("delay_probability", "max"),
                high_risk_shipments=("is_high_risk", "sum"),
            )
            .reset_index()
        )
        grouped["high_risk_rate"] = grouped["high_risk_shipments"] / grouped["shipments"].clip(lower=1)
        return grouped.sort_values(
            ["average_delay_risk", "high_risk_shipments", "shipments"],
            ascending=[False, False, False],
        ).head(top_n)

    if "predicted_delay_value" in predictions.columns:
        data = predictions.copy()
        value_column = (
            "estimated_delivery_days"
            if "estimated_delivery_days" in data.columns
            else "predicted_delay_value"
        )
        data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
        grouped = (
            data.groupby(group_column, dropna=False)
            .agg(
                shipments=(group_column, "size"),
                average_predicted_delay=(value_column, "mean"),
                highest_predicted_delay=(value_column, "max"),
            )
            .reset_index()
        )
        return grouped.sort_values(
            ["average_predicted_delay", "shipments"],
            ascending=[False, False],
        ).head(top_n)

    return pd.DataFrame()


def top_priority_rows(predictions, top_n=20):
    """Return the rows a business user should inspect first."""

    if "delay_probability" in predictions.columns:
        return predictions.sort_values("delay_probability", ascending=False).head(top_n)
    if "predicted_delay_value" in predictions.columns:
        return predictions.sort_values("predicted_delay_value", ascending=False).head(top_n)
    return predictions.head(top_n)


def readable_priority_columns(predictions):
    """Choose compact columns for the priority table."""

    identifier_keywords = ("shipment", "order", "delivery", "tracking", "id", "reference")
    business_keywords = (
        "origin",
        "destination",
        "city",
        "company",
        "supplier",
        "product",
        "customer",
        "date",
    )
    selected = []
    for column in predictions.columns:
        normalized = normalize_column_name(column)
        if any(keyword in normalized for keyword in identifier_keywords + business_keywords):
            selected.append(column)

    prediction_columns = [
        column
        for column in [
            "delay_probability",
            "risk_score",
            "risk_segment",
            "prediction_confidence",
            "confidence_level",
            "predicted_delay_class",
            "predicted_delay_value",
            "estimated_delivery_days",
            "estimated_delivery_date",
        ]
        if column in predictions.columns
    ]
    return list(dict.fromkeys(selected[:8] + prediction_columns))


def prediction_plain_language_summary(predictions, group_column=None):
    """Create short explanations for non-technical users."""

    if predictions is None or predictions.empty:
        return []

    data = enrich_prediction_segments(predictions)
    kpis = prediction_kpis(data)
    messages = ["The model analyzed {:,} records from the uploaded file.".format(kpis.get("rows", 0))]

    if "delay_probability" in data.columns:
        messages.append(
            "The average delay risk is {:.1%}; this is the typical probability that a shipment may be delayed.".format(
                kpis.get("average_risk", 0.0)
            )
        )
        messages.append(
            "{:,} records are high risk and should be reviewed first.".format(kpis.get("high_risk_count", 0))
        )
        if kpis.get("average_confidence") is not None:
            messages.append(
                "Average prediction confidence is {:.1%}; low-confidence rows should be reviewed with extra context.".format(
                    kpis.get("average_confidence", 0.0)
                )
            )
        if "estimated_delivery_days" in data.columns:
            messages.append(
                "The average estimated delivery duration is {:.2f} days.".format(
                    kpis.get("average_estimated_days", 0.0)
                )
            )
        if (
            "estimated_delivery_date" in data.columns
            and pd.to_datetime(data["estimated_delivery_date"], errors="coerce").notna().any()
        ):
            dates = pd.to_datetime(data["estimated_delivery_date"], errors="coerce")
            messages.append(
                "Estimated delivery dates range from {} to {} for rows with usable date inputs.".format(
                    dates.min().date(),
                    dates.max().date(),
                )
            )
        if group_column:
            grouped = group_risk_table(data, group_column, top_n=1)
            if not grouped.empty:
                top_group = grouped.iloc[0]
                messages.append(
                    "The highest-risk {} is '{}', with an average delay risk of {:.1%}.".format(
                        group_column,
                        top_group[group_column],
                        top_group["average_delay_risk"],
                    )
                )
    elif "predicted_delay_value" in data.columns:
        messages.append(
            "The average estimated delivery duration is {:.2f} days.".format(
                kpis.get("average_estimated_days", kpis.get("average_prediction", 0.0))
            )
        )
        if (
            "estimated_delivery_date" in data.columns
            and pd.to_datetime(data["estimated_delivery_date"], errors="coerce").notna().any()
        ):
            dates = pd.to_datetime(data["estimated_delivery_date"], errors="coerce")
            messages.append(
                "Estimated delivery dates range from {} to {} for rows with usable date inputs.".format(
                    dates.min().date(),
                    dates.max().date(),
                )
            )
        if kpis.get("average_confidence") is not None:
            messages.append(
                "Regression confidence is {:.1%}, based on validation strength from the trained model.".format(
                    kpis.get("average_confidence", 0.0)
                )
            )
        messages.append("Rows with the largest predicted values should be reviewed first.")

    missing_rate = kpis.get("missing_rate", 0.0)
    if missing_rate > 0.05:
        messages.append(
            "The uploaded data has {:.1%} missing cells, so predictions may be less reliable for incomplete rows.".format(
                missing_rate
            )
        )
    else:
        messages.append("The uploaded data is mostly complete, which supports more reliable predictions.")

    return messages


def _indicator_row(indicator, score, interpretation):
    score = 0.0 if pd.isna(score) else float(max(min(score, 1.0), 0.0))
    if score >= 0.80:
        status = "Strong"
    elif score >= 0.60:
        status = "Watch"
    else:
        status = "Review"
    return {
        "indicator": indicator,
        "value": "{:.1%}".format(score),
        "status": status,
        "interpretation": interpretation,
    }


def _validation_signal(artifact):
    if not isinstance(artifact, dict):
        return None, None
    metrics = artifact.get("metrics", {})
    problem_type = artifact.get("problem_type")
    if problem_type == "classification":
        value = metrics.get("roc_auc", metrics.get("f1_weighted"))
        if value is None:
            return None, None
        return float(
            max(min(value, 1.0), 0.0)
        ), "Validation quality for ranking delayed vs on-time shipments."
    if problem_type == "regression":
        r2 = metrics.get("r2")
        if r2 is None:
            return None, None
        return float(
            max(min(0.50 + 0.50 * float(r2), 1.0), 0.0)
        ), "Validation strength for delivery-duration estimates."
    return None, None


def plot_risk_gauge(predictions):
    """Gauge chart for average delay probability."""

    if "delay_probability" not in predictions.columns:
        return go.Figure()
    average_risk = pd.to_numeric(predictions["delay_probability"], errors="coerce").mean()
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(average_risk * 100),
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563EB"},
                "steps": [
                    {"range": [0, 35], "color": "#DCFCE7"},
                    {"range": [35, 65], "color": "#FEF3C7"},
                    {"range": [65, 100], "color": "#FEE2E2"},
                ],
            },
            title={"text": "Average delay risk"},
        )
    )


def plot_risk_segments(predictions):
    if "risk_segment" not in predictions.columns:
        return go.Figure()
    order = [
        "Low risk",
        "Medium risk",
        "High risk",
        "Unknown",
        "Shorter delay",
        "Typical delay",
        "Longer delay",
    ]
    counts = predictions["risk_segment"].value_counts(dropna=False).reindex(order).dropna().reset_index()
    counts.columns = ["risk_segment", "shipments"]
    if counts.empty:
        return go.Figure()
    return px.bar(
        counts,
        x="risk_segment",
        y="shipments",
        color="risk_segment",
        color_discrete_map={
            "Low risk": "#16A34A",
            "Medium risk": "#D97706",
            "High risk": "#DC2626",
            "Unknown": "#6B7280",
            "Shorter delay": "#16A34A",
            "Typical delay": "#D97706",
            "Longer delay": "#DC2626",
        },
        title="Prediction risk segments",
    )


def plot_probability_distribution(predictions):
    if "delay_probability" not in predictions.columns:
        return go.Figure()
    data = predictions.copy()
    data["delay_probability_percent"] = pd.to_numeric(data["delay_probability"], errors="coerce") * 100
    return px.histogram(
        data,
        x="delay_probability_percent",
        nbins=20,
        color="risk_segment" if "risk_segment" in data.columns else None,
        title="Distribution of predicted delay risk",
        labels={"delay_probability_percent": "Delay risk (%)"},
    )


def plot_group_risk(grouped, group_column):
    if grouped is None or grouped.empty:
        return go.Figure()
    if "average_delay_risk" in grouped.columns:
        chart = grouped.copy()
        chart["average_delay_risk_percent"] = chart["average_delay_risk"] * 100
        return px.bar(
            chart.sort_values("average_delay_risk_percent"),
            x="average_delay_risk_percent",
            y=group_column,
            orientation="h",
            text="shipments",
            title="Where delay risk is concentrated",
            labels={"average_delay_risk_percent": "Average delay risk (%)", group_column: group_column},
        )
    if "average_predicted_delay" in grouped.columns:
        return px.bar(
            grouped.sort_values("average_predicted_delay"),
            x="average_predicted_delay",
            y=group_column,
            orientation="h",
            text="shipments",
            title="Highest predicted delay by group",
        )
    return go.Figure()


def plot_prediction_value_distribution(predictions):
    if "predicted_delay_value" not in predictions.columns:
        return go.Figure()
    x_column = (
        "estimated_delivery_days"
        if "estimated_delivery_days" in predictions.columns
        else "predicted_delay_value"
    )
    return px.histogram(
        predictions,
        x=x_column,
        nbins=25,
        color="risk_segment" if "risk_segment" in predictions.columns else None,
        title="Distribution of estimated delivery days",
    )
