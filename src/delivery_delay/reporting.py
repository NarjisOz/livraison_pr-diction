"""Human-readable summaries for dashboard users and thesis reporting."""

import pandas as pd

from .cleaning import basic_quality_report
from .prediction_dashboard import (
    anomaly_alerts,
    prediction_plain_language_summary,
    prediction_reliability_indicators,
)
from .recommendations import recommendations_table


def dataset_summary(df):
    """Generate concise non-technical dataset observations."""

    quality = basic_quality_report(df)
    rows = quality["rows"]
    columns = quality["columns"]
    missing_rate = quality["missing_rate"] * 100
    duplicates = quality["duplicate_rows"]

    messages = [
        "The dataset contains {:,} rows and {:,} columns.".format(rows, columns),
        "Overall missing-data rate is {:.2f}%.".format(missing_rate),
    ]
    if duplicates:
        messages.append(
            "{:,} duplicate rows were detected and will be removed during training.".format(duplicates)
        )
    else:
        messages.append("No duplicate rows were detected.")
    return messages


def model_summary(artifact):
    """Generate business-friendly model-selection messages."""

    if not artifact:
        return []

    metrics = artifact.get("metrics", {})
    problem_type = artifact.get("problem_type", "classification")
    model_name = artifact.get("best_model_name", "selected model")

    messages = [
        "{} was selected automatically after comparing candidate models.".format(model_name),
        "Target column: {}.".format(artifact.get("target_column", "unknown")),
    ]

    if problem_type == "classification":
        if "roc_auc" in metrics:
            messages.append(
                "ROC-AUC is {:.3f}, which measures ranking quality for delayed shipments.".format(
                    metrics["roc_auc"]
                )
            )
        messages.append("Weighted F1-score is {:.3f}.".format(metrics.get("f1_weighted", 0.0)))
    else:
        messages.append(
            "RMSE is {:.3f}, the typical prediction error in target units.".format(metrics.get("rmse", 0.0))
        )
        messages.append("R-squared is {:.3f}.".format(metrics.get("r2", 0.0)))

    leakage_columns = artifact.get("leakage_columns", [])
    if leakage_columns:
        messages.append("Potential leakage columns were removed: {}.".format(", ".join(leakage_columns[:8])))
    return messages


def prediction_summary(predictions):
    """Summarize model outputs for uploaded prediction data."""

    if predictions is None or predictions.empty:
        return []

    messages = ["Predictions were generated for {:,} rows.".format(len(predictions))]
    if "delay_probability" in predictions.columns:
        avg_risk = predictions["delay_probability"].mean()
        high_risk = (predictions["delay_probability"] >= 0.5).sum()
        messages.append("Average predicted delay risk is {:.1%}.".format(avg_risk))
        messages.append("{:,} rows have predicted delay risk of at least 50%.".format(int(high_risk)))
    elif "predicted_delay_value" in predictions.columns:
        values = pd.to_numeric(predictions["predicted_delay_value"], errors="coerce")
        messages.append("Average predicted delay value is {:.2f}.".format(values.mean()))
    return messages


def prediction_dashboard_summary(predictions, group_column=None):
    """Business-facing explanation for the prediction dashboard."""

    return prediction_plain_language_summary(predictions, group_column=group_column)


def build_training_report_markdown(artifact):
    """Build a downloadable model-training report."""

    understanding = artifact.get("dataset_understanding", {})
    contamination = artifact.get("contamination_report", {})
    lines = [
        "# Delivery Delay Model Report",
        "",
        "## Selected Model",
        "",
        "- Best model: {}".format(artifact.get("best_model_name", "unknown")),
        "- Target column: {}".format(artifact.get("target_column", "unknown")),
        "- Problem type: {}".format(artifact.get("problem_type", "unknown")),
        "- Trained at: {}".format(artifact.get("trained_at", "unknown")),
        "",
        "## Dataset Understanding",
        "",
        "- Quality score: {}/100".format(understanding.get("quality_score", "n/a")),
        "- Compatibility score: {}/100".format(understanding.get("compatibility_score", "n/a")),
        "- Governance score: {}/100".format(understanding.get("governance_score", "n/a")),
        "- Duplicate rate: {}".format(understanding.get("duplicate_rate", "n/a")),
        "",
        "## Validation",
        "",
        "- Train/test contamination risk: {}".format(contamination.get("risk_level", "unknown")),
        "- Exact row overlap count: {}".format(contamination.get("exact_row_overlap_count", "unknown")),
        "",
        "## Metrics",
        "",
    ]
    for key, value in artifact.get("metrics", {}).items():
        lines.append("- {}: {}".format(key, value))

    lines.extend(["", "## Preprocessing Plan", ""])
    plan = artifact.get("preprocessing_plan", {})
    lines.append("```text")
    lines.append(str(plan))
    lines.append("```")
    return "\n".join(lines)


def build_prediction_report_markdown(predictions, recommendations):
    """Build a downloadable business report for prediction outputs."""

    lines = ["# Delivery Delay Prediction Report", ""]
    lines.extend(["## Summary", ""])
    for message in prediction_dashboard_summary(predictions):
        lines.append("- {}".format(message))

    lines.extend(["", "## Reliability Indicators", ""])
    reliability = prediction_reliability_indicators(predictions)
    if reliability.empty:
        lines.append("- No reliability indicators available.")
    else:
        for _, row in reliability.iterrows():
            lines.append(
                "- {}: {} ({}) - {}".format(
                    row["indicator"],
                    row["value"],
                    row["status"],
                    row["interpretation"],
                )
            )

    lines.extend(["", "## Operational Alerts", ""])
    for alert in anomaly_alerts(predictions):
        lines.append(
            "- [{}] {}: {} Action: {}".format(
                alert["severity"],
                alert["alert"],
                alert["evidence"],
                alert["action"],
            )
        )

    lines.extend(["", "## Recommendations", ""])
    table = recommendations_table(recommendations)
    if table.empty:
        lines.append("- No recommendations generated.")
    else:
        for _, row in table.iterrows():
            lines.append(
                "- [{}] {}: {} Action: {}".format(
                    row["priority"], row["theme"], row["insight"], row["action"]
                )
            )

    return "\n".join(lines)


def build_prediction_dashboard_html(predictions, recommendations, title="Logistics Prediction Dashboard"):
    """Build a portable HTML dashboard summary for download."""

    summary = prediction_dashboard_summary(predictions)
    recs = recommendations_table(recommendations)
    safe_title = _html_escape(title)
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>{}</title>".format(safe_title),
        "<style>",
        "body{font-family:Arial,sans-serif;margin:32px;color:#102033;background:#f5f8fb}",
        ".card{background:#fff;border:1px solid #dde6ef;border-radius:8px;padding:18px;margin:14px 0}",
        ".high{border-left:5px solid #c2413d}.medium{border-left:5px solid #b7791f}.low{border-left:5px solid #15803d}",
        "table{border-collapse:collapse;width:100%;background:#fff}td,th{border:1px solid #dde6ef;padding:8px;text-align:left}",
        "th{background:#e6f4f1}",
        "</style></head><body>",
        "<h1>{}</h1>".format(safe_title),
        "<div class='card'><h2>Summary</h2><ul>",
    ]
    for message in summary:
        lines.append("<li>{}</li>".format(_html_escape(message)))
    lines.extend(["</ul></div>", "<div class='card'><h2>Priority Recommendations</h2>"])
    if recs.empty:
        lines.append("<p>No recommendations generated.</p>")
    else:
        lines.append(recs.to_html(index=False, escape=True))
    lines.append("</div>")

    if predictions is not None and not predictions.empty:
        preview_columns = [
            column
            for column in [
                "delay_probability",
                "risk_segment",
                "predicted_delay_class",
                "predicted_delay_value",
                "estimated_delivery_days",
                "estimated_delivery_date",
                "prediction_confidence",
                "confidence_level",
                "prediction_system",
                "prediction_target",
            ]
            if column in predictions.columns
        ]
        preview = predictions[preview_columns].head(50) if preview_columns else predictions.head(50)
        lines.extend(
            [
                "<div class='card'><h2>Prediction Preview</h2>",
                preview.to_html(index=False, escape=True),
                "</div>",
            ]
        )

    lines.extend(["</body></html>"])
    return "\n".join(lines)


def _html_escape(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
