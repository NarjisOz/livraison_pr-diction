"""Operational recommendation engine for logistics prediction results."""

import pandas as pd

from .prediction_dashboard import group_risk_table, recommended_group_columns
from .schema import normalize_column_name


def generate_operational_recommendations(predictions=None, artifact=None, top_n=8):
    """Generate business recommendations from predictions and model metadata."""

    recommendations = []
    if artifact:
        recommendations.extend(_artifact_recommendations(artifact))

    if predictions is not None and not predictions.empty:
        recommendations.extend(_prediction_recommendations(predictions, top_n=top_n))

    if not recommendations:
        recommendations.append(
            {
                "priority": "medium",
                "theme": "Readiness",
                "insight": "No strong operational risk pattern was detected yet.",
                "action": "Upload more delivery records or train on a richer logistics dataset.",
            }
        )
    return recommendations


def recommendations_table(recommendations):
    """Return recommendations as a DataFrame."""

    return pd.DataFrame(recommendations, columns=["priority", "theme", "insight", "action"])


def _artifact_recommendations(artifact):
    recommendations = []
    understanding = artifact.get("dataset_understanding", {})

    if understanding.get("compatibility_score", 100) < 60:
        recommendations.append(
            {
                "priority": "high",
                "theme": "Dataset compatibility",
                "insight": "The dataset only partially matches expected logistics patterns.",
                "action": "Confirm target, date, origin, destination, carrier, supplier, and delay-related columns.",
            }
        )

    if understanding.get("quality_score", 100) < 70:
        recommendations.append(
            {
                "priority": "high",
                "theme": "Data quality",
                "insight": "The dataset quality score is below the recommended threshold.",
                "action": "Review missing values, duplicate rows, constant columns, and suspicious identifiers.",
            }
        )

    if understanding.get("governance_score", 100) < 100:
        recommendations.append(
            {
                "priority": "medium",
                "theme": "Governance",
                "insight": "Potential personal or sensitive data columns were detected.",
                "action": "Remove or anonymize sensitive columns before sharing, deployment, or reporting.",
            }
        )

    contamination = artifact.get("contamination_report", {})
    if contamination.get("risk_level") in ("medium", "high"):
        recommendations.append(
            {
                "priority": contamination.get("risk_level"),
                "theme": "Train/test contamination",
                "insight": "Some training rows or identifiers appear again in the test split.",
                "action": "Use time-based splitting or group-based splitting by shipment/order/customer identifier.",
            }
        )

    leakage_columns = artifact.get("leakage_columns", [])
    if leakage_columns:
        recommendations.append(
            {
                "priority": "high",
                "theme": "Target leakage",
                "insight": "Outcome or target-source columns were excluded from training.",
                "action": "Keep these columns out of prediction inputs: {}.".format(", ".join(leakage_columns[:8])),
            }
        )

    return recommendations


def _prediction_recommendations(predictions, top_n):
    recommendations = []
    if "delay_probability" not in predictions.columns and "predicted_delay_value" not in predictions.columns:
        return recommendations

    for group_column in recommended_group_columns(predictions)[:top_n]:
        grouped = group_risk_table(predictions, group_column, top_n=1)
        if grouped.empty:
            continue
        top_group = grouped.iloc[0]
        normalized = normalize_column_name(group_column)
        theme = _theme_from_column(normalized)

        if "average_delay_risk" in grouped.columns:
            risk = top_group["average_delay_risk"]
            high_count = int(top_group.get("high_risk_shipments", 0))
            if risk < 0.5 and high_count == 0:
                continue
            recommendations.append(
                {
                    "priority": "high" if risk >= 0.65 or high_count >= 5 else "medium",
                    "theme": theme,
                    "insight": "{} '{}' has elevated delay risk ({:.1%} average risk).".format(
                        group_column,
                        top_group[group_column],
                        risk,
                    ),
                    "action": _action_from_theme(theme, top_group[group_column]),
                }
            )
        elif "average_predicted_delay" in grouped.columns:
            recommendations.append(
                {
                    "priority": "medium",
                    "theme": theme,
                    "insight": "{} '{}' has the highest average predicted delay value ({:.2f}).".format(
                        group_column,
                        top_group[group_column],
                        top_group["average_predicted_delay"],
                    ),
                    "action": _action_from_theme(theme, top_group[group_column]),
                }
            )

    if "risk_segment" in predictions.columns:
        high_risk = predictions[predictions["risk_segment"] == "High risk"]
        if len(high_risk) > 0:
            recommendations.insert(
                0,
                {
                    "priority": "high",
                    "theme": "Operational triage",
                    "insight": "{:,} records are classified as high delay risk.".format(len(high_risk)),
                    "action": "Review high-risk rows first and contact responsible teams before the promised delivery date.",
                },
            )

    return recommendations


def _theme_from_column(normalized_column):
    if "supplier" in normalized_column:
        return "Risky suppliers"
    if "region" in normalized_column:
        return "Risky regions"
    if "city" in normalized_column or "destination" in normalized_column or "origin" in normalized_column:
        return "Route or location risk"
    if "company" in normalized_column or "carrier" in normalized_column or "logistics" in normalized_column:
        return "Carrier performance"
    if "product" in normalized_column:
        return "Product handling risk"
    return "Operational bottleneck"


def _action_from_theme(theme, value):
    if theme == "Risky suppliers":
        return "Audit supplier '{}' and compare pickup readiness, packaging, and handoff timing.".format(value)
    if theme == "Risky regions":
        return "Check capacity, route constraints, and weather or infrastructure issues for region '{}'.".format(value)
    if theme == "Route or location risk":
        return "Review route planning and delivery capacity for '{}'.".format(value)
    if theme == "Carrier performance":
        return "Compare carrier SLA, backlog, and escalation process for '{}'.".format(value)
    if theme == "Product handling risk":
        return "Check whether product '{}' needs special handling, packaging, or earlier dispatch.".format(value)
    return "Investigate process bottlenecks for '{}' and monitor this group over time.".format(value)

