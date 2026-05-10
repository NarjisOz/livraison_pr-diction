"""Plain-language analytics insights for logistics datasets."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Optional

import pandas as pd

from .eda import build_eda_report
from .schema import detect_columns, normalize_column_name
from .targeting import add_derived_delivery_targets


PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


@dataclass
class Insight:
    """One business-readable observation with an action."""

    priority: str
    theme: str
    title: str
    observation: str
    impact: str
    recommended_action: str
    metric: Optional[str] = None


def build_eda_story(df, target_column=None, max_insights=14):
    """Generate prioritized EDA storytelling insights from a dataframe."""

    if df is None or df.empty:
        return [
            Insight(
                priority="info",
                theme="Readiness",
                title="No dataset is active",
                observation="Upload a logistics dataset to begin the analysis.",
                impact="The platform cannot profile, train, or predict until data is available.",
                recommended_action="Upload a CSV, Excel, or Parquet file.",
            )
        ]

    data, _ = add_derived_delivery_targets(df)
    target_column = target_column if target_column in data.columns else _best_target(data, target_column)
    report = build_eda_report(data, target_column=target_column)
    insights = []
    insights.extend(_dataset_shape_insights(data))
    insights.extend(_missing_value_insights(data, report["missing_values"]))
    insights.extend(_duplicate_insights(data))
    insights.extend(_numeric_distribution_insights(report["numeric_profile"], report["outliers"]))
    insights.extend(_categorical_distribution_insights(data, report["categorical_profile"]))
    insights.extend(_datetime_insights(report["datetime_profile"]))
    insights.extend(_target_behavior_insights(data, target_column))
    insights.extend(_operational_group_insights(data, target_column))
    insights.extend(_geography_column_insights(data))

    return prioritize_insights(insights)[:max_insights]


def insights_to_frame(insights):
    """Return insights as a dataframe for display and download."""

    return pd.DataFrame([asdict(insight) for insight in insights])


def insights_to_markdown(insights, title="Logistics Insight Summary"):
    """Build a downloadable Markdown summary."""

    lines = ["# {}".format(title), ""]
    for insight in insights:
        lines.extend(
            [
                "## [{}] {}".format(insight.priority.upper(), insight.title),
                "",
                "- Theme: {}".format(insight.theme),
                "- Observation: {}".format(insight.observation),
                "- Impact: {}".format(insight.impact),
                "- Recommended action: {}".format(insight.recommended_action),
            ]
        )
        if insight.metric:
            lines.append("- Metric: {}".format(insight.metric))
        lines.append("")
    return "\n".join(lines)


def prioritize_insights(insights: Iterable[Insight]):
    """Sort insights so urgent operational findings appear first."""

    return sorted(
        [insight for insight in insights if insight is not None],
        key=lambda item: (PRIORITY_ORDER.get(item.priority, 99), item.theme, item.title),
    )


def top_priority_label(insights):
    """Return a compact label for the strongest visible signal."""

    prioritized = prioritize_insights(insights)
    if not prioritized:
        return "No critical issue detected"
    first = prioritized[0]
    return "{}: {}".format(first.priority.title(), first.title)


def _dataset_shape_insights(df):
    rows, columns = df.shape
    priority = "medium" if rows < 100 else "info"
    impact = (
        "Small datasets can make model validation unstable."
        if rows < 100
        else "The dataset is large enough for reliable profiling and initial model comparison."
    )
    action = (
        "Use model metrics as directional evidence and collect more historical shipments."
        if rows < 100
        else "Proceed with automated profiling, preprocessing, and model comparison."
    )
    return [
        Insight(
            priority=priority,
            theme="Dataset readiness",
            title="{} rows and {} columns available".format(f"{rows:,}", f"{columns:,}"),
            observation="The active dataset contains {:,} records across {:,} fields.".format(rows, columns),
            impact=impact,
            recommended_action=action,
            metric="rows={}, columns={}".format(rows, columns),
        )
    ]


def _missing_value_insights(df, missing_table):
    insights = []
    if missing_table is None or missing_table.empty:
        return insights

    total_missing_rate = float(df.isna().sum().sum() / max(df.shape[0] * df.shape[1], 1))
    if total_missing_rate == 0:
        insights.append(
            Insight(
                priority="info",
                theme="Data quality",
                title="No missing cells detected",
                observation="Every analyzed cell contains a value.",
                impact="This reduces imputation uncertainty and makes operational explanations cleaner.",
                recommended_action="Keep validating missingness on future uploads.",
                metric="missing_rate=0.00%",
            )
        )
        return insights

    top_missing = missing_table[missing_table["missing_count"] > 0].head(5)
    if top_missing.empty:
        return insights

    highest = top_missing.iloc[0]
    rate = float(highest["missing_rate"])
    priority = "critical" if rate >= 0.75 else "high" if rate >= 0.4 else "medium"
    insights.append(
        Insight(
            priority=priority,
            theme="Data quality",
            title="Missing values concentrate in {}".format(highest["column"]),
            observation="{:.1%} of '{}' is missing; overall missingness is {:.1%}.".format(
                rate, highest["column"], total_missing_rate
            ),
            impact=_missing_impact(rate),
            recommended_action=_missing_action(highest["column"], rate),
            metric="column_missing_rate={:.2%}, total_missing_rate={:.2%}".format(rate, total_missing_rate),
        )
    )

    for _, row in top_missing.iloc[1:3].iterrows():
        row_rate = float(row["missing_rate"])
        if row_rate >= 0.2:
            insights.append(
                Insight(
                    priority="medium" if row_rate < 0.5 else "high",
                    theme="Data quality",
                    title="Secondary missing-data issue in {}".format(row["column"]),
                    observation="{:.1%} of '{}' is unavailable.".format(row_rate, row["column"]),
                    impact="Predictions may rely less on this field because fewer historical records prove its value.",
                    recommended_action="Confirm whether this field is optional, delayed, or collected by only some teams.",
                    metric="missing_rate={:.2%}".format(row_rate),
                )
            )
    return insights


def _duplicate_insights(df):
    duplicates = int(df.duplicated().sum())
    if duplicates == 0:
        return []
    rate = duplicates / max(len(df), 1)
    return [
        Insight(
            priority="high" if rate >= 0.05 else "medium",
            theme="Data quality",
            title="Duplicate shipment records detected",
            observation="{:,} duplicate rows appear in the active dataset.".format(duplicates),
            impact="Duplicates can overstate repeated events and inflate model confidence.",
            recommended_action="Keep duplicate removal enabled before training unless duplicates represent real repeated shipments.",
            metric="duplicate_rate={:.2%}".format(rate),
        )
    ]


def _numeric_distribution_insights(numeric_profile, outliers):
    insights = []
    if numeric_profile is None or numeric_profile.empty:
        return insights

    profile = numeric_profile.copy()
    profile["abs_skewness"] = pd.to_numeric(profile.get("skewness"), errors="coerce").abs()
    skewed = profile.sort_values("abs_skewness", ascending=False).head(3)
    for _, row in skewed.iterrows():
        skew = row.get("skewness")
        if pd.notna(skew) and abs(float(skew)) >= 1.0:
            column = row["column"]
            direction = "right-skewed" if float(skew) > 0 else "left-skewed"
            insights.append(
                Insight(
                    priority="medium",
                    theme="Distribution behavior",
                    title="{} is strongly {}".format(column, direction),
                    observation="The skewness of '{}' is {:.2f}, so typical and extreme values differ sharply.".format(
                        column, float(skew)
                    ),
                    impact="Average-based rules may misrepresent normal logistics behavior for this field.",
                    recommended_action="Use median-based summaries and robust preprocessing for this column.",
                    metric="skewness={:.3f}".format(float(skew)),
                )
            )

    if outliers is not None and not outliers.empty:
        outlier_rows = outliers.sort_values("outlier_rate", ascending=False).head(3)
        for _, row in outlier_rows.iterrows():
            rate = float(row.get("outlier_rate", 0.0))
            if rate >= 0.03:
                insights.append(
                    Insight(
                        priority="high" if rate >= 0.10 else "medium",
                        theme="Anomaly detection",
                        title="Outliers detected in {}".format(row["column"]),
                        observation="{:.1%} of non-missing '{}' values fall outside the IQR range.".format(
                            rate, row["column"]
                        ),
                        impact="These records may represent urgent shipments, data entry errors, unusually long routes, or atypical costs.",
                        recommended_action="Review the largest values and keep IQR capping active for model training.",
                        metric="outlier_rate={:.2%}".format(rate),
                    )
                )
    return insights


def _categorical_distribution_insights(df, categorical_profile):
    insights = []
    if categorical_profile is None or categorical_profile.empty:
        return insights

    for _, row in categorical_profile.head(8).iterrows():
        column = row["column"]
        if column not in df.columns:
            continue
        counts = df[column].value_counts(dropna=False)
        if counts.empty:
            continue
        dominant_share = float(counts.iloc[0] / max(len(df), 1))
        unique_values = int(row.get("unique_values", counts.shape[0]))
        normalized = normalize_column_name(column)
        if dominant_share >= 0.70:
            insights.append(
                Insight(
                    priority="medium",
                    theme="Operational concentration",
                    title="{} is concentrated in one category".format(column),
                    observation="The leading '{}' value represents {:.1%} of records.".format(
                        column, dominant_share
                    ),
                    impact="The model may learn mostly from the dominant operation and have less evidence for rare groups.",
                    recommended_action="Compare performance across smaller groups before using predictions for escalation.",
                    metric="dominant_share={:.2%}".format(dominant_share),
                )
            )
        if unique_values >= 50 and any(
            token in normalized for token in ("supplier", "carrier", "city", "route")
        ):
            insights.append(
                Insight(
                    priority="medium",
                    theme="Operational complexity",
                    title="{} has many distinct values".format(column),
                    observation="'{}' contains {:,} unique values.".format(column, unique_values),
                    impact="This may reveal a complex supplier, carrier, city, or route network.",
                    recommended_action="Use grouped dashboards to find which categories carry the highest delay risk.",
                    metric="unique_values={}".format(unique_values),
                )
            )
    return insights


def _datetime_insights(datetime_profile):
    insights = []
    if datetime_profile is None or datetime_profile.empty:
        return insights

    for _, row in datetime_profile.head(5).iterrows():
        valid_dates = int(row.get("valid_dates", 0))
        missing_rate = float(row.get("missing_rate", 0.0))
        if valid_dates == 0:
            continue
        priority = "medium" if missing_rate >= 0.2 else "info"
        insights.append(
            Insight(
                priority=priority,
                theme="Time behavior",
                title="{} supports time-based analysis".format(row["column"]),
                observation="'{}' spans from {} to {} with {:.1%} missing values.".format(
                    row["column"], row.get("min"), row.get("max"), missing_rate
                ),
                impact="The platform can derive month, weekday, seasonality, and duration features from this field.",
                recommended_action="Use these time features to find weekly, seasonal, and promise-date delay patterns.",
                metric="valid_dates={}".format(valid_dates),
            )
        )
    return insights


def _target_behavior_insights(df, target_column):
    if target_column is None or target_column not in df.columns:
        return [
            Insight(
                priority="high",
                theme="Model readiness",
                title="No target column is confirmed",
                observation="The platform could not confidently identify a delay-risk or delivery-duration target.",
                impact="Training requires a reliable outcome column such as delivery status, delay flag, or delivery days.",
                recommended_action="Select or add a target before training the prediction system.",
            )
        ]

    target = df[target_column]
    numeric = pd.to_numeric(target, errors="coerce")
    if numeric.notna().sum() and numeric.nunique(dropna=True) <= 5:
        positive_rate = float((numeric > 0).mean())
        priority = "high" if positive_rate >= 0.35 else "medium" if positive_rate >= 0.15 else "info"
        return [
            Insight(
                priority=priority,
                theme="Delay behavior",
                title="Delay rate is {:.1%}".format(positive_rate),
                observation="'{}' behaves like a delay-risk target.".format(target_column),
                impact=_delay_rate_impact(positive_rate),
                recommended_action=_delay_rate_action(positive_rate),
                metric="positive_rate={:.2%}".format(positive_rate),
            )
        ]

    if numeric.notna().sum():
        median = float(numeric.median())
        p90 = float(numeric.quantile(0.90))
        priority = "high" if p90 > median * 2 and p90 > 1 else "medium"
        return [
            Insight(
                priority=priority,
                theme="Delivery duration",
                title="Typical duration is {:.2f}; p90 is {:.2f}".format(median, p90),
                observation="'{}' can support regression-style delivery-duration prediction.".format(
                    target_column
                ),
                impact="Long-tail delivery times can hide SLA risk even when the median looks acceptable.",
                recommended_action="Train the regression model and use the longest predicted durations as a planning queue.",
                metric="median={:.3f}, p90={:.3f}".format(median, p90),
            )
        ]
    return []


def _operational_group_insights(df, target_column):
    if target_column is None or target_column not in df.columns:
        return []

    schema = detect_columns(df)
    y = pd.to_numeric(df[target_column], errors="coerce")
    if y.notna().sum() < 10:
        return []

    insights = []
    preferred = [
        column
        for column in schema.categorical + schema.boolean
        if any(
            token in normalize_column_name(column)
            for token in ("supplier", "carrier", "city", "region", "route", "origin", "destination")
        )
        and 1 < df[column].nunique(dropna=True) <= 40
    ]
    for column in preferred[:4]:
        grouped = (
            pd.DataFrame({"group": df[column], "target": y})
            .dropna()
            .groupby("group")
            .agg(records=("target", "size"), average_target=("target", "mean"))
            .query("records >= 2")
            .sort_values(["average_target", "records"], ascending=[False, False])
        )
        if grouped.empty:
            continue
        top_group = grouped.iloc[0]
        baseline = float(y.mean())
        top_value = float(top_group["average_target"])
        if top_value <= baseline * 1.15 and top_value - baseline <= 0.05:
            continue
        insights.append(
            Insight(
                priority="high" if top_value >= baseline * 1.5 else "medium",
                theme="Operational hotspot",
                title="{} '{}' stands out".format(column, grouped.index[0]),
                observation="This group averages {:.2f} versus the dataset baseline of {:.2f}.".format(
                    top_value, baseline
                ),
                impact="The group may represent a risky supplier, route, carrier, or region.",
                recommended_action="Prioritize this group in the dashboard and compare process constraints against lower-risk groups.",
                metric="group_average={:.4f}, baseline={:.4f}, records={}".format(
                    top_value, baseline, int(top_group["records"])
                ),
            )
        )
    return insights


def _geography_column_insights(df):
    schema = detect_columns(df)
    columns = schema.numeric + schema.categorical
    geo_columns = [
        column
        for column in columns
        if any(
            token in normalize_column_name(column)
            for token in ("latitude", "longitude", "lat", "lon", "city", "region")
        )
    ]
    if not geo_columns:
        return []
    return [
        Insight(
            priority="info",
            theme="Geography",
            title="Geographic signals are present",
            observation="Detected geographic fields: {}.".format(", ".join(geo_columns[:8])),
            impact="Location, distance, weather, and regional capacity may help explain delivery behavior.",
            recommended_action="Run the external weather/geography intelligence check before training the final model.",
            metric="geo_columns={}".format(len(geo_columns)),
        )
    ]


def _best_target(df, requested):
    if requested in df.columns:
        return requested
    for column in ("is_delayed", "is_delayed_from_dates", "delay_days"):
        if column in df.columns:
            return column
    candidates = detect_columns(df).target_candidates
    return candidates[0] if candidates else None


def _missing_impact(rate):
    if rate >= 0.75:
        return "This field is too incomplete to trust as a primary model signal."
    if rate >= 0.4:
        return "This field may still help, but imputation will influence a large share of records."
    return "This field is usable, but missing-value handling should be explained to users."


def _missing_action(column, rate):
    if rate >= 0.75:
        return "Remove '{}' during modeling unless the missingness itself has business meaning.".format(
            column
        )
    if rate >= 0.4:
        return (
            "Validate why '{}' is missing and consider a missingness indicator in future iterations.".format(
                column
            )
        )
    return "Impute '{}' with the rule selected by the preprocessing engine.".format(column)


def _delay_rate_impact(rate):
    if rate >= 0.35:
        return "A substantial share of shipments need operational attention."
    if rate >= 0.15:
        return "Delay risk is meaningful enough for triage and monitoring workflows."
    return "Delays are relatively rare, so precision and recall should be watched carefully."


def _delay_rate_action(rate):
    if rate >= 0.35:
        return "Prioritize high-risk alerts, supplier reviews, and route-level dashboards."
    if rate >= 0.15:
        return "Use classification to rank daily work queues and regression to estimate planning impact."
    return "Tune thresholds carefully so the model still catches rare high-impact delays."
