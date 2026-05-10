import pandas as pd

from delivery_delay.prediction_dashboard import (
    anomaly_alerts,
    enrich_prediction_segments,
    executive_ai_summary,
    group_risk_table,
    prediction_kpis,
    prediction_reliability_indicators,
    recommended_group_columns,
    top_priority_rows,
)


def test_prediction_dashboard_builds_business_views():
    predictions = pd.DataFrame(
        {
            "shipment_id": ["S1", "S2", "S3", "S4"],
            "destination_city": ["Lagos", "Lagos", "Abuja", "Abuja"],
            "logistics_company": ["A", "B", "A", "B"],
            "delay_probability": [0.2, 0.45, 0.75, 0.9],
            "predicted_delay_class": [0, 0, 1, 1],
        }
    )

    enriched = enrich_prediction_segments(predictions)
    kpis = prediction_kpis(enriched)
    groups = recommended_group_columns(enriched)
    grouped = group_risk_table(enriched, "destination_city")
    priority = top_priority_rows(enriched, top_n=2)

    assert enriched["risk_segment"].tolist() == ["Low risk", "Medium risk", "High risk", "High risk"]
    assert kpis["high_risk_count"] == 2
    assert "destination_city" in groups
    assert grouped.iloc[0]["destination_city"] == "Abuja"
    assert priority["shipment_id"].tolist() == ["S4", "S3"]

    alerts = anomaly_alerts(enriched, group_column="destination_city")
    reliability = prediction_reliability_indicators(enriched)
    summary = executive_ai_summary(enriched, group_column="destination_city")

    assert alerts[0]["alert"] == "High delay-risk concentration"
    assert "Input completeness" in reliability["indicator"].tolist()
    assert any("Average delay risk" in item for item in summary)


def test_regression_prediction_dashboard_uses_estimated_days_and_confidence():
    predictions = pd.DataFrame(
        {
            "shipment_id": ["S1", "S2", "S3"],
            "destination_city": ["Lagos", "Abuja", "Kano"],
            "predicted_delay_value": [2.0, 7.5, 4.0],
            "estimated_delivery_days": [2.0, 7.5, 4.0],
            "prediction_confidence": [0.72, 0.72, 0.72],
        }
    )

    enriched = enrich_prediction_segments(predictions)
    kpis = prediction_kpis(enriched)
    priority = top_priority_rows(enriched, top_n=1)

    assert kpis["average_estimated_days"] == 4.5
    assert round(kpis["average_confidence"], 2) == 0.72
    assert priority.iloc[0]["shipment_id"] == "S2"
