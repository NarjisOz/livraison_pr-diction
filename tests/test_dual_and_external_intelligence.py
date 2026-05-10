import pandas as pd

from delivery_delay.dual_prediction import discover_prediction_tasks
from delivery_delay.external_intelligence import evaluate_external_weather_intelligence


def test_discover_prediction_tasks_supports_classification_and_regression():
    rows = 24
    expected = pd.date_range("2024-01-01", periods=rows)
    actual = expected + pd.to_timedelta([index % 12 for index in range(rows)], unit="D")
    df = pd.DataFrame(
        {
            "shipment_id": ["S{}".format(index) for index in range(rows)],
            "origin_city": ["Lagos", "Abuja"] * 12,
            "destination_city": ["Kano", "Lagos", "Abuja"] * 8,
            "expected_delivery_date": expected,
            "actual_delivery_date": actual,
            "delivery_status": ["delayed" if index % 12 > 0 else "delivered" for index in range(rows)],
        }
    )

    tasks = discover_prediction_tasks(df)
    compatible = {task.key: task for task in tasks if task.compatible}

    assert compatible["classification"].target_column in ("is_delayed", "is_delayed_from_dates")
    assert compatible["regression"].target_column == "delay_days"


def test_external_weather_intelligence_enriches_but_does_not_force_without_predictive_test(tmp_path):
    weather_path = tmp_path / "nigeria_cities_weather_data.csv"
    weather = pd.DataFrame(
        {
            "country": ["NG", "NG", "NG"],
            "city": ["Lagos", "Abuja", "Kano"],
            "latitude": [6.45, 9.07, 12.0],
            "longitude": [3.39, 7.49, 8.52],
            "temp": [303.0, 301.0, 306.0],
            "humidity": [75, 52, 33],
            "wind_speed": [4.0, 2.0, 3.0],
            "cloud": [40, 25, 12],
            "region": ["Lagos", "FCT", "Kano"],
            "population": [1000, 800, 700],
        }
    )
    weather.to_csv(weather_path, index=False)
    df = pd.DataFrame(
        {
            "origin_city": ["Lagos", "Abuja", "Kano", "Lagos"] * 10,
            "destination_city": ["Abuja", "Kano", "Lagos", "Abuja"] * 10,
            "expected_delivery_date": pd.date_range("2024-02-01", periods=40),
            "actual_delivery_date": pd.date_range("2024-02-03", periods=40),
        }
    )

    result = evaluate_external_weather_intelligence(
        df,
        weather_path=weather_path,
        run_predictive_test=False,
    )

    assert result.available
    assert result.match_rate == 1.0
    assert "weather_route_distance_km" in result.feature_columns
    assert not result.recommended
    assert result.enriched_data is not None
