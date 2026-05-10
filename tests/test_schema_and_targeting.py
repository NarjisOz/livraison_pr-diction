import pandas as pd

from delivery_delay.cleaning import clean_dataframe
from delivery_delay.schema import detect_columns
from delivery_delay.targeting import add_derived_delivery_targets, prepare_target


def test_detect_columns_finds_datetime_and_target_candidates():
    df = pd.DataFrame(
        {
            "shipment_id": ["A1", "A2"],
            "ship_date": ["2024-01-01", "2024-01-02"],
            "quantity": [10, 12],
            "delivery_status": ["delivered", "delayed"],
        }
    )

    schema = detect_columns(clean_dataframe(df))

    assert "ship_date" in schema.datetime
    assert "quantity" in schema.numeric
    assert "shipment_id" in schema.id_like
    assert "delivery_status" in schema.target_candidates


def test_prepare_target_derives_binary_delay_and_removes_leakage_columns():
    df = pd.DataFrame(
        {
            "shipment_id": ["A1", "A2", "A3", "A4"],
            "ship_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "expected_delivery_date": ["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"],
            "actual_delivery_date": ["2024-01-05", "2024-01-09", "2024-01-07", "2024-01-10"],
            "delivery_status": ["delivered", "delayed", "delivered", "delayed"],
            "quantity": [10, 20, 30, 40],
        }
    )

    data, sources = add_derived_delivery_targets(clean_dataframe(df))
    prepared = prepare_target(data, target_column="is_delayed")

    assert "is_delayed" in sources
    assert prepared.problem_type == "classification"
    assert prepared.y.tolist() == [0.0, 1.0, 0.0, 1.0]
    assert "delivery_status" not in prepared.X.columns
    assert "actual_delivery_date" not in prepared.X.columns

