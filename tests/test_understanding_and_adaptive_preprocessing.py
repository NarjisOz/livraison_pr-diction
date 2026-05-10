import numpy as np
import pandas as pd

from delivery_delay.preprocessing import build_preprocessor
from delivery_delay.understanding import analyze_dataset


def test_understanding_generates_scores_and_decisions():
    df = pd.DataFrame(
        {
            "shipment_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "ship_date": pd.date_range("2024-01-01", periods=6),
            "destination_city": ["A", "B", "A", "B", "C", "C"],
            "supplier_name": ["supplier_{}".format(i) for i in range(6)],
            "shipping_cost": [10, 12, 13, 1000, 11, 14],
            "mostly_missing": [np.nan, np.nan, np.nan, np.nan, np.nan, 1],
            "is_delayed": [0, 0, 1, 1, 0, 1],
        }
    )

    understanding = analyze_dataset(df, target_column="is_delayed", problem_type="classification")
    plan = understanding.preprocessing_plan

    assert understanding.quality_score > 0
    assert understanding.compatibility_score > 0
    assert "mostly_missing" in plan["drop_columns"]["high_missing_or_constant_or_id"]
    assert "shipment_id" in plan["drop_columns"]["high_missing_or_constant_or_id"]
    assert "destination_city" in plan["categorical_encoders"]["one_hot"]
    assert "shipping_cost" in plan["numeric_imputers"]["median"]


def test_adaptive_preprocessor_fits_from_understanding_plan():
    X = pd.DataFrame(
        {
            "destination_city": ["A", "B", "A", "B", "C", "C"],
            "supplier_name": ["supplier_{}".format(i) for i in range(6)],
            "shipping_cost": [10, 12, 13, 1000, 11, 14],
        }
    )
    y = pd.Series([0, 0, 1, 1, 0, 1])
    understanding = analyze_dataset(X.assign(__target__=y), target_column="__target__", problem_type="classification")
    preprocessor = build_preprocessor(X, y=y, problem_type="classification", understanding=understanding)

    transformed = preprocessor.fit_transform(X, y)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0

