"""Prediction helpers for saved pipeline artifacts."""

import numpy as np
import pandas as pd

from .cleaning import clean_dataframe
from .data_io import load_joblib
from .dual_prediction import is_portfolio, select_portfolio_model
from .exceptions import ArtifactError
from .schema import detect_columns, normalize_column_name


def load_model_artifact(path):
    """Load and validate a saved model artifact."""

    artifact = load_joblib(path)
    if not isinstance(artifact, dict) or ("pipeline" not in artifact and not is_portfolio(artifact)):
        raise ArtifactError("The selected file is not a valid delivery-delay model artifact.")
    return artifact


def predict_dataframe(artifact, df, model_key=None):
    """Generate predictions from a saved artifact and return an enriched DataFrame."""

    artifact, selected_model_key = select_portfolio_model(artifact, model_key=model_key)
    if "pipeline" not in artifact:
        raise ArtifactError("Artifact does not contain a trained pipeline.")

    pipeline = artifact["pipeline"]
    problem_type = artifact.get("problem_type", "classification")
    cleaned = clean_dataframe(df, drop_duplicates=False)

    predictions = pipeline.predict(cleaned)
    output = df.copy()
    output["prediction_system"] = selected_model_key
    output["prediction_target"] = artifact.get("target_column", "target")

    if problem_type == "classification":
        output["predicted_delay_class"] = predictions
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(cleaned)
            classes = list(getattr(pipeline.named_steps.get("model"), "classes_", []))
            if probabilities.shape[1] == 2:
                output["delay_probability"] = probabilities[:, 1]
                output["risk_score"] = output["delay_probability"] * 100.0
            for index, label in enumerate(classes):
                output["probability_{}".format(_safe_label(label))] = probabilities[:, index]
            if probabilities.shape[1] >= 2:
                output["prediction_confidence"] = probabilities.max(axis=1)
                output["confidence_level"] = output["prediction_confidence"].map(_confidence_level)
    else:
        output["predicted_delay_value"] = predictions
        output["estimated_delivery_days"] = np.maximum(pd.to_numeric(predictions, errors="coerce"), 0.0)
        output["estimated_delivery_date"] = _estimated_delivery_date(output, artifact, predictions)
        output["risk_score"] = _regression_risk_score(output["predicted_delay_value"])
        output["prediction_confidence"] = _regression_confidence(artifact)
        output["confidence_level"] = output["prediction_confidence"].map(_confidence_level)

    return output


def _safe_label(label):
    text = str(label).strip().lower().replace(" ", "_")
    return "".join(char for char in text if char.isalnum() or char == "_") or "class"


def _estimated_delivery_date(output, artifact, predictions):
    target = normalize_column_name(artifact.get("target_column", ""))
    offset = pd.to_numeric(pd.Series(predictions, index=output.index), errors="coerce")

    if "delay" in target:
        base_column = _find_expected_date_column(output)
        if base_column:
            base = pd.to_datetime(output[base_column], errors="coerce")
            return base + pd.to_timedelta(offset, unit="D")

    start_column = _find_start_date_column(output)
    if start_column:
        base = pd.to_datetime(output[start_column], errors="coerce")
        return base + pd.to_timedelta(offset.clip(lower=0), unit="D")

    expected_column = _find_expected_date_column(output)
    if expected_column:
        base = pd.to_datetime(output[expected_column], errors="coerce")
        return base + pd.to_timedelta(offset, unit="D")
    return pd.Series(pd.NaT, index=output.index)


def _find_expected_date_column(df):
    schema = detect_columns(df)
    for column in schema.datetime:
        normalized = normalize_column_name(column)
        if any(token in normalized for token in ("expected", "planned", "promise", "due", "estimated")):
            return column
    return None


def _find_start_date_column(df):
    schema = detect_columns(df)
    for column in schema.datetime:
        normalized = normalize_column_name(column)
        if any(token in normalized for token in ("order", "ship", "start", "created", "pickup", "dispatch", "sent")):
            return column
    return None


def _regression_risk_score(values):
    values = pd.to_numeric(values, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(np.nan, index=values.index)
    ranks = values.rank(pct=True)
    return ranks * 100.0


def _regression_confidence(artifact):
    metrics = artifact.get("metrics", {})
    r2 = metrics.get("r2")
    if r2 is not None:
        try:
            score = float(r2)
            return float(np.clip(0.50 + 0.50 * score, 0.05, 0.95))
        except Exception:
            pass
    return 0.50


def _confidence_level(score):
    try:
        score = float(score)
    except Exception:
        return "Unknown"
    if score >= 0.80:
        return "High confidence"
    if score >= 0.60:
        return "Medium confidence"
    return "Low confidence"
