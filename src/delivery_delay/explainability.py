"""Feature importance and SHAP explainability utilities."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .schema import normalize_column_name


def get_transformed_feature_names(pipeline):
    """Return feature names after preprocessing and optional selection."""

    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        return []

    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception:
        try:
            names = list(preprocessor.get_feature_names())
        except Exception:
            names = ["feature_{}".format(index) for index in range(_transformed_feature_count(pipeline))]

    selector = pipeline.named_steps.get("feature_selection")
    names = _apply_selector_names(selector, names)

    return names


def model_feature_importance(pipeline, top_n=30):
    """Return model importances or coefficients as a tidy DataFrame."""

    model = pipeline.named_steps.get("model")
    names = get_transformed_feature_names(pipeline)
    values = None

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        values = np.ravel(np.abs(coef).mean(axis=0) if getattr(coef, "ndim", 1) > 1 else np.abs(coef))

    if values is None:
        return pd.DataFrame(columns=["feature", "importance"])

    if len(names) != len(values):
        names = ["feature_{}".format(index) for index in range(len(values))]

    data = pd.DataFrame({"feature": names, "importance": values})
    data = data.sort_values("importance", ascending=False).head(top_n)
    return data.reset_index(drop=True)


def shap_importance(pipeline, X, max_rows=100, top_n=30):
    """Compute mean absolute SHAP values when SHAP supports the model."""

    try:
        import shap
    except Exception as exc:
        raise RuntimeError("SHAP is not installed or could not be imported: {}".format(exc))

    if X is None or len(X) == 0:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    sample = X.head(max_rows) if hasattr(X, "head") else X[:max_rows]
    transformer = Pipeline(steps=pipeline.steps[:-1])
    transformed = transformer.transform(sample)
    model = pipeline.named_steps["model"]
    names = get_transformed_feature_names(pipeline)

    explainer = shap.Explainer(model, transformed, feature_names=names)
    values = explainer(transformed)
    shap_values = values.values
    if getattr(shap_values, "ndim", 2) == 3:
        shap_values = np.mean(np.abs(shap_values), axis=2)
    else:
        shap_values = np.abs(shap_values)

    importance = np.mean(shap_values, axis=0)
    if len(names) != len(importance):
        names = ["feature_{}".format(index) for index in range(len(importance))]

    return (
        pd.DataFrame({"feature": names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def natural_language_feature_insights(importance_df, top_n=8):
    """Translate feature-importance rows into non-technical explanations."""

    if importance_df is None or importance_df.empty:
        return ["No feature-importance explanation is available for this model yet."]

    value_column = "importance" if "importance" in importance_df.columns else "mean_abs_shap"
    insights = []
    for _, row in importance_df.head(top_n).iterrows():
        feature = str(row["feature"])
        score = row[value_column]
        insights.append(
            "{} is influential in the model. In business terms: {} Impact score: {:.3f}.".format(
                _clean_feature_name(feature),
                _business_meaning(feature),
                float(score),
            )
        )
    return insights


def _transformed_feature_count(pipeline):
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        return len(preprocessor.get_feature_names_out())
    except Exception:
        try:
            return len(preprocessor.get_feature_names())
        except Exception:
            return 0


def _apply_selector_names(selector, names):
    if selector is None or selector == "passthrough":
        return names

    if hasattr(selector, "steps"):
        selected_names = names
        for _, step in selector.steps:
            if hasattr(step, "get_support"):
                support = step.get_support()
                selected_names = [name for name, keep in zip(selected_names, support) if keep]
        return selected_names

    if hasattr(selector, "get_support"):
        support = selector.get_support()
        return [name for name, keep in zip(names, support) if keep]

    return names


def _clean_feature_name(feature):
    text = feature.split("__")[-1]
    text = text.replace("_target_encoded", "")
    text = text.replace("_", " ")
    return text.strip().title()


def _business_meaning(feature):
    normalized = normalize_column_name(feature)
    if "duration" in normalized or "_to_" in normalized or "days" in normalized:
        return "timing and planned delivery duration are affecting delay risk."
    if "month" in normalized or "season" in normalized or "quarter" in normalized or "weekend" in normalized:
        return "seasonality or calendar timing changes delivery behavior."
    if "cost" in normalized or "price" in normalized or "fee" in normalized:
        return "shipping cost patterns may reflect route complexity or service level."
    if "distance" in normalized:
        return "longer or more complex routes may be more vulnerable to delay."
    if "quantity" in normalized or "weight" in normalized or "volume" in normalized:
        return "shipment size or handling load influences operational risk."
    if "supplier" in normalized:
        return "supplier behavior or readiness may affect delivery outcomes."
    if "region" in normalized or "city" in normalized or "route" in normalized:
        return "location and route conditions are important risk signals."
    if "company" in normalized or "carrier" in normalized or "logistics" in normalized:
        return "carrier performance differs across logistics partners."
    return "this field contains patterns that help separate risky and less risky deliveries."
