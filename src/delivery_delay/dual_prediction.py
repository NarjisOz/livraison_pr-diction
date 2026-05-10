"""Dual classification/regression prediction portfolio orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd

from .config import DEFAULT_MODEL_PATH, MODEL_PORTFOLIO_PATH
from .data_io import save_joblib
from .schema import detect_columns, infer_problem_type, normalize_column_name
from .targeting import add_derived_delivery_targets, prepare_target
from .training import train_and_select_model


@dataclass
class PredictionTask:
    """A compatible prediction task discovered in the active dataset."""

    key: str
    label: str
    target_column: str
    problem_type: str
    compatible: bool
    reason: str
    rows: int = 0
    unique_targets: int = 0


def discover_prediction_tasks(df):
    """Find classification and regression targets that can be trained honestly."""

    if df is None or df.empty:
        return []

    data, _ = add_derived_delivery_targets(df)
    candidates = {
        "classification": _classification_candidates(data),
        "regression": _regression_candidates(data),
    }
    tasks = []
    for problem_type, target_candidates in candidates.items():
        seen = set()
        for target_column in target_candidates:
            if target_column in seen or target_column not in data.columns:
                continue
            seen.add(target_column)
            tasks.append(_validate_task(data, target_column, problem_type))
            if tasks[-1].compatible:
                break

    if not any(task.key == "classification" for task in tasks):
        tasks.append(
            PredictionTask(
                key="classification",
                label="Delay Risk Classification",
                target_column="",
                problem_type="classification",
                compatible=False,
                reason="No binary delay-status, delay-flag, or delayed/not-delayed target was found.",
            )
        )

    if not any(task.key == "regression" for task in tasks):
        tasks.append(
            PredictionTask(
                key="regression",
                label="Delivery Duration Regression",
                target_column="",
                problem_type="regression",
                compatible=False,
                reason="No continuous delay-days, delivery-duration, or lead-time target was found.",
            )
        )

    return tasks


def tasks_to_frame(tasks):
    """Return discovered tasks as a dataframe."""

    return pd.DataFrame([asdict(task) for task in tasks])


def train_dual_prediction_system(
    df,
    tasks=None,
    selected_targets=None,
    exclude_columns=None,
    feature_selection_percentile=80,
    max_rows=100000,
    output_path=MODEL_PORTFOLIO_PATH,
    active_model_output_path=DEFAULT_MODEL_PATH,
    metadata=None,
):
    """Train all compatible classification/regression tasks and save a portfolio."""

    tasks = tasks or discover_prediction_tasks(df)
    selected_targets = selected_targets or {}
    trained_models = {}
    result_summaries = {}
    failed_models = []

    for task in tasks:
        if not task.compatible:
            result_summaries[task.key] = {
                "status": "skipped",
                "reason": task.reason,
                "problem_type": task.problem_type,
                "target_column": task.target_column,
            }
            continue

        target_column = selected_targets.get(task.key, task.target_column) or task.target_column
        try:
            result = train_and_select_model(
                df,
                target_column=target_column,
                problem_type=task.problem_type,
                exclude_columns=exclude_columns,
                output_path=None,
                feature_selection_percentile=feature_selection_percentile,
                max_rows=max_rows,
            )
            trained_models[task.key] = result.artifact
            result_summaries[task.key] = {
                "status": "trained",
                "best_model_name": result.best_model_name,
                "best_score": result.best_score,
                "target_column": target_column,
                "problem_type": task.problem_type,
                "metrics": result.metrics,
                "leaderboard": result.leaderboard,
            }
            failed_models.extend(
                [
                    {
                        "task": task.key,
                        "model": item.get("model", ""),
                        "error": item.get("error", ""),
                    }
                    for item in result.failed_models
                ]
            )
        except Exception as exc:
            result_summaries[task.key] = {
                "status": "failed",
                "reason": str(exc),
                "problem_type": task.problem_type,
                "target_column": target_column,
            }
            failed_models.append({"task": task.key, "model": "task", "error": str(exc)})

    comparison = compare_prediction_systems(result_summaries)
    recommended_key = comparison.get("recommended_model_key")

    portfolio = {
        "artifact_type": "dual_prediction_portfolio",
        "models": trained_models,
        "results": result_summaries,
        "comparison": comparison,
        "recommended_model_key": recommended_key,
        "failed_models": failed_models,
        "metadata": metadata or {},
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }

    portfolio_path = None
    if output_path:
        portfolio_path = str(save_joblib(portfolio, output_path))

    if recommended_key in trained_models and active_model_output_path:
        save_joblib(trained_models[recommended_key], active_model_output_path)

    return {
        "portfolio": portfolio,
        "portfolio_path": portfolio_path,
        "recommended_model_key": recommended_key,
        "failed_models": failed_models,
    }


def compare_prediction_systems(result_summaries):
    """Explain how classification and regression should be used."""

    classification = result_summaries.get("classification", {})
    regression = result_summaries.get("regression", {})
    messages = []
    trained = [
        key
        for key, result in result_summaries.items()
        if isinstance(result, dict) and result.get("status") == "trained"
    ]

    if "classification" in trained:
        metrics = classification.get("metrics", {})
        auc = metrics.get("roc_auc")
        f1 = metrics.get("f1_weighted")
        messages.append(
            "Classification is useful for daily triage because it separates delayed vs on-time risk. "
            "Current quality: ROC-AUC={} and weighted F1={}.".format(_fmt(auc), _fmt(f1))
        )
    elif classification:
        messages.append(
            "Classification was skipped: {}.".format(classification.get("reason", "not compatible"))
        )

    if "regression" in trained:
        metrics = regression.get("metrics", {})
        messages.append(
            "Regression is useful for capacity planning because it estimates delivery duration or delay days. "
            "Current quality: R2={} and RMSE={}.".format(_fmt(metrics.get("r2")), _fmt(metrics.get("rmse")))
        )
    elif regression:
        messages.append("Regression was skipped: {}.".format(regression.get("reason", "not compatible")))

    recommended_key = _recommend_model_key(classification, regression)
    if recommended_key == "classification":
        messages.append(
            "Recommended default: classification, because the dataset is currently stronger for risk alerts."
        )
    elif recommended_key == "regression":
        messages.append(
            "Recommended default: regression, because duration estimates are currently more reliable."
        )
    else:
        messages.append("No default model is recommended yet; train at least one compatible task.")

    return {
        "trained_tasks": trained,
        "recommended_model_key": recommended_key,
        "messages": messages,
        "classification_use_case": "Alert high-risk shipments before SLA breach.",
        "regression_use_case": "Estimate delivery days, delay days, and capacity pressure.",
    }


def portfolio_model_options(portfolio):
    """Return trained model keys with readable labels."""

    if not is_portfolio(portfolio):
        problem_type = portfolio.get("problem_type", "model") if isinstance(portfolio, dict) else "model"
        return [("single", problem_type.title())]

    options = []
    recommended = portfolio.get("recommended_model_key")
    for key, artifact in portfolio.get("models", {}).items():
        target = artifact.get("target_column", "target")
        suffix = " (Recommended)" if key == recommended else ""
        options.append((key, "{} - {}{}".format(key.title(), target, suffix)))
    return sorted(options, key=lambda item: 0 if item[0] == recommended else 1)


def is_portfolio(artifact):
    """Return True when a loaded artifact stores multiple trained systems."""

    return isinstance(artifact, dict) and artifact.get("artifact_type") == "dual_prediction_portfolio"


def select_portfolio_model(artifact, model_key=None):
    """Return the concrete model artifact to use for prediction."""

    if not is_portfolio(artifact):
        return artifact, "single"

    models = artifact.get("models", {})
    if not models:
        raise ValueError("The model portfolio does not contain any trained models.")
    model_key = model_key or artifact.get("recommended_model_key") or next(iter(models))
    if model_key not in models:
        model_key = artifact.get("recommended_model_key") or next(iter(models))
    return models[model_key], model_key


def _classification_candidates(data):
    preferred = [column for column in ("is_delayed", "is_delayed_from_dates") if column in data.columns]
    schema = detect_columns(data)
    candidates = []
    for column in schema.target_candidates:
        normalized = normalize_column_name(column)
        if any(token in normalized for token in ("delay", "delayed", "late", "status")):
            values = data[column].dropna()
            inferred = infer_problem_type(values, requested="auto")
            if inferred == "classification":
                candidates.append(column)
    return list(dict.fromkeys(preferred + candidates))


def _regression_candidates(data):
    preferred = [column for column in ("delay_days",) if column in data.columns]
    schema = detect_columns(data)
    candidates = []
    for column in schema.numeric + schema.target_candidates:
        if column not in data.columns:
            continue
        normalized = normalize_column_name(column)
        if any(
            token in normalized for token in ("duration", "delivery_time", "lead_time", "delay_days", "days")
        ):
            values = data[column].dropna()
            if values.nunique(dropna=True) > 10:
                candidates.append(column)
    return list(dict.fromkeys(preferred + candidates))


def _validate_task(data, target_column, problem_type):
    label = (
        "Delay Risk Classification" if problem_type == "classification" else "Delivery Duration Regression"
    )
    try:
        prepared = prepare_target(data, target_column=target_column, problem_type=problem_type)
    except Exception as exc:
        return PredictionTask(
            key=problem_type,
            label=label,
            target_column=target_column,
            problem_type=problem_type,
            compatible=False,
            reason=str(exc),
        )

    y = pd.Series(prepared.y).dropna()
    unique_targets = int(y.nunique(dropna=True))
    if len(y) < 10:
        return PredictionTask(
            key=problem_type,
            label=label,
            target_column=target_column,
            problem_type=problem_type,
            compatible=False,
            reason="At least 10 labeled rows are required.",
            rows=int(len(y)),
            unique_targets=unique_targets,
        )
    if problem_type == "classification" and unique_targets < 2:
        return PredictionTask(
            key=problem_type,
            label=label,
            target_column=target_column,
            problem_type=problem_type,
            compatible=False,
            reason="Classification needs at least two target classes.",
            rows=int(len(y)),
            unique_targets=unique_targets,
        )
    if problem_type == "regression" and unique_targets < 8:
        return PredictionTask(
            key=problem_type,
            label=label,
            target_column=target_column,
            problem_type=problem_type,
            compatible=False,
            reason="Regression needs a continuous target with enough distinct values.",
            rows=int(len(y)),
            unique_targets=unique_targets,
        )
    return PredictionTask(
        key=problem_type,
        label=label,
        target_column=target_column,
        problem_type=problem_type,
        compatible=True,
        reason="Compatible: {} labeled rows and {} distinct target values.".format(len(y), unique_targets),
        rows=int(len(y)),
        unique_targets=unique_targets,
    )


def _recommend_model_key(classification, regression):
    cls_ok = classification.get("status") == "trained"
    reg_ok = regression.get("status") == "trained"
    if cls_ok and not reg_ok:
        return "classification"
    if reg_ok and not cls_ok:
        return "regression"
    if not cls_ok and not reg_ok:
        return None

    cls_metrics = classification.get("metrics", {})
    reg_metrics = regression.get("metrics", {})
    cls_signal = cls_metrics.get("roc_auc", cls_metrics.get("f1_weighted", 0.0)) or 0.0
    reg_r2 = reg_metrics.get("r2", -999.0)
    reg_signal = reg_r2 if reg_r2 is not None else -999.0

    if cls_signal >= 0.70:
        return "classification"
    if reg_signal >= 0.45:
        return "regression"
    if cls_signal >= 0.55:
        return "classification"
    return "regression" if reg_signal > 0 else "classification"


def _fmt(value):
    if value is None:
        return "n/a"
    try:
        return "{:.3f}".format(float(value))
    except Exception:
        return str(value)
