"""End-to-end model training and model selection."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .cleaning import clean_dataframe
from .config import DEFAULT_MODEL_PATH, RANDOM_STATE
from .data_io import save_joblib
from .evaluation import evaluate_classifier, evaluate_regressor, leaderboard_from_results
from .exceptions import ModelTrainingError
from .features import DateFeatureEngineer, LogisticsFeatureEngineer
from .modeling import get_model_specs
from .preprocessing import build_preprocessor, make_feature_selector
from .schema import detect_columns
from .targeting import prepare_target
from .understanding import analyze_dataset, understanding_to_dict


@dataclass
class TrainingResult:
    best_model_name: str
    best_score: float
    leaderboard: pd.DataFrame
    metrics: Dict[str, Any]
    artifact: Dict[str, Any]
    artifact_path: Optional[str]
    failed_models: List[Dict[str, str]]


def train_and_select_model(
    df,
    target_column=None,
    problem_type="auto",
    exclude_columns=None,
    output_path=DEFAULT_MODEL_PATH,
    test_size=0.2,
    random_state=RANDOM_STATE,
    feature_selection_percentile=80,
    max_rows=100000,
):
    """Train multiple models, select the best one, and save one complete artifact."""

    cleaned = clean_dataframe(df)
    prepared = prepare_target(
        cleaned,
        target_column=target_column,
        problem_type=problem_type,
        exclude_columns=exclude_columns,
    )
    raw_understanding = analyze_dataset(
        cleaned,
        target_column=prepared.target_column,
        problem_type=prepared.problem_type,
        known_leakage_columns=prepared.leakage_columns,
    )

    X, y = _sample_if_needed(prepared.X, prepared.y, max_rows=max_rows, random_state=random_state)
    if X.shape[0] < 10:
        raise ModelTrainingError("At least 10 labeled rows are required to train reliable models.")

    stratify = _safe_stratify(y, prepared.problem_type)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    probe_engineer = DateFeatureEngineer()
    probe_logistics = LogisticsFeatureEngineer()
    probe_base = probe_engineer.fit(X_train).transform(X_train.head(min(len(X_train), 500)))
    probe_X = probe_logistics.fit(probe_base).transform(probe_base)
    probe_y = pd.Series(y_train).head(len(probe_X)).reset_index(drop=True)
    feature_understanding = analyze_dataset(
        probe_X.assign(__target__=probe_y.values),
        target_column="__target__",
        problem_type=prepared.problem_type,
    )
    preprocessor = build_preprocessor(
        probe_X,
        y=probe_y,
        problem_type=prepared.problem_type,
        understanding=feature_understanding,
    )
    selector = make_feature_selector(prepared.problem_type, feature_selection_percentile)
    contamination_report = _contamination_report(X_train, X_test)

    model_results = []
    best = None

    for spec in get_model_specs(prepared.problem_type):
        pipeline = Pipeline(
            steps=[
                ("features", DateFeatureEngineer()),
                ("logistics_features", LogisticsFeatureEngineer()),
                ("preprocessor", clone(preprocessor)),
                ("feature_selection", clone(selector) if selector != "passthrough" else "passthrough"),
                ("model", clone(spec.estimator)),
            ]
        )

        try:
            pipeline.fit(X_train, y_train)
            if prepared.problem_type == "classification":
                metrics, y_pred, y_proba = evaluate_classifier(pipeline, X_test, y_test)
            else:
                metrics, y_pred, y_proba = evaluate_regressor(pipeline, X_test, y_test)

            result = {
                "model_name": spec.name,
                "family": spec.family,
                "notes": spec.notes,
                "status": "ok",
                "metrics": metrics,
                "pipeline": pipeline,
                "y_pred": y_pred,
                "y_proba": y_proba,
            }
            model_results.append(result)
            if best is None or metrics["selection_score"] > best["metrics"]["selection_score"]:
                best = result
        except Exception as exc:
            model_results.append(
                {
                    "model_name": spec.name,
                    "family": spec.family,
                    "status": "failed",
                    "metrics": {},
                    "error": str(exc),
                }
            )

    if best is None:
        failures = [result.get("error", "unknown error") for result in model_results]
        raise ModelTrainingError("All model candidates failed: {}".format("; ".join(failures[:5])))

    leaderboard = leaderboard_from_results(model_results)
    artifact = _build_artifact(
        best=best,
        prepared=prepared,
        leaderboard=leaderboard,
        model_results=model_results,
        random_state=random_state,
        test_size=test_size,
        raw_understanding=raw_understanding,
        feature_understanding=feature_understanding,
        contamination_report=contamination_report,
    )

    artifact_path = None
    if output_path:
        artifact_path = str(save_joblib(artifact, output_path))

    failed_models = [
        {"model": result["model_name"], "error": result.get("error", "")}
        for result in model_results
        if result["status"] != "ok"
    ]

    return TrainingResult(
        best_model_name=best["model_name"],
        best_score=float(best["metrics"]["selection_score"]),
        leaderboard=leaderboard,
        metrics=best["metrics"],
        artifact=artifact,
        artifact_path=artifact_path,
        failed_models=failed_models,
    )


def _sample_if_needed(X, y, max_rows, random_state):
    if not max_rows or len(X) <= max_rows:
        return X, y
    sampled = X.assign(__target__=y).sample(n=max_rows, random_state=random_state)
    return sampled.drop(columns=["__target__"]), sampled["__target__"]


def _safe_stratify(y, problem_type):
    if problem_type != "classification":
        return None
    counts = pd.Series(y).value_counts(dropna=False)
    if len(counts) < 2 or counts.min() < 2:
        return None
    return y


def _build_artifact(
    best,
    prepared,
    leaderboard,
    model_results,
    random_state,
    test_size,
    raw_understanding,
    feature_understanding,
    contamination_report,
):
    serializable_results = []
    for result in model_results:
        serializable_results.append(
            {
                "model_name": result["model_name"],
                "family": result.get("family"),
                "notes": result.get("notes", ""),
                "status": result["status"],
                "metrics": result.get("metrics", {}),
                "error": result.get("error", ""),
            }
        )

    return {
        "pipeline": best["pipeline"],
        "best_model_name": best["model_name"],
        "problem_type": prepared.problem_type,
        "target_column": prepared.target_column,
        "metrics": best["metrics"],
        "leaderboard": leaderboard,
        "model_results": serializable_results,
        "derived_targets": prepared.derived_targets,
        "source_columns": prepared.source_columns,
        "leakage_columns": prepared.leakage_columns,
        "dataset_understanding": understanding_to_dict(raw_understanding),
        "feature_understanding": understanding_to_dict(feature_understanding),
        "preprocessing_plan": feature_understanding.preprocessing_plan,
        "contamination_report": contamination_report,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "random_state": random_state,
        "test_size": test_size,
    }


def _contamination_report(X_train, X_test):
    report = {
        "exact_row_overlap_count": 0,
        "exact_row_overlap_rate": 0.0,
        "identifier_overlap": [],
        "risk_level": "low",
    }
    if X_train.empty or X_test.empty:
        return report

    train_hashes = pd.util.hash_pandas_object(X_train.astype(str), index=False)
    test_hashes = pd.util.hash_pandas_object(X_test.astype(str), index=False)
    overlap = int(test_hashes.isin(set(train_hashes)).sum())
    report["exact_row_overlap_count"] = overlap
    report["exact_row_overlap_rate"] = float(overlap / max(len(X_test), 1))

    schema = detect_columns(pd.concat([X_train, X_test], axis=0))
    for column in schema.id_like[:10]:
        if column in X_train.columns and column in X_test.columns:
            train_values = set(X_train[column].dropna().astype(str))
            test_values = set(X_test[column].dropna().astype(str))
            shared = train_values.intersection(test_values)
            if shared:
                report["identifier_overlap"].append(
                    {
                        "column": column,
                        "overlap_count": len(shared),
                        "test_overlap_rate": len(shared) / max(len(test_values), 1),
                    }
                )

    if report["exact_row_overlap_rate"] > 0.02 or any(
        item["test_overlap_rate"] > 0.1 for item in report["identifier_overlap"]
    ):
        report["risk_level"] = "high"
    elif report["exact_row_overlap_count"] > 0 or report["identifier_overlap"]:
        report["risk_level"] = "medium"
    return report
