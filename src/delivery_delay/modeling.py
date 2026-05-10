"""Candidate model definitions for classification and regression."""

from dataclasses import dataclass
from typing import Any, List

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

from .config import N_JOBS, RANDOM_STATE

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
except Exception:
    HistGradientBoostingClassifier = None
    HistGradientBoostingRegressor = None


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    family: str
    notes: str = ""


def get_model_specs(problem_type):
    """Return baseline and optional advanced model candidates."""

    if problem_type == "classification":
        specs = [
            ModelSpec(
                "Logistic Regression",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
                "linear",
                "Interpretable baseline for binary delay risk.",
            ),
            ModelSpec(
                "Random Forest",
                RandomForestClassifier(
                    n_estimators=250,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                ),
                "tree_ensemble",
                "Robust nonlinear model with native feature importance.",
            ),
            _sklearn_classifier_boosting_spec(),
        ]
        specs.extend(_optional_classifiers())
        return specs

    specs = [
        ModelSpec(
            "Ridge Regression",
            Ridge(alpha=1.0, random_state=RANDOM_STATE),
            "linear",
            "Stable linear baseline for delivery-time regression.",
        ),
        ModelSpec(
            "Random Forest Regressor",
            RandomForestRegressor(
                n_estimators=250,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            ),
            "tree_ensemble",
            "Robust nonlinear regressor.",
        ),
        _sklearn_regressor_boosting_spec(),
    ]
    specs.extend(_optional_regressors())
    return specs


def _sklearn_classifier_boosting_spec():
    if HistGradientBoostingClassifier is not None:
        return ModelSpec(
            "Hist Gradient Boosting",
            HistGradientBoostingClassifier(max_iter=180, learning_rate=0.06, random_state=RANDOM_STATE),
            "boosting",
            "Fast sklearn gradient boosting baseline.",
        )
    return ModelSpec(
        "Gradient Boosting",
        GradientBoostingClassifier(n_estimators=150, learning_rate=0.06, random_state=RANDOM_STATE),
        "boosting",
        "Portable sklearn gradient boosting baseline.",
    )


def _sklearn_regressor_boosting_spec():
    if HistGradientBoostingRegressor is not None:
        return ModelSpec(
            "Hist Gradient Boosting Regressor",
            HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06, random_state=RANDOM_STATE),
            "boosting",
            "Fast sklearn gradient boosting regressor.",
        )
    return ModelSpec(
        "Gradient Boosting Regressor",
        GradientBoostingRegressor(n_estimators=150, learning_rate=0.06, random_state=RANDOM_STATE),
        "boosting",
        "Portable sklearn gradient boosting regressor.",
    )


def _optional_classifiers() -> List[ModelSpec]:
    specs = []
    try:
        from xgboost import XGBClassifier

        specs.append(
            ModelSpec(
                "XGBoost",
                XGBClassifier(
                    n_estimators=250,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                ),
                "boosting",
                "Optional high-performing gradient boosting model.",
            )
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        specs.append(
            ModelSpec(
                "LightGBM",
                LGBMClassifier(
                    n_estimators=250,
                    learning_rate=0.05,
                    num_leaves=31,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                    verbose=-1,
                ),
                "boosting",
                "Optional fast gradient boosting model for tabular data.",
            )
        )
    except Exception:
        pass

    return specs


def _optional_regressors() -> List[ModelSpec]:
    specs = []
    try:
        from xgboost import XGBRegressor

        specs.append(
            ModelSpec(
                "XGBoost Regressor",
                XGBRegressor(
                    n_estimators=250,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                ),
                "boosting",
                "Optional high-performing gradient boosting regressor.",
            )
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor

        specs.append(
            ModelSpec(
                "LightGBM Regressor",
                LGBMRegressor(
                    n_estimators=250,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                    verbose=-1,
                ),
                "boosting",
                "Optional fast gradient boosting regressor.",
            )
        )
    except Exception:
        pass

    return specs
