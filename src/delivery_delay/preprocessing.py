"""Preprocessing factories built on sklearn ColumnTransformer."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .schema import detect_columns
from .understanding import analyze_dataset


class IQRCapper(BaseEstimator, TransformerMixin):
    """Cap numerical outliers using training-set IQR limits."""

    def fit(self, X, y=None):
        df = _as_dataframe(X)
        self.columns_ = list(df.columns)
        self.bounds_ = {}
        for column in self.columns_:
            values = pd.to_numeric(df[column], errors="coerce")
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                self.bounds_[column] = (None, None)
            else:
                self.bounds_[column] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return self

    def transform(self, X):
        df = _as_dataframe(X, columns=getattr(self, "columns_", None)).copy()
        for column, (lower, upper) in getattr(self, "bounds_", {}).items():
            if lower is not None and upper is not None and column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce").clip(lower=lower, upper=upper)
        return df.values


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """Simple leakage-safe target mean encoder fitted only on the training fold."""

    def __init__(self, smoothing=10.0):
        self.smoothing = smoothing

    def fit(self, X, y):
        df = _as_dataframe(X).astype("object")
        target = _numeric_target(y)
        self.columns_ = list(df.columns)
        self.output_columns_ = ["target_encoded_{}".format(index) for index in range(len(self.columns_))]
        self.global_mean_ = float(target.mean()) if len(target) else 0.0
        self.mappings_ = {}

        for column in self.columns_:
            stats = pd.DataFrame({"category": df[column].fillna("__missing__"), "target": target})
            grouped = stats.groupby("category")["target"].agg(["mean", "count"])
            smooth = (grouped["count"] * grouped["mean"] + self.smoothing * self.global_mean_) / (
                grouped["count"] + self.smoothing
            )
            self.mappings_[column] = smooth.to_dict()
        return self

    def transform(self, X):
        df = _as_dataframe(X, columns=getattr(self, "columns_", None)).astype("object")
        encoded = pd.DataFrame(index=df.index)
        for column, output_column in zip(getattr(self, "columns_", []), getattr(self, "output_columns_", [])):
            mapping = self.mappings_.get(column, {})
            encoded[output_column] = (
                df[column].fillna("__missing__").map(mapping).fillna(self.global_mean_).astype(float)
            )
        return encoded.values

    def get_feature_names_out(self, input_features=None):
        if input_features is not None and len(input_features) == len(getattr(self, "columns_", [])):
            return np.array(["{}_target_encoded".format(column) for column in input_features], dtype=object)
        return np.array(getattr(self, "output_columns_", []), dtype=object)


def make_one_hot_encoder(max_categories=40, min_frequency=0.01):
    """Create a version-tolerant OneHotEncoder."""

    try:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            max_categories=max_categories,
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X, y=None, problem_type="auto", understanding=None, drop_identifier_columns=True):
    """Build a decision-driven ColumnTransformer from dataset understanding."""

    if understanding is None:
        analysis_df = X.copy()
        if y is not None:
            analysis_df = analysis_df.assign(__target__=pd.Series(y).reset_index(drop=True).values)
            understanding = analyze_dataset(analysis_df, target_column="__target__", problem_type=problem_type)
        else:
            understanding = analyze_dataset(analysis_df, problem_type=problem_type)

    plan = understanding.preprocessing_plan
    drop_columns = set()
    if drop_identifier_columns:
        for columns in plan.get("drop_columns", {}).values():
            drop_columns.update(columns)
    drop_columns = {column for column in drop_columns if column in X.columns}

    numeric_mean = _existing(plan.get("numeric_imputers", {}).get("mean", []), X, drop_columns)
    numeric_median = _existing(plan.get("numeric_imputers", {}).get("median", []), X, drop_columns)
    categorical_one_hot = _existing(plan.get("categorical_encoders", {}).get("one_hot", []), X, drop_columns)
    categorical_target = _existing(plan.get("categorical_encoders", {}).get("target_encoding", []), X, drop_columns)
    capped_columns = set(_existing(plan.get("outlier_handlers", {}).get("iqr_capping", []), X, drop_columns))

    transformers = []
    transformers.extend(_numeric_transformers("mean", numeric_mean, capped_columns))
    transformers.extend(_numeric_transformers("median", numeric_median, capped_columns))

    if categorical_one_hot:
        one_hot_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical_one_hot", one_hot_pipeline, categorical_one_hot))

    if categorical_target:
        target_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("target_encoder", TargetMeanEncoder()),
                ("scaler", RobustScaler()),
            ]
        )
        transformers.append(("categorical_target", target_pipeline, categorical_target))

    if not transformers:
        fallback = _fallback_transformers(X, drop_columns)
        transformers.extend(fallback)

    if not transformers:
        raise ValueError("No usable model features were detected after adaptive preprocessing decisions.")

    try:
        return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)
    except TypeError:
        return ColumnTransformer(transformers=transformers, remainder="drop")


def make_feature_selector(problem_type, percentile=80):
    """Create a lightweight supervised feature selector."""

    if percentile is None or percentile >= 100:
        return VarianceThreshold()

    percentile = int(np.clip(percentile, 1, 100))
    score_func = f_classif if problem_type == "classification" else f_regression
    return Pipeline(
        steps=[
            ("variance", VarianceThreshold()),
            ("supervised", SelectPercentile(score_func=score_func, percentile=percentile)),
        ]
    )


def _numeric_transformers(strategy, columns, capped_columns):
    transformers = []
    capped = [column for column in columns if column in capped_columns]
    uncapped = [column for column in columns if column not in capped_columns]
    if capped:
        transformers.append(
            (
                "numeric_{}_capped".format(strategy),
                Pipeline(
                    steps=[
                        ("capper", IQRCapper()),
                        ("imputer", SimpleImputer(strategy=strategy)),
                        ("scaler", RobustScaler()),
                    ]
                ),
                capped,
            )
        )
    if uncapped:
        transformers.append(
            (
                "numeric_{}".format(strategy),
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy=strategy)),
                        ("scaler", RobustScaler()),
                    ]
                ),
                uncapped,
            )
        )
    return transformers


def _fallback_transformers(X, drop_columns):
    schema = detect_columns(X)
    numeric_columns = [
        column
        for column in schema.numeric + schema.boolean
        if column not in drop_columns and column in X.columns
    ]
    categorical_columns = [
        column
        for column in schema.categorical
        if column not in drop_columns and column in X.columns
    ]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric_fallback",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical_fallback",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_one_hot_encoder())]),
                categorical_columns,
            )
        )
    return transformers


def _existing(columns, X, drop_columns):
    return [column for column in columns if column in X.columns and column not in drop_columns]


def _as_dataframe(X, columns=None):
    if isinstance(X, pd.DataFrame):
        return X.copy()
    if columns is not None and len(columns) == np.asarray(X).shape[1]:
        return pd.DataFrame(X, columns=columns)
    return pd.DataFrame(X)


def _numeric_target(y):
    target = pd.Series(y).reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(target):
        return pd.to_numeric(target, errors="coerce").fillna(pd.to_numeric(target, errors="coerce").median())
    codes, _ = pd.factorize(target)
    encoded = pd.Series(codes, dtype="float")
    encoded[target.isna()] = np.nan
    return encoded.fillna(encoded.median() if encoded.notna().any() else 0.0)
