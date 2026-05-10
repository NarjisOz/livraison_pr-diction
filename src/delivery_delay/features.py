"""Feature engineering transformers for sklearn pipelines."""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .schema import detect_columns, normalize_column_name


class DateFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create robust datetime-derived features inside an sklearn Pipeline."""

    def __init__(self, max_datetime_columns=8, drop_original_datetime=True):
        self.max_datetime_columns = max_datetime_columns
        self.drop_original_datetime = drop_original_datetime

    def fit(self, X, y=None):
        df = self._as_dataframe(X)
        self.input_columns_ = list(df.columns)
        schema = detect_columns(df)
        self.datetime_columns_ = schema.datetime[: self.max_datetime_columns]
        self.date_pairs_ = self._discover_date_pairs(self.datetime_columns_)
        return self

    def transform(self, X):
        df = self._as_dataframe(X).copy()

        for column in getattr(self, "input_columns_", []):
            if column not in df.columns:
                df[column] = np.nan
        df = df[getattr(self, "input_columns_", list(df.columns))]

        for column in getattr(self, "datetime_columns_", []):
            values = pd.to_datetime(df[column], errors="coerce")
            prefix = normalize_column_name(column)
            df[prefix + "_year"] = values.dt.year
            df[prefix + "_month"] = values.dt.month
            df[prefix + "_day"] = values.dt.day
            df[prefix + "_dayofweek"] = values.dt.dayofweek
            df[prefix + "_quarter"] = values.dt.quarter
            df[prefix + "_season"] = values.dt.month.map(_season_from_month)
            df[prefix + "_is_weekend"] = values.dt.dayofweek.isin([5, 6]).astype(float)
            df[prefix + "_month_sin"] = np.sin(2 * np.pi * values.dt.month / 12)
            df[prefix + "_month_cos"] = np.cos(2 * np.pi * values.dt.month / 12)
            df[prefix + "_days_from_min"] = self._days_from_min(values)

        for start_column, end_column in getattr(self, "date_pairs_", []):
            start = pd.to_datetime(df[start_column], errors="coerce")
            end = pd.to_datetime(df[end_column], errors="coerce")
            feature_name = "{}_to_{}_days".format(
                normalize_column_name(start_column),
                normalize_column_name(end_column),
            )
            df[feature_name] = (end - start).dt.total_seconds() / 86400.0
            df[feature_name + "_is_negative"] = (df[feature_name] < 0).astype(float)

        if self.drop_original_datetime:
            df = df.drop(columns=getattr(self, "datetime_columns_", []), errors="ignore")

        return df

    @staticmethod
    def _as_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    @staticmethod
    def _days_from_min(values):
        if values.notna().sum() == 0:
            return pd.Series(np.nan, index=values.index)
        return (values - values.min()).dt.total_seconds() / 86400.0

    @staticmethod
    def _discover_date_pairs(datetime_columns):
        pairs = []
        if len(datetime_columns) < 2:
            return pairs

        for first, second in combinations(datetime_columns[:5], 2):
            first_name = normalize_column_name(first)
            second_name = normalize_column_name(second)
            if _looks_like_start(first_name) and _looks_like_end(second_name):
                pairs.append((first, second))
            elif _looks_like_start(second_name) and _looks_like_end(first_name):
                pairs.append((second, first))

        if not pairs and 2 <= len(datetime_columns) <= 3:
            pairs.append((datetime_columns[0], datetime_columns[1]))

        return pairs[:4]


def _looks_like_start(name):
    return any(token in name for token in ("order", "ship", "start", "created", "pickup", "sent"))


def _looks_like_end(name):
    return any(token in name for token in ("expected", "actual", "deliver", "arrival", "due", "end"))


class LogisticsFeatureEngineer(BaseEstimator, TransformerMixin):
    """Generate logistics-specific features without assuming a fixed schema."""

    def __init__(self, max_numeric_indicators=12):
        self.max_numeric_indicators = max_numeric_indicators

    def fit(self, X, y=None):
        df = self._as_dataframe(X)
        self.input_columns_ = list(df.columns)
        self.numeric_indicator_columns_ = self._numeric_indicator_columns(df)[: self.max_numeric_indicators]
        self.numeric_quantiles_ = {}
        for column in self.numeric_indicator_columns_:
            values = pd.to_numeric(df[column], errors="coerce")
            self.numeric_quantiles_[column] = {
                "median": values.median(),
                "q75": values.quantile(0.75),
                "min": values.min(),
            }
        self.route_pairs_ = self._route_pairs(df)
        return self

    def transform(self, X):
        df = self._as_dataframe(X).copy()
        for column in getattr(self, "input_columns_", []):
            if column not in df.columns:
                df[column] = np.nan
        df = df[getattr(self, "input_columns_", list(df.columns))]

        for column in getattr(self, "numeric_indicator_columns_", []):
            values = pd.to_numeric(df[column], errors="coerce")
            quantiles = self.numeric_quantiles_.get(column, {})
            prefix = normalize_column_name(column)
            median = quantiles.get("median")
            q75 = quantiles.get("q75")
            min_value = quantiles.get("min")
            if min_value is not None and pd.notna(min_value) and min_value >= 0:
                df[prefix + "_log1p"] = np.log1p(values.clip(lower=0))
            if median is not None and pd.notna(median) and median != 0:
                df[prefix + "_relative_to_median"] = values / median
            if q75 is not None and pd.notna(q75):
                df[prefix + "_high_flag"] = (values > q75).astype(float)

        for origin_column, destination_column in getattr(self, "route_pairs_", []):
            route_name = "{}_to_{}_route".format(
                normalize_column_name(origin_column),
                normalize_column_name(destination_column),
            )
            df[route_name] = (
                df[origin_column].astype(str).fillna("unknown")
                + " -> "
                + df[destination_column].astype(str).fillna("unknown")
            )

        return df

    @staticmethod
    def _as_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    @staticmethod
    def _numeric_indicator_columns(df):
        keywords = ("cost", "price", "distance", "quantity", "weight", "volume", "amount", "fee", "duration")
        schema = detect_columns(df)
        columns = []
        for column in schema.numeric:
            normalized = normalize_column_name(column)
            if any(keyword in normalized for keyword in keywords):
                columns.append(column)
        return columns

    @staticmethod
    def _route_pairs(df):
        columns = list(df.columns)
        origins = [column for column in columns if "origin" in normalize_column_name(column)]
        destinations = [
            column
            for column in columns
            if any(token in normalize_column_name(column) for token in ("destination", "dest"))
        ]
        pairs = []
        for origin in origins:
            for destination in destinations:
                if origin != destination:
                    pairs.append((origin, destination))
        return pairs[:2]


def _season_from_month(month):
    if pd.isna(month):
        return np.nan
    month = int(month)
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3
