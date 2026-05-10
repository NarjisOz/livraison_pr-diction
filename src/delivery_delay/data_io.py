"""Input/output helpers for tabular datasets and model artifacts."""

from pathlib import Path

import joblib
import pandas as pd

from .exceptions import ArtifactError, DataLoadingError


def _resolve_name(file_or_path, filename=None):
    if filename:
        return filename
    if hasattr(file_or_path, "name"):
        return file_or_path.name
    return str(file_or_path)


def read_tabular_file(file_or_path, filename=None, max_rows=None):
    """Read CSV, Excel, or Parquet data into a DataFrame.

    The function accepts both filesystem paths and Streamlit UploadedFile
    objects. It keeps parsing rules conservative so malformed uploads fail
    with actionable errors instead of obscure pandas tracebacks.
    """

    name = _resolve_name(file_or_path, filename)
    suffix = Path(name).suffix.lower()

    try:
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)

        if suffix == ".csv":
            return _validate_loaded_dataframe(_read_csv(file_or_path, max_rows=max_rows), name)
        if suffix in (".xlsx", ".xls"):
            return _validate_loaded_dataframe(pd.read_excel(file_or_path, nrows=max_rows), name)
        if suffix == ".parquet":
            df = pd.read_parquet(file_or_path)
            return _validate_loaded_dataframe(df.head(max_rows) if max_rows else df, name)
    except Exception as exc:
        if isinstance(exc, DataLoadingError):
            raise
        raise DataLoadingError("Unable to load dataset '{}': {}".format(name, exc)) from exc

    raise DataLoadingError("Unsupported file type '{}'. Please upload CSV, Excel, or Parquet.".format(suffix))


def _read_csv(file_or_path, max_rows=None):
    """Read CSV with a conservative fallback for common non-UTF encodings."""

    try:
        return pd.read_csv(file_or_path, nrows=max_rows)
    except UnicodeDecodeError:
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
        return pd.read_csv(file_or_path, nrows=max_rows, encoding="latin1")


def _validate_loaded_dataframe(df, name):
    """Fail early with user-actionable messages for invalid tabular uploads."""

    if not isinstance(df, pd.DataFrame):
        raise DataLoadingError("The uploaded file '{}' did not produce a table.".format(name))
    if df.shape[1] == 0:
        raise DataLoadingError("The uploaded file '{}' has no columns.".format(name))
    if df.shape[0] == 0:
        raise DataLoadingError("The uploaded file '{}' has no rows.".format(name))

    unnamed_columns = [column for column in df.columns if str(column).lower().startswith("unnamed")]
    if len(unnamed_columns) == df.shape[1]:
        raise DataLoadingError(
            "The uploaded file '{}' appears to contain only unnamed columns. Check the header row.".format(
                name
            )
        )
    return df


def save_joblib(obj, path):
    """Persist an object with joblib and create the parent directory."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_joblib(path):
    """Load a joblib artifact with a clear project-level error."""

    path = Path(path)
    if not path.exists():
        raise ArtifactError("Artifact not found: {}".format(path))
    try:
        return joblib.load(path)
    except Exception as exc:
        raise ArtifactError("Unable to load artifact '{}': {}".format(path, exc)) from exc
