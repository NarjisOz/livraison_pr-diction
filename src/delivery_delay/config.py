"""Central project configuration."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
MODEL_PORTFOLIO_PATH = ARTIFACTS_DIR / "model_portfolio.joblib"
WEATHER_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nigeria_cities_weather_data.csv"

RANDOM_STATE = 42
N_JOBS = -1

SUPPORTED_UPLOAD_EXTENSIONS = (".csv", ".xlsx", ".xls", ".parquet")
