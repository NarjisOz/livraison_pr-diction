"""Generate predictions with a saved delivery-delay model artifact."""

import argparse
from pathlib import Path

from delivery_delay.config import DEFAULT_MODEL_PATH
from delivery_delay.data_io import read_tabular_file
from delivery_delay.prediction import load_model_artifact, predict_dataframe


def parse_args():
    parser = argparse.ArgumentParser(description="Predict delivery delays for new data.")
    parser.add_argument("input", help="Path to CSV, Excel, or Parquet prediction data.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Saved joblib artifact path.")
    parser.add_argument("--output", default="data/predictions.csv", help="CSV output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    artifact = load_model_artifact(args.model)
    df = read_tabular_file(args.input)
    predictions = predict_dataframe(artifact, df)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print("Predictions saved to {}".format(output))


if __name__ == "__main__":
    main()
