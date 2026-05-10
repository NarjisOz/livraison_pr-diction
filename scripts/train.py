"""Train and save the best delivery delay model from the command line."""

import argparse

from delivery_delay.config import DEFAULT_MODEL_PATH
from delivery_delay.data_io import read_tabular_file
from delivery_delay.training import train_and_select_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train delivery-delay prediction models.")
    parser.add_argument("input", help="Path to CSV, Excel, or Parquet training data.")
    parser.add_argument("--target", default=None, help="Target column. Auto-detected when omitted.")
    parser.add_argument(
        "--problem-type",
        default="auto",
        choices=["auto", "classification", "regression"],
        help="Prediction task type.",
    )
    parser.add_argument("--output", default=str(DEFAULT_MODEL_PATH), help="Path for the saved joblib artifact.")
    parser.add_argument("--max-rows", type=int, default=100000, help="Maximum rows to use for training.")
    parser.add_argument(
        "--feature-percentile",
        type=int,
        default=80,
        help="Percentile of transformed features to keep after supervised selection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = read_tabular_file(args.input)
    result = train_and_select_model(
        df,
        target_column=args.target,
        problem_type=args.problem_type,
        output_path=args.output,
        max_rows=args.max_rows,
        feature_selection_percentile=args.feature_percentile,
    )
    print("Best model: {}".format(result.best_model_name))
    print("Selection score: {:.4f}".format(result.best_score))
    print("Artifact: {}".format(result.artifact_path))
    print(result.leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
