#!/usr/bin/env python3
"""
Build windowed train/validation/test CSVs from scenario-level GraphLeak splits.

The output remains in CSV format for auditability, but each row represents one
time window ready to be converted into tensors by a future training script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PREFIX = "graphleak_volume"
DEFAULT_OUTPUT_PREFIX = None
DEFAULT_WINDOW_SIZE = 24
DEFAULT_STRIDE = 1
DEFAULT_FEATURE_COLS = ["V_N2"]
DEFAULT_LABEL_COL = "leak_binary"
NODE_OUTPUT_DIRS = {
    "V_N2": "mlp/n2",
    "V_N8": "mlp/n8",
    "V_N9": "mlp/n9",
}

REQUIRED_METADATA_COLUMNS = [
    "scenario_id",
    "source_file",
    "scenario_type",
    "time_s",
]


def validate_input(df: pd.DataFrame, feature_cols: list[str], label_col: str) -> None:
    required = set(REQUIRED_METADATA_COLUMNS + feature_cols + [label_col])
    missing = required - set(df.columns)
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise ValueError(f"Input CSV is missing required columns: {missing_csv}")


def build_window_rows(
    df: pd.DataFrame,
    split_name: str,
    feature_cols: list[str],
    label_col: str,
    window_size: int,
    stride: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for scenario_id, group in df.groupby("scenario_id", sort=True, dropna=False):
        g = group.sort_values("time_s").reset_index(drop=True)
        num_rows = len(g)

        if num_rows < window_size:
            continue

        for start_idx in range(0, num_rows - window_size + 1, stride):
            end_idx = start_idx + window_size - 1
            window = g.iloc[start_idx : start_idx + window_size]

            row: dict[str, object] = {
                "split": split_name,
                "scenario_id": int(scenario_id),
                "source_file": g["source_file"].iloc[0],
                "scenario_type": g["scenario_type"].iloc[0],
                "window_id": f"{int(scenario_id)}_{start_idx}_{end_idx}",
                "window_size": window_size,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time_s": int(window["time_s"].iloc[0]),
                "end_time_s": int(window["time_s"].iloc[-1]),
                "label_col": label_col,
                "y_true": int(window[label_col].iloc[-1]),
            }

            for feature_col in feature_cols:
                for offset, value in enumerate(window[feature_col].tolist()):
                    lag = window_size - 1 - offset
                    row[f"{feature_col}_t-{lag}"] = value

            rows.append(row)

    return rows


def build_windowed_split(
    input_path: Path,
    split_name: str,
    feature_cols: list[str],
    label_col: str,
    window_size: int,
    stride: int,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    validate_input(df, feature_cols, label_col)
    rows = build_window_rows(df, split_name, feature_cols, label_col, window_size, stride)
    return pd.DataFrame(rows)


def output_stem(
    base_prefix: str,
    feature_cols: list[str],
    label_col: str,
    window_size: int,
    stride: int,
) -> str:
    feature_tag = "-".join(feature_cols)
    return (
        f"{base_prefix}_w{window_size}_s{stride}"
        f"_{feature_tag}_label-{label_col}"
    )


def default_output_prefix(feature_cols: list[str]) -> str:
    if len(feature_cols) == 1 and feature_cols[0] in NODE_OUTPUT_DIRS:
        return f"{NODE_OUTPUT_DIRS[feature_cols[0]]}/graphleak_volume_windows"
    return "mlp/graphleak_volume_windows"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build windowed train/validation/test CSVs from GraphLeak splits."
    )
    parser.add_argument(
        "--input-prefix",
        default=DEFAULT_INPUT_PREFIX,
        help="Common prefix of the split CSVs, e.g. graphleak_volume.",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Common prefix for the output windowed CSVs. Defaults to mlp/<node>/graphleak_volume_windows.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Number of time steps per window.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Window stride in samples.",
    )
    parser.add_argument(
        "--feature-cols",
        nargs="+",
        default=DEFAULT_FEATURE_COLS,
        help="Feature columns copied into each flattened window.",
    )
    parser.add_argument(
        "--label-col",
        default=DEFAULT_LABEL_COL,
        help="Label column used to define y_true from the last sample in the window.",
    )
    args = parser.parse_args()

    if args.window_size <= 0:
        raise ValueError("window-size must be positive.")
    if args.stride <= 0:
        raise ValueError("stride must be positive.")

    output_prefix = args.output_prefix or default_output_prefix(args.feature_cols)
    output_base = output_stem(
        output_prefix,
        args.feature_cols,
        args.label_col,
        args.window_size,
        args.stride,
    )

    split_paths = {
        "train": Path(f"{args.input_prefix}_train.csv"),
        "val": Path(f"{args.input_prefix}_val.csv"),
        "test": Path(f"{args.input_prefix}_test.csv"),
    }

    for split_name, split_path in split_paths.items():
        split_df = build_windowed_split(
            split_path,
            split_name,
            args.feature_cols,
            args.label_col,
            args.window_size,
            args.stride,
        )
        output_path = Path(f"{output_base}_{split_name}.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path} ({len(split_df)} windows)")


if __name__ == "__main__":
    main()
