#!/usr/bin/env python3
"""
Build shared train/validation/test windowed CSVs for a single multi-node model.

Each window is derived from one local meter signal and carries:
- the flattened local time series
- a one-hot node identifier
- the local downstream target for that node
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PREFIX = "graphleak_volume"
DEFAULT_OUTPUT_PREFIX = "mlp/shared/graphleak_volume_shared_windows"
DEFAULT_WINDOW_SIZE = 24
DEFAULT_STRIDE = 1

NODE_SPECS = [
    ("N2", "V_N2", "leak_downstream_N2"),
    ("N8", "V_N8", "leak_downstream_N8"),
    ("N9", "V_N9", "leak_downstream_N9"),
]

REQUIRED_BASE_COLUMNS = [
    "scenario_id",
    "source_file",
    "scenario_type",
    "time_s",
]


def validate_input(df: pd.DataFrame) -> None:
    required = set(REQUIRED_BASE_COLUMNS)
    for node_id, signal_col, label_col in NODE_SPECS:
        required.add(signal_col)
        required.add(label_col)

    missing = required - set(df.columns)
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise ValueError(f"Input CSV is missing required columns: {missing_csv}")


def build_rows_for_node(
    group: pd.DataFrame,
    split_name: str,
    node_id: str,
    signal_col: str,
    label_col: str,
    window_size: int,
    stride: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    num_rows = len(group)

    if num_rows < window_size:
        return rows

    for start_idx in range(0, num_rows - window_size + 1, stride):
        end_idx = start_idx + window_size - 1
        window = group.iloc[start_idx : start_idx + window_size]

        row: dict[str, object] = {
            "split": split_name,
            "scenario_id": int(group["scenario_id"].iloc[0]),
            "source_file": group["source_file"].iloc[0],
            "scenario_type": group["scenario_type"].iloc[0],
            "node_id": node_id,
            "signal_col": signal_col,
            "label_col": label_col,
            "window_id": f"{int(group['scenario_id'].iloc[0])}_{node_id}_{start_idx}_{end_idx}",
            "window_size": window_size,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_time_s": int(window["time_s"].iloc[0]),
            "end_time_s": int(window["time_s"].iloc[-1]),
            "y_true": int(window[label_col].iloc[-1]),
            "node_is_N2": int(node_id == "N2"),
            "node_is_N8": int(node_id == "N8"),
            "node_is_N9": int(node_id == "N9"),
        }

        for offset, value in enumerate(window[signal_col].tolist()):
            lag = window_size - 1 - offset
            row[f"local_t-{lag}"] = value

        rows.append(row)

    return rows


def build_shared_split(input_path: Path, split_name: str, window_size: int, stride: int) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    validate_input(df)

    rows: list[dict[str, object]] = []
    for _, group in df.groupby("scenario_id", sort=True, dropna=False):
        group = group.sort_values("time_s").reset_index(drop=True)
        for node_id, signal_col, label_col in NODE_SPECS:
            rows.extend(
                build_rows_for_node(
                    group,
                    split_name,
                    node_id,
                    signal_col,
                    label_col,
                    window_size,
                    stride,
                )
            )

    return pd.DataFrame(rows)


def output_stem(base_prefix: str, window_size: int, stride: int) -> str:
    return f"{base_prefix}_w{window_size}_s{stride}_nodes-N2-N8-N9_local-downstream"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build shared node-aware GraphLeak train/val/test window CSVs."
    )
    parser.add_argument(
        "--input-prefix",
        default=DEFAULT_INPUT_PREFIX,
        help="Common prefix of the split CSVs, e.g. graphleak_volume.",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Common prefix for the output windowed CSVs.",
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
    args = parser.parse_args()

    if args.window_size <= 0:
        raise ValueError("window-size must be positive.")
    if args.stride <= 0:
        raise ValueError("stride must be positive.")

    output_base = output_stem(args.output_prefix, args.window_size, args.stride)
    split_paths = {
        "train": Path(f"{args.input_prefix}_train.csv"),
        "val": Path(f"{args.input_prefix}_val.csv"),
        "test": Path(f"{args.input_prefix}_test.csv"),
    }

    for split_name, split_path in split_paths.items():
        split_df = build_shared_split(split_path, split_name, args.window_size, args.stride)
        output_path = Path(f"{output_base}_{split_name}.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path} ({len(split_df)} windows)")


if __name__ == "__main__":
    main()
