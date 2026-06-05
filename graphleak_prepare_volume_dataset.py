#!/usr/bin/env python3
"""
Prepare a volume-only GraphLeak dataset for local leak detection experiments.

This script converts the original GraphLeak CSV files into a reproducible,
semantically named, volume-only dataset focused on Smart-Eyes-like experiments.

Original GraphLeak CSV structure, according to the dataset documentation:

- Column 1: time indicator in seconds
- Columns 2 to 9: flow rate at nodes N2, N3, N4, N5, N6, N7, N8, N9
- Columns 10 to 17: pressure at nodes N2, N3, N4, N5, N6, N7, N8, N9
- Columns 18 to 25: volume at nodes N2, N3, N4, N5, N6, N7, N8, N9
- Columns 26 to 49: coordinates of measurement points
- Columns 50 to 54: leakage flags at nodes N10, N11, N12, N13, N14
- Columns 55 to 57: leakage point coordinates
- Column 58: weekday flag

Pandas uses zero-based indexing, so:
- time_s: column 0
- volume: columns 17..24
- leakage flags: columns 49..53
- leakage coordinates: columns 54..56
- weekday flag: column 57
"""

from pathlib import Path
import argparse
import re
import sys

import numpy as np
import pandas as pd


MEASUREMENT_NODES = ["N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9"]
LEAK_NODES = ["N10", "N11", "N12", "N13", "N14"]


def natural_key(path: Path):
    """
    Sort paths like data_1.csv, data_2.csv, ..., data_10.csv correctly.
    """
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))]


def extract_scenario_id(filename: str) -> int:
    """
    Extract scenario id from filenames like data_16.csv.
    """
    match = re.search(r"data_(\d+)\.csv", filename)
    if not match:
        raise ValueError(f"Could not extract scenario id from filename: {filename}")
    return int(match.group(1))


def classify_scenario_type(labels: pd.Series) -> str:
    """
    Classify the scenario based on its label sequence.
    """
    first_label = int(labels.iloc[0])
    last_label = int(labels.iloc[-1])

    if first_label == 0 and last_label == 0:
        return "normal"

    if first_label != 0 and last_label != 0:
        return "persistent_leak"

    if first_label == 0 and last_label != 0:
        return "onset_leak"

    return "other"


def build_volume_dataset(data_folder: Path) -> pd.DataFrame:
    """
    Read GraphLeak original data_*.csv files and build a semantically named
    volume-only dataset.
    """
    files = sorted(data_folder.glob("data_*.csv"), key=natural_key)

    if not files:
        raise FileNotFoundError(f"No data_*.csv files found in: {data_folder}")

    rows = []

    for file in files:
        df_raw = pd.read_csv(file, header=None)

        if df_raw.shape[1] != 58:
            raise ValueError(
                f"Unexpected number of columns in {file.name}: "
                f"expected 58, got {df_raw.shape[1]}"
            )

        if df_raw.shape[0] != 145:
            print(
                f"Warning: {file.name} has {df_raw.shape[0]} rows, expected 145.",
                file=sys.stderr,
            )

        scenario_id = extract_scenario_id(file.name)

        out = pd.DataFrame({
            "time_s": df_raw.iloc[:, 0].astype(int).to_numpy()
        })

        out.insert(0, "source_file", file.name)
        out.insert(0, "scenario_id", scenario_id)

        # Hour of day: 0..23. The last sample, 86400 s, maps back to hour 0.
        out["hour"] = ((out["time_s"] // 3600) % 24).astype(int)

        # Volume at nodes N2..N9.
        # Original columns 18..25 => pandas indices 17..24.
        for idx, node in enumerate(MEASUREMENT_NODES):
            out[f"V_{node}"] = df_raw.iloc[:, 17 + idx].astype(float)

        # Leak flags at N10..N14.
        # Original columns 50..54 => pandas indices 49..53.
        for idx, node in enumerate(LEAK_NODES):
            out[f"leak_{node}"] = df_raw.iloc[:, 49 + idx].astype(int)

        # Original leakage coordinates.
        out["leak_x"] = df_raw.iloc[:, 54].astype(float)
        out["leak_y"] = df_raw.iloc[:, 55].astype(float)
        out["leak_z"] = df_raw.iloc[:, 56].astype(float)

        # Weekday flag, currently for future use in GraphLeak.
        out["weekday_flag"] = df_raw.iloc[:, 57].astype(int)

        leak_cols = [f"leak_{node}" for node in LEAK_NODES]
        leak_matrix = out[leak_cols]

        # Multi-class label:
        # 0 = no leak
        # 1 = leak at N10
        # 2 = leak at N11
        # 3 = leak at N12
        # 4 = leak at N13
        # 5 = leak at N14
        label = leak_matrix.idxmax(axis=1).str.extract(r"N(\d+)").astype(int)[0]
        label = label.map({10: 1, 11: 2, 12: 3, 13: 4, 14: 5})
        label[leak_matrix.sum(axis=1) == 0] = 0

        out["label"] = label.astype(int)
        out["leak_binary"] = (out["label"] > 0).astype(int)

        # Local downstream labels for Smart-Eyes-like experiments.
        #
        # N2 is the incoming water meter: all leakage points are downstream.
        # N8 is treated as a proxy for the upper branch: N13 and N14.
        # N9 is treated as a proxy for the lower branch: N11 and N12.
        #
        # N10 is downstream of N2 but not assigned to N8/N9.
        out["leak_downstream_N2"] = out["label"].isin([1, 2, 3, 4, 5]).astype(int)
        out["leak_downstream_N8"] = out["label"].isin([4, 5]).astype(int)
        out["leak_downstream_N9"] = out["label"].isin([2, 3]).astype(int)

        scenario_type = classify_scenario_type(out["label"])
        out["scenario_type"] = scenario_type

        rows.append(out)

    result = pd.concat(rows, ignore_index=True)

    # Reorder columns for readability.
    volume_cols = [f"V_{node}" for node in MEASUREMENT_NODES]
    leak_flag_cols = [f"leak_{node}" for node in LEAK_NODES]

    ordered_cols = (
        [
            "scenario_id",
            "source_file",
            "scenario_type",
            "time_s",
            "hour",
        ]
        + volume_cols
        + leak_flag_cols
        + [
            "label",
            "leak_binary",
            "leak_downstream_N2",
            "leak_downstream_N8",
            "leak_downstream_N9",
            "leak_x",
            "leak_y",
            "leak_z",
            "weekday_flag",
        ]
    )

    return result[ordered_cols]


def build_scenario_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per scenario with onset information.
    """
    rows = []

    for scenario_id, g in df.groupby("scenario_id", sort=True):
        g = g.sort_values("time_s")

        leak_rows = g[g["leak_binary"] == 1]

        if len(leak_rows) > 0:
            onset_time_s = int(leak_rows["time_s"].iloc[0])
            onset_idx = int(g.index.get_loc(leak_rows.index[0]))
            onset_time_h = onset_time_s / 3600
        else:
            onset_time_s = np.nan
            onset_idx = np.nan
            onset_time_h = np.nan

        rows.append(
            {
                "scenario_id": int(scenario_id),
                "source_file": g["source_file"].iloc[0],
                "scenario_type": g["scenario_type"].iloc[0],
                "first_label": int(g["label"].iloc[0]),
                "last_label": int(g["label"].iloc[-1]),
                "dominant_label": int(g["label"].mode()[0]),
                "onset_idx": onset_idx,
                "onset_time_s": onset_time_s,
                "onset_time_h": onset_time_h,
                "n_samples": int(len(g)),
                "n_normal_samples": int((g["leak_binary"] == 0).sum()),
                "n_leak_samples": int((g["leak_binary"] == 1).sum()),
            }
        )

    return pd.DataFrame(rows)


def print_diagnostics(df: pd.DataFrame, summary: pd.DataFrame):
    """
    Print concise diagnostics for reproducibility.
    """
    print("\nDataset generated successfully.")
    print("Shape:", df.shape)

    print("\nRows per scenario:")
    print(df.groupby("source_file").size().value_counts().sort_index())

    print("\nScenario types:")
    print(summary["scenario_type"].value_counts().sort_index())

    print("\nFinal label distribution by scenario type:")
    print(summary.groupby(["scenario_type", "last_label"]).size())

    print("\nSample-level label distribution:")
    print(df["label"].value_counts().sort_index())

    print("\nSample-level binary leak distribution:")
    print(df["leak_binary"].value_counts().sort_index())

    print("\nFirst rows:")
    print(df.head())


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a volume-only GraphLeak dataset for Smart-Eyes-like experiments."
    )

    parser.add_argument(
        "--data_folder",
        required=True,
        help="Path to the original GraphLeak folder containing data_*.csv files.",
    )

    parser.add_argument(
        "--out",
        default="graphleak_volume_experiments.csv",
        help="Output CSV file for the prepared volume dataset.",
    )

    parser.add_argument(
        "--summary_out",
        default="graphleak_volume_summary.csv",
        help="Output CSV file for the scenario-level summary.",
    )

    args = parser.parse_args()

    data_folder = Path(args.data_folder)

    df = build_volume_dataset(data_folder)
    summary = build_scenario_summary(df)

    df.to_csv(args.out, index=False)
    summary.to_csv(args.summary_out, index=False)

    print(f"Volume dataset saved to: {args.out}")
    print(f"Scenario summary saved to: {args.summary_out}")

    print_diagnostics(df, summary)


if __name__ == "__main__":
    main()
