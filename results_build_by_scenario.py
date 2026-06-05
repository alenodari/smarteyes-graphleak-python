#!/usr/bin/env python3
"""
Build scenario-level detector results from predictions_by_sample CSVs.

Detectors should emit only the granular sample-level file. This script then
derives one row per scenario with comparable metrics across algorithms.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "config",
    "scenario_id",
    "source_file",
    "scenario_type",
    "time_s",
    "hour",
    "y_true",
    "y_pred",
    "score",
}

CORE_GROUP_COLUMNS = ["config", "scenario_id", "source_file", "scenario_type"]
BASE_SAMPLE_COLUMNS = set(CORE_GROUP_COLUMNS + ["time_s", "hour", "y_true", "y_pred", "score"])


def validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise ValueError(f"Input CSV is missing required columns: {missing_csv}")


def grouping_columns(df: pd.DataFrame) -> list[str]:
    return CORE_GROUP_COLUMNS


def extract_constant_metadata(group: pd.DataFrame) -> dict[str, object]:
    metadata: dict[str, object] = {}
    extra_cols = [col for col in group.columns if col not in BASE_SAMPLE_COLUMNS]

    for col in extra_cols:
        non_null_values = group[col].dropna().unique()
        if len(non_null_values) == 1:
            metadata[col] = non_null_values[0]
    return metadata


def scenario_row(group: pd.DataFrame) -> dict[str, object]:
    g = group.sort_values("time_s").reset_index(drop=True)

    y_true = g["y_true"].astype(int).to_numpy()
    y_pred = g["y_pred"].astype(int).to_numpy()
    scores = g["score"].astype(float).to_numpy()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    alert_indices = np.where(y_pred == 1)[0]
    first_alert_idx = int(alert_indices[0]) if len(alert_indices) > 0 else None
    first_alert_time_s = int(g["time_s"].iloc[first_alert_idx]) if first_alert_idx is not None else None

    delay_min = None
    valid_detection = 0
    first_valid_alert_idx = None
    first_valid_alert_time_s = None

    if y_true.any():
        onset_idx = int(np.argmax(y_true == 1))
        fp_before_onset = int(y_pred[:onset_idx].sum())
        candidates = np.where((y_pred == 1) & (np.arange(len(y_pred)) >= onset_idx))[0]

        if len(candidates) > 0:
            first_valid_alert_idx = int(candidates[0])
            first_valid_alert_time_s = int(g["time_s"].iloc[first_valid_alert_idx])
            delay_min = int((first_valid_alert_idx - onset_idx) * 10)
            valid_detection = 1
    else:
        fp_before_onset = int(y_pred.sum())

    row = {
        "config": g["config"].iloc[0],
        "scenario_id": int(g["scenario_id"].iloc[0]),
        "source_file": g["source_file"].iloc[0],
        "scenario_type": g["scenario_type"].iloc[0],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "detected_any": int(y_pred.any()),
        "valid_detection": valid_detection,
        "first_alert_idx": first_alert_idx,
        "first_alert_time_s": first_alert_time_s,
        "first_valid_alert_idx": first_valid_alert_idx,
        "first_valid_alert_time_s": first_valid_alert_time_s,
        "delay_min": delay_min,
        "fp_before_onset": fp_before_onset,
        "clean_valid_detection": int(valid_detection == 1 and fp_before_onset == 0),
        "clean_delay_min": delay_min if valid_detection == 1 and fp_before_onset == 0 else None,
        "max_score": float(scores.max()),
        "mean_score": float(scores.mean()),
    }
    row.update(extract_constant_metadata(g))
    return row


def build_results(df: pd.DataFrame) -> pd.DataFrame:
    validate_input(df)
    group_cols = grouping_columns(df)

    rows = []
    for _, group in df.groupby(group_cols, sort=True, dropna=False):
        rows.append(scenario_row(group.copy()))

    results = pd.DataFrame(rows)

    preferred_order = [
        "config",
        "cols",
        "alpha",
        "drift",
        "delta",
        "threshold",
        "group_col",
        "meter",
        "scenario_id",
        "source_file",
        "scenario_type",
        "last_label",
        "tp",
        "fp",
        "fn",
        "tn",
        "detected_any",
        "valid_detection",
        "first_alert_idx",
        "first_alert_time_s",
        "first_valid_alert_idx",
        "first_valid_alert_time_s",
        "delay_min",
        "fp_before_onset",
        "clean_valid_detection",
        "clean_delay_min",
        "max_score",
        "mean_score",
    ]
    ordered_cols = [col for col in preferred_order if col in results.columns]
    remaining_cols = [col for col in results.columns if col not in ordered_cols]
    return results[ordered_cols + remaining_cols]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build results_by_scenario CSV from predictions_by_sample CSV."
    )
    parser.add_argument(
        "--input",
        default="baseline_ewma_cusum_predictions_by_sample.csv",
        help="Path to predictions_by_sample CSV.",
    )
    parser.add_argument(
        "--output",
        default="baseline_ewma_cusum_results_by_scenario.csv",
        help="Path to results_by_scenario CSV.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    results = build_results(df)

    output_path = Path(args.output)
    results.to_csv(output_path, index=False)

    print(f"Scenario results saved to: {output_path}")
    print(results.head().to_string(index=False))


if __name__ == "__main__":
    main()
