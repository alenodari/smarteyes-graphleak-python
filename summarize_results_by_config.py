#!/usr/bin/env python3
"""
Build a compact configuration-level summary from detector results_by_scenario CSVs.

This keeps the experiment contract simple:
- detectors emit per-scenario results
- this script derives comparable summary tables for papers and ranking
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "config",
    "scenario_type",
    "tp",
    "fp",
    "fn",
    "tn",
    "detected_any",
    "valid_detection",
    "delay_min",
    "fp_before_onset",
    "clean_valid_detection",
    "max_score",
    "mean_score",
}


def infer_meter(config: str) -> str:
    if config == "volume_multisensor":
        return "N2..N9"
    if "N2" in config:
        return "N2"
    if "N8" in config:
        return "N8"
    if "N9" in config:
        return "N9"
    return "unknown"


def infer_scope(config: str) -> str:
    return "multisensor" if "multisensor" in config else "local"


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def scenario_fraction(df: pd.DataFrame, scenario_type: str, column: str) -> float:
    subset = df[df["scenario_type"] == scenario_type]
    if subset.empty:
        return np.nan
    return float(subset[column].mean())


def scenario_delay_stat(
    df: pd.DataFrame,
    scenario_type: str,
    delay_column: str,
    reducer: str,
) -> float:
    subset = df[df["scenario_type"] == scenario_type]
    values = subset[delay_column].dropna()
    if values.empty:
        return np.nan
    if reducer == "mean":
        return float(values.mean())
    if reducer == "median":
        return float(values.median())
    raise ValueError(f"Unsupported reducer: {reducer}")


def summarize_config(group: pd.DataFrame) -> dict[str, object]:
    config = str(group["config"].iloc[0])
    tp_total = int(group["tp"].sum())
    fp_total = int(group["fp"].sum())
    fn_total = int(group["fn"].sum())
    tn_total = int(group["tn"].sum())

    onset = group[group["scenario_type"] == "onset_leak"]
    persistent = group[group["scenario_type"] == "persistent_leak"]

    return {
        "config": config,
        "meter": infer_meter(config),
        "scope": infer_scope(config),
        "n_scenarios": int(len(group)),
        "n_normal": int((group["scenario_type"] == "normal").sum()),
        "n_onset": int((group["scenario_type"] == "onset_leak").sum()),
        "n_persistent": int((group["scenario_type"] == "persistent_leak").sum()),
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total,
        "tn_total": tn_total,
        "precision": safe_ratio(tp_total, tp_total + fp_total),
        "recall": safe_ratio(tp_total, tp_total + fn_total),
        "f1": safe_ratio(2 * tp_total, 2 * tp_total + fp_total + fn_total),
        "normal_false_alert_rate": scenario_fraction(group, "normal", "detected_any"),
        "onset_valid_detection_rate": scenario_fraction(group, "onset_leak", "valid_detection"),
        "onset_clean_valid_detection_rate": scenario_fraction(
            group, "onset_leak", "clean_valid_detection"
        ),
        "persistent_detection_rate": scenario_fraction(group, "persistent_leak", "detected_any"),
        "persistent_valid_detection_rate": scenario_fraction(
            group, "persistent_leak", "valid_detection"
        ),
        "mean_delay_min_onset": scenario_delay_stat(group, "onset_leak", "delay_min", "mean"),
        "median_delay_min_onset": scenario_delay_stat(
            group, "onset_leak", "delay_min", "median"
        ),
        "mean_delay_min_persistent": scenario_delay_stat(
            group, "persistent_leak", "delay_min", "mean"
        ),
        "median_delay_min_persistent": scenario_delay_stat(
            group, "persistent_leak", "delay_min", "median"
        ),
        "mean_fp_before_onset_onset": float(onset["fp_before_onset"].mean()) if not onset.empty else np.nan,
        "mean_fp_before_onset_persistent": (
            float(persistent["fp_before_onset"].mean()) if not persistent.empty else np.nan
        ),
        "max_score_mean": float(group["max_score"].mean()),
        "mean_score_mean": float(group["mean_score"].mean()),
    }


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise ValueError(f"Input CSV is missing required columns: {missing_csv}")

    rows = []
    for _, group in df.groupby("config", sort=True):
        rows.append(summarize_config(group.copy()))

    summary = pd.DataFrame(rows)
    ordered_cols = [
        "config",
        "meter",
        "scope",
        "n_scenarios",
        "n_normal",
        "n_onset",
        "n_persistent",
        "tp_total",
        "fp_total",
        "fn_total",
        "tn_total",
        "precision",
        "recall",
        "f1",
        "normal_false_alert_rate",
        "onset_valid_detection_rate",
        "onset_clean_valid_detection_rate",
        "persistent_detection_rate",
        "persistent_valid_detection_rate",
        "mean_delay_min_onset",
        "median_delay_min_onset",
        "mean_delay_min_persistent",
        "median_delay_min_persistent",
        "mean_fp_before_onset_onset",
        "mean_fp_before_onset_persistent",
        "max_score_mean",
        "mean_score_mean",
    ]
    return summary[ordered_cols].sort_values(["scope", "config"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize detector results_by_scenario CSV into one row per config."
    )
    parser.add_argument(
        "--input",
        default="baseline_ewma_cusum_results_by_scenario.csv",
        help="Path to results_by_scenario CSV.",
    )
    parser.add_argument(
        "--output",
        default="baseline_ewma_cusum_summary_by_config.csv",
        help="Path to summary_by_config CSV.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    summary = build_summary(df)

    output_path = Path(args.output)
    summary.to_csv(output_path, index=False)

    print(f"Summary saved to: {output_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
