#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from graphleak_detector_common import NODE_MAP, load_experiment_dataset, robust_group_params
from page_hinkley import page_hinkley_predict
from results_summarize_by_config import build_summary
from results_build_by_scenario import build_results


GROUP_COL = "hour"
DEFAULT_DELTAS = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5]
DEFAULT_THRESHOLDS = [5, 10, 20, 30, 40, 50, 60, 80, 100]


LOCAL_CONFIGS = [
    ("local_N2", [NODE_MAP["N2"]], lambda g: g["leak_downstream_N2"], "N2"),
    ("local_N8_downstream_only", [NODE_MAP["N8"]], lambda g: g["leak_downstream_N8"], "N8"),
    ("local_N9_downstream_only", [NODE_MAP["N9"]], lambda g: g["leak_downstream_N9"], "N9"),
]


def parse_csv_floats(raw: str | None, default: list[float]) -> list[float]:
    if raw is None or raw.strip() == "":
        return default
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def generate_predictions_for_params(
    df: pd.DataFrame,
    delta: float,
    threshold: float,
) -> pd.DataFrame:
    normal_df = df[df["scenario_type"] == "normal"]
    sample_rows: list[dict[str, object]] = []

    for config_name, cols, label_fn, meter in LOCAL_CONFIGS:
        params = robust_group_params(normal_df, cols, group_col=GROUP_COL)

        for scenario_id, g in df.groupby("scenario_id"):
            g = g.sort_values("time_s").copy()
            y_true = label_fn(g).astype(int).to_numpy()
            y_pred, scores = page_hinkley_predict(
                g,
                cols,
                params,
                delta=delta,
                threshold=threshold,
                group_col=GROUP_COL,
            )

            for i, (_, row) in enumerate(g.iterrows()):
                sample_rows.append(
                    {
                        "config": config_name,
                        "cols": ",".join(cols),
                        "alpha": np.nan,
                        "drift": np.nan,
                        "delta": delta,
                        "threshold": threshold,
                        "group_col": GROUP_COL,
                        "meter": meter,
                        "scenario_id": int(scenario_id),
                        "source_file": row["source_file"],
                        "scenario_type": row["scenario_type"],
                        "time_s": int(row["time_s"]),
                        "hour": int(row["hour"]),
                        "last_label": int(g["label"].iloc[-1]),
                        "y_true": int(y_true[i]),
                        "y_pred": int(y_pred[i]),
                        "score": float(scores[i]),
                    }
                )

    return pd.DataFrame(sample_rows)


def aggregate_global_score(summary: pd.DataFrame) -> dict[str, float]:
    local = summary[summary["scope"] == "local"].copy()
    return {
        "mean_f1_local": float(local["f1"].mean()),
        "mean_recall_local": float(local["recall"].mean()),
        "mean_precision_local": float(local["precision"].mean()),
        "mean_normal_false_alert_rate_local": float(local["normal_false_alert_rate"].mean()),
        "mean_onset_valid_detection_rate_local": float(local["onset_valid_detection_rate"].mean()),
        "mean_onset_clean_valid_detection_rate_local": float(
            local["onset_clean_valid_detection_rate"].mean()
        ),
        "mean_persistent_detection_rate_local": float(local["persistent_detection_rate"].mean()),
        "mean_persistent_valid_detection_rate_local": float(
            local["persistent_valid_detection_rate"].mean()
        ),
        "mean_delay_min_onset_local": float(local["mean_delay_min_onset"].mean()),
        "mean_delay_min_persistent_local": float(local["mean_delay_min_persistent"].mean()),
    }


def rank_rows(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["rank_score"] = (
        ranked["mean_f1_local"] * 4.0
        + ranked["mean_onset_clean_valid_detection_rate_local"] * 2.0
        + ranked["mean_persistent_valid_detection_rate_local"] * 2.0
        - ranked["mean_normal_false_alert_rate_local"] * 2.0
        - ranked["mean_delay_min_onset_local"] / 1000.0
    )
    return ranked.sort_values(
        [
            "rank_score",
            "mean_f1_local",
            "mean_onset_clean_valid_detection_rate_local",
            "mean_persistent_valid_detection_rate_local",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search global Page-Hinkley parameters on local GraphLeak detectors."
    )
    parser.add_argument(
        "--input",
        default="graphleak_volume_experiments.csv",
        help="Path to experiment CSV.",
    )
    parser.add_argument(
        "--deltas",
        default=None,
        help="Comma-separated delta values. Default: built-in calibration grid.",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated threshold values. Default: built-in calibration grid.",
    )
    parser.add_argument(
        "--output",
        default="page_hinkley_global_calibration.csv",
        help="Output CSV with one row per delta-threshold pair.",
    )
    args = parser.parse_args()

    deltas = parse_csv_floats(args.deltas, DEFAULT_DELTAS)
    thresholds = parse_csv_floats(args.thresholds, DEFAULT_THRESHOLDS)
    df = load_experiment_dataset(args.input)
    output = Path(args.output)

    rows: list[dict[str, object]] = []
    total = len(deltas) * len(thresholds)
    done = 0

    for delta in deltas:
        for threshold in thresholds:
            done += 1
            print(f"[{done}/{total}] Evaluating delta={delta} threshold={threshold}", flush=True)
            predictions = generate_predictions_for_params(df, delta=delta, threshold=threshold)
            results = build_results(predictions)
            summary = build_summary(results)
            aggregated = aggregate_global_score(summary)

            row = {
                "delta": delta,
                "threshold": threshold,
                **aggregated,
            }

            for _, metric_row in summary.iterrows():
                config = str(metric_row["config"])
                safe_config = config.replace("local_", "").replace("_downstream_only", "")
                row[f"f1_{safe_config}"] = float(metric_row["f1"])
                row[f"precision_{safe_config}"] = float(metric_row["precision"])
                row[f"recall_{safe_config}"] = float(metric_row["recall"])
                row[f"normal_false_alert_rate_{safe_config}"] = float(
                    metric_row["normal_false_alert_rate"]
                )
                row[f"onset_valid_detection_rate_{safe_config}"] = float(
                    metric_row["onset_valid_detection_rate"]
                )
                row[f"onset_clean_valid_detection_rate_{safe_config}"] = float(
                    metric_row["onset_clean_valid_detection_rate"]
                )
                row[f"persistent_valid_detection_rate_{safe_config}"] = float(
                    metric_row["persistent_valid_detection_rate"]
                )
                row[f"mean_delay_min_onset_{safe_config}"] = float(
                    metric_row["mean_delay_min_onset"]
                )

            rows.append(row)
            pd.DataFrame(rows).to_csv(output, index=False)

    ranked = rank_rows(pd.DataFrame(rows))
    ranked.to_csv(output, index=False)

    print(f"Calibration results saved to: {output}")
    print("Top 10 configurations:")
    print(
        ranked[
            [
                "delta",
                "threshold",
                "rank_score",
                "mean_f1_local",
                "mean_onset_clean_valid_detection_rate_local",
                "mean_persistent_valid_detection_rate_local",
                "mean_normal_false_alert_rate_local",
                "mean_delay_min_onset_local",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
