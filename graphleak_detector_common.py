from __future__ import annotations

import numpy as np
import pandas as pd


NODE_MAP = {
    "N2": "V_N2",
    "N3": "V_N3",
    "N4": "V_N4",
    "N5": "V_N5",
    "N6": "V_N6",
    "N7": "V_N7",
    "N8": "V_N8",
    "N9": "V_N9",
}


def load_experiment_dataset(path: str = "graphleak_volume_experiments.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["hour"] = ((df["time_s"] // 3600) % 24).astype(int)
    return df


def robust_group_params(normal_df: pd.DataFrame, cols: list[str], group_col: str = "hour") -> dict[str, dict[int, tuple[float, float]]]:
    params: dict[str, dict[int, tuple[float, float]]] = {}

    for col in cols:
        params[col] = {}

        for group_value, g in normal_df.groupby(group_col):
            x = g[col].values.astype(float)

            median = float(np.median(x))
            q1 = float(np.percentile(x, 25))
            q3 = float(np.percentile(x, 75))
            iqr = q3 - q1

            scale = iqr / 1.349
            if scale < 1e-6:
                scale = float(np.std(x))
            if scale < 1e-6:
                scale = 1.0

            params[col][int(group_value)] = (median, scale)

    return params


def meter_name_from_config(config: str) -> str:
    return (
        config.replace("ewma_cusum_", "")
        .replace("page_hinkley_", "")
        .replace("shewhart_", "")
        .replace("local_", "")
        .replace("_downstream_only", "")
    )


def evaluate_configuration(
    *,
    df: pd.DataFrame,
    config: str,
    cols: list[str],
    label_fn,
    predict_fn,
    metadata: dict[str, object],
) -> pd.DataFrame:
    sample_rows = []

    for scenario_id, g in df.groupby("scenario_id"):
        g = g.sort_values("time_s").copy()

        y_true = label_fn(g).astype(int).to_numpy()
        y_pred, scores = predict_fn(g, cols)

        y_pred = np.asarray(y_pred)
        scores = np.asarray(scores)

        for i, (_, row) in enumerate(g.iterrows()):
            sample_rows.append({
                "config": config,
                "cols": ",".join(cols),
                "meter": meter_name_from_config(config),
                "scenario_id": scenario_id,
                "source_file": row["source_file"],
                "scenario_type": row["scenario_type"],
                "time_s": int(row["time_s"]),
                "hour": int(row["hour"]),
                "last_label": int(g["label"].iloc[-1]),
                "y_true": int(y_true[i]),
                "y_pred": int(y_pred[i]),
                "score": float(scores[i]),
                **metadata,
            })

    pred = pd.DataFrame(sample_rows)

    if pred.empty:
        raise RuntimeError(
            f"No sample rows were generated for configuration {config}. "
            "Check whether df has scenario_id and valid samples."
        )

    tp = int(((pred["y_true"] == 1) & (pred["y_pred"] == 1)).sum())
    fp = int(((pred["y_true"] == 0) & (pred["y_pred"] == 1)).sum())
    fn = int(((pred["y_true"] == 1) & (pred["y_pred"] == 0)).sum())
    tn = int(((pred["y_true"] == 0) & (pred["y_pred"] == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    print("\n===", config, "===")
    print("Colunas usadas:", cols)
    print({"TP": tp, "FP": fp, "FN": fn, "TN": tn})
    print({"precision": precision, "recall": recall, "f1": f1})

    return pred
