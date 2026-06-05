import numpy as np
import pandas as pd

from graphleak_detector_common import (
    NODE_MAP,
    evaluate_configuration,
    load_experiment_dataset,
    robust_group_params,
)


GROUP_COL = "hour"
THRESHOLD = 3.0


def shewhart_predict(
    g: pd.DataFrame,
    cols: list[str],
    params: dict[str, dict[int, tuple[float, float]]],
    threshold: float = THRESHOLD,
    group_col: str = GROUP_COL,
):
    preds = []
    scores = []

    for _, row in g.iterrows():
        max_score = 0.0
        group_value = int(row[group_col])

        for col in cols:
            median, scale = params[col][group_value]
            z = (float(row[col]) - median) / scale
            max_score = max(max_score, z)

        scores.append(max_score)
        preds.append(1 if max_score > threshold else 0)

    return np.array(preds), np.array(scores)


def evaluate_local_configuration(config: str, cols: list[str], label_fn) -> pd.DataFrame:
    df = load_experiment_dataset()
    normal_df = df[df["scenario_type"] == "normal"]
    params = robust_group_params(normal_df, cols, group_col=GROUP_COL)

    return evaluate_configuration(
        df=df,
        config=config,
        cols=cols,
        label_fn=label_fn,
        predict_fn=lambda g, detector_cols: shewhart_predict(
            g,
            detector_cols,
            params,
            threshold=THRESHOLD,
            group_col=GROUP_COL,
        ),
        metadata={
            "alpha": np.nan,
            "drift": np.nan,
            "delta": np.nan,
            "threshold": THRESHOLD,
            "group_col": GROUP_COL,
        },
    )


def main() -> None:
    all_volume_cols = [NODE_MAP[n] for n in ["N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9"]]

    pred_all = evaluate_local_configuration(
        config="volume_multisensor",
        cols=all_volume_cols,
        label_fn=lambda g: (g["label"] > 0),
    )

    pred_n2 = evaluate_local_configuration(
        config="local_N2",
        cols=[NODE_MAP["N2"]],
        label_fn=lambda g: g["leak_downstream_N2"],
    )

    pred_n8 = evaluate_local_configuration(
        config="local_N8_downstream_only",
        cols=[NODE_MAP["N8"]],
        label_fn=lambda g: g["leak_downstream_N8"],
    )

    pred_n9 = evaluate_local_configuration(
        config="local_N9_downstream_only",
        cols=[NODE_MAP["N9"]],
        label_fn=lambda g: g["leak_downstream_N9"],
    )

    all_predictions = pd.concat([pred_all, pred_n2, pred_n8, pred_n9], ignore_index=True)
    all_predictions.to_csv("baseline_shewhart_predictions_by_sample.csv", index=False)


if __name__ == "__main__":
    main()
