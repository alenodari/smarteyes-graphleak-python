import pandas as pd
import numpy as np

df = pd.read_csv("graphleak_volume_experiments.csv")

# Hora do dia: 0–23
# O ponto 86400 volta para hour=0
df["hour"] = ((df["time_s"] // 3600) % 24).astype(int)

# Ajuste se seus nomes ainda forem V1...V8
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

def robust_group_params(normal_df, cols, group_col="hour"):
    params = {}

    for col in cols:
        params[col] = {}

        for group_value, g in normal_df.groupby(group_col):
            x = g[col].values.astype(float)

            median = np.median(x)
            q1 = np.percentile(x, 25)
            q3 = np.percentile(x, 75)
            iqr = q3 - q1

            scale = iqr / 1.349
            if scale < 1e-6:
                scale = np.std(x)
            if scale < 1e-6:
                scale = 1.0

            params[col][int(group_value)] = (median, scale)

    return params

def ewma_cusum_predict(g, cols, params, alpha=0.25, drift=0.75, threshold=50.0, group_col="hour"):
    ewma = {col: 0.0 for col in cols}
    cusum = {col: 0.0 for col in cols}

    preds = []
    scores = []

    for _, row in g.iterrows():
        max_score = 0.0
        group_value = int(row[group_col])

        for col in cols:
            median, scale = params[col][group_value]
            z = (float(row[col]) - median) / scale

            ewma[col] = alpha * z + (1.0 - alpha) * ewma[col]
            cusum[col] = max(0.0, cusum[col] + ewma[col] - drift)

            max_score = max(max_score, cusum[col])

        scores.append(max_score)
        preds.append(1 if max_score > threshold else 0)

    return np.array(preds), np.array(scores)

def evaluate_configuration(name, cols, label_fn, alpha=0.25, drift=0.75, threshold=50.0):
    normal_df = df[df["scenario_type"] == "normal"]
    params = robust_group_params(normal_df, cols, group_col="hour")

    rows = []
    sample_rows = []

    for scenario_id, g in df.groupby("scenario_id"):
        g = g.sort_values("time_s").copy()

        y_true = label_fn(g).astype(int).to_numpy()

        y_pred, scores = ewma_cusum_predict(
            g,
            cols,
            params,
            alpha=alpha,
            drift=drift,
            threshold=threshold,
            group_col="hour",
        )

        y_pred = np.asarray(y_pred)
        scores = np.asarray(scores)

        scenario_type = g["scenario_type"].iloc[0]

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        first_alert_idx = None
        first_alert_time_s = None

        alert_indices = np.where(y_pred == 1)[0]
        if len(alert_indices) > 0:
            first_alert_idx = int(alert_indices[0])
            first_alert_time_s = int(g["time_s"].iloc[first_alert_idx])

        delay_min = None
        valid_detection = 0
        first_valid_alert_idx = None
        first_valid_alert_time_s = None

        if y_true.any():
            onset_idx = int(np.argmax(y_true == 1))
            fp_before_onset = int(y_pred[:onset_idx].sum())

            candidates = np.where(
                (y_pred == 1) & (np.arange(len(y_pred)) >= onset_idx)
            )[0]

            if len(candidates) > 0:
                first_valid_alert_idx = int(candidates[0])
                first_valid_alert_time_s = int(g["time_s"].iloc[first_valid_alert_idx])
                delay_min = int((first_valid_alert_idx - onset_idx) * 10)
                valid_detection = 1
        else:
            onset_idx = None
            fp_before_onset = int(y_pred.sum())

        clean_valid_detection = int(valid_detection == 1 and fp_before_onset == 0)
        clean_delay_min = delay_min if clean_valid_detection else None

        rows.append({
            "config": name,
            "cols": ",".join(cols),
            "alpha": alpha,
            "drift": drift,
            "threshold": threshold,
            "group_col": "hour",

            "scenario_id": scenario_id,
            "source_file": g["source_file"].iloc[0],
            "scenario_type": scenario_type,
            "last_label": int(g["label"].iloc[-1]),

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
            "clean_valid_detection": clean_valid_detection,
            "clean_delay_min": clean_delay_min,

            "max_score": float(scores.max()),
            "mean_score": float(scores.mean()),
        })

        for i, (_, row) in enumerate(g.iterrows()):
            sample_rows.append({
                "config": name,
                "scenario_id": scenario_id,
                "source_file": row["source_file"],
                "scenario_type": row["scenario_type"],
                "time_s": int(row["time_s"]),
                "hour": int(row["hour"]),
                "last_label": int(g["label"].iloc[-1]),
                "y_true": int(y_true[i]),
                "y_pred": int(y_pred[i]),
                "score": float(scores[i]),
            })

    res = pd.DataFrame(rows)
    pred = pd.DataFrame(sample_rows)

    if res.empty:
        raise RuntimeError(
            f"No scenario rows were generated for configuration {name}. "
            "Check the indentation of rows.append() and whether df has scenario_id."
        )

    TP = res["tp"].sum()
    FP = res["fp"].sum()
    FN = res["fn"].sum()
    TN = res["tn"].sum()

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    print("\n===", name, "===")
    print("Colunas usadas:", cols)
    print({"TP": TP, "FP": FP, "FN": FN, "TN": TN})
    print({"precision": precision, "recall": recall, "f1": f1})

    print("Cenários com qualquer alerta por tipo:")
    print(res.groupby("scenario_type")["detected_any"].mean())

    print("Detecção válida por tipo:")
    print(res.groupby("scenario_type")["valid_detection"].mean())

    print("FP antes do onset por tipo:")
    print(res.groupby("scenario_type")["fp_before_onset"].mean())

    print("Detecção limpa por tipo:")
    print(res.groupby("scenario_type")["clean_valid_detection"].mean())

    print("Delay médio limpo em onset_leak:")
    print(res[res["scenario_type"] == "onset_leak"]["clean_delay_min"].dropna().mean())
    
    print("Delay médio em onset_leak:")
    print(res[res["scenario_type"] == "onset_leak"]["delay_min"].dropna().mean())

    return res, pred

# Configuração 1: volume multissensor
all_volume_cols = [NODE_MAP[n] for n in ["N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9"]]

res_all, pred_all = evaluate_configuration(
    name="volume_multisensor",
    cols=all_volume_cols,
    label_fn=lambda g: (g["label"] > 0),
)

res_n2, pred_n2 = evaluate_configuration(
    name="local_N2",
    cols=[NODE_MAP["N2"]],
    label_fn=lambda g: g["leak_downstream_N2"],
)

res_n8, pred_n8 = evaluate_configuration(
    name="local_N8_downstream_only",
    cols=[NODE_MAP["N8"]],
    label_fn=lambda g: g["leak_downstream_N8"],
)

res_n9, pred_n9 = evaluate_configuration(
    name="local_N9_downstream_only",
    cols=[NODE_MAP["N9"]],
    label_fn=lambda g: g["leak_downstream_N9"],
)

all_results = pd.concat(
    [res_all, res_n2, res_n8, res_n9],
    ignore_index=True
)

all_predictions = pd.concat(
    [pred_all, pred_n2, pred_n8, pred_n9],
    ignore_index=True
)

all_results.to_csv("baseline_ewma_cusum_results_by_scenario.csv", index=False)
all_predictions.to_csv("baseline_ewma_cusum_predictions_by_sample.csv", index=False)