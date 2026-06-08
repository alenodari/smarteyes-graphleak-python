#!/usr/bin/env python3
"""
Train, validate, and test a shared node-aware MLP baseline on GraphLeak windows.

This variant trains a single model over N2, N8, and N9 windows using:
- flattened local signal windows
- one-hot node identity features
- local downstream labels per node
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_INPUT_PREFIX = "mlp/shared/graphleak_volume_shared_windows_w24_s1_nodes-N2-N8-N9_local-downstream"
DEFAULT_OUTPUT_PREFIX = "mlp/shared/graphleak_volume_shared_mlp_baseline"
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_HIDDEN_DIMS = [64, 32]
DEFAULT_DROPOUT = 0.10
DEFAULT_PATIENCE = 8
DEFAULT_SEED = 42

METADATA_COLUMNS = {
    "split",
    "scenario_id",
    "source_file",
    "scenario_type",
    "node_id",
    "signal_col",
    "label_col",
    "window_id",
    "window_size",
    "start_idx",
    "end_idx",
    "start_time_s",
    "end_time_s",
    "y_true",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a shared node-aware MLP baseline on GraphLeak windowed CSVs."
    )
    parser.add_argument("--input-prefix", default=DEFAULT_INPUT_PREFIX)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=DEFAULT_HIDDEN_DIMS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if col not in METADATA_COLUMNS]
    if not cols:
        raise ValueError("No feature columns found in input CSV.")
    return cols


def read_split(path: Path, expected_features: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    if "y_true" not in df.columns:
        raise ValueError(f"Input CSV is missing y_true: {path}")

    cols = feature_columns(df)
    if expected_features is not None and cols != expected_features:
        raise ValueError(
            f"Feature columns mismatch in {path}. Expected {expected_features}, got {cols}."
        )
    return df, cols


def standardize_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = train_df[feature_cols].to_numpy(dtype=np.float32)
    val_x = val_df[feature_cols].to_numpy(dtype=np.float32)
    test_x = test_df[feature_cols].to_numpy(dtype=np.float32)

    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0.0] = 1.0

    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std
    return train_x, val_x, test_x, mean, std


class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            batch_size = x_batch.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
    return total_loss / total_count


def predict_scores(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = make_loader(x, np.zeros(len(x), dtype=np.float32), batch_size=batch_size, shuffle=False)
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, _ in loader:
            logits = model(x_batch.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            scores.append(probs.astype(np.float32))
    return np.concatenate(scores, axis=0)


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(np.int32)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def select_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = binary_metrics(y_true, y_score, best_threshold)
    for threshold in np.linspace(0.05, 0.95, 19):
        metrics = binary_metrics(y_true, y_score, float(threshold))
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def build_predictions_df(df: pd.DataFrame, split_name: str, y_score: np.ndarray, threshold: float) -> pd.DataFrame:
    out = df[
        [
            "scenario_id",
            "source_file",
            "scenario_type",
            "node_id",
            "signal_col",
            "label_col",
            "window_id",
            "window_size",
            "start_idx",
            "end_idx",
            "start_time_s",
            "end_time_s",
            "y_true",
        ]
    ].copy()
    out.insert(0, "split", split_name)
    out["y_score"] = y_score.astype(np.float32)
    out["y_pred"] = (out["y_score"] >= threshold).astype(int)
    return out


def metrics_rows_for_split(
    df: pd.DataFrame,
    split_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    best_epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
    device: torch.device,
    input_dim: int,
    pos_weight_value: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    overall_metrics = binary_metrics(y_true, y_score, threshold)
    rows.append(
        {
            "split": split_name,
            "scope": "overall",
            "node_id": "ALL",
            "num_windows": len(df),
            "positive_windows": int(y_true.sum()),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "threshold": threshold,
            "device": str(device),
            "input_dim": input_dim,
            "hidden_dims": "-".join(str(v) for v in args.hidden_dims),
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "pos_weight": pos_weight_value,
            **overall_metrics,
        }
    )

    scored = df[["node_id"]].copy()
    scored["y_true"] = y_true
    scored["y_score"] = y_score

    for node_id, group in scored.groupby("node_id", sort=True):
        node_true = group["y_true"].to_numpy(dtype=np.int32)
        node_score = group["y_score"].to_numpy(dtype=np.float32)
        node_metrics = binary_metrics(node_true, node_score, threshold)
        rows.append(
            {
                "split": split_name,
                "scope": "node",
                "node_id": node_id,
                "num_windows": len(group),
                "positive_windows": int(node_true.sum()),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "threshold": threshold,
                "device": str(device),
                "input_dim": input_dim,
                "hidden_dims": "-".join(str(v) for v in args.hidden_dims),
                "dropout": args.dropout,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "pos_weight": pos_weight_value,
                **node_metrics,
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs, batch-size, and patience must be positive.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_paths = {
        "train": Path(f"{args.input_prefix}_train.csv"),
        "val": Path(f"{args.input_prefix}_val.csv"),
        "test": Path(f"{args.input_prefix}_test.csv"),
    }

    train_df, feature_cols = read_split(split_paths["train"])
    val_df, _ = read_split(split_paths["val"], expected_features=feature_cols)
    test_df, _ = read_split(split_paths["test"], expected_features=feature_cols)

    train_x, val_x, test_x, mean, std = standardize_splits(train_df, val_df, test_df, feature_cols)
    train_y = train_df["y_true"].to_numpy(dtype=np.float32)
    val_y = val_df["y_true"].to_numpy(dtype=np.float32)
    test_y = test_df["y_true"].to_numpy(dtype=np.float32)

    train_loader = make_loader(train_x, train_y, args.batch_size, shuffle=True)
    val_loader = make_loader(val_x, val_y, args.batch_size, shuffle=False)

    model = MLPBinaryClassifier(
        input_dim=len(feature_cols),
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    positive_count = float(train_y.sum())
    negative_count = float(len(train_y) - positive_count)
    pos_weight_value = negative_count / positive_count if positive_count > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_rows: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            batch_size = x_batch.shape[0]
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        train_loss = running_loss / seen
        val_loss = compute_loss(model, val_loader, criterion, device)
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            break

    model.load_state_dict(best_state)

    train_scores = predict_scores(model, train_x, args.batch_size, device)
    val_scores = predict_scores(model, val_x, args.batch_size, device)
    test_scores = predict_scores(model, test_x, args.batch_size, device)

    threshold, val_threshold_metrics = select_threshold(val_y.astype(np.int32), val_scores)

    metrics_rows: list[dict[str, object]] = []
    for split_name, df, y_true, y_score in (
        ("train", train_df, train_y.astype(np.int32), train_scores),
        ("val", val_df, val_y.astype(np.int32), val_scores),
        ("test", test_df, test_y.astype(np.int32), test_scores),
    ):
        metrics_rows.extend(
            metrics_rows_for_split(
                df,
                split_name,
                y_true,
                y_score,
                threshold,
                best_epoch,
                best_val_loss,
                args,
                device,
                len(feature_cols),
                pos_weight_value,
            )
        )

    predictions_df = pd.concat(
        [
            build_predictions_df(train_df, "train", train_scores, threshold),
            build_predictions_df(val_df, "val", val_scores, threshold),
            build_predictions_df(test_df, "test", test_scores, threshold),
        ],
        ignore_index=True,
    )

    history_df = pd.DataFrame(history_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    scaler_df = pd.DataFrame({"feature": feature_cols, "mean": mean.astype(np.float32), "std": std.astype(np.float32)})

    history_path = Path(f"{args.output_prefix}_history.csv")
    metrics_path = Path(f"{args.output_prefix}_metrics.csv")
    preds_path = Path(f"{args.output_prefix}_predictions.csv")
    scaler_path = Path(f"{args.output_prefix}_scaler.csv")
    model_path = Path(f"{args.output_prefix}_model.pt")

    for path in (history_path, metrics_path, preds_path, scaler_path, model_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    history_df.to_csv(history_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(preds_path, index=False)
    scaler_df.to_csv(scaler_path, index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "threshold": threshold,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "scaler_mean": mean,
            "scaler_std": std,
        },
        model_path,
    )

    print(f"Saved: {history_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {preds_path}")
    print(f"Saved: {scaler_path}")
    print(f"Saved: {model_path}")
    print()
    print("Validation threshold selection:")
    print(pd.DataFrame([val_threshold_metrics]).to_string(index=False))
    print()
    print("Metrics by split and node:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
