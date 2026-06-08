#!/usr/bin/env python3
"""
Build train/validation/test GraphLeak splits at scenario level.

This script reads the semantically named volume experiments dataset and creates
three reproducible CSV splits without leaking windows from the same scenario
across train, validation, and test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = "graphleak_volume_experiments.csv"
DEFAULT_OUTPUT_PREFIX = "graphleak_volume"
DEFAULT_TRAIN_RATIO = 0.60
DEFAULT_VAL_RATIO = 0.20
DEFAULT_TEST_RATIO = 0.20
DEFAULT_SEED = 42

REQUIRED_COLUMNS = {
    "scenario_id",
    "source_file",
    "scenario_type",
}


def validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise ValueError(f"Input CSV is missing required columns: {missing_csv}")


def scenario_metadata(df: pd.DataFrame) -> pd.DataFrame:
    meta = (
        df.groupby("scenario_id", sort=True, dropna=False)
        .agg(
            source_file=("source_file", "first"),
            scenario_type=("scenario_type", "first"),
            num_rows=("scenario_id", "size"),
        )
        .reset_index()
    )

    duplicates = (
        df.groupby("scenario_id", sort=True, dropna=False)[["source_file", "scenario_type"]]
        .nunique(dropna=False)
        .reset_index()
    )
    inconsistent = duplicates[
        (duplicates["source_file"] > 1) | (duplicates["scenario_type"] > 1)
    ]
    if not inconsistent.empty:
        scenario_ids = ", ".join(str(v) for v in inconsistent["scenario_id"].tolist())
        raise ValueError(
            f"Found scenario_id values with inconsistent source_file/scenario_type: {scenario_ids}"
        )

    return meta.sort_values("scenario_id").reset_index(drop=True)


def split_counts(n_items: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if n_items < 3:
        raise ValueError(
            f"Each scenario_type bucket must have at least 3 scenarios. Got {n_items}."
        )

    raw_train = n_items * train_ratio
    raw_val = n_items * val_ratio
    train_count = int(np.floor(raw_train))
    val_count = int(np.floor(raw_val))
    test_count = n_items - train_count - val_count

    counts = {"train": train_count, "val": val_count, "test": test_count}

    for split_name in ("val", "test"):
        if counts[split_name] == 0:
            donor = "train" if counts["train"] > 1 else None
            if donor is None:
                raise ValueError(f"Could not allocate at least one scenario to {split_name}.")
            counts[donor] -= 1
            counts[split_name] += 1

    while counts["train"] <= 0:
        donor = "test" if counts["test"] > 1 else "val"
        if counts[donor] <= 1:
            raise ValueError("Could not allocate a positive number of training scenarios.")
        counts[donor] -= 1
        counts["train"] += 1

    total = counts["train"] + counts["val"] + counts["test"]
    if total != n_items:
        raise ValueError(f"Split count mismatch: expected {n_items}, got {total}.")

    return counts["train"], counts["val"], counts["test"]


def assign_splits(
    meta: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for scenario_type, group in meta.groupby("scenario_type", sort=True, dropna=False):
        indices = group.index.to_numpy()
        shuffled = rng.permutation(indices)
        train_count, val_count, test_count = split_counts(len(group), train_ratio, val_ratio)

        train_idx = shuffled[:train_count]
        val_idx = shuffled[train_count : train_count + val_count]
        test_idx = shuffled[train_count + val_count : train_count + val_count + test_count]

        for idx in train_idx:
            rows.append((idx, "train"))
        for idx in val_idx:
            rows.append((idx, "val"))
        for idx in test_idx:
            rows.append((idx, "test"))

    split_df = pd.DataFrame(rows, columns=["index", "split"])
    out = meta.reset_index().merge(split_df, on="index", how="left").drop(columns=["index"])

    if out["split"].isna().any():
        raise ValueError("Some scenarios were not assigned to a split.")

    return out.sort_values("scenario_id").reset_index(drop=True)


def write_split_csvs(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
    output_prefix: str,
    output_dir: Path,
) -> list[Path]:
    paths = []
    manifest_path = output_dir / f"{output_prefix}_split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    paths.append(manifest_path)

    for split_name in ("train", "val", "test"):
        scenario_ids = manifest.loc[manifest["split"] == split_name, "scenario_id"]
        split_df = df[df["scenario_id"].isin(scenario_ids)].copy()
        split_df = split_df.sort_values(["scenario_id", "time_s"]).reset_index(drop=True)

        split_path = output_dir / f"{output_prefix}_{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        paths.append(split_path)

    return paths


def print_summary(df: pd.DataFrame, manifest: pd.DataFrame) -> None:
    scenario_summary = (
        manifest.groupby(["split", "scenario_type"], sort=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    row_summary = (
        df.merge(manifest[["scenario_id", "split"]], on="scenario_id", how="left")
        .groupby(["split", "scenario_type"], sort=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    print("Scenario counts by split and type:")
    print(scenario_summary.to_string())
    print()
    print("Row counts by split and type:")
    print(row_summary.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train/validation/test CSV splits from GraphLeak volume experiments."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to graphleak_volume_experiments.csv.",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Common prefix for the output CSV files.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Scenario-level ratio assigned to the training split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Scenario-level ratio assigned to the validation split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for reproducible scenario assignment.",
    )
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise ValueError("train-ratio and val-ratio must be positive.")

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train-ratio + val-ratio must be less than 1.0.")

    input_path = Path(args.input)
    output_dir = input_path.parent

    df = pd.read_csv(input_path)
    validate_input(df)

    meta = scenario_metadata(df)
    manifest = assign_splits(meta, args.train_ratio, args.val_ratio, args.seed)
    paths = write_split_csvs(df, manifest, args.output_prefix, output_dir)

    for path in paths:
        print(f"Saved: {path}")
    print()
    print_summary(df, manifest)


if __name__ == "__main__":
    main()
