#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculate_metrics.py

Publication-oriented calculation of validation-fold prediction metrics for
S1131 and M1101 under RCSV, PCSV, and ICSV.

This script reproduces the fold-level RMSE, MAE, and Pearson correlation
coefficient (PCC) values used in Supplementary Tables S2 and S3, together with
their arithmetic mean and sample standard deviation across the five validation
folds.

Outputs
-------
fold_metrics.csv
    One row per model/dataset/split/fold with:
        model
        dataset
        split
        fold
        n
        RMSE
        MAE
        PCC

fold_mean_std.csv
    One row per model/dataset/split with:
        model
        dataset
        split
        RMSE_mean
        RMSE_std
        MAE_mean
        MAE_std
        PCC_mean
        PCC_std

Definitions
-----------
For each validation fold containing n records:

    RMSE = sqrt(mean((prediction - observed)^2))
    MAE  = mean(abs(prediction - observed))
    PCC  = Pearson correlation between observed and predicted values

PCC is unavailable when either vector has zero variance.

The five-fold mean is the unweighted arithmetic mean of the five fold-level
metrics. The reported standard deviation is the sample standard deviation across
the five folds, using denominator K - 1.

The script intentionally does NOT calculate pooled OOF metrics, SCC, R2, CCC,
prediction contraction, paired tests, or development-only audit quantities.
Those analyses are handled by the other publication scripts.

Expected prediction layout
--------------------------
Prediction files may be nested below:

    PREDICTION_ROOT / MODEL / DATASET / SPLIT / {DATASET}_{SPLIT}_fold{K}_pred.csv

where DATASET is S1131 or M1101, SPLIT is RCSV, PCSV, or ICSV, and K is 0-4.
The relative path after the split directory must contain one fold marker such
as:

    fold0
    fold_0
    fold-0

Exactly one CSV must be identified for every model/dataset/split/fold
combination. Ambiguous inputs cause an error rather than an automatic guess.

Example
-------
python calculate_metrics.py \
    --prediction-root output/SHUJU_clean \
    --output-dir output/fold_metrics
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTION_ROOT = PROJECT_ROOT / "output" / "SHUJU_clean"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "fold_metrics"

DATASETS = ("S1131", "M1101")
SPLITS = ("RCSV", "PCSV", "ICSV")
FOLDS = tuple(range(5))

PRED_FILENAME_RE = re.compile(
    r"^(S1131|M1101)_(RCSV|PCSV|ICSV)_fold([0-4])_pred\.csv$",
    re.IGNORECASE,
)

TRUE_ALIASES = (
    "ddG_true",
    "ddg_true",
    "true_ddG",
    "experimental_ddG",
    "experimental",
    "label",
    "target",
    "ddG",
    "ddg",
)

PRED_ALIASES = (
    "ddG_pred",
    "ddg_pred",
    "pred_ddG",
    "predicted_ddG",
    "Prediction",
    "prediction",
    "pred",
    "y_pred",
)


def normalize_column_name(value: object) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).strip().casefold()
        .replace("δ", "delta")
        .replace("Δ", "delta"),
    )


def choose_column(
    columns: Sequence[object],
    aliases: Sequence[str],
    kind: str,
) -> str:
    column_names = [str(column) for column in columns]

    for alias in aliases:
        if alias in column_names:
            return alias

    normalized: dict[str, list[str]] = defaultdict(list)
    for column in column_names:
        normalized[normalize_column_name(column)].append(column)

    for alias in aliases:
        matches = normalized.get(normalize_column_name(alias), [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous {kind} columns for alias {alias!r}: {matches}"
            )

    if kind == "true":
        fuzzy = [
            column
            for column in column_names
            if any(
                token in normalize_column_name(column)
                for token in (
                    "true",
                    "label",
                    "target",
                    "actual",
                    "experimental",
                    "groundtruth",
                )
            )
            and "pred" not in normalize_column_name(column)
        ]
    else:
        fuzzy = [
            column
            for column in column_names
            if any(
                token in normalize_column_name(column)
                for token in ("pred", "prediction", "estimate")
            )
        ]

    if len(fuzzy) == 1:
        return fuzzy[0]

    raise ValueError(
        f"Could not uniquely identify the {kind} column. "
        f"Available columns: {column_names}"
    )


def read_csv_robust(path: Path) -> pd.DataFrame:
    errors: list[str] = []

    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
            )
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")

    raise RuntimeError(
        f"Could not decode CSV {path}: {'; '.join(errors)}"
    )


def discover_prediction_files(
    prediction_root: Path,
) -> list[dict[str, object]]:
    root = prediction_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Prediction root not found: {root}")

    records: list[dict[str, object]] = []

    for path in sorted(root.rglob("*.csv")):
        name_match = PRED_FILENAME_RE.fullmatch(path.name)
        if name_match is None:
            continue

        relative = path.relative_to(root)
        parts = relative.parts

        split_indices = [
            index
            for index, part in enumerate(parts[:-1])
            if part.upper() in SPLITS
        ]
        if len(split_indices) != 1:
            continue

        split_index = split_indices[0]
        if split_index < 2:
            continue

        split = parts[split_index].upper()
        dataset = parts[split_index - 1]
        if dataset not in DATASETS:
            continue

        model_parts = parts[: split_index - 1]
        if not model_parts:
            continue

        model = "/".join(model_parts)
        name_dataset, name_split, name_fold = name_match.groups()
        if name_dataset.upper() != dataset or name_split.upper() != split:
            continue

        fold = int(name_fold)

        records.append(
            {
                "model": model,
                "dataset": dataset,
                "split": split,
                "fold": fold,
                "path": path,
                "relative_path": str(relative),
            }
        )

    if not records:
        raise RuntimeError(
            "No prediction CSV files matching MODEL/DATASET/SPLIT/...fold... "
            f"were found below {root}."
        )

    return records


def require_unique_complete_design(
    records: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, str, str, int],
        list[dict[str, object]],
    ] = defaultdict(list)

    for record in records:
        key = (
            str(record["model"]),
            str(record["dataset"]),
            str(record["split"]),
            int(record["fold"]),
        )
        grouped[key].append(record)

    ambiguous = {
        key: group
        for key, group in grouped.items()
        if len(group) > 1
    }
    if ambiguous:
        preview = []
        for key, group in list(ambiguous.items())[:20]:
            preview.append(
                f"{key}: {[str(item['relative_path']) for item in group]}"
            )
        raise RuntimeError(
            "Multiple CSV files were identified for the same "
            "model/dataset/split/fold. Refusing to guess:\n"
            + "\n".join(preview)
        )

    unique_records = [group[0] for group in grouped.values()]
    models = sorted({str(record["model"]) for record in unique_records})

    missing: list[str] = []
    available_keys = {
        (
            str(record["model"]),
            str(record["dataset"]),
            str(record["split"]),
            int(record["fold"]),
        )
        for record in unique_records
    }

    for model in models:
        for dataset in DATASETS:
            for split in SPLITS:
                for fold in FOLDS:
                    key = (model, dataset, split, fold)
                    if key not in available_keys:
                        missing.append(
                            f"{model}/{dataset}/{split}/fold{fold}"
                        )

    if missing:
        preview = "\n".join(missing[:30])
        suffix = (
            ""
            if len(missing) <= 30
            else f"\n... and {len(missing) - 30} more"
        )
        raise RuntimeError(
            "The publication design is incomplete. Missing prediction files:\n"
            + preview
            + suffix
        )

    return sorted(
        unique_records,
        key=lambda record: (
            str(record["model"]),
            DATASETS.index(str(record["dataset"])),
            SPLITS.index(str(record["split"])),
            int(record["fold"]),
        ),
    )


def calculate_fold_metrics(
    path: Path,
) -> dict[str, object]:
    frame = read_csv_robust(path)

    true_column = choose_column(
        frame.columns,
        TRUE_ALIASES,
        "true",
    )
    pred_column = choose_column(
        frame.columns,
        PRED_ALIASES,
        "pred",
    )

    if true_column == pred_column:
        raise ValueError(
            f"{path}: true and prediction columns resolve to the same "
            f"column {true_column!r}."
        )

    observed = pd.to_numeric(
        frame[true_column],
        errors="coerce",
    ).to_numpy(dtype=float)
    predicted = pd.to_numeric(
        frame[pred_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    invalid = ~np.isfinite(observed) | ~np.isfinite(predicted)
    if invalid.any():
        rows = np.flatnonzero(invalid)[:20].tolist()
        raise ValueError(
            f"{path}: non-finite observed/predicted values at rows {rows}."
        )

    if len(observed) == 0:
        raise ValueError(f"{path}: no prediction records.")

    error = predicted - observed
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))

    if (
        len(observed) < 2
        or np.std(observed, ddof=0) == 0
        or np.std(predicted, ddof=0) == 0
    ):
        pcc = math.nan
    else:
        pcc = float(np.corrcoef(observed, predicted)[0, 1])

    return {
        "n": int(len(observed)),
        "RMSE": rmse,
        "MAE": mae,
        "PCC": pcc,
    }


def build_mean_std(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (model, dataset, split), group in fold_metrics.groupby(
        ["model", "dataset", "split"],
        sort=True,
        dropna=False,
    ):
        if sorted(group["fold"].astype(int).tolist()) != list(FOLDS):
            raise RuntimeError(
                f"{model}/{dataset}/{split}: expected folds 0-4."
            )

        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "split": split,
                "RMSE_mean": float(group["RMSE"].mean()),
                "RMSE_std": float(group["RMSE"].std(ddof=1)),
                "MAE_mean": float(group["MAE"].mean()),
                "MAE_std": float(group["MAE"].std(ddof=1)),
                "PCC_mean": float(group["PCC"].mean()),
                "PCC_std": float(group["PCC"].std(ddof=1)),
            }
        )

    summary = pd.DataFrame(rows)

    dataset_order = {
        dataset: index
        for index, dataset in enumerate(DATASETS)
    }
    split_order = {
        split: index
        for index, split in enumerate(SPLITS)
    }

    summary["_dataset_order"] = summary["dataset"].map(
        dataset_order
    )
    summary["_split_order"] = summary["split"].map(
        split_order
    )

    return (
        summary.sort_values(
            ["_dataset_order", "model", "_split_order"],
            kind="stable",
        )
        .drop(columns=["_dataset_order", "_split_order"])
        .reset_index(drop=True)
    )


def prepare_output_dir(
    output_dir: Path,
    prediction_root: Path,
    overwrite: bool,
) -> None:
    output = output_dir.resolve()
    prediction = prediction_root.resolve()

    if output == prediction or prediction in output.parents:
        raise RuntimeError(
            "Output directory must be outside the prediction input tree."
        )

    output_files = (
        "fold_metrics.csv",
        "fold_mean_std.csv",
    )

    if output.exists():
        existing = [
            output / filename
            for filename in output_files
            if (output / filename).exists()
        ]

        if existing and not overwrite:
            raise FileExistsError(
                f"Publication outputs already exist in {output}. "
                "Use --overwrite to replace them."
            )

        if overwrite:
            for path in existing:
                path.unlink()

    output.mkdir(
        parents=True,
        exist_ok=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate validation-fold RMSE, MAE, and PCC, plus their "
            "unweighted five-fold mean and sample standard deviation."
        )
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=DEFAULT_PREDICTION_ROOT,
        help=(
            "Directory of MODEL/DATASET/SPLIT/"
            "{DATASET}_{SPLIT}_foldK_pred.csv files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace fold_metrics.csv and fold_mean_std.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    prepare_output_dir(
        args.output_dir,
        args.prediction_root,
        args.overwrite,
    )

    print("========== calculate_metrics.py ==========")
    print(f"Prediction root: {args.prediction_root}")
    print(f"Output directory: {args.output_dir}")

    discovered = discover_prediction_files(
        args.prediction_root
    )
    records = require_unique_complete_design(
        discovered
    )

    models = sorted(
        {
            str(record["model"])
            for record in records
        }
    )

    print(f"Models: {len(models)}")
    print(f"Fold files: {len(records)}")

    rows: list[dict[str, object]] = []

    for index, record in enumerate(records, start=1):
        metrics = calculate_fold_metrics(
            Path(record["path"])
        )

        rows.append(
            {
                "model": str(record["model"]),
                "dataset": str(record["dataset"]),
                "split": str(record["split"]),
                "fold": int(record["fold"]),
                **metrics,
            }
        )

        if index % 25 == 0 or index == len(records):
            print(f"[CALCULATE] {index}/{len(records)} fold files")

    fold_metrics = pd.DataFrame(rows)

    dataset_order = {
        dataset: index
        for index, dataset in enumerate(DATASETS)
    }
    split_order = {
        split: index
        for index, split in enumerate(SPLITS)
    }

    fold_metrics["_dataset_order"] = fold_metrics["dataset"].map(
        dataset_order
    )
    fold_metrics["_split_order"] = fold_metrics["split"].map(
        split_order
    )

    fold_metrics = (
        fold_metrics.sort_values(
            [
                "_dataset_order",
                "model",
                "_split_order",
                "fold",
            ],
            kind="stable",
        )
        .drop(
            columns=[
                "_dataset_order",
                "_split_order",
            ]
        )
        .reset_index(drop=True)
    )

    mean_std = build_mean_std(
        fold_metrics
    )

    fold_path = (
        args.output_dir
        / "fold_metrics.csv"
    )
    summary_path = (
        args.output_dir
        / "fold_mean_std.csv"
    )

    fold_metrics.to_csv(
        fold_path,
        index=False,
        encoding="utf-8-sig",
    )
    mean_std.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[RESULT]")
    print(f"Fold-level rows: {len(fold_metrics)}")
    print(
        "Model-dataset-split combinations: "
        f"{len(mean_std)}"
    )

    print("\n[WRITE]")
    print(f"  {fold_path}")
    print(f"  {summary_path}")
    print("[DONE] Fold metrics completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
