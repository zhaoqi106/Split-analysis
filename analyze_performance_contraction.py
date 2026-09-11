#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_performance_contraction.py

Publication-oriented analysis of pooled out-of-fold (OOF) prediction performance
and prediction contraction for S1131 and M1101 under RCSV, PCSV, and ICSV.

The script reads the final five validation-fold prediction CSV files for each
model/dataset/split combination, standardizes their columns, pools the five
validation folds into OOF predictions, and writes only the outputs used by the
manuscript.

Outputs
-------
all_oof_predictions.csv
    Per-record OOF predictions with the columns required by the paired
    cross-split analysis:
        model, dataset, split, fold, ID, ddG_true, ddG_pred

pooled_oof_performance.csv
    Pooled OOF RMSE, MAE, and Pearson correlation coefficient (PCC).

prediction_contraction_metrics.csv
    Quantities used to characterize prediction contraction:
        observed range
        predicted range
        Rrange = predicted range / observed range
        observed standard deviation
        predicted standard deviation
        RSD = predicted SD / observed SD
        beta = prediction-on-observation attenuation slope

The regression used for beta is:

    ddG_pred = alpha + beta * ddG_true + error

Only beta is reported because it is the quantity used in the manuscript.
This is a prediction-on-observation attenuation slope, not a conventional
calibration slope.

Boundary rules
--------------
PCC is unavailable when either observed or predicted values have zero variance.
Rrange is unavailable when the observed range is zero.
RSD is unavailable when the observed standard deviation is zero.
beta is unavailable when the observed variance is zero.

The script intentionally does NOT calculate or output SCC, R2, CCC, Wilcoxon
tests, paired split statistics, figures, or development-only audit tables.

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
python analyze_performance_contraction.py \
    --prediction-root output/SHUJU_clean \
    --output-dir output/performance_contraction
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTION_ROOT = PROJECT_ROOT / "output" / "SHUJU_clean"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "performance_contraction"

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

ID_ALIASES = (
    "ID",
    "sample_id",
    "mutation_id",
    "mutant_id",
    "record_id",
    "id",
)


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\ufeff", "").strip()
    if text.casefold() in {"", "nan", "none", "null", "na"}:
        return ""
    return text


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

    # The benchmark files can contain both "ID" and "id". The exact uppercase
    # mutation target identifier must take precedence.
    if kind == "id" and "ID" in column_names:
        return "ID"

    # Prefer exact spelling first.
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
    elif kind == "pred":
        fuzzy = [
            column
            for column in column_names
            if any(
                token in normalize_column_name(column)
                for token in ("pred", "prediction", "estimate")
            )
        ]
    else:
        fuzzy = []

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


def canonical_prediction_frame(
    path: Path,
    model: str,
    dataset: str,
    split: str,
    fold: int,
    official_labels: pd.DataFrame,
) -> pd.DataFrame:
    raw = read_csv_robust(path)

    id_column = choose_column(raw.columns, ID_ALIASES, "id")
    true_column = choose_column(raw.columns, TRUE_ALIASES, "true")
    pred_column = choose_column(raw.columns, PRED_ALIASES, "pred")

    if true_column == pred_column:
        raise ValueError(
            f"{path}: true and prediction columns resolve to the same column "
            f"{true_column!r}."
        )

    result = pd.DataFrame(
        {
            "model": model,
            "dataset": dataset,
            "split": split,
            "fold": int(fold),
            "ID": raw[id_column].map(clean_text),
            "ddG_true": pd.to_numeric(
                raw[true_column],
                errors="coerce",
            ),
            "ddG_pred": pd.to_numeric(
                raw[pred_column],
                errors="coerce",
            ),
            "_row_order": np.arange(len(raw), dtype=int),
        }
    )

    invalid_id = result["ID"].eq("")
    invalid_values = (
        ~np.isfinite(result["ddG_true"])
        | ~np.isfinite(result["ddG_pred"])
    )

    if invalid_id.any():
        rows = result.index[invalid_id].tolist()[:20]
        raise ValueError(
            f"{path}: empty sample IDs at rows {rows}."
        )

    if invalid_values.any():
        rows = result.index[invalid_values].tolist()[:20]
        raise ValueError(
            f"{path}: non-finite ddG_true/ddG_pred values at rows {rows}."
        )

    return attach_official_labels(result, official_labels, path)


def load_official_master_labels(data_dir: Path, dataset: str) -> pd.DataFrame:
    path = data_dir / dataset / f"{dataset}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Official dataset CSV not found: {path}")

    raw = read_csv_robust(path)
    id_column = choose_column(raw.columns, ID_ALIASES, "id")
    true_column = choose_column(raw.columns, TRUE_ALIASES, "true")
    result = pd.DataFrame(
        {
            "ID": raw[id_column].map(clean_text),
            "ddG_true": pd.to_numeric(
                raw[true_column],
                errors="coerce",
            ),
        }
    )
    invalid = result["ID"].eq("") | ~np.isfinite(result["ddG_true"])
    if invalid.any():
        rows = result.index[invalid].tolist()[:20]
        raise ValueError(
            f"{path}: empty IDs or non-finite official ddG values at rows {rows}."
        )
    return result


def attach_official_labels(
    pred: pd.DataFrame,
    official: pd.DataFrame,
    source: Path,
) -> pd.DataFrame:
    """Replace prediction-file labels with official master-dataset ddG values.

    Some prediction CSVs store averaged or mixed replicate labels for repeated
    IDs such as 1DVF_YA49A. The master CSV keeps the original experimental
    values, so labels are reattached by ID in master-file order.
    """
    result = pred.copy()
    for target_id, pred_group in result.groupby("ID", sort=False):
        official_values = official.loc[
            official["ID"].eq(target_id),
            "ddG_true",
        ].to_numpy(dtype=float)
        if len(official_values) != len(pred_group):
            raise RuntimeError(
                f"{source}: ID {target_id!r} has {len(pred_group)} prediction "
                f"rows but {len(official_values)} official master labels."
            )
        result.loc[pred_group.index, "ddG_true"] = official_values
    return result


def label_map_by_id(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    mapping: dict[str, np.ndarray] = {}

    for target_id, group in frame.groupby("ID", sort=True):
        mapping[str(target_id)] = np.sort(
            group["ddG_true"].to_numpy(dtype=float)
        )

    return mapping


def compare_label_maps(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    tolerance: float,
) -> tuple[bool, str]:
    if set(reference) != set(candidate):
        missing = sorted(set(reference) - set(candidate))[:10]
        extra = sorted(set(candidate) - set(reference))[:10]
        return (
            False,
            f"ID sets differ, missing={missing}, extra={extra}",
        )

    for target_id in reference:
        first = reference[target_id]
        second = candidate[target_id]

        if len(first) != len(second):
            return (
                False,
                f"ID {target_id!r} has different multiplicity "
                f"({len(first)} vs {len(second)}).",
            )

        if not np.allclose(
            first,
            second,
            rtol=0.0,
            atol=tolerance,
            equal_nan=False,
        ):
            return (
                False,
                f"ID {target_id!r} has inconsistent ddG_true values "
                f"{first.tolist()} vs {second.tolist()}.",
            )

    return True, ""


def validate_oof_label_consistency(
    pooled: pd.DataFrame,
    tolerance: float,
) -> None:
    """Require the same complete labeled dataset for every model and split."""
    for dataset, dataset_frame in pooled.groupby(
        "dataset",
        sort=False,
    ):
        reference_map: dict[str, np.ndarray] | None = None
        reference_name = ""

        for (model, split), group in dataset_frame.groupby(
            ["model", "split"],
            sort=True,
        ):
            current_map = label_map_by_id(group)

            if reference_map is None:
                reference_map = current_map
                reference_name = f"{model}/{split}"
                continue

            consistent, message = compare_label_maps(
                reference_map,
                current_map,
                tolerance=tolerance,
            )
            if not consistent:
                raise RuntimeError(
                    f"{dataset}: pooled OOF label inconsistency between "
                    f"{reference_name} and {model}/{split}: {message}"
                )


def safe_pcc(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    if len(observed) < 2:
        return math.nan

    if np.std(observed, ddof=0) == 0:
        return math.nan
    if np.std(predicted, ddof=0) == 0:
        return math.nan

    return float(np.corrcoef(observed, predicted)[0, 1])


def attenuation_slope(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    if len(observed) < 2:
        return math.nan

    observed_variance = float(np.var(observed, ddof=0))
    if observed_variance == 0:
        return math.nan

    covariance = float(
        np.mean(
            (observed - np.mean(observed))
            * (predicted - np.mean(predicted))
        )
    )
    return covariance / observed_variance


def pooled_performance_record(
    group: pd.DataFrame,
) -> dict[str, object]:
    observed = group["ddG_true"].to_numpy(dtype=float)
    predicted = group["ddG_pred"].to_numpy(dtype=float)

    error = predicted - observed

    return {
        "n": int(len(group)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "pcc": safe_pcc(observed, predicted),
    }


def contraction_record(
    group: pd.DataFrame,
) -> dict[str, object]:
    observed = group["ddG_true"].to_numpy(dtype=float)
    predicted = group["ddG_pred"].to_numpy(dtype=float)

    observed_range = float(np.max(observed) - np.min(observed))
    predicted_range = float(np.max(predicted) - np.min(predicted))

    observed_sd = (
        float(np.std(observed, ddof=1))
        if len(observed) > 1
        else math.nan
    )
    predicted_sd = (
        float(np.std(predicted, ddof=1))
        if len(predicted) > 1
        else math.nan
    )

    range_ratio = (
        predicted_range / observed_range
        if observed_range > 0
        else math.nan
    )
    sd_ratio = (
        predicted_sd / observed_sd
        if np.isfinite(observed_sd) and observed_sd > 0
        else math.nan
    )

    return {
        "n": int(len(group)),
        "observed_range": observed_range,
        "predicted_range": predicted_range,
        "Rrange_predicted_over_observed": range_ratio,
        "observed_sd": observed_sd,
        "predicted_sd": predicted_sd,
        "RSD_predicted_over_observed": sd_ratio,
        "attenuation_slope_beta": attenuation_slope(
            observed,
            predicted,
        ),
    }


def build_publication_tables(
    pooled: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    performance_rows: list[dict[str, object]] = []
    contraction_rows: list[dict[str, object]] = []

    grouped = pooled.groupby(
        ["model", "dataset", "split"],
        sort=True,
        dropna=False,
    )

    for (model, dataset, split), group in grouped:
        base = {
            "model": model,
            "dataset": dataset,
            "split": split,
        }

        performance_rows.append(
            {
                **base,
                **pooled_performance_record(group),
            }
        )
        contraction_rows.append(
            {
                **base,
                **contraction_record(group),
            }
        )

    performance = pd.DataFrame(performance_rows)
    contraction = pd.DataFrame(contraction_rows)

    dataset_order = {
        dataset: index
        for index, dataset in enumerate(DATASETS)
    }
    split_order = {
        split: index
        for index, split in enumerate(SPLITS)
    }

    def sort_table(frame: pd.DataFrame) -> pd.DataFrame:
        work = frame.copy()
        work["_dataset_order"] = work["dataset"].map(dataset_order)
        work["_split_order"] = work["split"].map(split_order)
        work = work.sort_values(
            ["dataset", "model", "_split_order"],
            kind="stable",
        )
        return work.drop(
            columns=["_dataset_order", "_split_order"]
        ).reset_index(drop=True)

    return sort_table(performance), sort_table(contraction)


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

    publication_files = (
        "all_oof_predictions.csv",
        "pooled_oof_performance.csv",
        "prediction_contraction_metrics.csv",
    )

    if output.exists():
        existing = [
            output / filename
            for filename in publication_files
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

    output.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build pooled OOF predictions and calculate manuscript RMSE, MAE, "
            "PCC, range ratio, SD ratio, and prediction-on-observation "
            "attenuation slope."
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
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory of {dataset}/{dataset}.csv official label files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--label-tolerance",
        type=float,
        default=1e-6,
        help=(
            "Absolute tolerance used only to verify that ddG_true values are "
            "consistent across models and split conditions."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the three publication output CSV files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.label_tolerance < 0:
        raise ValueError("--label-tolerance must be non-negative.")


def main() -> int:
    args = parse_args()
    validate_args(args)

    prepare_output_dir(
        args.output_dir,
        args.prediction_root,
        args.overwrite,
    )

    print("========== analyze_performance_contraction.py ==========")
    print(f"Prediction root: {args.prediction_root}")
    print(f"Data directory:  {args.data_dir}")
    print(f"Output directory: {args.output_dir}")

    discovered = discover_prediction_files(args.prediction_root)
    records = require_unique_complete_design(discovered)

    models = sorted({str(record["model"]) for record in records})
    print(f"Models: {len(models)}")
    print(f"Expected combinations: {len(models) * len(DATASETS) * len(SPLITS)}")
    print(f"Prediction fold files: {len(records)}")

    official_by_dataset = {
        dataset: load_official_master_labels(args.data_dir, dataset)
        for dataset in DATASETS
    }

    pooled_pieces: list[pd.DataFrame] = []

    for index, record in enumerate(records, start=1):
        frame = canonical_prediction_frame(
            path=Path(record["path"]),
            model=str(record["model"]),
            dataset=str(record["dataset"]),
            split=str(record["split"]),
            fold=int(record["fold"]),
            official_labels=official_by_dataset[str(record["dataset"])],
        )
        pooled_pieces.append(frame)

        if index % 25 == 0 or index == len(records):
            print(f"[READ] {index}/{len(records)} fold files")

    pooled = pd.concat(
        pooled_pieces,
        ignore_index=True,
    )

    validate_oof_label_consistency(
        pooled,
        tolerance=args.label_tolerance,
    )

    performance, contraction = build_publication_tables(pooled)

    pooled = pooled.sort_values(
        ["dataset", "model", "split", "fold", "_row_order"],
        kind="stable",
    ).drop(columns="_row_order")

    pooled = pooled[
        [
            "model",
            "dataset",
            "split",
            "fold",
            "ID",
            "ddG_true",
            "ddG_pred",
        ]
    ].reset_index(drop=True)

    oof_path = args.output_dir / "all_oof_predictions.csv"
    performance_path = (
        args.output_dir / "pooled_oof_performance.csv"
    )
    contraction_path = (
        args.output_dir / "prediction_contraction_metrics.csv"
    )

    pooled.to_csv(
        oof_path,
        index=False,
        encoding="utf-8-sig",
    )
    performance.to_csv(
        performance_path,
        index=False,
        encoding="utf-8-sig",
    )
    contraction.to_csv(
        contraction_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[RESULT]")
    print(f"OOF prediction rows: {len(pooled)}")
    print(f"Model-dataset-split combinations: {len(performance)}")

    print("\n[WRITE]")
    print(f"  {oof_path}")
    print(f"  {performance_path}")
    print(f"  {contraction_path}")
    print("[DONE] Performance and contraction analysis completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
