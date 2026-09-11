#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paired_analysis.py

Publication-oriented same-target paired absolute-error analysis for the final
pooled out-of-fold predictions.

The script reproduces Supplementary Tables S5 and S6 by comparing PCSV and ICSV
with RCSV for the same target identifiers.

Primary definition
------------------
For target i:

    delta_AE_i =
        |prediction_grouped_i - true_i|
        - |prediction_RCSV_i - true_i|

Positive delta_AE indicates greater absolute error under PCSV or ICSV.

Repeated target identifiers
---------------------------
Repeated IDs are aggregated within each model/dataset/split before pairing.
The arithmetic mean prediction and arithmetic mean true value are used for each
unique ID. Experimental replicates that share an ID but have different measured
ddG values are averaged. This gives:

    S1131: n = 1131 paired targets
    M1101: n = 1096 paired targets from 1101 records

Complete-PDB-group inference
----------------------------
The primary uncertainty analysis is fixed to the method reported in the paper:

    10,000 complete-PDB-group bootstrap resamples
    100,000 complete-PDB-group Monte Carlo sign-flip iterations
    Benjamini-Hochberg correction across all 36 comparisons
    FDR alpha = 0.05
    random seed = 20260802

Complete PDB/complex groups are derived from target IDs while preserving the
HM_ prefix. For example:

    1YY9_AH98W    -> 1YY9
    HM_1YY9_AH98W -> HM_1YY9

Output
------
paired_error_summary.csv

One row per model/dataset/comparison, with exactly the quantities needed for
Supplementary Tables S5 and S6:

    model
    dataset
    comparison_split
    n
    delta_AE
    ci_95_low
    ci_95_high
    p
    q
    significant

The 95 percent confidence interval is pointwise and unadjusted. The p value is
from the two-sided complete-PDB-group Monte Carlo sign-flip test. The q value
is the Benjamini-Hochberg-adjusted p value across the complete 36-comparison
family.

The script intentionally does NOT output Wilcoxon tests, binomial sign tests,
duplicate-audit tables, record-weighted sensitivity tables, or other
development-only analyses.

Input
-----
Use all_oof_predictions.csv produced by analyze_performance_contraction.py.
Required columns are:

    model, dataset, split, ID, ddG_true, ddG_pred

Example
-------
python paired_analysis.py \
    --predictions output/performance_contraction/all_oof_predictions.csv \
    --output-dir output/paired_analysis
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT / "output" / "performance_contraction" / "all_oof_predictions.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "paired_analysis"

REQUIRED_COLUMNS = {
    "model",
    "dataset",
    "split",
    "ID",
    "ddG_true",
    "ddG_pred",
}

REFERENCE_SPLIT = "RCSV"
COMPARISON_SPLITS = ("PCSV", "ICSV")
DATASETS = ("S1131", "M1101")

BOOTSTRAP_ITERATIONS = 10000
SIGN_FLIP_ITERATIONS = 100000
FDR_ALPHA = 0.05
RANDOM_SEED = 20260802
TRUE_TOLERANCE = 1e-6
EXPECTED_BH_FAMILY_SIZE = 36


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\ufeff", "").strip()
    if text.casefold() in {"", "nan", "none", "null", "na"}:
        return ""
    return text


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


def derive_complete_pdb_group(target_id: object) -> str:
    """Derive the complete complex group without collapsing HM_* identifiers."""
    if target_id is None or pd.isna(target_id):
        return ""

    text = str(target_id).strip().upper()
    if not text:
        return ""

    parts = text.split("_")
    if not parts or not parts[0]:
        return ""

    if parts[0] == "HM":
        if len(parts) >= 2 and parts[1]:
            return f"HM_{parts[1]}"
        return ""

    return parts[0]


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input is missing required columns: {sorted(missing)}"
        )

    work = df[
        [
            "model",
            "dataset",
            "split",
            "ID",
            "ddG_true",
            "ddG_pred",
        ]
    ].copy()

    for column in ("model", "dataset", "split", "ID"):
        work[column] = work[column].map(clean_text)

    work["split"] = work["split"].str.upper()
    work["ddG_true"] = pd.to_numeric(
        work["ddG_true"],
        errors="coerce",
    )
    work["ddG_pred"] = pd.to_numeric(
        work["ddG_pred"],
        errors="coerce",
    )

    invalid_text = (
        work["model"].eq("")
        | work["dataset"].eq("")
        | work["split"].eq("")
        | work["ID"].eq("")
    )
    if invalid_text.any():
        rows = work.index[invalid_text].tolist()[:20]
        raise ValueError(
            f"Blank model/dataset/split/ID values at rows: {rows}"
        )

    invalid_numeric = (
        ~np.isfinite(work["ddG_true"])
        | ~np.isfinite(work["ddG_pred"])
    )
    if invalid_numeric.any():
        rows = work.index[invalid_numeric].tolist()[:20]
        raise ValueError(
            f"Non-finite ddG_true/ddG_pred values at rows: {rows}"
        )

    unexpected_dataset = ~work["dataset"].isin(DATASETS)
    if unexpected_dataset.any():
        values = sorted(work.loc[unexpected_dataset, "dataset"].unique())
        raise ValueError(
            f"Unexpected datasets: {values}. "
            f"Publication datasets are {DATASETS}."
        )

    required_splits = {REFERENCE_SPLIT, *COMPARISON_SPLITS}
    unexpected_split = ~work["split"].isin(required_splits)
    if unexpected_split.any():
        values = sorted(work.loc[unexpected_split, "split"].unique())
        raise ValueError(
            f"Unexpected splits: {values}. "
            f"Publication splits are {sorted(required_splits)}."
        )

    work["complete_pdb_group"] = work["ID"].map(
        derive_complete_pdb_group
    )
    invalid_group = work["complete_pdb_group"].eq("")
    if invalid_group.any():
        rows = work.index[invalid_group].tolist()[:20]
        raise ValueError(
            "Could not derive complete PDB/complex group from some IDs. "
            f"Example row indices: {rows}"
        )

    return work


def aggregate_unique_targets(
    frame: pd.DataFrame,
    true_tolerance: float,
) -> pd.DataFrame:
    """Aggregate repeated IDs within one model/dataset/split.

    Extra internal columns are retained only to preserve the exact finalized
    Monte Carlo random stream used for the published analysis. They are not
    written to the publication output.
    """
    if frame.empty:
        return pd.DataFrame()

    work = frame.copy()
    work["record_absolute_error"] = (
        work["ddG_pred"] - work["ddG_true"]
    ).abs()

    grouped = work.groupby("ID", sort=False, dropna=False)

    result = grouped.agg(
        ddG_true=("ddG_true", "mean"),
        ddG_pred=("ddG_pred", "mean"),
        duplicate_rows=("ID", "size"),
        true_min=("ddG_true", "min"),
        true_max=("ddG_true", "max"),
        mean_record_absolute_error=(
            "record_absolute_error",
            "mean",
        ),
        complete_pdb_group=("complete_pdb_group", "first"),
        group_nunique=("complete_pdb_group", "nunique"),
    ).reset_index()

    result["true_range"] = (
        result["true_max"] - result["true_min"]
    )

    inconsistent_group = result["group_nunique"] != 1
    if inconsistent_group.any():
        bad = result.loc[inconsistent_group, "ID"].head(20).tolist()
        raise RuntimeError(
            "Repeated target IDs map to multiple complete PDB groups: "
            f"{bad}"
        )

    return result


def weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    mask = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if not mask.any():
        return math.nan

    return float(
        np.average(
            values[mask],
            weights=weights[mask],
        )
    )


def complete_group_bootstrap_ci(
    values: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap the weighted mean by resampling complete PDB groups."""
    mask = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
        & pd.notna(groups)
    )

    values = values[mask]
    weights = weights[mask]
    groups = groups[mask]

    if len(values) == 0:
        return math.nan, math.nan

    grouped = (
        pd.DataFrame(
            {
                "group": groups.astype(str),
                "weighted_value": values * weights,
                "weight": weights,
            }
        )
        .groupby("group", sort=False, as_index=False)
        .agg(
            numerator=("weighted_value", "sum"),
            denominator=("weight", "sum"),
        )
    )

    numerators = grouped["numerator"].to_numpy(dtype=float)
    denominators = grouped["denominator"].to_numpy(dtype=float)
    n_groups = len(grouped)

    if n_groups == 0:
        return math.nan, math.nan

    estimates = np.empty(iterations, dtype=float)

    for start in range(0, iterations, 500):
        count = min(500, iterations - start)
        indices = rng.integers(
            0,
            n_groups,
            size=(count, n_groups),
        )
        estimates[start : start + count] = (
            numerators[indices].sum(axis=1)
            / denominators[indices].sum(axis=1)
        )

    low, high = np.quantile(
        estimates,
        [0.025, 0.975],
    )
    return float(low), float(high)


def complete_group_sign_flip_p(
    values: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> float:
    """Two-sided Monte Carlo sign-flip test at the complete-PDB-group level."""
    mask = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
        & pd.notna(groups)
    )

    values = values[mask]
    weights = weights[mask]
    groups = groups[mask]

    if len(values) == 0:
        return math.nan

    contributions = (
        pd.DataFrame(
            {
                "group": groups.astype(str),
                "value": values * weights,
            }
        )
        .groupby("group", sort=False)["value"]
        .sum()
        .to_numpy(dtype=float)
    )

    if len(contributions) == 0 or np.allclose(
        contributions,
        0,
    ):
        return math.nan

    observed = abs(float(contributions.sum()))
    extreme = 0

    for start in range(0, iterations, 1000):
        count = min(1000, iterations - start)
        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(count, len(contributions)),
        )
        permuted = np.abs(signs @ contributions)
        extreme += int(
            np.count_nonzero(
                permuted >= observed - 1e-15
            )
        )

    return float(
        (extreme + 1)
        / (iterations + 1)
    )


def benjamini_hochberg(
    p_values: pd.Series,
) -> pd.Series:
    """BH-FDR adjustment while preserving missing values."""
    p = pd.to_numeric(
        p_values,
        errors="coerce",
    ).to_numpy(dtype=float)

    q = np.full(
        len(p),
        np.nan,
        dtype=float,
    )

    valid = np.isfinite(p)
    if not valid.any():
        return pd.Series(
            q,
            index=p_values.index,
            dtype=float,
        )

    valid_indices = np.flatnonzero(valid)
    valid_p = p[valid]
    order = np.argsort(
        valid_p,
        kind="mergesort",
    )
    ranked_p = valid_p[order]
    m = len(ranked_p)

    adjusted = (
        ranked_p
        * m
        / np.arange(
            1,
            m + 1,
            dtype=float,
        )
    )
    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    q[valid_indices[order]] = np.clip(
        adjusted,
        0.0,
        1.0,
    )

    return pd.Series(
        q,
        index=p_values.index,
        dtype=float,
    )


def consume_finalized_sensitivity_rng_stream(
    paired: pd.DataFrame,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Preserve the exact random stream of the finalized analysis.

    The finalized script used to generate the manuscript tables also computed
    two record-level bootstrap sensitivity estimates after every primary
    comparison. Those sensitivity tables are not part of the publication
    output retained here, but the two bootstrap calls are deliberately executed
    and discarded so that the same seed reproduces the published Monte Carlo
    confidence intervals and p values for subsequent comparisons.
    """
    record_delta = (
        paired["mean_record_absolute_error_comparison"]
        - paired["mean_record_absolute_error_reference"]
    ).to_numpy(dtype=float)

    equal_weights = np.ones(
        len(paired),
        dtype=float,
    )
    record_weights = paired[
        "duplicate_rows_reference"
    ].to_numpy(dtype=float)

    complete_group_bootstrap_ci(
        values=record_delta,
        weights=equal_weights,
        groups=groups,
        iterations=BOOTSTRAP_ITERATIONS,
        rng=rng,
    )

    complete_group_bootstrap_ci(
        values=record_delta,
        weights=record_weights,
        groups=groups,
        iterations=BOOTSTRAP_ITERATIONS,
        rng=rng,
    )


def pair_one_comparison(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    true_tolerance: float,
) -> pd.DataFrame:
    merged = reference.merge(
        comparison,
        on="ID",
        how="inner",
        validate="one_to_one",
        suffixes=("_reference", "_comparison"),
    )

    reference_ids = set(reference["ID"])
    comparison_ids = set(comparison["ID"])
    if reference_ids != comparison_ids:
        missing = sorted(reference_ids - comparison_ids)[:20]
        extra = sorted(comparison_ids - reference_ids)[:20]
        raise RuntimeError(
            "Reference and comparison splits do not contain the same unique "
            f"target IDs, missing={missing}, extra={extra}."
        )

    true_difference = (
        merged["ddG_true_comparison"]
        - merged["ddG_true_reference"]
    )

    mismatch = true_difference.abs() > true_tolerance
    if mismatch.any():
        bad = merged.loc[mismatch, "ID"].head(20).tolist()
        raise RuntimeError(
            "Paired target true values differ across splits: "
            f"{bad}"
        )

    group_mismatch = (
        merged["complete_pdb_group_reference"].astype(str)
        != merged["complete_pdb_group_comparison"].astype(str)
    )
    if group_mismatch.any():
        bad = merged.loc[group_mismatch, "ID"].head(20).tolist()
        raise RuntimeError(
            "Paired targets map to different complete PDB groups: "
            f"{bad}"
        )

    multiplicity_mismatch = (
        merged["duplicate_rows_reference"]
        != merged["duplicate_rows_comparison"]
    )
    if multiplicity_mismatch.any():
        bad = merged.loc[
            multiplicity_mismatch,
            "ID",
        ].head(20).tolist()
        raise RuntimeError(
            "Repeated-target multiplicity differs across splits: "
            f"{bad}"
        )

    merged["complete_pdb_group"] = (
        merged["complete_pdb_group_reference"].astype(str)
    )
    merged["absolute_error_reference"] = (
        merged["ddG_pred_reference"]
        - merged["ddG_true_reference"]
    ).abs()
    merged["absolute_error_comparison"] = (
        merged["ddG_pred_comparison"]
        - merged["ddG_true_comparison"]
    ).abs()
    merged["delta_AE"] = (
        merged["absolute_error_comparison"]
        - merged["absolute_error_reference"]
    )

    return merged


def build_paired_summary(
    work: pd.DataFrame,
) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    rng = np.random.default_rng(RANDOM_SEED)

    # Preserve the iteration order of the finalized analysis:
    # model first, then dataset, then PCSV and ICSV.
    for (model, dataset), group in work.groupby(
        ["model", "dataset"],
        sort=True,
        dropna=False,
    ):
        reference_raw = group[
            group["split"].eq(REFERENCE_SPLIT)
        ]
        if reference_raw.empty:
            raise RuntimeError(
                f"Missing {REFERENCE_SPLIT} rows for {model}/{dataset}."
            )

        reference = aggregate_unique_targets(
            reference_raw,
            TRUE_TOLERANCE,
        )

        for comparison_split in COMPARISON_SPLITS:
            comparison_raw = group[
                group["split"].eq(comparison_split)
            ]
            if comparison_raw.empty:
                raise RuntimeError(
                    f"Missing {comparison_split} rows for {model}/{dataset}."
                )

            comparison = aggregate_unique_targets(
                comparison_raw,
                TRUE_TOLERANCE,
            )

            paired = pair_one_comparison(
                reference,
                comparison,
                TRUE_TOLERANCE,
            )

            delta = paired["delta_AE"].to_numpy(
                dtype=float,
            )
            equal_weights = np.ones(
                len(paired),
                dtype=float,
            )
            groups = paired[
                "complete_pdb_group"
            ].astype(str).to_numpy()

            ci_low, ci_high = complete_group_bootstrap_ci(
                values=delta,
                weights=equal_weights,
                groups=groups,
                iterations=BOOTSTRAP_ITERATIONS,
                rng=rng,
            )

            p_value = complete_group_sign_flip_p(
                values=delta,
                weights=equal_weights,
                groups=groups,
                iterations=SIGN_FLIP_ITERATIONS,
                rng=rng,
            )

            summaries.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "comparison_split": comparison_split,
                    "n": int(len(paired)),
                    "delta_AE": weighted_mean(
                        delta,
                        equal_weights,
                    ),
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                    "p": p_value,
                }
            )

            # See function docstring. No sensitivity output is retained.
            consume_finalized_sensitivity_rng_stream(
                paired=paired,
                groups=groups,
                rng=rng,
            )

    summary = pd.DataFrame(summaries)

    if len(summary) != EXPECTED_BH_FAMILY_SIZE:
        models = sorted(work["model"].unique())
        raise RuntimeError(
            "The publication BH family must contain exactly "
            f"{EXPECTED_BH_FAMILY_SIZE} comparisons, but {len(summary)} "
            f"were generated from {len(models)} models."
        )

    if summary["p"].notna().sum() != EXPECTED_BH_FAMILY_SIZE:
        raise RuntimeError(
            "At least one primary sign-flip p value is unavailable; "
            "the complete 36-comparison BH family cannot be reproduced."
        )

    summary["q"] = benjamini_hochberg(
        summary["p"]
    )
    summary["significant"] = np.where(
        summary["q"] < FDR_ALPHA,
        "Yes",
        "No",
    )

    dataset_order = {
        dataset: index
        for index, dataset in enumerate(DATASETS)
    }
    split_order = {
        split: index
        for index, split in enumerate(COMPARISON_SPLITS)
    }

    summary["_dataset_order"] = summary["dataset"].map(
        dataset_order
    )
    summary["_split_order"] = summary[
        "comparison_split"
    ].map(split_order)

    summary = (
        summary.sort_values(
            [
                "_dataset_order",
                "model",
                "_split_order",
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

    return summary[
        [
            "model",
            "dataset",
            "comparison_split",
            "n",
            "delta_AE",
            "ci_95_low",
            "ci_95_high",
            "p",
            "q",
            "significant",
        ]
    ]


def prepare_output_dir(
    output_dir: Path,
    predictions: Path,
    overwrite: bool,
) -> None:
    output = output_dir.resolve()
    input_file = predictions.resolve()

    if output == input_file.parent or input_file.parent in output.parents:
        raise RuntimeError(
            "Output directory must be outside the directory containing the "
            "input all_oof_predictions.csv."
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = output / "paired_error_summary.csv"
    if result_path.exists() and not overwrite:
        raise FileExistsError(
            f"{result_path} already exists. Use --overwrite to replace it."
        )

    if result_path.exists() and overwrite:
        result_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce manuscript same-target delta-AE analyses using "
            "complete-PDB-group bootstrap, group sign-flip inference, and "
            "36-comparison BH correction."
        )
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
        help="all_oof_predictions.csv from analyze_performance_contraction.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace paired_error_summary.csv if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.predictions.is_file():
        raise FileNotFoundError(
            f"Prediction file not found: {args.predictions}"
        )

    prepare_output_dir(
        args.output_dir,
        args.predictions,
        args.overwrite,
    )

    print("========== paired_analysis.py ==========")
    print(f"Input:                  {args.predictions}")
    print(f"Reference split:        {REFERENCE_SPLIT}")
    print(f"Comparison splits:      {', '.join(COMPARISON_SPLITS)}")
    print(f"Bootstrap unit:         complete PDB group")
    print(f"Bootstrap iterations:   {BOOTSTRAP_ITERATIONS}")
    print(f"Sign-flip iterations:   {SIGN_FLIP_ITERATIONS}")
    print(f"BH family size:         {EXPECTED_BH_FAMILY_SIZE}")
    print(f"FDR alpha:              {FDR_ALPHA}")
    print(f"Random seed:            {RANDOM_SEED}")

    raw = read_csv_robust(
        args.predictions
    )
    work = validate_input(raw)

    models = sorted(work["model"].unique())
    if len(models) != 9:
        raise RuntimeError(
            f"The publication analysis expects 9 models, found {len(models)}: "
            f"{models}"
        )

    summary = build_paired_summary(work)

    result_path = (
        args.output_dir
        / "paired_error_summary.csv"
    )
    summary.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[RESULT]")
    print(f"Comparisons:                     {len(summary)}")
    print(
        "Mean delta_AE > 0:              "
        f"{int((summary['delta_AE'] > 0).sum())}/{len(summary)}"
    )
    print(
        "95% CI entirely above zero:    "
        f"{int((summary['ci_95_low'] > 0).sum())}/{len(summary)}"
    )
    print(
        f"BH q < {FDR_ALPHA:g}:                    "
        f"{int((summary['q'] < FDR_ALPHA).sum())}/{len(summary)}"
    )

    counts = (
        summary.groupby("dataset")["n"]
        .unique()
        .to_dict()
    )
    print(f"Paired target counts:            {counts}")

    print("\n[WRITE]")
    print(f"  {result_path}")
    print("[DONE] Paired analysis completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
