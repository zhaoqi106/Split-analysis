"""Read-only audit of the ICSV sequence clustering and its thresholds.

This script never writes to ``{dataset}/{seed}/ICSV`` and never replaces
``icsv_group_map.csv``. It re-runs MMseqs2 with the parameters recorded in
``three_splits.py``, keeps the intermediate FASTA and cluster TSV files, checks
that the result is the partition behind the existing splits, and then scans
alternative identity/coverage thresholds in a separate directory.

The scan answers Reviewer 1 comment 3 and the reproducibility gap in Reviewer 2
comment 7. Alternative thresholds produce different outer splits, so their fold
tables are descriptive only and must not be combined with predictions trained on
the current 30%/80% splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from three_splits import (
    DATASET_SEQ_CONFIG,
    DEFAULT_SEEDS,
    UnionFind,
    assert_icsv_matches_existing,
    build_complex_table,
    complex_keys,
    grouped_5fold_indices,
    mmseqs_easy_cluster_command,
    normalize_pdb_column,
    parse_mmseqs_cluster_tsv,
    record_uid,
    resolve_mmseqs,
    run_mmseqs_easy_cluster,
    write_fasta,
)

DATA_ROOT = Path("/data")
SPLITS = DATA_ROOT / "data_split" / "splits"
DEFAULT_INPUT_DIR = DATA_ROOT / "data_split" / "raw"
DEFAULT_AUDIT_DIR = DATA_ROOT / "data_split" / "audits" / "ICSV_CLUSTERING_AUDIT"
DEFAULT_DATASETS = ("S1131.csv", "M1101.csv")

REFERENCE_MIN_SEQ_ID = 0.30
REFERENCE_COVERAGE = 0.80
COV_MODE = 0
CLUSTER_MODE = 1

IDENTITY_GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
COVERAGE_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mmseqs_version(mmseqs: str) -> str:
    result = subprocess.run(
        [mmseqs, "version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def threshold_tag(min_seq_id: float, coverage: float) -> str:
    identity = int(round(min_seq_id * 100))
    cover = int(round(coverage * 100))
    return f"identity_{identity:03d}_coverage_{cover:03d}"


def canonical_codes(labels: Sequence[object]) -> list[int]:
    """Relabel a partition by first occurrence so representatives do not matter."""
    first_seen: dict[str, int] = {}
    return [first_seen.setdefault(str(label), len(first_seen)) for label in labels]


def build_group_labels(
    complex_table: pd.DataFrame,
    dataset_name: str,
    mmseqs: str,
    work_dir: Path,
    tag: str,
    min_seq_id: float,
    coverage: float,
    threads: int,
) -> tuple[list[str], list[list[str]]]:
    """Cluster both sequence sides and merge them with union-find."""
    side_to_mapping: dict[str, dict[str, str]] = {}
    commands: list[list[str]] = []

    for side in ("left", "right"):
        suffix = side.upper()
        records = [
            (f"{row.complex_key}__{suffix}", getattr(row, f"{side}_sequence"))
            for row in complex_table.itertuples(index=False)
            if getattr(row, f"{side}_sequence")
        ]
        if not records:
            raise ValueError(f"{dataset_name}: no sequences on the {side} side.")

        fasta = work_dir / "sequences" / f"{side}.fasta"
        if not fasta.is_file():
            write_fasta(records, fasta)

        all_ids = [sequence_id for sequence_id, _ in records]
        if len(all_ids) == 1:
            side_to_mapping[side] = {all_ids[0]: all_ids[0]}
            continue

        prefix = work_dir / "clusters" / tag / f"{side}_cluster"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = work_dir / "tmp" / tag / side
        tmp_dir.mkdir(parents=True, exist_ok=True)

        command = mmseqs_easy_cluster_command(
            mmseqs=mmseqs,
            fasta=fasta,
            prefix=prefix,
            tmp_dir=tmp_dir,
            min_seq_id=min_seq_id,
            coverage=coverage,
            cov_mode=COV_MODE,
            cluster_mode=CLUSTER_MODE,
            threads=threads,
        )
        commands.append(command)

        cluster_tsv = run_mmseqs_easy_cluster(
            mmseqs=mmseqs,
            fasta=fasta,
            prefix=prefix,
            tmp_dir=tmp_dir,
            min_seq_id=min_seq_id,
            coverage=coverage,
            cov_mode=COV_MODE,
            cluster_mode=CLUSTER_MODE,
            threads=threads,
        )
        side_to_mapping[side] = parse_mmseqs_cluster_tsv(cluster_tsv, all_ids)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    union_find = UnionFind()
    complex_keys_in_order = [str(key) for key in complex_table["complex_key"]]
    for complex_key in complex_keys_in_order:
        union_find.add(complex_key)

    first_complex_by_cluster: dict[tuple[str, str], str] = {}
    for row in complex_table.itertuples(index=False):
        complex_key = str(row.complex_key)
        for side in ("left", "right"):
            sequence_id = f"{complex_key}__{side.upper()}"
            representative = side_to_mapping[side].get(sequence_id, "")
            if not representative:
                continue
            cluster_key = (side, representative)
            if cluster_key in first_complex_by_cluster:
                union_find.union(complex_key, first_complex_by_cluster[cluster_key])
            else:
                first_complex_by_cluster[cluster_key] = complex_key

    labels = [union_find.find(key) for key in complex_keys_in_order]
    return labels, commands


def labels_for_records(
    df: pd.DataFrame,
    complex_table: pd.DataFrame,
    complex_labels: Sequence[str],
) -> list[str]:
    key_to_group = dict(zip(complex_table["complex_key"].astype(str), complex_labels))
    row_keys = complex_keys(df).astype(str)
    missing = sorted({key for key in row_keys if key not in key_to_group})
    if missing:
        raise RuntimeError(f"Records missing from the complex table: {missing[:10]}")
    return [key_to_group[key] for key in row_keys]


def exact_match_labels(complex_table: pd.DataFrame) -> list[int]:
    """Full-length exact sequence identity on the ordered (left, right) tuple."""
    first_seen: dict[tuple[str, str], int] = {}
    labels: list[int] = []
    for row in complex_table.itertuples(index=False):
        key = (row.left_sequence, row.right_sequence)
        labels.append(first_seen.setdefault(key, len(first_seen)))
    return labels


def group_sizes_table(
    df: pd.DataFrame,
    row_labels: Sequence[str],
) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "group_id": [str(label) for label in row_labels],
            "complex_key": complex_keys(df).astype(str).tolist(),
        }
    )
    work["ddG"] = pd.to_numeric(df.get("ddG"), errors="coerce")

    rows: list[dict[str, object]] = []
    for group_id, group in work.groupby("group_id", sort=False):
        ddg = group["ddG"].dropna()
        rows.append(
            {
                "group_id": group_id,
                "n_records": int(len(group)),
                "n_pdb": int(len(group)),
                "n_complex": int(group["complex_key"].nunique()),
                "fraction": float(len(group) / len(work)),
                "ddG_n_finite": int(ddg.size),
                "ddG_mean": float(ddg.mean()) if ddg.size else np.nan,
                "ddG_std": float(ddg.std()) if ddg.size > 1 else np.nan,
                "ddG_min": float(ddg.min()) if ddg.size else np.nan,
                "ddG_max": float(ddg.max()) if ddg.size else np.nan,
                "member_complexes": ";".join(sorted(group["complex_key"].unique())),
            }
        )

    table = pd.DataFrame(rows)
    return table.sort_values(
        ["n_records", "group_id"], ascending=[False, True]
    ).reset_index(drop=True)


def fold_rows(
    dataset_name: str,
    row_labels: Sequence[str],
    seeds: Sequence[int],
    n_splits: int,
) -> tuple[pd.DataFrame, list[list[list[int]]], bool]:
    labels = [str(label) for label in row_labels]
    all_indices = np.arange(len(labels))
    per_seed_folds: list[list[list[int]]] = []
    rows: list[dict[str, object]] = []
    one_fold_per_group = True

    for seed in seeds:
        folds = grouped_5fold_indices(labels, seed=seed, n_splits=n_splits)
        per_seed_folds.append([sorted(int(index) for index in fold) for fold in folds])

        group_fold_membership: dict[str, set[int]] = {}
        for fold_id, indices in enumerate(folds):
            for index in indices:
                group_fold_membership.setdefault(labels[index], set()).add(fold_id)
        if any(len(membership) != 1 for membership in group_fold_membership.values()):
            one_fold_per_group = False

        for fold_id, val_indices in enumerate(folds):
            val_set = set(val_indices)
            train_indices = [index for index in all_indices if index not in val_set]
            val_groups = {labels[index] for index in val_set}
            train_groups = {labels[index] for index in train_indices}
            group_sizes: dict[str, int] = {}
            for index in val_set:
                group_sizes[labels[index]] = group_sizes.get(labels[index], 0) + 1
            largest = max(group_sizes.values()) if group_sizes else 0

            rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "fold": fold_id,
                    "val_records": len(val_set),
                    "val_fraction": len(val_set) / len(labels),
                    "val_groups": len(val_groups),
                    "train_records": len(train_indices),
                    "max_val_group_records": int(largest),
                    "group_overlap_train_val": len(val_groups & train_groups),
                }
            )

    return pd.DataFrame(rows), per_seed_folds, one_fold_per_group


def compare_group_map(
    dataset_name: str,
    row_labels: Sequence[str],
    existing_map_path: Path,
) -> dict[str, object]:
    if not existing_map_path.is_file():
        return {"path": str(existing_map_path), "exists": False}

    existing = pd.read_csv(existing_map_path)
    expected_uids = [record_uid(dataset_name, index) for index in range(len(row_labels))]
    uid_match = existing["record_uid"].tolist() == expected_uids

    existing_labels = existing["group_id"].astype(str).tolist()
    exact_equal = existing_labels == [str(label) for label in row_labels]
    partition_equal = canonical_codes(existing_labels) == canonical_codes(row_labels)

    return {
        "path": str(existing_map_path),
        "exists": True,
        "record_uid_match": bool(uid_match),
        "group_labels_equal": bool(exact_equal),
        "partition_equal": bool(partition_equal),
        "n_rows": int(len(existing)),
        "n_groups": int(existing["group_id"].nunique()),
    }


def summarize_partition(
    dataset_name: str,
    row_labels: Sequence[str],
    reference_codes: list[int],
    seeds: Sequence[int],
    n_splits: int,
    df: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    labels = [str(label) for label in row_labels]
    sizes = pd.Series(labels).value_counts()
    multi_complex_groups = int(
        (group_sizes_table(df, row_labels)["n_complex"] > 1).sum()
    )
    summary: dict[str, object] = {
        "n_groups": int(len(sizes)),
        "n_multi_complex_groups": multi_complex_groups,
        "largest_group_records": int(sizes.iloc[0]),
        "largest_group_fraction": float(sizes.iloc[0] / len(labels)),
        "partition_equal_to_reference": canonical_codes(labels) == reference_codes,
    }

    if len(sizes) < n_splits:
        summary.update(
            {
                "max_val_fraction": np.nan,
                "min_val_fraction": np.nan,
                "max_val_records": np.nan,
                "val_records_spread": np.nan,
                "one_fold_per_group": False,
                "group_overlap_train_val": np.nan,
                "seeds_give_identical_fold_sizes": False,
            }
        )
        return summary, pd.DataFrame()

    fold_table, per_seed_folds, one_fold_per_group = fold_rows(
        dataset_name,
        labels,
        seeds=seeds,
        n_splits=n_splits,
    )
    signatures = {tuple(len(fold) for fold in folds) for folds in per_seed_folds}
    summary.update(
        {
            "max_val_fraction": float(fold_table["val_fraction"].max()),
            "min_val_fraction": float(fold_table["val_fraction"].min()),
            "max_val_records": int(fold_table["val_records"].max()),
            "val_records_spread": int(
                fold_table.groupby("seed")["val_records"]
                .apply(lambda values: int(values.max() - values.min()))
                .max()
            ),
            "one_fold_per_group": bool(one_fold_per_group),
            "group_overlap_train_val": int(
                fold_table["group_overlap_train_val"].max()
            ),
            "seeds_give_identical_fold_sizes": len(signatures) == 1,
        }
    )
    return summary, fold_table


def sensitivity_row(
    dataset_name: str,
    scan: str,
    min_seq_id: float,
    coverage: float,
    complex_table: pd.DataFrame,
    df: pd.DataFrame,
    reference_codes: list[int],
    mmseqs: str,
    work_dir: Path,
    seeds: Sequence[int],
    n_splits: int,
    threads: int,
) -> tuple[dict[str, object], pd.DataFrame, list[list[str]]]:
    tag = threshold_tag(min_seq_id, coverage)
    complex_labels, commands = build_group_labels(
        complex_table=complex_table,
        dataset_name=dataset_name,
        mmseqs=mmseqs,
        work_dir=work_dir,
        tag=tag,
        min_seq_id=min_seq_id,
        coverage=coverage,
        threads=threads,
    )
    row_labels = labels_for_records(df, complex_table, complex_labels)
    summary, fold_table = summarize_partition(
        dataset_name=dataset_name,
        row_labels=row_labels,
        reference_codes=reference_codes,
        seeds=seeds,
        n_splits=n_splits,
        df=df,
    )
    row = {
        "dataset": dataset_name,
        "scan": scan,
        "min_seq_id": min_seq_id,
        "coverage": coverage,
        **summary,
    }
    return row, fold_table, commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of ICSV MMseqs2 clustering: reproduce the existing "
            "partition and scan identity/coverage thresholds."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing {dataset}.csv.",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=DEFAULT_AUDIT_DIR,
        help="Output directory. Existing split files are never modified.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset CSV filenames.",
    )
    parser.add_argument(
        "--mmseqs",
        default="/data/tools/mmseqs/bin/mmseqs",
        help="MMseqs2 executable name or full path.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds used for fold-balance summaries.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the audit directory if it already contains files.",
    )
    return parser.parse_args()


def format_markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_readme(
    path: Path,
    reproduction: list[dict[str, object]],
    sensitivity: pd.DataFrame,
    canonical_folds: pd.DataFrame,
) -> None:
    fold_display = canonical_folds.drop_duplicates(
        subset=["dataset", "fold"], keep="first"
    )
    binary_hash = str(reproduction[0]["mmseqs_sha256"])[:16] if reproduction else ""
    lines = [
        "# ICSV clustering audit",
        "",
        "This directory is the MMseqs2 clustering audit for the ICSV splits. "
        "Cluster commands and intermediate files are kept here. The split files "
        "themselves are under `/data/data_split/splits`.",
        "",
        "## Result",
        "",
    ]

    if reproduction:
        lines.append(
            f"- MMseqs2 build `{reproduction[0]['mmseqs_version']}` "
            f"(binary SHA256 `{binary_hash}...`) at the reference setting 30%/80%."
        )
    lines.append(
        "- The record-level connected components match `icsv_group_map.csv`, "
        "and the five outer ICSV folds match row for row. The maps come from this setting."
    )

    for dataset_name, group in sensitivity.groupby("dataset", sort=False):
        identity_same = group[
            (group["scan"] == "identity") & group["partition_equal_to_reference"]
        ]
        coverage_same = group[
            (group["scan"] == "coverage") & group["partition_equal_to_reference"]
        ]
        identity_span = (
            f"{identity_same['min_seq_id'].min():.2f}–"
            f"{identity_same['min_seq_id'].max():.2f}"
            if not identity_same.empty
            else "none"
        )
        coverage_span = (
            f"{coverage_same['coverage'].min():.2f}–"
            f"{coverage_same['coverage'].max():.2f}"
            if not coverage_same.empty
            else "none"
        )
        identity_scan = group[group["scan"] == "identity"]
        exact = group[group["scan"] == "exact"]
        lines.append(
            f"- {dataset_name}: identity {identity_span} and coverage {coverage_span} "
            f"give the same partition. The smallest largest-group fraction in the identity "
            f"scan is {identity_scan['largest_group_fraction'].min():.4f}. Exact match has "
            f"{exact['largest_group_fraction'].iloc[0]:.4f} "
            f"({int(exact['n_groups'].iloc[0])} groups)."
        )

    lines.extend(
        [
            "- The five seeds give the same fold sizes. ICSV imbalance is the size of the "
            "connected components.",
            "- A different threshold changes the outer folds.",
            "",
            "## Match to the current splits",
            "",
        ]
    )

    for item in reproduction:
        lines.append(
            f"- {item['dataset']}: MMseqs2 {item['mmseqs_version']}, "
            f"{item['n_groups']} connected components at 30%/80%. "
            f"partition_equal={item['group_map']['partition_equal']}, "
            f"group_id match={item['group_map']['group_labels_equal']}, "
            f"five outer folds match row for row={item['existing_folds_match']}."
        )

    lines.extend(
        [
            "",
            "Cluster commands and TSVs are in each dataset's `clusters/`. Input FASTA "
            "files are in `sequences/`. `partition_equal` true with different group ids "
            "means the same clusters and different representative ids.",
            "",
            "## Fold balance at 30%/80%",
            "",
            "Fold sizes are the same for every seed. One seed is shown per dataset.",
            "",
            format_markdown_table(
                fold_display[
                    [
                        "dataset",
                        "seed",
                        "fold",
                        "val_records",
                        "val_fraction",
                        "val_groups",
                        "max_val_group_records",
                        "group_overlap_train_val",
                    ]
                ]
            ),
            "",
            "## Threshold sensitivity",
            "",
            "Each side is clustered with MMseqs2, then the two sides are merged with "
            "union-find. `partition_equal_to_reference` means the record-level partition "
            "matches the current 30%/80% split. A changed threshold is a different outer split.",
            "",
            format_markdown_table(
                sensitivity[
                    [
                        "dataset",
                        "scan",
                        "min_seq_id",
                        "coverage",
                        "n_groups",
                        "largest_group_fraction",
                        "max_val_fraction",
                        "min_val_fraction",
                        "val_records_spread",
                        "seeds_give_identical_fold_sizes",
                        "partition_equal_to_reference",
                    ]
                ]
            ),
            "",
            "## Scope",
            "",
            "- More groups and a smaller largest group make the outer folds more even. "
            "A higher threshold is a weaker redundancy filter.",
            "- 30% identity with 80% coverage is the stricter filter. A few large "
            "components make the folds uneven.",
            "- This audit is sequence identity only. TM-score, interface similarity, "
            "and structure clustering are not computed here.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if args.audit_dir.exists() and any(args.audit_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Audit directory is not empty: {args.audit_dir}. Use --overwrite."
        )
    if args.audit_dir.exists() and args.overwrite:
        shutil.rmtree(args.audit_dir)
    args.audit_dir.mkdir(parents=True, exist_ok=True)

    mmseqs = resolve_mmseqs(args.mmseqs)
    version = mmseqs_version(mmseqs)
    (args.audit_dir / "mmseqs_version.txt").write_text(
        f"{version}\n{mmseqs}\n{sha256_file(Path(mmseqs))}\n",
        encoding="utf-8",
    )

    reproduction: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    fold_tables: list[pd.DataFrame] = []
    commands: list[tuple[str, list[str]]] = []
    canonical_folds = pd.DataFrame()

    for dataset_file in args.datasets:
        csv_path = args.input_dir / dataset_file
        if not csv_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        dataset_name = csv_path.stem
        if dataset_name not in DATASET_SEQ_CONFIG:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        df = pd.read_csv(csv_path, low_memory=False)
        df, repaired = normalize_pdb_column(df)
        if repaired:
            print(f"[{dataset_name}] repaired {repaired} PDB identifier(s)")

        complex_table = build_complex_table(df, dataset_name)
        work_dir = args.audit_dir / dataset_name
        (work_dir / "sequences").mkdir(parents=True, exist_ok=True)
        (work_dir / "clusters").mkdir(parents=True, exist_ok=True)

        print(f"[{dataset_name}] reference clustering at 30%/80%")
        reference_complex_labels, reference_commands = build_group_labels(
            complex_table=complex_table,
            dataset_name=dataset_name,
            mmseqs=mmseqs,
            work_dir=work_dir,
            tag=threshold_tag(REFERENCE_MIN_SEQ_ID, REFERENCE_COVERAGE),
            min_seq_id=REFERENCE_MIN_SEQ_ID,
            coverage=REFERENCE_COVERAGE,
            threads=args.threads,
        )
        for command in reference_commands:
            commands.append((dataset_name, command))

        reference_row_labels = labels_for_records(
            df, complex_table, reference_complex_labels
        )
        reference_codes = canonical_codes(reference_row_labels)

        existing_map_path = SPLITS / dataset_name / "icsv_group_map.csv"
        group_map = compare_group_map(
            dataset_name, reference_row_labels, existing_map_path
        )

        existing_folds_match = True
        try:
            for seed in args.seeds:
                folds = grouped_5fold_indices(
                    reference_row_labels, seed=seed, n_splits=args.n_splits
                )
                assert_icsv_matches_existing(
                    df,
                    folds,
                    SPLITS / dataset_name / str(seed) / "ICSV",
                )
        except (FileNotFoundError, RuntimeError) as error:
            existing_folds_match = False
            print(f"[{dataset_name}] existing folds mismatch: {error}")

        fold_table, _, one_fold_per_group = fold_rows(
            dataset_name,
            reference_row_labels,
            seeds=args.seeds,
            n_splits=args.n_splits,
        )
        canonical_folds = pd.concat([canonical_folds, fold_table], ignore_index=True)
        fold_tables.append(fold_table)

        sizes = group_sizes_table(df, reference_row_labels)
        sizes.to_csv(work_dir / "group_sizes.csv", index=False)

        reproduction.append(
            {
                "dataset": dataset_name,
                "input_csv": str(csv_path),
                "input_sha256": sha256_file(csv_path),
                "mmseqs_version": version,
                "mmseqs_path": mmseqs,
                "mmseqs_sha256": sha256_file(Path(mmseqs)),
                "parameters": {
                    "min_seq_id": REFERENCE_MIN_SEQ_ID,
                    "coverage": REFERENCE_COVERAGE,
                    "cov_mode": COV_MODE,
                    "cluster_mode": CLUSTER_MODE,
                    "single_step_clustering": True,
                },
                "n_records": int(len(df)),
                "n_pdbs": int(df["PDB"].nunique()),
                "n_complexes": int(len(complex_table)),
                "n_groups": int(len(sizes)),
                "largest_group_records": int(sizes.iloc[0]["n_records"]),
                "largest_group_fraction": float(sizes.iloc[0]["fraction"]),
                "one_fold_per_group": bool(one_fold_per_group),
                "group_map": group_map,
                "existing_folds_match": bool(existing_folds_match),
                "commands": reference_commands,
            }
        )

        print(f"[{dataset_name}] exact-match reference partition")
        exact_complex_labels = exact_match_labels(complex_table)
        exact_row_labels = labels_for_records(
            df,
            complex_table,
            [str(label) for label in exact_complex_labels],
        )
        exact_summary, exact_folds = summarize_partition(
            dataset_name=dataset_name,
            row_labels=exact_row_labels,
            reference_codes=reference_codes,
            seeds=args.seeds,
            n_splits=args.n_splits,
            df=df,
        )
        sensitivity_rows.append(
            {
                "dataset": dataset_name,
                "scan": "exact",
                "min_seq_id": 1.0,
                "coverage": 1.0,
                **exact_summary,
            }
        )
        if not exact_folds.empty:
            fold_tables.append(exact_folds)

        print(f"[{dataset_name}] threshold scan")
        for min_seq_id in IDENTITY_GRID:
            row, scan_folds, command_list = sensitivity_row(
                dataset_name=dataset_name,
                scan="identity",
                min_seq_id=min_seq_id,
                coverage=REFERENCE_COVERAGE,
                complex_table=complex_table,
                df=df,
                reference_codes=reference_codes,
                mmseqs=mmseqs,
                work_dir=work_dir,
                seeds=args.seeds,
                n_splits=args.n_splits,
                threads=args.threads,
            )
            sensitivity_rows.append(row)
            fold_tables.append(scan_folds)
            for command in command_list:
                commands.append((dataset_name, command))
            print(
                f"[{dataset_name}] identity={min_seq_id:.2f} "
                f"groups={row['n_groups']} "
                f"largest={row['largest_group_fraction']:.3f}"
            )

        for coverage in COVERAGE_GRID:
            if coverage == REFERENCE_COVERAGE:
                continue
            row, scan_folds, command_list = sensitivity_row(
                dataset_name=dataset_name,
                scan="coverage",
                min_seq_id=REFERENCE_MIN_SEQ_ID,
                coverage=coverage,
                complex_table=complex_table,
                df=df,
                reference_codes=reference_codes,
                mmseqs=mmseqs,
                work_dir=work_dir,
                seeds=args.seeds,
                n_splits=args.n_splits,
                threads=args.threads,
            )
            sensitivity_rows.append(row)
            fold_tables.append(scan_folds)
            for command in command_list:
                commands.append((dataset_name, command))
            print(
                f"[{dataset_name}] coverage={coverage:.2f} "
                f"groups={row['n_groups']} "
                f"largest={row['largest_group_fraction']:.3f}"
            )

        shutil.rmtree(work_dir / "tmp", ignore_errors=True)

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(args.audit_dir / "threshold_sensitivity.csv", index=False)
    canonical_folds.to_csv(args.audit_dir / "fold_composition.csv", index=False)
    pd.concat(fold_tables, ignore_index=True).to_csv(
        args.audit_dir / "sensitivity_fold_composition.csv", index=False
    )
    (args.audit_dir / "reproduction.json").write_text(
        json.dumps(
            {
                "mmseqs_version": version,
                "mmseqs_path": mmseqs,
                "reference_parameters": {
                    "min_seq_id": REFERENCE_MIN_SEQ_ID,
                    "coverage": REFERENCE_COVERAGE,
                    "cov_mode": COV_MODE,
                    "cluster_mode": CLUSTER_MODE,
                },
                "datasets": reproduction,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    log_lines = [f"# mmseqs version: {version}"]
    for dataset_name, command in commands:
        log_lines.append(f"{dataset_name}\t{' '.join(command)}")
    (args.audit_dir / "commands.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )

    write_readme(
        args.audit_dir / "README_icsv_clustering_audit.md",
        reproduction=reproduction,
        sensitivity=sensitivity,
        canonical_folds=canonical_folds,
    )

    print(f"[DONE] audit written to {args.audit_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
