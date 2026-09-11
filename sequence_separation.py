#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sequence_separation.py

Publication-oriented read-only audit of sequence separation in the final ICSV
folds for S1131 and M1101.

This script does not create or modify split files. It rebuilds the sequence
relationships used to audit the final ICSV partitions and writes only the
outputs that directly support the manuscript:

1. sequence_separation_summary.csv
   Dataset-level quantities reported or discussed in the manuscript, including
   the number of construction groups, threshold-passing sequence relationships,
   audited connected components, residual cross-fold relationships, and
   validation-record exposure.

2. sequence_relationships.csv
   All unique threshold-passing sequence relationships used to obtain the
   reported relationship counts and audited connected components.

3. cross_fold_sequence_relationships.csv
   The subset of threshold-passing relationships whose two complexes occur in
   different validation folds. These are the residual sequence relationships
   responsible for any nonzero ICSV sequence exposure.

No MMseqs2 FASTA files, cluster tables, search tables, logs, temporary files,
or development audit tables are retained.

Sequence definitions
--------------------
S1131 and M1101 share the same sequence columns.
The R/L prefixes are legacy storage labels; the audit interprets them only as
the left and right interaction sides and does not assign antibody or antigen
identity.

    left side, wild type  = R1-R6
    right side, wild type = L1-L3

Empty chain slots are skipped. Multiple wild-type chains on one interaction
side are joined by ten glycine residues, matching three_splits.py. ICSV
auditing uses only the wild-type sequences.

Primary sequence criteria
-------------------------
minimum sequence identity = 0.30
coverage                  = 0.80
coverage mode             = 0
cluster mode              = 1

The direct audit uses MMseqs2 easy-search with alignment mode 3 and sensitivity
7.5. Connected components are rebuilt from every direct relationship satisfying
the identity and coverage criteria.

PDB identifier normalization
----------------------------
Excel can convert a valid PDB identifier such as 1E96 to scientific notation
such as 1.00E+96. The script repairs that corruption before matching the master
dataset to the final fold files. Custom identifiers such as HM_1YY9 are
preserved.

Example
-------
python sequence_separation.py \
    --data-dir data \
    --split-root data \
    --output-dir output/sequence_separation \
    --datasets S1131 M1101 \
    --mmseqs mmseqs \
    --threads 16
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "sequence_separation"
DEFAULT_DATASETS = ("S1131", "M1101")

LINKER = "GGGGGGGGGG"

WT_LEFT_COLS = ("R1", "R2", "R3", "R4", "R5", "R6")
WT_RIGHT_COLS = ("L1", "L2", "L3")

SHARED_SEQ_CONFIG = {
    "left_cols": WT_LEFT_COLS,
    "right_cols": WT_RIGHT_COLS,
}

DATASET_SEQ_CONFIG = {
    "S1131": SHARED_SEQ_CONFIG,
    "M1101": SHARED_SEQ_CONFIG,
}

PAIR_FIELDS = (
    "query",
    "target",
    "fident",
    "qcov",
    "tcov",
    "alnlen",
    "qlen",
    "tlen",
    "evalue",
    "bits",
)

CORRUPTED_SCIENTIFIC_RE = re.compile(
    r"^(?:[0-9]+(?:\.[0-9]+)E\+?[0-9]+|[0-9]+E\+[0-9]+)$",
    re.IGNORECASE,
)
DIRECT_REPAIR_RE = re.compile(
    r"^([0-9])(?:\.0+)?E\+?([0-9]{2})$",
    re.IGNORECASE,
)
PDB_FROM_ID_RE = re.compile(
    r"^([0-9][A-Z0-9]{3})(?:[_:\-]|$)",
    re.IGNORECASE,
)


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\ufeff", "").strip()
    if text.casefold() in {"", "nan", "none", "null", "na"}:
        return ""
    return text


def clean_sequence(value: object) -> str:
    return re.sub(r"[^A-Za-z]", "", clean_text(value)).upper()


def resolve_id_column(df: pd.DataFrame) -> str:
    """Prefer the exact uppercase ID column when present."""
    if "ID" in df.columns:
        return "ID"

    for preferred in ("sample_id", "mutation_id", "mutant_id"):
        if preferred in df.columns:
            return preferred

    normalized: dict[str, list[str]] = defaultdict(list)
    for column in df.columns:
        key = re.sub(r"[^a-z0-9]", "", str(column).casefold())
        normalized[key].append(str(column))

    for alias in ("id", "sampleid", "mutationid", "mutantid"):
        matches = normalized.get(alias, [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous ID columns: {matches}")

    raise ValueError(f"No sample ID column found. Columns: {list(df.columns)}")


def normalize_pdb(value: object, fallback_id: object = "") -> str:
    """Normalize PDB IDs and repair Excel scientific-notation corruption."""
    raw = clean_text(value).upper()
    if not raw:
        return ""

    if CORRUPTED_SCIENTIFIC_RE.fullmatch(raw):
        fallback = clean_text(fallback_id).upper()
        match = PDB_FROM_ID_RE.match(fallback)
        if match:
            return match.group(1).upper()

        match = DIRECT_REPAIR_RE.fullmatch(raw)
        if match:
            return f"{match.group(1)}E{match.group(2)}".upper()

    return raw


def safe_id(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "-", clean_text(value))


def concat_sequence_cols(row: pd.Series, columns: Sequence[str]) -> str:
    sequences = []
    for column in columns:
        sequence = clean_sequence(row[column]) if column in row.index else ""
        if sequence:
            sequences.append(sequence)
    return LINKER.join(sequences)


def read_csv_text(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
                low_memory=False,
            )
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError(f"Cannot decode CSV {path}: {'; '.join(errors)}")


def add_normalized_keys(df: pd.DataFrame) -> pd.DataFrame:
    if "PDB" not in df.columns or "Partners" not in df.columns:
        raise ValueError("CSV must contain PDB and Partners columns.")

    result = df.copy()
    id_col = resolve_id_column(result)
    result["_PDB_norm"] = [
        normalize_pdb(pdb, fallback_id)
        for pdb, fallback_id in zip(result["PDB"], result[id_col])
    ]
    result["_complex_key"] = [
        f"{pdb}__{safe_id(partners)}"
        for pdb, partners in zip(result["_PDB_norm"], result["Partners"])
    ]

    if result["_PDB_norm"].eq("").any():
        rows = result.index[result["_PDB_norm"].eq("")].tolist()[:20]
        raise ValueError(f"Empty normalized PDB identifier at rows: {rows}")

    return result


def build_complex_table(master: pd.DataFrame, dataset: str) -> pd.DataFrame:
    config = DATASET_SEQ_CONFIG[dataset]
    required = [
        "PDB",
        "Partners",
        *config["left_cols"],
        *config["right_cols"],
    ]
    missing = [column for column in required if column not in master.columns]
    if missing:
        raise ValueError(f"{dataset} master CSV missing columns: {missing}")

    keyed = add_normalized_keys(master)
    rows: list[dict[str, object]] = []

    for complex_key, group in keyed.groupby("_complex_key", sort=False):
        first = group.iloc[0]

        for column in (*config["left_cols"], *config["right_cols"]):
            values = {
                clean_sequence(value)
                for value in group[column]
                if clean_sequence(value)
            }
            if len(values) > 1:
                raise ValueError(
                    f"{dataset}/{complex_key}: multiple wild-type sequences found "
                    f"in column {column!r}. Clean the master dataset before auditing."
                )

        left_sequence = concat_sequence_cols(first, config["left_cols"])
        right_sequence = concat_sequence_cols(first, config["right_cols"])

        if not left_sequence or not right_sequence:
            raise ValueError(
                f"{dataset}/{complex_key}: missing wild-type sequence on at least "
                "one interaction side."
            )

        rows.append(
            {
                "dataset": dataset,
                "complex_key": complex_key,
                "PDB": first["_PDB_norm"],
                "Partners": clean_text(first["Partners"]),
                "left_sequence": left_sequence,
                "right_sequence": right_sequence,
                "n_records": int(len(group)),
            }
        )

    return pd.DataFrame(rows)


def id_counter(df: pd.DataFrame, id_column: str) -> Counter:
    return Counter(
        clean_text(value).upper()
        for value in df[id_column]
        if clean_text(value)
    )


def counter_difference_count(first: Counter, second: Counter) -> int:
    return int(sum((first - second).values()))


def load_final_icsv_folds(
    dataset: str,
    master: pd.DataFrame,
    complex_table: pd.DataFrame,
    split_root: Path,
    split_name: str,
    folds: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Validate final ICSV files and return complex-to-validation-fold mapping."""
    split_dir = split_root / dataset / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    master_id_col = resolve_id_column(master)
    master_ids = id_counter(master, master_id_col)
    pooled_val_ids: Counter = Counter()

    master_complexes = set(complex_table["complex_key"])
    master_sequences = complex_table.set_index("complex_key")[
        ["left_sequence", "right_sequence"]
    ].to_dict("index")
    config = DATASET_SEQ_CONFIG[dataset]

    fold_records: list[dict[str, object]] = []
    validation_fold_by_complex: dict[str, int] = {}

    for fold in range(folds):
        train_path = split_dir / f"train_fold_{fold}.csv"
        val_path = split_dir / f"val_fold_{fold}.csv"

        if not train_path.is_file() or not val_path.is_file():
            missing = [
                str(path)
                for path in (train_path, val_path)
                if not path.is_file()
            ]
            raise FileNotFoundError(
                f"{dataset} fold {fold} missing split files: {missing}"
            )

        train = add_normalized_keys(read_csv_text(train_path))
        val = add_normalized_keys(read_csv_text(val_path))

        for role, frame in (("train", train), ("val", val)):
            missing_cols = [
                column
                for column in (
                    *config["left_cols"],
                    *config["right_cols"],
                )
                if column not in frame.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"{dataset} fold {fold} {role} missing sequence columns: "
                    f"{missing_cols}"
                )

        train_id_col = resolve_id_column(train)
        val_id_col = resolve_id_column(val)
        train_ids = id_counter(train, train_id_col)
        val_ids = id_counter(val, val_id_col)
        pooled_val_ids.update(val_ids)

        combined = train_ids + val_ids
        missing_from_fold = counter_difference_count(master_ids, combined)
        extra_in_fold = counter_difference_count(combined, master_ids)
        if missing_from_fold or extra_in_fold:
            raise RuntimeError(
                f"{dataset} fold {fold}: train+validation does not reproduce the "
                f"master ID multiset, missing={missing_from_fold}, extra={extra_in_fold}."
            )

        train_complexes = set(train["_complex_key"])
        val_complexes = set(val["_complex_key"])

        overlap = train_complexes & val_complexes
        if overlap:
            raise RuntimeError(
                f"{dataset} fold {fold}: {len(overlap)} complete complexes occur "
                "in both train and validation."
            )

        unknown = (train_complexes | val_complexes) - master_complexes
        if unknown:
            raise RuntimeError(
                f"{dataset} fold {fold}: complexes absent from master dataset: "
                f"{sorted(unknown)[:20]}"
            )

        for role, frame in (("train", train), ("val", val)):
            for complex_key, group in frame.groupby("_complex_key", sort=False):
                first = group.iloc[0]
                observed_left = concat_sequence_cols(first, config["left_cols"])
                observed_right = concat_sequence_cols(first, config["right_cols"])
                expected = master_sequences[complex_key]
                if observed_left != expected["left_sequence"]:
                    raise RuntimeError(
                        f"{dataset} fold {fold} {role}: left-side WT sequence "
                        f"mismatch for {complex_key}."
                    )
                if observed_right != expected["right_sequence"]:
                    raise RuntimeError(
                        f"{dataset} fold {fold} {role}: right-side WT sequence "
                        f"mismatch for {complex_key}."
                    )

        for complex_key in val_complexes:
            if complex_key in validation_fold_by_complex:
                raise RuntimeError(
                    f"{dataset}: complex {complex_key} occurs in multiple "
                    "validation folds."
                )
            validation_fold_by_complex[complex_key] = fold

        fold_records.append(
            {
                "fold": fold,
                "train_records": int(len(train)),
                "val_records": int(len(val)),
                "train_complexes": int(len(train_complexes)),
                "val_complexes": int(len(val_complexes)),
            }
        )

    pooled_missing = counter_difference_count(master_ids, pooled_val_ids)
    pooled_extra = counter_difference_count(pooled_val_ids, master_ids)
    if pooled_missing or pooled_extra:
        raise RuntimeError(
            f"{dataset}: pooled validation folds do not reproduce the master ID "
            f"multiset, missing={pooled_missing}, extra={pooled_extra}."
        )

    if set(validation_fold_by_complex) != master_complexes:
        missing_complexes = master_complexes - set(validation_fold_by_complex)
        extra_complexes = set(validation_fold_by_complex) - master_complexes
        raise RuntimeError(
            f"{dataset}: pooled validation complex coverage mismatch, "
            f"missing={sorted(missing_complexes)[:20]}, "
            f"extra={sorted(extra_complexes)[:20]}."
        )

    return fold_records, validation_fold_by_complex


def resolve_mmseqs(executable: str) -> str:
    candidate = Path(executable)
    if candidate.is_file():
        return str(candidate.resolve())

    located = shutil.which(executable)
    if located:
        return located

    raise FileNotFoundError(
        f"MMseqs2 executable not found: {executable!r}. "
        "Pass its command or full path with --mmseqs."
    )


def mmseqs_version(mmseqs: str) -> str:
    completed = subprocess.run(
        [mmseqs, "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        text = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"Cannot run MMseqs2 executable {mmseqs!r}: {text}")

    text = (completed.stdout or completed.stderr or "").strip()
    return text.splitlines()[0] if text else "unknown"


def run_command(command: Sequence[str]) -> None:
    completed = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        output = (completed.stdout or "").strip()
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + "\n\n"
            + output[-5000:]
        )


def write_fasta(
    path: Path,
    records: Sequence[tuple[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sequence_id, sequence in records:
            handle.write(f">{sequence_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start:start + 80] + "\n")


def run_easy_cluster(
    mmseqs: str,
    fasta: Path,
    prefix: Path,
    tmp_dir: Path,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    cluster_mode: int,
    threads: int,
) -> Path:
    command = [
        mmseqs,
        "easy-cluster",
        str(fasta),
        str(prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--cluster-mode",
        str(cluster_mode),
        "--threads",
        str(threads),
    ]

    if cluster_mode == 1:
        command.extend(["--single-step-clustering", "1"])

    run_command(command)

    cluster_tsv = Path(str(prefix) + "_cluster.tsv")
    if not cluster_tsv.is_file():
        raise FileNotFoundError(
            f"MMseqs2 cluster output not found: {cluster_tsv}"
        )
    return cluster_tsv


def run_easy_search(
    mmseqs: str,
    fasta: Path,
    output_tsv: Path,
    tmp_dir: Path,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    threads: int,
    sensitivity: float,
    max_seqs: int,
) -> None:
    command = [
        mmseqs,
        "easy-search",
        str(fasta),
        str(fasta),
        str(output_tsv),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--alignment-mode",
        "3",
        "-s",
        str(sensitivity),
        "--max-seqs",
        str(max_seqs),
        "--threads",
        str(threads),
        "--format-output",
        ",".join(PAIR_FIELDS),
    ]
    run_command(command)

    if not output_tsv.is_file():
        raise FileNotFoundError(
            f"MMseqs2 search output not found: {output_tsv}"
        )


def parse_cluster_tsv(
    path: Path,
    all_ids: Sequence[str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            representative, member = parts[0], parts[1]
            mapping[member] = representative
            mapping.setdefault(representative, representative)

    for item in all_ids:
        mapping.setdefault(item, item)

    return mapping


def unit_fraction(value: str) -> float:
    number = float(value)
    return number / 100.0 if number > 1.0 else number


def parse_search_tsv(
    path: Path,
    side: str,
    dataset: str,
) -> list[dict[str, object]]:
    """Return one best row per unordered complex pair for one interaction side."""
    best: dict[tuple[str, str], dict[str, object]] = {}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, parts in enumerate(reader, start=1):
            if not parts:
                continue
            if len(parts) != len(PAIR_FIELDS):
                raise RuntimeError(
                    f"Unexpected MMseqs2 output in {path}, line {line_number}: "
                    f"expected {len(PAIR_FIELDS)} fields, got {len(parts)}."
                )

            raw = dict(zip(PAIR_FIELDS, parts))
            query = raw["query"]
            target = raw["target"]

            if query == target:
                continue

            suffix = f"__{side.upper()}"
            complex_a = query.rsplit(suffix, 1)[0]
            complex_b = target.rsplit(suffix, 1)[0]

            if complex_a == complex_b:
                continue

            first, second = sorted((complex_a, complex_b))
            row = {
                "dataset": dataset,
                "side": side,
                "complex_a": first,
                "complex_b": second,
                "sequence_identity": unit_fraction(raw["fident"]),
                "query_coverage": unit_fraction(raw["qcov"]),
                "target_coverage": unit_fraction(raw["tcov"]),
                "alignment_length": int(float(raw["alnlen"])),
                "query_length": int(float(raw["qlen"])),
                "target_length": int(float(raw["tlen"])),
                "evalue": raw["evalue"],
                "bits": float(raw["bits"]),
            }

            key = (first, second)
            score = (
                float(row["sequence_identity"]),
                min(
                    float(row["query_coverage"]),
                    float(row["target_coverage"]),
                ),
                float(row["bits"]),
            )

            previous = best.get(key)
            if previous is None:
                best[key] = row
                continue

            previous_score = (
                float(previous["sequence_identity"]),
                min(
                    float(previous["query_coverage"]),
                    float(previous["target_coverage"]),
                ),
                float(previous["bits"]),
            )
            if score > previous_score:
                best[key] = row

    return [best[key] for key in sorted(best)]


class UnionFind:
    def __init__(self, items: Iterable[str] = ()) -> None:
        self.parent: dict[str, str] = {item: item for item in items}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: str, second: str) -> None:
        root_a = self.find(first)
        root_b = self.find(second)
        if root_a == root_b:
            return
        if root_a <= root_b:
            self.parent[root_b] = root_a
        else:
            self.parent[root_a] = root_b


def build_sequence_relationships(
    dataset: str,
    complex_table: pd.DataFrame,
    mmseqs: str,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    cluster_mode: int,
    threads: int,
    search_sensitivity: float,
    max_seqs: int,
) -> tuple[
    list[dict[str, object]],
    dict[str, str],
    dict[str, str],
]:
    """Rebuild construction groups and the independent direct relation graph."""
    complexes = complex_table["complex_key"].astype(str).tolist()

    with tempfile.TemporaryDirectory(
        prefix=f"{dataset}_sequence_audit_"
    ) as temp_name:
        work_dir = Path(temp_name)

        side_cluster_maps: dict[str, dict[str, str]] = {}
        all_relations: list[dict[str, object]] = []

        for side in ("left", "right"):
            records = [
                (
                    f"{row.complex_key}__{side.upper()}",
                    getattr(row, f"{side}_sequence"),
                )
                for row in complex_table.itertuples(index=False)
            ]

            fasta = work_dir / f"{side}.fasta"
            write_fasta(fasta, records)
            sequence_ids = [sequence_id for sequence_id, _ in records]

            cluster_tsv = run_easy_cluster(
                mmseqs=mmseqs,
                fasta=fasta,
                prefix=work_dir / f"{side}_cluster",
                tmp_dir=work_dir / f"tmp_cluster_{side}",
                min_seq_id=min_seq_id,
                coverage=coverage,
                cov_mode=cov_mode,
                cluster_mode=cluster_mode,
                threads=threads,
            )
            side_cluster_maps[side] = parse_cluster_tsv(
                cluster_tsv,
                sequence_ids,
            )

            search_tsv = work_dir / f"{side}_all_vs_all.tsv"
            run_easy_search(
                mmseqs=mmseqs,
                fasta=fasta,
                output_tsv=search_tsv,
                tmp_dir=work_dir / f"tmp_search_{side}",
                min_seq_id=min_seq_id,
                coverage=coverage,
                cov_mode=cov_mode,
                threads=threads,
                sensitivity=search_sensitivity,
                max_seqs=max_seqs,
            )
            all_relations.extend(
                parse_search_tsv(search_tsv, side, dataset)
            )

    construction_uf = UnionFind(complexes)
    first_by_cluster: dict[tuple[str, str], str] = {}

    for row in complex_table.itertuples(index=False):
        complex_key = str(row.complex_key)

        for side in ("left", "right"):
            sequence_id = f"{complex_key}__{side.upper()}"
            representative = side_cluster_maps[side].get(sequence_id, "")
            if not representative:
                continue

            cluster_key = (side, representative)
            if cluster_key in first_by_cluster:
                construction_uf.union(
                    complex_key,
                    first_by_cluster[cluster_key],
                )
            else:
                first_by_cluster[cluster_key] = complex_key

    construction_group = {
        complex_key: construction_uf.find(complex_key)
        for complex_key in complexes
    }

    relation_uf = UnionFind(complexes)
    for relation in all_relations:
        relation_uf.union(
            str(relation["complex_a"]),
            str(relation["complex_b"]),
        )

    relation_component = {
        complex_key: relation_uf.find(complex_key)
        for complex_key in complexes
    }

    return all_relations, construction_group, relation_component


def groups_crossing_folds(
    group_by_complex: dict[str, str],
    validation_fold_by_complex: dict[str, int],
) -> set[str]:
    folds_by_group: dict[str, set[int]] = defaultdict(set)

    for complex_key, group in group_by_complex.items():
        folds_by_group[group].add(validation_fold_by_complex[complex_key])

    return {
        group
        for group, fold_ids in folds_by_group.items()
        if len(fold_ids) > 1
    }


def build_cross_fold_relationships(
    relations: Sequence[dict[str, object]],
    validation_fold_by_complex: dict[str, int],
    complex_table: pd.DataFrame,
    relation_component: dict[str, str],
) -> list[dict[str, object]]:
    catalog = complex_table.set_index("complex_key").to_dict("index")
    rows: list[dict[str, object]] = []

    for relation in relations:
        complex_a = str(relation["complex_a"])
        complex_b = str(relation["complex_b"])
        fold_a = validation_fold_by_complex[complex_a]
        fold_b = validation_fold_by_complex[complex_b]

        if fold_a == fold_b:
            continue

        info_a = catalog[complex_a]
        info_b = catalog[complex_b]

        rows.append(
            {
                **relation,
                "fold_a": fold_a,
                "fold_b": fold_b,
                "PDB_a": info_a["PDB"],
                "PDB_b": info_b["PDB"],
                "Partners_a": info_a["Partners"],
                "Partners_b": info_b["Partners"],
                "records_a": int(info_a["n_records"]),
                "records_b": int(info_b["n_records"]),
                "audited_connected_component": relation_component[complex_a],
            }
        )

    return rows


def audit_dataset(
    dataset: str,
    data_dir: Path,
    split_root: Path,
    split_name: str,
    folds: int,
    mmseqs: str,
    mmseqs_version_text: str,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    cluster_mode: int,
    threads: int,
    search_sensitivity: float,
    max_seqs: int,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    master_path = data_dir / dataset / f"{dataset}.csv"
    if not master_path.is_file():
        raise FileNotFoundError(f"Master dataset not found: {master_path}")

    master = read_csv_text(master_path)
    complex_table = build_complex_table(master, dataset)

    _, validation_fold_by_complex = load_final_icsv_folds(
        dataset=dataset,
        master=master,
        complex_table=complex_table,
        split_root=split_root,
        split_name=split_name,
        folds=folds,
    )

    relationships, construction_group, relation_component = (
        build_sequence_relationships(
            dataset=dataset,
            complex_table=complex_table,
            mmseqs=mmseqs,
            min_seq_id=min_seq_id,
            coverage=coverage,
            cov_mode=cov_mode,
            cluster_mode=cluster_mode,
            threads=threads,
            search_sensitivity=search_sensitivity,
            max_seqs=max_seqs,
        )
    )

    cross_fold_relationships = build_cross_fold_relationships(
        relations=relationships,
        validation_fold_by_complex=validation_fold_by_complex,
        complex_table=complex_table,
        relation_component=relation_component,
    )

    construction_crossing = groups_crossing_folds(
        construction_group,
        validation_fold_by_complex,
    )
    component_crossing = groups_crossing_folds(
        relation_component,
        validation_fold_by_complex,
    )

    exposed_complexes = {
        str(row["complex_a"])
        for row in cross_fold_relationships
    } | {
        str(row["complex_b"])
        for row in cross_fold_relationships
    }

    records_by_complex = dict(
        zip(
            complex_table["complex_key"],
            complex_table["n_records"].astype(int),
        )
    )
    exposed_records = int(
        sum(records_by_complex[complex_key] for complex_key in exposed_complexes)
    )
    master_records = int(len(master))

    summary = {
        "dataset": dataset,
        "master_records": master_records,
        "master_complexes": int(len(complex_table)),
        "construction_groups": int(len(set(construction_group.values()))),
        "construction_groups_crossing_folds": int(len(construction_crossing)),
        "qualifying_sequence_relationships": int(len(relationships)),
        "audited_connected_components": int(
            len(set(relation_component.values()))
        ),
        "connected_components_crossing_folds": int(len(component_crossing)),
        "cross_fold_sequence_relationships": int(
            len(cross_fold_relationships)
        ),
        "exposed_validation_complexes": int(len(exposed_complexes)),
        "exposed_validation_records": exposed_records,
        "exposed_validation_fraction": (
            exposed_records / master_records if master_records else float("nan")
        ),
        "exposed_validation_percent": (
            100.0 * exposed_records / master_records
            if master_records
            else float("nan")
        ),
        "minimum_sequence_identity": min_seq_id,
        "coverage_threshold": coverage,
        "coverage_mode": cov_mode,
        "cluster_mode": cluster_mode,
        "search_sensitivity": search_sensitivity,
        "max_seqs": max_seqs,
        "linker": LINKER,
        "mmseqs_version": mmseqs_version_text,
    }

    return summary, relationships, cross_fold_relationships


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if any(path.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}. "
                "Use --overwrite to replace the three publication result tables."
            )
        if overwrite:
            for filename in (
                "sequence_separation_summary.csv",
                "sequence_relationships.csv",
                "cross_fold_sequence_relationships.csv",
            ):
                target = path / filename
                if target.exists():
                    target.unlink()
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only publication audit of sequence separation in final ICSV folds."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=list(DEFAULT_DATASETS),
    )
    parser.add_argument("--split-name", default="ICSV")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--mmseqs",
        default="mmseqs",
        help="MMseqs2 executable name or full path.",
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--min-seq-id", type=float, default=0.30)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--cluster-mode", type=int, default=1)
    parser.add_argument("--search-sensitivity", type=float, default=7.5)
    parser.add_argument("--max-seqs", type=int, default=100000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the three publication result tables in the output directory.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1.")
    if not 0 <= args.min_seq_id <= 1:
        raise ValueError("--min-seq-id must be between 0 and 1.")
    if not 0 <= args.coverage <= 1:
        raise ValueError("--coverage must be between 0 and 1.")
    if args.max_seqs < 1:
        raise ValueError("--max-seqs must be at least 1.")


def main() -> int:
    args = parse_args()
    validate_args(args)

    mmseqs = resolve_mmseqs(args.mmseqs)
    version = mmseqs_version(mmseqs)
    prepare_output_dir(args.output_dir, args.overwrite)

    print("========== sequence_separation.py ==========")
    print(f"Data directory: {args.data_dir}")
    print(f"Split root:     {args.split_root}")
    print(f"Split name:     {args.split_name}")
    print(f"Datasets:       {args.datasets}")
    print(f"MMseqs2:        {version}")
    print(
        "Criteria:       "
        f"identity>={args.min_seq_id}, coverage>={args.coverage}, "
        f"cov_mode={args.cov_mode}"
    )

    summaries: list[dict[str, object]] = []
    relationships: list[dict[str, object]] = []
    cross_fold: list[dict[str, object]] = []

    for dataset in args.datasets:
        print(f"\n[AUDIT] {dataset}")
        summary, dataset_relationships, dataset_cross_fold = audit_dataset(
            dataset=dataset,
            data_dir=args.data_dir,
            split_root=args.split_root,
            split_name=args.split_name,
            folds=args.folds,
            mmseqs=mmseqs,
            mmseqs_version_text=version,
            min_seq_id=args.min_seq_id,
            coverage=args.coverage,
            cov_mode=args.cov_mode,
            cluster_mode=args.cluster_mode,
            threads=args.threads,
            search_sensitivity=args.search_sensitivity,
            max_seqs=args.max_seqs,
        )
        summaries.append(summary)
        relationships.extend(dataset_relationships)
        cross_fold.extend(dataset_cross_fold)

        print(
            f"[RESULT] {dataset}: "
            f"relationships={summary['qualifying_sequence_relationships']}, "
            f"components={summary['audited_connected_components']}, "
            f"cross-fold relationships={summary['cross_fold_sequence_relationships']}, "
            f"exposure={summary['exposed_validation_percent']:.4f}%"
        )

    summary_df = pd.DataFrame(summaries)
    relationships_df = pd.DataFrame(
        relationships,
        columns=[
            "dataset",
            "side",
            "complex_a",
            "complex_b",
            "sequence_identity",
            "query_coverage",
            "target_coverage",
            "alignment_length",
            "query_length",
            "target_length",
            "evalue",
            "bits",
        ],
    )
    cross_fold_df = pd.DataFrame(
        cross_fold,
        columns=[
            "dataset",
            "side",
            "complex_a",
            "complex_b",
            "sequence_identity",
            "query_coverage",
            "target_coverage",
            "alignment_length",
            "query_length",
            "target_length",
            "evalue",
            "bits",
            "fold_a",
            "fold_b",
            "PDB_a",
            "PDB_b",
            "Partners_a",
            "Partners_b",
            "records_a",
            "records_b",
            "audited_connected_component",
        ],
    )

    summary_path = args.output_dir / "sequence_separation_summary.csv"
    relationships_path = args.output_dir / "sequence_relationships.csv"
    cross_fold_path = (
        args.output_dir / "cross_fold_sequence_relationships.csv"
    )

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    relationships_df.to_csv(
        relationships_path,
        index=False,
        encoding="utf-8-sig",
    )
    cross_fold_df.to_csv(
        cross_fold_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[WRITE]")
    print(f"  {summary_path}")
    print(f"  {relationships_path}")
    print(f"  {cross_fold_path}")
    print("[DONE] Sequence-separation audit completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
