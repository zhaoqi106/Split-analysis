from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DATA_ROOT = Path("/data")
DEFAULT_INPUT_DIR = DATA_ROOT / "data_split" / "raw"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "data_split" / "splits"
DEFAULT_SUMMARY_PATH = DATA_ROOT / "data_split" / "summary" / "split_summary.csv"
DEFAULT_DATASETS = ("S1131.csv", "M1101.csv")
DEFAULT_SEEDS = (0, 42, 1234, 2026, 1024)

LINKER = "GGGGGGGGGG"

# These legacy source-data columns represent the two interaction sides. Their
# R/L prefixes are storage labels and do not imply antibody/antigen identity.
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


def resolve_id_column(df: pd.DataFrame) -> str | None:
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
    return None


def normalize_pdb_id(value: object, fallback_id: object = "") -> str:
    """Normalize PDB IDs and repair Excel scientific-notation corruption.

    Examples
    --------
    1.00E+96 -> 1E96
    1E+96    -> 1E96
    1E96     -> 1E96
    HM_1YY9  -> HM_1YY9
    """
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


def normalize_pdb_column(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "PDB" not in df.columns:
        raise ValueError("Input CSV must contain column 'PDB'.")

    result = df.copy()
    id_col = resolve_id_column(result)
    fallback_ids: Iterable[object]
    if id_col is None:
        fallback_ids = [""] * len(result)
    else:
        fallback_ids = result[id_col]

    original = result["PDB"].map(clean_text)
    normalized = [
        normalize_pdb_id(pdb, fallback_id)
        for pdb, fallback_id in zip(result["PDB"], fallback_ids)
    ]
    result["PDB"] = normalized

    invalid = result["PDB"].eq("")
    if invalid.any():
        rows = result.index[invalid].tolist()[:20]
        raise ValueError(f"Empty PDB identifier after normalization at rows: {rows}")

    repaired = int(
        sum(
            CORRUPTED_SCIENTIFIC_RE.fullmatch(old.upper()) is not None
            and old.upper() != new
            for old, new in zip(original, normalized)
        )
    )
    return result, repaired


def safe_id(text: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "-", clean_text(text))


def clean_sequence(value: object) -> str:
    return re.sub(r"[^A-Za-z]", "", clean_text(value)).upper()


def concat_sequence_cols(
    row: pd.Series,
    columns: Sequence[str],
    linker: str = LINKER,
) -> str:
    sequences = []
    for column in columns:
        sequence = clean_sequence(row[column]) if column in row.index else ""
        if sequence:
            sequences.append(sequence)
    return linker.join(sequences)


def complex_key_from_values(pdb: object, partners: object) -> str:
    return f"{clean_text(pdb).upper()}__{safe_id(partners)}"


def complex_keys(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [
            complex_key_from_values(pdb, partners)
            for pdb, partners in zip(df["PDB"], df["Partners"])
        ],
        index=df.index,
        dtype="string",
    )


def random_5fold_indices(
    n: int,
    seed: int = 42,
    n_splits: int = 5,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    return [part.tolist() for part in np.array_split(indices, n_splits)]


def grouped_5fold_indices(
    group_labels: Sequence[object],
    seed: int = 42,
    n_splits: int = 5,
) -> list[list[int]]:
    """Greedily balance indivisible groups by record count.

    Group order is shuffled before the stable size sort so equally sized groups
    receive a reproducible seed-dependent tie break, matching the original
    construction.
    """
    rng = np.random.default_rng(seed)

    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_labels):
        group_to_indices[str(group)].append(index)

    groups = list(group_to_indices)
    if len(groups) < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} grouped folds from only {len(groups)} groups."
        )

    rng.shuffle(groups)
    groups = sorted(
        groups,
        key=lambda group: len(group_to_indices[group]),
        reverse=True,
    )

    folds: list[list[int]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    for group in groups:
        fold_id = int(np.argmin(fold_sizes))
        folds[fold_id].extend(group_to_indices[group])
        fold_sizes[fold_id] += len(group_to_indices[group])

    return folds


def resolve_dataset_csv(input_dir: Path, dataset_file: str) -> Path:
    """Resolve a dataset CSV from a flat directory or data/{dataset}/{dataset}.csv."""
    dataset_name = Path(dataset_file).stem
    csv_name = Path(dataset_file).name
    if Path(csv_name).suffix.casefold() != ".csv":
        csv_name = f"{dataset_name}.csv"

    flat_path = input_dir / csv_name
    if flat_path.is_file():
        return flat_path
    return input_dir / dataset_name / csv_name


def prepare_split_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        has_contents = any(path.iterdir())
        if has_contents and not overwrite:
            raise FileExistsError(
                f"Output directory already contains files: {path}. "
                "Use --overwrite to replace generated split files."
            )
        if overwrite:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def record_uid(dataset_name: str, row_index: int) -> str:
    """Match adapt.protocol.record_uid: official CSV row number, zero-padded."""
    return f"{dataset_name}:{int(row_index):06d}"


def serialized_row_keys(df: pd.DataFrame) -> list[tuple[str, ...]]:
    """Row identity after the same CSV round-trip used for fold files."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    loaded = pd.read_csv(buffer, dtype=str, keep_default_na=False)
    return [tuple(row) for row in loaded.itertuples(index=False, name=None)]


def write_icsv_group_map(
    dataset_name: str,
    identity_groups: pd.Series,
    output_root: Path,
) -> Path:
    """One row per official CSV record, using the groups that built ICSV folds."""
    labels = identity_groups.reset_index(drop=True).astype(str)
    frame = pd.DataFrame(
        {
            "record_uid": [
                record_uid(dataset_name, index) for index in range(len(labels))
            ],
            "group_id": labels.tolist(),
        }
    )
    path = output_root / dataset_name / "icsv_group_map.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(
        f"[WRITE] ICSV group map: {path} "
        f"rows={len(frame)} groups={frame['group_id'].nunique()}"
    )
    return path


def assert_icsv_matches_existing(
    df: pd.DataFrame,
    folds: Sequence[Sequence[int]],
    out_dir: Path,
) -> None:
    """Refuse to replace an outer ICSV split built from a different clustering."""
    if not out_dir.is_dir() or not any(out_dir.glob("*.csv")):
        return

    row_keys = serialized_row_keys(df)
    for fold_id, val_indices_raw in enumerate(folds):
        val_path = out_dir / f"val_fold_{fold_id}.csv"
        train_path = out_dir / f"train_fold_{fold_id}.csv"
        if not val_path.is_file() or not train_path.is_file():
            raise FileNotFoundError(
                f"Existing ICSV split is incomplete: {val_path} / {train_path}"
            )

        val_indices = set(int(index) for index in val_indices_raw)
        train_indices = set(range(len(df))) - val_indices
        expected = {
            "val": [row_keys[index] for index in sorted(val_indices)],
            "train": [row_keys[index] for index in sorted(train_indices)],
        }
        for split_name, path in (("val", val_path), ("train", train_path)):
            existing = pd.read_csv(path, dtype=str, keep_default_na=False)
            existing_keys = [
                tuple(row) for row in existing.itertuples(index=False, name=None)
            ]
            if existing_keys != expected[split_name]:
                raise RuntimeError(
                    f"Regenerated ICSV {split_name} fold {fold_id} does not match "
                    f"{path}. The new clustering is not the grouping used for the "
                    "current outer split, so the fold files were left unchanged."
                )


def write_fold_csvs(
    df: pd.DataFrame,
    folds: Sequence[Sequence[int]],
    out_dir: Path,
    overwrite: bool,
) -> None:
    prepare_split_directory(out_dir, overwrite)
    all_indices = np.arange(len(df))

    for fold_id, val_indices_raw in enumerate(folds):
        val_indices = np.asarray(sorted(val_indices_raw), dtype=int)
        train_indices = np.asarray(
            sorted(np.setdiff1d(all_indices, val_indices)),
            dtype=int,
        )

        train_df = df.iloc[train_indices].copy()
        val_df = df.iloc[val_indices].copy()

        train_path = out_dir / f"train_fold_{fold_id}.csv"
        val_path = out_dir / f"val_fold_{fold_id}.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        print(
            f"[WRITE] {out_dir.name} fold={fold_id} "
            f"train={len(train_df)} val={len(val_df)}"
        )


def resolve_mmseqs(executable: str) -> str:
    candidate = Path(executable)
    if candidate.is_file():
        return str(candidate.resolve())

    located = shutil.which(executable)
    if located:
        return located

    raise FileNotFoundError(
        f"MMseqs2 executable not found: {executable!r}. "
        "Pass the command or full path with --mmseqs."
    )


def write_fasta(records: Sequence[tuple[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sequence_id, sequence in records:
            if not sequence:
                continue
            handle.write(f">{sequence_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start:start + 80] + "\n")


def mmseqs_easy_cluster_command(
    mmseqs: str,
    fasta: Path,
    prefix: Path,
    tmp_dir: Path,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    cluster_mode: int,
    threads: int,
) -> list[str]:
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

    # MMseqs2 release 17 requires this compatibility flag for connected-
    # component clustering. It preserves the intended cluster-mode 1 behavior.
    if cluster_mode == 1:
        command.extend(["--single-step-clustering", "1"])

    return command


def run_mmseqs_easy_cluster(
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
    command = mmseqs_easy_cluster_command(
        mmseqs=mmseqs,
        fasta=fasta,
        prefix=prefix,
        tmp_dir=tmp_dir,
        min_seq_id=min_seq_id,
        coverage=coverage,
        cov_mode=cov_mode,
        cluster_mode=cluster_mode,
        threads=threads,
    )

    print("[MMSEQS]", " ".join(command))
    subprocess.run(command, check=True)

    cluster_tsv = Path(str(prefix) + "_cluster.tsv")
    if not cluster_tsv.is_file():
        raise FileNotFoundError(f"MMseqs2 cluster output not found: {cluster_tsv}")
    return cluster_tsv


def parse_mmseqs_cluster_tsv(
    cluster_tsv: Path,
    all_ids: Sequence[str],
) -> dict[str, str]:
    item_to_cluster: dict[str, str] = {}

    with cluster_tsv.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            representative, member = parts[0], parts[1]
            item_to_cluster[member] = representative
            item_to_cluster.setdefault(representative, representative)

    for item in all_ids:
        item_to_cluster.setdefault(item, item)

    return item_to_cluster


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: str, second: str) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if root_first <= root_second:
            self.parent[root_second] = root_first
        else:
            self.parent[root_first] = root_second


def build_complex_table(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    if dataset_name not in DATASET_SEQ_CONFIG:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if "Partners" not in df.columns:
        raise ValueError("Input CSV must contain column 'Partners'.")

    config = DATASET_SEQ_CONFIG[dataset_name]
    required = [
        "PDB",
        "Partners",
        *config["left_cols"],
        *config["right_cols"],
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} missing required columns: {missing}")

    work = df[
        ["PDB", "Partners", *config["left_cols"], *config["right_cols"]]
    ].copy()
    work["_complex_key"] = complex_keys(work)

    rows: list[dict[str, object]] = []

    for complex_key, group in work.groupby("_complex_key", sort=False):
        first = group.iloc[0]

        for column in (*config["left_cols"], *config["right_cols"]):
            values = {
                clean_sequence(value)
                for value in group[column]
                if clean_sequence(value)
            }
            if len(values) > 1:
                raise ValueError(
                    f"{dataset_name}/{complex_key}: multiple wild-type sequences "
                    f"found in column {column!r}. Clean the master CSV before "
                    "generating publication splits."
                )

        rows.append(
            {
                "complex_key": complex_key,
                "PDB": clean_text(first["PDB"]).upper(),
                "Partners": clean_text(first["Partners"]),
                "left_sequence": concat_sequence_cols(first, config["left_cols"]),
                "right_sequence": concat_sequence_cols(first, config["right_cols"]),
                "n_records": len(group),
            }
        )

    return pd.DataFrame(rows)


def build_identity_groups(
    df: pd.DataFrame,
    dataset_name: str,
    mmseqs: str,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    cluster_mode: int,
    threads: int,
) -> pd.Series:
    """Return one connected construction-group label for every input record."""
    complex_table = build_complex_table(df, dataset_name)

    with tempfile.TemporaryDirectory(prefix=f"{dataset_name}_icsv_mmseqs_") as temp_name:
        work_dir = Path(temp_name)

        side_to_mapping: dict[str, dict[str, str]] = {}

        for side in ("left", "right"):
            records: list[tuple[str, str]] = []
            suffix = side.upper()

            for row in complex_table.itertuples(index=False):
                sequence = getattr(row, f"{side}_sequence")
                if sequence:
                    records.append((f"{row.complex_key}__{suffix}", sequence))

            if not records:
                raise ValueError(
                    f"{dataset_name}: no valid wild-type sequences on the {side} side."
                )

            fasta = work_dir / f"{side}_side_sequences.fasta"
            write_fasta(records, fasta)

            all_ids = [sequence_id for sequence_id, _ in records]
            if len(all_ids) == 1:
                side_to_mapping[side] = {all_ids[0]: all_ids[0]}
                continue

            cluster_tsv = run_mmseqs_easy_cluster(
                mmseqs=mmseqs,
                fasta=fasta,
                prefix=work_dir / f"{side}_cluster30",
                tmp_dir=work_dir / f"tmp_{side}",
                min_seq_id=min_seq_id,
                coverage=coverage,
                cov_mode=cov_mode,
                cluster_mode=cluster_mode,
                threads=threads,
            )
            side_to_mapping[side] = parse_mmseqs_cluster_tsv(
                cluster_tsv,
                all_ids,
            )

        union_find = UnionFind()
        for complex_key in complex_table["complex_key"]:
            union_find.add(str(complex_key))

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
                    union_find.union(
                        complex_key,
                        first_complex_by_cluster[cluster_key],
                    )
                else:
                    first_complex_by_cluster[cluster_key] = complex_key

        complex_table["identity_group"] = complex_table["complex_key"].map(
            lambda key: union_find.find(str(key))
        )

    key_to_group = dict(
        zip(complex_table["complex_key"], complex_table["identity_group"])
    )
    row_keys = complex_keys(df)
    identity_groups = row_keys.map(key_to_group)

    if identity_groups.isna().any():
        rows = identity_groups.index[identity_groups.isna()].tolist()[:20]
        raise RuntimeError(
            f"{dataset_name}: some records failed to map to ICSV construction groups: {rows}"
        )

    return identity_groups.astype("string")


def summarize_split(
    df: pd.DataFrame,
    dataset_name: str,
    split_name: str,
    folds: Sequence[Sequence[int]],
    indivisible_group_labels: Sequence[object],
) -> list[dict[str, object]]:
    """Summarize only the split-composition quantities used in the study."""
    row_complex_keys = complex_keys(df)
    all_groups = pd.Series(indivisible_group_labels, dtype="string")

    rows: list[dict[str, object]] = []
    total_pdbs = int(df["PDB"].nunique())
    total_complexes = int(row_complex_keys.nunique())
    total_groups = int(all_groups.nunique())

    unit_name = {
        "RCSV": "record",
        "PCSV": "complete_PDB",
        "ICSV": "connected_sequence_group",
    }[split_name]

    for fold_id, val_indices_raw in enumerate(folds):
        val_indices = np.asarray(sorted(val_indices_raw), dtype=int)
        val = df.iloc[val_indices]
        val_complex_keys = row_complex_keys.iloc[val_indices]
        val_groups = all_groups.iloc[val_indices]

        rows.append(
            {
                "dataset": dataset_name,
                "split": split_name,
                "fold": fold_id,
                "grouping_unit": unit_name,
                "val_records": int(len(val)),
                "val_PDBs": int(val["PDB"].nunique()),
                "val_complexes": int(val_complex_keys.nunique()),
                "val_indivisible_groups": int(val_groups.nunique()),
                "total_records": int(len(df)),
                "total_PDBs": total_pdbs,
                "total_complexes": total_complexes,
                "total_indivisible_groups": total_groups,
            }
        )

    return rows


def process_dataset(
    csv_path: Path,
    output_root: Path,
    mmseqs: str,
    seeds: Sequence[int],
    n_splits: int,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    cluster_mode: int,
    threads: int,
    overwrite: bool,
    splits: Sequence[str] = ("RCSV", "PCSV", "ICSV"),
) -> list[dict[str, object]]:
    dataset_name = csv_path.stem

    if dataset_name not in DATASET_SEQ_CONFIG:
        raise ValueError(
            f"Unsupported dataset {dataset_name}. "
            f"Supported datasets: {sorted(DATASET_SEQ_CONFIG)}"
        )

    print("\n" + "=" * 80)
    print(f"[DATASET] {dataset_name}")
    print(f"[INPUT]   {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    if "Partners" not in df.columns:
        raise ValueError(f"{csv_path} missing column 'Partners'.")

    df, repaired_count = normalize_pdb_column(df)
    if repaired_count:
        print(
            f"[PDB] Repaired {repaired_count} Excel scientific-notation "
            "PDB identifier(s)."
        )

    requested = set(splits)
    identity_groups: pd.Series | None = None
    if "ICSV" in requested:
        print("[SPLIT] ICSV sequence groups")
        identity_groups = build_identity_groups(
            df=df,
            dataset_name=dataset_name,
            mmseqs=mmseqs,
            min_seq_id=min_seq_id,
            coverage=coverage,
            cov_mode=cov_mode,
            cluster_mode=cluster_mode,
            threads=threads,
        )
    pdb_groups = df["PDB"].astype("string")

    summary_rows: list[dict[str, object]] = []
    icsv_folds_by_seed: dict[int, list[list[int]]] = {}

    for seed in seeds:
        dataset_out = output_root / dataset_name / str(seed)
        dataset_out.mkdir(parents=True, exist_ok=True)
        print(f"[SEED] {seed} -> {dataset_out}")

        if "RCSV" in requested:
            print("[SPLIT] RCSV")
            rcsv_folds = random_5fold_indices(
                len(df),
                seed=seed,
                n_splits=n_splits,
            )
            write_fold_csvs(
                df,
                rcsv_folds,
                dataset_out / "RCSV",
                overwrite=overwrite,
            )
            for row in summarize_split(
                df,
                dataset_name,
                "RCSV",
                rcsv_folds,
                indivisible_group_labels=np.arange(len(df)),
            ):
                row["seed"] = seed
                summary_rows.append(row)

        if "PCSV" in requested:
            print("[SPLIT] PCSV")
            pcsv_folds = grouped_5fold_indices(
                pdb_groups,
                seed=seed,
                n_splits=n_splits,
            )
            write_fold_csvs(
                df,
                pcsv_folds,
                dataset_out / "PCSV",
                overwrite=overwrite,
            )
            for row in summarize_split(
                df,
                dataset_name,
                "PCSV",
                pcsv_folds,
                indivisible_group_labels=pdb_groups,
            ):
                row["seed"] = seed
                summary_rows.append(row)

        if "ICSV" in requested:
            assert identity_groups is not None
            print("[SPLIT] ICSV")
            icsv_folds_by_seed[seed] = grouped_5fold_indices(
                identity_groups,
                seed=seed,
                n_splits=n_splits,
            )

    if identity_groups is not None:
        for seed, icsv_folds in icsv_folds_by_seed.items():
            icsv_dir = output_root / dataset_name / str(seed) / "ICSV"
            print(f"[CHECK] ICSV seed={seed} against {icsv_dir}")
            assert_icsv_matches_existing(df, icsv_folds, icsv_dir)

        write_icsv_group_map(dataset_name, identity_groups, output_root)
        for seed, icsv_folds in icsv_folds_by_seed.items():
            write_fold_csvs(
                df,
                icsv_folds,
                output_root / dataset_name / str(seed) / "ICSV",
                overwrite=overwrite,
            )
            for row in summarize_split(
                df,
                dataset_name,
                "ICSV",
                icsv_folds,
                indivisible_group_labels=identity_groups,
            ):
                row["seed"] = seed
                summary_rows.append(row)

    group_count = (
        int(identity_groups.nunique()) if identity_groups is not None else "not built"
    )
    print(
        f"[INFO] {dataset_name}: records={len(df)}, "
        f"PDBs={df['PDB'].nunique()}, "
        f"complexes={complex_keys(df).nunique()}, "
        f"ICSV construction groups={group_count}"
    )
    print(f"[DONE] {dataset_name} -> {dataset_out}")

    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication RCSV, PCSV, and ICSV 5-fold splits."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing {dataset}.csv or {dataset}/{dataset}.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for {dataset}/{seed}/{RCSV,PCSV,ICSV} fold CSVs",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path for split_summary.csv",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset CSV filenames, e.g. S1131.csv M1101.csv",
    )
    parser.add_argument(
        "--mmseqs",
        default="mmseqs",
        help="MMseqs2 executable name or full path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single random seed. Ignored when --seeds is set.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Random seeds. Default: 0 42 1234 2026 1024.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("RCSV", "PCSV", "ICSV"),
        default=["RCSV", "PCSV", "ICSV"],
        help="Which outer splits to generate. ICSV also writes {dataset}/icsv_group_map.csv.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-seq-id", type=float, default=0.30)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--cluster-mode", type=int, default=1)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated split directories and split_summary.csv.",
    )
    parser.add_argument(
        "--entity-cv",
        nargs="+",
        choices=("AGCV", "PGCV", "PGCV_LEFT", "PGCV_RIGHT"),
        default=None,
        help=(
            "Opt-in entity-grouped splits. Does not change the default "
            "RCSV/PCSV/ICSV run. AGCV is M1101 antigen-side holdout. "
            "S1131 has no single biological target side, so use PGCV_LEFT "
            "and/or PGCV_RIGHT. A bare PGCV request is refused."
        ),
    )
    parser.add_argument(
        "--entity-output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Output root for entity-grouped splits, same layout as RCSV/PCSV/ICSV: "
            "{dataset}/{seed}/{AGCV,PGCV_LEFT,PGCV_RIGHT}. Existing files are never deleted."
        ),
    )
    parser.add_argument(
        "--entity-summary-path",
        type=Path,
        default=None,
        help="Summary CSV for entity splits. Default: {entity-output-root}/entity_split_summary.csv.",
    )
    parser.add_argument(
        "--skempi-csv",
        type=Path,
        default=Path("/data/external/SKEMPI/skempi_v2.csv"),
        help="Local SKEMPI v2 table used only as S1131 partner-name evidence.",
    )
    parser.add_argument(
        "--inner-val-fraction",
        type=float,
        default=0.2,
        help="Target fraction of outer_train records for inner_val. Groups stay intact.",
    )
    parser.add_argument(
        "--wt-pdb-dir",
        type=Path,
        default=None,
        help="Directory of local wt PDB files. Default: {input-dir}/{dataset}/wt.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if not 0 <= args.min_seq_id <= 1:
        raise ValueError("--min-seq-id must be between 0 and 1.")
    if not 0 <= args.coverage <= 1:
        raise ValueError("--coverage must be between 0 and 1.")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1.")
    if not 0 < args.inner_val_fraction < 0.5:
        raise ValueError("--inner-val-fraction must be between 0 and 0.5.")


def main() -> int:
    args = parse_args()
    validate_args(args)

    if args.entity_cv:
        from entity_group_cv import run_entity_cross_validation

        return run_entity_cross_validation(args)

    # MMseqs2 is an ICSV dependency. RCSV, PCSV, and entity-grouped splits
    # must be able to run when the executable is absent.
    mmseqs = resolve_mmseqs(args.mmseqs) if "ICSV" in args.splits else ""
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary_path = args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{summary_path} already exists. Use --overwrite to replace it."
        )

    print("========== three_splits.py ==========")
    print(f"Input directory: {args.input_dir}")
    print(f"Output root:     {args.output_root}")
    print(f"Summary path:    {summary_path}")
    print(f"Datasets:        {args.datasets}")
    if args.seeds:
        seeds = list(args.seeds)
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(DEFAULT_SEEDS)

    print(f"Seeds:           {seeds}")
    print(f"Splits:          {args.splits}")
    print(f"Folds:           {args.n_splits}")
    print(
        "MMseqs2:         "
        f"identity={args.min_seq_id}, coverage={args.coverage}, "
        f"cov_mode={args.cov_mode}, cluster_mode={args.cluster_mode}"
    )

    all_summary_rows: list[dict[str, object]] = []

    for dataset_file in args.datasets:
        csv_path = resolve_dataset_csv(args.input_dir, dataset_file)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        all_summary_rows.extend(
            process_dataset(
                csv_path=csv_path,
                output_root=args.output_root,
                mmseqs=mmseqs,
                seeds=seeds,
                n_splits=args.n_splits,
                min_seq_id=args.min_seq_id,
                coverage=args.coverage,
                cov_mode=args.cov_mode,
                cluster_mode=args.cluster_mode,
                threads=args.threads,
                overwrite=args.overwrite,
                splits=args.splits,
            )
        )

    summary = pd.DataFrame(all_summary_rows)
    requested = set(args.splits)
    if summary_path.exists() and requested != {"RCSV", "PCSV", "ICSV"}:
        existing = pd.read_csv(summary_path)
        datasets = [Path(dataset_file).stem for dataset_file in args.datasets]
        keep = ~(
            existing["dataset"].isin(datasets)
            & existing["split"].isin(requested)
            & existing["seed"].isin(seeds)
        )
        summary = pd.concat([existing.loc[keep], summary], ignore_index=True)
    summary.to_csv(summary_path, index=False)
    print(f"\n[WRITE] Split summary: {summary_path}")
    print("[DONE] All requested datasets completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
