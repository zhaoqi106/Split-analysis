#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
structure_exposure.py

Publication-oriented audit of cross-fold molecular exposure in the final RCSV,
PCSV, and ICSV files for S1131 and M1101.

The script retains only calculations used by the manuscript and Supporting
Information Table S1:

1. Complete-PDB exposure for RCSV, PCSV, and ICSV.
2. Exact wild-type sequence exposure on the left and right interaction sides
   for PCSV and ICSV.
3. Paired structural exposure for PCSV and ICSV.

Paired structural exposure
--------------------------
A validation complex is structurally exposed only when ONE SAME training
complex satisfies the structural criteria on BOTH interaction sides:

    symmetric TM-score >= 0.70
    bidirectional structural coverage >= 0.80

For each side, symmetric TM-score is the smaller of the two US-align
normalizations, and bidirectional coverage is the smaller of:

    aligned_length / validation_length
    aligned_length / training_length

The script uses only wild-type structures. Mutant structures are never used.

Outputs
-------
structure_exposure_summary.csv
    Dataset/split-level record exposure percentages used for Table S1.
    Measures not summarized in the manuscript are left blank rather than
    reported as zero.

paired_structure_exposure_detail.csv
    One row per validation complex/fold for PCSV and ICSV, including the number
    of validation records, exposure status, and the best same-training-complex
    structural match. This table directly supports the paired structural
    exposure percentages.

All extracted chain files and pairwise US-align results are kept in a temporary
directory and removed automatically after the run.

Sequence definitions
--------------------
S1131 and M1101 share the same sequence columns.
The R/L prefixes are legacy storage labels; the audit interprets them only as
the left and right interaction sides and does not assign antibody or antigen
identity.

    left side, wild type  = R1-R6
    right side, wild type = L1-L3

Empty chain slots are skipped. Exact-sequence exposure uses only the
wild-type sequences.

PDB identifier normalization
----------------------------
Excel can convert a valid PDB identifier such as 1E96 to scientific notation
such as 1.00E+96. The script repairs that corruption before structure lookup and
split comparison. Custom identifiers such as HM_1YY9 are preserved.

Expected structure layout
-------------------------
All wild-type PDB files for a dataset must occur somewhere below:

    STRUCTURE_ROOT/{dataset}/wt/

The search is recursive. Repair-named files or exact PDB filenames located in a
repair/repaired/repairpdb directory are preferred over raw/original files.

Example
-------
python structure_exposure.py \
    --split-root data \
    --structure-root data \
    --output-dir output/structure_exposure \
    --usalign USalign \
    --threads 8
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "data"
DEFAULT_STRUCTURE_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "structure_exposure"
DEFAULT_DATASETS = ("S1131", "M1101")
DEFAULT_SPLITS = ("RCSV", "PCSV", "ICSV")

MAIN_TM_THRESHOLD = 0.70
MAIN_COVERAGE_THRESHOLD = 0.80

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

FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


@dataclass(frozen=True)
class ComplexInfo:
    dataset: str
    complex_key: str
    pdb_id: str
    partners: str
    left_chains: tuple[str, ...]
    right_chains: tuple[str, ...]
    left_sequence: str
    right_sequence: str
    left_path: str
    right_path: str


@dataclass(frozen=True)
class AlignResult:
    status: str
    tm_symmetric: float = math.nan
    coverage_symmetric: float = math.nan
    aligned_length: int = 0
    length_structure_1: int = 0
    length_structure_2: int = 0
    error: str = ""


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


def concat_exact_sequences(row: pd.Series, columns: Sequence[str]) -> str:
    """Join chains with a delimiter for exact sequence comparison only."""
    sequences = []
    for column in columns:
        sequence = clean_sequence(row[column]) if column in row.index else ""
        if sequence:
            sequences.append(sequence)
    return "|".join(sequences)


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


def normalize_split_frame(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if "PDB" not in df.columns or "Partners" not in df.columns:
        raise ValueError("Split CSV must contain PDB and Partners columns.")

    config = DATASET_SEQ_CONFIG[dataset]
    missing = [
        column
        for column in (
            *config["left_cols"],
            *config["right_cols"],
        )
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"{dataset} split CSV missing sequence columns: {missing}")

    result = df.copy()
    id_column = resolve_id_column(result)
    result["_PDB_norm"] = [
        normalize_pdb(pdb, fallback_id)
        for pdb, fallback_id in zip(result["PDB"], result[id_column])
    ]
    result["_Partners_norm"] = result["Partners"].map(clean_text)
    result["_complex_key"] = [
        f"{pdb}__{safe_id(partners)}"
        for pdb, partners in zip(result["_PDB_norm"], result["_Partners_norm"])
    ]
    result["_left_sequence"] = [
        concat_exact_sequences(row, config["left_cols"])
        for _, row in result.iterrows()
    ]
    result["_right_sequence"] = [
        concat_exact_sequences(row, config["right_cols"])
        for _, row in result.iterrows()
    ]

    if result["_PDB_norm"].eq("").any():
        rows = result.index[result["_PDB_norm"].eq("")].tolist()[:20]
        raise ValueError(f"Empty normalized PDB identifier at rows: {rows}")

    if result["_left_sequence"].eq("").any() or result["_right_sequence"].eq("").any():
        raise ValueError(
            f"{dataset}: at least one split record has a missing wild-type "
            "sequence on one interaction side."
        )

    return result


def parse_partners(value: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    text = re.sub(r"\s+", "", clean_text(value))
    if text.count("_") != 1:
        raise ValueError(
            f"Partners must contain exactly one underscore separating sides: {text!r}"
        )

    left, right = text.split("_", 1)
    if not left or not right:
        raise ValueError(f"Partners has an empty interaction side: {text!r}")

    def parse_side(side: str) -> tuple[str, ...]:
        if any(separator in side for separator in (",", "+", ";", ":", "/", "|")):
            chains = tuple(
                token
                for token in re.split(r"[,;+:/|]+", side)
                if token
            )
        else:
            chains = tuple(side)

        if not chains or any(len(chain) != 1 for chain in chains):
            raise ValueError(
                f"Expected one-character PDB chain identifiers in Partners: {side!r}"
            )
        return chains

    return parse_side(left), parse_side(right)


def discover_fold_files(
    split_root: Path,
    datasets: Sequence[str],
    splits: Sequence[str],
    folds: int,
) -> dict[tuple[str, str, int, str], Path]:
    files: dict[tuple[str, str, int, str], Path] = {}

    for dataset in datasets:
        for split in splits:
            split_dir = split_root / dataset / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Split directory not found: {split_dir}")

            for fold in range(folds):
                for role in ("train", "val"):
                    path = split_dir / f"{role}_fold_{fold}.csv"
                    if not path.is_file():
                        raise FileNotFoundError(f"Split file not found: {path}")
                    files[(dataset, split, fold, role)] = path

    return files


def load_split_tables(
    files: Mapping[tuple[str, str, int, str], Path],
) -> dict[tuple[str, str, int, str], pd.DataFrame]:
    tables: dict[tuple[str, str, int, str], pd.DataFrame] = {}

    for key, path in sorted(files.items()):
        dataset, _, _, _ = key
        tables[key] = normalize_split_frame(read_csv_text(path), dataset)

    return tables


def validate_split_tables(
    tables: Mapping[tuple[str, str, int, str], pd.DataFrame],
    datasets: Sequence[str],
    splits: Sequence[str],
    folds: int,
) -> None:
    """Verify that each split provides a complete 5-fold OOF partition."""
    for dataset in datasets:
        for split in splits:
            validation_ids: Counter[str] = Counter()
            reference_ids: Counter[str] | None = None

            for fold in range(folds):
                train = tables[(dataset, split, fold, "train")]
                val = tables[(dataset, split, fold, "val")]

                train_id_column = resolve_id_column(train)
                val_id_column = resolve_id_column(val)

                train_ids = Counter(
                    clean_text(value).upper()
                    for value in train[train_id_column]
                    if clean_text(value)
                )
                val_ids = Counter(
                    clean_text(value).upper()
                    for value in val[val_id_column]
                    if clean_text(value)
                )
                full_ids = train_ids + val_ids

                if reference_ids is None:
                    reference_ids = full_ids
                elif full_ids != reference_ids:
                    raise RuntimeError(
                        f"{dataset}/{split}/fold {fold}: train+validation ID "
                        "multiset differs from the other folds."
                    )

                validation_ids.update(val_ids)

            if reference_ids is None or validation_ids != reference_ids:
                raise RuntimeError(
                    f"{dataset}/{split}: pooled validation folds do not "
                    "reproduce the complete record multiset."
                )


def index_pdb_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Structure directory not found: {root}")

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".pdb"
    )
    if not files:
        raise FileNotFoundError(f"No PDB files found below: {root}")
    return files


def path_has_marker(path: Path, markers: Sequence[str]) -> bool:
    parts = [
        re.sub(r"[^a-z0-9]+", "", part.casefold())
        for part in path.parts[:-1]
    ]
    normalized_markers = [
        re.sub(r"[^a-z0-9]+", "", marker.casefold())
        for marker in markers
    ]
    return any(
        marker and (part == marker or marker in part)
        for part in parts
        for marker in normalized_markers
    )


def choose_unique(
    candidates: Sequence[Path],
    description: str,
) -> Path | None:
    unique = sorted(set(candidates), key=lambda path: str(path).casefold())
    if not unique:
        return None
    if len(unique) > 1:
        raise RuntimeError(
            f"Ambiguous {description}: {[str(path) for path in unique[:10]]}"
        )
    return unique[0]


def resolve_wt_path(
    pdb_id: str,
    indexed: Sequence[Path],
    preference: str,
) -> Path:
    pid = pdb_id.casefold()

    by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in indexed:
        by_stem[path.stem.casefold()].append(path)

    explicitly_repaired = [
        path
        for stem in (f"{pid}_repair", f"{pid}_repaired", f"repair_{pid}")
        for path in by_stem.get(stem, [])
    ]
    exact_raw = list(by_stem.get(pid, []))

    repair_directory_raw = [
        path
        for path in exact_raw
        if path_has_marker(
            path,
            ("repair", "repaired", "repairpdb", "foldx"),
        )
    ]
    original_directory_raw = [
        path
        for path in exact_raw
        if path_has_marker(
            path,
            ("wt", "wildtype", "wild_type", "original", "raw"),
        )
        and path not in repair_directory_raw
    ]
    unlabelled_raw = [
        path
        for path in exact_raw
        if path not in repair_directory_raw
        and path not in original_directory_raw
    ]

    if preference == "repaired":
        ranked = (
            ("explicit repaired filename", explicitly_repaired),
            ("exact PDB filename in repair directory", repair_directory_raw),
            ("exact PDB filename in original directory", original_directory_raw),
            ("unlabelled exact PDB filename", unlabelled_raw),
        )
    else:
        ranked = (
            ("exact PDB filename in original directory", original_directory_raw),
            ("unlabelled exact PDB filename", unlabelled_raw),
            ("explicit repaired filename", explicitly_repaired),
            ("exact PDB filename in repair directory", repair_directory_raw),
        )

    for description, candidates in ranked:
        selected = choose_unique(candidates, description)
        if selected is not None:
            return selected

    fallback = [
        path
        for path in indexed
        if re.fullmatch(
            rf"{re.escape(pid)}(?:[_-](?:wt|wildtype|wild_type|repair|repaired))(?:[_-]1)?",
            path.stem.casefold(),
        )
    ]
    selected = choose_unique(fallback, f"WT structure for {pdb_id}")
    if selected is None:
        raise FileNotFoundError(f"WT structure not found for PDB {pdb_id}")
    return selected


def first_model_atom_lines(path: Path) -> Iterable[str]:
    in_model = False
    saw_model = False

    with path.open("r", encoding="ascii", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip().upper()

            if record == "MODEL":
                if saw_model:
                    break
                saw_model = True
                in_model = True
                continue

            if record == "ENDMDL" and in_model:
                break

            if saw_model and not in_model:
                continue

            if record in {"ATOM", "HETATM", "TER"}:
                yield line.rstrip("\r\n")


def extract_chains(
    source: Path,
    target: Path,
    chains: Sequence[str],
) -> None:
    requested = tuple(chains)
    requested_set = set(requested)
    lines_by_chain: dict[str, list[str]] = {chain: [] for chain in requested}
    residues: dict[str, set[tuple[str, str]]] = {
        chain: set()
        for chain in requested
    }

    for line in first_model_atom_lines(source):
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue

        chain = line[21:22]
        if chain not in requested_set:
            continue

        lines_by_chain[chain].append(line)
        if (
            line[12:16].strip() == "CA"
            and line[16:17] in {" ", "A", "1"}
        ):
            residues[chain].add((line[22:26], line[26:27]))

    missing = [
        chain
        for chain in requested
        if not residues[chain]
    ]
    if missing:
        raise RuntimeError(
            f"Missing or CA-free chains {missing} in structure {source}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="ascii", newline="\n") as handle:
        serial = 1
        for chain in requested:
            for line in lines_by_chain[chain]:
                handle.write(line + "\n")
            handle.write(f"TER   {serial:5d}      {chain:>3s} {chain:1s}\n")
            serial += 1
        handle.write("END\n")


def collect_unique_complexes(
    tables: Mapping[tuple[str, str, int, str], pd.DataFrame],
    dataset: str,
) -> pd.DataFrame:
    pieces = []

    for (current_dataset, _, _, _), frame in tables.items():
        if current_dataset != dataset:
            continue
        pieces.append(
            frame[
                [
                    "_complex_key",
                    "_PDB_norm",
                    "_Partners_norm",
                    "_left_sequence",
                    "_right_sequence",
                ]
            ].copy()
        )

    all_rows = pd.concat(pieces, ignore_index=True)
    rows: list[dict[str, str]] = []

    for complex_key, group in all_rows.groupby("_complex_key", sort=True):
        pdbs = set(group["_PDB_norm"])
        partners = set(group["_Partners_norm"])
        left_sequences = set(group["_left_sequence"])
        right_sequences = set(group["_right_sequence"])

        if (
            len(pdbs) != 1
            or len(partners) != 1
            or len(left_sequences) != 1
            or len(right_sequences) != 1
        ):
            raise RuntimeError(
                f"Inconsistent metadata or WT sequence for complex {complex_key}"
            )

        rows.append(
            {
                "complex_key": str(complex_key),
                "pdb_id": next(iter(pdbs)),
                "partners": next(iter(partners)),
                "left_sequence": next(iter(left_sequences)),
                "right_sequence": next(iter(right_sequences)),
            }
        )

    return pd.DataFrame(rows)


def prepare_complex_structures(
    tables: Mapping[tuple[str, str, int, str], pd.DataFrame],
    datasets: Sequence[str],
    structure_root: Path,
    extraction_root: Path,
    wt_preference: str,
) -> dict[tuple[str, str], ComplexInfo]:
    prepared: dict[tuple[str, str], ComplexInfo] = {}

    for dataset in datasets:
        complex_table = collect_unique_complexes(tables, dataset)
        indexed = index_pdb_files(structure_root / dataset / "wt")

        for row in complex_table.itertuples(index=False):
            left_chains, right_chains = parse_partners(row.partners)
            wt_path = resolve_wt_path(
                row.pdb_id,
                indexed,
                preference=wt_preference,
            )

            base = extraction_root / dataset / safe_id(row.complex_key)
            left_path = base / "left.pdb"
            right_path = base / "right.pdb"

            extract_chains(wt_path, left_path, left_chains)
            extract_chains(wt_path, right_path, right_chains)

            prepared[(dataset, row.complex_key)] = ComplexInfo(
                dataset=dataset,
                complex_key=row.complex_key,
                pdb_id=row.pdb_id,
                partners=row.partners,
                left_chains=left_chains,
                right_chains=right_chains,
                left_sequence=row.left_sequence,
                right_sequence=row.right_sequence,
                left_path=str(left_path),
                right_path=str(right_path),
            )

    return prepared


def resolve_usalign(executable: str) -> str:
    candidate = Path(executable)
    if candidate.is_file():
        return str(candidate.resolve())

    located = shutil.which(executable)
    if located:
        return located

    if sys.platform.startswith("win") and not executable.lower().endswith(".exe"):
        located = shutil.which(executable + ".exe")
        if located:
            return located

    raise FileNotFoundError(
        f"US-align executable not found: {executable!r}. "
        "Pass its full path with --usalign."
    )


def parse_usalign_output(text: str) -> AlignResult:
    lengths = re.findall(
        r"Length of Structure_[12]\s*:\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    if len(lengths) < 2:
        lengths = re.findall(
            r"Length of Chain_[12]\s*:\s*(\d+)",
            text,
            re.IGNORECASE,
        )

    aligned = re.search(
        rf"Aligned length\s*=\s*(\d+)\s*,\s*RMSD\s*=\s*{FLOAT_PATTERN}"
        rf"[^\r\n]*Seq_ID[^\r\n]*=\s*{FLOAT_PATTERN}\s*$",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    tm_scores = re.findall(
        rf"TM-score\s*=\s*({FLOAT_PATTERN})",
        text,
        re.IGNORECASE,
    )

    if len(lengths) < 2 or aligned is None or len(tm_scores) < 2:
        compact = re.sub(r"\s+", " ", text).strip()[:1000]
        return AlignResult(status="parse_error", error=compact)

    length_1 = int(lengths[0])
    length_2 = int(lengths[1])
    aligned_length = int(aligned.group(1))
    tm_1 = float(tm_scores[0])
    tm_2 = float(tm_scores[1])

    coverage_1 = aligned_length / length_1 if length_1 else math.nan
    coverage_2 = aligned_length / length_2 if length_2 else math.nan

    return AlignResult(
        status="ok",
        tm_symmetric=min(tm_1, tm_2),
        coverage_symmetric=min(coverage_1, coverage_2),
        aligned_length=aligned_length,
        length_structure_1=length_1,
        length_structure_2=length_2,
    )


def run_usalign(
    executable: str,
    path_1: Path,
    path_2: Path,
    timeout: float,
) -> AlignResult:
    command = [
        executable,
        str(path_1),
        str(path_2),
        "-mm",
        "1",
        "-ter",
        "1",
    ]

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return AlignResult(
            status="timeout",
            error=f"timeout after {timeout:g} seconds",
        )
    except OSError as exc:
        return AlignResult(status="execution_error", error=str(exc))

    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "").strip()[:2000]
        return AlignResult(
            status="nonzero_exit",
            error=f"returncode={completed.returncode}; {error}",
        )

    return parse_usalign_output(completed.stdout)


def required_structural_pairs(
    tables: Mapping[tuple[str, str, int, str], pd.DataFrame],
    datasets: Sequence[str],
    folds: int,
) -> set[tuple[str, str, str]]:
    """Collect unique cross-fold complex pairs needed for PCSV and ICSV."""
    pairs: set[tuple[str, str, str]] = set()

    for dataset in datasets:
        for split in ("PCSV", "ICSV"):
            for fold in range(folds):
                train = tables[(dataset, split, fold, "train")]
                val = tables[(dataset, split, fold, "val")]

                train_keys = set(train["_complex_key"])
                val_keys = set(val["_complex_key"])

                for val_key in val_keys:
                    for train_key in train_keys:
                        if val_key == train_key:
                            continue
                        first, second = sorted((str(val_key), str(train_key)))
                        pairs.add((dataset, first, second))

    return pairs


def calculate_pairwise_structures(
    pairs: Sequence[tuple[str, str, str]],
    prepared: Mapping[tuple[str, str], ComplexInfo],
    usalign: str,
    threads: int,
    timeout: float,
) -> dict[tuple[str, str, str], dict[str, float]]:
    lookup: dict[tuple[str, str, str], dict[str, float]] = {}

    def worker(
        pair: tuple[str, str, str]
    ) -> tuple[tuple[str, str, str], dict[str, float]]:
        dataset, complex_a, complex_b = pair
        info_a = prepared[(dataset, complex_a)]
        info_b = prepared[(dataset, complex_b)]

        left = run_usalign(
            usalign,
            Path(info_a.left_path),
            Path(info_b.left_path),
            timeout,
        )
        right = run_usalign(
            usalign,
            Path(info_a.right_path),
            Path(info_b.right_path),
            timeout,
        )

        if left.status != "ok":
            raise RuntimeError(
                f"US-align failed for {dataset}/{complex_a}/{complex_b}/left: "
                f"{left.status}: {left.error}"
            )
        if right.status != "ok":
            raise RuntimeError(
                f"US-align failed for {dataset}/{complex_a}/{complex_b}/right: "
                f"{right.status}: {right.error}"
            )

        return pair, {
            "left_tm_symmetric": left.tm_symmetric,
            "left_coverage_symmetric": left.coverage_symmetric,
            "right_tm_symmetric": right.tm_symmetric,
            "right_coverage_symmetric": right.coverage_symmetric,
            "pair_tm_symmetric": min(
                left.tm_symmetric,
                right.tm_symmetric,
            ),
            "pair_coverage_symmetric": min(
                left.coverage_symmetric,
                right.coverage_symmetric,
            ),
        }

    pairs = list(pairs)
    print(
        f"[US-ALIGN] {len(pairs)} unique complex pairs, "
        f"{2 * len(pairs)} side alignments"
    )

    completed_count = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {
            pool.submit(worker, pair): pair
            for pair in pairs
        }

        for future in as_completed(futures):
            pair, metrics = future.result()
            lookup[pair] = metrics
            completed_count += 1
            if completed_count % 250 == 0 or completed_count == len(pairs):
                print(
                    f"[US-ALIGN] completed {completed_count}/{len(pairs)} pairs"
                )

    return lookup


def pair_metrics(
    dataset: str,
    first: str,
    second: str,
    lookup: Mapping[tuple[str, str, str], dict[str, float]],
) -> dict[str, float]:
    if first == second:
        return {
            "left_tm_symmetric": 1.0,
            "left_coverage_symmetric": 1.0,
            "right_tm_symmetric": 1.0,
            "right_coverage_symmetric": 1.0,
            "pair_tm_symmetric": 1.0,
            "pair_coverage_symmetric": 1.0,
        }

    complex_a, complex_b = sorted((first, second))
    key = (dataset, complex_a, complex_b)

    if key not in lookup:
        raise RuntimeError(f"Missing US-align pair result for {key}")
    return lookup[key]


def summarize_record_exposure(
    tables: Mapping[tuple[str, str, int, str], pd.DataFrame],
    pairwise: Mapping[tuple[str, str, str], dict[str, float]],
    datasets: Sequence[str],
    splits: Sequence[str],
    folds: int,
    tm_threshold: float,
    coverage_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    structural_detail_rows: list[dict[str, object]] = []

    for dataset in datasets:
        for split in splits:
            total_validation_records = 0
            pdb_exposed_records = 0
            exact_left_exposed_records = 0
            exact_right_exposed_records = 0
            paired_structure_exposed_records = 0

            summarize_exact_sequence = split in {"PCSV", "ICSV"}
            summarize_structure = split in {"PCSV", "ICSV"}

            for fold in range(folds):
                train = tables[(dataset, split, fold, "train")]
                val = tables[(dataset, split, fold, "val")]

                total_validation_records += len(val)

                train_pdbs = set(train["_PDB_norm"])
                pdb_exposed_records += int(
                    val["_PDB_norm"].isin(train_pdbs).sum()
                )

                if summarize_exact_sequence:
                    train_left_sequences = set(train["_left_sequence"])
                    train_right_sequences = set(train["_right_sequence"])
                    exact_left_exposed_records += int(
                        val["_left_sequence"].isin(train_left_sequences).sum()
                    )
                    exact_right_exposed_records += int(
                        val["_right_sequence"].isin(train_right_sequences).sum()
                    )

                if not summarize_structure:
                    continue

                train_complexes = sorted(set(train["_complex_key"]))

                for val_complex, val_group in val.groupby(
                    "_complex_key",
                    sort=False,
                ):
                    best: dict[str, object] | None = None
                    exposed = False

                    for train_complex in train_complexes:
                        metrics = pair_metrics(
                            dataset,
                            str(val_complex),
                            str(train_complex),
                            pairwise,
                        )

                        qualifies = (
                            metrics["left_tm_symmetric"] >= tm_threshold
                            and metrics["right_tm_symmetric"] >= tm_threshold
                            and metrics["left_coverage_symmetric"] >= coverage_threshold
                            and metrics["right_coverage_symmetric"] >= coverage_threshold
                        )

                        candidate = {
                            "train_complex": str(train_complex),
                            **metrics,
                            "qualifies": qualifies,
                        }

                        if best is None or (
                            candidate["pair_tm_symmetric"],
                            candidate["pair_coverage_symmetric"],
                            candidate["train_complex"],
                        ) > (
                            best["pair_tm_symmetric"],
                            best["pair_coverage_symmetric"],
                            best["train_complex"],
                        ):
                            best = candidate

                        if qualifies:
                            exposed = True

                    n_val_records = int(len(val_group))
                    if exposed:
                        paired_structure_exposed_records += n_val_records

                    first = val_group.iloc[0]
                    structural_detail_rows.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "fold": fold,
                            "validation_complex": str(val_complex),
                            "validation_PDB": first["_PDB_norm"],
                            "validation_Partners": first["_Partners_norm"],
                            "validation_records": n_val_records,
                            "paired_structure_exposed": int(exposed),
                            "best_training_complex": (
                                "" if best is None else best["train_complex"]
                            ),
                            "best_pair_tm_symmetric": (
                                math.nan
                                if best is None
                                else best["pair_tm_symmetric"]
                            ),
                            "best_left_tm_symmetric": (
                                math.nan
                                if best is None
                                else best["left_tm_symmetric"]
                            ),
                            "best_right_tm_symmetric": (
                                math.nan
                                if best is None
                                else best["right_tm_symmetric"]
                            ),
                            "best_pair_coverage_symmetric": (
                                math.nan
                                if best is None
                                else best["pair_coverage_symmetric"]
                            ),
                            "best_left_coverage_symmetric": (
                                math.nan
                                if best is None
                                else best["left_coverage_symmetric"]
                            ),
                            "best_right_coverage_symmetric": (
                                math.nan
                                if best is None
                                else best["right_coverage_symmetric"]
                            ),
                            "tm_threshold": tm_threshold,
                            "coverage_threshold": coverage_threshold,
                        }
                    )

            def percent(count: int) -> float:
                return (
                    100.0 * count / total_validation_records
                    if total_validation_records
                    else math.nan
                )

            summary_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "validation_records": total_validation_records,
                    "pdb_exposed_records": pdb_exposed_records,
                    "pdb_exposure_percent": percent(pdb_exposed_records),
                    "exact_left_sequence_exposed_records": (
                        exact_left_exposed_records
                        if summarize_exact_sequence
                        else pd.NA
                    ),
                    "exact_left_sequence_exposure_percent": (
                        percent(exact_left_exposed_records)
                        if summarize_exact_sequence
                        else math.nan
                    ),
                    "exact_right_sequence_exposed_records": (
                        exact_right_exposed_records
                        if summarize_exact_sequence
                        else pd.NA
                    ),
                    "exact_right_sequence_exposure_percent": (
                        percent(exact_right_exposed_records)
                        if summarize_exact_sequence
                        else math.nan
                    ),
                    "paired_structure_exposed_records": (
                        paired_structure_exposed_records
                        if summarize_structure
                        else pd.NA
                    ),
                    "paired_structure_exposure_percent": (
                        percent(paired_structure_exposed_records)
                        if summarize_structure
                        else math.nan
                    ),
                    "structural_tm_threshold": (
                        tm_threshold if summarize_structure else math.nan
                    ),
                    "structural_coverage_threshold": (
                        coverage_threshold if summarize_structure else math.nan
                    ),
                }
            )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(structural_detail_rows),
    )


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    output_files = (
        "structure_exposure_summary.csv",
        "paired_structure_exposure_detail.csv",
    )

    if path.exists():
        existing = [
            path / filename
            for filename in output_files
            if (path / filename).exists()
        ]
        if existing and not overwrite:
            raise FileExistsError(
                f"Publication outputs already exist in {path}. "
                "Use --overwrite to replace them."
            )
        if overwrite:
            for target in existing:
                target.unlink()

    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publication audit of PDB, exact-sequence, and paired structural "
            "exposure in RCSV/PCSV/ICSV."
        )
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=DEFAULT_SPLIT_ROOT,
    )
    parser.add_argument(
        "--structure-root",
        type=Path,
        default=DEFAULT_STRUCTURE_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=list(DEFAULT_SPLITS),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--usalign",
        default="USalign",
        help="US-align executable name or full path.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Parallel US-align worker count.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Maximum seconds per US-align side comparison.",
    )
    parser.add_argument(
        "--tm-threshold",
        type=float,
        default=MAIN_TM_THRESHOLD,
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=MAIN_COVERAGE_THRESHOLD,
    )
    parser.add_argument(
        "--wt-preference",
        choices=("repaired", "original"),
        default="repaired",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the two publication result tables.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive.")
    if not 0 < args.tm_threshold <= 1:
        raise ValueError("--tm-threshold must be in (0, 1].")
    if not 0 < args.coverage_threshold <= 1:
        raise ValueError("--coverage-threshold must be in (0, 1].")


def main() -> int:
    args = parse_args()
    validate_args(args)

    started = time.perf_counter()
    usalign = resolve_usalign(args.usalign)
    prepare_output_dir(args.output_dir, args.overwrite)

    print("========== structure_exposure.py ==========")
    print(f"Split root:      {args.split_root}")
    print(f"Structure root:  {args.structure_root}")
    print(f"Datasets:        {args.datasets}")
    print(f"Splits:          {args.splits}")
    print(f"US-align:        {usalign}")
    print(
        "Structural rule: "
        f"TM>={args.tm_threshold}, coverage>={args.coverage_threshold}, "
        "same training complex on both interaction sides"
    )

    files = discover_fold_files(
        args.split_root,
        args.datasets,
        args.splits,
        args.folds,
    )
    tables = load_split_tables(files)
    validate_split_tables(
        tables,
        args.datasets,
        args.splits,
        args.folds,
    )

    with tempfile.TemporaryDirectory(
        prefix="structure_exposure_"
    ) as temp_name:
        temp_root = Path(temp_name)

        prepared = prepare_complex_structures(
            tables=tables,
            datasets=args.datasets,
            structure_root=args.structure_root,
            extraction_root=temp_root / "extracted",
            wt_preference=args.wt_preference,
        )

        pairs = required_structural_pairs(
            tables=tables,
            datasets=args.datasets,
            folds=args.folds,
        )

        pairwise = calculate_pairwise_structures(
            pairs=sorted(pairs),
            prepared=prepared,
            usalign=usalign,
            threads=args.threads,
            timeout=args.timeout,
        )

        summary, detail = summarize_record_exposure(
            tables=tables,
            pairwise=pairwise,
            datasets=args.datasets,
            splits=args.splits,
            folds=args.folds,
            tm_threshold=args.tm_threshold,
            coverage_threshold=args.coverage_threshold,
        )

    summary_path = args.output_dir / "structure_exposure_summary.csv"
    detail_path = (
        args.output_dir / "paired_structure_exposure_detail.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    detail.to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[SUMMARY]")
    display_columns = [
        "dataset",
        "split",
        "pdb_exposure_percent",
        "exact_left_sequence_exposure_percent",
        "exact_right_sequence_exposure_percent",
        "paired_structure_exposure_percent",
    ]
    print(
        summary[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\n[WRITE]")
    print(f"  {summary_path}")
    print(f"  {detail_path}")
    print(f"[DONE] elapsed={time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
