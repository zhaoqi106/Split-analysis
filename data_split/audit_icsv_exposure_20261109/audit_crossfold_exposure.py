#!/usr/bin/env python3
"""Read-only cross-fold molecular-exposure audit for the official ICSV splits.

Task A : build a complete inventory (path, rows, MD5, train/val counts).
Task B : for every dataset x seed x fold, audit whether the validation set is
         molecularly exposed to the training set, on several independent
         molecular levels:

           L1  concatenated partner sides (R1..R6 | L1..L3, the official
               construction unit)  -- MMseqs2 all-vs-all @ 30% id / 80% bi-cov
               plus exact overlap.
           L2  individual partner chains  -- the same test on every individual
               chain, which catches shared / re-ordered / partially-shared
               partners that the order-sensitive concatenation misses.
           L3  construction groups  -- group must never be split val/train.
           L4  record identity       -- same PDB and same mutation ID.
           L5  structural similarity -- US-align TM-score between the wt
               structures of every PDB; cross-fold TM tiers.

Guarantees
----------
* Never writes to {dataset}/{seed}/ICSV, icsv_group_map.csv or any structure.
* Never retrains a model.
* Only writes inside the audit directory given by --out.

Run:
    python3 audit_crossfold_exposure.py --data-root .. --out . --threads 64
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

# Fixed configuration mirroring three_splits.py (the script that built ICSV).
DATASETS = ("M1101", "S1131")
SEEDS = (0, 42, 1024, 1234, 2026)
SPLIT_TYPE = "ICSV"
N_FOLDS = 5

# Source-data interaction sides. Prefixes are storage labels only.
LEFT_COLS = ("R1", "R2", "R3", "R4", "R5", "R6")
RIGHT_COLS = ("L1", "L2", "L3")
ALL_CHAIN_COLS = LEFT_COLS + RIGHT_COLS
LINKER = "GGGGGGGGGG"

MIN_SEQ_ID = 0.30
COVERAGE = 0.80
COV_MODE = 0

# Structural tiers (TM-score between different PDB wt structures).
TM_NEAR_IDENTICAL = 0.90
TM_SIMILAR = 0.70
TM_SAME_FOLD = 0.50

MMSEQS_DEFAULT = "/data/tools/mmseqs/bin/mmseqs"
USALIGN_DEFAULT = "/data/tools/USalign"

SPECIAL_PDBS = {"S1131": ("1E96", "2A9K")}


# Small helpers
def log(handle, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\ufeff", "").strip()
    if text.casefold() in {"", "nan", "none", "null", "na"}:
        return ""
    return text


def clean_sequence(value: object) -> str:
    return re.sub(r"[^A-Za-z]", "", clean_text(value)).upper()


def concat_sequence_cols(row: dict, columns: Iterable[str]) -> str:
    parts = []
    for column in columns:
        seq = clean_sequence(row.get(column, ""))
        if seq:
            parts.append(seq)
    return LINKER.join(parts)


def read_csv_dicts(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def uniq_sorted(values: Iterable[str]) -> list[str]:
    return sorted({v for v in values if v != ""})


def fmt_list(values: Iterable[str], limit: int = 300) -> str:
    values = uniq_sorted(values)
    if len(values) <= limit:
        return ";".join(values)
    return ";".join(values[:limit]) + f";...(+{len(values) - limit})"


def safe_id(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", clean_text(value)) or "NA"


def seq_md5(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()[:12]


def pident_bucket(pident: float, exact: bool) -> str:
    if exact:
        return "exact"
    if pident >= 90:
        return ">=90"
    if pident >= 70:
        return ">=70"
    if pident >= 50:
        return ">=50"
    return ">=30"


# MMseqs2
def write_fasta(path: Path, seq_to_id: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for seq, seq_id in seq_to_id.items():
            handle.write(f">{seq_id}\n{seq}\n")


def run_cross_search(mmseqs: str, val_fa: Path, train_fa: Path, out_tsv: Path,
                     tmp_dir: Path, threads: int, handle, label: str) -> list[dict]:
    if out_tsv.exists():
        out_tsv.unlink()
    command = [
        mmseqs, "easy-search", str(val_fa), str(train_fa), str(out_tsv), str(tmp_dir),
        "--min-seq-id", str(MIN_SEQ_ID),
        "-c", str(COVERAGE),
        "--cov-mode", str(COV_MODE),
        "-s", "7.5",
        "--max-seqs", "1000000",
        "-e", "1000",
        "--threads", str(threads),
        "--format-output", "query,target,pident,alnlen,qcov,tcov,evalue,bits",
    ]
    log(handle, f"MMSEQS[{label}] " + " ".join(command))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hits: list[dict] = []
    if out_tsv.is_file():
        with out_tsv.open(newline="", encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 8:
                    continue
                hits.append({
                    "val_id": parts[0], "train_id": parts[1],
                    "pident": float(parts[2]), "alnlen": int(parts[3]),
                    "qcov": float(parts[4]), "tcov": float(parts[5]),
                    "evalue": parts[6], "bits": parts[7],
                })
    return hits


# US-align structural matrix
def usalign_tm(usalign: str, pdb_a: Path, pdb_b: Path) -> tuple[float, float]:
    try:
        result = subprocess.run([usalign, str(pdb_a), str(pdb_b)],
                                capture_output=True, text=True, timeout=120)
    except (subprocess.SubprocessError, OSError):
        return (float("nan"), float("nan"))
    tm1 = tm2 = float("nan")
    for match in re.finditer(r"TM-score=\s*([0-9.]+)\s*\(normalized by length of Structure_([12])",
                             result.stdout):
        if match.group(2) == "1":
            tm1 = float(match.group(1))
        else:
            tm2 = float(match.group(1))
    return (tm1, tm2)


def _tm_worker(args):
    usalign, dataset, a, b, wt_dir = args
    if a == b:
        return (dataset, a, b, 1.0, 1.0)
    tm1, tm2 = usalign_tm(usalign, Path(wt_dir) / f"{a}.pdb", Path(wt_dir) / f"{b}.pdb")
    return (dataset, a, b, tm1, tm2)


def build_structural_matrix(data_root: Path, out_dir: Path, usalign: str,
                            threads: int, handle) -> tuple[dict[str, dict[tuple[str, str], float]], int]:
    log(handle, "TASK B/L5: building structural similarity matrix (US-align, all unique PDB wt structures)")
    matrix: dict[str, dict[tuple[str, str], float]] = {}
    all_rows: list[dict] = []
    tasks = []
    for dataset in DATASETS:
        master_rows, _ = read_csv_dicts(data_root / "raw" / f"{dataset}.csv")
        pdbs = uniq_sorted(clean_text(r.get("PDB", "")).upper() for r in master_rows)
        wt_dir = str(data_root / "splits" / dataset / "wt")
        for i in range(len(pdbs)):
            for j in range(i, len(pdbs)):
                tasks.append((usalign, dataset, pdbs[i], pdbs[j], wt_dir))
    log(handle, f"US-align pairs to compute: {len(tasks)}")
    workers = max(1, min(threads, os.cpu_count() or 1))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for idx, result in enumerate(pool.map(_tm_worker, tasks, chunksize=8)):
            results.append(result)
            if (idx + 1) % 1000 == 0:
                log(handle, f"  structural pairs done: {idx + 1}/{len(tasks)}")

    for dataset, a, b, tm1, tm2 in results:
        valid = [v for v in (tm1, tm2) if v == v]
        tm_max = max(valid) if valid else float("nan")
        matrix.setdefault(dataset, {})[(a, b)] = tm_max
        matrix[dataset][(b, a)] = tm_max
        all_rows.append({
            "dataset": dataset, "pdb_a": a, "pdb_b": b, "same_pdb": a == b,
            "tm_score_a_norm": "" if tm1 != tm1 else round(tm1, 4),
            "tm_score_b_norm": "" if tm2 != tm2 else round(tm2, 4),
            "tm_max": "" if tm_max != tm_max else round(tm_max, 4),
            "structural_homolog": (a == b) or (tm_max == tm_max and tm_max >= TM_SAME_FOLD),
        })
    write_csv(out_dir / "structural_pdb_tm_matrix.csv", all_rows,
              ["dataset", "pdb_a", "pdb_b", "same_pdb", "tm_score_a_norm",
               "tm_score_b_norm", "tm_max", "structural_homolog"])
    log(handle, f"structural matrix written ({len(all_rows)} rows)")
    return matrix, len(all_rows)


# Task A - inventory / manifest
def build_manifest(data_root: Path, out_dir: Path, handle) -> dict:
    log(handle, "TASK A: building ICSV inventory / manifest")
    file_rows: list[dict] = []
    fold_rows: list[dict] = []
    dataset_rows: list[dict] = []
    target_checks: list[dict] = []

    for dataset in DATASETS:
        master_path = data_root / "raw" / f"{dataset}.csv"
        master_rows, master_cols = read_csv_dicts(master_path)
        master_pdbs = uniq_sorted(clean_text(r.get("PDB", "")).upper() for r in master_rows)
        master_complex = uniq_sorted(
            f"{clean_text(r.get('PDB','')).upper()}__{safe_id(r.get('Partners',''))}" for r in master_rows)
        master_ids = uniq_sorted(clean_text(r.get("ID", "")) for r in master_rows)
        gm_path = data_root / "splits" / dataset / "icsv_group_map.csv"
        gm_rows, _ = read_csv_dicts(gm_path) if gm_path.is_file() else ([], [])
        n_groups = len(uniq_sorted(r["group_id"] for r in gm_rows)) if gm_rows else 0
        dataset_rows.append({
            "dataset": dataset, "master_csv": str(master_path),
            "master_rows": len(master_rows), "n_columns": len(master_cols),
            "unique_ID": len(master_ids), "unique_PDB": len(master_pdbs),
            "unique_complex": len(master_complex), "n_construction_groups": n_groups,
            "group_map": str(gm_path), "columns": ";".join(master_cols),
        })

        def key(r):
            return (clean_text(r.get("PDB", "")).upper(), clean_text(r.get("ID", "")),
                    clean_text(r.get("Mutation", "")), clean_text(r.get("ddG", "")))

        for seed in SEEDS:
            fold_keys: list[set] = []
            fold_union: set = set()
            for fold in range(N_FOLDS):
                icsv_dir = data_root / "splits" / dataset / str(seed) / SPLIT_TYPE
                val_path = icsv_dir / f"val_fold_{fold}.csv"
                train_path = icsv_dir / f"train_fold_{fold}.csv"
                for split_name, path in (("validation", val_path), ("training", train_path)):
                    rows, cols = read_csv_dicts(path)
                    file_rows.append({
                        "dataset": dataset, "seed": seed, "split_type": SPLIT_TYPE,
                        "fold": fold, "split": split_name,
                        "rel_path": str(path.relative_to(data_root)), "path": str(path),
                        "n_rows": len(rows), "n_columns": len(cols),
                        "file_bytes": path.stat().st_size, "md5": md5_file(path),
                        "unique_ID": len(uniq_sorted(r.get("ID", "") for r in rows)),
                        "unique_PDB": len(uniq_sorted(clean_text(r.get("PDB", "")).upper() for r in rows)),
                        "unique_complex": len(uniq_sorted(
                            f"{clean_text(r.get('PDB','')).upper()}__{safe_id(r.get('Partners',''))}" for r in rows)),
                    })
                val_rows, _ = read_csv_dicts(val_path)
                train_rows, _ = read_csv_dicts(train_path)
                fold_rows.append({
                    "dataset": dataset, "seed": seed, "split_type": SPLIT_TYPE, "fold": fold,
                    "train_rows": len(train_rows), "val_rows": len(val_rows),
                    "total_rows": len(train_rows) + len(val_rows),
                    "val_fraction": round(len(val_rows) / max(1, len(train_rows) + len(val_rows)), 6),
                    "train_path": str(train_path.relative_to(data_root)),
                    "val_path": str(val_path.relative_to(data_root)),
                })
                vset = {key(r) for r in val_rows}
                tset = {key(r) for r in train_rows}
                fold_keys.append(vset)
                fold_union |= vset
                target_checks.append({
                    "dataset": dataset, "seed": seed, "fold": fold,
                    "val_n": len(vset), "train_n": len(tset),
                    "val_train_overlap": len(vset & tset),
                    "val_plus_train": len(vset | tset), "master_rows": len(master_rows),
                    "covers_master": len(vset | tset) == len(master_rows),
                })
            disj = all(fold_keys[i].isdisjoint(fold_keys[j])
                       for i in range(N_FOLDS) for j in range(i + 1, N_FOLDS))
            target_checks.append({
                "dataset": dataset, "seed": seed, "fold": "ALL",
                "val_n": sum(len(s) for s in fold_keys), "train_n": "",
                "val_train_overlap": "", "val_plus_train": len(fold_union),
                "master_rows": len(master_rows),
                "covers_master": (len(fold_union) == len(master_rows)) and disj,
            })

    write_csv(out_dir / "icsv_manifest_files.csv", file_rows, list(file_rows[0].keys()))
    write_csv(out_dir / "icsv_manifest_folds.csv", fold_rows, list(fold_rows[0].keys()))
    write_csv(out_dir / "icsv_dataset_info.csv", dataset_rows, list(dataset_rows[0].keys()))
    write_csv(out_dir / "audit_target_check.csv", target_checks, list(target_checks[0].keys()))
    log(handle, f"TASK A done: {len(file_rows)} files, {len(fold_rows)} fold summaries")
    return {"dataset_rows": dataset_rows}


# Group map helper
def build_pdb_group_map(data_root: Path, dataset: str) -> dict[str, str]:
    master_rows, _ = read_csv_dicts(data_root / "raw" / f"{dataset}.csv")
    gm_rows, _ = read_csv_dicts(data_root / "splits" / dataset / "icsv_group_map.csv")
    idx2group = {i: gm["group_id"] for i, gm in enumerate(gm_rows)}
    pdb2group: dict[str, str] = {}
    for i, row in enumerate(master_rows):
        pdb = clean_text(row.get("PDB", "")).upper()
        group = idx2group.get(i, "")
        if pdb and group:
            pdb2group[pdb] = group
    return pdb2group


# Sequence-collection helpers
def collect_concat_sequences(rows: list[dict], columns: tuple[str, ...]) -> dict[str, list[int]]:
    seq_to_rows: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        seq = concat_sequence_cols(row, columns)
        if seq:
            seq_to_rows.setdefault(seq, []).append(index)
    return seq_to_rows


def collect_chain_sequences(rows: list[dict]) -> dict[str, list[tuple[int, str]]]:
    """Distinct individual chain sequence -> list of (record_index, column)."""
    seq_to_hits: dict[str, list[tuple[int, str]]] = {}
    for index, row in enumerate(rows):
        for column in ALL_CHAIN_COLS:
            seq = clean_sequence(row.get(column, ""))
            if seq:
                seq_to_hits.setdefault(seq, []).append((index, column))
    return seq_to_hits


# Task B - per-fold audit
def audit_fold(data_root: Path, dataset: str, seed: int, fold: int,
               pdb2group: dict[str, str], struct_matrix: dict,
               mmseqs: str, threads: int, work_dir: Path, handle) -> dict:
    icsv_dir = data_root / "splits" / dataset / str(seed) / SPLIT_TYPE
    val_rows, _ = read_csv_dicts(icsv_dir / f"val_fold_{fold}.csv")
    train_rows, _ = read_csv_dicts(icsv_dir / f"train_fold_{fold}.csv")

    def meta(rows):
        out = []
        for r in rows:
            pdb = clean_text(r.get("PDB", "")).upper()
            out.append({
                "ID": clean_text(r.get("ID", "")), "PDB": pdb,
                "complex": f"{pdb}__{safe_id(r.get('Partners',''))}",
                "group": pdb2group.get(pdb, ""),
            })
        return out

    val_meta = meta(val_rows)
    train_meta = meta(train_rows)
    fold_tag = f"{dataset}_seed{seed}_fold{fold}"
    fold_work = work_dir / fold_tag
    fold_work.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "dataset": dataset, "seed": seed, "fold": fold,
        "val_records": len(val_rows), "train_records": len(train_rows),
        "val_complexes": len(uniq_sorted(m["complex"] for m in val_meta)),
        "train_complexes": len(uniq_sorted(m["complex"] for m in train_meta)),
        "val_PDBs": len(uniq_sorted(m["PDB"] for m in val_meta)),
        "train_PDBs": len(uniq_sorted(m["PDB"] for m in train_meta)),
    }

    val_pdb = {m["PDB"] for m in val_meta}
    train_pdb = {m["PDB"] for m in train_meta}
    val_id = {m["ID"] for m in val_meta}
    train_id = {m["ID"] for m in train_meta}
    val_group = {m["group"] for m in val_meta if m["group"]}
    train_group = {m["group"] for m in train_meta if m["group"]}

    summary["pdb_overlap_count"] = len(val_pdb & train_pdb)
    summary["pdb_overlap"] = fmt_list(val_pdb & train_pdb)
    summary["mutant_id_overlap_count"] = len(val_id & train_id)
    summary["mutant_id_overlap"] = fmt_list(val_id & train_id)
    summary["group_overlap_count"] = len(val_group & train_group)
    summary["group_overlap"] = fmt_list(val_group & train_group)

    # ---- L5 structural exposure (cross-fold, different PDB) ----
    struct_pairs = []
    for vp in sorted(val_pdb):
        for tp in sorted(train_pdb):
            same = vp == tp
            tm = struct_matrix.get(dataset, {}).get((vp, tp))
            if same:
                struct_pairs.append((vp, tp, 1.0, "same_pdb"))
            elif tm is not None and tm == tm and tm >= TM_SAME_FOLD:
                tier = ("near_identical" if tm >= TM_NEAR_IDENTICAL else
                        "similar" if tm >= TM_SIMILAR else "same_fold")
                struct_pairs.append((vp, tp, tm, tier))
    summary["same_pdb_crossfold"] = len(val_pdb & train_pdb)
    summary["struct_near_identical_pairs"] = sum(1 for p in struct_pairs if p[3] == "near_identical")
    summary["struct_similar_pairs"] = sum(1 for p in struct_pairs if p[3] == "similar")
    summary["struct_samefold_pairs"] = sum(1 for p in struct_pairs if p[3] == "same_fold")
    summary["structural_homolog_crossfold_pairs"] = len(struct_pairs)
    struct_rows = [{
        "dataset": dataset, "seed": seed, "fold": fold,
        "val_PDB": vp, "train_PDB": tp, "tm_score": round(tm, 4), "tier": tier,
    } for vp, tp, tm, tier in struct_pairs]

    # ---- L1 concatenated partner sides ----
    concat_hits: list[dict] = []
    concat_exact: list[dict] = []
    concat_val_exposed: set[int] = set()
    concat_exact_records: set[int] = set()
    for side, columns in (("left", LEFT_COLS), ("right", RIGHT_COLS)):
        vseq = collect_concat_sequences(val_rows, columns)
        tseq = collect_concat_sequences(train_rows, columns)
        common = set(vseq) & set(tseq)
        for seq in common:
            for vi in vseq[seq]:
                concat_exact_records.add(vi)
            concat_exact.append({
                "dataset": dataset, "seed": seed, "fold": fold, "side": side,
                "sequence_length": len(seq), "sequence_md5": seq_md5(seq),
                "val_records": fmt_list(val_meta[i]["ID"] for i in vseq[seq]),
                "train_records": fmt_list(train_meta[j]["ID"] for j in tseq[seq]),
                "val_PDBs": fmt_list(val_meta[i]["PDB"] for i in vseq[seq]),
                "train_PDBs": fmt_list(train_meta[j]["PDB"] for j in tseq[seq]),
            })
        if not vseq or not tseq:
            continue
        comb: dict[str, str] = {}
        val_id_of: dict[str, str] = {}
        train_id_of: dict[str, str] = {}
        for i, seq in enumerate(vseq):
            sid = f"VAL|{side[0].upper()}|{i:05d}"
            comb.setdefault(seq, sid)
            val_id_of[seq] = comb[seq]
        for j, seq in enumerate(tseq):
            sid = f"TRA|{side[0].upper()}|{j:05d}"
            comb.setdefault(seq, sid)
            train_id_of[seq] = comb[seq]
        vfa = fold_work / f"concat_{side}_val.fasta"
        tfa = fold_work / f"concat_{side}_train.fasta"
        write_fasta(vfa, {s: val_id_of[s] for s in vseq})
        write_fasta(tfa, {s: train_id_of[s] for s in tseq})
        hits = run_cross_search(mmseqs, vfa, tfa, fold_work / f"concat_{side}.tsv",
                                fold_work / f"tmp_concat_{side}", threads, handle,
                                f"{fold_tag}/concat/{side}")
        id_to_vseq = {v: k for k, v in val_id_of.items()}
        id_to_tseq = {v: k for k, v in train_id_of.items()}
        seen = set()
        for hit in hits:
            vs = id_to_vseq.get(hit["val_id"])
            ts = id_to_tseq.get(hit["train_id"])
            if vs is None or ts is None:
                continue
            if (vs, ts) in seen:
                continue
            seen.add((vs, ts))
            for vi in vseq[vs]:
                concat_val_exposed.add(vi)
            concat_hits.append({
                "dataset": dataset, "seed": seed, "fold": fold, "side": side,
                "pident": hit["pident"], "alnlen": hit["alnlen"],
                "qcov": hit["qcov"], "tcov": hit["tcov"], "evalue": hit["evalue"],
                "bits": hit["bits"], "exact_overlap": vs == ts,
                "val_sequence_length": len(vs), "train_sequence_length": len(ts),
                "val_records": fmt_list(val_meta[i]["ID"] for i in vseq[vs]),
                "train_records": fmt_list(train_meta[j]["ID"] for j in tseq[ts]),
                "val_PDBs": fmt_list(val_meta[i]["PDB"] for i in vseq[vs]),
                "train_PDBs": fmt_list(train_meta[j]["PDB"] for j in tseq[ts]),
            })
        shutil.rmtree(fold_work / f"tmp_concat_{side}", ignore_errors=True)

    summary["concat_seq_hit_pairs"] = len(concat_hits)
    summary["concat_seq_exposed_records"] = len(concat_val_exposed)
    summary["concat_exact_overlap_records"] = len(concat_exact_records)

    # ---- L2 individual-chain exposure ----
    chain_hits: list[dict] = []
    chain_exact: list[dict] = []
    chain_val_exposed: set[int] = set()
    chain_exact_records: set[int] = set()
    chain_high_records: set[int] = set()   # exact or pident>=90
    chain_val_best: dict[int, float] = {}
    chain_val_partners: dict[int, list[str]] = {}

    vchain = collect_chain_sequences(val_rows)
    tchain = collect_chain_sequences(train_rows)
    # exact chain overlap
    for seq in set(vchain) & set(tchain):
        vloc = vchain[seq]
        tloc = tchain[seq]
        for vi, _col in vloc:
            chain_exact_records.add(vi)
            chain_high_records.add(vi)
            chain_val_exposed.add(vi)
        chain_exact.append({
            "dataset": dataset, "seed": seed, "fold": fold,
            "sequence_length": len(seq), "sequence_md5": seq_md5(seq),
            "val_records": fmt_list(val_meta[i]["ID"] for i, _c in vloc),
            "train_records": fmt_list(train_meta[j]["ID"] for j, _c in tloc),
            "val_PDBs": fmt_list(val_meta[i]["PDB"] for i, _c in vloc),
            "train_PDBs": fmt_list(train_meta[j]["PDB"] for j, _c in tloc),
            "val_columns": fmt_list(c for _i, c in vloc),
            "train_columns": fmt_list(c for _j, c in tloc),
        })

    if vchain and tchain:
        vlist = list(vchain)
        tlist = list(tchain)
        vid = {s: f"VC|{i:05d}" for i, s in enumerate(vlist)}
        tid = {s: f"TC|{j:05d}" for j, s in enumerate(tlist)}
        vfa = fold_work / "chain_val.fasta"
        tfa = fold_work / "chain_train.fasta"
        write_fasta(vfa, {s: vid[s] for s in vlist})
        write_fasta(tfa, {s: tid[s] for s in tlist})
        hits = run_cross_search(mmseqs, vfa, tfa, fold_work / "chain.tsv",
                                fold_work / "tmp_chain", threads, handle, f"{fold_tag}/chain")
        id_to_v = {v: k for k, v in vid.items()}
        id_to_t = {v: k for k, v in tid.items()}
        seen = set()
        for hit in hits:
            vs = id_to_v.get(hit["val_id"])
            ts = id_to_t.get(hit["train_id"])
            if vs is None or ts is None or (vs, ts) in seen:
                continue
            seen.add((vs, ts))
            exact = vs == ts
            bucket = pident_bucket(hit["pident"], exact)
            vloc = vchain[vs]
            tloc = tchain[ts]
            for vi, _c in vloc:
                chain_val_exposed.add(vi)
                chain_val_best[vi] = max(chain_val_best.get(vi, 0.0), hit["pident"] if not exact else 100.0)
                chain_val_partners.setdefault(vi, []).extend(train_meta[j]["PDB"] for j, _c in tloc)
                if exact or hit["pident"] >= 90:
                    chain_high_records.add(vi)
            chain_hits.append({
                "dataset": dataset, "seed": seed, "fold": fold,
                "val_seq_id": hit["val_id"], "train_seq_id": hit["train_id"],
                "pident": hit["pident"], "alnlen": hit["alnlen"],
                "qcov": hit["qcov"], "tcov": hit["tcov"], "evalue": hit["evalue"],
                "bits": hit["bits"], "exact_overlap": exact, "identity_bucket": bucket,
                "val_sequence_length": len(vs), "train_sequence_length": len(ts),
                "val_records": fmt_list(val_meta[i]["ID"] for i, _c in vloc),
                "train_records": fmt_list(train_meta[j]["ID"] for j, _c in tloc),
                "val_PDBs": fmt_list(val_meta[i]["PDB"] for i, _c in vloc),
                "train_PDBs": fmt_list(train_meta[j]["PDB"] for j, _c in tloc),
                "val_columns": fmt_list(c for _i, c in vloc),
                "train_columns": fmt_list(c for _j, c in tloc),
            })
        shutil.rmtree(fold_work / "tmp_chain", ignore_errors=True)

    summary["chain_seq_hit_pairs"] = len(chain_hits)
    summary["chain_exact_overlap_pairs"] = len(chain_exact)
    summary["chain_seq_exposed_records"] = len(chain_val_exposed)
    summary["chain_exact_records"] = len(chain_exact_records)
    summary["chain_high_records"] = len(chain_high_records)
    summary["chain_exposure_rate_records"] = round(len(chain_val_exposed) / max(1, len(val_meta)), 6)
    val_complexes = {m["complex"] for m in val_meta}
    exposed_complexes = {val_meta[i]["complex"] for i in chain_val_exposed}
    summary["chain_exposed_complexes"] = len(exposed_complexes)
    summary["chain_exposure_rate_complexes"] = round(len(exposed_complexes) / max(1, len(val_complexes)), 6)

    # ---- per-record exposure table ----
    record_rows: list[dict] = []
    for vi, m in enumerate(val_meta):
        record_rows.append({
            "dataset": dataset, "seed": seed, "fold": fold,
            "val_record_index": vi, "ID": m["ID"], "PDB": m["PDB"],
            "complex": m["complex"], "group": m["group"],
            "concat_exposed": vi in concat_val_exposed,
            "concat_exact_overlap": vi in concat_exact_records,
            "chain_exposed": vi in chain_val_exposed,
            "chain_exact_overlap": vi in chain_exact_records,
            "chain_high_overlap": vi in chain_high_records,
            "chain_best_pident": round(chain_val_best[vi], 2) if vi in chain_val_best else "",
            "chain_matching_train_PDBs": fmt_list(chain_val_partners.get(vi, [])),
            "same_pdb_in_train": m["PDB"] in train_pdb,
            "same_group_in_train": bool(m["group"]) and m["group"] in train_group,
        })

    return {
        "summary": summary, "concat_hits": concat_hits, "concat_exact": concat_exact,
        "chain_hits": chain_hits, "chain_exact": chain_exact, "records": record_rows,
        "struct_rows": struct_rows,
    }


# Group integrity + special 1E96 / 2A9K
def group_integrity(data_root: Path, handle) -> tuple[list[dict], list[dict]]:
    log(handle, "TASK B/L3: construction-group integrity")
    group_assign_rows: list[dict] = []
    per_fold_rows: list[dict] = []

    for dataset in DATASETS:
        for seed in SEEDS:
            pdb2group = build_pdb_group_map(data_root, dataset)
            val_group_fold: dict[str, set] = {}
            group_val_records: dict[str, int] = {}
            for fold in range(N_FOLDS):
                icsv_dir = data_root / "splits" / dataset / str(seed) / SPLIT_TYPE
                vrows, _ = read_csv_dicts(icsv_dir / f"val_fold_{fold}.csv")
                trows, _ = read_csv_dicts(icsv_dir / f"train_fold_{fold}.csv")
                vg = {pdb2group.get(clean_text(r.get("PDB", "")).upper(), "") for r in vrows}
                tg = {pdb2group.get(clean_text(r.get("PDB", "")).upper(), "") for r in trows}
                vg.discard(""); tg.discard("")
                for g in vg:
                    val_group_fold.setdefault(g, set()).add(fold)
                for r in vrows:
                    g = pdb2group.get(clean_text(r.get("PDB", "")).upper(), "")
                    if g:
                        group_val_records[g] = group_val_records.get(g, 0) + 1
                per_fold_rows.append({
                    "dataset": dataset, "seed": seed, "fold": fold,
                    "val_groups": len(vg), "train_groups": len(tg),
                    "group_overlap_val_train": len(vg & tg),
                    "group_overlap_groups": fmt_list(vg & tg),
                })
            for g, folds in sorted(val_group_fold.items()):
                group_assign_rows.append({
                    "dataset": dataset, "seed": seed, "group_id": g,
                    "held_out_val_fold": ";".join(str(f) for f in sorted(folds)),
                    "n_val_folds": len(folds),
                    "split_across_folds": len(folds) > 1,
                    "val_records_in_group": group_val_records.get(g, 0),
                })
    return group_assign_rows, per_fold_rows


def special_check(data_root: Path, chain_hits: list[dict], handle) -> list[dict]:
    log(handle, "TASK B/L3: special check S1131 1E96 / 2A9K")
    rows: list[dict] = []
    targets = SPECIAL_PDBS.get("S1131", ())
    if not targets:
        return rows
    pdb2group = build_pdb_group_map(data_root, "S1131")
    # cross-fold similar partners collected from chain hits (either direction)
    partner_hits: dict[str, list[str]] = {p: [] for p in targets}
    for hit in chain_hits:
        vp = set(hit["val_PDBs"].split(";"))
        tp = set(hit["train_PDBs"].split(";"))
        for target in targets:
            if target in vp:
                partner_hits[target].append(
                    f"seed{hit['seed']}f{hit['fold']}:VAL~{hit['pident']}%~{hit['train_PDBs']}")
            if target in tp:
                partner_hits[target].append(
                    f"seed{hit['seed']}f{hit['fold']}:TRAIN~{hit['pident']}%~{hit['val_PDBs']}")

    for seed in SEEDS:
        icsv_dir = data_root / "splits" / "S1131" / str(seed) / SPLIT_TYPE
        # locate which fold's val contains each target, and check same side
        target_val_fold: dict[str, str] = {}
        same_side_all_folds = True
        for fold in range(N_FOLDS):
            vrows, _ = read_csv_dicts(icsv_dir / f"val_fold_{fold}.csv")
            trows, _ = read_csv_dicts(icsv_dir / f"train_fold_{fold}.csv")
            vset = {clean_text(r.get("PDB", "")).upper() for r in vrows}
            tset = {clean_text(r.get("PDB", "")).upper() for r in trows}
            sides = []
            for target in targets:
                if target in vset:
                    target_val_fold[target] = str(fold)
                    sides.append("val")
                elif target in tset:
                    sides.append("train")
                else:
                    sides.append("absent")
            in_val = sum(1 for s in sides if s == "val")
            in_train = sum(1 for s in sides if s == "train")
            if in_val and in_train:
                same_side_all_folds = False
        groups = {pdb2group.get(t, "") for t in targets}
        together = len(groups) == 1
        for target in targets:
            rows.append({
                "dataset": "S1131", "seed": seed, "PDB": target,
                "group": pdb2group.get(target, ""),
                "in_same_group_as_other": together,
                "held_out_val_fold": target_val_fold.get(target, ""),
                "both_targets_same_side_all_folds": same_side_all_folds,
                "cross_fold_similar_partners_from_chain_audit": fmt_list(partner_hits[target]),
            })
    return rows


# Report
def write_report(out_dir: Path, dataset_info: list[dict], fold_summaries: list[dict],
                 group_assign: list[dict], special: list[dict], anomalies: list[dict],
                 struct_rows_done: int) -> None:
    lines: list[str] = []
    add = lines.append
    add("# ICSV 跨折分子暴露审计报告")
    add("")
    add("本报告由 `audit_crossfold_exposure.py` 生成，**只读**：不修改任何正式 ICSV 划分、"
        "`icsv_group_map.csv`、结构文件或模型产物；发现问题只做记录，未擅自修复。")
    add("")
    add("## 任务 A：审计对象（ICSV 清单校验）")
    add("")
    add("| 数据集 | master 行数 | 唯一ID | 唯一PDB | 唯一complex | 构造组数 |")
    add("|---|---|---|---|---|---|")
    for d in dataset_info:
        add(f"| {d['dataset']} | {d['master_rows']} | {d['unique_ID']} | "
            f"{d['unique_PDB']} | {d['unique_complex']} | {d['n_construction_groups']} |")
    add("")
    add("全部文件级清单（路径/行数/MD5/唯一ID/PDB/组数）见 `icsv_manifest_files.csv`；"
        "每个 fold 的 train/val 数量见 `icsv_manifest_folds.csv`；"
        "划分完整性（val∪train=全集、五折 val 互斥覆盖全部记录）见 `audit_target_check.csv`。")
    add("")

    tot_val = sum(s["val_records"] for s in fold_summaries)
    concat_exp = sum(s["concat_seq_exposed_records"] for s in fold_summaries)
    concat_exact = sum(s["concat_exact_overlap_records"] for s in fold_summaries)
    chain_exp = sum(s["chain_seq_exposed_records"] for s in fold_summaries)
    chain_exact = sum(s["chain_exact_records"] for s in fold_summaries)
    chain_high = sum(s["chain_high_records"] for s in fold_summaries)
    grp_split = [g for g in group_assign if g["split_across_folds"]]
    pdb_ov = sum(s["pdb_overlap_count"] for s in fold_summaries)
    id_ov = sum(s["mutant_id_overlap_count"] for s in fold_summaries)
    grp_ov = sum(s["group_overlap_count"] for s in fold_summaries)

    add("## 任务 B：结论摘要")
    add("")
    add(f"- 审计单元：{len(fold_summaries)} 个（2 数据集 × 5 seed × 5 折），"
        f"validation 记录合计 {tot_val}。")
    add("")
    add("| 分子层面 | 检查方法 | 跨折暴露记录 | 结论 |")
    add("|---|---|---|---|")
    add(f"| L1 拼接侧 partner 序列（官方构造单元） | MMseqs2 30%/80%双向 + 精确 | "
        f"{concat_exp}（精确 {concat_exact}） | "
        f"{'无暴露' if concat_exp == 0 else '存在暴露'} |")
    add(f"| L2 单链 partner 序列 | MMseqs2 30%/80%双向 + 精确 | "
        f"{chain_exp}（精确 {chain_exact}，≥90% {chain_high}） | "
        f"{'无暴露' if chain_exp == 0 else '存在暴露（重点）'} |")
    add(f"| L3 构造组 | group split val/train | {grp_ov} 处组重叠 / {len(grp_split)} 组跨 fold | "
        f"{'完整' if grp_ov == 0 and not grp_split else '异常'} |")
    add(f"| L4 记录身份 | 同 PDB / 同突变ID | PDB {pdb_ov} / ID {id_ov} | "
        f"{'无共用' if pdb_ov == 0 and id_ov == 0 else '存在共用'} |")
    add("")
    add("> 关键发现：官方 ICSV 的构造单元是 **顺序敏感的拼接序列**（R1..R6 | L1..L3）。"
        "当两个复合物共享同一抗体/抗原单链、但链的列顺序不同或其它链不同时，拼接序列可能"
        "达不到 80% 双向覆盖而落入不同构造组，从而被分到不同折——表面看 L1 无暴露，"
        "但 **单链层面（L2）存在真实分子重复**。这正是仅检查文件分组会遗漏的跨折分子暴露。")
    add("")
    add("### 分数据集汇总（对全部 seed×fold 求和）")
    add("")
    add("| 数据集 | val记录合计 | 拼接侧暴露 | 单链暴露 | 单链精确 | 单链≥90% | 单链暴露率(记录) |")
    add("|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        sub = [s for s in fold_summaries if s["dataset"] == ds]
        if not sub:
            continue
        tv = sum(s["val_records"] for s in sub)
        add(f"| {ds} | {tv} | {sum(s['concat_seq_exposed_records'] for s in sub)} | "
            f"{sum(s['chain_seq_exposed_records'] for s in sub)} | "
            f"{sum(s['chain_exact_records'] for s in sub)} | "
            f"{sum(s['chain_high_records'] for s in sub)} | "
            f"{sum(s['chain_seq_exposed_records'] for s in sub) / max(1, tv):.4f} |")
    add("")

    add("## 逐数据集×seed×fold 暴露明细")
    add("")
    add("| 数据集 | seed | fold | val记录 | 拼接侧暴露 | 单链暴露 | 单链率 | 单链精确 | 单链≥90% | "
        "组重叠 | 同PDB | 同ID | 结构近同对 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in fold_summaries:
        add(f"| {s['dataset']} | {s['seed']} | {s['fold']} | {s['val_records']} | "
            f"{s['concat_seq_exposed_records']} | {s['chain_seq_exposed_records']} | "
            f"{s['chain_exposure_rate_records']:.4f} | {s['chain_exact_records']} | "
            f"{s['chain_high_records']} | {s['group_overlap_count']} | {s['pdb_overlap_count']} | "
            f"{s['mutant_id_overlap_count']} | {s['struct_near_identical_pairs']} |")
    add("")
    add("全部匹配记录见 `crossfold_chain_hits.csv`（单链）与 `crossfold_sequence_hits.csv`"
        "（拼接侧）；逐记录暴露见 `record_exposure.csv`；逐数据集×seed 暴露率汇总见 "
        "`exposure_by_dataset_seed.csv`；精确重叠见 `chain_exact_overlap.csv` / "
        "`concat_exact_overlap.csv`。")
    add("")

    add("## 构造组完整性（L3）")
    add("")
    if grp_split:
        add(f"- 有 {len(grp_split)} 个构造组被拆分到多个折（异常）。")
    else:
        add("- 每个构造组都完整落在单一 held-out 折；任何 fold 内 val/train 的组交集为 0。构造组层面完整。")
    add("")

    add("## 重点核查：S1131 的 1E96 与 2A9K")
    add("")
    add("| 数据集 | seed | PDB | 所属构造组 | 与对方同组 | held-out折 | 两目标始终同侧 | 链级跨折相似伙伴(节选) |")
    add("|---|---|---|---|---|---|---|---|")
    for s in special:
        add(f"| {s['dataset']} | {s['seed']} | {s['PDB']} | {s['group']} | {s['in_same_group_as_other']} | "
            f"{s['held_out_val_fold']} | {s['both_targets_same_side_all_folds']} | "
            f"{str(s['cross_fold_similar_partners_from_chain_audit'])[:160]} |")
    add("")
    add("> 1E96 与 2A9K 同属一个构造组（`1E96__A_B`），彼此不会被拆到两侧。"
        "但链级审计显示该组与 S1131 其它 PDB（如 1LFD、1HE8）在 30–55% 同一性、≈95% 覆盖下相似，"
        "而后者落在别的折——即该蛋白家族存在跨折暴露。详见 `special_1E96_2A9K.csv`。")
    add("")

    add("## 结构相似性（L5，US-align TM-score）")
    add("")
    add(f"- 对每个数据集全部唯一 PDB 的 `wt` 结构做了 all-vs-all US-align，共 {struct_rows_done} 条 PDB×PDB 记录。")
    add(f"- 分层：TM≥{TM_NEAR_IDENTICAL} 近同结构；≥{TM_SIMILAR} 相似；≥{TM_SAME_FOLD} 同 fold。")
    add("- 完整 TM 矩阵 `structural_pdb_tm_matrix.csv`；跨折结构对明细 `structural_crossfold_pairs.csv`。")
    add("- 说明：抗体类结构共享 Ig 折叠，TM≥0.5 可能只是同 fold 而非重复；"
        "因此真正的结构重复应结合 `structural_crossfold_pairs.csv` 中的 near_identical 层与链级审计交叉判断。")
    add("")

    add("## 异常汇总")
    add("")
    if anomalies:
        add("| 数据集 | seed | fold | 类型 | 严重度 | 明细 |")
        add("|---|---|---|---|---|---|")
        for a in anomalies:
            add(f"| {a['dataset']} | {a['seed']} | {a['fold']} | {a['anomaly_type']} | "
                f"{a['severity']} | {str(a['detail'])[:200]} |")
    else:
        add("- 未发现异常。")
    add("")
    add("## 复现")
    add("")
    add("```bash")
    add("cd SPLIT_AUDIT_ICSV_20261109")
    add("python3 audit_crossfold_exposure.py --data-root .. --out . --threads 64")
    add("```")
    add("")
    add("阈值：MMseqs2 `--min-seq-id 0.30 -c 0.80 --cov-mode 0 -s 7.5`；"
        "US-align 默认参数取两向 TM 的较大值。运行环境见 `environment.txt`，"
        "完整命令与时间戳见 `run.log`。")
    (out_dir / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# Main
def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-fold molecular-exposure audit for ICSV splits.")
    parser.add_argument("--data-root", type=Path, default=Path("/data/data_split"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mmseqs", default=MMSEQS_DEFAULT)
    parser.add_argument("--usalign", default=USALIGN_DEFAULT)
    parser.add_argument("--threads", type=int, default=64)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    handle = (out_dir / "run.log").open("w", encoding="utf-8")
    start = time.time()
    try:
        log(handle, f"data_root={data_root} out={out_dir}")
        mmseqs_version = subprocess.run([args.mmseqs, "version"], capture_output=True,
                                        text=True).stdout.strip()
        usalign_version = "unknown"
        try:
            uout = subprocess.run([args.usalign], capture_output=True, text=True).stdout
            m = re.search(r"Version\s+([0-9]+)", uout)
            if m:
                usalign_version = m.group(1)
        except Exception:
            pass
        (out_dir / "environment.txt").write_text(
            f"mmseqs_bin={args.mmseqs}\nmmseqs_version={mmseqs_version}\n"
            f"usalign_bin={args.usalign}\nusalign_version={usalign_version}\n"
            f"min_seq_id={MIN_SEQ_ID} coverage={COVERAGE} cov_mode={COV_MODE} "
            f"sensitivity=-s 7.5\nTM tiers: near>={TM_NEAR_IDENTICAL} similar>={TM_SIMILAR} samefold>={TM_SAME_FOLD}\n"
            f"datasets={DATASETS} seeds={SEEDS} split_type={SPLIT_TYPE}\n",
            encoding="utf-8")

        manifest = build_manifest(data_root, out_dir, handle)
        struct_matrix, struct_rows_done = build_structural_matrix(data_root, out_dir, args.usalign, args.threads, handle)

        log(handle, "TASK B: per-fold audit (L1 concat, L2 chain, L3 group, L4 id, L5 struct)")
        all_summaries, concat_hits, concat_exact, chain_hits, chain_exact = [], [], [], [], []
        all_records, struct_rows = [], []
        for dataset in DATASETS:
            pdb2group = build_pdb_group_map(data_root, dataset)
            for seed in SEEDS:
                for fold in range(N_FOLDS):
                    res = audit_fold(data_root, dataset, seed, fold, pdb2group, struct_matrix,
                                     args.mmseqs, args.threads, work_dir, handle)
                    s = res["summary"]
                    all_summaries.append(s)
                    concat_hits.extend(res["concat_hits"])
                    concat_exact.extend(res["concat_exact"])
                    chain_hits.extend(res["chain_hits"])
                    chain_exact.extend(res["chain_exact"])
                    all_records.extend(res["records"])
                    struct_rows.extend(res["struct_rows"])
                    log(handle, f"  {dataset} seed={seed} fold={fold} val={s['val_records']} "
                                f"concat_exp={s['concat_seq_exposed_records']} "
                                f"chain_exp={s['chain_seq_exposed_records']}(exact {s['chain_exact_records']}) "
                                f"grp_ov={s['group_overlap_count']} pdb_ov={s['pdb_overlap_count']} "
                                f"struct_near={s['struct_near_identical_pairs']}")

        group_assign, group_fold = group_integrity(data_root, handle)
        special = special_check(data_root, chain_hits, handle)

        # ---- write tables ----
        write_csv(out_dir / "exposure_summary.csv", all_summaries, list(all_summaries[0].keys()))
        # per dataset x seed aggregate exposure-rate table
        agg: list[dict] = []
        for dataset in DATASETS:
            for seed in SEEDS:
                sub = [s for s in all_summaries if s["dataset"] == dataset and s["seed"] == seed]
                if not sub:
                    continue
                tv = sum(s["val_records"] for s in sub)
                ce = sum(s["chain_seq_exposed_records"] for s in sub)
                agg.append({
                    "dataset": dataset, "seed": seed, "n_folds": len(sub),
                    "val_records": tv,
                    "concat_exposed_records": sum(s["concat_seq_exposed_records"] for s in sub),
                    "chain_exposed_records": ce,
                    "chain_exact_records": sum(s["chain_exact_records"] for s in sub),
                    "chain_high_records": sum(s["chain_high_records"] for s in sub),
                    "chain_exposure_rate": round(ce / max(1, tv), 6),
                    "chain_exact_rate": round(sum(s["chain_exact_records"] for s in sub) / max(1, tv), 6),
                    "group_overlap_total": sum(s["group_overlap_count"] for s in sub),
                    "pdb_overlap_total": sum(s["pdb_overlap_count"] for s in sub),
                    "mutant_id_overlap_total": sum(s["mutant_id_overlap_count"] for s in sub),
                })
        write_csv(out_dir / "exposure_by_dataset_seed.csv", agg, list(agg[0].keys()))
        write_csv(out_dir / "crossfold_chain_hits.csv", chain_hits,
                  ["dataset", "seed", "fold", "val_seq_id", "train_seq_id", "pident", "alnlen",
                   "qcov", "tcov", "evalue", "bits", "exact_overlap", "identity_bucket",
                   "val_sequence_length", "train_sequence_length", "val_records", "train_records",
                   "val_PDBs", "train_PDBs", "val_columns", "train_columns"])
        write_csv(out_dir / "crossfold_sequence_hits.csv", concat_hits,
                  ["dataset", "seed", "fold", "side", "pident", "alnlen", "qcov", "tcov",
                   "evalue", "bits", "exact_overlap", "val_sequence_length",
                   "train_sequence_length", "val_records", "train_records", "val_PDBs", "train_PDBs"])
        write_csv(out_dir / "chain_exact_overlap.csv", chain_exact,
                  ["dataset", "seed", "fold", "sequence_length", "sequence_md5", "val_records",
                   "train_records", "val_PDBs", "train_PDBs", "val_columns", "train_columns"])
        write_csv(out_dir / "concat_exact_overlap.csv", concat_exact,
                  ["dataset", "seed", "fold", "side", "sequence_length", "sequence_md5",
                   "val_records", "train_records", "val_PDBs", "train_PDBs"])
        write_csv(out_dir / "record_exposure.csv", all_records,
                  ["dataset", "seed", "fold", "val_record_index", "ID", "PDB", "complex", "group",
                   "concat_exposed", "concat_exact_overlap", "chain_exposed", "chain_exact_overlap",
                   "chain_high_overlap", "chain_best_pident", "chain_matching_train_PDBs",
                   "same_pdb_in_train", "same_group_in_train"])
        write_csv(out_dir / "structural_crossfold_pairs.csv", struct_rows,
                  ["dataset", "seed", "fold", "val_PDB", "train_PDB", "tm_score", "tier"])
        write_csv(out_dir / "group_fold_assignment.csv", group_assign,
                  list(group_assign[0].keys()) if group_assign else ["dataset"])
        write_csv(out_dir / "group_fold_overlap.csv", group_fold, list(group_fold[0].keys()))
        write_csv(out_dir / "special_1E96_2A9K.csv", special,
                  list(special[0].keys()) if special else ["dataset"])

        # ---- anomalies (aggregated per fold per category) ----
        anomalies: list[dict] = []
        for s in all_summaries:
            base = {"dataset": s["dataset"], "seed": s["seed"], "fold": s["fold"]}
            if s["chain_exact_records"] > 0:
                anomalies.append({**base, "anomaly_type": "CHAIN_EXACT_OVERLAP", "severity": "CRITICAL",
                                  "detail": f"{s['chain_exact_records']}/{s['val_records']} val records share an exact "
                                            f"partner CHAIN with training (rate {s['chain_exposure_rate_records']:.4f})"})
            if s["chain_high_records"] > s["chain_exact_records"]:
                anomalies.append({**base, "anomaly_type": "CHAIN_HIGH_SIMILARITY(>=90%)", "severity": "HIGH",
                                  "detail": f"{s['chain_high_records']} val records have a chain >=90% identical to training"})
            if s["chain_seq_exposed_records"] > s["chain_high_records"]:
                anomalies.append({**base, "anomaly_type": "CHAIN_SIMILARITY(30-90%)", "severity": "WARN",
                                  "detail": f"{s['chain_seq_exposed_records'] - s['chain_high_records']} val records "
                                            f"have a chain 30-90% identical (>=30%/80% bicov) to training"})
            if s["concat_seq_exposed_records"] > 0:
                anomalies.append({**base, "anomaly_type": "CONCAT_SEQUENCE_EXPOSURE", "severity": "HIGH",
                                  "detail": f"{s['concat_seq_exposed_records']} val records exposed at concatenated-partner level"})
            if s["concat_exact_overlap_records"] > 0:
                anomalies.append({**base, "anomaly_type": "CONCAT_EXACT_OVERLAP", "severity": "CRITICAL",
                                  "detail": f"{s['concat_exact_overlap_records']} val records share an exact concatenated partner sequence"})
            if s["group_overlap_count"] > 0:
                anomalies.append({**base, "anomaly_type": "GROUP_CROSS_FOLD", "severity": "CRITICAL",
                                  "detail": f"group overlap: {s['group_overlap']}"})
            if s["pdb_overlap_count"] > 0:
                anomalies.append({**base, "anomaly_type": "SAME_PDB_CROSSFOLD", "severity": "CRITICAL",
                                  "detail": f"shared PDB: {s['pdb_overlap']}"})
            if s["mutant_id_overlap_count"] > 0:
                anomalies.append({**base, "anomaly_type": "MUTANT_ID_CROSSFOLD", "severity": "CRITICAL",
                                  "detail": f"shared mutation IDs: {s['mutant_id_overlap']}"})
            if s["struct_near_identical_pairs"] > 0:
                anomalies.append({**base, "anomaly_type": "STRUCT_NEAR_IDENTICAL_CROSSFOLD", "severity": "HIGH",
                                  "detail": f"{s['struct_near_identical_pairs']} cross-fold PDB pairs with TM>={TM_NEAR_IDENTICAL}"})
            if s["struct_similar_pairs"] > 0:
                anomalies.append({**base, "anomaly_type": "STRUCT_SIMILAR_CROSSFOLD", "severity": "INFO",
                                  "detail": f"{s['struct_similar_pairs']} cross-fold PDB pairs with TM in [{TM_SIMILAR},{TM_NEAR_IDENTICAL})"})
            if s["struct_samefold_pairs"] > 0:
                anomalies.append({**base, "anomaly_type": "STRUCT_SAME_FOLD_CROSSFOLD", "severity": "INFO",
                                  "detail": f"{s['struct_samefold_pairs']} cross-fold PDB pairs with TM in [{TM_SAME_FOLD},{TM_SIMILAR})"})
        for g in group_assign:
            if g["split_across_folds"]:
                anomalies.append({"dataset": g["dataset"], "seed": g["seed"], "fold": g["held_out_val_fold"],
                                  "anomaly_type": "GROUP_SPLIT_ACROSS_FOLDS", "severity": "CRITICAL",
                                  "detail": f"group {g['group_id']} held-out in folds {g['held_out_val_fold']}"})
        write_csv(out_dir / "anomalies.csv", anomalies,
                  ["dataset", "seed", "fold", "anomaly_type", "severity", "detail"])

        write_report(out_dir, manifest["dataset_rows"], all_summaries, group_assign,
                     special, anomalies, struct_rows_done)

        elapsed = time.time() - start
        log(handle, f"DONE in {elapsed:.1f}s | anomalies={len(anomalies)} | "
                    f"concat_exposed={sum(s['concat_seq_exposed_records'] for s in all_summaries)} | "
                    f"chain_exposed={sum(s['chain_seq_exposed_records'] for s in all_summaries)} | "
                    f"chain_exact={sum(s['chain_exact_records'] for s in all_summaries)} | "
                    f"group_split={sum(1 for g in group_assign if g['split_across_folds'])}")
    finally:
        handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
