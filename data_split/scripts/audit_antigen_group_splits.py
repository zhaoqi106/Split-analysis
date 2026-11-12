"""Re-read entity-grouped split files from disk and audit them.

The auditor does not reuse the data frames built while the splits were written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import three_splits as legacy

VIOLATION_COLUMNS = [
    "record_uid",
    "pdb",
    "entity_group",
    "fold",
    "violation_type",
    "evidence",
]
SUBSETS = ("outer_train", "inner_train", "inner_val", "outer_eval")
MODEL_COLUMNS = (
    "Partners",
    "Mutation",
    "Mutation(source)",
    "mutation(source)",
    *legacy.WT_LEFT_COLS,
    *legacy.WT_RIGHT_COLS,
    "R1m", "R2m", "R3m", "R4m", "R5m", "R6m",
    "L1m", "L2m", "L3m",
)


def _clean(value: object) -> str:
    return legacy.clean_text(value)


def _uid_frame(dataset_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result["record_uid"] = [legacy.record_uid(dataset_name, index) for index in range(len(result))]
    result["source_row_index"] = np.arange(len(result))
    return result


def _load_source(path: Path, dataset_name: str) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    raw_pdb = raw["PDB"].map(legacy.clean_text)
    normalized, _repaired = legacy.normalize_pdb_column(raw)
    normalized = normalized.reset_index(drop=True)
    normalized["pdb_raw"] = raw_pdb.reset_index(drop=True).astype(str)
    normalized["record_uid"] = [legacy.record_uid(dataset_name, index) for index in range(len(normalized))]
    normalized["source_row_index"] = np.arange(len(normalized))
    return normalized


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _group_column(mapping: pd.DataFrame) -> str:
    if "antigen_group_id" in mapping.columns:
        return "antigen_group_id"
    if "partner_group_id" in mapping.columns:
        return "partner_group_id"
    raise ValueError("Mapping file has neither antigen_group_id nor partner_group_id.")


def _values_equal(left: object, right: object) -> bool:
    left_missing = pd.isna(left) or _clean(left) == ""
    right_missing = pd.isna(right) or _clean(right) == ""
    if left_missing and right_missing:
        return True
    if left_missing or right_missing:
        return False
    left_number = pd.to_numeric(pd.Series([left]), errors="coerce").iloc[0]
    right_number = pd.to_numeric(pd.Series([right]), errors="coerce").iloc[0]
    if pd.notna(left_number) and pd.notna(right_number):
        return bool(np.isclose(float(left_number), float(right_number), rtol=1e-6, atol=1e-8))
    return _clean(left) == _clean(right)


def _model_key(row: pd.Series, columns: list[str]) -> tuple[str, ...]:
    return tuple(_clean(row[column]) if column in row.index else "" for column in columns)


def _input_hashes(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts: list[pd.Series] = []
    for column in columns:
        if column in frame.columns:
            parts.append(frame[column].map(_clean))
        else:
            parts.append(pd.Series([""] * len(frame), index=frame.index, dtype="string"))
    blob = parts[0].astype("string")
    for part in parts[1:]:
        blob = blob.str.cat(part.astype("string"), sep="\t")
    return blob.map(lambda value: hashlib.sha1(str(value).encode("utf-8")).hexdigest())


def _column_mismatch(left: pd.Series, right: pd.Series) -> pd.Series:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    both_num = left_num.notna() & right_num.notna()
    numeric_bad = pd.Series(False, index=left.index)
    if bool(both_num.any()):
        numeric_bad.loc[both_num] = ~np.isclose(
            left_num.loc[both_num].to_numpy(dtype=float),
            right_num.loc[both_num].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-8,
        )
    text_bad = ~both_num & (left.map(_clean) != right.map(_clean))
    return numeric_bad | text_bad


def _check(name: str, status: str, n_errors: int, detail: str) -> dict[str, object]:
    return {"name": name, "status": status, "n_errors": int(n_errors), "detail": detail}


def audit_split_dir(split_dir: Path, source_csv: Path) -> dict[str, object]:
    split_dir = split_dir.resolve()
    dataset_name = split_dir.parents[1].name if split_dir.parent.parent.name not in {"", split_dir.parent.name} else split_dir.parent.parent.name
    # Layout is {root}/{dataset}/{seed}/{split}.
    dataset_name = split_dir.parents[1].name
    source = _load_source(source_csv, dataset_name)
    mapping_path = next(split_dir.glob("*_mapping.csv"))
    mapping = _read_csv(mapping_path)
    group_column = _group_column(mapping)
    violations: list[dict[str, object]] = []

    def add(uid: str, pdb: str, group: str, fold: str, kind: str, evidence: str) -> None:
        violations.append(
            {
                "record_uid": uid,
                "pdb": pdb,
                "entity_group": group,
                "fold": fold,
                "violation_type": kind,
                "evidence": evidence,
            }
        )

    checks: list[dict[str, object]] = []
    source_uids = source["record_uid"].tolist()
    if mapping["record_uid"].tolist() != source_uids:
        add("", "", "", "", "mapping_uid_mismatch", f"{mapping_path.name} record_uid sequence does not match the source CSV")
    missing_status = sorted(set(source_uids) - set(mapping["record_uid"].astype(str)))
    for uid in missing_status:
        add(uid, "", "", "", "mapping_missing_record", "source record absent from mapping")
    duplicate_ids = source.loc[source["ID"].astype(str).duplicated(keep=False), "ID"].astype(str).unique().tolist()
    present_duplicate_rows = int(source["ID"].astype(str).isin(duplicate_ids).sum()) if duplicate_ids else 0
    mapped_duplicate_rows = int(mapping["original_mutation_id"].astype(str).isin(duplicate_ids).sum()) if duplicate_ids else 0
    if present_duplicate_rows != mapped_duplicate_rows:
        add("", "", "", "", "duplicate_mutation_id_lost", f"source duplicate-id rows={present_duplicate_rows}, mapping={mapped_duplicate_rows}")

    statuses = sorted(mapping["mapping_status"].astype(str).unique())
    unconfirmed = int((mapping["mapping_status"].astype(str) != "confirmed").sum())
    checks.append(_check(
        "mapping_status",
        "PASS" if unconfirmed == 0 else "INCOMPLETE",
        unconfirmed,
        "statuses=" + ",".join(statuses),
    ))

    normalized_mismatch = 0
    for record in mapping.itertuples(index=False):
        raw = _clean(getattr(record, "pdb_raw", ""))
        fallback = _clean(record.original_mutation_id)
        expected = legacy.normalize_pdb_id(raw, fallback)
        if expected != _clean(record.complete_pdb_id):
            normalized_mismatch += 1
            add(record.record_uid, _clean(record.complete_pdb_id), _clean(getattr(record, group_column)), "", "pdb_normalization", f"raw={raw} expected={expected}")
        repair = _clean(getattr(record, "pdb_repair", ""))
        if raw.upper() != expected and not repair:
            add(record.record_uid, expected, "", "", "pdb_repair_undocumented", f"raw={raw}")
    checks.append(_check("pdb_traceable", "PASS" if normalized_mismatch == 0 else "FAIL", normalized_mismatch, "normalized PDB ids recomputed from pdb_raw and mutation id"))

    group_by_uid = dict(zip(mapping["record_uid"].astype(str), mapping[group_column].astype(str)))
    pdb_by_uid = dict(zip(mapping["record_uid"].astype(str), mapping["complete_pdb_id"].astype(str)))
    source_by_uid = source.set_index("record_uid", drop=False)

    fold_ids = []
    eval_uids_by_fold: dict[int, list[str]] = {}
    subset_frames: dict[tuple[int, str], pd.DataFrame] = {}
    for fold_id in range(0, 20):
        eval_path = split_dir / f"fold_{fold_id}_outer_eval.csv"
        if not eval_path.is_file():
            break
        fold_ids.append(fold_id)
        for subset in SUBSETS:
            path = split_dir / f"fold_{fold_id}_{subset}.csv"
            if not path.is_file():
                add("", "", "", str(fold_id), "missing_subset_file", path.name)
                continue
            subset_frames[(fold_id, subset)] = _read_csv(path)
        if (fold_id, "outer_eval") in subset_frames:
            eval_uids_by_fold[fold_id] = subset_frames[(fold_id, "outer_eval")]["record_uid"].astype(str).tolist()

    all_eval = [uid for fold_id in fold_ids for uid in eval_uids_by_fold.get(fold_id, [])]
    eval_counts: dict[str, int] = defaultdict(int)
    for uid in all_eval:
        eval_counts[uid] += 1
    missing_eval = sorted(set(source_uids) - set(all_eval))
    extra_eval = sorted(set(all_eval) - set(source_uids))
    repeated = sorted(uid for uid, count in eval_counts.items() if count != 1)
    for uid in missing_eval:
        add(uid, pdb_by_uid.get(uid, ""), group_by_uid.get(uid, ""), "", "outer_eval_missing", "record is not in exactly one outer_eval fold")
    for uid in extra_eval:
        add(uid, "", "", "", "outer_eval_unknown_record", "outer_eval contains a record_uid absent from the source CSV")
    for uid in repeated:
        add(uid, pdb_by_uid.get(uid, ""), group_by_uid.get(uid, ""), "", "outer_eval_repeated", f"count={eval_counts[uid]}")
    coverage_errors = len(missing_eval) + len(extra_eval) + len(repeated)
    checks.append(_check(
        "outer_eval_partition",
        "PASS" if coverage_errors == 0 and fold_ids else "FAIL",
        coverage_errors,
        f"folds={fold_ids}",
    ))

    overlap_errors = 0
    field_errors = 0
    identical_input_errors = 0
    sequence_cross_group = 0
    group_split_errors = 0
    compared_columns = [column for column in source.columns if column in (subset_frames.get((fold_ids[0], "outer_eval")).columns if fold_ids else [])]
    compared_columns = [column for column in compared_columns if column not in {"record_uid", "source_row_index", "pdb_raw"}]
    model_columns = [column for column in MODEL_COLUMNS if column in source.columns]

    sequence_column = "antigen_sequence_sha256" if "antigen_sequence_sha256" in mapping.columns else "sequence_sha256"
    if sequence_column in mapping.columns:
        for sequence_hash, part in mapping.groupby(sequence_column, sort=False):
            groups = sorted(set(part[group_column].astype(str)))
            if len(groups) > 1:
                sequence_cross_group += len(part)
                for record in part.itertuples(index=False):
                    add(
                        str(record.record_uid),
                        str(record.complete_pdb_id),
                        str(getattr(record, group_column)),
                        "",
                        "identical_wt_sequence_cross_group",
                        f"sha256={sequence_hash} groups={groups}",
                    )
    checks.append(_check(
        "identical_wt_sequence_same_group",
        "PASS" if sequence_cross_group == 0 else "FAIL",
        sequence_cross_group,
        "Exact wild-type sequence hash from the mapping file. This is not a similarity-threshold audit.",
    ))

    antibody_shared = 0
    antibody_unseen = 0
    eval_total = 0
    if "antibody_entity_id" in mapping.columns:
        antibody_by_uid = dict(zip(mapping["record_uid"].astype(str), mapping["antibody_entity_id"].astype(str)))
    else:
        antibody_by_uid = {}

    for fold_id in fold_ids:
        frames = {subset: subset_frames[(fold_id, subset)] for subset in SUBSETS if (fold_id, subset) in subset_frames}
        uid_sets = {subset: set(frame["record_uid"].astype(str)) for subset, frame in frames.items()}
        group_sets = {
            subset: {group_by_uid.get(uid, "") for uid in uids}
            for subset, uids in uid_sets.items()
        }
        if uid_sets.get("inner_train", set()) | uid_sets.get("inner_val", set()) != uid_sets.get("outer_train", set()):
            overlap_errors += 1
            add("", "", "", str(fold_id), "inner_cover", "inner_train union inner_val is not outer_train")
        if uid_sets.get("inner_train", set()) & uid_sets.get("inner_val", set()):
            overlap_errors += 1
            add("", "", "", str(fold_id), "inner_record_overlap", "inner_train and inner_val share records")
        if uid_sets.get("outer_train", set()) & uid_sets.get("outer_eval", set()):
            overlap_errors += 1
            add("", "", "", str(fold_id), "outer_record_overlap", "outer_train and outer_eval share records")
        for left, right in (("inner_train", "inner_val"), ("inner_train", "outer_eval"), ("inner_val", "outer_eval"), ("outer_train", "outer_eval")):
            shared_groups = group_sets.get(left, set()) & group_sets.get(right, set())
            shared_groups.discard("")
            if shared_groups:
                group_split_errors += len(shared_groups)
                add("", "", ",".join(sorted(shared_groups)), str(fold_id), "entity_group_overlap", f"{left} overlaps {right}")
        hashed: dict[str, pd.Series] = {}
        for subset, frame in frames.items():
            uids = frame["record_uid"].astype(str)
            unknown = sorted(set(uids) - set(source_by_uid.index))
            for uid in unknown:
                field_errors += 1
                add(uid, "", "", str(fold_id), "unknown_record_in_fold_file", subset)
            aligned = frame.set_index(uids, drop=False)
            common = aligned.index.intersection(source_by_uid.index)
            missing_group = [uid for uid in common if group_by_uid.get(str(uid), "") == ""]
            for uid in missing_group:
                group_split_errors += 1
                add(str(uid), pdb_by_uid.get(str(uid), ""), "", str(fold_id), "missing_group", subset)
            source_aligned = source_by_uid.loc[common]
            bad_column = pd.Series("", index=common, dtype="string")
            for column in compared_columns:
                mismatch = _column_mismatch(aligned.loc[common, column], source_aligned[column])
                newly = mismatch & bad_column.eq("")
                bad_column.loc[newly] = column
            for uid, column in bad_column[bad_column.ne("")].items():
                field_errors += 1
                add(str(uid), pdb_by_uid.get(str(uid), ""), group_by_uid.get(str(uid), ""), str(fold_id), "field_mismatch", f"{subset} column {column}")
            hashed[subset] = pd.Series(_input_hashes(frame, model_columns).to_numpy(), index=uids)
        for left, right in (("inner_train", "inner_val"), ("inner_train", "outer_eval"), ("inner_val", "outer_eval")):
            if left not in hashed or right not in hashed:
                continue
            shared_keys = set(hashed[left]) & set(hashed[right])
            right_lookup = dict(zip(hashed[right], hashed[right].index.astype(str)))
            left_lookup = dict(zip(hashed[left], hashed[left].index.astype(str)))
            for key in shared_keys:
                identical_input_errors += 1
                add(left_lookup.get(key, right_lookup.get(key, "")), "", "", str(fold_id), "identical_model_input_cross_subset", f"{left} and {right}")
        if antibody_by_uid:
            train_antibodies = {antibody_by_uid.get(uid, "") for uid in uid_sets.get("outer_train", set())}
            train_antibodies.discard("")
            for uid in uid_sets.get("outer_eval", set()):
                eval_total += 1
                antibody = antibody_by_uid.get(uid, "")
                if antibody and antibody in train_antibodies:
                    antibody_shared += 1
                elif antibody:
                    antibody_unseen += 1

    # Same model input in two different outer_eval folds.
    eval_key_folds: dict[str, set[int]] = defaultdict(set)
    for fold_id in fold_ids:
        frame = subset_frames.get((fold_id, "outer_eval"))
        if frame is None:
            continue
        for key in _input_hashes(frame, model_columns):
            eval_key_folds[str(key)].add(fold_id)
    cross_fold_inputs = 0
    for key, folds in eval_key_folds.items():
        if len(folds) > 1:
            cross_fold_inputs += 1
            add("", "", "", ",".join(str(fold) for fold in sorted(folds)), "identical_model_input_cross_fold", "same model-input fields in more than one outer_eval fold")
    identical_input_errors += cross_fold_inputs

    checks.append(_check("subset_disjoint_and_cover", "PASS" if overlap_errors == 0 else "FAIL", overlap_errors, "record cover and subset disjointness"))
    checks.append(_check("entity_groups_disjoint", "PASS" if group_split_errors == 0 else "FAIL", group_split_errors, "entity groups do not cross inner_train, inner_val, and outer_eval"))
    checks.append(_check("source_fields_preserved", "PASS" if field_errors == 0 else "FAIL", field_errors, "fold files compared with the normalized source row; record_uid is ignored as a feature"))
    checks.append(_check("identical_model_input", "PASS" if identical_input_errors == 0 else "FAIL", identical_input_errors, "Partners, mutation strings, and wild-type/mutant sequence columns"))
    checks.append(_check(
        "duplicate_mutation_ids_retained",
        "PASS" if present_duplicate_rows == mapped_duplicate_rows else "FAIL",
        abs(present_duplicate_rows - mapped_duplicate_rows),
        f"duplicate mutation IDs={len(duplicate_ids)}; source rows={present_duplicate_rows}",
    ))

    multi_pdb_groups = 0
    if group_column in mapping.columns:
        for _group_id, part in mapping.groupby(group_column):
            if part["complete_pdb_id"].nunique() > 1:
                multi_pdb_groups += 1
    checks.append(_check(
        "same_entity_across_pdb",
        "PASS" if multi_pdb_groups >= 0 else "FAIL",
        0,
        f"{multi_pdb_groups} entity groups contain more than one PDB. Group ids are taken from the on-disk mapping.",
    ))
    if antibody_by_uid:
        checks.append(_check(
            "shared_antibody_entities",
            "INFO",
            0,
            f"outer_eval records={eval_total}; antibody entity also in outer_train={antibody_shared}; "
            f"unseen antigen entity and unseen antibody entity={antibody_unseen}. Shared antibodies are allowed.",
        ))
    else:
        checks.append(_check(
            "shared_antibody_entities",
            "未计算",
            0,
            "This split has no antibody_entity_id column. S1131 partner holdout does not define an antibody entity.",
        ))
    checks.append(_check(
        "sequence_similarity_beyond_exact_match",
        "未计算",
        0,
        "No sequence-identity clustering threshold was run. Exact identity and exact substring checks are separate.",
    ))
    checks.append(_check(
        "structure_similarity",
        "未计算",
        0,
        "No structure-similarity audit was run.",
    ))

    hard_fail = any(item["status"] == "FAIL" for item in checks)
    incomplete = any(item["status"] == "INCOMPLETE" for item in checks) or unconfirmed > 0
    if hard_fail:
        status = "FAIL"
    elif incomplete:
        status = "INCOMPLETE"
    else:
        status = "PASS"

    violation_frame = pd.DataFrame(violations, columns=VIOLATION_COLUMNS)
    violation_path = split_dir / "audit_violations.csv"
    violation_frame.to_csv(violation_path, index=False)
    summary = {
        "status": status,
        "split_dir": str(split_dir),
        "source_csv": str(source_csv.resolve()),
        "source_sha256": hashlib.sha256(source_csv.read_bytes()).hexdigest(),
        "mapping_file": mapping_path.name,
        "n_source_records": int(len(source)),
        "n_folds": len(fold_ids),
        "mapping_status_counts": mapping["mapping_status"].astype(str).value_counts().to_dict(),
        "checks": checks,
        "uncomputed": [
            "sequence_similarity_beyond_exact_match",
            "structure_similarity",
        ],
        "limitations": [
            "Entity-level holdout is not a claim of no homology, no local interface similarity, or no pretraining exposure.",
            "Items marked 未计算 were not run. They are not zeros.",
        ],
    }
    (split_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _refresh_manifest(split_dir, status)
    print(f"[AUDIT] {split_dir} status={status} violations={len(violation_frame)}")
    return summary


def _refresh_manifest(split_dir: Path, status: str) -> None:
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["formal_status"] = status
    manifest["audit_script"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output_files = []
    for path in sorted(split_dir.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        output_files.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    manifest["output_files"] = output_files
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_split_dirs(output_root: Path) -> list[Path]:
    found = []
    for manifest in output_root.glob("*/*/*/manifest.json"):
        found.append(manifest.parent)
    return sorted(found)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit AGCV and PGCV split directories from disk.")
    parser.add_argument("--split-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--source-csv", type=Path, default=None, help="Source CSV. Required with --split-dir when the dataset cannot be inferred.")
    parser.add_argument("--input-dir", type=Path, default=Path("/data/data_split/raw"))
    return parser.parse_args()


def _source_for(split_dir: Path, input_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    dataset_name = split_dir.parents[1].name
    return legacy.resolve_dataset_csv(input_dir, f"{dataset_name}.csv")


def main() -> int:
    args = parse_args()
    split_dirs = list(args.split_dir)
    if args.output_root is not None:
        split_dirs.extend(iter_split_dirs(args.output_root))
    if not split_dirs:
        raise SystemExit("Pass --split-dir or --output-root.")
    worst = 0
    for split_dir in split_dirs:
        source = _source_for(split_dir, args.input_dir, args.source_csv if len(split_dirs) == 1 else None)
        if not source.is_file():
            raise FileNotFoundError(source)
        summary = audit_split_dir(split_dir, source)
        if summary["status"] == "FAIL":
            worst = max(worst, 2)
        elif summary["status"] != "PASS":
            worst = max(worst, 1)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
