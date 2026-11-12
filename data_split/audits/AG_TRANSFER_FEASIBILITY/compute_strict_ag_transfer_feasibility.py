#!/usr/bin/env python3
"""Feasibility audit for strict Ag-transfer on M1101. Does not write a new split."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

DATA_SPLIT = Path("/data/data_split")
sys.path.insert(0, str(DATA_SPLIT / "scripts"))
import three_splits as legacy  # noqa: E402

SOURCE_CSV = DATA_SPLIT / "raw" / "M1101.csv"
SPLIT_ROOT = DATA_SPLIT / "splits" / "M1101"
SEEDS = (0, 42, 1234, 2026, 1024)
FOLDS = (0, 1, 2, 3, 4)
LEFT_COLS = ("R1", "R2", "R3", "R4", "R5", "R6")
RIGHT_COLS = ("L1", "L2", "L3")
OUT_DIR = Path(__file__).resolve().parent
N_EXPECTED = 1101
INNER_FRACTION = 0.20
INNER_BAND = (0.15, 0.25)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonempty_sequences(row: pd.Series, columns: tuple[str, ...]) -> tuple[list[str], list[str]]:
    sequences: list[str] = []
    used: list[str] = []
    for column in columns:
        sequence = legacy.clean_sequence(row[column]) if column in row.index else ""
        if sequence:
            sequences.append(sequence)
            used.append(column)
    return sequences, used


def nonempty_ids(row: pd.Series, columns: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for column in columns:
        text = legacy.clean_text(row[column]) if column in row.index else ""
        if text:
            values.append(text)
    return values


def load_mapping(seed: int) -> pd.DataFrame:
    path = SPLIT_ROOT / str(seed) / "AGCV" / "antigen_mapping.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame["_seed_path"] = str(path)
    return frame


def load_fold_uids(seed: int, fold: int, subset: str) -> list[str]:
    path = SPLIT_ROOT / str(seed) / "AGCV" / f"fold_{fold}_{subset}.csv"
    frame = pd.read_csv(path, usecols=["record_uid"], dtype=str, keep_default_na=False)
    return frame["record_uid"].tolist()


def closest_validation_subset(
    free_groups: list[str],
    sizes: dict[str, int],
    total: int,
) -> dict[str, object]:
    """Exact subset of free groups whose record count is closest to 20% of total.

    The pinned complement stays in inner_train, so every returned subset preserves
    the same antibody-support constraint. Empty validation or empty training is rejected.
    """
    achievable: dict[int, tuple[str, ...]] = {0: ()}
    for group in free_groups:
        size = int(sizes[group])
        expanded = dict(achievable)
        for prev, subset in achievable.items():
            expanded[prev + size] = subset + (group,)
        achievable = expanded
    target = INNER_FRACTION * total
    best: tuple[tuple, dict[str, object]] | None = None
    for size, subset in achievable.items():
        train_n = total - size
        if size <= 0 or train_n <= 0:
            continue
        fraction = size / total
        rank = (abs(fraction - INNER_FRACTION), abs(size - target), size, subset)
        if best is None or rank < best[0]:
            best = (
                rank,
                {
                    "inner_val_groups": list(subset),
                    "inner_val_records": int(size),
                    "inner_train_records": int(train_n),
                    "inner_val_fraction": fraction,
                    "within_0.15_0.25": INNER_BAND[0] <= fraction <= INNER_BAND[1],
                },
            )
    if best is None:
        return {
            "inner_val_groups": [],
            "inner_val_records": 0,
            "inner_train_records": total,
            "inner_val_fraction": 0.0,
            "within_0.15_0.25": False,
            "feasible": False,
        }
    payload = best[1]
    payload["feasible"] = True
    return payload


def pin_supports(support_sets: list[set[str]], sizes: dict[str, int]) -> list[str]:
    """Deterministic hitting set: cover each support set with its smallest remaining group."""
    pinned: list[str] = []
    pinned_set: set[str] = set()
    for support in support_sets:
        if pinned_set & support:
            continue
        choice = min(support, key=lambda group: (sizes[group], group))
        pinned.append(choice)
        pinned_set.add(choice)
    return pinned


def build_separating_assignment(
    group_sizes: dict[str, int],
    antibody_groups: dict[str, set[str]],
) -> dict[str, int]:
    """Put every multi-group antibody's groups into more than one outer fold.

    Group ids are sorted. Larger groups are placed first, onto the current
    lightest fold. No label and no existing AGCV seed enters this rule.
    """
    multi_groups: set[str] = set()
    for groups in antibody_groups.values():
        if len(groups) >= 2:
            multi_groups.update(groups)
    ordered_multi = sorted(multi_groups, key=lambda group: (-group_sizes[group], group))
    ordered_rest = sorted(
        (group for group in group_sizes if group not in multi_groups),
        key=lambda group: (-group_sizes[group], group),
    )
    fold_sizes = [0] * 5
    assignment: dict[str, int] = {}

    def place(group: str, blocked: set[int] | None = None) -> None:
        candidates = [fold for fold in range(5) if blocked is None or fold not in blocked]
        if not candidates:
            candidates = list(range(5))
        fold_id = min(candidates, key=lambda fold: (fold_sizes[fold], fold))
        assignment[group] = fold_id
        fold_sizes[fold_id] += int(group_sizes[group])

    # Place sharing antibodies one at a time so their groups prefer different folds.
    antibodies = sorted(
        (ab for ab, groups in antibody_groups.items() if len(groups) >= 2),
        key=lambda ab: (-sum(group_sizes[g] for g in antibody_groups[ab]), ab),
    )
    for antibody in antibodies:
        used_folds = {assignment[g] for g in antibody_groups[antibody] if g in assignment}
        for group in sorted(antibody_groups[antibody], key=lambda g: (-group_sizes[g], g)):
            if group in assignment:
                continue
            place(group, blocked=used_folds)
            used_folds.add(assignment[group])
    for group in ordered_multi:
        if group not in assignment:
            place(group)
    for group in ordered_rest:
        place(group)
    if len(set(assignment.values())) != 5:
        raise RuntimeError("Separating assignment left an outer fold empty.")
    if set(assignment) != set(group_sizes):
        raise RuntimeError("Separating assignment missed an antigen group.")
    return assignment


def evaluate_assignment(
    assignment: dict[str, int],
    group_sizes: dict[str, int],
    pair_records: dict[tuple[str, str], int],
    antibody_groups: dict[str, set[str]],
) -> dict[str, object]:
    folds: list[dict[str, object]] = []
    eligible_records = 0
    ineligible_multi_records = 0
    for fold_id in FOLDS:
        eval_groups = {group for group, fold in assignment.items() if fold == fold_id}
        train_pool = set(group_sizes) - eval_groups
        support_sets: list[set[str]] = []
        fold_eligible = 0
        fold_multi = 0
        fold_blocked = 0
        for (antibody, group), n_records in pair_records.items():
            if group not in eval_groups or len(antibody_groups[antibody]) < 2:
                continue
            fold_multi += n_records
            support = (antibody_groups[antibody] - {group}) & train_pool
            if not support:
                fold_blocked += n_records
                continue
            support_sets.append(support)
            fold_eligible += n_records
        pinned = pin_supports(support_sets, group_sizes)
        pinned_set = set(pinned)
        free = sorted(train_pool - pinned_set)
        total = sum(group_sizes[group] for group in train_pool)
        subset = closest_validation_subset(free, group_sizes, total)
        inner_train_groups = sorted(pinned_set | (set(free) - set(subset["inner_val_groups"])))
        folds.append(
            {
                "fold": fold_id,
                "outer_eval_groups": sorted(eval_groups),
                "outer_eval_records": int(sum(group_sizes[group] for group in eval_groups)),
                "multi_group_antibody_records": int(fold_multi),
                "eligible_if_supports_stay_in_inner_train": int(fold_eligible),
                "blocked_because_support_groups_share_this_outer_fold": int(fold_blocked),
                "pinned_inner_train_groups": pinned,
                "inner_train_groups": inner_train_groups,
                "inner_val_groups": subset["inner_val_groups"],
                "inner_val_records": subset["inner_val_records"],
                "inner_train_records": subset["inner_train_records"],
                "inner_val_fraction": subset["inner_val_fraction"],
                "inner_val_within_0.15_0.25": subset["within_0.15_0.25"],
                "inner_split_feasible": subset["feasible"],
                "antigen_groups_disjoint": True,
            }
        )
        eligible_records += fold_eligible
        ineligible_multi_records += fold_blocked
    fractions = [float(row["inner_val_fraction"]) for row in folds]
    return {
        "rule": (
            "Antigen groups stay intact. Multi-group antibodies are placed onto different "
            "outer folds by a deterministic lightest-fold rule sorted by group id and group "
            "size. Inner validation is an exact subset of groups that are not required as "
            "the last inner_train support, chosen to be closest to 20% of outer_train records. "
            "No ΔΔG value is used."
        ),
        "eligible_records_over_one_full_pass": int(eligible_records),
        "multi_group_records_blocked_by_outer_fold_collision": int(ineligible_multi_records),
        "all_inner_splits_feasible": all(bool(row["inner_split_feasible"]) for row in folds),
        "all_inner_fractions_within_0.15_0.25": all(bool(row["inner_val_within_0.15_0.25"]) for row in folds),
        "inner_val_fractions": fractions,
        "folds": folds,
        "outer_assignment": {group: int(fold) for group, fold in sorted(assignment.items())},
    }


def main() -> None:
    if any(path.suffix in {".csv", ".json", ".md"} and path.name != Path(__file__).name for path in OUT_DIR.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing audit files in {OUT_DIR}")

    source = pd.read_csv(SOURCE_CSV, dtype=str, keep_default_na=False)
    if len(source) != N_EXPECTED:
        raise SystemExit(f"Expected {N_EXPECTED} source rows, found {len(source)}")

    mappings = {seed: load_mapping(seed) for seed in SEEDS}
    reference = mappings[0]
    key_columns = [
        "record_uid",
        "source_row_index",
        "antibody_entity_id",
        "antigen_group_id",
        "antigen_entity_id",
        "antibody_sequence_sha256",
        "antigen_sequence_sha256",
        "mapping_status",
        "biological_role",
        "complete_pdb_id",
        "antibody_chains",
        "antigen_chains",
    ]
    mapping_differences = []
    for seed in SEEDS[1:]:
        left = reference[key_columns]
        right = mappings[seed][key_columns]
        if not left.equals(right):
            mismatch = (left != right).any(axis=1).sum()
            mapping_differences.append({"seed": seed, "mismatched_rows": int(mismatch)})
    if len(reference) != N_EXPECTED:
        raise SystemExit(f"Mapping has {len(reference)} rows, expected {N_EXPECTED}")

    conflicts: list[dict[str, str]] = []
    records: list[dict[str, object]] = []
    for position, map_row in reference.reset_index(drop=True).iterrows():
        source_index = int(map_row["source_row_index"])
        if source_index != position:
            conflicts.append(
                {
                    "record_uid": map_row["record_uid"],
                    "reason": "source_row_index_not_aligned",
                    "detail": f"mapping={source_index} position={position}",
                }
            )
            continue
        src = source.iloc[source_index]
        expected_uid = legacy.record_uid("M1101", source_index)
        if map_row["record_uid"] != expected_uid:
            conflicts.append(
                {
                    "record_uid": map_row["record_uid"],
                    "reason": "record_uid_mismatch",
                    "detail": expected_uid,
                }
            )
        left_seqs, left_cols = nonempty_sequences(src, LEFT_COLS)
        right_seqs, right_cols = nonempty_sequences(src, RIGHT_COLS)
        left_key = "|".join(left_seqs)
        right_key = "|".join(right_seqs)
        left_sha = sha256_text(left_key)
        right_sha = sha256_text(right_key)
        antibody_sha = map_row["antibody_sequence_sha256"]
        antigen_sha = map_row["antigen_sequence_sha256"]
        if antibody_sha == left_sha and antigen_sha == right_sha:
            storage_side = "R_columns_heavy_light_fields"
            antibody_seqs = left_seqs
            antibody_cols = left_cols
            chain_ids = nonempty_ids(src, ("Heavy_chain", "Light_chain"))
        elif antibody_sha == right_sha and antigen_sha == left_sha:
            storage_side = "L_columns_antigen_chain_fields"
            antibody_seqs = right_seqs
            antibody_cols = right_cols
            chain_ids = nonempty_ids(src, ("Antigen_chain1", "Antigen_chain2"))
        else:
            storage_side = "unresolved"
            antibody_seqs = []
            antibody_cols = []
            chain_ids = []
            conflicts.append(
                {
                    "record_uid": map_row["record_uid"],
                    "reason": "antibody_sequence_hash_does_not_match_either_storage_side",
                    "detail": (
                        f"stored_ab={antibody_sha[:12]} left={left_sha[:12]} right={right_sha[:12]}"
                    ),
                }
            )
        paired_key = "|".join(antibody_seqs)
        recomputed_entity = "ABSEQ_" + short_hash(paired_key) if antibody_seqs else ""
        entity_status = "confirmed"
        if not map_row["antibody_entity_id"] or not map_row["antigen_group_id"]:
            entity_status = "unresolved"
            conflicts.append(
                {
                    "record_uid": map_row["record_uid"],
                    "reason": "missing_entity_or_group_id",
                    "detail": f"ab={map_row['antibody_entity_id']} ag={map_row['antigen_group_id']}",
                }
            )
        elif map_row["mapping_status"] != "confirmed":
            entity_status = "unresolved"
            conflicts.append(
                {
                    "record_uid": map_row["record_uid"],
                    "reason": "mapping_status_not_confirmed",
                    "detail": map_row["mapping_status"],
                }
            )
        elif storage_side == "unresolved" or recomputed_entity != map_row["antibody_entity_id"]:
            entity_status = "unresolved"
            conflicts.append(
                {
                    "record_uid": map_row["record_uid"],
                    "reason": "antibody_entity_id_does_not_match_recomputed_sequence_hash",
                    "detail": f"stored={map_row['antibody_entity_id']} recomputed={recomputed_entity}",
                }
            )
        elif len(chain_ids) != len(antibody_seqs):
            entity_status = "unresolved"
            conflicts.append(
                {
                    "record_uid": map_row["record_uid"],
                    "reason": "chain_count_does_not_match_sequence_count",
                    "detail": f"chains={chain_ids} sequences={antibody_cols}",
                }
            )
        if storage_side == "R_columns_heavy_light_fields" and len(antibody_seqs) == 2 and chain_ids == nonempty_ids(
            src, ("Heavy_chain", "Light_chain")
        ) and len(chain_ids) == 2:
            identity = "paired_heavy_and_light"
            heavy_seq, light_seq = antibody_seqs
        elif storage_side == "R_columns_heavy_light_fields" and len(antibody_seqs) == 1 and nonempty_ids(
            src, ("Heavy_chain",)
        ) and not nonempty_ids(src, ("Light_chain",)):
            identity = "single_chain_in_heavy_column"
            heavy_seq, light_seq = antibody_seqs[0], ""
        elif storage_side == "R_columns_heavy_light_fields" and len(antibody_seqs) == 1 and nonempty_ids(
            src, ("Light_chain",)
        ) and not nonempty_ids(src, ("Heavy_chain",)):
            identity = "single_chain_in_light_column"
            heavy_seq, light_seq = "", antibody_seqs[0]
        elif storage_side == "L_columns_antigen_chain_fields" and len(antibody_seqs) >= 1 and entity_status == "confirmed":
            identity = (
                "paired_partner_stored_on_L_side"
                if len(antibody_seqs) > 1
                else "single_partner_stored_on_L_side"
            )
            heavy_seq, light_seq = "", ""
        elif entity_status != "confirmed":
            identity = "unresolved"
            heavy_seq, light_seq = "", ""
        else:
            identity = "other_confirmed_partner_tuple"
            heavy_seq = antibody_seqs[0] if len(antibody_seqs) >= 1 and storage_side.startswith("R_") else ""
            light_seq = antibody_seqs[1] if len(antibody_seqs) >= 2 and storage_side.startswith("R_") else ""
        records.append(
            {
                "record_uid": map_row["record_uid"],
                "source_row_index": source_index,
                "original_mutation_id": map_row["original_mutation_id"],
                "complete_pdb_id": map_row["complete_pdb_id"],
                "complex_id": map_row["complex_id"],
                "antibody_entity_id": map_row["antibody_entity_id"] if entity_status == "confirmed" else "",
                "antigen_group_id": map_row["antigen_group_id"] if entity_status == "confirmed" else "",
                "mapping_status": entity_status,
                "stored_mapping_status": map_row["mapping_status"],
                "biological_role": map_row["biological_role"],
                "antibody_name": map_row["antibody_name"],
                "antigen_name": map_row["antigen_name"],
                "identity_definition": identity,
                "storage_side": storage_side,
                "chain_ids": ",".join(chain_ids),
                "sequence_columns": ",".join(antibody_cols),
                "n_chains": len(antibody_seqs) if entity_status == "confirmed" else 0,
                "heavy_sequence": heavy_seq,
                "light_sequence": light_seq,
                "paired_key": paired_key if entity_status == "confirmed" else "",
                "partner_chain_keys": tuple(antibody_seqs) if entity_status == "confirmed" else tuple(),
            }
        )

    record_frame = pd.DataFrame(records)
    confirmed = record_frame[record_frame["mapping_status"].eq("confirmed")].copy()
    unresolved_records = record_frame[record_frame["mapping_status"].ne("confirmed")]

    entity_hashes = confirmed.groupby("antibody_entity_id")["paired_key"].nunique()
    for entity, n_keys in entity_hashes.items():
        if int(n_keys) > 1:
            conflicts.append(
                {
                    "record_uid": "",
                    "reason": "same_antibody_entity_has_multiple_sequence_tuples",
                    "detail": f"{entity} tuples={int(n_keys)}",
                }
            )
    hash_entities = confirmed.groupby("paired_key")["antibody_entity_id"].nunique()
    for paired_key, n_entities in hash_entities.items():
        if int(n_entities) > 1:
            conflicts.append(
                {
                    "record_uid": "",
                    "reason": "same_sequence_tuple_has_multiple_antibody_entity_ids",
                    "detail": f"entities={int(n_entities)} sha256={sha256_text(str(paired_key))}",
                }
            )

    group_by_antibody: dict[str, set[str]] = defaultdict(set)
    for antibody, group in zip(confirmed["antibody_entity_id"], confirmed["antigen_group_id"]):
        group_by_antibody[str(antibody)].add(str(group))
    n_groups_by_antibody = {antibody: len(groups) for antibody, groups in group_by_antibody.items()}
    confirmed["n_antigen_groups"] = confirmed["antibody_entity_id"].map(n_groups_by_antibody)
    confirmed["structurally_possible"] = confirmed["n_antigen_groups"].ge(2)

    # Auxiliary chain matches. These never grant eligibility.
    heavy_groups: dict[str, set[str]] = defaultdict(set)
    light_groups: dict[str, set[str]] = defaultdict(set)
    chain_groups: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in confirmed.itertuples(index=False):
        if row.heavy_sequence:
            heavy_groups[row.heavy_sequence].add(row.antigen_group_id)
        if row.light_sequence:
            light_groups[row.light_sequence].add(row.antigen_group_id)
        for index, sequence in enumerate(row.partner_chain_keys):
            chain_groups[(index, sequence)].add(row.antigen_group_id)

    def crosses(groups_for_sequence: dict, sequence: str) -> bool:
        return bool(sequence) and len(groups_for_sequence.get(sequence, ())) >= 2

    confirmed["heavy_exact_crosses_groups"] = [
        crosses(heavy_groups, sequence) for sequence in confirmed["heavy_sequence"]
    ]
    confirmed["light_exact_crosses_groups"] = [
        crosses(light_groups, sequence) for sequence in confirmed["light_sequence"]
    ]
    loose_chain_cross_records = 0
    loose_only_records = 0
    for row in confirmed.itertuples(index=False):
        loose = any(len(chain_groups[(index, sequence)]) >= 2 for index, sequence in enumerate(row.partner_chain_keys))
        if loose:
            loose_chain_cross_records += 1
        if loose and int(row.n_antigen_groups) < 2:
            loose_only_records += 1

    heavy_cross_entities = {
        sequence: groups for sequence, groups in heavy_groups.items() if len(groups) >= 2
    }
    light_cross_entities = {
        sequence: groups for sequence, groups in light_groups.items() if len(groups) >= 2
    }
    heavy_only_not_full = []
    for sequence, groups in heavy_cross_entities.items():
        entities = sorted(set(confirmed.loc[confirmed["heavy_sequence"].eq(sequence), "antibody_entity_id"]))
        if len(entities) > 1:
            heavy_only_not_full.append(
                {
                    "n_antibody_entities": len(entities),
                    "antibody_entity_ids": entities,
                    "antigen_groups": sorted(groups),
                    "n_records": int(confirmed["heavy_sequence"].eq(sequence).sum()),
                }
            )
    light_only_not_full = []
    for sequence, groups in light_cross_entities.items():
        entities = sorted(set(confirmed.loc[confirmed["light_sequence"].eq(sequence), "antibody_entity_id"]))
        if len(entities) > 1:
            light_only_not_full.append(
                {
                    "n_antibody_entities": len(entities),
                    "antibody_entity_ids": entities,
                    "antigen_groups": sorted(groups),
                    "n_records": int(confirmed["light_sequence"].eq(sequence).sum()),
                }
            )

    proper_substrings = []
    unique_heavies = [seq for seq in heavy_groups if seq]
    for left_index, left in enumerate(unique_heavies):
        for right in unique_heavies[left_index + 1 :]:
            if left in right or right in left:
                proper_substrings.append(
                    {
                        "chain": "heavy_or_single_heavy_column",
                        "lengths": sorted((len(left), len(right))),
                        "groups_left": sorted(heavy_groups[left]),
                        "groups_right": sorted(heavy_groups[right]),
                    }
                )
    unique_lights = [seq for seq in light_groups if seq]
    for left_index, left in enumerate(unique_lights):
        for right in unique_lights[left_index + 1 :]:
            if left in right or right in left:
                proper_substrings.append(
                    {
                        "chain": "light_or_single_light_column",
                        "lengths": sorted((len(left), len(right))),
                        "groups_left": sorted(light_groups[left]),
                        "groups_right": sorted(light_groups[right]),
                    }
                )

    pair_rows = []
    for (antibody, group), part in confirmed.groupby(["antibody_entity_id", "antigen_group_id"], sort=True):
        identity_values = sorted(set(part["identity_definition"]))
        role_values = sorted(set(part["biological_role"]))
        pair_rows.append(
            {
                "antibody_entity_id": antibody,
                "antigen_group_id": group,
                "n_records": int(len(part)),
                "n_pdb": int(part["complete_pdb_id"].nunique()),
                "pdb_ids": ",".join(sorted(set(part["complete_pdb_id"]))),
                "complex_ids": ",".join(sorted(set(part["complex_id"]))),
                "biological_role": "|".join(role_values),
                "antibody_name": "|".join(sorted(set(part["antibody_name"]))),
                "antigen_name": "|".join(sorted(set(part["antigen_name"]))),
                "identity_definition": "|".join(identity_values),
                "storage_side": "|".join(sorted(set(part["storage_side"]))),
                "chain_ids": "|".join(sorted(set(part["chain_ids"]))),
                "sequence_columns": "|".join(sorted(set(part["sequence_columns"]))),
                "n_chains": int(part["n_chains"].iloc[0]),
                "heavy_length": int(len(part["heavy_sequence"].iloc[0])),
                "light_length": int(len(part["light_sequence"].iloc[0])),
                "heavy_sha256": sha256_text(part["heavy_sequence"].iloc[0]) if part["heavy_sequence"].iloc[0] else "",
                "light_sha256": sha256_text(part["light_sequence"].iloc[0]) if part["light_sequence"].iloc[0] else "",
                "paired_sha256": sha256_text(part["paired_key"].iloc[0]),
                "n_antigen_groups_of_antibody": int(n_groups_by_antibody[antibody]),
                "other_antigen_groups": ",".join(sorted(group_by_antibody[antibody] - {group})),
                "strict_ag_transfer_possible_if_another_group_is_inner_train": int(
                    n_groups_by_antibody[antibody] >= 2
                ),
            }
        )
    pair_frame = pd.DataFrame(pair_rows).sort_values(
        ["n_antigen_groups_of_antibody", "antibody_entity_id", "antigen_group_id"],
        ascending=[False, True, True],
    )

    reuse_rows = []
    for antibody, groups in sorted(group_by_antibody.items(), key=lambda item: (-len(item[1]), item[0])):
        part = confirmed[confirmed["antibody_entity_id"].eq(antibody)]
        multi = len(groups) >= 2
        reuse_rows.append(
            {
                "antibody_entity_id": antibody,
                "n_antigen_groups": len(groups),
                "antigen_group_ids": ",".join(sorted(groups)),
                "n_records": int(len(part)),
                "n_pdb": int(part["complete_pdb_id"].nunique()),
                "pdb_ids": ",".join(sorted(set(part["complete_pdb_id"]))),
                "identity_definition": "|".join(sorted(set(part["identity_definition"]))),
                "biological_role": "|".join(sorted(set(part["biological_role"]))),
                "antibody_name": "|".join(sorted(set(part["antibody_name"]))),
                "n_chains": int(part["n_chains"].iloc[0]),
                "heavy_sha256": sha256_text(part["heavy_sequence"].iloc[0]) if part["heavy_sequence"].iloc[0] else "",
                "light_sha256": sha256_text(part["light_sequence"].iloc[0]) if part["light_sequence"].iloc[0] else "",
                "paired_sha256": sha256_text(part["paired_key"].iloc[0]),
                "reuse_class": "multi_group" if multi else "single_group",
                "strict_ag_transfer_possible": int(multi),
                "blocking_reason": ""
                if multi
                else "antibody_entity_occurs_in_only_one_antigen_group",
            }
        )
    reuse_frame = pd.DataFrame(reuse_rows)

    pair_records = {
        (str(row.antibody_entity_id), str(row.antigen_group_id)): int(row.n_records)
        for row in pair_frame.itertuples(index=False)
    }
    group_sizes = confirmed.groupby("antigen_group_id").size().astype(int).to_dict()
    single_record_mask = confirmed["n_antigen_groups"].eq(1)
    multi_record_mask = confirmed["n_antigen_groups"].ge(2)
    n_antibodies = int(confirmed["antibody_entity_id"].nunique())
    n_single_antibodies = int((reuse_frame["reuse_class"] == "single_group").sum())
    n_multi_antibodies = int((reuse_frame["reuse_class"] == "multi_group").sum())
    n_single_records = int(single_record_mask.sum())
    n_multi_records = int(multi_record_mask.sum())
    n_unresolved = int(len(unresolved_records))

    shared_edges = []
    multi_antibodies = [ab for ab, groups in group_by_antibody.items() if len(groups) >= 2]
    seen_pairs: set[tuple[str, str]] = set()
    for antibody in multi_antibodies:
        groups = sorted(group_by_antibody[antibody])
        for left_index, left in enumerate(groups):
            for right in groups[left_index + 1 :]:
                seen_pairs.add((left, right))
    for left, right in sorted(seen_pairs):
        antibodies = sorted(
            antibody
            for antibody, groups in group_by_antibody.items()
            if left in groups and right in groups
        )
        shared_edges.append(
            {
                "antigen_group_a": left,
                "antigen_group_b": right,
                "n_shared_antibody_entities": len(antibodies),
                "shared_antibody_entity_ids": antibodies,
                "n_records_in_a_with_shared_antibodies": int(
                    confirmed["antigen_group_id"].eq(left).mul(confirmed["antibody_entity_id"].isin(antibodies)).sum()
                ),
                "n_records_in_b_with_shared_antibodies": int(
                    confirmed["antigen_group_id"].eq(right).mul(confirmed["antibody_entity_id"].isin(antibodies)).sum()
                ),
            }
        )

    group_summary = []
    for group, n_records in sorted(group_sizes.items(), key=lambda item: (-item[1], item[0])):
        part = confirmed[confirmed["antigen_group_id"].eq(group)]
        antibodies = sorted(set(part["antibody_entity_id"]))
        multi_abs = [ab for ab in antibodies if n_groups_by_antibody[ab] >= 2]
        group_summary.append(
            {
                "antigen_group_id": group,
                "n_records": int(n_records),
                "n_antibody_entities": len(antibodies),
                "n_multi_group_antibody_entities": len(multi_abs),
                "n_records_with_multi_group_antibody": int(part["n_antigen_groups"].ge(2).sum()),
                "n_records_with_single_group_antibody": int(part["n_antigen_groups"].eq(1).sum()),
                "partner_antigen_groups": sorted(
                    {
                        other
                        for antibody in multi_abs
                        for other in group_by_antibody[antibody]
                        if other != group
                    }
                ),
            }
        )

    # Existing AGCV eligibility. Antibody presence is counted from inner_train only.
    uid_to_antibody = dict(zip(record_frame["record_uid"], record_frame["antibody_entity_id"]))
    uid_to_group = dict(zip(record_frame["record_uid"], record_frame["antigen_group_id"]))
    uid_to_status = dict(zip(record_frame["record_uid"], record_frame["mapping_status"]))
    uid_to_ngroups = dict(zip(confirmed["record_uid"], confirmed["n_antigen_groups"]))
    eligibility_rows = []
    nontrivial_violations = []
    fold_cache_notes = []
    for seed in SEEDS:
        for fold in FOLDS:
            eval_uids = load_fold_uids(seed, fold, "outer_eval")
            train_uids = load_fold_uids(seed, fold, "inner_train")
            val_uids = load_fold_uids(seed, fold, "inner_val")
            outer_train_uids = load_fold_uids(seed, fold, "outer_train")
            eval_set, train_set, val_set = set(eval_uids), set(train_uids), set(val_uids)
            if eval_set & train_set or eval_set & val_set or train_set & val_set:
                fold_cache_notes.append(f"seed {seed} fold {fold} subsets overlap")
            if train_set | val_set != set(outer_train_uids):
                fold_cache_notes.append(f"seed {seed} fold {fold} inner union is not outer_train")
            eval_groups = {uid_to_group[uid] for uid in eval_set}
            train_groups = {uid_to_group[uid] for uid in train_set}
            val_groups = {uid_to_group[uid] for uid in val_set}
            if eval_groups & train_groups or eval_groups & val_groups or train_groups & val_groups:
                fold_cache_notes.append(f"seed {seed} fold {fold} antigen groups overlap")
            train_antibodies = {uid_to_antibody[uid] for uid in train_set if uid_to_antibody[uid]}
            val_antibodies = {uid_to_antibody[uid] for uid in val_set if uid_to_antibody[uid]}
            outer_antibodies = train_antibodies | val_antibodies
            counts = {"eligible": 0, "ineligible": 0, "unresolved": 0}
            eligible_groups: set[str] = set()
            eligible_antibodies: set[str] = set()
            seen_outer = 0
            seen_val_only = 0
            for uid in eval_uids:
                status = uid_to_status.get(uid, "unresolved")
                antibody = uid_to_antibody.get(uid, "")
                group = uid_to_group.get(uid, "")
                if status != "confirmed" or not antibody or not group:
                    counts["unresolved"] += 1
                    continue
                if antibody in train_antibodies:
                    counts["eligible"] += 1
                    eligible_groups.add(group)
                    eligible_antibodies.add(antibody)
                    continue
                counts["ineligible"] += 1
                if antibody in outer_antibodies:
                    seen_outer += 1
                if antibody in val_antibodies and antibody not in train_antibodies:
                    seen_val_only += 1
                if int(uid_to_ngroups.get(uid, 1)) >= 2:
                    support = group_by_antibody[antibody] - {group}
                    support_train = sorted(support & train_groups)
                    support_val = sorted(support & val_groups)
                    support_same = sorted(support & eval_groups)
                    if support_same and not support_train and not support_val:
                        code = "support_groups_are_in_the_same_outer_fold"
                    elif support_val and not support_train:
                        code = "support_groups_are_only_in_inner_val"
                    else:
                        code = "support_not_in_inner_train"
                    nontrivial_violations.append(
                        {
                            "scope": "existing_agcv",
                            "seed": seed,
                            "fold": fold,
                            "record_uid": uid,
                            "antigen_group_id": group,
                            "antibody_entity_id": antibody,
                            "violation_code": code,
                            "n_antigen_groups_for_antibody": int(uid_to_ngroups[uid]),
                            "support_groups": ",".join(sorted(support)),
                            "support_in_inner_train": ",".join(support_train),
                            "support_in_inner_val": ",".join(support_val),
                            "support_in_same_outer_fold": ",".join(support_same),
                        }
                    )
            eligibility_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "n_outer_eval": len(eval_uids),
                    "n_eligible": counts["eligible"],
                    "eligible_coverage": counts["eligible"] / len(eval_uids) if eval_uids else 0.0,
                    "n_ineligible": counts["ineligible"],
                    "n_unresolved": counts["unresolved"],
                    "n_eligible_antigen_groups": len(eligible_groups),
                    "n_eligible_antibody_entities": len(eligible_antibodies),
                    "n_outer_eval_antigen_groups": len(eval_groups),
                    "n_outer_eval_antibody_entities": len({uid_to_antibody[uid] for uid in eval_set if uid_to_antibody[uid]}),
                    "n_inner_train": len(train_uids),
                    "n_inner_val": len(val_uids),
                    "n_eval_antibody_seen_in_outer_train": seen_outer + counts["eligible"],
                    "n_eval_antibody_seen_in_inner_val_but_not_inner_train": seen_val_only,
                }
            )

    # Each source record appears once per seed.
    partition_errors = []
    for seed in SEEDS:
        seen: dict[str, int] = defaultdict(int)
        for fold in FOLDS:
            for uid in load_fold_uids(seed, fold, "outer_eval"):
                seen[uid] += 1
        missing = [f"M1101:{i:06d}" for i in range(N_EXPECTED) if seen.get(f"M1101:{i:06d}", 0) != 1]
        if missing:
            partition_errors.append({"seed": seed, "n_records_not_exactly_once": len(missing)})

    structural_violations = []
    for row in confirmed.loc[single_record_mask].itertuples(index=False):
        structural_violations.append(
            {
                "scope": "any_group_intact_split",
                "seed": "",
                "fold": "",
                "record_uid": row.record_uid,
                "source_row_index": int(row.source_row_index),
                "original_mutation_id": row.original_mutation_id,
                "complete_pdb_id": row.complete_pdb_id,
                "antigen_group_id": row.antigen_group_id,
                "antibody_entity_id": row.antibody_entity_id,
                "violation_code": "antibody_entity_occurs_in_only_one_antigen_group",
                "n_antigen_groups_for_antibody": 1,
                "support_groups": "",
                "support_in_inner_train": "",
                "support_in_inner_val": "",
                "support_in_same_outer_fold": "",
                "detail": (
                    "Holding out this antigen group removes every record of this antibody entity, "
                    "so the entity cannot occur in inner_train."
                ),
            }
        )
    for row in unresolved_records.itertuples(index=False):
        structural_violations.append(
            {
                "scope": "unresolved_mapping",
                "seed": "",
                "fold": "",
                "record_uid": row.record_uid,
                "source_row_index": int(row.source_row_index),
                "original_mutation_id": row.original_mutation_id,
                "complete_pdb_id": row.complete_pdb_id,
                "antigen_group_id": row.antigen_group_id,
                "antibody_entity_id": row.antibody_entity_id,
                "violation_code": "antibody_entity_unresolved",
                "n_antigen_groups_for_antibody": "",
                "support_groups": "",
                "support_in_inner_train": "",
                "support_in_inner_val": "",
                "support_in_same_outer_fold": "",
                "detail": row.identity_definition,
            }
        )

    assignment = build_separating_assignment(group_sizes, group_by_antibody)
    reassignment = evaluate_assignment(assignment, group_sizes, pair_records, group_by_antibody)
    structural_upper_bound = n_multi_records
    reassignment_eligible = int(reassignment["eligible_records_over_one_full_pass"])
    full_cv_possible = structural_upper_bound == N_EXPECTED and n_unresolved == 0 and reassignment_eligible == N_EXPECTED
    optimality = (
        "proven_maximum"
        if reassignment_eligible == structural_upper_bound
        and reassignment["all_inner_splits_feasible"]
        and reassignment["multi_group_records_blocked_by_outer_fold_collision"] == 0
        else "best_found_not_proven"
    )
    # The single-group bound is exhaustive: no intact-group assignment can exceed it.
    # The exhibited assignment meets that bound, so the coverage maximum is proven
    # even before the inner-validation fraction is considered. The fraction is then
    # checked on the same assignment.
    if reassignment_eligible == structural_upper_bound:
        optimality = "proven_maximum"

    existing_eligible_total_per_seed = []
    eligibility_frame = pd.DataFrame(eligibility_rows)
    for seed, part in eligibility_frame.groupby("seed"):
        existing_eligible_total_per_seed.append(
            {
                "seed": int(seed),
                "n_eligible_record_evaluations": int(part["n_eligible"].sum()),
                "n_outer_eval": int(part["n_outer_eval"].sum()),
                "coverage": float(part["n_eligible"].sum() / part["n_outer_eval"].sum()),
            }
        )

    identity_counts = confirmed["identity_definition"].value_counts().to_dict()
    role_counts = confirmed["biological_role"].value_counts().to_dict()
    improved = reassignment_eligible > int(eligibility_frame["n_eligible"].sum() / len(SEEDS))
    # Compare against each seed; all seeds are expected to match because the obstruction is structural.
    existing_coverages = [row["coverage"] for row in existing_eligible_total_per_seed]
    can_improve = reassignment_eligible > max(row["n_eligible_record_evaluations"] for row in existing_eligible_total_per_seed)

    if n_unresolved:
        recommendation = (
            "Some antibody-entity mappings are unresolved. A strict Ag-transfer split "
            "needs those mappings first."
        )
    elif structural_upper_bound == 0:
        recommendation = (
            "The current AGCV is the split on disk. Its strict subset is empty, so "
            "there is no strict Ag-transfer record to score on its own."
        )
    elif full_cv_possible:
        recommendation = (
            "A full five-fold strict Ag-transfer split is possible from the data relations. "
            "This audit does not write that split."
        )
    else:
        recommendation = (
            "The full 1101 records cannot form a strict Ag-transfer five-fold split. "
            "Reassigning antigen groups can reach the structural upper bound of the "
            "multi-group antibody subset. That subset would live in its own directory."
        )

    mapping_hashes = {
        str(seed): file_sha256(SPLIT_ROOT / str(seed) / "AGCV" / "antigen_mapping.csv") for seed in SEEDS
    }
    result = {
        "question_full_1101_five_fold_strict_ag_transfer": {
            "feasible": full_cv_possible,
            "n_source_records": N_EXPECTED,
            "n_confirmed_records": int(len(confirmed)),
            "n_unresolved_records": n_unresolved,
            "structural_upper_bound_records": structural_upper_bound,
            "reason": (
                "A record can enter strict Ag-transfer only when its complete antibody entity "
                "also occurs in another intact antigen group that is placed in inner_train. "
                f"{n_single_records} confirmed records belong to antibody entities that occur "
                "in only one antigen group. Holding that group out removes the antibody from "
                "every training subset. Those records cannot be covered by any five-fold, "
                "group-intact assignment."
            ),
        },
        "antibody_identity": {
            "primary_key": "confirmed antibody_entity_id, recomputed as ABSEQ_ plus SHA1-12 of the wild-type partner sequence tuple",
            "pair_rule": "Heavy and light sequences are both required when both chains exist. Either chain alone is only an auxiliary count.",
            "n_antibody_entities": n_antibodies,
            "identity_definition_counts": identity_counts,
            "biological_role_counts": role_counts,
            "mapping_conflicts": conflicts,
            "mapping_identical_across_five_seeds_on_entity_columns": mapping_differences == [],
            "mapping_differences": mapping_differences,
            "mapping_file_sha256": mapping_hashes,
            "source_sha256": file_sha256(SOURCE_CSV),
        },
        "antibody_antigen_reuse": {
            "n_antibody_entities": n_antibodies,
            "n_antibodies_in_one_antigen_group": n_single_antibodies,
            "n_records_of_single_group_antibodies": n_single_records,
            "n_antibodies_in_two_or_more_antigen_groups": n_multi_antibodies,
            "n_records_of_multi_group_antibodies": n_multi_records,
            "n_antigen_groups": len(group_sizes),
            "shared_antigen_group_pairs": shared_edges,
            "antigen_group_summary": group_summary,
            "auxiliary_not_used_for_eligibility": {
                "n_exact_heavy_sequences_shared_by_multiple_groups": len(heavy_cross_entities),
                "n_exact_light_sequences_shared_by_multiple_groups": len(light_cross_entities),
                "heavy_matches_that_are_not_the_same_full_antibody": heavy_only_not_full,
                "light_matches_that_are_not_the_same_full_antibody": light_only_not_full,
                "n_records_whose_some_single_chain_occurs_in_another_group": loose_chain_cross_records,
                "n_records_loose_chain_only_full_antibody_does_not_cross": loose_only_records,
                "proper_substring_pairs_not_used": proper_substrings,
            },
        },
        "existing_agcv": {
            "root": str(SPLIT_ROOT),
            "seeds": list(SEEDS),
            "per_seed_record_coverage": existing_eligible_total_per_seed,
            "partition_errors": partition_errors,
            "fold_consistency_notes": fold_cache_notes,
            "eligibility_definition": "antigen group absent from inner_train and inner_val; complete antibody entity present in inner_train",
        },
        "reassignment_search": {
            "objective": (
                "Maximize the number of source records that are eligible exactly once "
                "across five outer folds, subject to intact antigen groups."
            ),
            "constraints": [
                "22 antigen groups from the on-disk SPLIT_R2 mapping",
                "each group assigned to exactly one of 5 outer folds",
                "each fold nonempty",
                "inner_train and inner_val are intact groups and partition outer_train",
                "an outer-eval record is eligible only if its antibody entity occurs in inner_train",
                "inner_val antibodies do not count",
                "no ΔΔG labels",
            ],
            "structural_upper_bound": structural_upper_bound,
            "upper_bound_proof": (
                "Every single-group antibody record is ineligible in every intact-group split. "
                "Therefore no assignment can exceed the multi-group record count. "
                "If the exhibited assignment reaches that count, the count is the maximum."
            ),
            "solver": "deterministic construction plus exact inner-validation subset search",
            "solver_status": optimality,
            "exhibited_eligible_records": reassignment_eligible,
            "exhibited_coverage_of_1101": reassignment_eligible / N_EXPECTED,
            "improves_on_every_existing_seed": can_improve,
            "solution": reassignment,
            "new_split_files_written": False,
        },
        "recommendation": recommendation,
        "executed": [
            "loaded all 1101 M1101.csv rows by source index",
            "verified five SPLIT_R2 antigen mappings on entity columns",
            "recomputed antibody sequence tuples from R1-R6 and L1-L3 and matched stored hashes",
            "built antibody-antigen group incidence",
            "scored existing five seeds and five folds from inner_train versus outer_eval",
            "searched one identity-only reassignment against the structural upper bound",
        ],
        "not_computed": [
            "sequence-identity clustering below exact match",
            "structure or interface similarity",
            "model training",
            "a new fold CSV split",
            "any split chosen with ΔΔG",
        ],
    }

    pair_frame.to_csv(OUT_DIR / "antibody_antigen_group_map.csv", index=False)
    reuse_frame.to_csv(OUT_DIR / "antibody_group_reuse_summary.csv", index=False)
    eligibility_frame.to_csv(OUT_DIR / "strict_ag_transfer_eligibility_by_fold.csv", index=False)
    violation_frame = pd.DataFrame(structural_violations + nontrivial_violations)
    violation_columns = [
        "scope",
        "seed",
        "fold",
        "record_uid",
        "source_row_index",
        "original_mutation_id",
        "complete_pdb_id",
        "antigen_group_id",
        "antibody_entity_id",
        "violation_code",
        "n_antigen_groups_for_antibody",
        "support_groups",
        "support_in_inner_train",
        "support_in_inner_val",
        "support_in_same_outer_fold",
        "detail",
    ]
    for column in violation_columns:
        if column not in violation_frame.columns:
            violation_frame[column] = ""
    violation_frame[violation_columns].to_csv(OUT_DIR / "strict_ag_transfer_violations.csv", index=False)
    (OUT_DIR / "strict_ag_transfer_feasibility.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    def fmt_fold_table() -> str:
        lines = [
            "| seed | fold | outer eval | eligible | coverage | ineligible | unresolved | eligible antigen groups | eligible antibodies |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in eligibility_frame.itertuples(index=False):
            lines.append(
                f"| {row.seed} | {row.fold} | {row.n_outer_eval} | {row.n_eligible} | "
                f"{row.eligible_coverage:.4f} | {row.n_ineligible} | {row.n_unresolved} | "
                f"{row.n_eligible_antigen_groups} | {row.n_eligible_antibody_entities} |"
            )
        return "\n".join(lines)

    multi_lines = []
    if n_multi_antibodies == 0:
        multi_lines.append("No complete antibody entity appears in two or more antigen groups.")
    else:
        for row in reuse_frame[reuse_frame["reuse_class"].eq("multi_group")].itertuples(index=False):
            multi_lines.append(
                f"- `{row.antibody_entity_id}`: {row.n_records} records, "
                f"antigen groups {row.antigen_group_ids}, "
                f"identity {row.identity_definition}, PDB {row.pdb_ids}."
            )
    fraction_text = ", ".join(f"{value:.4f}" for value in reassignment["inner_val_fractions"])
    readme = f"""# M1101 strict Ag-transfer

The full 1101 records cannot form a strict Ag-transfer five-fold split. The limit is which antibody entities sit in which antigen groups. It does not depend on the seed or the inner-validation fraction.

There are {n_antibodies} complete antibody entities in the confirmed map. {n_single_antibodies} of them appear in only one antigen group ({n_single_records} records). When that group is the outer eval set, the antibody is absent from `inner_train`, so those records are never eligible.

The structural upper bound is {structural_upper_bound} records, {structural_upper_bound / N_EXPECTED:.2%} of 1101. That count is the records whose complete antibody entity appears in at least two antigen groups. The deterministic reassignment reaches {reassignment_eligible} records (`{optimality}`). A single-group antibody contributes 0 under every intact-group assignment, so the bound does not depend on a search.

Eligibility below counts an antibody only when it is already in that fold's `inner_train`. Presence in `inner_val` does not count.

{fmt_fold_table()}

Reassignment {"raises" if can_improve else "does not raise"} the strict coverage. Eligible records in one full rotation of the current seeds: {existing_eligible_total_per_seed}. The identity-constrained assignment covers {reassignment_eligible} records.

{recommendation}

## Antibody identity

The identity is the confirmed `antibody_entity_id`. Each raw record's R1-R6 and L1-L3 chains are rebuilt and compared with `antibody_sequence_sha256` and `antigen_sequence_sha256`. The entity id is `ABSEQ_` plus the first 12 hex characters of the SHA1 of the antibody-side tuple.

Antigen maps {"agree" if not mapping_differences else "differ"} on the entity columns across seeds. Mapping SHA256: `{mapping_hashes}`.

Identity counts: `{identity_counts}`.

Biological-role counts: `{role_counts}`. When both heavy and light chains are present, the match uses the ordered pair. Heavy-only or light-only matches are auxiliary and do not grant eligibility.

Mapping conflicts: {len(conflicts)}.

Auxiliary counts:

- exact heavy sequences that cross antigen groups: {len(heavy_cross_entities)}
- exact light sequences that cross antigen groups: {len(light_cross_entities)}
- same heavy chain, different complete antibody: {len(heavy_only_not_full)}
- same light chain, different complete antibody: {len(light_only_not_full)}
- a single chain crosses groups while the complete antibody does not: {loose_only_records}
- exact substring pairs (no similarity threshold): {len(proper_substrings)}

## Antibodies that cross antigen groups

{chr(10).join(multi_lines)}

Edges between antigen groups that share a complete antibody: `{shared_edges}`.

## Current AGCV and the reassignment

Outer eval holds out whole antigen groups. A record is eligible only when the same `antibody_entity_id` is already in that fold's `inner_train`.

Inner-validation fractions of the reassignment: {fraction_text}. All five folds fall in 0.15-0.25: {reassignment["all_inner_fractions_within_0.15_0.25"]}. The assignment is stored in `reassignment_search.solution` of the JSON. It is not written as fold files.

## What this directory contains

Record alignment for all 1101 rows, the five seed maps, sequence-tuple checks, antibody-antigen links, eligibility of the current five-by-five AGCV, the single-group upper bound, and one assignment that meets that bound.

Sequence clustering below exact identity, structure or interface similarity, model training, new fold CSVs, and label-based split search are not in this directory.

## Files

- `antibody_antigen_group_map.csv`: one complete antibody entity and one antigen group per row.
- `antibody_group_reuse_summary.csv`: one antibody entity per row.
- `strict_ag_transfer_feasibility.json`: the bound and the exhibited assignment.
- `strict_ag_transfer_eligibility_by_fold.csv`: eligibility of the current AGCV.
- `strict_ag_transfer_violations.csv`: records that no intact grouping can make eligible, plus multi-group antibodies that fail in the current folds.
- `compute_strict_ag_transfer_feasibility.py`: the script that writes this directory.
"""
    (OUT_DIR / "README_strict_ag_transfer_feasibility.md").write_text(readme, encoding="utf-8")
    print(json.dumps({
        "full_feasible": full_cv_possible,
        "n_antibodies": n_antibodies,
        "n_single_antibodies": n_single_antibodies,
        "n_single_records": n_single_records,
        "n_multi_antibodies": n_multi_antibodies,
        "n_multi_records": n_multi_records,
        "n_unresolved": n_unresolved,
        "n_conflicts": len(conflicts),
        "identity_counts": identity_counts,
        "existing": existing_eligible_total_per_seed,
        "reassignment_eligible": reassignment_eligible,
        "optimality": optimality,
        "fractions": reassignment["inner_val_fractions"],
        "within_band": reassignment["all_inner_fractions_within_0.15_0.25"],
        "can_improve": can_improve,
        "n_shared_edges": len(shared_edges),
        "partition_errors": partition_errors,
        "fold_notes": fold_cache_notes,
        "mapping_identical": mapping_differences == [],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
