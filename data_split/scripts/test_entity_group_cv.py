"""Lightweight checks for entity-grouped splits. No model training."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

import entity_group_cv as eg
import three_splits as legacy

ROOT = Path(__file__).resolve().parent
PY = sys.executable
LEGACY_HASHES = {
    ROOT / "output" / "split_summary.csv": "a32c0811ef49e7a2593fc434d7792da3c59af7b15957d052c85d7099ca019afb",
    ROOT / "M1101" / "0" / "RCSV" / "val_fold_0.csv": "69cb0f88e5e8d458741b3a965f0bc5bf35f5be4323261da794f29a1cc00d8935",
    ROOT / "M1101" / "icsv_group_map.csv": "8cb93289fe3fb972f7926fac5725f4bb80835b9279e0a00e2ab2fe7919109359",
    ROOT / "S1131" / "42" / "ICSV" / "train_fold_0.csv": "2af86fc3c08739b6e96364101216ffe67fec8978e70ed2dbb45e4870473ec558",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_balance_is_reproducible_and_refuses_too_few_groups() -> None:
    sizes = {f"G{index}": index + 1 for index in range(8)}
    first = eg.assign_outer_groups(sizes, seed=2026, n_splits=5)
    second = eg.assign_outer_groups(sizes, seed=2026, n_splits=5)
    assert first == second
    assert set(first.values()) == {0, 1, 2, 3, 4}
    try:
        eg.assign_outer_groups({"A": 3, "B": 1, "C": 1}, seed=0, n_splits=5)
    except eg.InfeasibleSplit as exc:
        assert "silently reduce" in str(exc)
    else:
        raise AssertionError("too few groups did not raise")


def test_inner_groups_are_disjoint() -> None:
    sizes = {f"G{index}": (index + 1) * 3 for index in range(6)}
    train, validation = eg.assign_inner_groups(sizes, seed=42, fraction=0.2)
    assert train.isdisjoint(validation)
    assert train | validation == set(sizes)
    try:
        eg.assign_inner_groups({"only": 10}, seed=1, fraction=0.2)
    except eg.InfeasibleSplit as exc:
        assert "outer_eval" in str(exc)
    else:
        raise AssertionError("single-group inner split did not raise")


def test_unknown_antigen_does_not_write_folds() -> None:
    header = pd.read_csv(ROOT / "M1101.csv", nrows=0)
    row = {column: "" for column in header.columns}
    row.update({"PDB": "9ZZZ", "ID": "9ZZZ_AA1A", "Partners": "A_B", "ddG": 1.0, "Heavy_chain": "A", "Antigen_chain1": "B", "R1": "ACDEFG", "L1": "HHHHHH"})
    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)
        csv_path = temp / "M1101.csv"
        pd.DataFrame([row]).to_csv(csv_path, index=False)
        out = temp / "out"
        completed = subprocess.run(
            [
                PY, str(ROOT / "three_splits.py"),
                "--entity-cv", "AGCV",
                "--datasets", "M1101.csv",
                "--seeds", "0",
                "--input-dir", str(temp),
                "--entity-output-root", str(out),
                "--wt-pdb-dir", str(temp / "missing_wt"),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode != 0, completed.stdout + completed.stderr
        assert not list(out.rglob("fold_*_outer_eval.csv"))
        blocked = list(out.rglob("unresolved_issues.txt"))
        assert blocked, completed.stdout + completed.stderr


def test_output_directory_conflict_exits() -> None:
    with tempfile.TemporaryDirectory() as temp_name:
        out = Path(temp_name) / "out"
        occupied = out / "M1101" / "0" / "AGCV"
        occupied.mkdir(parents=True)
        (occupied / "existing.txt").write_text("keep\n", encoding="utf-8")
        completed = subprocess.run(
            [
                PY, str(ROOT / "three_splits.py"),
                "--entity-cv", "AGCV",
                "--datasets", "M1101.csv",
                "--seeds", "0",
                "--input-dir", str(ROOT),
                "--entity-output-root", str(out),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode != 0
        assert (occupied / "existing.txt").read_text(encoding="utf-8") == "keep\n"
        assert "Refusing" in (completed.stderr + completed.stdout)


def test_pgcv_without_a_side_is_refused() -> None:
    with tempfile.TemporaryDirectory() as temp_name:
        out = Path(temp_name) / "out"
        completed = subprocess.run(
            [
                PY, str(ROOT / "three_splits.py"),
                "--entity-cv", "PGCV",
                "--datasets", "S1131.csv",
                "--seeds", "0",
                "--input-dir", str(ROOT),
                "--entity-output-root", str(out),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode != 0, completed.stdout + completed.stderr
        decision = json.loads((out / "S1131" / "PGCV_UNAVAILABLE" / "decision.json").read_text(encoding="utf-8"))
        assert decision["status"] == "REFUSED"
        assert decision["mutation_side_counts"]["LEFT"] > 0
        assert decision["mutation_side_counts"]["RIGHT"] > 0
        assert not list(out.rglob("fold_*_outer_eval.csv"))


def test_real_m1101_duplicate_ids_and_multi_pdb_groups() -> None:
    frame, raw_pdb, repair = eg.load_master(ROOT / "M1101.csv")
    mapping = eg.build_m1101_mapping(frame, raw_pdb, repair, ROOT / "M1101" / "wt")
    assert len(mapping) == len(frame) == 1101
    assert set(mapping["mapping_status"]) == {"confirmed"}
    source_counts = frame["ID"].astype(str).value_counts()
    repeated = set(source_counts[source_counts > 1].index)
    assert repeated
    assert set(mapping.loc[mapping["original_mutation_id"].isin(repeated), "record_uid"]) == set(
        mapping.loc[frame["ID"].astype(str).isin(repeated).to_numpy(), "record_uid"]
    )
    lysozyme = mapping.loc[mapping["antigen_group_id"] == "AG_LYSOZYME", "complete_pdb_id"].unique()
    assert set(lysozyme) == {"1DQJ", "1MLC", "1VFB", "3HFM"}
    egfr = set(mapping.loc[mapping["antigen_group_id"] == "AG_EGFR", "complete_pdb_id"].unique())
    assert egfr == {"1YY9", "HM_1YY9"}
    assert "HM_" in " ".join(egfr)
    her2 = set(mapping.loc[mapping["antigen_group_id"] == "AG_HER2", "complete_pdb_id"].unique())
    assert her2 == {"1N8Z", "3BE1"}
    assert set(mapping.loc[mapping["complete_pdb_id"] == "1KTZ", "antigen_group_id"]) == {"AG_TGFB3"}
    assert set(mapping.loc[mapping["complete_pdb_id"] == "HM_1KTZ", "antigen_group_id"]) == {"AG_TGFB1"}
    assert set(mapping.loc[mapping["complete_pdb_id"] == "1KTZ", "antigen_chains"]) == {"A"}
    assert set(mapping.loc[mapping["complete_pdb_id"] == "HM_1KTZ", "antigen_chains"]) == {"A"}


def test_repeat_real_assignment_matches() -> None:
    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)
        first = temp / "a"
        second = temp / "b"
        command = [
            PY, str(ROOT / "three_splits.py"),
            "--entity-cv", "AGCV",
            "--datasets", "M1101.csv",
            "--seeds", "1234",
            "--input-dir", str(ROOT),
            "--entity-output-root",
        ]
        one = subprocess.run(command + [str(first)], cwd=ROOT, text=True, capture_output=True, check=False)
        two = subprocess.run(command + [str(second)], cwd=ROOT, text=True, capture_output=True, check=False)
        assert one.returncode == 0, one.stdout + one.stderr
        assert two.returncode == 0, two.stdout + two.stderr
        left = first / "M1101" / "1234" / "AGCV" / "outer_assignments.csv"
        right = second / "M1101" / "1234" / "AGCV" / "outer_assignments.csv"
        assert left.read_bytes() == right.read_bytes()


def test_legacy_split_files_unchanged() -> None:
    for path, digest in LEGACY_HASHES.items():
        assert path.is_file(), path
        assert _sha(path) == digest, path


def main() -> int:
    tests = [
        test_balance_is_reproducible_and_refuses_too_few_groups,
        test_inner_groups_are_disjoint,
        test_unknown_antigen_does_not_write_folds,
        test_output_directory_conflict_exits,
        test_pgcv_without_a_side_is_refused,
        test_real_m1101_duplicate_ids_and_multi_pdb_groups,
        test_repeat_real_assignment_matches,
        test_legacy_split_files_unchanged,
    ]
    for test in tests:
        print(f"[TEST] {test.__name__}")
        test()
        print("  ok")
    print("[TEST] all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
