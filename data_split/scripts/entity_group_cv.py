"""Entity-grouped outer splits for M1101 (AGCV) and S1131 (PGCV_LEFT / PGCV_RIGHT).

This module is imported by three_splits.py only when --entity-cv is set.
It does not change RCSV, PCSV, or ICSV, and it never deletes an existing output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import three_splits as legacy

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}

# Curated M1101 antigen-side entities. Group membership is a hypothesis that
# the generator accepts only after exact sequence and local PDB checks pass.
# Similar names are not enough: HER2, gp120, BoNT/A1 versus A2, and MT-SP1
# versus Epithin are intentionally separate groups.
M1101_ANTIGEN_GROUPS: tuple[dict[str, object], ...] = (
    {
        "group_id": "AG_HIV1_CAPSID_NTERM",
        "antigen_name": "HIV-1 CAPSID (N-Term)",
        "antibody_name": "huCyc-A",
        "role": "non_antibody_complex",
        "pdbs": ("1AK4",),
        "construct": "Crystal 1AK4. Antigen chain D. Chain A is cyclophilin A, not an antibody.",
        "evidence": "M1101.csv Protein-2 plus Antigen_chain1=D. Local wt/1AK4.pdb CA sequence of chain D matches L1 and chain A matches R1.",
        "notes": "Not a classical antibody-antigen complex. The held-out entity is the antigen-side partner, HIV-1 capsid N-terminal domain.",
    },
    {
        "group_id": "AG_VEGF",
        "antigen_name": "VEGF",
        "antibody_name": "fab-12 / Y0317 / bH1, kept as separate antibodies",
        "role": "classical_antibody_antigen",
        "pdbs": ("1BJ1", "1CZ8", "3BDY"),
        "construct": "1BJ1 and 1CZ8 use the same two 94-residue VEGF chains (V and W). 3BDY chain V is that chain plus one C-terminal residue.",
        "evidence": "M1101.csv names all three VEGF. 1BJ1 and 1CZ8 antigen chains are identical. Each of those chains is an exact substring of the 3BDY antigen chain. Local PDB CA sequences match the CSV chains.",
        "notes": "Same wild-type VEGF entity across antibodies and across a one-residue construct difference. Antibodies are not used to merge or split the group.",
    },
    {
        "group_id": "AG_LYSOZYME",
        "antigen_name": "Lysozyme",
        "antibody_name": "HYHEL-63 / D44.1 / D1.3 / HyHEL-10",
        "role": "classical_antibody_antigen",
        "pdbs": ("1DQJ", "1MLC", "1VFB", "3HFM"),
        "construct": "Four crystal complexes. Antigen WT sequence is the same 129-residue chain (C, E, C, and Y respectively).",
        "evidence": "Antigen-chain WT sequences of 1DQJ, 1MLC, 1VFB, and 3HFM are identical, and each matches the named antigen chain in the local PDB.",
        "notes": "3HFM Protein-2 is recorded as 'FV'. That label is not used. Chain Y is exactly the lysozyme sequence of the other three complexes.",
        "pdb_notes": {
            "3HFM": "Protein-2 says FV, but antigen chain Y is identical to lysozyme in 1DQJ, 1MLC, and 1VFB.",
        },
    },
    {
        "group_id": "AG_E52_FV",
        "antigen_name": "E5.2 Fv",
        "antibody_name": "IgG1-kappa D1.3 Fv",
        "role": "antibody_idiotope",
        "pdbs": ("1DVF",),
        "construct": "Crystal 1DVF. Antigen-side chains C and D are the E5.2 Fv. Chains A and B are D1.3.",
        "evidence": "M1101.csv Protein-2 is E5.2 Fv and Antigen_chain1/2 are C and D. Local 1DVF.pdb CA sequences match R1/R2 to A/B and L1/L2 to C/D.",
        "notes": "Idiotope complex: the antigen-side entity is itself an antibody Fv. It is not merged with lysozyme-bound D1.3 in 1VFB.",
    },
    {
        "group_id": "AG_CHEY",
        "antigen_name": "CHEY",
        "antibody_name": "CHEA",
        "role": "non_antibody_complex",
        "pdbs": ("1FFW",),
        "construct": "Crystal 1FFW. Antigen-side chain B. Chain A is CheA.",
        "evidence": "M1101.csv Protein-2 is CHEY and Antigen_chain1=B. Local PDB chain lengths match R1=A and L1=B.",
        "notes": "Not a classical antibody-antigen complex. Held-out entity is CheY.",
    },
    {
        "group_id": "AG_IFNGR1_108",
        "antigen_name": "IFNgammaR1-108",
        "antibody_name": "A6 fab",
        "role": "classical_antibody_antigen",
        "pdbs": ("1JRH",),
        "construct": "Crystal 1JRH. Antigen chain I is the receptor fragment bound by A6 Fab (chains H and L).",
        "evidence": "M1101.csv Protein-2 and Antigen_chain1=I. Local PDB CA sequences match the CSV chain assignment.",
        "notes": "The antigen is a receptor fragment bound by an antibody.",
    },
    {
        "group_id": "AG_BLIP",
        "antigen_name": "Beta-Lactamase inhibitor",
        "antibody_name": "TEM-1 Beta-Lactamase",
        "role": "non_antibody_complex",
        "pdbs": ("1JTG",),
        "construct": "Crystal 1JTG. Antigen-side chain B is BLIP. Chain A is TEM-1.",
        "evidence": "M1101.csv Protein-2 is Beta-Lactamase inhibitor and Antigen_chain1=B. Local PDB chain B matches L1.",
        "notes": "Not a classical antibody-antigen complex. Held-out entity is BLIP.",
    },
    {
        "group_id": "AG_TGFB3",
        "antigen_name": "TGF-beta3",
        "antibody_name": "TGF-beta type-II receptor",
        "role": "non_antibody_complex",
        "pdbs": ("1KTZ",),
        "construct": "Crystal 1KTZ. The ligand is chain A, local residues 13-112, with an unmodeled internal loop. Chain B, residues 25-130, is the type-II receptor and is not the grouped entity.",
        "evidence": "Chain A carries TGF-beta3 motifs (QDLGWKWVHEP, CSGPCPYLR, DLEPLTIL, SNMVVK). Chain B is an exact fragment of the TGF-beta type II receptor sequence and is identical to HM_1KTZ chain B. The CSV Antigen_chain column points at the receptor; the ligand was assigned from the chain sequence and residue numbers.",
        "notes": "Not grouped with HM_1KTZ. The shared chain is the receptor. The ligands are different isoforms. This is a cytokine-receptor complex, not an antibody-antigen complex.",
    },
    {
        "group_id": "AG_INTEGRIN_I",
        "antigen_name": "Integrin-I",
        "antibody_name": "Neutralizing hu-FAB",
        "role": "classical_antibody_antigen",
        "pdbs": ("1MHP",),
        "construct": "Crystal 1MHP. Antigen chain A. Antibody chains H and L.",
        "evidence": "Antigen_chain1=A and L1 match local PDB chain A (184 residues). Protein-1 is Integrin-I, so the antigen name is not assumed from Protein-1/Protein-2 column order.",
        "notes": "Protein-1 is the antigen and Protein-2 is the antibody. Column order was checked against chain sequences.",
    },
    {
        "group_id": "AG_HER2",
        "antigen_name": "human HER2/ERBB2 extracellular region",
        "antibody_name": "Herceptin (fab) and bH1 Fab, not used as a merge key",
        "role": "classical_antibody_antigen",
        "pdbs": ("1N8Z", "3BE1"),
        "construct": "Crystals 1N8Z chain C and 3BE1 chain A. Both are numbered residues 1-607. Built coordinates are 581 and 583 residues because different loops lack coordinates.",
        "evidence": "Every residue number present in both local PDB files has the same amino acid. Positions present in only one file are unmodeled loops, including Leu101, 303-305, and the 361-365 stretch. Merge uses residue numbers, not a sequence-identity threshold and not the HER2 name alone.",
        "notes": "Coordinate sequences are not exact substrings of each other. They stay one antigen entity because the shared residue numbers match.",
        "pdb_notes": {
            "1N8Z": "Antigen chain C. Antibody is Herceptin Fab, chains H and L. Protein-1 is hu-HER2.",
            "3BE1": "Antigen chain A. Antibody is bH1 Fab. Protein-2 is HER2/ErbB-2.",
        },
    },
    {
        "group_id": "AG_FCGR3_1T83",
        "antigen_name": "hu-FCgIIIb",
        "antibody_name": "Fc-IGG1",
        "role": "fc_receptor_complex",
        "pdbs": ("1T83",),
        "construct": "Crystal 1T83. Antigen-side chain C is 167 residues. Chains A and B are the Fc dimer (local residue numbers start at 235 and 232).",
        "evidence": "Antigen_chain1=C and L1 match local chain C. CSV Protein-1 is hu-FCgIIIb. Fc chains occupy the heavy/light columns but are not an antibody Fv.",
        "notes": "Fc-receptor complex, not a classical antibody-antigen pair. The held-out entity is the receptor on the dataset antigen-chain side. It is not merged with 3WJJ.",
    },
    {
        "group_id": "AG_EGFR",
        "antigen_name": "EGFR",
        "antibody_name": "Cetuximab / hu225",
        "role": "classical_antibody_antigen",
        "pdbs": ("1YY9", "HM_1YY9"),
        "construct": "Crystal 1YY9 antigen chain A is 613 residues. HM_1YY9 chain A is 201 residues and is an exact substring of the crystal chain.",
        "evidence": "Both entries are named EGFR. HM antigen sequence is an exact substring of 1YY9 L1, and both match local PDB chain A. HM_ prefix is kept in the PDB id.",
        "notes": "Cetuximab and hu225 are different antibodies and do not define the group. The HM construct stays with the crystal EGFR entity.",
    },
    {
        "group_id": "AG_HPR",
        "antigen_name": "HPr",
        "antibody_name": "Jel42 antibody",
        "role": "classical_antibody_antigen",
        "pdbs": ("2JEL",),
        "construct": "Crystal 2JEL. Antigen chain C. Antibody chains B and A, in that heavy/light order.",
        "evidence": "Protein-2 is HPr and Antigen_chain1=C. Local PDB chain C matches L1. Heavy_chain=B and Light_chain=A match R1 and R2.",
        "notes": "Heavy/light chain order follows the CSV, not alphabetical order.",
    },
    {
        "group_id": "AG_GP120_2NY7",
        "antigen_name": "HIV-1 gp120 (2NY7 construct)",
        "antibody_name": "b12 Ab",
        "role": "classical_antibody_antigen",
        "pdbs": ("2NY7",),
        "construct": "Crystal 2NY7. Antigen chain G is 290 residues. Not merged with 3NGB.",
        "evidence": "Protein-1 is HIV-1 gp120 and L1 matches local chain G. The 3NGB gp120-named chain is not identical and is not an exact substring.",
        "notes": "Shared antigen name with 3NGB was not used to merge distinct wild-type sequences.",
    },
    {
        "group_id": "AG_BONTA1",
        "antigen_name": "Bont/A1",
        "antibody_name": "mAb CR1 and mAb AR2, not used as a merge key",
        "role": "classical_antibody_antigen",
        "pdbs": ("2NYY", "2NZ9"),
        "construct": "Crystals 2NYY and 2NZ9. Antigen chain A is the same 1267-residue BoNT/A1 sequence. Antibodies differ.",
        "evidence": "Protein-1 is Bont/A1 in both. Antigen WT sequences are identical and match local PDB chain A.",
        "notes": "Not merged with BoNT/A2 homology models. Those share the CR1 or AR2 antibody sequences, and antibody sharing is not a reason to merge antigens.",
    },
    {
        "group_id": "AG_MTSP1",
        "antigen_name": "MT-SP1",
        "antibody_name": "Fab Inhibitor E2 and Fab Inhibitor S4",
        "role": "classical_antibody_antigen",
        "pdbs": ("3BN9", "3NPS"),
        "construct": "Crystals 3BN9 and 3NPS. Antigen chain A is the same 241-residue sequence.",
        "evidence": "Protein-1 is MT-SP1. Antigen WT sequences are identical and match local PDB chain A.",
        "notes": "Not merged with HM_3BN9 Epithin. The Fab E2 antibody sequence is shared, and that sharing is ignored for antigen grouping.",
    },
    {
        "group_id": "AG_SH2_3K2M",
        "antigen_name": "SH2 Domain",
        "antibody_name": "Monobody HA4/Abl1",
        "role": "non_antibody_complex",
        "pdbs": ("3K2M",),
        "construct": "Crystal 3K2M. Antigen-side chain D. Chain A is a monobody, not an antibody.",
        "evidence": "Protein-2 is SH2 Domain and Antigen_chain1=D. Local PDB chain D matches L1.",
        "notes": "Not a classical antibody-antigen complex.",
    },
    {
        "group_id": "AG_GP120_3NGB",
        "antigen_name": "HIV-1 gp120 (3NGB construct)",
        "antibody_name": "VRC01",
        "role": "classical_antibody_antigen",
        "pdbs": ("3NGB",),
        "construct": "Crystal 3NGB. Antigen chain G is 347 residues. Not merged with 2NY7.",
        "evidence": "Protein-2 is HIV-1 gp120 and L1 matches local chain G. Not an exact substring of the 2NY7 gp120 chain.",
        "notes": "Separate gp120 construct from AG_GP120_2NY7.",
    },
    {
        "group_id": "AG_FCGR2B_3WJJ",
        "antigen_name": "FcgRIIb",
        "antibody_name": "IIb-sel Fc(P238D)",
        "role": "fc_receptor_complex",
        "pdbs": ("3WJJ",),
        "construct": "Crystal 3WJJ. Antigen-side chain C is 164 residues. Chains A and B follow Fc numbering (local residues start at 233 and 238).",
        "evidence": "Antigen_chain1=C and L1 match local chain C. Protein-2 is FcgRIIb.",
        "notes": "Fc-receptor complex. Not merged with 1T83. The Fc side is not treated as the antigen.",
    },
    {
        "group_id": "AG_BONTA2",
        "antigen_name": "Bont/A2",
        "antibody_name": "mAb CR1 and mAb AR2, not used as a merge key",
        "role": "classical_antibody_antigen",
        "pdbs": ("HM_2NYY", "HM_2NZ9"),
        "construct": "Homology models HM_2NYY and HM_2NZ9. Antigen chain A is the same 1294-residue BoNT/A2 sequence. HM_ prefix is retained.",
        "evidence": "Protein-1 is Bont/A2. Antigen sequences are identical and match local PDB chain A. They are not exact substrings of BoNT/A1.",
        "notes": "A1 and A2 stay apart despite shared antibody sequences.",
    },
    {
        "group_id": "AG_EPITHIN",
        "antigen_name": "Epithin",
        "antibody_name": "Fab Inhibitor E2",
        "role": "classical_antibody_antigen",
        "pdbs": ("HM_3BN9",),
        "construct": "Homology model HM_3BN9. Antigen chain A is 241 residues and is not the MT-SP1 sequence. HM_ prefix is retained.",
        "evidence": "Protein-1 is Epithin. L1 matches local chain A and is not identical to MT-SP1.",
        "notes": "Same Fab E2 sequence as 3BN9. Antibody identity was not used to merge Epithin with MT-SP1.",
    },
    {
        "group_id": "AG_TGFB1",
        "antigen_name": "TGF-beta1",
        "antibody_name": "TGF-beta type-II receptor",
        "role": "non_antibody_complex",
        "pdbs": ("HM_1KTZ",),
        "construct": "Homology model HM_1KTZ. The ligand is chain A. Chain B, residues 25-130, is the same type-II receptor fragment as 1KTZ and is not the grouped entity. HM_ prefix is retained.",
        "evidence": "Chain A carries TGF-beta1 motifs (KDLGWKWIHEP, CLGPCPYIW, ALEPLPIV, SNMIVR). It is not the 1KTZ TGF-beta3 ligand. Chain B matches the TGF-beta type II receptor fragment. The CSV Antigen_chain column points at the receptor.",
        "notes": "Not grouped with 1KTZ. Identical receptor sequences are not a reason to merge different cytokine isoforms.",
    },
)

# Pairs that must remain distinct. This blocks an accidental name-based merge.
M1101_MUST_STAY_SEPARATE: tuple[tuple[str, str], ...] = (
    ("2NY7", "3NGB"),
    ("1KTZ", "HM_1KTZ"),
    ("2NYY", "HM_2NYY"),
    ("2NZ9", "HM_2NZ9"),
    ("3BN9", "HM_3BN9"),
    ("3NPS", "HM_3BN9"),
    ("1T83", "3WJJ"),
)

# Same biological construct whose deposited coordinates differ only by unmodeled loops.
# Checked by residue number, not by a sequence-identity cutoff.
M1101_NUMBERED_CONSTRUCT_PAIRS: tuple[tuple[str, str], ...] = (
    ("1N8Z", "3BE1"),
)

# CSV Antigen_chain points at the receptor in these entries. The ligand is the storage-left chain.
M1101_ANTIGEN_IS_STORAGE_LEFT = frozenset({"1KTZ", "HM_1KTZ"})

SUBSETS = ("outer_train", "inner_train", "inner_val", "outer_eval")
MODEL_INPUT_COLUMNS = (
    "Partners",
    "Mutation",
    "Mutation(source)",
    "mutation(source)",
    *legacy.WT_LEFT_COLS,
    *legacy.WT_RIGHT_COLS,
    "R1m", "R2m", "R3m", "R4m", "R5m", "R6m",
    "L1m", "L2m", "L3m",
)

VERSION_CANDIDATES = (
    ("M1101_official_used", Path("/data/data_split/M1101.csv"), True),
    ("M1101_SAMAffinity_not_used", Path("/data/external/SAMAffinity/data/M1101.csv"), False),
    ("M1101_GeoPPI_not_used", Path("/data/external/GeoPPI/data/M1101.csv"), False),
    ("S1131_official_used", Path("/data/data_split/S1131.csv"), True),
    ("S1131_SAMAffinity_not_used", Path("/data/external/SAMAffinity/data/S1131.csv"), False),
    ("S1131_GeoPPI_not_used", Path("/data/external/GeoPPI/data/S1131.csv"), False),
)


class InfeasibleSplit(RuntimeError):
    """The requested fold count cannot be built without breaking an entity."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def require_empty_directory(path: Path) -> None:
    if path.exists():
        contents = sorted(item.name for item in path.iterdir())
        if contents:
            raise FileExistsError(
                f"Output directory already contains files: {path}. "
                "Refusing to delete or overwrite them. "
                f"First entries: {contents[:8]}"
            )
    path.mkdir(parents=True, exist_ok=True)


def read_ca_numbered(path: Path, chain: str) -> list[tuple[int, str]]:
    """CA residues as (residue number, one-letter code). Insertion codes are rejected."""
    residues: list[tuple[int, str]] = []
    seen: set[str] = set()
    for line in path.open("r", encoding="utf-8", errors="replace"):
        if not line.startswith("ATOM") or line[12:16].strip() != "CA" or line[21].strip() != chain:
            continue
        residue_key = line[22:27]
        if residue_key in seen:
            continue
        seen.add(residue_key)
        if line[26].strip():
            raise ValueError(f"{path.name} chain {chain} has an insertion code at {residue_key!r}.")
        residues.append((int(line[22:26]), AA3_TO_1.get(line[17:20].strip(), "X")))
    return residues


def same_numbered_construct(path_a: Path, chain_a: str, path_b: Path, chain_b: str) -> tuple[bool, str]:
    """True when shared residue numbers match and other positions are unmodeled."""
    first = read_ca_numbered(path_a, chain_a)
    second = read_ca_numbered(path_b, chain_b)
    if not first or not second:
        return False, "missing CA residues"
    span_a = (first[0][0], first[-1][0])
    span_b = (second[0][0], second[-1][0])
    if span_a != span_b:
        return False, f"residue spans differ: {span_a} versus {span_b}"
    map_a = {number: amino for number, amino in first}
    map_b = {number: amino for number, amino in second}
    if len(map_a) != len(first) or len(map_b) != len(second):
        return False, "duplicate residue numbers"
    mismatches = sorted(number for number in set(map_a) & set(map_b) if map_a[number] != map_b[number])
    if mismatches:
        return False, f"shared residue numbers disagree at {mismatches[:8]}"
    only_a = sorted(set(map_a) - set(map_b))
    only_b = sorted(set(map_b) - set(map_a))
    return True, (
        f"shared numbered span {span_a[0]}-{span_a[1]}; "
        f"unmodeled in {path_a.stem} {only_a or 'none'}; "
        f"unmodeled in {path_b.stem} {only_b or 'none'}"
    )


def read_ca_numbered(path: Path, chain: str) -> list[tuple[int, str]]:
    """CA residues for one chain as (residue number, one-letter code)."""
    residues: list[tuple[int, str]] = []
    seen: set[str] = set()
    for line in path.open("r", encoding="utf-8", errors="replace"):
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        if (line[21].strip() or "_") != chain:
            continue
        residue_key = line[22:27]
        if residue_key in seen:
            continue
        seen.add(residue_key)
        if line[26].strip():
            raise ValueError(f"{path} chain {chain} has insertion code at {residue_key}")
        residues.append((int(line[22:26]), AA3_TO_1.get(line[17:20].strip(), "X")))
    return residues


def same_numbered_construct(path_left: Path, chain_left: str, path_right: Path, chain_right: str) -> tuple[bool, str]:
    """True when shared residue numbers match and differences are unmodeled loops."""
    left = read_ca_numbered(path_left, chain_left)
    right = read_ca_numbered(path_right, chain_right)
    if not left or not right:
        return False, "one chain has no CA residues"
    left_span = (left[0][0], left[-1][0])
    right_span = (right[0][0], right[-1][0])
    if left_span != right_span:
        return False, f"residue spans differ: {left_span} versus {right_span}"
    left_map = {number: amino for number, amino in left}
    right_map = {number: amino for number, amino in right}
    if len(left_map) != len(left) or len(right_map) != len(right):
        return False, "duplicate residue numbers"
    shared = sorted(set(left_map) & set(right_map))
    mismatches = [number for number in shared if left_map[number] != right_map[number]]
    if mismatches:
        return False, f"shared residue numbers differ at {mismatches[:8]}"
    only_left = sorted(set(left_map) - set(right_map))
    only_right = sorted(set(right_map) - set(left_map))
    return True, (
        f"residue span {left_span[0]}-{left_span[1]}; "
        f"unmodeled in {path_left.stem}:{only_left or 'none'}; "
        f"unmodeled in {path_right.stem}:{only_right or 'none'}"
    )


def read_ca_sequences(path: Path) -> dict[str, str]:
    chains: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for line in path.open("r", encoding="utf-8", errors="replace"):
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        chain = line[21].strip() or "_"
        residue_key = line[22:27]
        if residue_key in seen[chain]:
            continue
        seen[chain].add(residue_key)
        chains[chain].append(AA3_TO_1.get(line[17:20].strip(), "X"))
    return {chain: "".join(residues) for chain, residues in chains.items()}


def nonempty_chains(values: Iterable[object]) -> list[str]:
    chains = []
    for value in values:
        text = legacy.clean_text(value)
        if text:
            chains.append(text)
    return chains


def side_sequences(row: pd.Series, columns: Sequence[str]) -> list[str]:
    sequences = []
    for column in columns:
        sequence = legacy.clean_sequence(row[column]) if column in row.index else ""
        if sequence:
            sequences.append(sequence)
    return sequences


def _chains_contained(fragment: Sequence[str], parent: Sequence[str]) -> bool:
    """Every whole fragment chain is an exact substring of some parent chain."""
    if not fragment:
        return False
    return all(
        any(chain and (chain == other or chain in other) for other in parent)
        for chain in fragment
    )


def sequence_relation(left: Sequence[str], right: Sequence[str]) -> str:
    """Exact identity or whole-chain exact containment. No similarity cutoff."""
    left_t = tuple(left)
    right_t = tuple(right)
    if left_t == right_t:
        return "identical"
    if _chains_contained(left_t, right_t) or _chains_contained(right_t, left_t):
        return "exact_substring"
    return "distinct"


def pdb_catalog() -> dict[str, dict[str, object]]:
    catalog: dict[str, dict[str, object]] = {}
    for group in M1101_ANTIGEN_GROUPS:
        pdb_notes = dict(group.get("pdb_notes", {}))
        for pdb in group["pdbs"]:
            if pdb in catalog:
                raise RuntimeError(f"PDB {pdb} is assigned to more than one antigen group.")
            catalog[str(pdb)] = {
                "group_id": group["group_id"],
                "antigen_name": group["antigen_name"],
                "antibody_name": group["antibody_name"],
                "role": group["role"],
                "construct": group["construct"],
                "evidence": group["evidence"],
                "notes": " ".join(
                    part for part in (str(group["notes"]), str(pdb_notes.get(pdb, ""))) if part
                ),
            }
    return catalog


def _repair_note(raw_pdb: str, normalized_pdb: str, fallback_id: str) -> str:
    raw = raw_pdb.upper()
    if raw == normalized_pdb:
        return "unchanged"
    if legacy.CORRUPTED_SCIENTIFIC_RE.fullmatch(raw):
        match = legacy.PDB_FROM_ID_RE.match(fallback_id.upper())
        if match and match.group(1).upper() == normalized_pdb:
            return f"scientific_notation_repaired_from_mutation_id:{fallback_id}"
        if legacy.DIRECT_REPAIR_RE.fullmatch(raw):
            return "scientific_notation_repaired_by_direct_digit_rule"
    return "normalized_by_existing_rule"


def load_master(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    raw = pd.read_csv(csv_path, low_memory=False)
    raw_pdb = raw["PDB"].map(legacy.clean_text)
    id_column = legacy.resolve_id_column(raw)
    fallback = raw[id_column] if id_column else pd.Series([""] * len(raw))
    frame, _repaired = legacy.normalize_pdb_column(raw)
    repair = [
        _repair_note(old, new, fb)
        for old, new, fb in zip(raw_pdb, frame["PDB"], fallback)
    ]
    frame = frame.reset_index(drop=True)
    return frame, raw_pdb.reset_index(drop=True), pd.Series(repair, dtype="string")


def _assert_wt_constant(frame: pd.DataFrame, dataset_name: str) -> None:
    columns = [*legacy.WT_LEFT_COLS, *legacy.WT_RIGHT_COLS]
    for (pdb, partners), group in frame.groupby(["PDB", "Partners"], sort=False):
        for column in columns:
            if column not in group.columns:
                continue
            values = {legacy.clean_sequence(value) for value in group[column]}
            values.discard("")
            if len(values) > 1:
                raise ValueError(
                    f"{dataset_name}/{pdb}__{partners}: multiple wild-type sequences in {column}."
                )


def _match_pdb_chains(
    pdb_sequences: dict[str, str],
    chains: Sequence[str],
    sequences: Sequence[str],
    pdb: str,
    side: str,
) -> None:
    if len(chains) != len(sequences):
        raise ValueError(
            f"{pdb} {side}: {len(chains)} chain ids but {len(sequences)} sequences."
        )
    for chain, sequence in zip(chains, sequences):
        observed = pdb_sequences.get(chain, "")
        if observed != sequence:
            raise ValueError(
                f"{pdb} {side} chain {chain}: local PDB CA sequence does not match the CSV "
                f"(pdb={len(observed)} csv={len(sequence)})."
            )


def build_m1101_mapping(frame: pd.DataFrame, raw_pdb: pd.Series, repair: pd.Series, wt_dir: Path) -> pd.DataFrame:
    catalog = pdb_catalog()
    _assert_wt_constant(frame, "M1101")
    per_pdb_sequences: dict[str, tuple[list[str], list[str]]] = {}
    pdb_cache: dict[str, dict[str, str]] = {}
    rows: list[dict[str, object]] = []
    issues: list[str] = []

    for position, record in frame.iterrows():
        pdb = legacy.clean_text(record["PDB"]).upper()
        partners = legacy.clean_text(record["Partners"])
        antibody_chains = nonempty_chains((record.get("Heavy_chain"), record.get("Light_chain")))
        antigen_chains = nonempty_chains((record.get("Antigen_chain1"), record.get("Antigen_chain2")))
        antibody_seqs = side_sequences(record, legacy.WT_LEFT_COLS)
        antigen_seqs = side_sequences(record, legacy.WT_RIGHT_COLS)
        if pdb in M1101_ANTIGEN_IS_STORAGE_LEFT:
            antibody_chains, antigen_chains = antigen_chains, antibody_chains
            antibody_seqs, antigen_seqs = antigen_seqs, antibody_seqs
        status = "confirmed"
        notes: list[str] = []
        entry = catalog.get(pdb)
        if entry is None:
            status = "unresolved"
            issues.append(f"{pdb} has no confirmed antigen catalog entry")
            entry = {
                "group_id": "",
                "antigen_name": "",
                "antibody_name": "",
                "role": "",
                "construct": "",
                "evidence": "",
                "notes": "No catalog entry. Refusing to invent a singleton antigen group.",
            }
        if len(antibody_chains) != len(antibody_seqs) or len(antigen_chains) != len(antigen_seqs):
            status = "unresolved"
            issues.append(
                f"{pdb}: chain-count mismatch antibody {antibody_chains}/{len(antibody_seqs)} "
                f"antigen {antigen_chains}/{len(antigen_seqs)}"
            )
        pdb_path = wt_dir / f"{pdb}.pdb"
        if not pdb_path.is_file():
            status = "unresolved"
            issues.append(f"Missing local PDB file {pdb_path}")
        elif status != "unresolved":
            try:
                if pdb not in pdb_cache:
                    pdb_cache[pdb] = read_ca_sequences(pdb_path)
                _match_pdb_chains(pdb_cache[pdb], antibody_chains, antibody_seqs, pdb, "antibody-side")
                _match_pdb_chains(pdb_cache[pdb], antigen_chains, antigen_seqs, pdb, "antigen-side")
            except ValueError as exc:
                status = "unresolved"
                issues.append(str(exc))
        per_pdb_sequences.setdefault(pdb, (antibody_seqs, antigen_seqs))
        if entry.get("notes"):
            notes.append(str(entry["notes"]))
        antibody_entity = "ABSEQ_" + short_hash("|".join(antibody_seqs)) if antibody_seqs else ""
        rows.append(
            {
                "record_uid": legacy.record_uid("M1101", int(position)),
                "source_row_index": int(position),
                "original_mutation_id": legacy.clean_text(record.get("ID")),
                "complete_pdb_id": pdb,
                "complex_id": legacy.complex_key_from_values(pdb, partners),
                "antibody_chains": ",".join(antibody_chains),
                "antigen_chains": ",".join(antigen_chains),
                "antibody_entity_id": antibody_entity,
                "antigen_entity_id": entry["group_id"] if status != "unresolved" else "",
                "antigen_group_id": entry["group_id"] if status != "unresolved" else "",
                "antigen_name": entry["antigen_name"],
                "species": "not_annotated",
                "construct_information": entry["construct"],
                "mapping_status": status,
                "evidence_source": entry["evidence"],
                "mapping_notes": " ".join(notes),
                "pdb_raw": raw_pdb.iloc[int(position)],
                "pdb_repair": repair.iloc[int(position)],
                "biological_role": entry["role"],
                "antibody_name": entry["antibody_name"],
                "antigen_sequence_sha256": sha256_text("|".join(antigen_seqs)),
                "antibody_sequence_sha256": sha256_text("|".join(antibody_seqs)),
            }
        )

    mapping = pd.DataFrame(rows)
    _validate_m1101_groups(mapping, per_pdb_sequences, issues, wt_dir)
    if issues:
        mapping.attrs["issues"] = issues
        mapping["blocking_issues"] = " | ".join(issues)
    return mapping


def _validate_m1101_groups(
    mapping: pd.DataFrame,
    per_pdb_sequences: dict[str, tuple[list[str], list[str]]],
    issues: list[str],
    wt_dir: Path,
) -> None:
    by_group: dict[str, list[str]] = defaultdict(list)
    for pdb, group_id in (
        mapping.loc[mapping["antigen_group_id"].astype(str) != "", ["complete_pdb_id", "antigen_group_id"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        by_group[str(group_id)].append(str(pdb))

    chain_to_groups: dict[str, set[str]] = defaultdict(set)
    antigen_chain_by_pdb = (
        mapping.groupby("complete_pdb_id")["antigen_chains"].first().astype(str).to_dict()
    )
    numbered_pairs = {frozenset(pair) for pair in M1101_NUMBERED_CONSTRUCT_PAIRS}
    for group_id, pdbs in by_group.items():
        sequences = {pdb: per_pdb_sequences[pdb][1] for pdb in pdbs if pdb in per_pdb_sequences}
        pdb_list = sorted(sequences)
        for index, pdb in enumerate(pdb_list):
            for other in pdb_list[index + 1 :]:
                relation = sequence_relation(sequences[pdb], sequences[other])
                if relation == "distinct" and frozenset((pdb, other)) in numbered_pairs:
                    left_chain = antigen_chain_by_pdb.get(pdb, "").split(",")[0]
                    right_chain = antigen_chain_by_pdb.get(other, "").split(",")[0]
                    ok, detail = same_numbered_construct(
                        wt_dir / f"{pdb}.pdb",
                        left_chain,
                        wt_dir / f"{other}.pdb",
                        right_chain,
                    )
                    if not ok:
                        issues.append(f"{group_id}: {pdb} and {other} failed residue-number check: {detail}")
                elif relation == "distinct":
                    issues.append(
                        f"{group_id}: {pdb} and {other} antigen sequences are not identical "
                        "and neither construct's chains are exact substrings of the other."
                    )
            for chain in sequences[pdb]:
                chain_to_groups[chain].add(group_id)

    for chain, groups in chain_to_groups.items():
        if len(groups) > 1:
            issues.append(
                "Identical antigen-chain WT sequence appears in multiple groups: "
                + ", ".join(sorted(groups))
            )

    for left_pdb, right_pdb in M1101_MUST_STAY_SEPARATE:
        if left_pdb not in per_pdb_sequences or right_pdb not in per_pdb_sequences:
            issues.append(f"Separation check missing PDB {left_pdb} or {right_pdb}")
            continue
        relation = sequence_relation(per_pdb_sequences[left_pdb][1], per_pdb_sequences[right_pdb][1])
        if relation != "distinct":
            issues.append(
                f"Refusing merge check failed: {left_pdb} and {right_pdb} are {relation}."
            )
        groups = set(mapping.loc[mapping["complete_pdb_id"].isin([left_pdb, right_pdb]), "antigen_group_id"])
        if len(groups) < 2:
            issues.append(f"{left_pdb} and {right_pdb} were placed in the same antigen group.")

    if issues:
        mapping.attrs["blocked"] = True


def load_skempi_names(
    path: Path,
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[tuple[str, ...], tuple[str, ...]]]]:
    table = pd.read_csv(path, sep=";")
    names: dict[str, tuple[str, str]] = {}
    conflicts: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for key, group in table.groupby(table["#Pdb"].astype(str).str.upper(), sort=False):
        protein_1 = tuple(sorted({legacy.clean_text(value) for value in group["Protein 1"] if legacy.clean_text(value)}))
        protein_2 = tuple(sorted({legacy.clean_text(value) for value in group["Protein 2"] if legacy.clean_text(value)}))
        if len(protein_1) == 1 and len(protein_2) == 1:
            names[str(key)] = (protein_1[0], protein_2[0])
        else:
            conflicts[str(key)] = (protein_1, protein_2)
    return names, conflicts


class _UnionFind:
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


def build_s1131_mapping(
    frame: pd.DataFrame,
    raw_pdb: pd.Series,
    repair: pd.Series,
    wt_dir: Path,
    skempi_csv: Path,
    partner_side: str,
) -> pd.DataFrame:
    if partner_side not in {"LEFT", "RIGHT"}:
        raise ValueError(f"Unsupported partner side: {partner_side}")
    if not skempi_csv.is_file():
        raise FileNotFoundError(f"SKEMPI table not found: {skempi_csv}")

    _assert_wt_constant(frame, "S1131")
    skempi_names, skempi_conflicts = load_skempi_names(skempi_csv)

    complex_rows: list[dict[str, object]] = []
    for (pdb, partners), group in frame.groupby(["PDB", "Partners"], sort=False):
        first = group.iloc[0]
        left_chains = nonempty_chains((first.get("Heavy_chain"), first.get("Light_chain")))
        right_chains = nonempty_chains((first.get("Antigen_chain1"), first.get("Antigen_chain2")))
        left_seqs = side_sequences(first, legacy.WT_LEFT_COLS)
        right_seqs = side_sequences(first, legacy.WT_RIGHT_COLS)
        pdb_text = legacy.clean_text(pdb).upper()
        partner_text = legacy.clean_text(partners)
        expected = tuple(part for part in partner_text.split("_") if part)
        observed = tuple(left_chains + right_chains)
        skempi_key = f"{pdb_text}_{partner_text}".upper()
        pdb_path = wt_dir / f"{pdb_text}.pdb"
        chain_ok = True
        chain_note = ""
        if not pdb_path.is_file():
            chain_ok = False
            chain_note = f"Missing local PDB {pdb_path}"
        else:
            pdb_sequences = read_ca_sequences(pdb_path)
            try:
                _match_pdb_chains(pdb_sequences, left_chains, left_seqs, pdb_text, "left")
                _match_pdb_chains(pdb_sequences, right_chains, right_seqs, pdb_text, "right")
            except ValueError as exc:
                chain_ok = False
                chain_note = str(exc)
        name_conflict = skempi_key in skempi_conflicts
        if name_conflict:
            protein_1 = " || ".join(skempi_conflicts[skempi_key][0])
            protein_2 = " || ".join(skempi_conflicts[skempi_key][1])
            name_note = (
                f"SKEMPI v2 rows for {skempi_key} do not agree on a single protein label. "
                f"Protein 1 labels={skempi_conflicts[skempi_key][0]}; "
                f"Protein 2 labels={skempi_conflicts[skempi_key][1]}. "
                "The sequence entity is kept, but the name is candidate rather than confirmed."
            )
        elif skempi_key in skempi_names:
            protein_1, protein_2 = skempi_names[skempi_key]
            name_note = f"SKEMPI v2 #Pdb {skempi_key} Protein 1 = left, Protein 2 = right."
        else:
            protein_1, protein_2 = "", ""
            name_note = (
                f"{skempi_key} is absent from local SKEMPI v2. "
                "The chain sequences still define the entity; the protein name is not confirmed."
            )
        complex_rows.append(
            {
                "pdb": pdb_text,
                "partners": partner_text,
                "left_chains": left_chains,
                "right_chains": right_chains,
                "left_seqs": left_seqs,
                "right_seqs": right_seqs,
                "expected_chains": expected,
                "observed_chains": observed,
                "protein_left": protein_1,
                "protein_right": protein_2,
                "name_note": name_note,
                "chain_ok": chain_ok,
                "chain_note": chain_note,
                "skempi_key": skempi_key,
                "name_conflict": name_conflict,
            }
        )

    side_key = "left" if partner_side == "LEFT" else "right"
    union = _UnionFind()
    sequence_names: dict[str, set[str]] = defaultdict(set)
    for row in complex_rows:
        sequences = tuple(row[f"{side_key}_seqs"])
        sequence_key = "|".join(sequences)
        union.add(sequence_key)
        name = legacy.clean_text(row[f"protein_{side_key}"])
        if name:
            sequence_names[sequence_key].add(name)

    keys = sorted(sequence_names)
    for index, first in enumerate(keys):
        for second in keys[index + 1 :]:
            names_first = sequence_names[first]
            names_second = sequence_names[second]
            if not names_first or names_first != names_second:
                continue
            first_chains = first.split("|") if first else []
            second_chains = second.split("|") if second else []
            if sequence_relation(first_chains, second_chains) == "exact_substring":
                union.union(first, second)

    components: dict[str, list[str]] = defaultdict(list)
    for key in sorted(union.parent):
        components[union.find(key)].append(key)

    component_meta: dict[str, dict[str, object]] = {}
    for root, members in components.items():
        names: set[str] = set()
        for member in members:
            names.update(sequence_names.get(member, set()))
        signature = "\n".join(sorted(members))
        group_id = f"PG{partner_side}_{short_hash(signature)}"
        component_meta[root] = {
            "group_id": group_id,
            "names": sorted(names),
            "n_sequences": len(members),
        }

    rows: list[dict[str, object]] = []
    complex_lookup = {(row["pdb"], row["partners"]): row for row in complex_rows}
    for position, record in frame.iterrows():
        pdb = legacy.clean_text(record["PDB"]).upper()
        partners = legacy.clean_text(record["Partners"])
        info = complex_lookup[(pdb, partners)]
        sequences = tuple(info[f"{side_key}_seqs"])
        chains = list(info[f"{side_key}_chains"])
        sequence_key = "|".join(sequences)
        meta = component_meta[union.find(sequence_key)]
        names = list(meta["names"])
        status = "confirmed"
        notes = [
            str(info["name_note"]),
            "Heavy_chain and Antigen_chain column names in S1131 are storage labels, not antibody or antigen roles.",
            "Partner groups are built only within this side. Left and right entities are not unioned.",
            "Constructs are merged only when the protein name is identical and one wild-type sequence is an exact substring of the other, or when the full wild-type sequences are identical.",
        ]
        if info["expected_chains"] != info["observed_chains"]:
            status = "unresolved"
            notes.append(
                f"Partners {info['expected_chains']} does not match chain columns {info['observed_chains']}."
            )
        if not info["chain_ok"]:
            status = "unresolved"
            notes.append(str(info["chain_note"]))
        if info["name_conflict"]:
            status = "candidate" if status == "confirmed" else status
        if not names:
            status = "candidate" if status == "confirmed" else status
            notes.append(
                "Protein name is unresolved. The entity id is the observed wild-type sequence component, "
                "which does not exact-match or exactly contain any named partner sequence on this side. "
                "This is not a PASS mapping."
            )
        elif len(names) > 1:
            status = "candidate" if status == "confirmed" else status
            notes.append("Identical or nested sequences carry more than one SKEMPI protein name: " + " | ".join(names))
        mutation_source = legacy.clean_text(
            record["mutation(source)"] if "mutation(source)" in record.index else record.get("Mutation(source)")
        )
        mutation_chain = mutation_source.split(":", 1)[0] if ":" in mutation_source else ""
        on_side = mutation_chain in set(chains)
        notes.append(
            f"mutation_chain={mutation_chain or 'unknown'}; mutation_on_grouped_side={'yes' if on_side else 'no'}. "
            "Mutation side does not create a new partner group."
        )
        group_id = str(meta["group_id"]) if status != "unresolved" else ""
        rows.append(
            {
                "record_uid": legacy.record_uid("S1131", int(position)),
                "source_row_index": int(position),
                "original_mutation_id": legacy.clean_text(record.get("ID")),
                "complete_pdb_id": pdb,
                "complex_id": legacy.complex_key_from_values(pdb, partners),
                "partner_side": partner_side,
                "partner_chains": ",".join(chains),
                "partner_entity_id": group_id,
                "partner_group_id": group_id,
                "partner_name": " | ".join(names),
                "mapping_status": status,
                "evidence_source": (
                    f"S1131.csv wild-type {side_key} sequence and chain columns; "
                    f"local wt PDB CA match; {info['skempi_key']} in {skempi_csv}."
                ),
                "mapping_notes": " ".join(notes),
                "pdb_raw": raw_pdb.iloc[int(position)],
                "pdb_repair": repair.iloc[int(position)],
                "sequence_sha256": sha256_text(sequence_key),
                "skempi_key": info["skempi_key"],
            }
        )
    mapping = pd.DataFrame(rows)
    return mapping


def assign_outer_groups(group_sizes: dict[str, int], seed: int, n_splits: int) -> dict[str, int]:
    """Greedy record-count balance. Ties use a seeded shuffle of sorted group ids."""
    groups = sorted(group_sizes)
    if len(groups) < n_splits:
        raise InfeasibleSplit(
            f"Cannot build {n_splits} outer folds from {len(groups)} indivisible entity groups. "
            "Refusing to split a group or to silently reduce the fold count. "
            "Pass a smaller --n-splits only if that smaller count is an explicit choice."
        )
    rng = np.random.default_rng(int(seed))
    shuffled = list(groups)
    rng.shuffle(shuffled)
    ordered = sorted(shuffled, key=lambda group: group_sizes[group], reverse=True)
    fold_sizes = [0] * n_splits
    assignment: dict[str, int] = {}
    for group in ordered:
        fold_id = int(np.argmin(np.asarray(fold_sizes)))
        assignment[group] = fold_id
        fold_sizes[fold_id] += int(group_sizes[group])
    return assignment


def assign_inner_groups(
    group_sizes: dict[str, int],
    seed: int,
    fraction: float,
) -> tuple[set[str], set[str]]:
    """Hold out about `fraction` of outer_train records as whole groups."""
    groups = sorted(group_sizes)
    if len(groups) < 2:
        raise InfeasibleSplit(
            "This outer fold has fewer than 2 entity groups in outer_train. "
            "inner_train and inner_val cannot both be non-empty without splitting a group "
            "or using outer_eval for model selection. Refusing that fallback."
        )
    rng = np.random.default_rng(int(seed))
    shuffled = list(groups)
    rng.shuffle(shuffled)
    ordered = sorted(shuffled, key=lambda group: group_sizes[group], reverse=True)
    total = sum(int(size) for size in group_sizes.values())
    target_val = float(fraction) * total
    target_train = total - target_val
    train: list[str] = []
    validation: list[str] = []
    train_n = 0
    val_n = 0
    for index, group in enumerate(ordered):
        size = int(group_sizes[group])
        is_last = index == len(ordered) - 1
        val_ratio = (val_n / target_val) if target_val else 1.0
        train_ratio = (train_n / target_train) if target_train else 1.0
        choose_val = val_ratio < train_ratio
        if is_last and val_n == 0:
            choose_val = True
        elif is_last and train_n == 0:
            choose_val = False
        if choose_val:
            validation.append(group)
            val_n += size
        else:
            train.append(group)
            train_n += size
    if train_n == 0 or val_n == 0:
        raise InfeasibleSplit(
            "Inner split left one side empty. outer_eval will not be used as a substitute."
        )
    return set(train), set(validation)


def derived_inner_seed(seed: int, fold_id: int) -> int:
    """Traceable seed: outer seed, zero-based fold id, and a fixed scale of 1000."""
    return int(seed) * 1000 + int(fold_id)


def _group_column(mapping: pd.DataFrame) -> str:
    if "antigen_group_id" in mapping.columns:
        return "antigen_group_id"
    return "partner_group_id"


def _label_stats(values: pd.Series) -> dict[str, object]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric.to_numpy(dtype=float))]
    if finite.empty:
        return {
            "ddg_n_finite": 0,
            "ddg_mean": np.nan,
            "ddg_std_sample": np.nan,
            "ddg_min": np.nan,
            "ddg_max": np.nan,
        }
    std = float(finite.std(ddof=1)) if len(finite) >= 2 else np.nan
    return {
        "ddg_n_finite": int(len(finite)),
        "ddg_mean": float(finite.mean()),
        "ddg_std_sample": std,
        "ddg_min": float(finite.min()),
        "ddg_max": float(finite.max()),
    }


def _write_subset(frame: pd.DataFrame, indices: Sequence[int], dataset_name: str, path: Path) -> None:
    subset = frame.iloc[list(indices)].copy()
    subset["record_uid"] = [legacy.record_uid(dataset_name, int(index)) for index in subset.index]
    subset.to_csv(path, index=False)


def write_group_summary(mapping: pd.DataFrame, path: Path) -> pd.DataFrame:
    group_column = _group_column(mapping)
    name_column = "antigen_name" if "antigen_name" in mapping.columns else "partner_name"
    rows = []
    for group_id, part in mapping.groupby(group_column, sort=True):
        if not legacy.clean_text(group_id):
            continue
        rows.append(
            {
                "entity_group_id": group_id,
                "entity_name": " | ".join(sorted(set(part[name_column].astype(str)))),
                "n_records": int(len(part)),
                "n_pdb": int(part["complete_pdb_id"].nunique()),
                "n_complex": int(part["complex_id"].nunique()),
                "pdb_list": ",".join(sorted(part["complete_pdb_id"].astype(str).unique())),
                "mapping_status": ",".join(sorted(part["mapping_status"].astype(str).unique())),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["n_records", "entity_group_id"], ascending=[False, True])
    summary.to_csv(path, index=False)
    return summary


def formal_gate(mapping: pd.DataFrame) -> tuple[str, str]:
    statuses = set(mapping["mapping_status"].astype(str))
    group_column = _group_column(mapping)
    if "blocking_issues" in mapping.columns and mapping["blocking_issues"].astype(str).str.strip().ne("").any():
        return ("BLOCKED", str(mapping["blocking_issues"].iloc[0]))
    if (mapping[group_column].astype(str) == "").any() or "unresolved" in statuses:
        return (
            "BLOCKED",
            "At least one record has no confirmed entity group. "
            "Formal folds were not written, and unresolved records were not assigned singleton groups.",
        )
    if statuses - {"confirmed"}:
        return (
            "INCOMPLETE",
            "Every record has a sequence-defined group, but at least one mapping is not confirmed. "
            "Folds are written for inspection and must not be cited as a passed formal split.",
        )
    return ("CONFIRMED_PENDING_AUDIT", "All mappings used for grouping are confirmed. Independent audit still required.")


def build_folds(
    frame: pd.DataFrame,
    mapping: pd.DataFrame,
    dataset_name: str,
    seed: int,
    n_splits: int,
    inner_fraction: float,
    out_dir: Path,
) -> pd.DataFrame:
    group_column = _group_column(mapping)
    groups = mapping[group_column].astype(str)
    if (groups == "").any():
        raise InfeasibleSplit("Refusing to fold records that have no entity group.")
    sizes = groups.value_counts().to_dict()
    print("[GROUPS] entity sizes before assignment")
    for group_id, size in sorted(sizes.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {group_id}\trecords={int(size)}")
    outer = assign_outer_groups({str(key): int(value) for key, value in sizes.items()}, seed, n_splits)
    positions = list(range(len(frame)))
    by_fold_positions: dict[int, list[int]] = {fold_id: [] for fold_id in range(n_splits)}
    for position, group_id in zip(positions, groups.tolist()):
        by_fold_positions[outer[str(group_id)]].append(position)

    assignment_rows = []
    summary_rows = []
    for fold_id in range(n_splits):
        eval_positions = by_fold_positions[fold_id]
        train_positions = [position for position in positions if position not in set(eval_positions)]
        eval_groups = {str(groups.iloc[position]) for position in eval_positions}
        train_group_sizes: dict[str, int] = defaultdict(int)
        for position in train_positions:
            train_group_sizes[str(groups.iloc[position])] += 1
        inner_seed = derived_inner_seed(seed, fold_id)
        inner_train_groups, inner_val_groups = assign_inner_groups(
            dict(train_group_sizes),
            inner_seed,
            inner_fraction,
        )
        overlap = (
            eval_groups & inner_train_groups
            or eval_groups & inner_val_groups
            or inner_train_groups & inner_val_groups
        )
        if overlap:
            raise RuntimeError(f"Entity groups overlap between inner and outer subsets at fold {fold_id}.")
        role_positions = {
            "outer_eval": eval_positions,
            "outer_train": train_positions,
            "inner_train": [position for position in train_positions if str(groups.iloc[position]) in inner_train_groups],
            "inner_val": [position for position in train_positions if str(groups.iloc[position]) in inner_val_groups],
        }
        if set(role_positions["inner_train"]) | set(role_positions["inner_val"]) != set(train_positions):
            raise RuntimeError(f"Inner subsets do not cover outer_train at fold {fold_id}.")
        for subset, subset_positions in role_positions.items():
            _write_subset(frame, subset_positions, dataset_name, out_dir / f"fold_{fold_id}_{subset}.csv")
            part = frame.iloc[subset_positions]
            part_groups = groups.iloc[subset_positions]
            stats = _label_stats(part["ddG"])
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": int(seed),
                    "split": out_dir.name,
                    "fold": int(fold_id),
                    "subset": subset,
                    "n_records": int(len(subset_positions)),
                    "n_entity_groups": int(part_groups.nunique()),
                    "n_pdb": int(part["PDB"].nunique()),
                    "n_complex": int(legacy.complex_keys(part).nunique()),
                    "max_group_records": int(part_groups.value_counts().max()) if len(subset_positions) else 0,
                    "inner_seed": int(inner_seed),
                    "inner_val_fraction_target": float(inner_fraction),
                    "inner_val_fraction_of_outer_train": (
                        len(role_positions["inner_val"]) / len(train_positions) if train_positions else np.nan
                    ),
                    **stats,
                }
            )
        for position in positions:
            group_id = str(groups.iloc[position])
            if outer[group_id] != fold_id:
                continue
            if group_id in eval_groups:
                inner_role = "outer_eval"
            elif group_id in inner_val_groups:
                inner_role = "inner_val"
            else:
                inner_role = "inner_train"
            assignment_rows.append(
                {
                    "record_uid": legacy.record_uid(dataset_name, position),
                    "source_row_index": position,
                    "outer_fold": fold_id,
                    "entity_group_id": group_id,
                    "assignment": inner_role,
                    "inner_seed": inner_seed,
                }
            )
        print(
            f"[FOLD] seed={seed} fold={fold_id} "
            f"outer_eval={len(eval_positions)} groups={len(eval_groups)} "
            f"inner_train={len(role_positions['inner_train'])} "
            f"inner_val={len(role_positions['inner_val'])} "
            f"inner_val_fraction={len(role_positions['inner_val']) / len(train_positions):.3f}"
        )

    assignments = pd.DataFrame(assignment_rows).sort_values(["outer_fold", "source_row_index"])
    assignments.to_csv(out_dir / "outer_assignments.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "fold_summary.csv", index=False)
    return summary


def readme_text(dataset_name: str, split_name: str, gate: str, gate_reason: str) -> str:
    task = {
        "AGCV": "antigen-grouped cross-validation",
        "PGCV_LEFT": "left-partner grouped cross-validation",
        "PGCV_RIGHT": "right-partner grouped cross-validation",
    }[split_name]
    return f"""# {dataset_name} {split_name}

This directory is {task}. Mapping gate: `{gate}`.

{gate_reason}

## Files the model may read

Use the three files below separately. Do not concatenate them into one training set.

- `fold_{{k}}_inner_train.csv`: fit the model.
- `fold_{{k}}_inner_val.csv`: early stopping and checkpoint selection.
- `fold_{{k}}_outer_eval.csv`: final evaluation of that outer fold. Do not use it to stop, select, or refit.
- `fold_{{k}}_outer_train.csv`: `inner_train` plus `inner_val` for the same fold. It is a boundary check, not a third training protocol.

`record_uid` is an audit id (`dataset:row`). Ignore it as a feature. Labels are unfiltered, and ddG units and signs are unchanged.

## Grouping

An entity group from the mapping file is atomic. Every record in a group stays in the same outer fold and the same inner subset.

Assignment uses group id, group size, and a fixed seed. Group ids are sorted, ties are shuffled with that seed, then groups are placed largest first. Labels and model scores are not used. The inner seed is `outer_seed * 1000 + fold_id`. The inner-validation target is 20% of outer-train records; group size can move the realized fraction.

Outer seeds are 0, 42, 1234, 2026, and 1024. Seeds may reassign groups. The five folds, five seeds, and 20% inner split are project settings, not a literature requirement.

On M1101, 1N8Z and 3BE1 share the same HER2 sequence on numbered residues 1-607 and are one antigen entity. 1KTZ and HM_1KTZ share the TGF-beta type II receptor but have different ligands, so they are two groups.

## Source of the idea

The holdout idea follows Li et al., iScience 2026;29(8):116522, and https://github.com/amazon-science/ab-binding-eval. This directory uses only the ideas of holding out a whole antigen system, rotating systems, and separating model selection from the final test. It does not use that paper's model, loss, or split ratios. `PGCV_LEFT` and `PGCV_RIGHT` are partner holdouts, not antigen holdouts, and the two sides are not merged.

## Mapping states

`confirmed`: chain, wild-type sequence, and source note agree. `candidate`: the group is fixed by sequence, but the name is not fully checked. `unresolved`: the entity is unknown; those records are not forced into singleton groups, and formal fold files are not written for them.

`audit_antigen_group_splits.py` rereads the files on disk. Cite a split as formal only when that audit returns `PASS`. Sequence or structure similarity that was not run is uncomputed, not zero. Leaving out an entity does not mean there is no homolog, no interface similarity, or that pretraining never saw the sequence.
"""


def _version_notes(used_csv: Path) -> list[dict[str, object]]:
    notes = []
    for label, path, used in VERSION_CANDIDATES:
        if not path.is_file():
            notes.append({"label": label, "path": str(path), "exists": False, "used": False})
            continue
        notes.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "exists": True,
                "used": bool(used and path.resolve() == used_csv.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    notes.append(
        {
            "label": "data_root",
            "path": "/data",
            "exists": Path("/data").exists(),
            "used": False,
            "note": "Requested legacy server path is not present on this machine.",
        }
    )
    return notes


def write_manifest(
    out_dir: Path,
    *,
    dataset_name: str,
    split_name: str,
    csv_path: Path,
    skempi_csv: Path | None,
    seed: int,
    n_splits: int,
    inner_fraction: float,
    gate: str,
    gate_reason: str,
    mapping_path: Path,
) -> None:
    script_paths = [Path(legacy.__file__).resolve(), Path(__file__).resolve()]
    output_files = []
    for path in sorted(out_dir.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        output_files.append({"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest = {
        "dataset": dataset_name,
        "split": split_name,
        "seed": int(seed),
        "n_splits": int(n_splits),
        "inner_val_fraction_target": float(inner_fraction),
        "inner_seed_rule": "inner_seed = outer_seed * 1000 + zero_based_fold_id",
        "mapping_gate": gate,
        "mapping_gate_reason": gate_reason,
        "formal_status": "PENDING_INDEPENDENT_AUDIT" if gate != "BLOCKED" else "BLOCKED",
        "input_csv": {"path": str(csv_path.resolve()), "sha256": sha256_file(csv_path)},
        "skempi_csv": (
            {"path": str(skempi_csv.resolve()), "sha256": sha256_file(skempi_csv)}
            if skempi_csv is not None and skempi_csv.is_file()
            else None
        ),
        "scripts": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in script_paths
        ],
        "mapping_file": mapping_path.name,
        "version_comparison": _version_notes(csv_path),
        "grouping_rule": (
            "M1101 groups are curated antigen-side entities accepted only after exact wild-type "
            "sequence identity or exact whole-chain substring containment, plus local PDB CA matches. "
            "No sequence-identity threshold is used. Shared antibodies do not merge antigens."
            if split_name == "AGCV"
            else (
                "S1131 groups are one partner side only. Identical wild-type sequences are one entity. "
                "Exact substring containment merges records only when the SKEMPI protein name is identical. "
                "Left and right sides are never unioned. There is no antibody-antigen assumption."
            )
        ),
        "balance_algorithm": (
            "Sort group ids lexicographically, shuffle with numpy default_rng(seed), "
            "then stable-sort by record count descending. Assign each group to the current "
            "smallest fold. argmin breaks remaining size ties toward the lowest fold index. "
            "Labels and model metrics are not used."
        ),
        "inner_validation_rule": (
            "Same greedy rule inside outer_train, using inner_seed. "
            "Target validation size is inner_val_fraction times outer_train records. "
            "Equal relative fill assigns the group to inner_train. "
            "outer_eval is never a fallback validation set."
        ),
        "ignored_by_model_reader": ["record_uid"],
        "limitations": [
            "Sequence-identity clustering beyond exact identity or exact substring was not used and is not computed here.",
            "Structure-similarity search was not run.",
            "Entity holdout does not imply absence of homologs, local interface similarity, or pretraining exposure.",
            "Five folds, five seeds, and the 20% inner validation target are project settings, not MochiBind parameters.",
        ],
        "output_files": output_files,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _mutation_side_counts(frame: pd.DataFrame) -> dict[str, int]:
    source_column = "mutation(source)" if "mutation(source)" in frame.columns else "Mutation(source)"
    counts = {"LEFT": 0, "RIGHT": 0, "OTHER": 0}
    for _, record in frame.iterrows():
        source = legacy.clean_text(record[source_column])
        chain = source.split(":", 1)[0] if ":" in source else ""
        left = set(nonempty_chains((record.get("Heavy_chain"), record.get("Light_chain"))))
        right = set(nonempty_chains((record.get("Antigen_chain1"), record.get("Antigen_chain2"))))
        if chain in left:
            counts["LEFT"] += 1
        elif chain in right:
            counts["RIGHT"] += 1
        else:
            counts["OTHER"] += 1
    return counts


def prepare_split_dir(root: Path, dataset_name: str, seed: int, split_name: str) -> Path:
    path = root / dataset_name / str(seed) / split_name
    require_empty_directory(path)
    return path


def run_one_split(
    *,
    dataset_name: str,
    split_name: str,
    csv_path: Path,
    output_root: Path,
    seeds: Sequence[int],
    n_splits: int,
    inner_fraction: float,
    skempi_csv: Path,
    wt_dir: Path,
) -> tuple[pd.DataFrame, int]:
    frame, raw_pdb, repair = load_master(csv_path)
    if dataset_name == "M1101" and split_name == "AGCV":
        mapping = build_m1101_mapping(frame, raw_pdb, repair, wt_dir)
        skempi_used = None
    elif dataset_name == "S1131" and split_name in {"PGCV_LEFT", "PGCV_RIGHT"}:
        mapping = build_s1131_mapping(
            frame,
            raw_pdb,
            repair,
            wt_dir,
            skempi_csv,
            "LEFT" if split_name.endswith("LEFT") else "RIGHT",
        )
        skempi_used = skempi_csv
    elif dataset_name == "S1131" and split_name == "PGCV":
        counts = _mutation_side_counts(frame)
        reason = (
            "S1131 has no single biological target side. "
            f"Mutation chains fall on the stored left side for {counts['LEFT']} records, "
            f"the stored right side for {counts['RIGHT']} records, and {counts['OTHER']} elsewhere. "
            "Heavy/Antigen column names are not antibody-antigen roles. "
            "Refusing a single PGCV split. Use PGCV_LEFT and PGCV_RIGHT, which are built separately."
        )
        refused = output_root / dataset_name / "PGCV_UNAVAILABLE"
        require_empty_directory(refused)
        (refused / "decision.json").write_text(
            json.dumps({"status": "REFUSED", "reason": reason, "mutation_side_counts": counts}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[REFUSED] {reason}")
        return pd.DataFrame(), 2
    else:
        raise ValueError(f"{split_name} is not defined for {dataset_name}.")

    gate, gate_reason = formal_gate(mapping)
    issues = list(mapping.attrs.get("issues", []))
    if issues and gate != "BLOCKED":
        gate = "BLOCKED"
        gate_reason = "Sequence relation checks failed. " + " | ".join(issues[:8])

    summary_frames = []
    exit_code = 0 if gate != "BLOCKED" else 2
    report_dir = output_root / dataset_name / f"{split_name}_mapping"
    # Mapping diagnostics live with each seed. A blocked split still writes one
    # diagnostic directory so the unresolved rows are on disk.
    if gate == "BLOCKED":
        diagnostic = output_root / dataset_name / f"{split_name}_BLOCKED"
        require_empty_directory(diagnostic)
        mapping.to_csv(diagnostic / ("antigen_mapping.csv" if split_name == "AGCV" else "partner_mapping.csv"), index=False)
        (diagnostic / "unresolved_issues.txt").write_text(
            gate_reason + "\n" + "\n".join(issues) + "\n",
            encoding="utf-8",
        )
        (diagnostic / "README.md").write_text(
            readme_text(dataset_name, split_name if split_name != "PGCV" else "PGCV_LEFT", "BLOCKED", gate_reason),
            encoding="utf-8",
        )
        print(f"[BLOCKED] {dataset_name} {split_name}: {gate_reason}")
        return pd.DataFrame(), 2

    for seed in seeds:
        out_dir = prepare_split_dir(output_root, dataset_name, seed, split_name)
        mapping_name = "antigen_mapping.csv" if split_name == "AGCV" else "partner_mapping.csv"
        mapping_path = out_dir / mapping_name
        mapping.to_csv(mapping_path, index=False)
        summary_name = "antigen_group_summary.csv" if split_name == "AGCV" else "partner_group_summary.csv"
        group_summary = write_group_summary(mapping, out_dir / summary_name)
        print(
            f"[MAP] {dataset_name} {split_name} seed={seed} "
            f"groups={len(group_summary)} gate={gate}"
        )
        fold_summary = build_folds(
            frame,
            mapping,
            dataset_name,
            int(seed),
            n_splits,
            inner_fraction,
            out_dir,
        )
        summary_frames.append(fold_summary)
        (out_dir / f"README_{split_name}.md").write_text(
            readme_text(dataset_name, split_name, gate, gate_reason),
            encoding="utf-8",
        )
        write_manifest(
            out_dir,
            dataset_name=dataset_name,
            split_name=split_name,
            csv_path=csv_path,
            skempi_csv=skempi_used,
            seed=int(seed),
            n_splits=n_splits,
            inner_fraction=inner_fraction,
            gate=gate,
            gate_reason=gate_reason,
            mapping_path=mapping_path,
        )
    if report_dir.exists():
        pass
    return (pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()), exit_code


def _refuse_existing_split(path: Path) -> None:
    """Refuse only the destination split directory, so AGCV can share the RCSV root."""
    if path.exists():
        contents = sorted(item.name for item in path.iterdir())
        if contents:
            raise FileExistsError(
                f"Output directory already contains files: {path}. "
                "Refusing to delete or overwrite them. "
                f"First entries: {contents[:8]}"
            )


def run_entity_cross_validation(args: argparse.Namespace) -> int:
    output_root = Path(args.entity_output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.seeds:
        seeds = [int(seed) for seed in args.seeds]
    elif args.seed is not None:
        seeds = [int(args.seed)]
    else:
        seeds = list(legacy.DEFAULT_SEEDS)
    requested = list(dict.fromkeys(args.entity_cv))
    print("========== entity grouped CV ==========")
    print(f"Output root: {output_root}")
    print(f"Requested:   {requested}")
    print(f"Seeds:       {seeds}")
    print(f"Folds:       {args.n_splits}")
    print("Historical RCSV/PCSV/ICSV files and split_summary.csv are not modified.")

    summaries = []
    exit_code = 0
    for dataset_file in args.datasets:
        csv_path = legacy.resolve_dataset_csv(Path(args.input_dir), dataset_file)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        dataset_name = csv_path.stem
        wt_dir = Path(args.wt_pdb_dir) if args.wt_pdb_dir else csv_path.parent / dataset_name / "wt"
        for split_name in requested:
            supported = {"M1101": {"AGCV"}, "S1131": {"PGCV", "PGCV_LEFT", "PGCV_RIGHT"}}
            if split_name not in supported.get(dataset_name, set()):
                print(f"[SKIP] {split_name} is not defined for {dataset_name}.")
                continue
            for seed in seeds:
                _refuse_existing_split(output_root / dataset_name / str(seed) / split_name)
            summary, code = run_one_split(
                dataset_name=dataset_name,
                split_name=split_name,
                csv_path=csv_path,
                output_root=output_root,
                seeds=seeds,
                n_splits=int(args.n_splits),
                inner_fraction=float(args.inner_val_fraction),
                skempi_csv=Path(args.skempi_csv),
                wt_dir=wt_dir,
            )
            exit_code = max(exit_code, code)
            if not summary.empty:
                summaries.append(summary)

    summary_path = (
        Path(args.entity_summary_path)
        if args.entity_summary_path is not None
        else output_root / "entity_split_summary.csv"
    )
    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if summary_path.exists():
            raise FileExistsError(f"Refusing to overwrite {summary_path}")
        combined.to_csv(summary_path, index=False)
        print(f"[WRITE] {summary_path}")
    print(f"[DONE] entity CV exit={exit_code}")
    return exit_code
