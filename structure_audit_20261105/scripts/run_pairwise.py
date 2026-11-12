#!/usr/bin/env python3
"""Formal two-sided global structural similarity: all-vs-all US-align over
unique WT complexes of one dataset.

For every unordered complex pair (i<j) run:
    USalign wt_i.pdb wt_j.pdb -mm 1 -ter 1
and record bidirectional normalized TM-score (TM1 normalized by structure_1,
TM2 normalized by structure_2), RMSD, aligned length, sequence identity, and
bidirectional coverage = aligned_length / length_of_each_structure.

Self pairs are set analytically (TM=1, coverage=1).

Outputs:
  pairwise/{ds}_complex_lengths.csv
  pairwise/{ds}_pairwise.csv
"""

import os
import sys
import csv
import re
import subprocess
import argparse
import itertools
from multiprocessing import Pool

ROOT = "/data"
DS_ROOT = os.path.join(ROOT, "data_split")
OUT = os.path.join(ROOT, "structure_audit_20261105")
USALIGN = os.path.expanduser("~/.local/bin/USalign")

RE_L1 = re.compile(r"Length of Structure_1:\s*([0-9]+) residues")
RE_L2 = re.compile(r"Length of Structure_2:\s*([0-9]+) residues")
RE_ALN = re.compile(r"Aligned length=\s*([0-9]+), RMSD=\s*([0-9.]+), Seq_ID=n_identical/n_aligned=\s*([0-9.]+)")
RE_TM1 = re.compile(r"TM-score=\s*([0-9.]+) \(normalized by length of Structure_1")
RE_TM2 = re.compile(r"TM-score=\s*([0-9.]+) \(normalized by length of Structure_2")
RE_NAME1 = re.compile(r"Name of Structure_1:\s*(.+?)\s*\(to be")
RE_NAME2 = re.compile(r"Name of Structure_2:\s*(.+)")


def run_one(path1, path2):
    cmd = [USALIGN, path1, path2, "-mm", "1", "-ter", "1"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        out = p.stdout + "\n" + p.stderr
    except Exception as e:
        return None, str(e)
    rec = {}
    m = RE_L1.search(out); rec["len1"] = int(m.group(1)) if m else None
    m = RE_L2.search(out); rec["len2"] = int(m.group(1)) if m else None
    m = RE_ALN.search(out)
    if m:
        rec["aligned_len"] = int(m.group(1)); rec["rmsd"] = float(m.group(2)); rec["seq_id"] = float(m.group(3))
    else:
        rec["aligned_len"] = None; rec["rmsd"] = None; rec["seq_id"] = None
    m = RE_TM1.search(out); rec["tm1"] = float(m.group(1)) if m else None
    m = RE_TM2.search(out); rec["tm2"] = float(m.group(1)) if m else None
    m = RE_NAME1.search(out); rec["name1"] = m.group(1) if m else ""
    m = RE_NAME2.search(out); rec["name2"] = m.group(1).strip() if m else ""
    if rec["tm1"] is None or rec["tm2"] is None:
        return None, out[-500:]
    return rec, None


def self_len(args):
    pdb, path = args
    rec, err = run_one(path, path)
    if rec is None:
        return (pdb, None, None, None, err)
    return (pdb, rec["len1"], rec["name1"], rec.get("seq_id"), None)


def pair_job(args):
    i, j, p1, p2 = args
    rec, err = run_one(p1, p2)
    if rec is None:
        return (i, j, None, None, None, None, None, None, err)
    # coverage relative to each structure's own length
    cov1 = (rec["aligned_len"] / rec["len1"]) if (rec["aligned_len"] and rec["len1"]) else None
    cov2 = (rec["aligned_len"] / rec["len2"]) if (rec["aligned_len"] and rec["len2"]) else None
    return (i, j, rec["len1"], rec["len2"], rec["aligned_len"], rec["rmsd"],
            rec["tm1"], rec["tm2"], None, cov1, cov2, rec["seq_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    ds = args.dataset
    wt_dir = os.path.join(DS_ROOT, "splits", ds, "wt")

    # unique PDBs in canonical order of first appearance
    with open(os.path.join(DS_ROOT, "raw", f"{ds}.csv"), newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    pdbs = []
    seen = set()
    for r in rows:
        p = r["PDB"]
        if p not in seen:
            seen.add(p); pdbs.append(p)

    paths = [os.path.join(wt_dir, p + ".pdb") for p in pdbs]
    for p in paths:
        if not os.path.exists(p):
            print("MISSING", p, file=sys.stderr); sys.exit(2)

    outdir = os.path.join(OUT, "pairwise")
    os.makedirs(outdir, exist_ok=True)

    # self lengths
    print(f"[{ds}] computing self lengths for {len(pdbs)} complexes", file=sys.stderr)
    with Pool(args.workers) as pool:
        lens = list(pool.imap_unordered(self_len, list(zip(pdbs, paths)), chunksize=1))
    len_map = {}
    with open(os.path.join(outdir, f"{ds}_complex_lengths.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["PDB", "length", "usalign_name1", "self_seq_id", "error"])
        for pdb, L, name, sid, err in sorted(lens):
            len_map[pdb] = L
            w.writerow([pdb, L, name, sid, err or ""])
    print(f"[{ds}] self lengths done", file=sys.stderr)

    pairs = []
    for i, j in itertools.combinations(range(len(pdbs)), 2):
        pairs.append((i, j, paths[i], paths[j]))
    print(f"[{ds}] running {len(pairs)} pairwise alignments with {args.workers} workers", file=sys.stderr)

    out_path = os.path.join(outdir, f"{ds}_pairwise.csv")
    failures = 0
    done = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pdb_i", "pdb_j", "len_i", "len_j", "aligned_len", "rmsd",
                    "tm_norm_by_i", "tm_norm_by_j", "cov_by_i", "cov_by_j", "seq_id", "error"])
        with Pool(args.workers) as pool:
            for res in pool.imap_unordered(pair_job, pairs, chunksize=4):
                (i, j, L1, L2, aln, rmsd, tm1, tm2, err, cov1, cov2, sid) = (list(res) + [None] * (12 - len(res)))[:12]
                done += 1
                if err is not None and tm1 is None:
                    failures += 1
                    w.writerow([pdbs[i], pdbs[j], L1, L2, aln, rmsd, "", "", "", "", "", err[:200]])
                else:
                    w.writerow([pdbs[i], pdbs[j], L1, L2, aln, rmsd, tm1, tm2, cov1, cov2, sid, ""])
                if done % 500 == 0:
                    f.flush()
                    print(f"[{ds}] {done}/{len(pairs)} pairs, failures={failures}", file=sys.stderr)
    print(f"[{ds}] DONE {done} pairs, failures={failures} -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
