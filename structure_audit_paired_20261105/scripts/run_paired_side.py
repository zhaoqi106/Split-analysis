#!/usr/bin/env python3
"""Per-side all-vs-all US-align for one dataset and one side (left/right).

For every unordered complex pair run:
    USalign {i}.{side}.pdb {j}.{side}.pdb -mm 1 -ter 1
Record both directional TM-scores (normalized by partner length of i and j),
both coverages = aligned/len, and the manuscript side metrics:
    S_p = min(TM_{p,v}, TM_{p,t})
    C_p = min(C_{p,v}, C_{p,t})
Self pairs are analytic (TM=1, coverage=1) and additionally verified by an
actual self-alignment (QC).

Outputs:
  paired/{ds}_{side}_self.csv
  paired/pairwise_{ds}_{side}.csv
"""
import os, sys, csv, re, subprocess, argparse, itertools
from multiprocessing import Pool

ROOT = "/data"
OUT = os.path.join(ROOT, "structure_audit_paired_20261105")
USALIGN = os.path.expanduser("~/.local/bin/USalign")

RE_L1 = re.compile(r"Length of Structure_1:\s*([0-9]+) residues")
RE_L2 = re.compile(r"Length of Structure_2:\s*([0-9]+) residues")
RE_ALN = re.compile(r"Aligned length=\s*([0-9]+), RMSD=\s*([0-9.]+), Seq_ID=n_identical/n_aligned=\s*([0-9.]+)")
RE_TM1 = re.compile(r"TM-score=\s*([0-9.]+) \(normalized by length of Structure_1")
RE_TM2 = re.compile(r"TM-score=\s*([0-9.]+) \(normalized by length of Structure_2")
RE_NAME1 = re.compile(r"Name of Structure_1:\s*(.+?)\s*\(to be")
RE_NAME2 = re.compile(r"Name of Structure_2:\s*(.+)")

_LOG = None


def _init_logger(logdir):
    global _LOG
    os.makedirs(logdir, exist_ok=True)
    _LOG = open(os.path.join(logdir, f"worker_{os.getpid()}.log"), "a", buffering=1, encoding="utf-8")
    _LOG.write(f"# worker pid={os.getpid()}\n")


def _log_cmd(cmd, out):
    if _LOG is not None:
        _LOG.write("### CMD: " + " ".join(cmd) + "\n")
        _LOG.write(out.rstrip() + "\n\n")


def run_one(p1, p2):
    cmd = [USALIGN, p1, p2, "-mm", "1", "-ter", "1"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        out = p.stdout + "\n" + p.stderr
    except Exception as e:
        _log_cmd(cmd, f"EXCEPTION: {e}")
        return None, str(e)
    _log_cmd(cmd, out)
    rec = {}
    m = RE_L1.search(out); rec["len1"] = int(m.group(1)) if m else None
    m = RE_L2.search(out); rec["len2"] = int(m.group(1)) if m else None
    m = RE_ALN.search(out)
    rec["aligned_len"] = int(m.group(1)) if m else None
    rec["rmsd"] = float(m.group(2)) if m else None
    rec["seq_id"] = float(m.group(3)) if m else None
    m = RE_TM1.search(out); rec["tm1"] = float(m.group(1)) if m else None
    m = RE_TM2.search(out); rec["tm2"] = float(m.group(1)) if m else None
    m = RE_NAME1.search(out); rec["name1"] = m.group(1) if m else ""
    m = RE_NAME2.search(out); rec["name2"] = m.group(1).strip() if m else ""
    if rec["tm1"] is None or rec["tm2"] is None or rec["aligned_len"] is None:
        return None, out[-600:]
    return rec, None


def self_job(args):
    pdb, path = args
    rec, err = run_one(path, path)
    if rec is None:
        return (pdb, None, None, None, None, None, err)
    c1 = rec["aligned_len"] / rec["len1"]; c2 = rec["aligned_len"] / rec["len2"]
    chains = rec["name1"].split(":")[1:] if ":" in rec["name1"] else []
    return (pdb, rec["len1"], rec["tm1"], rec["tm2"], c1, c2, rec["name1"], len(chains), rec["seq_id"], None)


def pair_job(args):
    i, j, p1, p2 = args
    rec, err = run_one(p1, p2)
    if rec is None:
        return (i, j, None, None, None, None, None, None, None, None, None, None, err)
    c1 = rec["aligned_len"] / rec["len1"]; c2 = rec["aligned_len"] / rec["len2"]
    S = min(rec["tm1"], rec["tm2"]); C = min(c1, c2)
    return (i, j, rec["len1"], rec["len2"], rec["aligned_len"], rec["rmsd"],
            rec["tm1"], rec["tm2"], c1, c2, rec["seq_id"], S, C, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--side", required=True, choices=["left", "right"])
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    ds, side = args.dataset, args.side

    # canonical order of complexes from partner manifest
    pdbs = []
    manifest = {}
    with open(os.path.join(OUT, "partner_chain_manifest.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == ds:
                pdbs.append(r["PDB"]); manifest[r["PDB"]] = r
    paths = [os.path.join(OUT, "partners", ds, f"{p}.{side}.pdb") for p in pdbs]
    for p in paths:
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            print("MISSING/empty partner file", p, file=sys.stderr); sys.exit(2)
    logdir = os.path.join(OUT, "logs", "usalign_raw", f"{ds}_{side}")

    # self alignments (QC)
    print(f"[{ds}/{side}] self-align {len(pdbs)} partners", file=sys.stderr)
    with Pool(args.workers, initializer=_init_logger, initargs=(logdir,)) as pool:
        selfs = list(pool.imap_unordered(self_job, list(zip(pdbs, paths)), chunksize=1))
    with open(os.path.join(OUT, "paired", f"{ds}_{side}_self.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PDB", "partner_length", "self_tm1", "self_tm2", "self_cov1", "self_cov2",
                    "usalign_name1", "n_chains_in_name", "self_seq_id", "error"])
        for row in sorted(selfs):
            w.writerow(row)

    pairs = [(i, j, paths[i], paths[j]) for i, j in itertools.combinations(range(len(pdbs)), 2)]
    print(f"[{ds}/{side}] {len(pairs)} pair alignments, workers={args.workers}", file=sys.stderr)
    out_path = os.path.join(OUT, "paired", f"pairwise_{ds}_{side}.csv")
    done = fail = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pdb_i", "pdb_j", "len_i", "len_j", "aligned_len", "rmsd",
                    "tm_norm_by_i", "tm_norm_by_j", "cov_by_i", "cov_by_j", "seq_id", "S_p", "C_p", "error"])
        with Pool(args.workers, initializer=_init_logger, initargs=(logdir,)) as pool:
            for r in pool.imap_unordered(pair_job, pairs, chunksize=4):
                done += 1
                if r[-1] is not None and r[6] is None:
                    fail += 1
                    w.writerow([pdbs[r[0]], pdbs[r[1]], "", "", "", "", "", "", "", "", "", "", "", str(r[-1])[:300]])
                else:
                    (i, j, L1, L2, aln, rmsd, tm1, tm2, c1, c2, sid, S, C, err) = r
                    w.writerow([pdbs[i], pdbs[j], L1, L2, aln, rmsd, tm1, tm2, c1, c2, sid, S, C, ""])
                if done % 1000 == 0:
                    f.flush(); print(f"[{ds}/{side}] {done}/{len(pairs)} fail={fail}", file=sys.stderr)
    print(f"[{ds}/{side}] DONE {done} pairs fail={fail} -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
