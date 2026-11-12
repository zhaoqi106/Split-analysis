#!/usr/bin/env python3
"""Auxiliary formal-audit tables: HM comparison records, provenance, failures."""

import os
import csv
import json
import hashlib
import subprocess

ROOT = "/data"
OUT = os.path.join(ROOT, "structure_audit_20261105")
USALIGN = os.path.expanduser("~/.local/bin/USalign")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(os.path.join(OUT, "exposure"), exist_ok=True)

    # ---- pairwise error / failure file ----
    fail_rows = []
    for ds in ["S1131", "M1101"]:
        p = os.path.join(OUT, "pairwise", f"{ds}_pairwise.csv")
        for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
            if r["error"].strip():
                fail_rows.append([ds, r["pdb_i"], r["pdb_j"], r["error"]])
    with open(os.path.join(OUT, "exposure", "failed_or_undetermined.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "pdb_i", "pdb_j", "error"])
        w.writerows(fail_rows)

    # ---- M1101 HM comparison vs all base complexes, and template check ----
    pair = {}
    for r in csv.DictReader(open(os.path.join(OUT, "pairwise", "M1101_pairwise.csv"), newline="", encoding="utf-8")):
        if not r["tm_norm_by_i"]:
            continue
        pair[(r["pdb_i"], r["pdb_j"])] = r

    def get(a, b):
        if (a, b) in pair:
            r = pair[(a, b)]
            return float(r["tm_norm_by_i"]), float(r["tm_norm_by_j"]), r["cov_by_i"], r["cov_by_j"]
        if (b, a) in pair:
            r = pair[(b, a)]
            return float(r["tm_norm_by_j"]), float(r["tm_norm_by_i"]), r["cov_by_j"], r["cov_by_i"]
        if a == b:
            return 1.0, 1.0, 1.0, 1.0
        return None, None, None, None

    lengths = {r["PDB"]: r["length"] for r in csv.DictReader(open(os.path.join(OUT, "pairwise", "M1101_complex_lengths.csv"), newline="", encoding="utf-8"))}
    hm_pdbs = sorted(p for p in lengths if p.startswith("HM_"))
    base_pdbs = sorted(p for p in lengths if not p.startswith("HM_"))

    # full HM vs base table
    with open(os.path.join(OUT, "exposure", "hm_vs_base_all.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hm_pdb", "len_hm", "base_pdb", "len_base", "tm_norm_by_hm", "tm_norm_by_base",
                    "cov_hm", "cov_base", "is_template"])
        for hm in hm_pdbs:
            template = hm[len("HM_"):]
            for b in base_pdbs:
                tm_hm, tm_b, cov_hm, cov_b = get(hm, b)
                w.writerow([hm, lengths[hm], b, lengths[b], tm_hm, tm_b, cov_hm, cov_b,
                            "yes" if b == template else "no"])

    # template summary + top non-self base match
    with open(os.path.join(OUT, "exposure", "hm_template_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hm_pdb", "template_base", "template_present", "tm_norm_by_hm_vs_template",
                    "tm_norm_by_base_vs_template", "cov_hm", "cov_base",
                    "top_non_template_base", "top_non_template_tm_norm_by_hm"])
        for hm in hm_pdbs:
            template = hm[len("HM_"):]
            present = template in lengths
            if present:
                t_hm, t_b, c_hm, c_b = get(hm, template)
            else:
                t_hm = t_b = c_hm = c_b = None
            best = (-1, None)
            for b in base_pdbs:
                if b == template:
                    continue
                tm_hm, _, _, _ = get(hm, b)
                if tm_hm is not None and tm_hm > best[0]:
                    best = (tm_hm, b)
            w.writerow([hm, template, "yes" if present else "no", t_hm, t_b, c_hm, c_b,
                        best[1], best[0] if best[0] >= 0 else None])

    # ---- software provenance ----
    prov = {
        "tool": "US-align",
        "version": "20260328",
        "binary_path": USALIGN,
        "binary_copy": os.path.join(OUT, "usalign_bin", "USalign"),
        "binary_sha256": sha256(USALIGN),
        "download_url": "https://zhanggroup.org/US-align/bin/module/USalignLinux64.zip",
        "architecture": "ELF 64-bit LSB executable, x86-64, statically linked",
        "alignment_mode": "-mm 1 -ter 1 (global alignment of multi-chain complexes; -ter 1 aligns all chains of the first model)",
        "command_template": "USalign {wt_i}.pdb {wt_j}.pdb -mm 1 -ter 1",
        "normalization": "TM-score recorded in both directions: normalized by length of structure_1 and structure_2",
        "coverage": "aligned_length / length(structure_1) and aligned_length / length(structure_2)",
    }
    with open(os.path.join(OUT, "usalign_bin", "software_provenance.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, ensure_ascii=False, indent=2)

    print("fail rows:", len(fail_rows))
    print(json.dumps(prov, ensure_ascii=False, indent=2))
    print()
    print("=== HM template summary ===")
    for r in csv.DictReader(open(os.path.join(OUT, "exposure", "hm_template_summary.csv"), newline="", encoding="utf-8")):
        print(r)


if __name__ == "__main__":
    main()
