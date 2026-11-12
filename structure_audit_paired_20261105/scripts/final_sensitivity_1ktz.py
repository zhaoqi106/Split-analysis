#!/usr/bin/env python3
"""Targeted sensitivity for the 1KTZ / HM_1KTZ left/right role conflict.

Main mapping (metadata):  left = antibody_chains = B (type-II receptor),
                          right = antigen_chains = A (TGF-beta ligand).
Alternative (CSV columns): left = A, right = B  (i.e. swap).

Only pairs involving 1KTZ or HM_1KTZ are recomputed, via the already-extracted
partner PDBs (no full all-vs-all rerun). All other pairs reuse the existing
combined matrix. Then AGCV exposure is recomputed with the alternative mapping.
"""
import os, csv, re, subprocess, collections

ROOT = "/data"
DS = os.path.join(ROOT, "data_split")
OUT = os.path.join(ROOT, "structure_audit_paired_20261105")
USALIGN = os.path.expanduser("~/.local/bin/USalign")
CONFLICTS = {"1KTZ", "HM_1KTZ"}
TAU, COV = 0.70, 0.80
SEEDS = ["0", "42", "1024", "1234", "2026"]

RE_L1 = re.compile(r"Length of Structure_1:\s*([0-9]+) residues")
RE_L2 = re.compile(r"Length of Structure_2:\s*([0-9]+) residues")
RE_ALN = re.compile(r"Aligned length=\s*([0-9]+)")
RE_TM1 = re.compile(r"TM-score=\s*([0-9.]+) \(normalized by length of Structure_1")
RE_TM2 = re.compile(r"TM-score=\s*([0-9.]+) \(normalized by length of Structure_2")


def align(p1, p2):
    out = subprocess.run([USALIGN, p1, p2, "-mm", "1", "-ter", "1"], capture_output=True, text=True).stdout
    L1 = int(RE_L1.search(out).group(1)); L2 = int(RE_L2.search(out).group(1))
    aln = int(RE_ALN.search(out).group(1))
    t1 = float(RE_TM1.search(out).group(1)); t2 = float(RE_TM2.search(out).group(1))
    c1 = aln / L1; c2 = aln / L2
    return min(t1, t2), min(c1, c2)


def pl(p):
    return os.path.join(OUT, "partners", "M1101", p)


def main():
    pdbs = []
    for r in csv.DictReader(open(os.path.join(OUT, "partner_chain_manifest.csv"), encoding="utf-8")):
        if r["dataset"] == "M1101":
            pdbs.append(r["PDB"])
    normal = [p for p in pdbs if p not in CONFLICTS]

    # main combined
    main = {}
    for r in csv.DictReader(open(os.path.join(OUT, "paired", "pairwise_M1101_combined.csv"), encoding="utf-8")):
        if r["S_pair"]:
            main[frozenset((r["pdb_i"], r["pdb_j"]))] = (float(r["S_pair"]), float(r["C_pair"]))

    # alt S_pair for pairs involving a conflicted complex
    alt = {}
    log = []
    for c in sorted(CONFLICTS):
        c_altL = pl(f"{c}.right.pdb")   # A (ligand) becomes alt-left
        c_altR = pl(f"{c}.left.pdb")    # B (receptor) becomes alt-right
        for x in pdbs:
            if x == c:
                continue
            xL = pl(f"{x}.right.pdb") if x in CONFLICTS else pl(f"{x}.left.pdb")
            xR = pl(f"{x}.left.pdb") if x in CONFLICTS else pl(f"{x}.right.pdb")
            Sl, Cl = align(c_altL, xL)
            Sr, Cr = align(c_altR, xR)
            alt[frozenset((c, x))] = (min(Sl, Sr), min(Cl, Cr))
            log.append([c, x, round(Sl, 4), round(Sr, 4), round(min(Sl, Sr), 4), round(min(Cl, Cr), 4)])

    def lookup(a, b, use_alt):
        if a == b:
            return 1.0, 1.0
        key = frozenset((a, b))
        if use_alt and key in alt:
            return alt[key]
        return main.get(key, (None, None))

    def agcv_exposure(use_alt):
        # returns per seed record-weighted, and grand pooled
        per_seed = {}
        num_all = den_all = 0
        for seed in SEEDS:
            num = den = 0
            for k in range(5):
                d = os.path.join(DS, "splits", "M1101", seed, "AGCV")
                tr = list(csv.DictReader(open(os.path.join(d, f"fold_{k}_outer_train.csv"), encoding="utf-8-sig")))
                ev = list(csv.DictReader(open(os.path.join(d, f"fold_{k}_outer_eval.csv"), encoding="utf-8-sig")))
                train = set(r["PDB"] for r in tr)
                evc = collections.Counter(r["PDB"] for r in ev)
                for v, n in evc.items():
                    den += n
                    exposed = False
                    for t in train:
                        s, cc = lookup(v, t, use_alt)
                        if s is not None and s >= TAU and cc >= COV:
                            exposed = True; break
                    if exposed:
                        num += n
            per_seed[seed] = num / den if den else None
            num_all += num; den_all += den
        vals = list(per_seed.values())
        m = sum(vals) / len(vals)
        sd = (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
        return per_seed, m, sd, num_all / den_all, num_all, den_all

    main_ps, main_m, main_sd, main_gp, mn, md = agcv_exposure(False)
    alt_ps, alt_m, alt_sd, alt_gp, an, ad = agcv_exposure(True)

    with open(os.path.join(OUT, "sensitivity_1KTZ_alt_pairwise.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["conflict_complex", "other", "altS_left", "altS_right", "altS_pair", "altC_pair"])
        w.writerows(log)

    print("=== AGCV exposure: main (metadata roles) vs alt (CSV swap for 1KTZ/HM_1KTZ) ===")
    print(f"main seed-pooled recW mean = {main_m:.4f}  sd(n=5) = {main_sd:.4f}  grand pooled = {main_gp:.4f}  num/den = {mn}/{md}")
    print(f"alt  seed-pooled recW mean = {alt_m:.4f}  sd(n=5) = {alt_sd:.4f}  grand pooled = {alt_gp:.4f}  num/den = {an}/{ad}")
    print(f"delta (alt-main) = {alt_m - main_m:+.4f}")
    print("per-seed main:", {k: round(v, 4) for k, v in main_ps.items()})
    print("per-seed alt :", {k: round(v, 4) for k, v in alt_ps.items()})
    # which conflicted-complex pairs change decision?
    changed = []
    for key, (s_alt, c_alt) in alt.items():
        a, b = tuple(key)
        s_main, c_main = main.get(key, (None, None))
        if s_main is None: continue
        d_main = s_main >= TAU and c_main >= COV
        d_alt = s_alt >= TAU and c_alt >= COV
        if d_main != d_alt:
            changed.append((a, b, round(s_main, 3), round(s_alt, 3), "main_exposed" if d_main else "alt_exposed"))
    print("\npairs (involving 1KTZ/HM_1KTZ) whose exposure decision flips:", len(changed))
    for x in changed:
        print("   ", x)


if __name__ == "__main__":
    main()
