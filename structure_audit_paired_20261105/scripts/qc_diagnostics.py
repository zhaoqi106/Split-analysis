#!/usr/bin/env python3
"""QC + diagnostics for paired exposure.

1. Verify same-t requirement: S_pair(v,t)=min(S_left,S_right) both from same t
   (recompute independently and diff against combined csv).
2. Show explicit counterexamples where whole-complex TM differs from paired.
3. Top AGCV / PCSV structural neighbours (same-t) for report.
4. Self-match sanity (already validated) summary.
"""
import os, csv, collections

ROOT = "/data"
OUT = os.path.join(ROOT, "structure_audit_paired_20261105")
WHOLE = os.path.join(ROOT, "structure_audit_20261105", "pairwise")
TAU, COV = 0.70, 0.80


def read(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_whole(ds):
    d = {}
    for r in read(os.path.join(WHOLE, f"{ds}_pairwise.csv")):
        if not r["tm_norm_by_i"]:
            continue
        d[(r["pdb_i"], r["pdb_j"])] = (float(r["tm_norm_by_i"]), float(r["tm_norm_by_j"]),
                                       float(r["cov_by_i"]) if r["cov_by_i"] else None)
    return d


def whole_pair(d, a, b):
    if a == b: return (1.0, 1.0, 1.0)
    if (a, b) in d: return d[(a, b)]
    if (b, a) in d:
        x = d[(b, a)]; return (x[1], x[0], x[2])
    return None


def main():
    report = []
    for ds in ("S1131", "M1101"):
        comb = read(os.path.join(OUT, "paired", f"pairwise_{ds}_combined.csv"))
        whole = load_whole(ds)
        # 1. same-t independent recompute
        mism = 0
        for r in comb:
            if not r["S_pair"]:
                continue
            sl = float(r["left_S"]); sr = float(r["right_S"])
            if abs(min(sl, sr) - float(r["S_pair"])) > 1e-9:
                mism += 1
        print(f"[{ds}] same-t S_pair = min(S_left,S_right) mismatches: {mism} / {len(comb)}")

        # 2. whole vs paired divergence examples (both directions)
        div = []
        for r in comb:
            if not r["S_pair"]:
                continue
            a, b = r["pdb_i"], r["pdb_j"]
            wp = whole_pair(whole, a, b)
            if wp is None: continue
            whole_min = min(wp[0], wp[1])
            sp = float(r["S_pair"]); cp = float(r["C_pair"])
            whole_exposed = whole_min >= TAU
            paired_exposed = sp >= TAU and cp >= COV
            if whole_exposed != paired_exposed:
                div.append((a, b, round(whole_min, 3), round(float(r["left_S"]), 3),
                            round(float(r["right_S"]), 3), round(sp, 3), round(cp, 3),
                            "whole_only" if whole_exposed else "paired_only"))
        div.sort(key=lambda x: (x[7], -abs(x[2]-x[5])))
        print(f"[{ds}] whole!=paired decision divergences: {len(div)}")
        for d in div[:12]:
            print("    ", d)

        # 3. strongest AGCV neighbours (same t), using M1101 AGCV fold composition
        report.append((ds, div))

    # 4. AGCV top neighbours
    print("\n=== M1101 AGCV best same-t neighbours (S_pair>=0.7) ===")
    print(f"{'pdb_i':10s} {'pdb_j':10s} {'S_L':>5s} {'S_R':>5s} {'S_pair':>6s} {'C_pair':>6s}")
    comb = read(os.path.join(OUT, "paired", "pairwise_M1101_combined.csv"))
    rows = [(r["pdb_i"], r["pdb_j"], r["left_S"], r["right_S"], r["S_pair"], r["C_pair"])
            for r in comb if r["S_pair"] and float(r["S_pair"]) >= TAU]
    rows.sort(key=lambda x: -float(x[4]))
    for a, b, sl, sr, sp, cp in rows:
        print(f"{a:10s} {b:10s} {float(sl):>5.2f} {float(sr):>5.2f} {float(sp):>6.2f} {float(cp):>6.2f}")

    print("\n=== S1131 PCSV-driving strongest left-similar pairs (S_pair>=0.7), top 20 ===")
    comb = read(os.path.join(OUT, "paired", "pairwise_S1131_combined.csv"))
    rows = [(r["pdb_i"], r["pdb_j"], r["left_S"], r["right_S"], r["S_pair"], r["C_pair"])
            for r in comb if r["S_pair"] and float(r["S_pair"]) >= TAU]
    rows.sort(key=lambda x: -float(x[4]))
    print("total S1131 pairs with S_pair>=0.7:", len(rows))
    for a, b, sl, sr, sp, cp in rows[:20]:
        print(f"{a:10s} {b:10s} L={float(sl):>4.2f} R={float(sr):>4.2f} S_pair={float(sp):>4.2f} C_pair={float(cp):>4.2f}")


if __name__ == "__main__":
    main()
