#!/usr/bin/env python3
"""Fold-level structural exposure from the all-vs-all TM-score matrices.

Definitions (documented in the report):
- For an ordered (eval complex e, train complex t) pair we use the *global*
  complex alignment from USalign -mm 1 -ter 1.
- tm_eval  = TM-score normalized by the length of the eval complex e
             (fraction of e's structure that can be matched).
- tm_train = TM-score normalized by the length of the train complex t.
- coverage of e = aligned_length / length(e).
- "best match" of e = training complex maximising tm_eval.
- "bilateral" similarity of a pair = min(tm_eval, tm_train); best bilateral
  = max_t min(tm_eval, tm_train).
- Exposure is reported both including and excluding the trivial case where the
  identical complex (same PDB) is also present in the training fold.
"""

import os
import csv
import json
import collections
import statistics as st

ROOT = "/data"
DS_ROOT = os.path.join(ROOT, "data_split")
OUT = os.path.join(ROOT, "structure_audit_20261105")
DATASETS = ["S1131", "M1101"]
SEEDS = ["0", "42", "1024", "1234", "2026"]
NORMAL_SPLITS = ["RCSV", "PCSV", "ICSV"]
THRESH = [0.5, 0.7]


def load_pair(ds):
    """dict: frozenset? use ordered tuple key (a,b) -> dict with tm_norm_by_a, tm_norm_by_b."""
    d = {}
    path = os.path.join(OUT, "pairwise", f"{ds}_pairwise.csv")
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["tm_norm_by_i"]:
                continue
            d[(r["pdb_i"], r["pdb_j"])] = {
                "tm_norm_by_i": float(r["tm_norm_by_i"]),
                "tm_norm_by_j": float(r["tm_norm_by_j"]),
                "cov_by_i": float(r["cov_by_i"]) if r["cov_by_i"] else None,
                "cov_by_j": float(r["cov_by_j"]) if r["cov_by_j"] else None,
                "aligned_len": int(r["aligned_len"]) if r["aligned_len"] else None,
            }
    return d


def lookup(pair, a, b):
    """Return (tm_norm_by_a, tm_norm_by_b, cov_a, cov_b) for ordered (a,b)."""
    if a == b:
        return 1.0, 1.0, 1.0, 1.0
    if (a, b) in pair:
        p = pair[(a, b)]
        return p["tm_norm_by_i"], p["tm_norm_by_j"], p["cov_by_i"], p["cov_by_j"]
    p = pair[(b, a)]
    # stored as (b,a): tm_norm_by_i is norm by b, tm_norm_by_j is norm by a
    return p["tm_norm_by_j"], p["tm_norm_by_i"], p["cov_by_j"], p["cov_by_i"]


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def folds_for(ds, split, seed):
    if split == "AGCV":
        d = os.path.join(DS_ROOT, "splits", ds, seed, "AGCV")
        out = []
        for k in range(5):
            tr = read_rows(os.path.join(d, f"fold_{k}_outer_train.csv"))
            ev = read_rows(os.path.join(d, f"fold_{k}_outer_eval.csv"))
            out.append((k, tr, ev))
        return out
    d = os.path.join(DS_ROOT, "splits", ds, seed, split)
    out = []
    for k in range(5):
        tr = read_rows(os.path.join(d, f"train_fold_{k}.csv"))
        ev = read_rows(os.path.join(d, f"val_fold_{k}.csv"))
        out.append((k, tr, ev))
    return out


def pdb_counts(rows):
    c = collections.Counter(r["PDB"] for r in rows)
    return c


def analyse_fold(pair, train_rows, eval_rows):
    train_c = pdb_counts(train_rows)
    eval_c = pdb_counts(eval_rows)
    train_pdbs = set(train_c)
    eval_pdbs = sorted(eval_c)
    n_eval_records = sum(eval_c.values())
    per_eval = []
    for e in eval_pdbs:
        shared = e in train_pdbs
        # candidates excluding exact same complex for the "novel" variant
        cand_all = train_pdbs
        cand_novel = train_pdbs - {e}
        def best_of(cands):
            bt_e = -1.0; bt_t = None; bcov = None; bt_id = None
            bb = -1.0; bb_id = None
            for t in cands:
                tm_e, tm_t, cov_e, _ = lookup(pair, e, t)
                if tm_e > bt_e:
                    bt_e = tm_e; bt_t = tm_t; bcov = cov_e; bt_id = t
                bval = min(tm_e, tm_t)
                if bval > bb:
                    bb = bval; bb_id = t
            return bt_e, bt_t, bcov, bt_id, bb, bb_id
        b_all = best_of(cand_all)
        b_nov = best_of(cand_novel) if cand_novel else (None, None, None, None, None, None)
        per_eval.append({
            "PDB": e, "n_records": eval_c[e], "shared_with_train": shared,
            "best_eval_tm": b_all[0], "best_train_pdb": b_all[3],
            "best_train_tm_by_train": b_all[1], "best_cov_eval": b_all[2],
            "best_bilateral": b_all[4],
            "best_eval_tm_novel": b_nov[0], "best_train_pdb_novel": b_nov[3],
            "best_bilateral_novel": b_nov[4],
        })
    return {
        "n_eval_records": n_eval_records,
        "n_train_records": sum(train_c.values()),
        "n_eval_complexes": len(eval_pdbs),
        "n_train_complexes": len(train_pdbs),
        "n_shared_complexes": sum(1 for e in eval_pdbs if e in train_pdbs),
        "per_eval": per_eval,
        "eval_c": eval_c,
    }


def summarize_fold(res):
    pe = res["per_eval"]
    n = len(pe)
    ei = [p["best_eval_tm"] for p in pe]
    bi = [p["best_bilateral"] for p in pe]
    # novel (exclude identical complex); if no novel candidate -> None
    en = [p["best_eval_tm_novel"] for p in pe if p["best_eval_tm_novel"] is not None]
    bn = [p["best_bilateral_novel"] for p in pe if p["best_bilateral_novel"] is not None]
    # record-weighted fractions
    tot = sum(p["n_records"] for p in pe)
    def rec_frac(items):
        if tot == 0:
            return None
        s = 0
        for it in items:
            for th in []:
                pass
        return None
    out = {
        "n_eval_complexes": n,
        "n_shared_complexes": res["n_shared_complexes"],
        "frac_eval_complexes_shared": res["n_shared_complexes"] / n if n else None,
        "mean_best_eval_tm": sum(ei) / n if n else None,
        "median_best_eval_tm": st.median(ei) if n else None,
        "min_best_eval_tm": min(ei) if n else None,
        "max_best_eval_tm": max(ei) if n else None,
        "mean_best_bilateral": sum(bi) / n if n else None,
        "median_best_bilateral": st.median(bi) if n else None,
        "mean_best_eval_tm_novel": (sum(en) / len(en)) if en else None,
        "median_best_eval_tm_novel": st.median(en) if en else None,
        "mean_best_bilateral_novel": (sum(bn) / len(bn)) if bn else None,
    }
    for th in THRESH:
        out[f"frac_eval_complexes_best_eval_tm_ge{th}"] = sum(1 for x in ei if x >= th) / n if n else None
        out[f"frac_eval_complexes_bilateral_ge{th}"] = sum(1 for x in bi if x >= th) / n if n else None
        out[f"frac_eval_complexes_best_eval_tm_novel_ge{th}"] = (
            sum(1 for x in en if x >= th) / len(en)) if en else None
        out[f"frac_eval_complexes_bilateral_novel_ge{th}"] = (
            sum(1 for x in bn if x >= th) / len(bn)) if bn else None
        # record-weighted
        if tot:
            num = 0
            for p in pe:
                if p["best_eval_tm"] >= th:
                    num += p["n_records"]
            out[f"record_frac_eval_best_eval_tm_ge{th}"] = num / tot
            num2 = 0; den2 = 0
            for p in pe:
                if p["best_eval_tm_novel"] is not None:
                    den2 += p["n_records"]
                    if p["best_eval_tm_novel"] >= th:
                        num2 += p["n_records"]
            out[f"record_frac_eval_best_eval_tm_novel_ge{th}"] = (num2 / den2) if den2 else None
    return out


def build_matrix(ds, pdbs, pair):
    n = len(pdbs)
    idx = {p: i for i, p in enumerate(pdbs)}
    # row-normalized: entry (r,c) = TM normalized by length of r (coverage of r)
    tm_row = [[0.0] * n for _ in range(n)]
    cov_row = [[0.0] * n for _ in range(n)]
    for r in range(n):
        tm_row[r][r] = 1.0; cov_row[r][r] = 1.0
    for (a, b), p in pair.items():
        ia, ib = idx[a], idx[b]
        tm_row[ia][ib] = p["tm_norm_by_i"]      # norm by a (row a coverage)
        tm_row[ib][ia] = p["tm_norm_by_j"]      # norm by b (row b coverage)
        cov_row[ia][ib] = p["cov_by_i"] or 0.0
        cov_row[ib][ia] = p["cov_by_j"] or 0.0
    return tm_row, cov_row


def main():
    os.makedirs(os.path.join(OUT, "exposure"), exist_ok=True)
    all_fold_rows = []
    all_eval_rows = []
    summary = {}
    for ds in DATASETS:
        pair = load_pair(ds)
        splits = NORMAL_SPLITS + (["AGCV"] if ds == "M1101" else [])
        summary[ds] = {}

        # unique complex order (from lengths csv) for matrices
        with open(os.path.join(OUT, "pairwise", f"{ds}_complex_lengths.csv"), newline="", encoding="utf-8") as f:
            pdbs = [r["PDB"] for r in csv.DictReader(f)]
        tm_row, cov_row = build_matrix(ds, pdbs, pair)
        with open(os.path.join(OUT, "exposure", f"{ds}_matrix_tm_normalized_by_row.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["PDB"] + pdbs)
            for i, p in enumerate(pdbs):
                w.writerow([p] + [f"{x:.5f}" for x in tm_row[i]])
        with open(os.path.join(OUT, "exposure", f"{ds}_matrix_coverage_normalized_by_row.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["PDB"] + pdbs)
            for i, p in enumerate(pdbs):
                w.writerow([p] + [f"{x:.4f}" for x in cov_row[i]])

        for split in splits:
            summary[ds][split] = {}
            for seed in SEEDS:
                folds = folds_for(ds, split, seed)
                for k, tr, ev in folds:
                    res = analyse_fold(pair, tr, ev)
                    s = summarize_fold(res)
                    row = {"dataset": ds, "split_type": split, "split_seed": seed, "outer_fold": k,
                           "n_eval_records": res["n_eval_records"],
                           "n_train_records": res["n_train_records"],
                           "n_eval_complexes": res["n_eval_complexes"],
                           "n_train_complexes": res["n_train_complexes"]}
                    row.update({k2: v for k2, v in s.items() if k2 not in ("n_eval_complexes",)})
                    all_fold_rows.append(row)
                    for pe in res["per_eval"]:
                        r2 = {"dataset": ds, "split_type": split, "split_seed": seed, "outer_fold": k}
                        r2.update(pe)
                        all_eval_rows.append(r2)

    # per-fold CSV
    fold_fields = ["dataset", "split_type", "split_seed", "outer_fold",
                   "n_eval_records", "n_train_records", "n_eval_complexes", "n_train_complexes",
                   "n_shared_complexes", "frac_eval_complexes_shared",
                   "mean_best_eval_tm", "median_best_eval_tm", "min_best_eval_tm", "max_best_eval_tm",
                   "mean_best_bilateral", "median_best_bilateral",
                   "mean_best_eval_tm_novel", "median_best_eval_tm_novel", "mean_best_bilateral_novel"]
    for th in THRESH:
        fold_fields += [f"frac_eval_complexes_best_eval_tm_ge{th}",
                        f"frac_eval_complexes_bilateral_ge{th}",
                        f"frac_eval_complexes_best_eval_tm_novel_ge{th}",
                        f"frac_eval_complexes_bilateral_novel_ge{th}",
                        f"record_frac_eval_best_eval_tm_ge{th}",
                        f"record_frac_eval_best_eval_tm_novel_ge{th}"]
    with open(os.path.join(OUT, "exposure", "per_fold_exposure.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fold_fields, extrasaction="ignore")
        w.writeheader()
        for r in all_fold_rows:
            w.writerow(r)

    # per eval complex best-match CSV
    eval_fields = ["dataset", "split_type", "split_seed", "outer_fold", "PDB", "n_records",
                   "shared_with_train", "best_eval_tm", "best_train_pdb", "best_train_tm_by_train",
                   "best_cov_eval", "best_bilateral", "best_eval_tm_novel", "best_train_pdb_novel",
                   "best_bilateral_novel"]
    with open(os.path.join(OUT, "exposure", "per_eval_complex_best_match.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=eval_fields, extrasaction="ignore")
        w.writeheader()
        for r in all_eval_rows:
            w.writerow(r)

    # aggregated summary table (Supplementary Table S1 candidate)
    s1_fields = ["dataset", "split_type", "n_cells",
                 "frac_eval_complexes_shared_mean", "frac_eval_complexes_shared_sd",
                 "mean_best_eval_tm_mean", "mean_best_eval_tm_sd",
                 "mean_best_eval_tm_novel_mean", "mean_best_eval_tm_novel_sd",
                 "mean_best_bilateral_mean", "mean_best_bilateral_sd"]
    for th in THRESH:
        s1_fields += [f"frac_eval_complexes_best_eval_tm_ge{th}_mean",
                      f"frac_eval_complexes_best_eval_tm_ge{th}_sd",
                      f"frac_eval_complexes_best_eval_tm_novel_ge{th}_mean",
                      f"frac_eval_complexes_best_eval_tm_novel_ge{th}_sd",
                      f"record_frac_eval_best_eval_tm_novel_ge{th}_mean"]

    def mean_sd(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None
        m = sum(vals) / len(vals)
        sd = st.pstdev(vals) if len(vals) > 1 else 0.0
        return m, sd

    s1_rows = []
    for ds in DATASETS:
        splits = NORMAL_SPLITS + (["AGCV"] if ds == "M1101" else [])
        for split in splits:
            rs = [r for r in all_fold_rows if r["dataset"] == ds and r["split_type"] == split]
            if not rs:
                continue
            row = {"dataset": ds, "split_type": split, "n_cells": len(rs)}
            m, sd = mean_sd([r["frac_eval_complexes_shared"] for r in rs])
            row["frac_eval_complexes_shared_mean"], row["frac_eval_complexes_shared_sd"] = m, sd
            m, sd = mean_sd([r["mean_best_eval_tm"] for r in rs])
            row["mean_best_eval_tm_mean"], row["mean_best_eval_tm_sd"] = m, sd
            m, sd = mean_sd([r["mean_best_eval_tm_novel"] for r in rs])
            row["mean_best_eval_tm_novel_mean"], row["mean_best_eval_tm_novel_sd"] = m, sd
            m, sd = mean_sd([r["mean_best_bilateral"] for r in rs])
            row["mean_best_bilateral_mean"], row["mean_best_bilateral_sd"] = m, sd
            for th in THRESH:
                m, sd = mean_sd([r[f"frac_eval_complexes_best_eval_tm_ge{th}"] for r in rs])
                row[f"frac_eval_complexes_best_eval_tm_ge{th}_mean"] = m
                row[f"frac_eval_complexes_best_eval_tm_ge{th}_sd"] = sd
                m, sd = mean_sd([r[f"frac_eval_complexes_best_eval_tm_novel_ge{th}"] for r in rs])
                row[f"frac_eval_complexes_best_eval_tm_novel_ge{th}_mean"] = m
                row[f"frac_eval_complexes_best_eval_tm_novel_ge{th}_sd"] = sd
                m, _ = mean_sd([r[f"record_frac_eval_best_eval_tm_novel_ge{th}"] for r in rs])
                row[f"record_frac_eval_best_eval_tm_novel_ge{th}_mean"] = m
            s1_rows.append(row)

    with open(os.path.join(OUT, "supplementary_table_S1_structure_exposure.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=s1_fields, extrasaction="ignore")
        w.writeheader()
        for r in s1_rows:
            w.writerow(r)

    print("Wrote exposure tables.")
    print("Per-fold cells:", len(all_fold_rows))
    print()
    print(f"{'dataset':8s} {'split':6s} {'cells':5s} {'shared':>7s} {'tm>=0.7':>8s} {'novel>=0.7':>10s} {'meanBestTM':>10s} {'meanBilat':>9s}")
    for r in s1_rows:
        def f(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "
        print(f"{r['dataset']:8s} {r['split_type']:6s} {r['n_cells']:5d} "
              f"{f(r['frac_eval_complexes_shared_mean']):>7s} "
              f"{f(r['frac_eval_complexes_best_eval_tm_ge0.7_mean']):>8s} "
              f"{f(r['frac_eval_complexes_best_eval_tm_novel_ge0.7_mean']):>10s} "
              f"{f(r['mean_best_eval_tm_mean']):>10s} "
              f"{f(r['mean_best_bilateral_mean']):>9s}")


if __name__ == "__main__":
    main()
