#!/usr/bin/env python3
"""Combine per-side results and compute paired global-partner structural exposure.

Implements manuscript eq 4-8 exactly:
  per side p in {left,right}, per (v,t):
      S_p = min(TM_{p,v}, TM_{p,t})           (partner-length normalized)
      C_p = min(C_{p,v},  C_{p,t})            (bidirectional coverage)
  per (v,t):
      S_pair = min(S_left, S_right)
      C_pair = min(C_left, C_right)
  E_struct(v) = 1 iff exists a SINGLE training complex t with
      S_pair(v,t) >= 0.70 AND C_pair(v,t) >= 0.80
The same t must satisfy both sides (never mix left/right from different t).

Primary statistic: record-weighted exposure, INCLUDING identical-complex
matches (RCSV expected exposure). Excluding-identical is reported alongside.
UNRESOLVED is never counted as NOT_EXPOSED.
"""
import os, csv, collections, statistics as st, math, json

ROOT = "/data"
DS_ROOT = os.path.join(ROOT, "data_split")
OUT = os.path.join(ROOT, "structure_audit_paired_20261105")
DATASETS = ["S1131", "M1101"]
SEEDS = ["0", "42", "1024", "1234", "2026"]
SPLITS = {"S1131": ["RCSV", "PCSV", "ICSV"], "M1101": ["RCSV", "PCSV", "ICSV", "AGCV"]}
TAU = 0.70
COV = 0.80


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_manifest(ds):
    pdbs, man = [], {}
    for r in read_csv(os.path.join(OUT, "partner_chain_manifest.csv")):
        if r["dataset"] == ds:
            pdbs.append(r["PDB"]); man[r["PDB"]] = r
    return pdbs, man


def load_side(ds, side):
    d = {}
    for r in read_csv(os.path.join(OUT, "paired", f"pairwise_{ds}_{side}.csv")):
        if not r["S_p"]:
            continue
        d[(r["pdb_i"], r["pdb_j"])] = {
            "len_i": int(r["len_i"]), "len_j": int(r["len_j"]),
            "aligned": int(r["aligned_len"]), "rmsd": float(r["rmsd"]),
            "tm_i": float(r["tm_norm_by_i"]), "tm_j": float(r["tm_norm_by_j"]),
            "cov_i": float(r["cov_by_i"]), "cov_j": float(r["cov_by_j"]),
            "seq_id": float(r["seq_id"]), "S": float(r["S_p"]), "C": float(r["C_p"]),
        }
    return d


def lookup_side(d, a, b):
    if a == b:
        return {"len_i": None, "len_j": None, "aligned": None, "rmsd": 0.0,
                "tm_i": 1.0, "tm_j": 1.0, "cov_i": 1.0, "cov_j": 1.0, "seq_id": 1.0, "S": 1.0, "C": 1.0}
    if (a, b) in d:
        return d[(a, b)]
    if (b, a) in d:
        return d[(b, a)]
    return None


def build_combined(ds):
    pdbs, man = load_manifest(ds)
    L = load_side(ds, "left"); R = load_side(ds, "right")
    combined = {}
    failures = []
    rows = []
    for i in range(len(pdbs)):
        for j in range(i + 1, len(pdbs)):
            a, b = pdbs[i], pdbs[j]
            l = lookup_side(L, a, b); r = lookup_side(R, a, b)
            err = ""
            if l is None or r is None:
                err = "missing_side:" + (("left" if l is None else "") + ("right" if r is None else ""))
                failures.append([ds, a, b, "pair_alignment_missing", err])
                S_pair = C_pair = None
            else:
                S_pair = min(l["S"], r["S"]); C_pair = min(l["C"], r["C"])
            combined[(a, b)] = {"left": l, "right": r, "S_pair": S_pair, "C_pair": C_pair,
                                "S_side_best": None if (l is None or r is None) else max(l["S"], r["S"])}
            def g(x, k):
                return "" if x is None or x.get(k) is None else x[k]
            rows.append([a, b,
                         g(l, "len_i"), g(l, "len_j"), g(l, "aligned"), g(l, "tm_i"), g(l, "tm_j"), g(l, "cov_i"), g(l, "cov_j"), g(l, "S"), g(l, "C"),
                         g(r, "len_i"), g(r, "len_j"), g(r, "aligned"), g(r, "tm_i"), g(r, "tm_j"), g(r, "cov_i"), g(r, "cov_j"), g(r, "S"), g(r, "C"),
                         "" if S_pair is None else round(S_pair, 6), "" if C_pair is None else round(C_pair, 6),
                         "" if S_pair is None else int(S_pair >= TAU and C_pair >= COV), err])
    with open(os.path.join(OUT, "paired", f"pairwise_{ds}_combined.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pdb_i", "pdb_j",
                    "left_len_i", "left_len_j", "left_aligned", "left_tm_i", "left_tm_j", "left_cov_i", "left_cov_j", "left_S", "left_C",
                    "right_len_i", "right_len_j", "right_aligned", "right_tm_i", "right_tm_j", "right_cov_i", "right_cov_j", "right_S", "right_C",
                    "S_pair", "C_pair", "exposed_pair", "error"])
        w.writerows(rows)
    # matrices
    n = len(pdbs); idx = {p: i for i, p in enumerate(pdbs)}
    def mat(getter, diag):
        M = [[diag for _ in range(n)] for _ in range(n)]
        for (a, b), v in combined.items():
            ia, ib = idx[a], idx[b]
            x = getter(v); y = x
            M[ia][ib] = x; M[ib][ia] = y
        return M
    def write_mat(path, M, fmt):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["PDB"] + pdbs)
            for i, p in enumerate(pdbs):
                w.writerow([p] + [("" if x is None else fmt.format(x)) for x in M[i]])
    S_pair_M = mat(lambda v: v["S_pair"], 1.0)
    C_pair_M = mat(lambda v: v["C_pair"], 1.0)
    S_left_M = mat(lambda v: None if v["left"] is None else v["left"]["S"], 1.0)
    S_right_M = mat(lambda v: None if v["right"] is None else v["right"]["S"], 1.0)
    write_mat(os.path.join(OUT, "paired_complex_similarity_matrix.csv"), S_pair_M, "{:.5f}")
    write_mat(os.path.join(OUT, "paired_complex_coverage_matrix.csv"), C_pair_M, "{:.4f}")
    write_mat(os.path.join(OUT, "paired_complex_S_left_matrix.csv"), S_left_M, "{:.5f}")
    write_mat(os.path.join(OUT, "paired_complex_S_right_matrix.csv"), S_right_M, "{:.5f}")
    # stats of matrix
    stat = {"n_complexes": n, "n_pairs": len(combined),
            "n_pairs_exposed_tau0.7_cov0.8": sum(1 for v in combined.values() if v["S_pair"] is not None and v["S_pair"] >= TAU and v["C_pair"] >= COV)}
    return pdbs, man, combined, failures, stat


def folds_for(ds, split, seed):
    out = []
    if split == "AGCV":
        d = os.path.join(DS_ROOT, "splits", ds, seed, "AGCV")
        for k in range(5):
            tr = read_csv(os.path.join(d, f"fold_{k}_outer_train.csv"))
            ev = read_csv(os.path.join(d, f"fold_{k}_outer_eval.csv"))
            out.append((k, tr, ev))
    else:
        d = os.path.join(DS_ROOT, "splits", ds, seed, split)
        for k in range(5):
            tr = read_csv(os.path.join(d, f"train_fold_{k}.csv"))
            ev = read_csv(os.path.join(d, f"val_fold_{k}.csv"))
            out.append((k, tr, ev))
    return out


def analyse_fold(combined, train_rows, eval_rows):
    train_c = collections.Counter(r["PDB"] for r in train_rows)
    eval_c = collections.Counter(r["PDB"] for r in eval_rows)
    train_pdbs = set(train_c)
    eval_pdbs = sorted(eval_c)
    per_eval = []
    for v in eval_pdbs:
        def best(cands):
            bs = bc = None; bt = None; n_valid = 0
            for t in cands:
                if t == v:
                    # identical complex: S_pair = C_pair = 1 by definition
                    s, c = 1.0, 1.0
                else:
                    key = (v, t) if (v, t) in combined else (t, v)
                    rec = combined.get(key)
                    if rec is None or rec["S_pair"] is None:
                        continue
                    s, c = rec["S_pair"], rec["C_pair"]
                n_valid += 1
                if bs is None or s > bs:
                    bs, bc, bt = s, c, t
            return bs, bc, bt, n_valid
        cand_all = train_pdbs
        cand_nov = train_pdbs - {v}
        bs_i, bc_i, bt_i, nv_i = best(cand_all)
        bs_e, bc_e, bt_e, nv_e = best(cand_nov)
        exp_i = (bs_i is not None and bs_i >= TAU and bc_i >= COV)
        exp_e = (bs_e is not None and bs_e >= TAU and bc_e >= COV)
        # unresolved only if there were candidates but none had usable pair data
        unresolved = (len(cand_all) > 0 and nv_i == 0)
        per_eval.append({
            "PDB": v, "n_records": eval_c[v], "shared_with_train": v in train_pdbs,
            "n_train_candidates": len(cand_all),
            "best_S_pair_incl": bs_i, "best_C_pair_incl": bc_i, "best_train_pdb_incl": bt_i,
            "exposed_incl": int(exp_i), "unresolved": int(unresolved),
            "best_S_pair_excl": bs_e, "best_C_pair_excl": bc_e, "best_train_pdb_excl": bt_e,
            "exposed_excl": int(exp_e),
        })
    total_rec = sum(eval_c.values())
    total_cpx = len(eval_pdbs)
    unres_cpx = sum(p["unresolved"] for p in per_eval)
    unres_rec = sum(p["n_records"] for p in per_eval if p["unresolved"])
    # UNRESOLVED must NOT be counted as NOT_EXPOSED: rates use resolved denominators
    res_cpx = total_cpx - unres_cpx
    res_rec = total_rec - unres_rec
    exp_cpx_i = sum(p["exposed_incl"] for p in per_eval if not p["unresolved"])
    exp_rec_i = sum(p["n_records"] for p in per_eval if p["exposed_incl"] and not p["unresolved"])
    exp_cpx_e = sum(p["exposed_excl"] for p in per_eval if not p["unresolved"])
    exp_rec_e = sum(p["n_records"] for p in per_eval if p["exposed_excl"] and not p["unresolved"])
    return {
        "eval_rows": eval_rows, "per_eval": per_eval,
        "total_eval_complexes": total_cpx, "total_eval_records": total_rec,
        "resolved_eval_complexes": res_cpx, "resolved_eval_records": res_rec,
        "exposed_eval_complexes_incl": exp_cpx_i, "exposed_eval_records_incl": exp_rec_i,
        "complex_weighted_exposure_incl": exp_cpx_i / res_cpx if res_cpx else None,
        "record_weighted_exposure_incl": exp_rec_i / res_rec if res_rec else None,
        "exposed_eval_complexes_excl": exp_cpx_e, "exposed_eval_records_excl": exp_rec_e,
        "complex_weighted_exposure_excl": exp_cpx_e / res_cpx if res_cpx else None,
        "record_weighted_exposure_excl": exp_rec_e / res_rec if res_rec else None,
        "unresolved_complexes": unres_cpx, "unresolved_records": unres_rec,
    }


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return None, None
    return sum(vals) / len(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


def main():
    os.makedirs(os.path.join(OUT, "exposure"), exist_ok=True)
    fold_rows, eval_rows_out = [], []
    failures = []
    matrix_stat = {}
    combined_map = {}
    for ds in DATASETS:
        pdbs, man, combined, fails, stat = build_combined(ds)
        combined_map[ds] = combined
        failures += fails
        matrix_stat[ds] = stat
        for split in SPLITS[ds]:
            for seed in SEEDS:
                for k, tr, ev in folds_for(ds, split, seed):
                    res = analyse_fold(combined, tr, ev)
                    row = {"dataset": ds, "split_type": split, "split_seed": seed, "outer_fold": k,
                           "n_train_records": len(tr), "n_eval_records": res["total_eval_records"],
                           "n_train_complexes": len(set(r["PDB"] for r in tr)),
                           "n_eval_complexes": res["total_eval_complexes"]}
                    for key in ["exposed_eval_complexes_incl", "total_eval_complexes",
                                "resolved_eval_complexes", "total_eval_records", "resolved_eval_records",
                                "exposed_eval_records_incl",
                                "complex_weighted_exposure_incl", "record_weighted_exposure_incl",
                                "exposed_eval_complexes_excl", "exposed_eval_records_excl",
                                "complex_weighted_exposure_excl", "record_weighted_exposure_excl",
                                "unresolved_complexes", "unresolved_records"]:
                        row[key] = res[key]
                    fold_rows.append(row)
                    for p in res["per_eval"]:
                        r2 = {"dataset": ds, "split_type": split, "split_seed": seed, "outer_fold": k}
                        r2.update(p)
                        eval_rows_out.append(r2)
                    if res["unresolved_complexes"]:
                        for p in res["per_eval"]:
                            if p["unresolved"]:
                                failures.append([ds, p["PDB"], "", "unresolved_eval_complex",
                                                 f"{split}/{seed}/fold{k}: no usable pair data among {p['n_train_candidates']} candidates"])

    fold_fields = ["dataset", "split_type", "split_seed", "outer_fold",
                   "n_train_records", "n_eval_records", "n_train_complexes", "n_eval_complexes",
                   "exposed_eval_complexes_incl", "total_eval_complexes", "resolved_eval_complexes",
                   "exposed_eval_records_incl", "total_eval_records", "resolved_eval_records",
                   "complex_weighted_exposure_incl", "record_weighted_exposure_incl",
                   "exposed_eval_complexes_excl", "exposed_eval_records_excl",
                   "complex_weighted_exposure_excl", "record_weighted_exposure_excl",
                   "unresolved_complexes", "unresolved_records"]
    with open(os.path.join(OUT, "per_fold_paired_exposure.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fold_fields, extrasaction="ignore")
        w.writeheader()
        for r in fold_rows: w.writerow(r)

    eval_fields = ["dataset", "split_type", "split_seed", "outer_fold", "PDB", "n_records",
                   "shared_with_train", "n_train_candidates",
                   "best_S_pair_incl", "best_C_pair_incl", "best_train_pdb_incl", "exposed_incl", "unresolved",
                   "best_S_pair_excl", "best_C_pair_excl", "best_train_pdb_excl", "exposed_excl"]
    with open(os.path.join(OUT, "per_eval_complex_paired.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=eval_fields, extrasaction="ignore")
        w.writeheader()
        for r in eval_rows_out: w.writerow(r)

    # per seed (pool 5 folds)
    seed_rows = []
    for ds in DATASETS:
        for split in SPLITS[ds]:
            for seed in SEEDS:
                rs = [r for r in fold_rows if r["dataset"] == ds and r["split_type"] == split and r["split_seed"] == seed]
                tr = sum(r["resolved_eval_records"] for r in rs)
                tc = sum(r["resolved_eval_complexes"] for r in rs)
                eri = sum(r["exposed_eval_records_incl"] for r in rs)
                eci = sum(r["exposed_eval_complexes_incl"] for r in rs)
                ere = sum(r["exposed_eval_records_excl"] for r in rs)
                ece = sum(r["exposed_eval_complexes_excl"] for r in rs)
                seed_rows.append({
                    "dataset": ds, "split_type": split, "split_seed": seed, "n_folds": len(rs),
                    "total_eval_complexes": sum(r["total_eval_complexes"] for r in rs),
                    "total_eval_records": sum(r["total_eval_records"] for r in rs),
                    "resolved_eval_complexes": tc, "resolved_eval_records": tr,
                    "exposed_eval_complexes_incl": eci, "exposed_eval_records_incl": eri,
                    "complex_weighted_exposure_incl": eci / tc if tc else None,
                    "record_weighted_exposure_incl": eri / tr if tr else None,
                    "exposed_eval_complexes_excl": ece, "exposed_eval_records_excl": ere,
                    "complex_weighted_exposure_excl": ece / tc if tc else None,
                    "record_weighted_exposure_excl": ere / tr if tr else None,
                    "unresolved_records": sum(r["unresolved_records"] for r in rs),
                    "record_weighted_incl_mean_fold": mean_sd([r["record_weighted_exposure_incl"] for r in rs])[0],
                    "record_weighted_incl_sd_fold": mean_sd([r["record_weighted_exposure_incl"] for r in rs])[1],
                })
    with open(os.path.join(OUT, "per_seed_paired_exposure.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(seed_rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        for r in seed_rows: w.writerow(r)

    # S1 aggregated across seeds & folds
    s1 = []
    for ds in DATASETS:
        for split in SPLITS[ds]:
            rs = [r for r in fold_rows if r["dataset"] == ds and r["split_type"] == split]
            tr = sum(r["resolved_eval_records"] for r in rs); tc = sum(r["resolved_eval_complexes"] for r in rs)
            eri = sum(r["exposed_eval_records_incl"] for r in rs); eci = sum(r["exposed_eval_complexes_incl"] for r in rs)
            ere = sum(r["exposed_eval_records_excl"] for r in rs); ece = sum(r["exposed_eval_complexes_excl"] for r in rs)
            rw_i_mean, rw_i_sd = mean_sd([r["record_weighted_exposure_incl"] for r in rs])
            rw_e_mean, rw_e_sd = mean_sd([r["record_weighted_exposure_excl"] for r in rs])
            cw_i_mean, cw_i_sd = mean_sd([r["complex_weighted_exposure_incl"] for r in rs])
            cw_e_mean, cw_e_sd = mean_sd([r["complex_weighted_exposure_excl"] for r in rs])
            s1.append({
                "dataset": ds, "split_type": split, "n_cells": len(rs),
                "total_eval_complexes": sum(r["total_eval_complexes"] for r in rs),
                "total_eval_records": sum(r["total_eval_records"] for r in rs),
                "resolved_eval_complexes": tc, "resolved_eval_records": tr,
                "exposed_eval_complexes_incl": eci, "exposed_eval_records_incl": eri,
                "exposed_eval_complexes_excl": ece, "exposed_eval_records_excl": ere,
                "record_weighted_exposure_incl_pooled": eri / tr if tr else None,
                "record_weighted_exposure_incl_mean": rw_i_mean, "record_weighted_exposure_incl_sd": rw_i_sd,
                "record_weighted_exposure_excl_pooled": ere / tr if tr else None,
                "record_weighted_exposure_excl_mean": rw_e_mean, "record_weighted_exposure_excl_sd": rw_e_sd,
                "complex_weighted_exposure_incl_pooled": eci / tc if tc else None,
                "complex_weighted_exposure_incl_mean": cw_i_mean, "complex_weighted_exposure_incl_sd": cw_i_sd,
                "complex_weighted_exposure_excl_pooled": ece / tc if tc else None,
                "complex_weighted_exposure_excl_mean": cw_e_mean, "complex_weighted_exposure_excl_sd": cw_e_sd,
                "unresolved_records": sum(r["unresolved_records"] for r in rs),
            })
    with open(os.path.join(OUT, "supplementary_table_S1_paired_exposure.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(s1[0].keys()), extrasaction="ignore")
        w.writeheader()
        for r in s1: w.writerow(r)

    # failures
    with open(os.path.join(OUT, "failed_or_undetermined.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "object_a", "object_b", "category", "detail"])
        w.writerows(failures)
    with open(os.path.join(OUT, "matrix_summary.json"), "w", encoding="utf-8") as f:
        json.dump(matrix_stat, f, indent=2)

    print("fold rows:", len(fold_rows), "eval rows:", len(eval_rows_out), "failures:", len(failures))
    print()
    print(f"{'dataset':7s} {'split':5s} {'cells':5s} {'recW_incl':>9s} {'recW_excl':>9s} {'cpxW_incl':>9s} {'cpxW_excl':>9s} {'unres':>5s}")
    for r in s1:
        def f(x): return f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "
        print(f"{r['dataset']:7s} {r['split_type']:5s} {r['n_cells']:5d} "
              f"{f(r['record_weighted_exposure_incl_mean']):>9s} {f(r['record_weighted_exposure_excl_mean']):>9s} "
              f"{f(r['complex_weighted_exposure_incl_mean']):>9s} {f(r['complex_weighted_exposure_excl_mean']):>9s} "
              f"{r['unresolved_records']:>5d}")


if __name__ == "__main__":
    main()
