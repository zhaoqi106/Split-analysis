#!/usr/bin/env python3
"""Build FINAL_TABLE_S1.csv with explicit numerator/denominator/aggregation.

Main statistic (Section 3.1):
  seed-pooled record-weighted exposure  = for each seed, pool its 5 folds
  (sum exposed eval records / sum eval records), then mean and SAMPLE SD (n=5)
  across the 5 seeds.  Identical-complex matches INCLUDED (RCSV expected).
Supplementary:
  25-fold arithmetic mean and sample SD (n=25) of fold-level record-weighted;
  complex-weighted exposure (seed-pooled mean/SD and grand pooled);
  excluding-identical variant for RCSV (denominator = all valid eval records).
"""
import os, csv, statistics as st

OUT = "/data/structure_audit_paired_20261105"
SEEDS = ["0", "42", "1024", "1234", "2026"]


def read(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def mean_sd(vals):
    vals = [float(v) for v in vals if v not in (None, "", "None")]
    m = sum(vals) / len(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0  # sample SD (ddof=1)
    return m, sd


def main():
    fold = read(os.path.join(OUT, "per_fold_paired_exposure.csv"))
    seed = read(os.path.join(OUT, "per_seed_paired_exposure.csv"))
    ds_order = ["S1131", "M1101"]
    split_order = {"S1131": ["RCSV", "PCSV", "ICSV"], "M1101": ["RCSV", "PCSV", "ICSV", "AGCV"]}

    fields = ["dataset", "split_type", "n_seeds", "n_folds",
              "MAIN_caliber",
              "MAIN_record_incl_numerator", "MAIN_record_incl_denominator",
              "MAIN_record_incl_seedpooled_mean", "MAIN_record_incl_seedpooled_sd_n5",
              "MAIN_record_incl_grandpooled",
              "SUPP_record_incl_fold25_mean", "SUPP_record_incl_fold25_sd_n25",
              "EXCL_record_numerator", "EXCL_record_denominator",
              "EXCL_record_seedpooled_mean", "EXCL_record_seedpooled_sd_n5", "EXCL_record_grandpooled",
              "SUPP_cpx_incl_seedpooled_mean", "SUPP_cpx_incl_seedpooled_sd_n5", "SUPP_cpx_incl_grandpooled",
              "SUPP_cpx_incl_fold25_mean", "SUPP_cpx_incl_fold25_sd_n25",
              "SUPP_cpx_excl_seedpooled_mean", "SUPP_cpx_excl_grandpooled",
              "identical_pdb_possible", "note"]
    rows = []
    for ds in ds_order:
        for sp in split_order[ds]:
            srows = [r for r in seed if r["dataset"] == ds and r["split_type"] == sp]
            frows = [r for r in fold if r["dataset"] == ds and r["split_type"] == sp]
            # per-seed pooled values (5)
            rec_i = [r["record_weighted_exposure_incl"] for r in srows]
            rec_e = [r["record_weighted_exposure_excl"] for r in srows]
            cpx_i = [r["complex_weighted_exposure_incl"] for r in srows]
            cpx_e = [r["complex_weighted_exposure_excl"] for r in srows]
            mi_m, mi_sd = mean_sd(rec_i)
            me_m, me_sd = mean_sd(rec_e)
            ci_m, ci_sd = mean_sd(cpx_i)
            ce_m, ce_sd = mean_sd(cpx_e)
            # fold25
            f25 = [r["record_weighted_exposure_incl"] for r in frows]
            f25_m, f25_sd = mean_sd(f25)
            f25c = [r["complex_weighted_exposure_incl"] for r in frows]
            f25c_m, f25c_sd = mean_sd(f25c)
            # grand pooled numerator/denominator (incl)
            num_i = sum(int(r["exposed_eval_records_incl"]) for r in frows)
            den = sum(int(r["resolved_eval_records"]) for r in frows)
            num_e = sum(int(r["exposed_eval_records_excl"]) for r in frows)
            cnum_i = sum(int(r["exposed_eval_complexes_incl"]) for r in frows)
            cden = sum(int(r["resolved_eval_complexes"]) for r in frows)
            cnum_e = sum(int(r["exposed_eval_complexes_excl"]) for r in frows)
            identical_possible = "yes" if sp == "RCSV" else "no"
            note = ("RCSV: identical-PDB cross-fold matching is EXPECTED and kept in MAIN; "
                    "EXCL removes identical complex but denominator stays ALL valid eval records (%.0f)." % den) if sp == "RCSV" else \
                   ("No identical PDB across folds, so incl==excl by construction." if sp in ("PCSV", "ICSV") else
                    "AGCV: no identical PDB; incl==excl. 0.350=seed-pooled recW, 0.330=fold25 arithmetic mean.")
            rows.append({
                "dataset": ds, "split_type": sp, "n_seeds": len(srows), "n_folds": len(frows),
                "MAIN_caliber": "seed-pooled record-weighted, INCLUDING identical complex",
                "MAIN_record_incl_numerator": num_i, "MAIN_record_incl_denominator": den,
                "MAIN_record_incl_seedpooled_mean": round(mi_m, 4), "MAIN_record_incl_seedpooled_sd_n5": round(mi_sd, 4),
                "MAIN_record_incl_grandpooled": round(num_i / den, 4) if den else None,
                "SUPP_record_incl_fold25_mean": round(f25_m, 4), "SUPP_record_incl_fold25_sd_n25": round(f25_sd, 4),
                "EXCL_record_numerator": num_e, "EXCL_record_denominator": den,
                "EXCL_record_seedpooled_mean": round(me_m, 4), "EXCL_record_seedpooled_sd_n5": round(me_sd, 4),
                "EXCL_record_grandpooled": round(num_e / den, 4) if den else None,
                "SUPP_cpx_incl_seedpooled_mean": round(ci_m, 4), "SUPP_cpx_incl_seedpooled_sd_n5": round(ci_sd, 4),
                "SUPP_cpx_incl_grandpooled": round(cnum_i / cden, 4) if cden else None,
                "SUPP_cpx_incl_fold25_mean": round(f25c_m, 4), "SUPP_cpx_incl_fold25_sd_n25": round(f25c_sd, 4),
                "SUPP_cpx_excl_seedpooled_mean": round(ce_m, 4),
                "SUPP_cpx_excl_grandpooled": round(cnum_e / cden, 4) if cden else None,
                "identical_pdb_possible": identical_possible, "note": note,
            })

    with open(os.path.join(OUT, "FINAL_TABLE_S1.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # pretty print
    print(f"{'dataset':6s}{'split':5s} | MAIN recW incl seed-pooled mean±sd(n=5) | num/den | EXCL seed-pooled mean±sd | fold25 recW mean±sd | cpx-incl seed-pooled ± | cpx-incl fold25 ±")
    for r in rows:
        print(f"{r['dataset']:6s}{r['split_type']:5s} | "
              f"{r['MAIN_record_incl_seedpooled_mean']:.3f} ± {r['MAIN_record_incl_seedpooled_sd_n5']:.3f}  | "
              f"{r['MAIN_record_incl_numerator']}/{r['MAIN_record_incl_denominator']} = {r['MAIN_record_incl_grandpooled']:.3f} | "
              f"{r['EXCL_record_seedpooled_mean']:.3f} ± {r['EXCL_record_seedpooled_sd_n5']:.3f} | "
              f"{r['SUPP_record_incl_fold25_mean']:.3f} ± {r['SUPP_record_incl_fold25_sd_n25']:.3f} | "
              f"{r['SUPP_cpx_incl_seedpooled_mean']:.3f} ± {r['SUPP_cpx_incl_seedpooled_sd_n5']:.3f} | "
              f"{r['SUPP_cpx_incl_fold25_mean']:.3f} ± {r['SUPP_cpx_incl_fold25_sd_n25']:.3f}")
    print("\nwrote FINAL_TABLE_S1.csv")


if __name__ == "__main__":
    main()
