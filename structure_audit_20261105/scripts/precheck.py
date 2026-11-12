#!/usr/bin/env python3
"""Phase-1 read-only precheck for the two-sided structural similarity audit.

Does NOT run any structural alignment. Only inventories files, parses PDB
chain IDs, maps data records to structures, and audits split CSVs.

Outputs (all under structure_audit_20261105/):
  manifests/structure_manifest.csv
  manifests/chain_mapping_audit.csv
  precheck/csv_integrity_report.csv
  precheck/csv_integrity_summary.json
  precheck/inventory_summary.json
  precheck/anomalies.csv
"""

import os
import csv
import json
import collections

ROOT = "/data"
DS = os.path.join(ROOT, "data_split")
OUT = os.path.join(ROOT, "structure_audit_20261105")
DATASETS = ["S1131", "M1101"]
SEEDS = ["0", "42", "1024", "1234", "2026"]
NORMAL_SPLITS = ["RCSV", "PCSV", "ICSV"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        rows = list(r)
        return rows, r.fieldnames


def parse_pdb_chains(path):
    cs = set()
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                ch = line[21]
                cs.add(ch if ch.strip() else "_")
    return cs


def classify_wt_files(files):
    """Split wt listing into full complexes, per-chain slices, and xvg."""
    full = {}
    perchain = collections.defaultdict(list)
    xvg = []
    other = []
    for f in sorted(files):
        if f.endswith(".xvg"):
            xvg.append(f)
            continue
        if not f.endswith(".pdb"):
            other.append(f)
            continue
        stem = f[:-4]  # drop .pdb
        parts = stem.split(".")
        if len(parts) == 1:
            full[parts[0]] = f
        elif len(parts) == 2 and 1 <= len(parts[1]) <= 2:
            perchain[parts[0]].append((parts[1], f))
        else:
            other.append(f)
    return full, perchain, xvg, other


def classify_mut_files(files):
    pdb = [f for f in files if f.endswith(".pdb")]
    xvg = [f for f in files if f.endswith(".xvg")]
    other = [f for f in files if not f.endswith((".pdb", ".xvg"))]
    return pdb, xvg, other


def clean_chain(s):
    return s.strip() if s else ""


def group_chains(*vals):
    out = []
    for v in vals:
        v = clean_chain(v)
        if v:
            for ch in v.split(","):
                ch = ch.strip()
                if ch:
                    out.append(ch)
    # dedupe preserving order
    seen = set()
    res = []
    for ch in out:
        if ch not in seen:
            seen.add(ch)
            res.append(ch)
    return res


def sig_of(row, fields):
    return tuple((row.get(c) or "") for c in fields)


def counter_over_shape(seq_of_counters):
    tot = collections.Counter()
    for c in seq_of_counters:
        tot.update(c)
    return tot


def main():
    os.makedirs(os.path.join(OUT, "manifests"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "precheck"), exist_ok=True)
    anomalies = []

    inventory = {}
    manifest_rows = []
    chain_rows = []

    canon = {}          # ds -> list of dict
    canon_fields = {}   # ds -> fieldnames
    canon_counter = {}  # ds -> Counter(sig)
    pdb_to_complex = {} # ds -> {PDB: info}

    for ds in DATASETS:
        wt_dir = os.path.join(DS, "splits", ds, "wt")
        mut_dir = os.path.join(DS, "splits", ds, "mut")
        wt_files = os.listdir(wt_dir)
        mut_files = os.listdir(mut_dir)
        full, perchain, wt_xvg, wt_other = classify_wt_files(wt_files)
        mut_pdb, mut_xvg, mut_other = classify_mut_files(mut_files)

        # canonical records
        cpath = os.path.join(DS, "raw", f"{ds}.csv")
        rows, fields = read_csv(cpath)
        canon[ds] = rows
        canon_fields[ds] = fields
        canon_counter[ds] = collections.Counter(sig_of(r, fields) for r in rows)

        # map PDB -> wt full file
        # verify every PDB referenced by records has a full wt file
        pdb_refs = collections.defaultdict(list)
        for idx, r in enumerate(rows):
            pdb_refs[r["PDB"]].append(idx)

        inv = {
            "dataset": ds,
            "wt_files_total": len(wt_files),
            "wt_full_complexes": len(full),
            "wt_perchain_slices": sum(len(v) for v in perchain.values()),
            "wt_xvg": len(wt_xvg),
            "wt_other": len(wt_other),
            "mut_files_total": len(mut_files),
            "mut_pdb": len(mut_pdb),
            "mut_xvg": len(mut_xvg),
            "mut_other": len(mut_other),
            "canonical_records": len(rows),
            "unique_pdb_in_records": len(pdb_refs),
            "unique_ID_in_records": len(set(r["ID"] for r in rows)),
        }
        inventory[ds] = inv

        # PDBs referenced but no full wt file
        missing_complex = sorted(p for p in pdb_refs if p not in full)
        extra_complex = sorted(p for p in full if p not in pdb_refs)
        inv["referenced_pdb_without_wt_full"] = missing_complex
        inv["wt_full_without_records"] = extra_complex
        if missing_complex:
            anomalies.append([ds, "missing_wt_complex", ";".join(missing_complex)])
        if extra_complex:
            anomalies.append([ds, "unused_wt_complex", ";".join(extra_complex)])

        # per complex manifest and chain audit
        for pdb in sorted(pdb_refs):
            idxs = pdb_refs[pdb]
            wt_full = full.get(pdb, "")
            wt_path = os.path.join(wt_dir, wt_full) if wt_full else ""
            wt_chains = parse_pdb_chains(wt_path) if wt_full and os.path.exists(wt_path) else set()
            is_hm = pdb.startswith("HM_")
            partner_patterns = collections.Counter()
            left_all = collections.Counter()
            right_all = collections.Counter()
            mut_expected = 0
            mut_present = 0
            per_rec_missing = 0
            ids = set()
            for idx in idxs:
                r = rows[idx]
                ids.add(r["ID"])
                heavy = clean_chain(r.get("Heavy_chain"))
                light = clean_chain(r.get("Light_chain"))
                a1 = clean_chain(r.get("Antigen_chain1"))
                a2 = clean_chain(r.get("Antigen_chain2"))
                left = group_chains(heavy, light)
                right = group_chains(a1, a2)
                partner_patterns[r.get("Partners", "")] += 1
                left_all["".join(left)] += 1
                right_all["".join(right)] += 1

                # partners string consistency
                parts = (r.get("Partners", "") or "").split("_")
                expected_left = parts[0] if len(parts) >= 1 else ""
                expected_right = parts[1] if len(parts) >= 2 else ""
                consistent = (expected_left == "".join(left) and expected_right == "".join(right))
                if not consistent:
                    anomalies.append([ds, "partners_vs_chain_columns_mismatch",
                                      f"{r['record_uid'] if 'record_uid' in r else idx}:{r.get('Partners')}:{''.join(left)}_{''.join(right)}"])

                need = set(left) | set(right)
                missing = sorted(ch for ch in need if ch not in wt_chains)

                # mut file expected = ID + .pdb
                mut_name = r["ID"] + ".pdb"
                mut_expected += 1
                mp = os.path.join(mut_dir, mut_name)
                mp_exists = os.path.exists(mp)
                if mp_exists:
                    mut_present += 1
                else:
                    per_rec_missing += 1

                chain_rows.append([
                    ds,
                    f"{ds}:{idx:06d}",
                    r["ID"],
                    pdb,
                    r.get("Partners", ""),
                    heavy, light, a1, a2,
                    "".join(left), "".join(right),
                    "yes" if consistent else "no",
                    wt_full,
                    "".join(sorted(c for c in wt_chains if c.strip())) or "-",
                    "".join(missing),
                    mut_name,
                    "present" if mp_exists else "MISSING",
                    "OK" if (not missing and consistent and mp_exists) else "CHECK",
                ])

            manifest_rows.append([
                ds, pdb,
                "HM" if is_hm else "standard",
                wt_full,
                "present" if (wt_full and os.path.exists(wt_path)) else "MISSING",
                "".join(sorted(c for c in wt_chains if c.strip())) or "-",
                ";".join(f"{ch}:{fn}" for ch, fn in perchain.get(pdb, [])),
                "present" if any(x.startswith(pdb + ".pdb.xvg") for x in wt_xvg) else "-",
                len(idxs),
                len(ids),
                ";".join(sorted(partner_patterns)),
                ";".join(f"{k}:{v}" for k, v in left_all.most_common()),
                ";".join(f"{k}:{v}" for k, v in right_all.most_common()),
                mut_expected, mut_present, per_rec_missing,
            ])

            pdb_to_complex[(ds, pdb)] = {
                "wt_full": wt_full,
                "chains": sorted(c for c in wt_chains if c.strip()),
                "is_hm": is_hm,
            }

    # write manifest
    with open(os.path.join(OUT, "manifests", "structure_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "PDB", "complex_type", "wt_full_file", "wt_full_status",
            "wt_chains", "wt_perchain_slices", "wt_xvg_present",
            "n_records", "n_unique_ID", "partner_patterns",
            "left_partner_chains_and_counts", "right_partner_chains_and_counts",
            "mut_files_expected", "mut_files_present", "mut_files_missing",
        ])
        w.writerows(manifest_rows)

    with open(os.path.join(OUT, "manifests", "chain_mapping_audit.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "record_uid", "ID", "PDB", "Partners",
            "Heavy_chain", "Light_chain", "Antigen_chain1", "Antigen_chain2",
            "left_partner_chains", "right_partner_chains", "partners_consistent",
            "wt_full_file", "chains_in_wt", "missing_chains_in_wt",
            "mut_file", "mut_file_status", "mapping_status",
        ])
        w.writerows(chain_rows)

    # ---- Split CSV integrity audit ----
    report_rows = []
    summary = {}

    for ds in DATASETS:
        cfields = canon_fields[ds]
        ccount = canon_counter[ds]
        N = sum(ccount.values())
        summary[ds] = {"canonical_multiset_size": N, "unique_signatures": len(ccount)}

        # normal splits
        for split in NORMAL_SPLITS:
            seed_alloc = {}
            all_ok = True
            for seed in SEEDS:
                d = os.path.join(DS, "splits", ds, seed, split)
                train_counters = []
                val_counters = []
                fold_ok_all = True
                for k in range(5):
                    tpath = os.path.join(d, f"train_fold_{k}.csv")
                    vpath = os.path.join(d, f"val_fold_{k}.csv")
                    if not (os.path.exists(tpath) and os.path.exists(vpath)):
                        anomalies.append([ds, "missing_fold_file", f"{seed}/{split}/fold{k}"])
                        fold_ok_all = False
                        continue
                    trows, tf = read_csv(tpath)
                    vrows, vf = read_csv(vpath)
                    tc = collections.Counter(sig_of(r, cfields) for r in trows)
                    vc = collections.Counter(sig_of(r, cfields) for r in vrows)
                    train_counters.append(tc)
                    val_counters.append(vc)
                    # partition check: train+val multiset == canonical
                    part_ok = (tc + vc) == ccount
                    if not part_ok:
                        fold_ok_all = False
                        anomalies.append([ds, "fold_partition_mismatch", f"{seed}/{split}/fold{k} train={sum(tc.values())} val={sum(vc.values())}"])
                    report_rows.append([ds, split, seed, k, "normal",
                                        sum(tc.values()), sum(vc.values()),
                                        "PASS" if part_ok else "FAIL"])
                # val union across folds == canonical
                val_union = counter_over_shape(val_counters)
                val_union_ok = (val_union == ccount)
                if not val_union_ok:
                    anomalies.append([ds, "val_union_not_full", f"{seed}/{split}"])
                # build val fold assignment signature for uniqueness across seeds
                fold_of = {}
                for k, vc in enumerate(val_counters):
                    for s in vc:
                        fold_of.setdefault(s, []).append(k)
                alloc_key = tuple(sorted((s, tuple(sorted(v))) for s, v in fold_of.items()))
                seed_alloc[seed] = hash(alloc_key)
                summary.setdefault(ds, {}).setdefault(split, {})[seed] = {
                    "fold_ok": fold_ok_all,
                    "val_union_full": val_union_ok,
                    "n_train_per_fold": [sum(c.values()) for c in train_counters],
                    "n_val_per_fold": [sum(c.values()) for c in val_counters],
                }
                all_ok = all_ok and fold_ok_all and val_union_ok
            nunique = len(set(seed_alloc.values()))
            summary[ds][split]["unique_outer_allocations_across_seeds"] = nunique
            summary[ds][split]["all_pass"] = all_ok

        # AGCV
        if ds == "M1101":
            agcv_seed_alloc = {}
            agcv_ok = True
            for seed in SEEDS:
                d = os.path.join(DS, "splits", ds, seed, "AGCV")
                agcv_summary = {"folds": {}}
                fold_assign = {}
                for k in range(5):
                    def upath(tag):
                        return os.path.join(d, f"fold_{k}_{tag}.csv")
                    tr_out, e_out, tr_in, v_in = [], [], [], []
                    okf = True
                    paths = {t: upath(t) for t in ["outer_train", "outer_eval", "inner_train", "inner_val"]}
                    for t, p in paths.items():
                        if not os.path.exists(p):
                            anomalies.append([ds, "missing_agcv_file", f"{seed}/fold{k}/{t}"])
                            okf = False
                    if okf:
                        tr_out, _ = read_csv(paths["outer_train"])
                        e_out, _ = read_csv(paths["outer_eval"])
                        tr_in, _ = read_csv(paths["inner_train"])
                        v_in, _ = read_csv(paths["inner_val"])
                        uout = collections.Counter(r["record_uid"] for r in tr_out)
                        uev = collections.Counter(r["record_uid"] for r in e_out)
                        uin = collections.Counter(r["record_uid"] for r in tr_in)
                        uiv = collections.Counter(r["record_uid"] for r in v_in)
                        outer_is_union = (uin + uiv) == uout
                        inner_disjoint = len(set(uin) & set(uiv)) == 0
                        outer_eval_disjoint = len(set(uout) & set(uev)) == 0
                        eval_frac = sum(uiv.values()) / max(1, sum(uout.values()))
                        for r in e_out:
                            fold_assign.setdefault(r["record_uid"], []).append(k)
                        report_rows.append([ds, "AGCV", seed, k, "agcv",
                                            sum(uout.values()), sum(uev.values()),
                                            "PASS" if (outer_is_union and inner_disjoint and outer_eval_disjoint and eval_frac < 0.5) else "FAIL"])
                        if not (outer_is_union and inner_disjoint and outer_eval_disjoint):
                            okf = False
                            anomalies.append([ds, "agcv_partition_problem", f"{seed}/fold{k} union={outer_is_union} in_dis={inner_disjoint} out_dis={outer_eval_disjoint}"])
                        agcv_summary["folds"][k] = {
                            "n_outer_train": sum(uout.values()),
                            "n_outer_eval": sum(uev.values()),
                            "n_inner_train": sum(uin.values()),
                            "n_inner_val": sum(uiv.values()),
                            "outer_train_eq_inner_union": outer_is_union,
                            "inner_disjoint": inner_disjoint,
                            "outer_eval_disjoint": outer_eval_disjoint,
                            "inner_val_frac_of_outer_train": round(eval_frac, 4),
                        }
                    agcv_ok = agcv_ok and okf
                # unique outer allocation
                key = tuple(sorted((u, tuple(sorted(v))) for u, v in fold_assign.items()))
                agcv_seed_alloc[seed] = hash(key)
                summary[ds].setdefault("AGCV", {})[seed] = agcv_summary
            summary[ds]["AGCV"]["unique_outer_allocations_across_seeds"] = len(set(agcv_seed_alloc.values()))
            summary[ds]["AGCV"]["all_pass"] = agcv_ok

    with open(os.path.join(OUT, "precheck", "csv_integrity_report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "split", "seed", "fold", "mode", "n_train", "n_eval", "partition_status"])
        w.writerows(report_rows)

    with open(os.path.join(OUT, "precheck", "csv_integrity_summary.json"), "w", encoding="utf-8") as f:
        # strip non-serializable? all json-safe here
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT, "precheck", "inventory_summary.json"), "w", encoding="utf-8") as f:
        json.dump(inventory, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT, "precheck", "anomalies.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "anomaly_type", "detail"])
        w.writerows(anomalies)

    print("=== INVENTORY ===")
    for ds in DATASETS:
        print(ds, json.dumps(inventory[ds], ensure_ascii=False))
    print()
    print("=== SPLIT SUMMARY (key) ===")
    for ds in DATASETS:
        for split in NORMAL_SPLITS + (["AGCV"] if ds == "M1101" else []):
            info = summary[ds].get(split, {})
            print(ds, split, "all_pass=", info.get("all_pass"),
                  "unique_alloc=", info.get("unique_outer_allocations_across_seeds"))
    print()
    print("anomalies:", len(anomalies))
    for a in anomalies[:40]:
        print("  ", a)


if __name__ == "__main__":
    main()
