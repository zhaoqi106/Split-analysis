#!/usr/bin/env python3
"""Build partner-chain manifest and extract left/right partner WT structures.

Partner definitions (paper / Supplementary Methods):
  S1131 : left  = chains carrying R1-R6 (verified == Heavy_chain + Light_chain);
          right = chains carrying L1-L3 (verified == Antigen_chain1 + Antigen_chain2).
  M1101 : left  = antibody side from metadata (antibody_chains);
          right = antigen  side from metadata (antigen_chains);
          heavy+light are one partner; multi-chain antigen one partner.
          metadata source: /data/data_split/splits/M1101/0/AGCV/antigen_mapping.csv
          (for 1KTZ / HM_1KTZ the metadata roles are swapped vs CSV columns;
           metadata is authoritative, and this swap is recorded explicitly).

Does NOT touch structure_audit_20261105/.
"""
import os, csv, collections, json

ROOT = "/data"
DS_ROOT = os.path.join(ROOT, "data_split")
OUT = os.path.join(ROOT, "structure_audit_paired_20261105")
DATASETS = ["S1131", "M1101"]


def ne(x):
    return (x or "").strip() != ""


def chains_in_pdb(path):
    cs = set()
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                ch = line[21]
                cs.add(ch if ch.strip() else "_")
    return cs


def atom_chains_in_pdb(path):
    cs = set()
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM"):
                ch = line[21]
                cs.add(ch if ch.strip() else "_")
    return cs


def write_partner(src, chains, dst):
    keep = set(chains)
    n = 0
    with open(src, errors="ignore") as f, open(dst, "w") as o:
        for line in f:
            if line.startswith("ATOM") and line[21] in keep:
                o.write(line); n += 1
        o.write("END\n")
    return n


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "partners", "S1131"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "partners", "M1101"), exist_ok=True)

    m1101_meta = {}
    with open(os.path.join(DS_ROOT, "splits", "M1101", "0", "AGCV", "antigen_mapping.csv"), newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            m1101_meta[r["record_uid"]] = r

    manifest_rows = []
    role_conflicts = []
    summary = {}

    for ds in DATASETS:
        with open(os.path.join(DS_ROOT, "raw", f"{ds}.csv"), newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        # group records by PDB
        by_pdb = collections.OrderedDict()
        for idx, r in enumerate(rows):
            by_pdb.setdefault(r["PDB"], []).append((idx, r))

        n_multi_left = n_multi_right = 0
        for pdb, recs in by_pdb.items():
            left_sets = set(); right_sets = set(); role_notes = set()
            role_source = ""
            for idx, r in recs:
                csv_left = [c for c in [r.get("Heavy_chain"), r.get("Light_chain")] if ne(c)]
                csv_right = [c for c in [r.get("Antigen_chain1"), r.get("Antigen_chain2")] if ne(c)]
                # R/L column counts
                nR = sum(ne(r.get(f"R{i}")) for i in range(1, 7))
                nL = sum(ne(r.get(f"L{i}")) for i in range(1, 4))
                if ds == "S1131":
                    left = csv_left; right = csv_right
                    role_source = "S1131:R1-R6/L1-L3+Heavy/Light/Antigen"
                    if nR != len(left) or nL != len(right):
                        role_conflicts.append([ds, pdb, f"record{idx}", f"nR={nR}/L={len(left)} nL={nL}/R={len(right)}"])
                else:
                    uid = f"M1101:{idx:06d}"
                    a = m1101_meta.get(uid)
                    if a is None:
                        role_conflicts.append([ds, pdb, f"record{idx}", "no metadata"])
                        left, right = csv_left, csv_right
                        role_source = "M1101:CSV_fallback"
                    else:
                        left = [x for x in a["antibody_chains"].replace(" ", "").split(",") if x]
                        right = [x for x in a["antigen_chains"].replace(" ", "").split(",") if x]
                        role_source = "M1101:antigen_mapping.antibody_chains/antigen_chains"
                        # cross-check vs CSV (order-insensitive)
                        if set(left) == set(csv_left) and set(right) == set(csv_right):
                            role_notes.add("metadata==CSV")
                        elif set(left) == set(csv_right) and set(right) == set(csv_left):
                            role_notes.add("metadata_swapped_vs_CSV_columns")
                            role_conflicts.append([ds, pdb, f"record{idx}",
                                                   f"metadata_left={''.join(left)} metadata_right={''.join(right)} "
                                                   f"CSV_left={''.join(csv_left)} CSV_right={''.join(csv_right)}"])
                        else:
                            role_notes.add("metadata_vs_CSV_INCONSISTENT")
                            role_conflicts.append([ds, pdb, f"record{idx}",
                                                   f"metadata={''.join(left)}|{''.join(right)} CSV={''.join(csv_left)}|{''.join(csv_right)}"])
                left_sets.add(tuple(left)); right_sets.add(tuple(right))
            if len(left_sets) > 1 or len(right_sets) > 1:
                role_conflicts.append([ds, pdb, "MULTIPLE", f"left={left_sets} right={right_sets}"])
            left = list(sorted(left_sets)[0]) if left_sets else []
            right = list(sorted(right_sets)[0]) if right_sets else []

            wt_file = pdb + ".pdb"
            wt_path = os.path.join(DS_ROOT, "splits", ds, "wt", wt_file)
            wt_chains = atom_chains_in_pdb(wt_path) if os.path.exists(wt_path) else set()
            left_missing = [c for c in left if c not in wt_chains]
            right_missing = [c for c in right if c not in wt_chains]
            partition_ok = (set(left) | set(right)) == wt_chains and not (set(left) & set(right)) and not left_missing and not right_missing

            lp = os.path.join(OUT, "partners", ds, f"{pdb}.left.pdb")
            rp = os.path.join(OUT, "partners", ds, f"{pdb}.right.pdb")
            nl = write_partner(wt_path, left, lp) if not left_missing and os.path.exists(wt_path) else 0
            nr = write_partner(wt_path, right, rp) if not right_missing and os.path.exists(wt_path) else 0
            if len(left) > 1: n_multi_left += 1
            if len(right) > 1: n_multi_right += 1

            manifest_rows.append([
                ds, pdb, "HM" if pdb.startswith("HM_") else "standard",
                wt_file, role_source,
                ",".join(left), ",".join(right), len(left), len(right),
                len(recs),
                ", ".join(sorted(role_notes)) if role_notes else "",
                f"{pdb}.left.pdb", f"{pdb}.right.pdb",
                "".join(sorted(left_missing)), "".join(sorted(right_missing)),
                "yes" if partition_ok else "NO",
                nl, nr,
            ])
        summary[ds] = {"complexes": len(by_pdb), "multi_left": n_multi_left, "multi_right": n_multi_right}

    with open(os.path.join(OUT, "partner_chain_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "PDB", "complex_type", "wt_full_file", "roles_source",
                    "left_chains", "right_chains", "n_left_chains", "n_right_chains", "n_records",
                    "role_notes", "left_partner_pdb", "right_partner_pdb",
                    "left_missing_chains", "right_missing_chains", "chain_partition_ok",
                    "left_atom_lines", "right_atom_lines"])
        w.writerows(manifest_rows)

    with open(os.path.join(OUT, "partner_chain_conflicts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "PDB", "record", "detail"])
        w.writerows(role_conflicts)

    print("partner manifest rows:", len(manifest_rows))
    print("summary:", json.dumps(summary))
    print("role conflicts (incl. recorded swaps):", len(role_conflicts))
    # show non-swap conflicts only
    hard = [c for c in role_conflicts if "swapped" not in c[3]]
    print("hard conflicts:", len(hard))
    for c in hard[:20]:
        print("   ", c)


if __name__ == "__main__":
    main()
