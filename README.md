# Antibody–antigen ddG benchmark

Root directory: `/data`.

This tree holds the benchmark tables, cross-validation splits, exposure audits, ESM-FNN code, and FoldX helpers used for the antibody–antigen ddG study.

## Layout

```
/data
├── data_split/                         M1101 and S1131 tables, splits, and audits
│   ├── raw/                            M1101.csv, S1131.csv
│   ├── splits/                         five seeds × RCSV, PCSV, ICSV; M1101 also has AGCV
│   ├── scripts/                        split builders and split audits
│   ├── summary/                        split size summaries
│   ├── audits/                         ICSV clustering audit, strict Ag-transfer audit
│   └── audit_icsv_exposure_20261109/   cross-fold sequence and structure exposure
├── ESM-FNN/                            ESM-2 + FNN training, external eval, size-control scripts
├── structure_audit_20261105/           whole-complex US-align exposure
├── structure_audit_paired_20261105/    paired left/right structural exposure
└── foldx/                              FoldX recomputation and insertion-code checks
```

## Data

- `M1101` is the antibody–antigen set (1101 records). `S1131` is the general protein–protein set (1131 records). Master tables are `/data/data_split/raw/`.
- Seeds are `0`, `42`, `1024`, `1234`, and `2026`.
- `RCSV` is a random record split, `PCSV` holds out whole PDB complexes, and `ICSV` holds out MMseqs2 sequence-identity clusters (30% identity, 80% coverage). `AGCV` holds out antigen entities and exists only for M1101.
- RCSV, PCSV, and ICSV folds are `train_fold_{k}.csv` and `val_fold_{k}.csv`. AGCV folds are `fold_{k}_inner_train.csv`, `fold_{k}_inner_val.csv`, `fold_{k}_outer_train.csv`, and `fold_{k}_outer_eval.csv`.
- `record_uid` (`dataset:row`) is an audit id, not a feature. `fold_{k}_outer_train.csv` is the union of the inner train and inner validation files.
- Wild-type PDBs are `/data/data_split/splits/{M1101,S1131}/wt/`.
- ICSV group maps are `/data/data_split/splits/{dataset}/icsv_group_map.csv`.
- Cluster-size imbalance in ICSV comes from the connected components, not from the seed. Changing the clustering thresholds changes the folds.
- The full 1101-record M1101 set cannot support a strict antibody-transfer split. The structural upper bound is 131 records (11.90%). See `data_split/audits/AG_TRANSFER_FEASIBILITY/`.
- Whole-complex TM-score exposure is in `structure_audit_20261105/`. Paired left/right exposure is in `structure_audit_paired_20261105/`.
- For Figure 3 and Figure 4, SAMAffinity and StaB-ddG numbers come from the `adapt_code_v1` runs.

## Code

Paths below are under `/data`.

1. Splits: `data_split/scripts/three_splits.py` and `entity_group_cv.py`. Check with `test_entity_group_cv.py`.
2. ICSV clustering audit: `data_split/scripts/audit_icsv_clustering.py`. MMseqs2 is `/data/tools/mmseqs/bin/mmseqs`.
3. Cross-fold exposure: `data_split/audit_icsv_exposure_20261109/audit_crossfold_exposure.py`. US-align is `/data/tools/USalign`.
4. Structure audits: `structure_audit_20261105/scripts/` and `structure_audit_paired_20261105/scripts/`.
5. ESM-FNN outer folds: `ESM-FNN/adapt_outer.py` with `protocol.py`. Weights: `/data/ESM-FNN/esm2_t33_650M_UR50D.pt`.
6. Training-size control: `ESM-FNN/launch_size_control.py`, `run_size_control.py`, `summarize_size_control.py`. See `ESM-FNN/SIZE_CONTROL_EXPERIMENT.md`.
7. FoldX: `foldx/foldx_recompute_all.py`. Binary: `/data/tools/foldx`. Mutation lists: `/data/external/M1101/pdb/txt` and `/data/external/S1131/pdb/txt`.

SKEMPI names used only as S1131 evidence: `/data/external/SKEMPI/skempi_v2.csv`. External evaluation tables and MAGMA inputs, when present, live under `/data/external/`.

Large run outputs, model weights, and mutant structures are not in this tree.

## Check

```bash
sha256sum -c /data/SHA256SUMS.txt
```

Per-file notes are in `/data/MANIFEST.csv`.
