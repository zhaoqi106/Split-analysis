# Paired structural exposure

Paired left/right structural exposure for S1131 and M1101 (manuscript equations 4–8). This directory is `/data/structure_audit_paired_20261105`. It is separate from the whole-complex audit in `/data/structure_audit_20261105`.

For a validation complex `v`, a training complex `t`, and a side `p`:

- `S_p(v,t)` is the minimum of the two directional TM-scores, each normalized by that side's own length.
- `C_p(v,t)` is the minimum of the two directional coverages.
- `S_pair` and `C_pair` are the minima across the two sides.
- A complex is exposed at `τ = 0.70` and `c = 0.80` only when one training complex meets both thresholds on both sides. The best left and right matches are not taken from different training complexes.
- `UNRESOLVED` is not counted as unexposed. The primary rate is record-weighted and includes identical complexes.

S1131 left chains carry `R1–R6` and right chains carry `L1–L3`. M1101 left is the antibody side and right is the antigen side, taken from `/data/data_split/splits/M1101/0/AGCV/antigen_mapping.csv`. For `1KTZ` and `HM_1KTZ` the metadata roles are swapped relative to the CSV columns; the metadata is the one used here. `HM_` structures are separate objects.

## Run

```bash
python3 /data/structure_audit_paired_20261105/scripts/build_partners.py
python3 /data/structure_audit_paired_20261105/scripts/run_paired_side.py --dataset S1131 --side left --workers 64
python3 /data/structure_audit_paired_20261105/scripts/run_paired_side.py --dataset S1131 --side right --workers 64
python3 /data/structure_audit_paired_20261105/scripts/run_paired_side.py --dataset M1101 --side left --workers 64
python3 /data/structure_audit_paired_20261105/scripts/run_paired_side.py --dataset M1101 --side right --workers 64
python3 /data/structure_audit_paired_20261105/scripts/paired_exposure.py
python3 /data/structure_audit_paired_20261105/scripts/qc_diagnostics.py
```

US-align is `/data/tools/USalign`.

## Outputs

- `partner_chain_manifest.csv`: left and right chains for each complex.
- `paired/`: per-side and combined pair tables.
- `paired_complex_similarity_matrix.csv` and the coverage and per-side matrices.
- `per_fold_paired_exposure.csv`, `per_seed_paired_exposure.csv`, `per_eval_complex_paired.csv`.
- `supplementary_table_S1_paired_exposure.csv`.
- `failed_or_undetermined.csv`: empty for this run.
- `PAIRED_AUDIT_REPORT_zh.md`: longer write-up.
