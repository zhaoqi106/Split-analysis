# Whole-complex structural exposure

US-align TM-score exposure of wild-type complexes for S1131 and M1101. Files in this directory are under `/data/structure_audit_20261105`. Split inputs are `/data/data_split`.

## Read first

- `PRECHECK_REPORT_zh.md`: structure counts, chain mapping, split integrity, and the US-align check.
- `STRUCTURE_AUDIT_RESULTS_zh.md`: method, supplementary table, and file index.

## Contents

- `supplementary_table_S1_structure_exposure.csv`: exposure by dataset and split.
- `manifests/`: 144 wild-type complexes and the chain map for 2232 records.
- `precheck/`: fold-by-seed integrity. The anomaly list is empty.
- `pairwise/`: all-vs-all TM and coverage (6216 S1131 pairs, 496 M1101 pairs).
- `exposure/`: per-fold and per-complex exposure, row-normalized TM matrices, and the HM comparison.
- `scripts/`: `precheck.py`, `run_pairwise.py`, `fold_exposure.py`, `aux_tables.py`.
- `logs/`: run logs.

## Result in short

Random splits mostly share complexes with training. PDB holdout removes identical names but still leaves many TM ≥ 0.7 neighbors. Sequence-cluster holdout removes nearly all of that global exposure. Antigen holdout does not. Whole-complex TM includes the antibody scaffold, so high M1101 scores partly reflect the immunoglobulin fold. Interface similarity is not computed. Failed alignments are not treated as zero exposure.
