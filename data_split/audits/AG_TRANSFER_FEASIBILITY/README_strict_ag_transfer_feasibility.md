# M1101 strict Ag-transfer

The full 1101 records cannot form a strict Ag-transfer five-fold split. The limit is which antibody entities sit in which antigen groups. It does not depend on the seed or the inner-validation fraction.

The confirmed map has 29 complete antibody entities. 26 of them appear in only one antigen group (970 records). When that group is the outer eval set, the antibody is absent from `inner_train`, so those 970 records are never eligible.

The structural upper bound is 131 records, 11.90% of 1101. That count is the records whose complete antibody entity appears in at least two antigen groups. The deterministic reassignment reaches 131, so 131 is the maximum. A single-group antibody contributes 0 under every intact-group assignment.

Eligibility below counts an antibody only when it is already in that fold's `inner_train`. Presence in `inner_val` does not count. Every current seed-fold cell is 0.

| seed | fold | outer eval | eligible | coverage | ineligible | unresolved | eligible antigen groups | eligible antibodies |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 246 | 0 | 0.0000 | 246 | 0 | 0 | 0 |
| 0 | 1 | 217 | 0 | 0.0000 | 217 | 0 | 0 | 0 |
| 0 | 2 | 210 | 0 | 0.0000 | 210 | 0 | 0 | 0 |
| 0 | 3 | 212 | 0 | 0.0000 | 212 | 0 | 0 | 0 |
| 0 | 4 | 216 | 0 | 0.0000 | 216 | 0 | 0 | 0 |
| 42 | 0 | 246 | 0 | 0.0000 | 246 | 0 | 0 | 0 |
| 42 | 1 | 217 | 0 | 0.0000 | 217 | 0 | 0 | 0 |
| 42 | 2 | 210 | 0 | 0.0000 | 210 | 0 | 0 | 0 |
| 42 | 3 | 212 | 0 | 0.0000 | 212 | 0 | 0 | 0 |
| 42 | 4 | 216 | 0 | 0.0000 | 216 | 0 | 0 | 0 |
| 1234 | 0 | 246 | 0 | 0.0000 | 246 | 0 | 0 | 0 |
| 1234 | 1 | 217 | 0 | 0.0000 | 217 | 0 | 0 | 0 |
| 1234 | 2 | 210 | 0 | 0.0000 | 210 | 0 | 0 | 0 |
| 1234 | 3 | 212 | 0 | 0.0000 | 212 | 0 | 0 | 0 |
| 1234 | 4 | 216 | 0 | 0.0000 | 216 | 0 | 0 | 0 |
| 2026 | 0 | 246 | 0 | 0.0000 | 246 | 0 | 0 | 0 |
| 2026 | 1 | 217 | 0 | 0.0000 | 217 | 0 | 0 | 0 |
| 2026 | 2 | 210 | 0 | 0.0000 | 210 | 0 | 0 | 0 |
| 2026 | 3 | 212 | 0 | 0.0000 | 212 | 0 | 0 | 0 |
| 2026 | 4 | 216 | 0 | 0.0000 | 216 | 0 | 0 | 0 |
| 1024 | 0 | 246 | 0 | 0.0000 | 246 | 0 | 0 | 0 |
| 1024 | 1 | 217 | 0 | 0.0000 | 217 | 0 | 0 | 0 |
| 1024 | 2 | 210 | 0 | 0.0000 | 210 | 0 | 0 | 0 |
| 1024 | 3 | 212 | 0 | 0.0000 | 212 | 0 | 0 | 0 |
| 1024 | 4 | 216 | 0 | 0.0000 | 216 | 0 | 0 | 0 |

One full rotation of each seed is 0/1101. Outer-eval sizes are the same for every seed: 246, 217, 210, 212, 216. The only two antibody-sharing pairs land in the same outer fold. `AG_BONTA1`, `AG_BONTA2`, `AG_MTSP1`, and `AG_EPITHIN` are all in outer fold 4 for seeds 0, 42, 1234, 2026, and 1024. Their sizes are 47, 41, 62, and 8. The greedy rule sorts by size and shuffles only equal-size groups, so these seeds do not separate the pairs. When fold 4 is evaluated, the paired groups are in the eval set, not in `inner_train`.

Reassignment covers 131/1101:

- Two complete heavy-plus-light entities shared by BoNT/A1 and BoNT/A2, 88 records. `ABSEQ_24b4b264b9d1` links 2NYY and HM_2NYY (53). `ABSEQ_2befe52357b7` links 2NZ9 and HM_2NZ9 (35).
- One complete heavy-plus-light entity shared by MT-SP1 and Epithin, `ABSEQ_9a8f59575a34`, 43 records: 35 in MT-SP1 and 8 in Epithin. Another 27 MT-SP1 records belong to antibodies that appear only in that group.

The exhibited assignment puts each pair in different outer folds and keeps the paired groups in `inner_train`. Inner-validation fractions are 0.1998, 0.1995, 0.1995, 0.2002, and 0.2000. The assignment is in the JSON. It is not written as fold files.

The current AGCV stays the antigen holdout on disk. Its strict subset is empty. The 131-record subset, if written later, would live in its own directory and would not stand for the full M1101 antigen holdout.

## Antibody identity

The identity is the confirmed `antibody_entity_id`. Each raw record's R1-R6 and L1-L3 chains are rebuilt and compared with `antibody_sequence_sha256` and `antigen_sequence_sha256`. The entity id is `ABSEQ_` plus the first 12 hex characters of the SHA1 of the antibody-side tuple.

The five `antigen_mapping.csv` files are byte-identical. SHA256: `38afa297d4a04752c176c45bbba8e5acd6e2d3d373e212c1b93d39608f8f4fd5`. `M1101.csv` SHA256: `be8aad0f99ee9706b695875fb906d59d9d442b29b649b28e610d0ae19eb2ce3c`.

All 1101 antibody-side hashes fall on R1-R6 and the antigen side on L1-L3. 993 records have both a heavy and a light chain, and the identity is that ordered pair. 108 records have a single sequence in the heavy columns and an empty light chain; all 108 are `non_antibody_complex`. The remaining roles are 683 classic antibody-antigen records, 271 Fc-receptor complexes, and 39 antibody idiotopes. Fc complexes use the heavy and light columns as the Fc dimer pair. A heavy-only or light-only match does not grant eligibility.

Mapping conflicts: 0.

Auxiliary counts:

- exact heavy sequences that cross antigen groups: 3
- exact light sequences that cross antigen groups: 2
- same heavy chain, different complete antibody: 0
- same light chain, different complete antibody: 1
- a single chain crosses groups while the complete antibody does not: 0
- exact substring pairs (no similarity threshold): 1

## Antibodies that cross antigen groups

- `ABSEQ_24b4b264b9d1`: 53 records, antigen groups AG_BONTA1 and AG_BONTA2, paired heavy and light, PDB 2NYY and HM_2NYY.
- `ABSEQ_2befe52357b7`: 35 records, antigen groups AG_BONTA1 and AG_BONTA2, paired heavy and light, PDB 2NZ9 and HM_2NZ9.
- `ABSEQ_9a8f59575a34`: 43 records, antigen groups AG_EPITHIN and AG_MTSP1, paired heavy and light, PDB 3BN9 and HM_3BN9.

The two BoNT entities share a light chain and differ in the heavy chain, so they remain two antibodies. The VEGF and HER2/3BE1 light chains have a 216-versus-217 exact substring. They are not treated as one antibody.

## Current AGCV and the reassignment

Outer eval holds out whole antigen groups. A record is eligible only when the same `antibody_entity_id` is already in that fold's `inner_train`. All 25 seed-fold cells are 0. Counting `inner_val` as fitted data is still 0, because the paired groups are not in `outer_train`.

## What this directory contains

Record alignment for all 1101 rows, the five seed maps, sequence-tuple checks, antibody-antigen links, eligibility of the current five-by-five AGCV, the single-group upper bound, and one assignment that meets that bound.

Sequence clustering below exact identity, structure or interface similarity, model training, new fold CSVs, and label-based split search are not in this directory.

## Files

- `antibody_antigen_group_map.csv`: one complete antibody entity and one antigen group per row.
- `antibody_group_reuse_summary.csv`: one antibody entity per row.
- `strict_ag_transfer_feasibility.json`: the bound and the exhibited assignment.
- `strict_ag_transfer_eligibility_by_fold.csv`: eligibility of the current AGCV.
- `strict_ag_transfer_violations.csv`: records that no intact grouping can make eligible, plus multi-group antibodies that fail in the current folds.
- `compute_strict_ag_transfer_feasibility.py`: the script that writes this directory.
