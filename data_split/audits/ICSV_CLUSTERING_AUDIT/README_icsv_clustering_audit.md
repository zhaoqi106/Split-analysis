# ICSV clustering audit

This directory is the MMseqs2 clustering audit for the ICSV splits. Cluster commands and intermediate files are kept here. The split files themselves are under `/data/data_split/splits`.

## Result

- MMseqs2 build `503743960fc57733a36951f5fa803cb3389af7e2` (binary SHA256 `f393cd8d4c433261...`) at the reference setting 30%/80%.
- The record-level connected components match `icsv_group_map.csv`, and the five outer ICSV folds match row for row. The maps come from this setting.
- S1131: identity 0.20-0.30 and coverage 0.50-0.90 give the same partition. The smallest largest-group fraction in the identity scan is 0.5119. Exact match has 0.1680 (112 groups).
- M1101: identity 0.20-0.50 and coverage 0.50-0.70 give the same partition. The smallest largest-group fraction in the identity scan is 0.2461. Exact match has 0.2234 (32 groups).
- The five seeds give the same fold sizes. ICSV imbalance is the size of the connected components.
- A different threshold changes the outer folds.

## Match to the current splits

- S1131: MMseqs2 503743960fc57733a36951f5fa803cb3389af7e2, 36 connected components at 30%/80%. partition_equal=True, group_id match=True, five outer folds match row for row=True.
- M1101: MMseqs2 503743960fc57733a36951f5fa803cb3389af7e2, 8 connected components at 30%/80%. partition_equal=True, group_id match=True, five outer folds match row for row=True.

Cluster commands and TSVs are in each dataset's `clusters/`. Input FASTA files are in `sequences/`. `partition_equal` true with different group ids means the same clusters and different representative ids.

## Fold balance at 30%/80%

Fold sizes are the same for every seed. One seed is shown per dataset.

| dataset | seed | fold | val_records | val_fraction | val_groups | max_val_group_records | group_overlap_train_val |
|---|---|---|---|---|---|---|---|
| S1131 | 0 | 0 | 621 | 0.5491 | 1 | 621 | 0 |
| S1131 | 0 | 1 | 128 | 0.1132 | 9 | 60 | 0 |
| S1131 | 0 | 2 | 128 | 0.1132 | 9 | 58 | 0 |
| S1131 | 0 | 3 | 128 | 0.1132 | 9 | 58 | 0 |
| S1131 | 0 | 4 | 126 | 0.1114 | 8 | 42 | 0 |
| M1101 | 0 | 0 | 683 | 0.6203 | 1 | 683 | 0 |
| M1101 | 0 | 1 | 271 | 0.2461 | 1 | 271 | 0 |
| M1101 | 0 | 2 | 51 | 0.0463 | 2 | 44 | 0 |
| M1101 | 0 | 3 | 48 | 0.0436 | 2 | 39 | 0 |
| M1101 | 0 | 4 | 48 | 0.0436 | 2 | 31 | 0 |

## Threshold sensitivity

Each side is clustered with MMseqs2, then the two sides are merged with union-find. `partition_equal_to_reference` means the record-level partition matches the current 30%/80% split. A changed threshold is a different outer split.

| dataset | scan | min_seq_id | coverage | n_groups | largest_group_fraction | max_val_fraction | min_val_fraction | val_records_spread | seeds_give_identical_fold_sizes | partition_equal_to_reference |
|---|---|---|---|---|---|---|---|---|---|---|
| S1131 | exact | 1.0000 | 1.0000 | 112 | 0.1680 | 0.2007 | 0.1998 | 1 | True | False |
| S1131 | identity | 0.2000 | 0.8000 | 36 | 0.5491 | 0.5491 | 0.1114 | 495 | True | True |
| S1131 | identity | 0.2500 | 0.8000 | 36 | 0.5491 | 0.5491 | 0.1114 | 495 | True | True |
| S1131 | identity | 0.3000 | 0.8000 | 36 | 0.5491 | 0.5491 | 0.1114 | 495 | True | True |
| S1131 | identity | 0.3500 | 0.8000 | 37 | 0.5491 | 0.5491 | 0.1123 | 494 | True | False |
| S1131 | identity | 0.4000 | 0.8000 | 37 | 0.5491 | 0.5491 | 0.1123 | 494 | True | False |
| S1131 | identity | 0.5000 | 0.8000 | 38 | 0.5119 | 0.5119 | 0.1211 | 442 | True | False |
| S1131 | identity | 0.6000 | 0.8000 | 38 | 0.5119 | 0.5119 | 0.1211 | 442 | True | False |
| S1131 | identity | 0.7000 | 0.8000 | 38 | 0.5119 | 0.5119 | 0.1211 | 442 | True | False |
| S1131 | identity | 0.8000 | 0.8000 | 38 | 0.5119 | 0.5119 | 0.1211 | 442 | True | False |
| S1131 | identity | 0.9000 | 0.8000 | 38 | 0.5119 | 0.5119 | 0.1211 | 442 | True | False |
| S1131 | coverage | 0.3000 | 0.5000 | 36 | 0.5491 | 0.5491 | 0.1114 | 495 | True | True |
| S1131 | coverage | 0.3000 | 0.6000 | 36 | 0.5491 | 0.5491 | 0.1114 | 495 | True | True |
| S1131 | coverage | 0.3000 | 0.7000 | 36 | 0.5491 | 0.5491 | 0.1114 | 495 | True | True |
| S1131 | coverage | 0.3000 | 0.9000 | 36 | 0.5491 | 0.5491 | 0.1114 | 495 | True | True |
| S1131 | coverage | 0.3000 | 1.0000 | 51 | 0.3378 | 0.3378 | 0.1653 | 195 | True | False |
| M1101 | exact | 1.0000 | 1.0000 | 32 | 0.2234 | 0.2234 | 0.1926 | 34 | True | False |
| M1101 | identity | 0.2000 | 0.8000 | 8 | 0.6203 | 0.6203 | 0.0436 | 635 | True | True |
| M1101 | identity | 0.2500 | 0.8000 | 8 | 0.6203 | 0.6203 | 0.0436 | 635 | True | True |
| M1101 | identity | 0.3000 | 0.8000 | 8 | 0.6203 | 0.6203 | 0.0436 | 635 | True | True |
| M1101 | identity | 0.3500 | 0.8000 | 8 | 0.6203 | 0.6203 | 0.0436 | 635 | True | True |
| M1101 | identity | 0.4000 | 0.8000 | 8 | 0.6203 | 0.6203 | 0.0436 | 635 | True | True |
| M1101 | identity | 0.5000 | 0.8000 | 8 | 0.6203 | 0.6203 | 0.0436 | 635 | True | True |
| M1101 | identity | 0.6000 | 0.8000 | 9 | 0.6104 | 0.6104 | 0.0454 | 622 | True | False |
| M1101 | identity | 0.7000 | 0.8000 | 11 | 0.4078 | 0.4078 | 0.0781 | 363 | True | False |
| M1101 | identity | 0.8000 | 0.8000 | 15 | 0.3588 | 0.3588 | 0.1217 | 261 | True | False |
| M1101 | identity | 0.9000 | 0.8000 | 19 | 0.2461 | 0.2461 | 0.1853 | 67 | True | False |
| M1101 | coverage | 0.3000 | 0.5000 | 8 | 0.6203 | 0.6203 | 0.0436 | 635 | True | True |
| M1101 | coverage | 0.3000 | 0.6000 | 8 | 0.6203 | 0.6203 | 0.0436 | 635 | True | True |
| M1101 | coverage | 0.3000 | 0.7000 | 8 | 0.6203 | 0.6203 | 0.0436 | 635 | True | True |
| M1101 | coverage | 0.3000 | 0.9000 | 9 | 0.6104 | 0.6104 | 0.0454 | 622 | True | False |
| M1101 | coverage | 0.3000 | 1.0000 | 21 | 0.2234 | 0.2234 | 0.1926 | 34 | True | False |

## Scope

- More groups and a smaller largest group make the outer folds more even. A higher threshold is a weaker redundancy filter.
- 30% identity with 80% coverage is the stricter filter. A few large components make the folds uneven.
- This audit is sequence identity only. TM-score, interface similarity, and structure clustering are not computed here.
