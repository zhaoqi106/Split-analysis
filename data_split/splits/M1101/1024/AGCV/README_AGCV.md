# M1101 AGCV

This directory is antigen-grouped cross-validation. Mapping gate: `CONFIRMED_PENDING_AUDIT`.

All mappings used for grouping are confirmed. Independent audit still required.

## Files the model may read

Use the three files below separately. Do not concatenate them into one training set.

- `fold_{k}_inner_train.csv`: fit the model.
- `fold_{k}_inner_val.csv`: early stopping and checkpoint selection.
- `fold_{k}_outer_eval.csv`: final evaluation of that outer fold. Do not use it to stop, select, or refit.
- `fold_{k}_outer_train.csv`: `inner_train` plus `inner_val` for the same fold. It is a boundary check, not a third training protocol.

`record_uid` is an audit id (`dataset:row`). Ignore it as a feature. Labels are unfiltered, and ddG units and signs are unchanged.

## Grouping

An entity group from the mapping file is atomic. Every record in a group stays in the same outer fold and the same inner subset.

Assignment uses group id, group size, and a fixed seed. Group ids are sorted, ties are shuffled with that seed, then groups are placed largest first. Labels and model scores are not used. The inner seed is `outer_seed * 1000 + fold_id`. The inner-validation target is 20% of outer-train records; group size can move the realized fraction.

Outer seeds are 0, 42, 1234, 2026, and 1024. Seeds may reassign groups. The five folds, five seeds, and 20% inner split are project settings, not a literature requirement.

On M1101, 1N8Z and 3BE1 share the same HER2 sequence on numbered residues 1-607 and are one antigen entity. 1KTZ and HM_1KTZ share the TGF-beta type II receptor but have different ligands, so they are two groups.

## Source of the idea

The holdout idea follows Li et al., iScience 2026;29(8):116522, and https://github.com/amazon-science/ab-binding-eval. This directory uses only the ideas of holding out a whole antigen system, rotating systems, and separating model selection from the final test. It does not use that paper's model, loss, or split ratios. `PGCV_LEFT` and `PGCV_RIGHT` are partner holdouts, not antigen holdouts, and the two sides are not merged.

## Mapping states

`confirmed`: chain, wild-type sequence, and source note agree. `candidate`: the group is fixed by sequence, but the name is not fully checked. `unresolved`: the entity is unknown; those records are not forced into singleton groups, and formal fold files are not written for them.

`audit_antigen_group_splits.py` rereads the files on disk. Cite a split as formal only when that audit returns `PASS`. Sequence or structure similarity that was not run is uncomputed, not zero. Leaving out an entity does not mean there is no homolog, no interface similarity, or that pretraining never saw the sequence.
