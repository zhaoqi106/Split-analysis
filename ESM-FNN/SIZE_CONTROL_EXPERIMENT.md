# ESM-FNN training-size control

The outer RCSV evaluation set and the inner validation set stay fixed. Only the number of inner-training records that receive gradients changes. This is a fit-size curve on a fixed RCSV eval set, not a claim that the train and validation composition matches ICSV.

The trainer is `adapt_outer.py`: frozen ESM-2, `SimpleFFN` (1280 to 512 to 128 to 1), Adam at 0.001, full-batch MSE, at most 50 epochs, gradient clip 1.0, CUDA FP16. The checkpoint is the best inner-validation PCC and is applied directly to the outer eval set.

## Scripts

- `run_size_control.py`: one task.
- `launch_size_control.py`: plan, shards, and retries.
- `summarize_size_control.py`: per-fold, per-seed, and full-versus-reduced tables.

Reduced-scale training adds 75 tasks. S1131 uses one size per seed (438, 446, 451, 454, 457) on all five RCSV folds. M1101 uses two sizes per seed (365/744, 371/744, 372/723, 374/731, 370/722). ICSV sizes larger than the available RCSV pool are recorded in the plan and are not trained. Fifty full-scale points reuse the existing full runs. The plan has 125 rows.

## Run

```bash
python /data/ESM-FNN/launch_size_control.py \
  --project-root /data \
  --split-root /data/data_split/splits \
  --esm-model /data/ESM-FNN/esm2_t33_650M_UR50D.pt \
  --output-root /data/esmfnn_size_control \
  --run

python /data/ESM-FNN/summarize_size_control.py \
  --project-root /data \
  --output-root /data/esmfnn_size_control
```

Five folds and five seeds are not 25 independent experiments.
