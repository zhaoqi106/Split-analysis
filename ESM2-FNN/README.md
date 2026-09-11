# ESM2-FNN baseline

This directory contains the study-specific ESM2-FNN baseline used for the
S1131 and M1101 case study. It combines mutation-site representations from
ESM-2 (`esm2_t33_650M_UR50D`) with a feed-forward neural network regressor.

## Required files

- `esm2line.py`: feature extraction, five-fold FFN training, validation, model
  checkpointing, and prediction export.
- `run_m1101_s1131_bf16.sh`: optional Linux/WSL launcher for the six
  dataset/split experiments.
- `requirements.txt`: Python dependencies.

The predefined folds are read from the parent repository:

```text
../data/{S1131,M1101}/{RCSV,PCSV,ICSV}/
```

## Model input

Each fold CSV must contain:

- `ID` and `ddG`;
- wild-type chains `R1`–`R6` and `L1`–`L3`; and
- mutant chains `R1m`–`R6m` and `L1m`–`L3m`.

The `R` and `L` prefixes are legacy source-data column names for the two
interaction sides. They do not imply that every input is an
antibody–antigen complex.

Non-empty chains are concatenated in that order. ESM-2 representations are
calculated for the wild-type and mutant sequences. Representation differences
at detected mutation positions are summed into a 1,280-dimensional feature
vector, which is used by the FFN regressor.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Reference environment

The reported ESM2-FNN experiments were run with:

- Python `3.10.13` (conda-forge build, GCC `12.3.0`)
- NumPy `1.26.4`
- PyTorch `2.6.0+cu124`
- PyTorch CUDA build `12.4`
- cuDNN `9.1.0` (`torch.backends.cudnn.version() = 90100`)
- GPU: NVIDIA H20
- CUDA available in PyTorch: `True`

[Download the official `esm2_t33_650M_UR50D` checkpoint](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt)
separately. Do not commit the checkpoint to this repository.

## Run one experiment

From this directory:

```bash
python esm2line.py \
  --dataset S1131 \
  --split RCSV \
  --esm-model /path/to/esm2_t33_650M_UR50D.pt
```

The default input root is `../data`. The default output root is:

```text
../output/SHUJU_clean/ESM-FNN/
```

Prediction files follow the format required by the parent analysis scripts:

```text
../output/SHUJU_clean/ESM-FNN/{dataset}/{split}/
  {dataset}_{split}_fold0_pred.csv
  ...
  {dataset}_{split}_fold4_pred.csv
```

Each prediction file contains:

```text
ID,ddG_true,ddG_pred
```

Use `--split-root` and `--result-root` to override the default locations.
The checkpoint can also be supplied through the `ESM_MODEL` environment
variable.

## Run all six experiments

The launcher requires Linux or WSL, Bash, `taskset`, `nohup`, six available
GPUs, and BF16 support.

```bash
export ESM_MODEL=/path/to/esm2_t33_650M_UR50D.pt
export GPU_IDS=0,1,2,3,4,5
bash run_m1101_s1131_bf16.sh
```

`GPU_IDS` must contain six comma-separated device numbers. Input and output
roots can be overridden with `SPLIT_ROOT` and `RESULT_ROOT`.

## Generated files

The following files are generated during training and should normally not be
committed:

```text
logs_bf16/
*.log
*.pid
fold_*_best_model.pt
fold_metrics_summary.csv
predicate_all_folds.csv
summary.txt
```

The standardized per-fold prediction CSV files used by the manuscript analysis
should be retained and released.

