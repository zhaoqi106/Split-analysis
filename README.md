# Data Splits Define the Prediction Task

This repository contains the data partitions, structure-preparation utilities,
molecular-exposure audits, and statistical analysis code used in the case study
for:

> **Data Splits Define the Prediction Task: A Perspective on Protein–Protein
> Binding ΔΔG Model Evaluation**

The study does not propose a new model architecture as its primary
contribution. It includes a study-specific ESM2-FNN comparison baseline and
evaluates how record-level random cross-validation (RCSV), PDB-grouped
cross-validation (PCSV), and identity-controlled sequence-group
cross-validation (ICSV) change the prediction task, molecular exposure,
performance estimates, prediction contraction, and model rankings.

The case study spans a general protein–protein interaction benchmark and an
antibody-focused benchmark. It also compares methods with different intended
scopes, including antibody-focused predictors, broadly formulated
protein–protein binding predictors, a PPI-oriented method with
antibody-informed representations, and a study-specific comparison baseline.

## Repository contents

```text
Split-analysis/
├── data/
│   ├── S1131/
│   │   ├── S1131.csv
│   │   ├── RCSV/
│   │   ├── PCSV/
│   │   ├── ICSV/
│   │   └── wt/
│   └── M1101/
│       ├── M1101.csv
│       ├── RCSV/
│       ├── PCSV/
│       ├── ICSV/
│       └── wt/
├── ESM2-FNN/
│   ├── esm2line.py
│   ├── run_m1101_s1131_bf16.sh
│   ├── requirements.txt
│   └── README.md
├── three_splits.py
├── foldxprocess.py
├── foldxpdb.py
├── sequence_separation.py
├── structure_exposure.py
├── calculate_metrics.py
├── analyze_performance_contraction.py
└── paired_analysis.py
```

Each split directory contains five predefined folds:

```text
train_fold_0.csv ... train_fold_4.csv
val_fold_0.csv   ... val_fold_4.csv
```

The released split CSV files are the authoritative partitions used in the
study. Users who want to evaluate a model on the same tasks should use these
files directly.

## Data and split definitions

### S1131

S1131 is a general protein–protein interaction benchmark derived by Xiong et
al. from the original SKEMPI database. The canonical benchmark reports 1,131
nonredundant interface single-point mutations across 114 protein complexes.
The processed `S1131.csv` used here contains all 1,131 records, 1,131 unique
`ID` values, and 112 unique complete PDB identifiers. The original literature
complex count and the number of complete PDB identifiers in the processed file
are therefore reported separately.

### M1101

M1101 corresponds to AB-Bind, which focuses on antibody interactions with large
globular antigens and additionally includes Fc–receptor, nanobody–antigen, and
antibody–antigen-like complexes. It contains 1,101 experimental mutation
records from 32 complexes. The original records comprise 645 single-point and
456 multiple-point variants. The processed file contains 1,096 unique
identifiers: five identifiers occur twice, and their experimental ΔΔG values
are not necessarily identical. The same-target paired analysis in
`paired_analysis.py` aggregates repeated identifiers explicitly and therefore
uses 1,096 targets, comprising 642 single-point and 454 multiple-point targets.
Fold-level and pooled OOF performance analyses retain all 1,101 records.

The sequence columns retain the source-data names `R1`–`R6` and `L1`–`L3`
with corresponding mutant columns ending in `m`. Throughout the split and
audit code, these are treated generically as the left and right interaction
sides. The column prefixes do not imply that every record is an
antibody–antigen complex.

The thermodynamic convention is `ΔΔG = ΔG_mutant - ΔG_wild-type`, reported in
`kcal mol⁻¹`. Positive values indicate weaker binding after mutation, whereas
negative values indicate stronger binding.

### RCSV

RCSV assigns individual mutation records to folds. It evaluates prediction of
held-out records while allowing molecular backgrounds, PDB entries, and related
sequences to occur in both training and validation data.

### PCSV

PCSV treats each complete PDB identifier as an indivisible group. Records with
the same complete PDB identifier are assigned to the same validation fold.
PCSV removes direct PDB sharing but does not automatically remove homologous
sequences or structurally similar complexes across different PDB entries.

### ICSV

ICSV clusters the wild-type sequences on the two interaction sides separately.
Complexes sharing either a left-side or right-side MMseqs2 cluster are merged
recursively into connected construction groups. Complete groups are assigned
to folds with seeded greedy load balancing.

The primary construction parameters are:

- minimum sequence identity: `0.30`
- alignment coverage: `0.80`
- MMseqs2 coverage mode: `0`
- MMseqs2 cluster mode: `1`
- folds: `5`
- split seed: `42`
- multi-chain linker: ten glycine residues (`GGGGGGGGGG`)

The final ICSV files must still be audited directly. A split name or clustering
procedure alone does not prove that all qualifying cross-fold relationships
have been removed.

## Software versions

The analyses reported in the study used:

- **MMseqs2:** `17b688d21dda57fc5f5b7286ecba7ec003d4717f`
- **US-align:** `20260527`
- **FoldX:** `5.1`, academic licensed version

MMseqs2 reports the build revision shown above rather than a short semantic
version number.

FoldX is licensed software and is not redistributed in this repository. Users
must obtain an appropriate FoldX licence and executable separately.

The top-level split, audit, structure-preparation, and statistical-analysis
scripts require Python 3.10 or later and use the following packages:

```text
numpy
pandas
biopython
```

ESM2-FNN has additional dependencies listed in
`ESM2-FNN/requirements.txt`.

External commands must be available through `PATH`, or their executable paths
must be supplied with the corresponding command-line options:

```bash
mmseqs version
USalign -v
foldx --help
```

## Compared models and source repositories

The case study compares nine models. Eight models use code released by their
original authors or the repository identified by the corresponding
publication:

- [abCAN](https://github.com/ChenGong57/abCAN)
- [AttABseq](https://github.com/ruofanjin/AttABseq)
- [DDGPred / binding-ddg-predictor](https://github.com/HeliXonProtein/binding-ddg-predictor)
- [DSSA-PPI](https://github.com/Agitw/DSSA-PPI)
- [GeoPPI](https://github.com/Liuxg16/GeoPPI)
- [RDE-PPI](https://github.com/luost26/RDE-PPI)
- [SAMAffinity](https://github.com/iobio-zjut/SAMAffinity)
- [StaB-ddG](https://github.com/LDeng0205/StaB-ddG)

The ninth model, ESM2-FNN, is the study-specific comparison baseline included
in `ESM2-FNN/`. The pretrained ESM-2 backbone originates from the
[facebookresearch/esm](https://github.com/facebookresearch/esm) project. The
links above are provided for attribution and implementation reference. Each
external repository has its own environment, pretrained-weight, and licence
requirements.

abCAN and AttABseq were developed explicitly for antibody-focused prediction.
DDGPred, DSSA-PPI, GeoPPI, RDE-PPI, and StaB-ddG are more broadly formulated
for protein–protein binding prediction. SAMAffinity is PPI-oriented but uses
antibody-informed representations. ESM2-FNN has no antibody-specific
architectural constraint, but its empirical scope is determined by the
training data used in this study. Running an antibody-focused method on S1131
is therefore an empirical cross-domain comparison and does not make S1131 an
antibody-specific dataset.

## Running ESM2-FNN

ESM2-FNN uses the `esm2_t33_650M_UR50D` backbone to calculate differences
between wild-type and mutant representations at mutation positions. Their sum
is passed to a feed-forward neural-network regressor.

Install its Python dependencies:

```bash
cd ESM2-FNN
python -m pip install -r requirements.txt
```

[Download the official `esm2_t33_650M_UR50D` checkpoint](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt)
separately. The checkpoint is not included in this repository.

Run one dataset/split experiment:

```bash
python esm2line.py \
  --dataset S1131 \
  --split RCSV \
  --esm-model /path/to/esm2_t33_650M_UR50D.pt
```

Valid datasets are `S1131` and `M1101`; valid split names are `RCSV`, `PCSV`,
and `ICSV`. The default input root is `../data`.

To run all six combinations concurrently under Linux or WSL:

```bash
export ESM_MODEL=/path/to/esm2_t33_650M_UR50D.pt
export GPU_IDS=0,1,2,3,4,5
bash run_m1101_s1131_bf16.sh
```

The parallel launcher requires six BF16-capable GPUs, Bash, `taskset`, and
`nohup`. For systems with fewer GPUs, run `esm2line.py` separately for each
dataset/split combination.

Standardized predictions are written to:

```text
../output/SHUJU_clean/ESM-FNN/{dataset}/{split}/
  {dataset}_{split}_fold0_pred.csv
  ...
  {dataset}_{split}_fold4_pred.csv
```

Each prediction file contains `ID`, `ddG_true`, and `ddG_pred`, making it
directly compatible with `calculate_metrics.py` and
`analyze_performance_contraction.py`. Additional implementation details and
known reproducibility limitations are documented in
`ESM2-FNN/README.md`.

## Script descriptions

### `three_splits.py`

Constructs five-fold RCSV, PCSV, and ICSV partitions for S1131 and M1101.

Its main operations are:

1. normalize PDB identifiers, including repair of Excel scientific-notation
   corruption such as `1.00E+96`;
2. generate record-level RCSV folds;
3. generate complete-PDB-grouped PCSV folds;
4. cluster wild-type sequences with MMseqs2;
5. merge complexes connected through either interaction-side cluster;
6. assign indivisible ICSV groups with seeded greedy load balancing; and
7. write train/validation CSV files and a split-composition summary.

Example:

```bash
python three_splits.py \
  --input-dir data \
  --output-root regenerated_data \
  --summary-path regenerated_output/split_summary.csv \
  --datasets S1131.csv M1101.csv \
  --mmseqs mmseqs \
  --seed 42 \
  --n-splits 5 \
  --min-seq-id 0.30 \
  --coverage 0.80 \
  --cov-mode 0 \
  --cluster-mode 1 \
  --threads 16
```

Split generation can depend on the input row order, especially for
record-level random splitting and tie breaking. The supplied split files,
rather than newly regenerated files, define the exact evaluation partitions
used in the study.

### `foldxprocess.py`

Prepares repaired wild-type structures with FoldX RepairPDB.

For each unique PDB identifier in the dataset CSV, the script:

1. reads the corresponding structure from `data/{dataset}/wt`;
2. removes hetero residues before FoldX processing;
3. runs FoldX `RepairPDB` in an isolated temporary directory;
4. saves the repaired structure to `data/{dataset}/repairpdb`; and
5. writes a processing log.

FoldX can be specified with `--foldx` or the `FOLDX_EXE` environment variable.

```bash
python foldxprocess.py \
  --dataset S1131 \
  --foldx /path/to/foldx \
  --workers 10
```

Run the command separately for `S1131` and `M1101`.

### `foldxpdb.py`

Generates mutant structures with FoldX BuildModel from repaired wild-type
structures.

The default locations are:

```text
data/{dataset}/repairpdb
data/{dataset}/foldx_txt/individual_list_{ID}.txt
data/{dataset}/mut
```

The mutation-list files must already exist and must follow FoldX
`individual_list` syntax. The script performs three FoldX runs per target,
selects the first sorted mutant PDB output, renames it to `{ID}.pdb`, and keeps
per-target logs.

```bash
python foldxpdb.py \
  --dataset S1131 \
  --foldx /path/to/foldx \
  --workers 16
```

Mutant structures generated by `foldxpdb.py` are not used by either
molecular-exposure audit in this repository.

### `sequence_separation.py`

Audits sequence separation in the final ICSV files without modifying those
files.

The script:

1. validates that the five validation folds reproduce the complete dataset;
2. checks that complete complexes are not divided within a train/validation
   pair;
3. reconstructs the MMseqs2 construction groups;
4. performs an independent all-against-all MMseqs2 search;
5. builds connected components from all relationships passing the identity and
   coverage criteria; and
6. identifies qualifying relationships that cross validation folds.

Example:

```bash
python sequence_separation.py \
  --data-dir data \
  --split-root data \
  --output-dir output/sequence_separation \
  --datasets S1131 M1101 \
  --mmseqs mmseqs \
  --threads 16
```

Outputs:

```text
output/sequence_separation/sequence_separation_summary.csv
output/sequence_separation/sequence_relationships.csv
output/sequence_separation/cross_fold_sequence_relationships.csv
```

### `structure_exposure.py`

Audits cross-fold molecular exposure in RCSV, PCSV, and ICSV.

It reports complete-PDB exposure for RCSV, PCSV, and ICSV. Exact wild-type
sequence exposure on the left and right interaction sides, and paired
structural exposure, are summarized for PCSV and ICSV.

Paired structural exposure requires the same training complex to satisfy the
criteria on both interaction sides. The primary thresholds are:

- symmetric TM-score at least `0.70`;
- bidirectional structural coverage at least `0.80`.

Wild-type PDB files are read recursively from `data/{dataset}/wt`. Structures
are aligned with US-align. Temporary chain files and pairwise alignment files
are deleted automatically after the run.

With the default paths, `structure_exposure.py` searches only below
`data/{dataset}/wt`. Repaired structures written to the sibling
`data/{dataset}/repairpdb` directory by `foldxprocess.py` are not discovered
automatically.

```bash
python structure_exposure.py \
  --split-root data \
  --structure-root data \
  --output-dir output/structure_exposure \
  --usalign USalign \
  --threads 8
```

Outputs:

```text
output/structure_exposure/structure_exposure_summary.csv
output/structure_exposure/paired_structure_exposure_detail.csv
```

### `calculate_metrics.py`

Calculates validation-fold RMSE, MAE, and Pearson correlation coefficient
(PCC), followed by the unweighted five-fold mean and sample standard deviation.

The script expects one prediction file for every
model/dataset/split/fold combination.

```bash
python calculate_metrics.py \
  --prediction-root output/SHUJU_clean \
  --output-dir output/fold_metrics
```

Outputs:

```text
output/fold_metrics/fold_metrics.csv
output/fold_metrics/fold_mean_std.csv
```

PCC is reported as unavailable when the observed or predicted vector has zero
variance.

### `analyze_performance_contraction.py`

Pools the five validation folds into out-of-fold (OOF) predictions and
calculates:

- pooled OOF RMSE, MAE, and PCC;
- observed and predicted ranges;
- predicted-to-observed range ratio;
- observed and predicted standard deviations;
- predicted-to-observed standard-deviation ratio; and
- prediction-on-observation attenuation slope.

The attenuation model is:

```text
ddG_pred = alpha + beta * ddG_true + error
```

Here, `beta` is a descriptive prediction-amplitude measure, not a conventional
calibration slope.

```bash
python analyze_performance_contraction.py \
  --prediction-root output/SHUJU_clean \
  --data-dir data \
  --output-dir output/performance_contraction
```

Outputs:

```text
output/performance_contraction/all_oof_predictions.csv
output/performance_contraction/pooled_oof_performance.csv
output/performance_contraction/prediction_contraction_metrics.csv
```

### `paired_analysis.py`

Compares same-target absolute error under PCSV or ICSV against RCSV:

```text
delta_AE =
    |prediction_grouped - observed|
    - |prediction_RCSV - observed|
```

A positive value indicates greater absolute error under the grouped split.
Repeated M1101 identifiers are aggregated before pairing.

The finalized inference procedure uses:

- 10,000 complete-PDB-group bootstrap resamples;
- 100,000 complete-PDB-group Monte Carlo sign-flip iterations;
- two-sided tests;
- Benjamini–Hochberg correction over 36 comparisons;
- false-discovery-rate threshold `0.05`; and
- random seed `20260802`.

```bash
python paired_analysis.py \
  --predictions output/performance_contraction/all_oof_predictions.csv \
  --output-dir output/paired_analysis
```

Output:

```text
output/paired_analysis/paired_error_summary.csv
```

The reported bootstrap confidence intervals are pointwise and unadjusted. The
Benjamini–Hochberg correction is applied to the sign-flip-test p-values.

## Prediction-file layout

The metric scripts expect prediction CSV files below a common root:

```text
output/SHUJU_clean/
└── {model}/
    └── {dataset}/
        └── {split}/
            └── {dataset}_{split}_fold{k}_pred.csv
```

For example:

```text
output/SHUJU_clean/SAMAffinity/S1131/RCSV/S1131_RCSV_fold0_pred.csv
```

Each model must provide all combinations of:

- datasets: `S1131`, `M1101`;
- splits: `RCSV`, `PCSV`, `ICSV`; and
- folds: `0`, `1`, `2`, `3`, `4`.

Prediction files must contain an identifier column, an observed ΔΔG column, and
a predicted ΔΔG column. The scripts recognize several common column aliases but
reject ambiguous inputs.

## Recommended analysis workflow

Use the predefined split files for model evaluation. After obtaining one
prediction CSV per model, dataset, split, and fold, run:

```bash
python calculate_metrics.py \
  --prediction-root output/SHUJU_clean \
  --output-dir output/fold_metrics

python analyze_performance_contraction.py \
  --prediction-root output/SHUJU_clean \
  --data-dir data \
  --output-dir output/performance_contraction

python paired_analysis.py \
  --predictions output/performance_contraction/all_oof_predictions.csv \
  --output-dir output/paired_analysis
```

Run the molecular-exposure audits independently:

```bash
python sequence_separation.py \
  --data-dir data \
  --split-root data \
  --output-dir output/sequence_separation \
  --mmseqs mmseqs \
  --threads 16

python structure_exposure.py \
  --split-root data \
  --structure-root data \
  --output-dir output/structure_exposure \
  --usalign USalign \
  --threads 8
```

Add `--overwrite` when intentionally replacing an existing publication output.

## Reproducibility notes

- Every source record must occur in exactly one validation fold for each split.
- Within each fold, the training and validation files must reproduce the full
  source dataset without missing or extra records.
- PDB grouping controls complete PDB identifiers, not sequence or structural
  similarity across different entries.
- Sequence clustering does not automatically control global structure, local
  interface similarity, experimental source, temporal exposure, or label
  distribution.
- Fold-mean and pooled OOF metrics have different statistical interpretations
  and should be reported together.
- The grouped bootstrap and sign-flip analyses are conditional on the fixed
  folds and fixed OOF predictions. They do not include retraining,
  hyperparameter-selection, or alternative-split uncertainty.

## Data and licence notes

S1131 is derived from the original SKEMPI database through the BindProfX
benchmark construction, whereas M1101 corresponds to AB-Bind. The corresponding
PDB structures are also derived from previously published resources. Users
should cite the original datasets, databases, and prediction methods listed in
the associated manuscript.

The inclusion of a structure or dataset file in this repository does not change
its original licence or attribution requirements. FoldX is not open-source
software and must not be redistributed without permission from its copyright
holder.

## Citation

If these partitions or analysis scripts are used, please cite the associated
article:

```text
Ge W, Han T, Zhao Q. Data Splits Define the Prediction Task:
A Perspective on Protein–Protein Binding ΔΔG Model Evaluation.
```
