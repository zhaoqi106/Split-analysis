#!/usr/bin/env bash
# Four GPUs, one dataset each. family then legacy, FP16.
#   bash run_family_legacy_bf16.sh
#
# GPU4 -> S1131, GPU5 -> S645, GPU6 -> S4169, GPU7 -> M1101.
# Logs: ./{dataset}_log.txt
# Data: /data/external/model/{DS}/{family|legacy}/
# Output: /data/external/model/{DS}/esmfnn/{family|legacy}/predicate_fold*.csv

set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN=${PYTHON_BIN:-python}
SPLIT_ROOT=${SPLIT_ROOT:-/data/external/model}
RESULT_ROOT=${RESULT_ROOT:-/data/external/model}
ESM_MODEL=${ESM_MODEL:-/data/ESM-FNN/esm2_t33_650M_UR50D.pt}
LOG_DIR=${LOG_DIR:-.}
mkdir -p "${LOG_DIR}"

# dataset:gpu
JOBS=(
  "S1131:4"
  "S645:5"
  "S4169:7"
  "M1101:6"
)
NUM_JOBS=${#JOBS[@]}
SPLITS=(family legacy)

TOTAL_CPUS=$("${PYTHON_BIN}" -c 'import os; print(len(os.sched_getaffinity(0)))')
MAX_CPU_THREADS=$(( TOTAL_CPUS * 20 / 100 ))

if [[ -z "${CPU_THREADS:-}" ]]; then
  CPU_THREADS=${MAX_CPU_THREADS}
fi

if (( CPU_THREADS < 1 )); then
  echo "error: cores per task < 1 (available=${TOTAL_CPUS})." >&2
  exit 1
fi
if (( CPU_THREADS > MAX_CPU_THREADS )); then
  echo "error: CPU_THREADS=${CPU_THREADS} exceeds the 20% cap ${MAX_CPU_THREADS}." >&2
  exit 1
fi

REQUIRED=$(( NUM_JOBS * CPU_THREADS ))
if (( REQUIRED > TOTAL_CPUS )); then
  echo "error: ${NUM_JOBS} jobs need ${CPU_THREADS} cores each, ${REQUIRED} > ${TOTAL_CPUS} available." >&2
  exit 1
fi

echo "CPU=${TOTAL_CPUS}; ${CPU_THREADS} cores per task; ${REQUIRED} cores in total."
echo "data: ${SPLIT_ROOT}/{DS}/{family|legacy}/"
echo "results: ${RESULT_ROOT}/{DS}/esmfnn/{family|legacy}/"
echo "logs: ${LOG_DIR}/{DS}_log.txt"

mapfile -t CPU_GROUPS < <(
  "${PYTHON_BIN}" - "${CPU_THREADS}" "${NUM_JOBS}" <<'PY'
import os
import sys

per_job = int(sys.argv[1])
jobs = int(sys.argv[2])
cpus = sorted(os.sched_getaffinity(0))
for i in range(jobs):
    group = cpus[i * per_job:(i + 1) * per_job]
    print(",".join(map(str, group)))
PY
)

launch_dataset() {
  local dataset=$1
  local gpu=$2
  local job_idx=$3
  local cores="${CPU_GROUPS[${job_idx}]}"
  local log="${LOG_DIR}/${dataset}_log.txt"
  local pid_file="${LOG_DIR}/${dataset}.pid"

  # family then legacy on the same GPU; both append to the same log
  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    OMP_NUM_THREADS="${CPU_THREADS}" \
    MKL_NUM_THREADS="${CPU_THREADS}" \
    OPENBLAS_NUM_THREADS="${CPU_THREADS}" \
    NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
    VECLIB_MAXIMUM_THREADS="${CPU_THREADS}" \
    PYTHONUNBUFFERED=1 \
    bash -c '
      set -euo pipefail
      dataset="$1"
      cores="$2"
      python_bin="$3"
      split_root="$4"
      result_root="$5"
      esm_model="$6"
      cpu_threads="$7"
      shift 7
      for split in "$@"; do
        echo "======== $(date "+%F %T") START ${dataset}/${split} ========"
        taskset -c "${cores}" \
          "${python_bin}" esm2line.py \
            --dataset "${dataset}" \
            --split "${split}" \
            --split-root "${split_root}" \
            --result-root "${result_root}" \
            --esm-model "${esm_model}" \
            --cpu-threads "${cpu_threads}"
        echo "======== $(date "+%F %T") DONE  ${dataset}/${split} ========"
      done
    ' _ "${dataset}" "${cores}" "${PYTHON_BIN}" "${SPLIT_ROOT}" "${RESULT_ROOT}" "${ESM_MODEL}" "${CPU_THREADS}" "${SPLITS[@]}" \
    >"${log}" 2>&1 &

  echo $! > "${pid_file}"
  echo "started ${dataset}: GPU=${gpu}, CPU=${cores}, PID=$(<"${pid_file}"), LOG=${log}"
}

job_idx=0
for item in "${JOBS[@]}"; do
  ds="${item%%:*}"
  gpu="${item##*:}"
  launch_dataset "${ds}" "${gpu}" "${job_idx}"
  job_idx=$((job_idx + 1))
done

echo "4 jobs are in the background (family, then legacy, on each GPU). Progress:"
echo "  tail -f ${LOG_DIR}/S1131_log.txt"
echo "  cat ${LOG_DIR}/*.pid"
