#!/usr/bin/env bash
# Launch M1101 and S1131 x ICSV/PCSV/RCSV, FP16, five folds.
#   bash run_m1101_s1131_bf16.sh
#
# M1101: ICSV->1, PCSV->2, RCSV->3
# S1131: ICSV->4, PCSV->5, RCSV->6
# Each task uses at most floor(0.14 * available cores), on disjoint tasksets.

set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN=${PYTHON_BIN:-python}
SPLIT_ROOT=${SPLIT_ROOT:-/data/data_split/splits}
RESULT_ROOT=${RESULT_ROOT:-/data/external/attab/ESM}
ESM_MODEL=${ESM_MODEL:-/data/ESM-FNN/esm2_t33_650M_UR50D.pt}
LOG_ROOT=${LOG_ROOT:-./logs_bf16}
mkdir -p "${LOG_ROOT}"

TOTAL_CPUS=$("${PYTHON_BIN}" -c 'import os; print(len(os.sched_getaffinity(0)))')
NUM_JOBS=6
MAX_CPU_THREADS=$(( TOTAL_CPUS * 14 / 100 ))

if [[ -z "${CPU_THREADS:-}" ]]; then
  CPU_THREADS=${MAX_CPU_THREADS}
fi

if (( CPU_THREADS < 1 )); then
  echo "error: cores per task < 1 (available=${TOTAL_CPUS})." >&2
  exit 1
fi
if (( CPU_THREADS > MAX_CPU_THREADS )); then
  echo "error: CPU_THREADS=${CPU_THREADS} exceeds the 14% cap ${MAX_CPU_THREADS}." >&2
  exit 1
fi

REQUIRED=$(( NUM_JOBS * CPU_THREADS ))
if (( REQUIRED > TOTAL_CPUS )); then
  echo "error: six jobs need ${CPU_THREADS} cores each, ${REQUIRED} > ${TOTAL_CPUS} available." >&2
  exit 1
fi

echo "CPU=${TOTAL_CPUS}; ${CPU_THREADS} cores per task (at most 14%); six jobs use ${REQUIRED} cores."

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

launch() {
  local dataset=$1
  local split=$2
  local gpu=$3
  local job_idx=$4
  local cores="${CPU_GROUPS[${job_idx}]}"
  local name="${dataset}_${split}"
  local log="${LOG_ROOT}/${name}.log"
  local pid_file="${LOG_ROOT}/${name}.pid"

  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    OMP_NUM_THREADS="${CPU_THREADS}" \
    MKL_NUM_THREADS="${CPU_THREADS}" \
    OPENBLAS_NUM_THREADS="${CPU_THREADS}" \
    NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
    VECLIB_MAXIMUM_THREADS="${CPU_THREADS}" \
    PYTHONUNBUFFERED=1 \
    taskset -c "${cores}" \
    "${PYTHON_BIN}" esm2line.py \
      --dataset "${dataset}" \
      --split "${split}" \
      --split-root "${SPLIT_ROOT}" \
      --result-root "${RESULT_ROOT}" \
      --esm-model "${ESM_MODEL}" \
      --cpu-threads "${CPU_THREADS}" \
      >"${log}" 2>&1 &

  echo $! > "${pid_file}"
  echo "started ${name}: GPU=${gpu}, CPU=${cores}, PID=$(<"${pid_file}"), LOG=${log}"
}

launch M1101 ICSV 1 0
launch M1101 PCSV 2 1
launch M1101 RCSV 3 2
launch S1131 ICSV 4 3
launch S1131 PCSV 5 4
launch S1131 RCSV 6 5

echo "six ESM jobs are in the background. Progress:"
echo "  tail -f ${LOG_ROOT}/M1101_ICSV.log"
echo "  cat ${LOG_ROOT}/*.pid"
echo "result directories:"
echo "  ${RESULT_ROOT}/M1101/result/ICSV/"