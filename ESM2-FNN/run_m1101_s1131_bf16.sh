#!/usr/bin/env bash
# 并行运行 M1101/S1131 × ICSV/PCSV/RCSV 的 BF16 五折训练。

set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN=${PYTHON_BIN:-python}
PROJECT_ROOT=$(cd .. && pwd)
SPLIT_ROOT=${SPLIT_ROOT:-"${PROJECT_ROOT}/data"}
RESULT_ROOT=${RESULT_ROOT:-"${PROJECT_ROOT}/output/SHUJU_clean/ESM-FNN"}
: "${ESM_MODEL:?Set ESM_MODEL to the ESM-2 checkpoint path}"
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5}
LOG_ROOT=${LOG_ROOT:-./logs_bf16}
mkdir -p "${LOG_ROOT}"

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
if (( ${#GPU_LIST[@]} != 6 )); then
  echo "错误：GPU_IDS 必须提供6个逗号分隔的GPU编号。" >&2
  exit 1
fi

TOTAL_CPUS=$("${PYTHON_BIN}" -c 'import os; print(len(os.sched_getaffinity(0)))')
NUM_JOBS=6
MAX_CPU_THREADS=$(( TOTAL_CPUS * 14 / 100 ))

if [[ -z "${CPU_THREADS:-}" ]]; then
  CPU_THREADS=${MAX_CPU_THREADS}
fi

if (( CPU_THREADS < 1 )); then
  echo "错误：计算出的每任务 CPU 核数 < 1（可用核数=${TOTAL_CPUS}）。" >&2
  exit 1
fi
if (( CPU_THREADS > MAX_CPU_THREADS )); then
  echo "错误：CPU_THREADS=${CPU_THREADS} 超过 14% 上限 ${MAX_CPU_THREADS}。" >&2
  exit 1
fi

REQUIRED=$(( NUM_JOBS * CPU_THREADS ))
if (( REQUIRED > TOTAL_CPUS )); then
  echo "错误：六组任务各需 ${CPU_THREADS} 核，合计 ${REQUIRED} > 可用 ${TOTAL_CPUS}。" >&2
  exit 1
fi

echo "检测到 CPU=${TOTAL_CPUS} 核；每任务绑定 ${CPU_THREADS} 核（≤14%）；六组共占用 ${REQUIRED} 核。"

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
  echo "已启动 ${name}: GPU=${gpu}, CPU=${cores}, PID=$(<"${pid_file}"), LOG=${log}"
}

launch M1101 ICSV "${GPU_LIST[0]}" 0
launch M1101 PCSV "${GPU_LIST[1]}" 1
launch M1101 RCSV "${GPU_LIST[2]}" 2
launch S1131 ICSV "${GPU_LIST[3]}" 3
launch S1131 PCSV "${GPU_LIST[4]}" 4
launch S1131 RCSV "${GPU_LIST[5]}" 5

echo "六组 ESM 任务均已转入后台。查看进度示例:"
echo "  tail -f ${LOG_ROOT}/M1101_ICSV.log"
echo "  cat ${LOG_ROOT}/*.pid"
echo "结果目录示例:"
echo "  ${RESULT_ROOT}/M1101/ICSV/"