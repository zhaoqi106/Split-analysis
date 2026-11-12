#!/usr/bin/env bash
# Stop esm2line.py on a watched GPU when its memory use reaches the threshold.
#   nohup bash vram_watchdog.sh > vram_watchdog_nohup.txt 2>&1 &
# WATCH_GPUS=4,6,7
# VRAM_THRESH=92
# INTERVAL_SEC=5
# LOG_FILE=./vram_watchdog.log

set -uo pipefail
cd "$(dirname "$0")"

WATCH_GPUS=${WATCH_GPUS:-4,6,7}
VRAM_THRESH=${VRAM_THRESH:-92}
INTERVAL_SEC=${INTERVAL_SEC:-5}
LOG_FILE=${LOG_FILE:-./vram_watchdog.log}
MATCH_PATTERN=${MATCH_PATTERN:-esm2line.py}

IFS=',' read -r -a GPU_ARR <<< "${WATCH_GPUS}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

kill_our_jobs_on_gpu() {
  local gpu=$1
  local pids
  pids=$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -u || true)

  for pid in ${pids}; do
    [[ -z "${pid}" || ! "${pid}" =~ ^[0-9]+$ ]] && continue
    [[ ! -d "/proc/${pid}" ]] && continue
    local cmd
    cmd=$(ps -p "${pid}" -o args= 2>/dev/null || true)
    if [[ "${cmd}" == *"${MATCH_PATTERN}"* ]]; then
      log "GPU${gpu} over the memory limit, stopping PID=${pid}: ${cmd}"
      kill -TERM "${pid}" 2>/dev/null || true
      sleep 2
      if [[ -d "/proc/${pid}" ]]; then
        kill -KILL "${pid}" 2>/dev/null || true
      fi
    fi
  done
}

log "vram watchdog: GPUs=${WATCH_GPUS}, threshold=${VRAM_THRESH}%, interval=${INTERVAL_SEC}s"

while true; do
  for gpu in "${GPU_ARR[@]}"; do
    line=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used,memory.total \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)
    [[ -z "${line}" ]] && continue
    used=${line%%,*}
    total=${line##*,}
    [[ -z "${used}" || -z "${total}" || "${total}" -le 0 ]] && continue
    pct=$(( used * 100 / total ))
    if (( pct >= VRAM_THRESH )); then
      log "GPU${gpu} memory ${used}/${total} MiB = ${pct}% >= ${VRAM_THRESH}%, stopping esm2line on that GPU"
      kill_our_jobs_on_gpu "${gpu}"
    fi
  done
  sleep "${INTERVAL_SEC}"
done
