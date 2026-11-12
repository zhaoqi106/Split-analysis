#!/usr/bin/env bash
# Pack ESM prediction CSVs as {dataset}/{split}/predicate_fold*.csv
#   bash pack_predictions.sh
# RESULT_ROOT=/data/external/model
# OUT_ZIP=/data/external/model/ESM_family_legacy_predictions.zip

set -euo pipefail
cd "$(dirname "$0")"

RESULT_ROOT=${RESULT_ROOT:-/data/external/model}
OUT_ZIP=${OUT_ZIP:-${RESULT_ROOT}/ESM_family_legacy_predictions.zip}
STAGING=$(mktemp -d)
trap 'rm -rf "${STAGING}"' EXIT

DATASETS=(S1131 S645 S4169 M1101)
SPLITS=(family legacy)

echo "result root: ${RESULT_ROOT}"
echo "zip: ${OUT_ZIP}"

copied=0
missing=0

for ds in "${DATASETS[@]}"; do
  for spl in "${SPLITS[@]}"; do
    src="${RESULT_ROOT}/${ds}/esmfnn/${spl}"
    dst="${STAGING}/${ds}/${spl}"
    mkdir -p "${dst}"

    if [[ ! -d "${src}" ]]; then
      echo "missing directory: ${src}"
      missing=$((missing + 1))
      continue
    fi

    for fold in 0 1 2 3 4; do
      f="${src}/predicate_fold${fold}.csv"
      if [[ -f "${f}" ]]; then
        cp -f "${f}" "${dst}/"
        copied=$((copied + 1))
      else
        echo "missing: ${f}"
        missing=$((missing + 1))
      fi
    done

    summary="${src}/predicate_all_folds.csv"
    if [[ -f "${summary}" ]]; then
      cp -f "${summary}" "${dst}/"
      copied=$((copied + 1))
    else
      echo "optional summary missing: ${summary}"
    fi
  done
done

if (( missing > 0 )); then
  echo "${missing} files are missing; packing stopped." >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_ZIP}")"
rm -f "${OUT_ZIP}"
(
  cd "${STAGING}"
  zip -r "${OUT_ZIP}" .
)

echo "packed ${copied} CSVs -> ${OUT_ZIP}"
echo "example path inside the zip: S1131/family/predicate_fold0.csv"
unzip -l "${OUT_ZIP}" | sed -n '1,40p'
