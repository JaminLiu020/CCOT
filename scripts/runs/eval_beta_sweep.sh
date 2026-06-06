#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

BETAS=${BETAS:-"0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0"}
DEVICE=${DEVICE:-cuda:0}
OUTDIR_ROOT=${OUTDIR_ROOT:-./results/nine_drugs/977genes/drug_pert/CCOT}
ALIGN_BY_CELL_TYPE=${ALIGN_BY_CELL_TYPE:-True}

for beta in $BETAS; do
  python scripts/eval.py \
    --outdir "${OUTDIR_ROOT}/beta_${beta}" \
    --config "${OUTDIR_ROOT}/beta_${beta}/config.yaml" \
    --config.datasplit.align_by_cell_type "${ALIGN_BY_CELL_TYPE}" \
    --config.training.device "${DEVICE}"
done
