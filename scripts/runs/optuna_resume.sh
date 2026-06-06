#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

OUTDIR=${OUTDIR:-./results/ood_split_finetuning/977genes_full_dataset/drug_pert/NAOT_CCOT_250225/alpha_sigma_1.0_based_on_beta_10.0}
CONFIG_PATH=${CONFIG_PATH:-${OUTDIR}/config.yaml}
DEVICE=${DEVICE:-cuda:0}
OPTUNA_CONFIG=${OPTUNA_CONFIG:-./configs/optuna.yaml}

python scripts/train.py \
  --outdir "${OUTDIR}" \
  --config "${CONFIG_PATH}" \
  --config "${OPTUNA_CONFIG}" \
  --config.training.device "${DEVICE}"
