#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

BETAS=${BETAS:-"0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0"}
DEVICE=${DEVICE:-cuda:0}
OUTDIR_ROOT=${OUTDIR_ROOT:-./results/nine_drugs/977genes/drug_pert/CCOT}
TASK_CONFIG=${TASK_CONFIG:-./configs/tasks/sciplex3_lincs_genes.yaml}
MODEL_CONFIG=${MODEL_CONFIG:-./configs/condot.yaml}
EMBEDDING_PATH=${EMBEDDING_PATH:-embeddings/rdkit2D_embedding_lincs_trapnell_chemCPA.parquet}
BATCH_SIZE=${BATCH_SIZE:-512}
LR=${LR:-0.0002}
N_ITERS=${N_ITERS:-50001}
ALIGN_BY_CELL_TYPE=${ALIGN_BY_CELL_TYPE:-True}
EMBEDDING_TYPE=${EMBEDDING_TYPE:-smiles}

for beta in $BETAS; do
  python scripts/train.py \
    --outdir "${OUTDIR_ROOT}/beta_${beta}" \
    --config "${MODEL_CONFIG}" \
    --config "${TASK_CONFIG}" \
    --config.model.classifier_free_guidance.beta "${beta}" \
    --config.datasplit.align_by_cell_type "${ALIGN_BY_CELL_TYPE}" \
    --config.model.embedding.type "${EMBEDDING_TYPE}" \
    --config.model.embedding.path "${EMBEDDING_PATH}" \
    --config.training.device "${DEVICE}" \
    --config.dataloader.batch_size "${BATCH_SIZE}" \
    --config.optim.lr "${LR}" \
    --config.training.n_iters "${N_ITERS}"
done
