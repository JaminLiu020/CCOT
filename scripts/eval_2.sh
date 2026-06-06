for beta in 7.0 6.0 5.0 4.0 3.0 2.0 1.0; do
  python scripts/eval.py \
    --outdir ./results/nine_drugs/977genes/drug_pert/CCOT/beta_${beta} \
    --config ./results/nine_drugs/977genes/drug_pert/CCOT/beta_${beta}/config.yaml \
    --config.datasplit.align_by_cell_type True \
    --config.training.device cuda:1
done
