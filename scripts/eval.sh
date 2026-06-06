#for beta in 4.0; do
for beta in 5.0 7.0 9.0 11.0 13.0 15.0 17.0 19.0 16.0 18.0; do
  python scripts/eval.py \
    --outdir ./results/nine_drugs/977genes/drug_pert/CCOT/beta_${beta} \
    --config ./results/nine_drugs/977genes/drug_pert/CCOT/beta_${beta}/config.yaml \
    --config.datasplit.align_by_cell_type True \
    --config.training.device cuda:1
done
