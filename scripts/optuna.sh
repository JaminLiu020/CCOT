python scripts/train.py \
  --outdir ./results/ood_split_finetuning/977genes_full_dataset/drug_pert/NAOT_CCOT_250225/alpha_sigma_1.0_based_on_beta_10.0 \
  --config ./results/ood_split_finetuning/977genes_full_dataset/drug_pert/NAOT_CCOT_250225/alpha_sigma_1.0_based_on_beta_10.0/config.yaml \
  --config ./configs/optuna.yaml \
  --config.training.device cuda:0