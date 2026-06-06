#for beta in 1.0; do
#  python scripts/train.py \
#    --outdir ./results/nine_drugs/977genes/drug_pert/ablation_experiment/CondOT_drug_encoder_cell_type_alignment/beta_${beta} \
#    --config ./configs/condot.yaml \
#    --config ./configs/tasks/sciplex3_lincs_genes.yaml \
#    --config.model.classifier_free_guidance.beta ${beta} \
#    --config.training.device cuda:2
#done

for beta in 15.0 14.0 15.0 20.0 3.0 2.0 1.0 0.0; do
  python scripts/train.py \
    --outdir ./results/nine_drugs/977genes/drug_pert/CCOT/beta_${beta} \
    --config ./configs/condot.yaml \
    --config ./configs/tasks/sciplex3_lincs_genes.yaml \
    --config.model.classifier_free_guidance.beta ${beta} \
    --config.datasplit.align_by_cell_type True \
    --config.model.embedding.type smiles \
    --config.model.embedding.path /home/jamin/CSG2A/chemCPA_inference_kit/pretrained/rdkit2D_embedding_lincs_trapnell_chemCPA.parquet \
    --config.training.device cuda:1 \
    --config.dataloader.batch_size 512 \
    --config.optim.lr 0.0002 \
    --config.training.n_iters 50001
done
