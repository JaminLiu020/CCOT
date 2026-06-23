# CCOT: Condition-Controlled Optimal Transport

<p align="center"><img src="assets/CCOT.svg" alt="CCOT architecture overview" width="100%"></p>

<p align="center">
  <strong>English</strong> | <a href="README_CN.md">中文</a>
</p>

---

## Project Introduction

CCOT is the companion code for the paper *Condition-Controlled Optimal Transport for Cellular Perturbation Response Prediction*, targeting the task of single-cell perturbation response prediction. A key challenge in this setting is that scRNA-seq data rarely provides paired cells before and after perturbation, so the model must learn a mapping from unperturbed to perturbed cell distributions while also generalizing to unseen drugs. Built on an optimal transport framework, CCOT unifies conditional and unconditional transport within a single model to predict cell state changes under unobserved perturbations.

This repository extends implementations and experimental pipelines from `CondOT` and `chemCPA`, and adds training, evaluation, and analysis scripts along with inference utilities specific to CCOT.

## Highlights

- **Condition-Controlled mechanism**: A tunable parameter controls the balance between conditional and unconditional transport.
- **Cross-Drug Generalization**: SMILES-based pretrained drug representations enable response prediction for unseen compounds.
- **Integration of Biological Priors**: Perturbation condition encoding and cell-type alignment are incorporated into the modeling process for improved biological plausibility.

## Overview

The diagram above illustrates the main components of CCOT; a more detailed method description can be found in `assets/template.tex`.

## Requirements

- Python 3.9.7
- PyTorch 2.0.1
- JAX 0.4.13 / OTT-JAX 0.4.2
- Scanpy 1.9.3
- A CUDA-capable GPU is recommended for training

## Installation

```bash
conda create -n ccot python=3.9.7
conda activate ccot
pip install -r requirements.txt

# Optional: Install in editable mode so that ccot, condot,
# and chemCPA are registered as importable packages,
# allowing code changes to take effect without reinstallation.
pip install -e .
```

## Data and Resources

The repository includes configuration files, reference CSVs, and some model assets, but the default training and evaluation configurations depend on externally prepared SciPlex3 data, drug embeddings, and checkpoint files referenced in the configs. Following the batch training scripts shipped in this repo, the default task configuration is `configs/tasks/sciplex3_lincs_genes.yaml`, whose data path points to `datasets/chemCPA/sciplex_complete_lincs_genes_v3.h5ad`; these large files are not fully tracked in the repository.

Smaller reference files are kept under `datasets/reference/smiles/`, and classifier-related weights are located in `notebooks/classifier/`.

## Quick Start

### Training

The examples below follow the default training combinations used by the shell scripts in the repository:

```bash
python scripts/train.py \
  --outdir ./results/quickstart \
  --config ./configs/condot.yaml \
  --config ./configs/tasks/sciplex3_lincs_genes.yaml \
  --config.model.embedding.type smiles \
  --config.model.embedding.path embeddings/rdkit2D_embedding_lincs_trapnell_chemCPA.parquet \
  --config.datasplit.align_by_cell_type True \
  --config.dataloader.batch_size 512 \
  --config.optim.lr 0.0002 \
  --config.training.n_iters 100 \
  --config.training.device cuda:0
```

To adjust the `beta` parameter in the condition-controlled mechanism, override the corresponding config entry:

```bash
python scripts/train.py \
  --outdir ./results/beta10 \
  --config ./configs/condot.yaml \
  --config ./configs/tasks/sciplex3_lincs_genes.yaml \
  --config.model.classifier_free_guidance.beta 10.0 \
  --config.model.embedding.type smiles \
  --config.model.embedding.path embeddings/rdkit2D_embedding_lincs_trapnell_chemCPA.parquet \
  --config.datasplit.align_by_cell_type True \
  --config.dataloader.batch_size 512 \
  --config.optim.lr 0.0002 \
  --config.training.device cuda:0
```

### Evaluation

After training, reuse the `config.yaml` saved in the output directory for evaluation:

```bash
python scripts/eval.py \
  --outdir ./results/quickstart \
  --config ./results/quickstart/config.yaml \
  --config.training.device cuda:0
```

### Batch Scripts

Batch training, beta sweep, and batch evaluation scripts are located in `scripts/runs/`, for example:

```bash
bash scripts/runs/train_beta_sweep.sh
bash scripts/runs/eval_beta_sweep.sh
```

`train_beta_sweep.sh` uses the following defaults:

- `./configs/condot.yaml`
- `./configs/tasks/sciplex3_lincs_genes.yaml`
- `embeddings/rdkit2D_embedding_lincs_trapnell_chemCPA.parquet`

Note that `configs/experiments/val.yaml` and `configs/projections/pca.yaml` are not part of the default training pipeline invoked by these shell scripts.

## Directory Structure

```text
.
├── assets/                    # Architecture diagrams and paper draft
├── ccot/                      # CCOT evaluation, loss functions, and utilities
├── chemCPA/                   # chemCPA-related modules
├── condot/                    # OT models, training logic, and data processing
├── configs/                   # Training, task, projection, and experiment configs
├── datasets/                  # Reference data and data split files
├── embeddings/                # Pretrained drug embedding files
├── inference_kits/            # Standalone inference toolkits
├── notebooks/                 # Experiment notebooks and resources
├── scripts/                   # Training, evaluation, analysis, and batch scripts
├── biolord_reproducibility/   # Biolord comparison experiments
├── PerturbNet/                # PerturbNet comparison experiments
└── README.md
```

## Comparison Experiments

This repository includes two directories for fair comparison with baseline methods under identical data splits and evaluation protocols.

### Biolord (`biolord_reproducibility/`)

Reproduce Biolord's sciplex3 experiments under CCOT's condot-aligned split.

```bash
conda create -n biolord python=3.10 -y
conda activate biolord
cd biolord_reproducibility
pip install -r requirements.txt
jupyter notebook preprocess.ipynb   # Data preprocessing
jupyter notebook run_biolord.ipynb  # Training + evaluation
```

### PerturbNet (`PerturbNet/`)

Reproduce PerturbNet's sciplex3 experiments under CCOT's condot-aligned split.

```bash
conda create -n PerturbNet python=3.7 -y
conda activate PerturbNet
cd PerturbNet
pip install -r requirements.txt
pip install -e .
jupyter notebook run_perturbnet.ipynb  # Training + evaluation
```

### Environment Notes

Due to incompatible dependency versions, the three projects require separate conda environments:

| Project | conda env | Python | PyTorch |
|---------|-----------|--------|---------|
| CCOT | `ccot` | 3.9.7 | 2.0.1 |
| Biolord | `biolord` | 3.10 | 2.0.1 |
| PerturbNet | `PerturbNet` | 3.7 | 1.13.1 |

**Important:** `matplotlib` must be pinned to `3.7.2` across all three environments; otherwise, figure dimensions will differ and visual comparisons will be misleading.

See the README in each subdirectory for further details.

## Notes

- Both `scripts/train.py` and `scripts/eval.py` read and override configuration via `--config` and `--config.xxx.yyy` flags.
- `inference_kits/` contains two standalone inference toolkits decoupled from the main training pipeline:
  - `ccot_inference_kit/`: A lightweight inference wrapper for CCOT pretrained models, with no condot dependency, designed to be portable across projects.
  - `chemCPA_inference_kit/`: An inference wrapper for chemCPA pretrained models, supporting drug response prediction via `drug_idx` or SMILES.
- For precise method definitions and experimental settings, refer to the paper draft in `assets/template.tex`.
