# CCOT: Condition-Controlled Optimal Transport

<p align="center"><img src="assets/CCOT_CN.drawio.svg" alt="CCOT 架构图" width="100%"></p>

## 项目简介 (Project Introduction)

CCOT 对应论文 *Condition-Controlled Optimal Transport for Cellular Perturbation Response Prediction*，面向单细胞扰动响应预测任务。该任务的核心难点在于：scRNA-seq 数据通常无法获得扰动前后的配对细胞，因此模型需要学习从未扰动细胞分布到扰动后细胞分布的映射，同时还要具备对新药物的泛化能力。CCOT 基于最优传输框架，在同一模型中统一了 conditional transport 和 unconditional transport，用于预测未观测扰动条件下的细胞状态变化。

本仓库是论文配套研究代码，基于 `CondOT` 与 `chemCPA` 的部分实现和实验流程扩展而来，并补充了 CCOT 的训练、评估、分析脚本与推理工具。

## 方法要点 (Highlights)

- **Condition-Controlled mechanism**: 用可调参数控制 conditional transport 与 unconditional transport 的组合强度。
- **Cross-Drug Generalization**: 利用基于 SMILES 的预训练药物表示，支持对未见药物的响应预测。
- **Integration of Biological Priors**: 将扰动条件编码与细胞类型对齐纳入建模过程，提升生物学合理性。

## 架构说明 (Overview)

上方架构图展示了 CCOT 的主要组件；更完整的方法描述可参考 `assets/template.tex`。

## 环境要求 (Requirements)

- Python 3.9.7
- PyTorch 2.0.1
- JAX 0.4.13 / OTT-JAX 0.4.2
- Scanpy 1.9.3
- 训练建议使用支持 CUDA 的 GPU

## 安装 (Installation)

```bash
conda create -n ccot python=3.9.7
conda activate ccot
pip install -r requirements.txt

# 可选：以 editable 模式安装仓库
pip install -e .
```

## 数据与依赖资源 (Data and Resources)

仓库包含配置文件、参考 CSV 和部分模型资源，但默认训练 / 评估配置依赖额外准备好的 SciPlex3 数据、药物 embedding，以及配置中引用的权重文件。按仓库中的批量训练脚本，默认任务配置是 `configs/tasks/sciplex3_lincs_genes.yaml`，其数据路径指向 `datasets/chemCPA/sciplex_complete_lincs_genes_v3.h5ad`；这些大文件不随仓库完整跟踪。

仓库中已保留的小型参考文件主要位于 `datasets/reference/smiles/`，分类器相关权重位于 `notebooks/classifier/`。

## 快速开始 (Quick Start)

### 训练

下面的示例按照仓库现有 shell 脚本的默认训练组合给出：

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

如果需要调整 condition-controlled 机制中的 `beta`，可通过真实配置项覆盖，例如：

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

### 评估

训练完成后，可直接复用输出目录中的 `config.yaml` 进行评估：

```bash
python scripts/eval.py \
  --outdir ./results/quickstart \
  --config ./results/quickstart/config.yaml \
  --config.training.device cuda:0
```

### 批量脚本

批量训练、beta 扫描和批量评估脚本位于 `scripts/runs/`，例如：

```bash
bash scripts/runs/train_beta_sweep.sh
bash scripts/runs/eval_beta_sweep.sh
```

其中 `train_beta_sweep.sh` 默认使用：

- `./configs/condot.yaml`
- `./configs/tasks/sciplex3_lincs_genes.yaml`
- `embeddings/rdkit2D_embedding_lincs_trapnell_chemCPA.parquet`

`configs/experiments/val.yaml` 与 `configs/projections/pca.yaml` 并不是这组 shell 脚本默认训练流程的一部分。

## 目录结构 (Directory Structure)

```text
.
├── ccot/             # CCOT 相关评估、损失函数与辅助工具
├── condot/           # OT 模型、训练逻辑与数据处理
├── chemCPA/          # chemCPA 相关模块
├── configs/          # 训练、任务、投影与实验配置
├── scripts/          # 训练、评估、分析与批量运行脚本
├── datasets/         # 参考数据与数据划分文件
├── inference_kits/   # 独立推理工具包
├── notebooks/        # 实验 notebook 与部分资源
└── README.md
```

## 补充说明 (Notes)

- `scripts/train.py` 和 `scripts/eval.py` 都通过 `--config` 与 `--config.xxx.yyy` 的形式读取和覆盖配置。
- `inference_kits/` 用于存放与训练主流程解耦的推理工具，不是根 README 的主要使用入口。
- 如需核对方法定义与实验设定，请以 `assets/template.tex` 中的论文原文为准。
