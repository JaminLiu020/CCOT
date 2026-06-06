# CCOT: Conditional Cellular Optimal Transport

<p align="center"><img src="assets/overview.png" alt="Overview" width="100%"></p>

## 项目简介 (Project Introduction)

CCOT (Conditional Cellular Optimal Transport) 是一个基于条件最优传输理论的单细胞扰动预测框架。该项目旨在通过学习细胞在不同扰动条件下的分布变化规律，预测未观测扰动下的细胞状态。

**核心创新**：
- **自适应质量传输 (Adaptive Mass Transport, AMT)**：动态调整源分布和目标分布之间的质量传输策略，更好地捕捉生物学异质性
- **分类器自由引导 (Classifier-Free Guidance, CFG)**：通过结合条件生成和无条件生成，增强模型对扰动条件的响应能力
- **条件 OT 建模**：使用输入凸神经网络 (PICNN) 参数化最优传输映射，保证传输的凸性和稳定性

CCOT 建立在 CondOT 和 chemCPA 代码库之上，将可复用代码、实验配置、批量运行脚本和分析输出分离，使项目结构更清晰。

## 架构示意图 (Architecture)

<p align="center"><img src="assets/CCOT_CN.drawio.svg" alt="CCOT Architecture" width="100%"></p>

完整的架构图展示了数据流、模型组件以及训练/推理流程。详见 `assets/CCOT_CN.drawio.svg`。

## 环境要求 (Requirements)

- **Python**: 3.9.7
- **GPU**: CUDA-capable GPU (推荐用于训练，评估可用 CPU)
- **内存**: 建议 16GB+ RAM
- **核心依赖**:
  - PyTorch 2.0.1 (深度学习框架)
  - JAX 0.4.13 + OTT-JAX 0.4.2 (最优传输计算)
  - Scanpy 1.9.3 (单细胞数据分析)
  - NumPy 1.24.4, Pandas 2.0.3 (数值计算与数据处理)

## 安装步骤 (Installation)

### 1. 创建 Conda 环境

```bash
# 创建 Python 3.9.7 环境
conda create --name cell python=3.9.7
conda activate cell
```

### 2. 安装依赖

项目提供了两种依赖安装方式：

```bash
# 方式 1：通过 requirements.txt 安装（推荐用于开发）
pip install -r requirements.txt

# 方式 2：通过 setup.py 安装核心依赖
pip install -e .
```

**区别说明**：
- `requirements.txt` 包含所有依赖（包括开发工具如 jupyterlab, ipython）
- `setup.py` 只包含核心运行时依赖，适合部署环境

### 3. 验证安装

```bash
# 验证核心模块可以正常导入
python -c "from ccot.evaluate import evaluate; from condot.train import train; print('✓ Installation successful')"
```

## 快速开始 (Quick Start)

以下是一个最小化的端到端训练示例，使用 sciplex3 数据集训练 100 步：

```bash
# 训练一个简单的 CondOT 模型
python scripts/train.py \
  --outdir ./results/quickstart \
  --config ./configs/condot.yaml \
  --config ./configs/tasks/sciplex3-top1k.yaml \
  --config ./configs/experiments/val.yaml \
  --config ./configs/projections/pca.yaml \
  --config.training.n_iters 100

# 评估训练好的模型
python scripts/eval.py \
  --outdir ./results/quickstart
```

训练完成后，结果保存在 `./results/quickstart/` 目录下。

## 目录结构 (Directory Structure)

```
ccot/
├── ccot/                          # CCOT 特定功能（12 个文件）
│   ├── evaluate/                  # 评估框架和指标计算
│   ├── losses/                    # 自定义损失函数（FID, Energy Distance, R², KL, PS）
│   └── utils/                     # 工具函数（日志记录、Optuna 集成）
├── condot/                        # 核心传输模型（31 个文件）
│   ├── data/                      # 数据加载器（cell data, chemCPA data）
│   ├── losses/                    # 损失实现（MMD, Wasserstein, etc.）
│   ├── models/                    # 模型定义（Autoencoder, CondOT）
│   ├── networks/                  # 网络组件（PICNN, NPICNN, Embedding layers）
│   ├── train/                     # 训练循环和实验管理
│   ├── utils/                     # 辅助工具（配置加载、日志、Git 追踪）
│   └── valid/                     # 验证工具
├── chemCPA/                       # chemCPA 集成（部分 CCOT 实验使用）
├── configs/                       # YAML 配置文件
│   ├── tasks/                     # 数据集特定配置（sciplex3, norman）
│   ├── experiments/               # 训练/验证/测试分割配置
│   ├── projections/               # 降维配置（PCA, Autoencoder）
│   ├── condot.yaml                # CondOT 模型基础配置
│   ├── autoencoder.yaml           # Autoencoder 配置
│   └── optuna.yaml                # 超参数搜索配置
├── scripts/                       # 实验脚本
│   ├── train.py                   # 训练入口（支持单次运行和 Optuna 搜索）
│   ├── eval.py                    # 评估入口
│   ├── runs/                      # 批量运行脚本
│   │   ├── train_beta_sweep.sh   # Beta 参数扫描训练
│   │   ├── eval_beta_sweep.sh    # Beta 扫描批量评估
│   │   └── optuna_resume.sh      # 恢复 Optuna 超参数搜索
│   ├── analysis/                  # 后处理分析和可视化
│   │   ├── aggregate_metrics.py  # 聚合实验指标
│   │   ├── plot_eval_umap.py     # UMAP 降维可视化
│   │   ├── plot_heatmaps.py      # 热图绘制（性能对比、细胞类型分析）
│   │   ├── plot_hyperparam_sensitivity.py  # 超参数敏感性分析
│   │   └── plot_transport_map.py # 最优传输映射可视化
│   └── examples/                  # 独立演示脚本
│       └── transport_map_demo.py # 传输映射使用示例
├── datasets/                      # 数据存储（大型数据集需自行下载）
├── notebooks/                     # Jupyter notebooks（探索性分析）
├── results/                       # 训练输出（模型、日志、检查点）
└── README.md                      # 本文件
```

**注意**：
- `datasets/scrna-norman/*.csv` 文件保留在 git 中，因为它们是小的分割定义文件
- 大型数据集、嵌入向量、日志、实验输出、notebook 产物和生成图像在 `.gitignore` 中被忽略

## 训练模型 (Training)

### 单次训练 (Single Run)

使用 `scripts/train.py` 进行单次训练。该脚本支持多配置文件合并和命令行覆盖：

```bash
python scripts/train.py \
  --outdir ./results/models-pca-50d/scrna-sciplex3/drug-trametinib/emb-val/holdout-10/model-condot \
  --config ./configs/condot.yaml \
  --config ./configs/tasks/sciplex3-top1k.yaml \
  --config ./configs/experiments/val.yaml \
  --config ./configs/projections/pca.yaml \
  --config.data.property dose \
  --config.data.target trametinib \
  --config.datasplit.holdout.dose 100
```

**参数说明**：
- `--outdir`: 输出目录（保存模型、日志、检查点）
- `--config`: YAML 配置文件（可多次指定，按顺序合并）
- `--config.X.Y Z`: 命令行覆盖配置项（最高优先级）

### 批量实验 (Batch Experiments)

对于参数扫描和批量评估，使用 `scripts/runs/` 下的脚本：

#### Beta 参数扫描

```bash
# 训练多个 beta 值的模型
bash scripts/runs/train_beta_sweep.sh

# 批量评估所有 beta 模型
bash scripts/runs/eval_beta_sweep.sh
```

这些脚本支持环境变量覆盖：

```bash
# 自定义 beta 值和设备
BETAS="1.0 1.5 2.0 3.0" DEVICE="cuda:1" bash scripts/runs/train_beta_sweep.sh

# 自定义输出目录
OUTDIR_ROOT="./results/beta_sweep_custom" bash scripts/runs/eval_beta_sweep.sh
```

**可用环境变量**：
- `BETAS`: Beta 值列表（默认: "0.0 0.5 1.0 1.5 2.0 2.5 3.0 4.0"）
- `DEVICE`: CUDA 设备（默认: "cuda:0"）
- `OUTDIR_ROOT`: 输出根目录
- `EMBEDDING_PATH`: 预训练嵌入路径

#### Optuna 超参数搜索

```bash
# 开始新的超参数搜索
python scripts/train.py \
  --config ./configs/optuna.yaml \
  --config ./configs/tasks/sciplex3-top1k.yaml \
  --outdir ./results/optuna_search

# 恢复中断的搜索
bash scripts/runs/optuna_resume.sh
```

Optuna 会自动记录所有试验，支持并行搜索和断点恢复。

## 评估与分析 (Evaluation and Analysis)

### 模型评估

```bash
# 评估单个模型
python scripts/eval.py \
  --outdir ./results/models-pca-50d/scrna-sciplex3/drug-trametinib/emb-val/holdout-10/model-condot

# 评估时使用不同的 beta 值（CFG 强度）
python scripts/eval.py \
  --outdir ./results/some_model \
  --beta 2.0
```

评估指标包括：
- **FID** (Fréchet Inception Distance): 分布相似度
- **Energy Distance**: 分布距离度量
- **R²**: 差异表达基因的预测准确度
- **KL Divergence**: 分布散度
- **PS** (Perturbation Score): 扰动响应强度

### 结果分析与可视化

所有分析脚本位于 `scripts/analysis/`，接受显式路径参数：

```bash
# 1. 聚合实验指标（生成 Excel 报表）
python scripts/analysis/aggregate_metrics.py \
  --root-dir ./results/beta_sweep

# 2. UMAP 可视化（降维后的细胞分布）
python scripts/analysis/plot_eval_umap.py \
  --eval-dir ./results/some_experiment/eval \
  --output-dir ./figures

# 3. 热图绘制（性能对比、细胞类型分析）
python scripts/analysis/plot_heatmaps.py \
  --input-dir ./results/visualization_exp \
  --include-violin

# 4. 超参数敏感性分析
python scripts/analysis/plot_hyperparam_sensitivity.py \
  --input-file ./results/optuna_search/DEGs_id_test.xlsx \
  --output-dir ./figures

# 5. 最优传输映射可视化
python scripts/analysis/plot_transport_map.py \
  --input-dir ./results/visualization_exp/CCOT/ood_DEGs \
  --output-dir ./figures
```

生成的图像默认保存到 `scripts/visualization_results/` 或指定的输出目录。

## 实验配置 (Configuration System)

CCOT 使用 **多文件 YAML 配置合并机制**，允许模块化和复用：

### 配置优先级（从低到高）

1. **基础模型配置** (`configs/condot.yaml`, `configs/autoencoder.yaml`)
2. **任务配置** (`configs/tasks/sciplex3-top1k.yaml`)
3. **实验配置** (`configs/experiments/val.yaml`)
4. **投影配置** (`configs/projections/pca.yaml`)
5. **命令行覆盖** (`--config.X.Y value`)

### 配置示例

```yaml
# configs/condot.yaml (基础配置)
model:
  name: condot
  dim: 128
  layers: [256, 256]

training:
  n_iters: 10000
  batch_size: 512
  lr: 0.001
```

```yaml
# configs/tasks/sciplex3-top1k.yaml (数据集配置)
data:
  name: sciplex3
  genes: top1k
  path: ./datasets/scrna-sciplex3
```

```yaml
# configs/experiments/val.yaml (训练/验证分割)
datasplit:
  type: stratified
  train_ratio: 0.8
  val_ratio: 0.2
```

### 命令行覆盖

```bash
# 覆盖训练迭代次数和批量大小
python scripts/train.py \
  --config ./configs/condot.yaml \
  --config.training.n_iters 5000 \
  --config.training.batch_size 1024
```

所有配置最终会合并为一个配置对象，命令行参数具有最高优先级。

## 常见问题 (Troubleshooting)

### 1. CUDA Out of Memory

**问题**：训练时出现 `RuntimeError: CUDA out of memory`

**解决方案**：
```bash
# 减小批量大小
python scripts/train.py \
  --config ./configs/condot.yaml \
  --config.training.batch_size 256

# 或使用 CPU（速度较慢）
python scripts/train.py \
  --config ./configs/condot.yaml \
  --device cpu
```

### 2. Import Errors

**问题**：`ModuleNotFoundError: No module named 'absl'` 或其他导入错误

**解决方案**：
```bash
# 确保依赖完整安装
pip install -r requirements.txt

# 验证安装
python -c "import absl; import yaml; import optuna; print('OK')"
```

### 3. Config Path Issues

**问题**：配置文件找不到或路径错误

**解决方案**：
- 使用相对路径时，确保从项目根目录运行
- 或使用绝对路径：
```bash
python scripts/train.py \
  --config /absolute/path/to/configs/condot.yaml
```

### 4. Missing Embeddings

**问题**：`FileNotFoundError: embedding file not found`

**解决方案**：
- 检查嵌入文件路径是否正确
- 确保预训练嵌入已下载到 `datasets/` 目录
- 嵌入文件应为 `.pt` 或 `.pkl` 格式的 PyTorch tensor

### 5. Git Tracking Issues

**问题**：意外跟踪了大文件或生成的输出

**解决方案**：
```bash
# 使用 .gitignore 忽略大文件
echo "results/" >> .gitignore
echo "datasets/*.h5ad" >> .gitignore
git rm --cached -r results/
```

## 引用 (Citation)

如果您在研究中使用了 CCOT，请引用：

```bibtex
@article{ccot2024,
  title={CCOT: Conditional Cellular Optimal Transport for Single-Cell Perturbation Prediction},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## 许可证 (License)

本项目遵循 MIT 许可证。详见 `LICENSE` 文件。

## 致谢 (Acknowledgments)

CCOT 建立在以下优秀项目之上：
- [CondOT](https://github.com/bunnech/condot): Supervised Training of Conditional Monge Maps
- [chemCPA](https://github.com/facebookresearch/chemCPA): Chemical Perturbation Autoencoder
- [OTT-JAX](https://github.com/ott-jax/ott): Optimal Transport Tools in JAX

感谢开源社区的贡献！
