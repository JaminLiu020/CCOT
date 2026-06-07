# PerturbNet 对比实验

本目录用于在 CCOT 的数据划分和评估标准下复现 [PerturbNet](https://doi.org/10.1038/s44320-025-00131-3) 的 sciplex3 实验结果，以便与 CCOT 进行公平对比。

## 环境安装

PerturbNet 依赖 scvi-tools 0.7.1 + Python 3.7，**必须使用独立 conda 环境**：

```bash
conda create -n PerturbNet python=3.7 -y
conda activate PerturbNet
pip install -r requirements.txt
pip install -e .
```

**重要：** `matplotlib` 必须为 `3.7.2`，与 CCOT 保持一致，否则图形比例会有差异。

## 运行步骤

从 `PerturbNet/` 目录运行：

### 1. 训练 + 评估

```bash
jupyter notebook run_perturbnet.ipynb
```

修改 `cfg.data_path` 指向 sciplex 数据文件，按需调整 `GPU_ID`、训练开关（`DO_TRAIN_SCVI`/`DO_TRAIN_CINN`）。运行全部 cell。结果保存到 `results/sciplex3/`。

### 2. 离线绘图（可选）

```bash
python plot_eval.py
```

## 目录结构

```text
PerturbNet/
├── run_perturbnet.ipynb      # 训练+评估入口
├── plot_eval.py              # 离线 UMAP 绘图
├── condot_repro/             # CCOT 对齐的评估桥接模块
├── perturbnet/               # PerturbNet 核心包
├── chemical_vae/             # chemicalVAE 依赖
├── pretrained_model/         # 预训练权重
├── embeddings/               # 药物 embedding 文件
├── notebooks/                # 原始实验 notebook
├── results/                  # 实验结果
│   └── sciplex3/
│       └── sciplex_comp2condot_rdkit2d/
│           ├── models/       # scVI + cINN checkpoint
│           ├── eval/         # 评估指标 CSV + per-drug NPZ
│           └── ...
├── setup.py                  # 包安装
└── requirements.txt          # 依赖
```

## 评估指标

与 CCOT 对齐的 5 个指标：MMD、L2、FID、R²、Energy Distance。评估在全基因空间和 DEG-50 子空间上分别进行，使用 condot 风格的 5000 cell cap 和 size 对齐。
