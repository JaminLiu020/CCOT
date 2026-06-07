# Biolord 对比实验

本目录用于在 CCOT 的数据划分和评估标准下复现 [Biolord](https://www.biorxiv.org/content/10.1101/2023.03.05.531195) 的 sciplex3 实验结果，以便与 CCOT 进行公平对比。

## 环境安装

Biolord 需要独立的 conda 环境（与 CCOT 的 `cell` 环境不兼容）：

```bash
conda create -n biolord python=3.10 -y
conda activate biolord
pip install -r requirements.txt
```

**重要：** `matplotlib` 必须为 `3.7.2`，与 CCOT 保持一致，否则图形比例会有差异。

## 运行步骤

从 `biolord_reproducibility/` 目录运行：

### 1. 数据预处理

```bash
jupyter notebook preprocess.ipynb
```

修改 `RAW_DATA_PATH` 指向 sciplex 数据文件，运行全部 cell。输出保存到 `results/sciplex3/preprocessed/`。

### 2. 训练 + 评估

```bash
jupyter notebook run_biolord.ipynb
```

按需修改 `GPU_ID`、训练超参数等，运行全部 cell。结果保存到 `results/sciplex3/`。

## 目录结构

```text
biolord_reproducibility/
├── preprocess.ipynb          # 预处理入口
├── run_biolord.ipynb         # 训练+评估入口
├── condot_compat/            # CCOT 对齐的评估桥接模块
├── utils/                    # 模型超参数配置
├── notebooks/                # 原始 notebook（含完整实验记录）
├── results/                  # 实验结果
│   └── sciplex3/
│       ├── preprocessed/     # 预处理数据
│       └── biolord_.../      # 模型 checkpoint、评估指标、UMAP 图
└── requirements.txt          # 依赖
```

## 评估指标

与 CCOT 对齐的 5 个指标：MMD、L2、FID、R²、Energy Distance。评估在 DEGs 子空间上进行，使用 condot 风格的 5000 cell cap 和按固定 cell type 数（3）除的归一化。
