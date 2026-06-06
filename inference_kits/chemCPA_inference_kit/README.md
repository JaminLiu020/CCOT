# chemCPA Inference Kit

独立的 chemCPA 推理模块。将预训练的 chemCPA 模型参数冻结后，作为预测组件嵌入到其他项目中。

在本仓库中，该套件位于 `inference_kits/chemCPA_inference_kit/`。

## 功能

- 加载预训练 chemCPA 模型（参数自动冻结，`eval()` 模式）
- 输入：对照组基因表达 + 药物（idx 或 SMILES） + 剂量
- 输出：扰动后基因表达预测值（均值），shape `[batch_size, num_genes]`
- 支持嵌入文件中全部 **17869 种药物**，通过 idx 或 SMILES 指定

## 推理管线

无论通过 `drug_idx` 还是 `smiles` 指定药物，内部都统一走同一条推理路径：

```
parquet 查找表 (17869×194)
        │
        ├─ drug_idx → 取第 idx 行
        └─ smiles   → 查 SMILES 对应行
        │
        ▼
194 维 rdkit2D 嵌入
        │
        ├──→ drug_embedding_encoder (MLP 194→32) ──→ 编码后的药物向量
        └──→ dosers (amortized MLP 195→1)        ──→ 剂量缩放因子
        │
        ▼
drug_effect = scaled_dosages × encoded_drugs
        │
        ▼
latent_treated = encoder(对照组) + drug_effect
        │
        ▼
decoder → 预测的扰动后基因表达
```

`drug_embedding_encoder` 和 `dosers` 是参数化的 MLP，学到的是 **rdkit2D 特征空间 → 隐空间的通用映射**，对 17869 种药物中的任何一种都适用。

## 目录结构

```
chemCPA_inference_kit/
├── __init__.py              # 包入口，导出 ChemCPAPredictor
├── model.py                 # ComPert 模型定义（精简版）
├── embedding.py             # 药物嵌入加载工具
├── inference.py             # ChemCPAPredictor 推理器
├── setup_pretrained.py      # 预训练文件复制脚本
├── test_inference.py        # 测试脚本
├── README.md                # 本文档
└── pretrained/              # 预训练文件目录
    ├── manual_2025-03-08_03-24-40.pt              # 模型权重
    └── rdkit2D_embedding_lincs_trapnell.parquet   # 药物嵌入 (17869×194)
```

## 快速开始

### 1. 复制到你的项目

```bash
cp -r inference_kits/chemCPA_inference_kit /path/to/your_project/
```

### 2. 准备预训练文件

```bash
cd inference_kits/chemCPA_inference_kit
python setup_pretrained.py /path/to/condot   # condot 项目根目录
```

或手动复制以下文件到 `pretrained/` 目录：
- 模型权重: `notebooks/evaluate_chemCPA/manual_2025-03-08_03-24-40.pt`
- 药物嵌入: `embeddings/rdkit2D_embedding_lincs_trapnell.parquet` (17869×194)

### 3. 运行测试

```bash
python test_inference.py
```

### 4. 在你的项目中使用

```python
import numpy as np
from chemCPA_inference_kit import ChemCPAPredictor

# 初始化（加载模型，参数自动冻结）
predictor = ChemCPAPredictor.from_pretrained(
    model_path="chemCPA_inference_kit/pretrained/manual_2025-03-08_03-24-40.pt",
    embedding_path="chemCPA_inference_kit/pretrained/rdkit2D_embedding_lincs_trapnell.parquet",
    device="cuda:0",  # 或 "cpu"
)

# 获取模型信息
num_genes = predictor.get_num_genes()   # 977
num_drugs = predictor.get_num_drugs()   # 17869
emb_dim = predictor.get_drug_embedding_dim()  # 194

# 准备对照组数据 (替换为真实数据)
control_expression = np.random.randn(64, num_genes).astype(np.float32)

# === 方式 1: 通过 drug_idx 预测 ===
# drug_idx 是药物在 parquet 嵌入文件中的行索引，范围 [0, 17868]
prediction = predictor.predict(
    control_expression=control_expression,
    drug_idx=42,
    dosage=1.0,
)
# prediction.shape = (64, 977)

# === 方式 2: 通过 SMILES 预测 ===
prediction = predictor.predict_by_smiles(
    control_expression=control_expression,
    smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
    dosage=1.0,
)

# === 批量预测（idx） ===
results = predictor.predict_batch(
    control_expression=control_expression,
    drug_indices=[0, 100, 5000],
    dosages=[1.0, 0.5, 2.0],
)
# results = {0: array(...), 100: array(...), 5000: array(...)}

# === 批量预测（SMILES） ===
results = predictor.predict_batch_by_smiles(
    control_expression=control_expression,
    smiles_list=["CCO", "CC(=O)O", "c1ccccc1"],
    dosages=[1.0, 0.5, 2.0],
)
# results = {"CCO": array(...), ...}

# === idx ↔ SMILES 互转 ===
smiles = predictor.idx_to_smiles(42)
idx = predictor.smiles_to_idx("CCO")
```

## API 参考

### `ChemCPAPredictor.from_pretrained(model_path, embedding_path, device)`
工厂方法，从 checkpoint + parquet 创建推理器。

### `predictor.predict(control_expression, drug_idx, dosage, covariates)`
通过 drug_idx（parquet 行索引）推理。返回 `np.ndarray [batch_size, num_genes]`。

### `predictor.predict_by_smiles(control_expression, smiles, dosage, covariates)`
通过 SMILES 推理（内部转为 idx 后调用 `predict`）。

### `predictor.predict_batch(control_expression, drug_indices, dosages, covariates)`
批量推理多种药物（idx）。返回 `Dict[int, np.ndarray]`。

### `predictor.predict_batch_by_smiles(control_expression, smiles_list, dosages, covariates)`
批量推理多种药物（SMILES）。返回 `Dict[str, np.ndarray]`。

### `predictor.predict_with_raw_embedding(control_expression, drug_embedding, dosage, covariates)`
直接传入 194 维嵌入向量推理（适合不在 parquet 中但你自行计算了 rdkit2D 特征的药物）。

### `predictor.smiles_to_idx(smiles)` / `predictor.idx_to_smiles(idx)`
SMILES ↔ idx 互转。

### `predictor.get_drug_embedding(drug_idx)`
获取指定药物的 194 维嵌入向量。

### `predictor.get_all_smiles()`
获取全部 17869 个 SMILES。

## 模型结构

- **Encoder**: 977 → [256]×4 → 32 (BN+ReLU)
- **Decoder**: 32 → [256]×4 → 1954 (decoder_activation=ReLU)
- **Drug embedding encoder**: 194 → [128]×4 → 32
- **Dosers**: amortized MLP, input=195(=194+1), output=1
- **Covariates**: 1 embedding (3 cell types)

## 依赖

```
torch>=1.10
numpy
pandas
```

## 注意事项

1. 模型参数在加载后自动冻结（`requires_grad=False`），所有推理方法使用 `@torch.no_grad()`
2. 预训练模型输入基因数为 **977** (LINCS shared genes)
3. 药物嵌入使用 **194维 rdkit2D** 特征，文件为 `rdkit2D_embedding_lincs_trapnell.parquet`
4. 如需作为更大模型的可微分组件，需在 `from_pretrained` 后手动取消参数冻结
