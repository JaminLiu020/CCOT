# CCOT Inference Kit - Assets 说明

本目录包含两个预训练的CCOT模型，分别对应不同的训练配置。

## 目录结构

```
assets/
├── CCOT_beta_10.0/          # 条件模型（SMILES embedding）
│   ├── best_model.pt
│   └── config_reference.yaml
├── CellOT/                  # 无条件模型（cell_type embedding, beta=0）
│   ├── best_model.pt
│   └── config_reference.yaml
└── rdkit2D_embedding_lincs_trapnell_chemCPA.parquet  # SMILES预训练嵌入
```

---

## 模型1: CCOT_beta_10.0（条件模型）

### 配置
- **embedding type**: `smiles` (32维 rdkit2D)
- **beta**: 10.0（强条件引导）
- **input_dim**: 977 (基因数)
- **input_dim_label**: 32 (SMILES嵌入维度)
- **用途**: 根据药物SMILES预测扰动效果

### 使用示例

```python
from ccot_wrapper import CCOTInferenceWrapper
import pandas as pd

# 1. 加载模型
wrapper = CCOTInferenceWrapper(
    model_path='assets/CCOT_beta_10.0/best_model.pt',
    embedding_path='assets/rdkit2D_embedding_lincs_trapnell_chemCPA.parquet',
    unique_drug_list=my_drug_list,  # 数据集中的所有SMILES列表
    beta=10.0,
    device='cuda:0'
)

# 2. 获取SMILES到索引的映射
smiles_to_index = wrapper.get_smiles_to_index()

# 3. 准备数据
control_cells = ...  # [N, 977] numpy array
drug_smiles = ['CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O', ...]  # N个SMILES
smiles_indices = [smiles_to_index[s] for s in drug_smiles]

# 4. 推理
transported = wrapper.transport(
    source_data=control_cells,
    smiles_indices=smiles_indices,
    return_numpy=True
)
```

### 特点
- ✓ 支持任意SMILES药物（如果在预训练embedding中）
- ✓ 强条件引导（beta=10.0）
- ✓ 需要提供SMILES条件输入

---

## 模型2: CellOT（无条件模型）

### 配置
- **embedding type**: `cell_type` (3维 one-hot)
- **beta**: 0.0（无条件推理）
- **input_dim**: 977 (基因数)
- **input_dim_label**: 3 (A549/K562/MCF7)
- **训练步数**: 1700 steps
- **最佳MMD**: 0.063637
- **用途**: 学习从control到treated的无条件映射

### 使用示例

```python
from ccot_wrapper import CCOTInferenceWrapper

# 1. 加载模型（跳过embedding加载）
wrapper = CCOTInferenceWrapper(
    model_path='assets/CellOT/best_model.pt',
    beta=0.0,
    device='cuda:0',
    skip_embedding=True  # 关键：跳过embedding加载
)

# 2. 检查是否为无条件模式
print(f"无条件模式: {wrapper.is_unconditional()}")  # True

# 3. 准备数据（只需要对照组数据）
control_cells = ...  # [N, 977] numpy array

# 4. 推理（无需提供条件输入）
transported = wrapper.transport(
    source_data=control_cells,
    smiles_indices=None,  # beta=0时不需要
    return_numpy=True
)
```

### 特点
- ✓ 无条件推理（beta=0）
- ✓ 不需要任何条件输入（SMILES或cell_type）
- ✓ 只学习control→treated的通用映射
- ✓ 更快的初始化（跳过embedding加载）
- ⚠️ 不区分不同药物/细胞类型

### 原理说明

当 `beta=0` 时：
```python
# CFG公式
transport = beta * transport_cond - (beta - 1) * transport_uncond
         = 0.0 * transport_cond - (0.0 - 1) * transport_uncond
         = transport_uncond

# PICNN.forward (enable_conditional_generation=False)
# 完全不使用条件输入y，只使用wx和wz层
z = wx(x)  # 无条件前向传播
```

因此：
- 条件输入（y）完全不参与计算
- embedding维度（3维 vs 32维）不影响推理
- 可以加载任何 `input_dim_label` 的模型，只要 `beta=0`

---

## 模型选择指南

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 预测特定药物的扰动效果 | CCOT_beta_10.0 | 强条件引导，区分不同药物 |
| 学习通用的control→treated映射 | CellOT | 无条件，不依赖特定条件 |
| 没有SMILES信息 | CellOT | 不需要条件输入 |
| 需要插值不同药物效果 | CCOT_beta_10.0 | 可调整beta值进行插值 |

---

## 技术细节

### Beta参数的作用

```python
if abs(beta - 1.0) < 1e-5:
    # beta=1: 纯条件推理
    transport = g.transport(x, condition, is_conditional_generation=True)
elif abs(beta) < 1e-5:
    # beta=0: 纯无条件推理
    transport = g.transport(x, condition, is_conditional_generation=False)
else:
    # beta>1: 分类器自由引导（Classifier-Free Guidance）
    transport_cond = g.transport(x, condition, is_conditional_generation=True)
    transport_uncond = g.transport(x, condition, is_conditional_generation=False)
    transport = beta * transport_cond - (beta - 1) * transport_uncond
```

### 网络架构（两个模型相同）

```
输入层: 977 (基因数)
隐藏层: [64, 64, 64, 64]
条件输入: 32 (CCOT_beta_10.0) / 3 (CellOT)
fnorm_penalty: g网络为1, f网络为0
```

---

## 兼容性说明

修改后的 `ccot_wrapper.py` 完全向后兼容：

- ✓ 原有的 `CCOT_beta_10.0` 模型正常使用
- ✓ 新增的 `CellOT` 模型（beta=0）可正常加载
- ✓ 自动检测beta值并调整推理逻辑
- ✓ 支持 `skip_embedding=True` 跳过embedding加载

---

## 测试

运行测试脚本验证两个模型：

```bash
conda activate cell
cd /home/jamin/github/ccot/inference_kits/ccot_inference_kit

# 测试CellOT模型（beta=0）
python test_beta0_model.py

# 测试CCOT_beta_10.0模型
python example_usage.py  # 如果有的话
```

---

## 常见问题

### Q1: 为什么CellOT训练时用cell_type但推理时不需要？

**A**: 因为 `beta=0` 时，PICNN的 `forward` 方法会完全忽略条件输入，只使用无条件路径（`wx`和`wz`层）。训练时的 `cell_type` 只是为了初始化网络架构，推理时不影响结果。

### Q2: 可以对CellOT模型设置beta>0吗？

**A**: 技术上可以，但由于模型训练时 `beta=0`，条件相关的权重（`w`, `wu`, `wxu`, `wzu`）没有被优化，设置 `beta>0` 会导致不可预测的结果。

### Q3: 两个模型的input_dim_label不同，会冲突吗？

**A**: 不会。Wrapper会从checkpoint自动推断 `input_dim_label`，并相应初始化网络。只要使用正确的checkpoint路径，就能正确加载。

---

## 更新日志

**2026-02-07**
- 添加 CellOT 模型（beta=0.0, cell_type embedding）
- 将 beta=0 checkpoint 整理到 `assets/CellOT/`
- 创建 `config_reference.yaml` 配置文件
- 训练信息：1700 steps, MinMMD=0.063637
