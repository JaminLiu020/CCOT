# CCOT推理包装器

快速加载和使用预训练的CCOT模型进行药物扰动效应预测。

在本仓库中，该套件位于 `inference_kits/ccot_inference_kit/`。

## 功能特点

- ✅ **快速初始化**: 2-3秒完成模型加载（原方案需30-60秒）
- ✅ **自动推断架构**: 从checkpoint自动提取网络结构
- ✅ **SMILES索引优化**: 保留原项目的高效索引方案
- ✅ **参数冻结**: 所有参数完全冻结，仅用于推理
- ✅ **灵活配置**: 支持beta、路径等参数覆盖
- ✅ **自包含**: 无需依赖原项目代码，可直接拷贝使用

## 文件结构

```
ccot_inference_kit/
├── ccot_wrapper.py          # 核心推理类
├── requirements.txt         # 依赖
├── assets/
│   ├── CCOT_beta_10.0/
│   │   ├── best_model.pt        # 预训练权重（977维基因）
│   │   └── config_reference.yaml # 配置文件
│   └── rdkit2D_embedding_lincs_trapnell_chemCPA.parquet  # SMILES嵌入
├── networks/
│   ├── __init__.py
│   ├── picnn.py             # PICNN网络
│   ├── layers.py            # 网络层
│   └── embedding.py         # 嵌入层（精简版）
├── example_usage.py         # 使用示例
└── README.md                # 本文件
```

## 安装依赖

```bash
pip install -r requirements.txt
```

依赖项：
- torch>=1.10.0
- pandas>=1.3.0
- pyarrow>=6.0.0
- numpy>=1.21.0

## 快速开始

### 基本用法

```python
from ccot_wrapper import CCOTInferenceWrapper
import pandas as pd

# 1. 从数据集获取药物列表
df = pd.read_csv('your_data.csv')
unique_drugs = df['SMILES'].drop_duplicates().tolist()

# 2. 初始化CCOT推理器
model = CCOTInferenceWrapper(
    model_path='assets/CCOT_beta_10.0/best_model.pt',
    embedding_path='assets/rdkit2D_embedding_lincs_trapnell_chemCPA.parquet',
    unique_drug_list=unique_drugs,  # 过滤嵌入表，加速查询
    beta=10.0,                      # Classifier-free guidance参数
    device='cuda:0'
)

# 3. 获取SMILES索引映射（用于构造DataLoader）
smiles_to_index = model.get_smiles_to_index()

# 4. 准备数据（假设已从DataLoader获取）
control_cells = ...           # [N, 977] 对照组基因表达
drug_indices = [smiles_to_index[smiles] for smiles in drug_smiles_list]

# 5. 执行推理
predicted_treated = model.transport(control_cells, drug_indices)
# 返回: [N, 977] 预测的扰动后基因表达
```

### 覆盖beta参数

```python
# 使用不同的beta值
result_beta5 = model.transport(control_cells, drug_indices, beta=5.0)
result_beta1 = model.transport(control_cells, drug_indices, beta=1.0)  # 无guidance
```

## 详细示例

查看 [`example_usage.py`](example_usage.py) 获取完整示例，包括：

1. **场景1**: 基本推理
2. **场景2**: 覆盖beta参数
3. **场景3**: 批量推理（多个药物）
4. **场景4**: 错误处理

运行示例：

```bash
python example_usage.py
```

## API文档

### `CCOTInferenceWrapper`

#### 初始化参数

- `model_path` (str): CCOT权重文件路径
- `embedding_path` (str): 预训练SMILES嵌入parquet文件路径
- `unique_drug_list` (list, optional): 数据集中包含的SMILES列表（用于过滤嵌入表）
- `beta` (float): Classifier-free guidance参数，默认10.0
- `device` (str): 推理设备，默认'cuda:0'

#### 方法

##### `get_smiles_to_index()`

返回SMILES到索引的映射字典，供构造DataLoader使用。

**返回:**
- `Dict[str, int]`: SMILES → 索引映射

##### `transport(source_data, smiles_indices, beta=None, return_numpy=True)`

执行转运映射预测。

**参数:**
- `source_data` (numpy.ndarray or torch.Tensor): 对照组基因表达 [N, 977]
- `smiles_indices` (list or torch.Tensor): 已转为索引的药物列表 [N]
- `beta` (float, optional): 覆盖默认beta值
- `return_numpy` (bool): 是否返回numpy数组，默认True

**返回:**
- `numpy.ndarray or torch.Tensor`: 预测的扰动后基因表达 [N, 977]

**异常:**
- `ValueError`: 当source_data和smiles_indices长度不匹配时
- `KeyError`: 当SMILES不在预训练嵌入中时

## 与原评估流程的对比

| 特性 | 原evaluate.py | CCOTInferenceWrapper |
|------|-------------|---------------------|
| 初始化时间 | 30-60秒 | 2-3秒 |
| 代码行数 | ~500行 | ~250行 |
| 依赖项数量 | 15+ | 4个 |
| 需要h5ad数据 | ✓ | ✗ |
| 需要DataLoader | ✓ | ✗ |
| transform_matrix | 包含但未使用 | 完全移除 |
| SMILES索引优化 | ✓ | ✓ |

## 工作流程

```
初始化阶段:
1. 加载SMILES嵌入 → 用unique_drug_list过滤 → 构建smiles_to_index
2. 从checkpoint推断网络架构（input_dim, hidden_units等）
3. 初始化f、g网络（都使用PICNN）
4. 加载预训练权重
5. 冻结所有参数

推理阶段:
1. 接收control cells + drug indices
2. 通过smiles_to_index查询嵌入
3. 执行g.transport()（带或不带classifier-free guidance）
4. 返回预测的treated cells
```

## 注意事项

1. **SMILES索引**: 必须使用`smiles_to_index`映射后的索引，而不是原始SMILES字符串
2. **数据维度**: source_data必须是[N, 977]维度（与训练时的基因数一致）
3. **设备匹配**: 确保有足够的GPU内存，或使用CPU（device='cpu'）
4. **beta参数**: 推荐使用10.0（与原训练一致），调整可能影响预测质量

## 迁移到新项目

整个 `ccot_inference_kit/` 文件夹是自包含的，可以直接拷贝到新项目中使用：

```bash
# 拷贝到新项目
cp -r inference_kits/ccot_inference_kit /path/to/your/new/project/

# 在新项目中使用
from ccot_inference_kit.ccot_wrapper import CCOTInferenceWrapper
```

## 故障排除

### 问题1: CUDA out of memory

**解决方案**: 减小batch_size或使用CPU

```python
model = CCOTInferenceWrapper(device='cpu', ...)
```

### 问题2: SMILES不在嵌入表中

**原因**: 该SMILES未在unique_drug_list中

**解决方案**: 确保所有推理数据的SMILES都在初始化时的unique_drug_list中

### 问题3: 维度不匹配

**原因**: source_data维度不是977

**解决方案**: 确保基因表达数据与训练时一致（977个基因）

## 技术细节

- **网络架构**: f和g都是PICNN（Partially Input Convex Neural Network）
- **嵌入维度**: 32维（rdkit2D）
- **基因数**: 977维（LINCS genes）
- **优化器**: Adam（仅训练时使用，推理不需要）
- **Beta范围**: 推荐1.0-10.0，更高的beta增强条件引导

## 许可证

与原CCOT项目保持一致。

## 问题反馈

如有问题，请联系原项目维护者。
