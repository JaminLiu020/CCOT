# 使用 beta=0 模型进行无条件推理

## 背景

当训练CCOT模型时设置 `beta=0.0`，模型会进行**无条件推理**（unconditional inference），此时：
- 条件输入（如SMILES embedding或cell_type）完全不被使用
- 只使用 `wx` 和 `wz` 层（输入和隐藏层）
- 不涉及 `w`, `wu`, `wxu`, `wzu` 等条件相关的层

因此，即使模型训练时使用了不同的embedding类型（如 `cell_type` 而非 `smiles`），只要 `beta=0`，就可以直接加载并推理。

## 关键修改

已对 `ccot_wrapper.py` 进行以下修改：

1. **新增 `skip_embedding` 参数**: 允许在 `beta=0` 时跳过embedding加载
2. **允许 `smiles_indices=None`**: 当 `beta=0` 时不需要提供条件输入
3. **自动使用dummy条件向量**: 在 `beta=0` 时自动构造零向量（不会被使用）
4. **新增 `is_unconditional()` 方法**: 检查是否为无条件推理模式

## 使用方法

### 加载 beta=0 模型

```python
from ccot_wrapper import CCOTInferenceWrapper

# 初始化wrapper（跳过embedding加载）
wrapper = CCOTInferenceWrapper(
    model_path='/path/to/your/best_model.pt',
    beta=0.0,                      # 设置beta=0.0
    device='cuda:0',
    skip_embedding=True            # 跳过embedding加载
)
```

### 执行推理

```python
import numpy as np

# 准备对照组基因表达数据 [N, 977]
control_cells = np.random.randn(100, 977).astype(np.float32)

# 执行无条件推理（无需提供smiles_indices）
transported = wrapper.transport(
    source_data=control_cells,
    smiles_indices=None,           # beta=0时不需要
    return_numpy=True
)

print(f"输出形状: {transported.shape}")  # (100, 977)
```

### 检查模式

```python
# 检查是否为无条件模式
if wrapper.is_unconditional():
    print("当前为无条件推理模式")
```

## 模型兼容性

### 你的新模型 (`results/.../beta_0.0/`)

- **embedding type**: `cell_type` (3维 one-hot)
- **input_dim**: 977 (基因数)
- **input_dim_label**: 3 (A549/K562/MCF7)
- **beta**: 0.0
- **✓ 可以使用修改后的代码加载并推理**

### 原inference kit模型 (`assets/CCOT_beta_10.0/best_model.pt`)

- **embedding type**: `smiles` (32维 rdkit2D)
- **input_dim**: 977
- **input_dim_label**: 32
- **beta**: 10.0 (默认)
- **✓ 仍然可以正常使用（向后兼容）**

## 测试验证

运行测试脚本验证功能：

```bash
conda activate cell
cd /home/jamin/github/ccot/inference_kits/ccot_inference_kit
python test_beta0_model.py
```

测试结果：
- ✓ 成功加载 beta=0 模型
- ✓ input_dim_label=3 (cell_type embedding)
- ✓ 无条件推理成功
- ✓ 输出维度正确 (10, 977)

## 原理说明

### Beta=0 时的推理逻辑

```python
# ccot_wrapper.py 中的 transport 方法
if abs(beta) < 1e-5:
    # beta=0: 只执行 unconditional transport
    transport = g.transport(
        source_data, 
        dummy_condition,  # 会被忽略
        is_conditional_generation=False
    )
```

### PICNN.forward 的 enable_conditional_generation=False 分支

```python
# networks/picnn.py
def forward(self, x, y, enable_conditional_generation=True):
    if enable_conditional_generation:
        # 使用条件输入y，涉及 w, wu, wxu, wzu 等层
        ...
    else:
        # 完全不使用y，只使用 wx 和 wz 层
        for i in range(self.n_layers):
            if i == 0:
                z = self.sigma(0.2)(self.wx[i](x))
                z = z * z
            else:
                z = self.sigma(0.2)(self.wz[i - 1](z) + self.wx[i](x))
        return self.wz[-1](z) + self.wx[-1](x)
```

**关键**: 当 `enable_conditional_generation=False` 时，y 参数完全不参与计算，因此其维度和内容都无关紧要。

## 注意事项

1. **只适用于 beta=0 的模型**: 如果模型训练时 beta>0，推理时必须提供正确的条件输入
2. **embedding维度不影响**: 只要 beta=0，任何 input_dim_label 都可以正常工作
3. **性能**: 跳过embedding加载可以加快初始化速度

## 总结

你的观察完全正确！当 `beta=0` 时：
- ✓ 不需要条件输入
- ✓ embedding维度不重要
- ✓ 可以直接加载并推理

修改后的代码已经支持这种用法，测试也验证了功能正常。
