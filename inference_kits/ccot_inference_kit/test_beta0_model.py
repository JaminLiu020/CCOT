#!/usr/bin/env python3
"""
测试加载 beta=0.0 的新模型（cell_type embedding）
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加当前目录到路径
kit_dir = Path(__file__).parent
sys.path.insert(0, str(kit_dir))

from ccot_wrapper import CCOTInferenceWrapper


def _resolve_model_path() -> Path:
    env_path = os.environ.get("CCOT_BETA0_MODEL_PATH")
    if env_path:
        return Path(env_path)
    return kit_dir / "assets" / "CellOT" / "best_model.pt"


def test_load_beta0_model():
    """测试加载并推理 beta=0 的新模型"""
    
    print("=" * 80)
    print("测试加载 beta=0.0 的新模型（cell_type embedding）")
    print("=" * 80)
    
    # 模型路径
    model_path = _resolve_model_path()
    
    # 初始化wrapper（跳过embedding加载，因为是无条件推理）
    print("\n【步骤1】初始化CCOTInferenceWrapper")
    wrapper = CCOTInferenceWrapper(
        model_path=model_path,
        beta=0.0,                      # 设置beta=0.0
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        skip_embedding=True            # 跳过embedding加载
    )
    
    print(f"\n【步骤2】验证模型信息")
    print(f"  - 模型是否为无条件模式: {wrapper.is_unconditional()}")
    print(f"  - g网络架构: input_dim={wrapper.g.input_dim}, input_dim_label={wrapper.g.input_dim_label}")
    print(f"  - g网络隐藏层数: {wrapper.g.n_layers}")
    
    # 创建测试数据
    print(f"\n【步骤3】创建测试数据")
    batch_size = 10
    n_genes = 977
    
    # 随机生成对照组基因表达数据
    source_data = np.random.randn(batch_size, n_genes).astype(np.float32)
    print(f"  - source_data shape: {source_data.shape}")
    
    # 执行推理（无需提供smiles_indices，因为beta=0）
    print(f"\n【步骤4】执行无条件推理")
    try:
        transported = wrapper.transport(
            source_data=source_data,
            smiles_indices=None,        # beta=0时不需要
            return_numpy=True
        )
        print(f"  ✓ 推理成功！")
        print(f"  - transported shape: {transported.shape}")
        print(f"  - transported mean: {transported.mean():.6f}")
        print(f"  - transported std: {transported.std():.6f}")
        
        # 验证输出维度
        assert transported.shape == source_data.shape, f"Shape mismatch: {transported.shape} != {source_data.shape}"
        print(f"  ✓ 输出维度验证通过")
        
    except Exception as e:
        print(f"  ✗ 推理失败: {e}")
        raise
    
    print("\n" + "=" * 80)
    print("✓ 测试通过！新模型可以成功加载并推理")
    print("=" * 80)


def test_load_checkpoint_details():
    """详细检查checkpoint内容"""
    
    print("\n" + "=" * 80)
    print("附加测试：检查checkpoint详细信息")
    print("=" * 80)
    
    model_path = _resolve_model_path()
    
    ckpt = torch.load(model_path, map_location='cpu')
    
    print("\n【Checkpoint结构】")
    print(f"  - Keys: {list(ckpt.keys())}")
    print(f"  - Step: {ckpt.get('step', 'N/A')}")
    print(f"  - MinMMD: {ckpt.get('minmmd', 'N/A')}")
    
    f_state = ckpt['f_state']
    g_state = ckpt['g_state']
    
    print("\n【网络架构参数推断】")
    print(f"  - input_dim (基因数): {f_state['wx.0.weight'].shape[1]}")
    print(f"  - input_dim_label (embedding维度): {f_state['w.0.weight'].shape[1]}")
    print(f"  - hidden_units:")
    for i in range(5):
        if f'wx.{i}.weight' in f_state:
            print(f"    - layer {i}: {f_state[f'wx.{i}.weight'].shape[0]}")
    
    print("\n【条件相关层的权重统计】")
    print("  这些层在 beta=0 时不会被使用：")
    print(f"  - w.0.weight (条件编码): shape={f_state['w.0.weight'].shape}, mean={f_state['w.0.weight'].mean():.6f}")
    print(f"  - wzu.0.weight (条件门控): shape={f_state['wzu.0.weight'].shape}, mean={f_state['wzu.0.weight'].mean():.6f}")
    
    print("\n【无条件路径层的权重统计】")
    print("  这些层在 beta=0 时会被使用：")
    print(f"  - wx.0.weight (输入层): shape={f_state['wx.0.weight'].shape}, mean={f_state['wx.0.weight'].mean():.6f}")
    print(f"  - wz.0.weight (隐藏层): shape={f_state['wz.0.weight'].shape}, mean={f_state['wz.0.weight'].mean():.6f}")


if __name__ == "__main__":
    # 测试加载和推理
    test_load_beta0_model()
    
    # 检查checkpoint详情
    test_load_checkpoint_details()
