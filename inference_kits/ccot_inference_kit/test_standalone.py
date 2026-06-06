#!/usr/bin/env python3
"""
CCOT推理包装器 - 独立测试脚本

此脚本可以从任何位置运行，会自动查找ccot_inference_kit包
使用方法:
    python test_standalone.py
或
    python /path/to/ccot_inference_kit/test_standalone.py
"""

import sys
import os
from pathlib import Path

# 获取脚本所在的ccot_inference_kit目录
kit_dir = Path(__file__).parent
parent_dir = kit_dir.parent

# 添加到Python路径
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    print("=" * 60)
    print("CCOT推理包装器测试")
    print("=" * 60)
    
    print(f"\n系统信息:")
    print(f"  Python版本: {sys.version.split()[0]}")
    print(f"  工作目录: {os.getcwd()}")
    print(f"  包位置: {kit_dir}")
    
    print("\n[1/5] 检查文件...")
    required_files = [
        'ccot_wrapper.py',
        'networks/picnn.py',
        'networks/layers.py',
        'networks/embedding.py',
        'assets/CCOT_beta_10.0/best_model.pt',
        'assets/rdkit2D_embedding_lincs_trapnell_chemCPA.parquet'
    ]
    for fname in required_files:
        fpath = kit_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024*1024)
            print(f"  ✓ {fname:<50} ({size_mb:.1f}MB)")
        else:
            print(f"  ✗ {fname:<50} (缺失)")
            raise FileNotFoundError(f"缺少文件: {fname}")
    
    print("\n[2/5] 检查依赖...")
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        raise
    
    try:
        import pandas
        print(f"  ✓ pandas {pandas.__version__}")
    except ImportError as e:
        print(f"  ✗ pandas: {e}")
        raise
    
    try:
        import numpy
        print(f"  ✓ numpy {numpy.__version__}")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        raise
    
    print("\n[3/5] 导入模块...")
    from ccot_inference_kit.ccot_wrapper import CCOTInferenceWrapper
    import numpy as np
    print("  ✓ 模块导入成功")
    
    print("\n[4/5] 初始化模型...")
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    print(f"  使用设备: {device}")
    
    model = CCOTInferenceWrapper(
        model_path=str(kit_dir / 'assets/CCOT_beta_10.0/best_model.pt'),
        embedding_path=str(kit_dir / 'assets/rdkit2D_embedding_lincs_trapnell_chemCPA.parquet'),
        unique_drug_list=None,  # 加载全部嵌入
        beta=10.0,
        device=device
    )
    print("  ✓ 模型初始化成功")
    
    print("\n[5/5] 执行推理测试...")
    smiles_to_index = model.get_smiles_to_index()
    print(f"  嵌入表大小: {len(smiles_to_index)} 个药物")
    
    # 准备测试数据
    test_control = np.random.randn(5, 977).astype(np.float32)
    test_indices = [0] * 5
    
    result = model.transport(test_control, test_indices)
    print(f"  ✓ 推理成功")
    print(f"    输入形状: {test_control.shape}")
    print(f"    输出形状: {result.shape}")
    print(f"    输出范围: [{result.min():.3f}, {result.max():.3f}]")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！CCOT推理包装器工作正常。")
    print("=" * 60)
    print("\n提示:")
    print("  - 快速开始: 查看 README.md")
    print("  - 完整示例: 运行 example_usage.py")
    print("  - 导入方式: from ccot_inference_kit import CCOTInferenceWrapper")
    
except Exception as e:
    print(f"\n✗ 测试失败: {type(e).__name__}")
    print(f"  错误信息: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
