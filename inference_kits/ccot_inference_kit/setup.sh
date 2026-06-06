#!/bin/bash
# CCOT推理包装器 - 快速使用指南

set -e

echo "======================================"
echo "CCOT推理包装器部署脚本"
echo "======================================"

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "脚本位置: $SCRIPT_DIR"

# 检查文件
echo ""
echo "[1/3] 检查文件..."
for file in ccot_wrapper.py networks/picnn.py networks/embedding.py networks/layers.py __init__.py; do
    if [ -f "$SCRIPT_DIR/$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ 缺失: $file"
        exit 1
    fi
done

for file in assets/CCOT_beta_10.0/best_model.pt assets/rdkit2D_embedding_lincs_trapnell_chemCPA.parquet; do
    if [ -f "$SCRIPT_DIR/$file" ]; then
        size=$(du -h "$SCRIPT_DIR/$file" | cut -f1)
        echo "  ✓ $file ($size)"
    else
        echo "  ✗ 缺失: $file"
        exit 1
    fi
done

# 检查Python依赖
echo ""
echo "[2/3] 检查Python依赖..."
python -c "import torch; print(f'  ✓ torch {torch.__version__}')"
python -c "import pandas; print(f'  ✓ pandas {pandas.__version__}')"
python -c "import numpy; print(f'  ✓ numpy {numpy.__version__}')"

# 运行测试
echo ""
echo "[3/3] 运行测试..."
cd "$SCRIPT_DIR"
python test_standalone.py

echo ""
echo "======================================"
echo "✓ 部署检查完成！"
echo "======================================"
echo ""
echo "使用方法:"
echo "  1. 导入模块:"
echo "     from ccot_inference_kit import CCOTInferenceWrapper"
echo ""
echo "  2. 初始化模型:"
echo "     model = CCOTInferenceWrapper("
echo "         model_path='path/to/inference_kits/ccot_inference_kit/assets/CCOT_beta_10.0/best_model.pt',"
echo "         embedding_path='path/to/inference_kits/ccot_inference_kit/assets/rdkit2D_embedding_*.parquet',"
echo "         unique_drug_list=your_drugs,"
echo "         device='cuda:0'"
echo "     )"
echo ""
echo "  3. 执行推理:"
echo "     result = model.transport(control_data, drug_indices)"
echo ""
echo "详见: README.md"
