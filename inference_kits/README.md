# Inference Kits

本目录存放与训练主流程解耦的可复用推理封装。

- `ccot_inference_kit/`: CCOT 预训练模型的独立推理包装器
- `chemCPA_inference_kit/`: chemCPA 预训练模型的独立推理包装器

约定：

- 代码、测试脚本和说明文档保留在仓库中
- 体积较大的本地权重和 parquet 资产继续放在各自目录下，但默认不纳入 git 跟踪
- 如果只是临时调试某个推理想法，不要再放回 `tmp/`，应在对应 kit 内补充脚本或文档
