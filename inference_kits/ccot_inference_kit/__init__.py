"""
CCOT推理包装器 - 快速推理任意新数据的药物扰动效应

用法:
    from ccot_inference_kit.ccot_wrapper import CCOTInferenceWrapper
    
    model = CCOTInferenceWrapper(
        model_path='assets/CCOT_beta_10.0/best_model.pt',
        embedding_path='assets/rdkit2D_embedding_lincs_trapnell_chemCPA.parquet',
        unique_drug_list=your_drugs,
        device='cuda:0'
    )
    
    result = model.transport(control_data, drug_indices)
"""

from .ccot_wrapper import CCOTInferenceWrapper

__version__ = '1.0.0'
__all__ = ['CCOTInferenceWrapper']
