"""
chemCPA 推理器 - 核心推理接口。

加载预训练的 chemCPA 模型，冻结参数，用于预测扰动后的基因表达。
完全复刻原始 chemCPA 的推理逻辑（compute_prediction → 取均值）。

药物指定方式（二选一）：
- drug_idx: 药物在嵌入文件（parquet, 17869种）中的行索引
- smiles: 药物 SMILES 字符串（自动从嵌入文件查找行索引）

无论哪种方式，嵌入都从 parquet 查找表获取，再经
drug_embedding_encoder + dosers 网络处理。
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Dict

from .model import ComPert
from .embedding import load_drug_embedding


class ChemCPAPredictor:
    """
    chemCPA 预测器：加载预训练模型，推理出扰动后的基因表达。

    药物通过在嵌入文件（17869种）中的索引或 SMILES 指定，
    内部统一从 parquet 查嵌入 → drug_embedding_encoder + dosers 管线推理。

    使用示例::

        predictor = ChemCPAPredictor.from_pretrained(
            model_path="pretrained/manual_2025-03-08_03-24-40.pt",
            embedding_path="pretrained/rdkit2D_embedding_lincs_trapnell.parquet",
        )

        # 通过药物在嵌入文件中的索引推理
        prediction = predictor.predict(control_genes, drug_idx=42, dosage=1.0)

        # 通过 SMILES 推理
        prediction = predictor.predict_by_smiles(control_genes, smiles="CCO", dosage=1.0)
    """

    def __init__(
        self,
        model: ComPert,
        drug_embeddings_table: np.ndarray,
        smiles_list: List[str],
        device: str = "cpu",
    ):
        """
        低级构造函数，通常使用 from_pretrained() 工厂方法。

        Args:
            model: 已加载权重的 ComPert 模型
            drug_embeddings_table: 完整药物嵌入查找表 [num_drugs_total, emb_dim]，
                                   来自 parquet 文件 (17869×194)
            smiles_list: 对应的 SMILES 列表（与 drug_embeddings_table 行顺序一致）
            device: 推理设备
        """
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.device = device

        # 药物嵌入查找表（来自 parquet，17869×194），常驻 GPU
        self._drug_table = torch.tensor(
            drug_embeddings_table, dtype=torch.float32, device=device
        )
        self._smiles_list = smiles_list
        self._smiles_to_idx = {s: i for i, s in enumerate(smiles_list)}
        self.num_genes = model.num_genes
        self.num_drugs_total = len(smiles_list)  # 17869
        self._embedding_dim = drug_embeddings_table.shape[1]  # 194

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        embedding_path: str,
        device: str = None,
    ) -> "ChemCPAPredictor":
        """
        从预训练 checkpoint 加载推理器。

        Args:
            model_path: 预训练模型 .pt 文件路径
            embedding_path: 药物嵌入 parquet 文件路径 (17869×194 rdkit2D)
            device: 推理设备。None 时自动选择 cuda/cpu

        Returns:
            ChemCPAPredictor 实例
        """
        import pandas as pd

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---- 加载 checkpoint ----
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if not isinstance(checkpoint, (list, tuple)):
            raise ValueError(f"不支持的 checkpoint 格式: {type(checkpoint)}")

        if len(checkpoint) == 3:
            state_dict, model_config, history = checkpoint
            cov_embeddings_state_dicts = []
        elif len(checkpoint) == 4:
            state_dict, _, model_config, history = checkpoint
            cov_embeddings_state_dicts = []
        elif len(checkpoint) == 5:
            state_dict, _, cov_embeddings_state_dicts, model_config, history = checkpoint
        else:
            raise ValueError(f"无法解析的 checkpoint 格式，包含 {len(checkpoint)} 个元素")

        # ---- 解析模型配置 ----
        num_genes = model_config["num_genes"]
        num_drugs = model_config["num_drugs"]
        num_covariates = model_config["num_covariates"]
        doser_type = model_config.get("doser_type", "amortized")
        decoder_activation = model_config.get("decoder_activation", "ReLU")
        use_drugs_idx = bool(model_config.get("use_drugs_idx", True))
        hparams = model_config.get("hparams", {})

        # ---- 加载 parquet 嵌入文件 (17869×194) ----
        embedding_path = Path(embedding_path)
        assert embedding_path.exists(), f"嵌入文件不存在: {embedding_path}"

        df = pd.read_parquet(str(embedding_path))
        smiles_list = df.index.tolist()
        drug_embeddings_table = df.values.astype(np.float32)  # (17869, 194)

        # ---- 构建 nn.Embedding（仅用于保持 state_dict 键兼容） ----
        if "drug_embeddings.weight" in state_dict:
            emb_weight = state_dict["drug_embeddings.weight"].to(device)
            drug_embeddings = torch.nn.Embedding.from_pretrained(emb_weight, freeze=True)
        else:
            drug_embeddings = torch.nn.Embedding(num_drugs, 194).to(device)

        # ---- 构建模型 ----
        model = ComPert(
            num_genes=num_genes,
            num_drugs=num_drugs,
            num_covariates=num_covariates,
            device=device,
            seed=model_config.get("seed", 0),
            patience=model_config.get("patience", 5),
            doser_type=doser_type,
            decoder_activation=decoder_activation,
            hparams=hparams,
            drug_embeddings=drug_embeddings,
            use_drugs_idx=use_drugs_idx,
            append_layer_width=None,
            enable_cpa_mode=False,
        )

        # 加载权重 (strict=False 因为 adversary 等可能缺失)
        incomp = model.load_state_dict(state_dict, strict=False)
        missing_important = [
            k for k in incomp.missing_keys
            if not k.startswith("adversary") and k != "drug_embeddings.weight"
        ]
        if missing_important:
            print(f"[ChemCPAPredictor 警告] 以下推理相关权重未加载: {missing_important}")

        # 加载协变量嵌入
        if cov_embeddings_state_dicts:
            for emb, sd in zip(model.covariates_embeddings, cov_embeddings_state_dicts):
                emb.load_state_dict(sd)

        emb_dim = drug_embeddings_table.shape[1]
        print(f"[ChemCPAPredictor] 模型加载成功:")
        print(f"  num_genes={num_genes}, num_covariates={num_covariates}")
        print(f"  decoder_activation={decoder_activation}, doser_type={doser_type}")
        print(f"  嵌入文件药物数={len(smiles_list)}, embedding_dim={emb_dim}")
        print(f"  hparams.dim={hparams.get('dim', '?')}, device={device}")

        return cls(
            model=model,
            drug_embeddings_table=drug_embeddings_table,
            smiles_list=smiles_list,
            device=device,
        )

    # ------------------------------------------------------------------
    #  核心推理：统一通过 parquet 嵌入 → drug_embedding_encoder + doser
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_from_embedding(
        self,
        genes: torch.Tensor,
        drug_embedding: torch.Tensor,
        dosage_val: torch.Tensor,
        covariates: Optional[List[torch.Tensor]] = None,
    ) -> np.ndarray:
        """
        内部方法：给定已准备好的张量，执行前向推理。

        复刻 chemCPA 推理路径：
        1. encoder(genes) → latent_basal
        2. drug_embedding → drug_embedding_encoder → encoded_drugs
        3. [drug_embedding, dosage] → dosers → scaled_dosages
        4. drug_effect = scaled_dosages * encoded_drugs
        5. latent_treated = latent_basal + drug_effect (+ covariates)
        6. decoder(latent_treated) → 取前半部分（均值）
        """
        latent_basal = self.model.encoder(genes)

        # Amortized doser
        if self.model.doser_type == "amortized":
            doser_input = torch.cat(
                [drug_embedding, dosage_val.unsqueeze(-1)], dim=1
            )
            scaled_dosages = self.model.dosers(doser_input).squeeze(-1)
            if scaled_dosages.dim() == 0:
                scaled_dosages = scaled_dosages.unsqueeze(0)
        else:
            scaled_dosages = dosage_val

        # Drug embedding encoder
        if not self.model.enable_cpa_mode and self.model.drug_embedding_encoder is not None:
            encoded_drugs = self.model.drug_embedding_encoder(drug_embedding)
        else:
            encoded_drugs = drug_embedding

        # drug_effect = scaled_dosages * encoded_drugs
        drug_effect = torch.einsum("b,be->be", [scaled_dosages, encoded_drugs])

        latent_treated = latent_basal + drug_effect

        # 协变量
        if covariates is not None and self.model.num_covariates[0] > 0:
            for cov_type, emb_cov in enumerate(self.model.covariates_embeddings):
                emb_cov = emb_cov.to(self.device)
                cov_idx = covariates[cov_type].to(self.device).argmax(1)
                latent_treated = latent_treated + emb_cov(cov_idx)

        gene_reconstructions = self.model.decoder(latent_treated)

        dim = gene_reconstructions.size(1) // 2
        mean_pred = gene_reconstructions[:, :dim]

        return mean_pred.cpu().numpy()

    # ------------------------------------------------------------------
    #  公开 API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        control_expression: Union[torch.Tensor, np.ndarray],
        drug_idx: Union[int, List[int], torch.Tensor, np.ndarray],
        dosage: Union[float, torch.Tensor, np.ndarray] = 1.0,
        covariates: Optional[List[torch.Tensor]] = None,
    ) -> np.ndarray:
        """
        通过药物在嵌入文件中的索引预测扰动后的基因表达均值。

        drug_idx 是药物在 parquet 嵌入文件（17869种）中的行索引，
        嵌入从 parquet 查找表获取后经 drug_embedding_encoder + dosers 处理。

        Args:
            control_expression: 对照组基因表达 [batch_size, num_genes] 或 [num_genes]
            drug_idx: 药物索引 (int) 或 batch 药物索引 [batch_size]
                      索引范围 [0, 17868]
            dosage: 药物剂量 (float) 或 batch 剂量 [batch_size]
            covariates: 协变量 one-hot 编码列表

        Returns:
            prediction: 预测均值 [batch_size, num_genes]
        """
        # --- control_expression ---
        if isinstance(control_expression, np.ndarray):
            control_expression = torch.tensor(control_expression, dtype=torch.float32)
        if control_expression.dim() == 1:
            control_expression = control_expression.unsqueeze(0)
        batch_size = control_expression.shape[0]
        genes = control_expression.to(self.device)

        # --- drug_embedding: 从 parquet 查找表取 ---
        if isinstance(drug_idx, (int, np.integer)):
            emb = self._drug_table[drug_idx].unsqueeze(0).expand(batch_size, -1)
        elif isinstance(drug_idx, (list, np.ndarray)):
            idx_tensor = torch.tensor(drug_idx, dtype=torch.long, device=self.device)
            emb = self._drug_table[idx_tensor]
        elif isinstance(drug_idx, torch.Tensor):
            emb = self._drug_table[drug_idx.to(self.device)]
        else:
            raise TypeError(f"不支持的 drug_idx 类型: {type(drug_idx)}")

        # --- dosage ---
        if isinstance(dosage, (float, int, np.floating, np.integer)):
            dosage_val = torch.full((batch_size,), float(dosage),
                                   dtype=torch.float32, device=self.device)
        elif isinstance(dosage, np.ndarray):
            dosage_val = torch.tensor(dosage, dtype=torch.float32, device=self.device)
        elif isinstance(dosage, torch.Tensor):
            dosage_val = dosage.to(self.device)
        else:
            raise TypeError(f"不支持的 dosage 类型: {type(dosage)}")

        return self._predict_from_embedding(genes, emb, dosage_val, covariates)

    @torch.no_grad()
    def predict_by_smiles(
        self,
        control_expression: Union[torch.Tensor, np.ndarray],
        smiles: str,
        dosage: float = 1.0,
        covariates: Optional[List[torch.Tensor]] = None,
    ) -> np.ndarray:
        """
        通过 SMILES 字符串预测。自动从嵌入文件查找索引。

        Args:
            control_expression: 对照组基因表达 [batch_size, num_genes]
            smiles: 药物 SMILES 字符串（须在嵌入文件的 17869 种中）
            dosage: 剂量
            covariates: 协变量列表

        Returns:
            prediction: 预测均值 [batch_size, num_genes]
        """
        if smiles not in self._smiles_to_idx:
            raise ValueError(
                f"SMILES '{smiles}' 不在嵌入文件中 (共 {self.num_drugs_total} 种药物)。\n"
                f"请确保 SMILES 格式正确（canonical SMILES）。"
            )
        drug_idx = self._smiles_to_idx[smiles]
        return self.predict(control_expression, drug_idx, dosage, covariates)

    @torch.no_grad()
    def predict_batch(
        self,
        control_expression: Union[torch.Tensor, np.ndarray],
        drug_indices: List[int],
        dosages: Optional[List[float]] = None,
        covariates: Optional[List[torch.Tensor]] = None,
    ) -> Dict[int, np.ndarray]:
        """
        批量预测：对同一组对照数据，分别施加不同药物扰动。

        Args:
            control_expression: 对照组基因表达 [batch_size, num_genes]
            drug_indices: 药物索引列表（嵌入文件中的行索引）
            dosages: 每种药物的剂量列表（默认全为1.0）
            covariates: 协变量列表

        Returns:
            results: {drug_idx: prediction_array} 字典
        """
        if dosages is None:
            dosages = [1.0] * len(drug_indices)

        results = {}
        for drug_idx, dose in zip(drug_indices, dosages):
            results[drug_idx] = self.predict(
                control_expression, drug_idx, dose, covariates
            )
        return results

    @torch.no_grad()
    def predict_batch_by_smiles(
        self,
        control_expression: Union[torch.Tensor, np.ndarray],
        smiles_list: List[str],
        dosages: Optional[List[float]] = None,
        covariates: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        批量预测：通过 SMILES 列表分别施加不同药物扰动。

        Args:
            control_expression: 对照组基因表达 [batch_size, num_genes]
            smiles_list: 药物 SMILES 列表
            dosages: 对应的剂量列表
            covariates: 协变量列表

        Returns:
            results: {smiles: prediction_array} 字典
        """
        if dosages is None:
            dosages = [1.0] * len(smiles_list)

        results = {}
        for smi, dose in zip(smiles_list, dosages):
            results[smi] = self.predict_by_smiles(
                control_expression, smi, dose, covariates
            )
        return results

    @torch.no_grad()
    def predict_with_raw_embedding(
        self,
        control_expression: Union[torch.Tensor, np.ndarray],
        drug_embedding: Union[torch.Tensor, np.ndarray],
        dosage: float = 1.0,
        covariates: Optional[List[torch.Tensor]] = None,
    ) -> np.ndarray:
        """
        直接使用 194 维药物嵌入向量进行推理（绕过查找表）。

        适用于自行计算了 rdkit2D 特征、不在嵌入文件中的药物。

        Args:
            control_expression: 对照组基因表达 [batch_size, num_genes]
            drug_embedding: 药物嵌入 [emb_dim] 或 [batch_size, emb_dim]
            dosage: 剂量
            covariates: 协变量列表

        Returns:
            prediction: 预测均值 [batch_size, num_genes]
        """
        if isinstance(control_expression, np.ndarray):
            control_expression = torch.tensor(control_expression, dtype=torch.float32)
        if isinstance(drug_embedding, np.ndarray):
            drug_embedding = torch.tensor(drug_embedding, dtype=torch.float32)
        if control_expression.dim() == 1:
            control_expression = control_expression.unsqueeze(0)
        if drug_embedding.dim() == 1:
            drug_embedding = drug_embedding.unsqueeze(0).expand(
                control_expression.shape[0], -1
            )

        batch_size = control_expression.shape[0]
        genes = control_expression.to(self.device)
        emb = drug_embedding.to(self.device)
        dosage_val = torch.full(
            (batch_size,), float(dosage), dtype=torch.float32, device=self.device
        )

        return self._predict_from_embedding(genes, emb, dosage_val, covariates)

    # ------------------------------------------------------------------
    #  查询方法
    # ------------------------------------------------------------------

    def smiles_to_idx(self, smiles: str) -> int:
        """将 SMILES 转换为嵌入文件中的行索引"""
        if smiles not in self._smiles_to_idx:
            raise ValueError(f"SMILES '{smiles}' 不在嵌入文件中")
        return self._smiles_to_idx[smiles]

    def idx_to_smiles(self, idx: int) -> str:
        """将嵌入文件行索引转换为 SMILES"""
        if idx < 0 or idx >= self.num_drugs_total:
            raise IndexError(f"索引 {idx} 超出范围 [0, {self.num_drugs_total - 1}]")
        return self._smiles_list[idx]

    def get_drug_embedding(self, drug_idx: int) -> np.ndarray:
        """获取指定索引药物的 194 维嵌入向量"""
        return self._drug_table[drug_idx].cpu().numpy()

    def get_all_smiles(self) -> List[str]:
        """返回嵌入文件中所有 SMILES 列表（17869个）"""
        return list(self._smiles_list)

    def get_num_genes(self) -> int:
        """返回基因数量"""
        return self.num_genes

    def get_num_drugs(self) -> int:
        """返回嵌入文件中的药物总数（17869）"""
        return self.num_drugs_total

    def get_drug_embedding_dim(self) -> int:
        """返回药物嵌入维度（194 for rdkit2D）"""
        return self._embedding_dim
