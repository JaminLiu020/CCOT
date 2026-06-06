"""
chemCPA 核心模型定义，从原始 chemCPA 项目中精简提取。
仅保留推理所需的网络结构，完全复刻原始模型的 forward 逻辑。
"""

import json
from collections import OrderedDict
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


def _move_inputs(*inputs, device="cuda"):
    """将输入移动到指定设备"""
    def mv_input(x):
        if x is None:
            return None
        elif isinstance(x, torch.Tensor):
            return x.to(device)
        else:
            return [mv_input(y) for y in x]
    return [mv_input(x) for x in inputs]


class MLP(torch.nn.Module):
    """
    多层感知机，带 ReLU 激活和可选 BatchNorm。
    完全复刻原始 chemCPA 的 MLP 实现。
    """

    def __init__(
        self,
        sizes,
        batch_norm=True,
        last_layer_act="linear",
        append_layer_width=None,
        append_layer_position=None,
    ):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.ReLU(),
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        if append_layer_width:
            assert append_layer_position in ("first", "last")
            if append_layer_position == "first":
                layers_dict = OrderedDict()
                layers_dict["append_linear"] = torch.nn.Linear(
                    append_layer_width, sizes[0]
                )
                layers_dict["append_bn1d"] = torch.nn.BatchNorm1d(sizes[0])
                layers_dict["append_relu"] = torch.nn.ReLU()
                for i, module in enumerate(layers):
                    layers_dict[str(i)] = module
            else:
                layers_dict = OrderedDict(
                    {str(i): module for i, module in enumerate(layers)}
                )
                layers_dict["append_bn1d"] = torch.nn.BatchNorm1d(sizes[-1])
                layers_dict["append_relu"] = torch.nn.ReLU()
                layers_dict["append_linear"] = torch.nn.Linear(
                    sizes[-1], append_layer_width
                )
        else:
            layers_dict = OrderedDict(
                {str(i): module for i, module in enumerate(layers)}
            )

        self.network = torch.nn.Sequential(layers_dict)

    def forward(self, x):
        if self.activation == "ReLU":
            x = self.network(x)
            dim = x.size(1) // 2
            return torch.cat((self.relu(x[:, :dim]), x[:, dim:]), dim=1)
        return self.network(x)


class GeneralizedSigmoid(torch.nn.Module):
    """
    Sigmoid / log-sigmoid 剂量响应函数。
    """

    def __init__(self, dim, device, nonlin="sigm"):
        super(GeneralizedSigmoid, self).__init__()
        assert nonlin in ("sigm", "logsigm", None)
        self.nonlin = nonlin
        self.beta = torch.nn.Parameter(
            torch.ones(1, dim, device=device), requires_grad=True
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(1, dim, device=device), requires_grad=True
        )

    def forward(self, x, idx=None):
        if self.nonlin == "logsigm":
            if idx is None:
                c0 = self.bias.sigmoid()
                return (torch.log1p(x) * self.beta + self.bias).sigmoid() - c0
            else:
                bias = self.bias[0][idx]
                beta = self.beta[0][idx]
                c0 = bias.sigmoid()
                return (torch.log1p(x) * beta + bias).sigmoid() - c0
        elif self.nonlin == "sigm":
            if idx is None:
                c0 = self.bias.sigmoid()
                return (x * self.beta + self.bias).sigmoid() - c0
            else:
                bias = self.bias[0][idx]
                beta = self.beta[0][idx]
                c0 = bias.sigmoid()
                return (x * beta + bias).sigmoid() - c0
        else:
            return x

    def one_drug(self, x, i):
        if self.nonlin == "logsigm":
            c0 = self.bias[0][i].sigmoid()
            return (torch.log1p(x) * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        elif self.nonlin == "sigm":
            c0 = self.bias[0][i].sigmoid()
            return (x * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        else:
            return x


class ComPert(torch.nn.Module):
    """
    chemCPA (ComPert) 自编码器模型 - 推理精简版。

    保留完整的推理路径：encoder → compute_drug_embeddings → decoder。
    去掉训练相关的优化器、损失调度、early stopping 等。
    """

    def __init__(
        self,
        num_genes: int,
        num_drugs: int,
        num_covariates: int,
        device="cpu",
        seed=0,
        patience=5,
        doser_type="logsigm",
        decoder_activation="linear",
        hparams="",
        drug_embeddings: Union[None, torch.nn.Embedding] = None,
        use_drugs_idx=False,
        append_layer_width=None,
        multi_task: bool = False,
        enable_cpa_mode=False,
    ):
        super(ComPert, self).__init__()
        self.num_genes = num_genes
        self.num_drugs = num_drugs
        self.num_covariates = num_covariates
        self.device = device
        self.seed = seed
        self.patience = patience
        self.best_score = -1e3
        self.patience_trials = 0
        self.use_drugs_idx = use_drugs_idx
        self.multi_task = multi_task
        self.enable_cpa_mode = enable_cpa_mode

        # 设置超参数
        if isinstance(hparams, dict):
            self.hparams = hparams
        else:
            self.set_hparams_(seed, hparams)

        # 存储初始化参数（兼容 checkpoint 格式）
        self.init_args = {
            "num_genes": num_genes,
            "num_drugs": num_drugs,
            "num_covariates": num_covariates,
            "seed": seed,
            "patience": patience,
            "doser_type": doser_type,
            "decoder_activation": decoder_activation,
            "hparams": hparams,
            "use_drugs_idx": use_drugs_idx,
        }

        # Encoder
        self.encoder = MLP(
            [num_genes]
            + [self.hparams["autoencoder_width"]] * self.hparams["autoencoder_depth"]
            + [self.hparams["dim"]],
            append_layer_width=append_layer_width,
            append_layer_position="first",
        )

        # Decoder
        self.decoder = MLP(
            [self.hparams["dim"]]
            + [self.hparams["autoencoder_width"]] * self.hparams["autoencoder_depth"]
            + [num_genes * 2],
            last_layer_act=decoder_activation,
            append_layer_width=2 * append_layer_width if append_layer_width else None,
            append_layer_position="last",
        )

        if append_layer_width:
            self.num_genes = append_layer_width

        # Drug components
        if self.num_drugs > 0:
            # Adversary (需要用于加载 state_dict)
            self.adversary_drugs = MLP(
                [self.hparams["dim"]]
                + [self.hparams["adversary_width"]] * self.hparams["adversary_depth"]
                + [self.num_drugs]
            )

            if drug_embeddings is None:
                self.drug_embeddings = torch.nn.Embedding(
                    self.num_drugs, self.hparams["dim"]
                )
            else:
                self.drug_embeddings = drug_embeddings

            if self.enable_cpa_mode:
                self.drug_embedding_encoder = None
            else:
                self.drug_embedding_encoder = MLP(
                    [self.drug_embeddings.embedding_dim]
                    + [self.hparams["embedding_encoder_width"]]
                    * self.hparams["embedding_encoder_depth"]
                    + [self.hparams["dim"]],
                    last_layer_act="linear",
                )

            # Dosers
            assert doser_type in ("mlp", "sigm", "logsigm", "amortized", None)
            if doser_type == "mlp":
                self.dosers = torch.nn.ModuleList()
                for _ in range(self.num_drugs):
                    self.dosers.append(
                        MLP(
                            [1]
                            + [self.hparams["dosers_width"]]
                            * self.hparams["dosers_depth"]
                            + [1],
                            batch_norm=False,
                        )
                    )
            elif doser_type == "amortized":
                self.dosers = MLP(
                    [self.drug_embeddings.embedding_dim + 1]
                    + [self.hparams["dosers_width"]] * self.hparams["dosers_depth"]
                    + [1],
                )
            else:
                self.dosers = GeneralizedSigmoid(
                    self.num_drugs, self.device, nonlin=doser_type
                )
            self.doser_type = doser_type

        # Covariate embeddings
        if self.num_covariates == [0]:
            self.adversary_covariates = []
            self.covariates_embeddings = []
        else:
            assert 0 not in self.num_covariates
            self.adversary_covariates = []
            self.covariates_embeddings = []
            for num_covariate in self.num_covariates:
                self.adversary_covariates.append(
                    MLP(
                        [self.hparams["dim"]]
                        + [self.hparams["adversary_width"]]
                        * self.hparams["adversary_depth"]
                        + [num_covariate]
                    )
                )
                self.covariates_embeddings.append(
                    torch.nn.Embedding(num_covariate, self.hparams["dim"])
                )

        # Loss (需要存在以匹配 state_dict)
        self.loss_autoencoder = torch.nn.GaussianNLLLoss()
        self.iteration = 0
        self.history = {"epoch": [], "stats_epoch": []}

        self.to(self.device)

    def set_hparams_(self, seed, hparams):
        """设置默认超参数"""
        default = seed == 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.hparams = {
            "dim": 256 if default else int(np.random.choice([128, 256, 512])),
            "dosers_width": 64 if default else int(np.random.choice([32, 64, 128])),
            "dosers_depth": 2 if default else int(np.random.choice([1, 2, 3])),
            "dosers_lr": 1e-3 if default else float(10 ** np.random.uniform(-4, -2)),
            "dosers_wd": 1e-7 if default else float(10 ** np.random.uniform(-8, -5)),
            "autoencoder_width": 512 if default else int(np.random.choice([256, 512, 1024])),
            "autoencoder_depth": 4 if default else int(np.random.choice([3, 4, 5])),
            "adversary_width": 128 if default else int(np.random.choice([64, 128, 256])),
            "adversary_depth": 3 if default else int(np.random.choice([2, 3, 4])),
            "reg_adversary": 5 if default else float(10 ** np.random.uniform(-2, 2)),
            "penalty_adversary": 3 if default else float(10 ** np.random.uniform(-2, 1)),
            "autoencoder_lr": 1e-3 if default else float(10 ** np.random.uniform(-4, -2)),
            "adversary_lr": 3e-4 if default else float(10 ** np.random.uniform(-5, -3)),
            "autoencoder_wd": 1e-6 if default else float(10 ** np.random.uniform(-8, -4)),
            "adversary_wd": 1e-4 if default else float(10 ** np.random.uniform(-6, -3)),
            "adversary_steps": 3 if default else int(np.random.choice([1, 2, 3, 4, 5])),
            "batch_size": 128 if default else int(np.random.choice([64, 128, 256, 512])),
            "step_size_lr": 45 if default else int(np.random.choice([15, 25, 45])),
            "embedding_encoder_width": 512,
            "embedding_encoder_depth": 0,
        }
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)
        return self.hparams

    def compute_drug_embeddings_(self, drugs=None, drugs_idx=None, dosages=None):
        """
        计算药物嵌入：embedding × dose_response。
        完全复刻原始 chemCPA 的 compute_drug_embeddings_ 方法。
        """
        assert (drugs is not None) or (drugs_idx is not None and dosages is not None)

        drugs, drugs_idx, dosages = _move_inputs(
            drugs, drugs_idx, dosages, device=self.device
        )

        latent_drugs = self.drug_embeddings.weight

        if drugs is None:
            if len(drugs_idx.size()) == 0:
                drugs_idx = drugs_idx.unsqueeze(0)
            if len(dosages.size()) == 0:
                dosages = dosages.unsqueeze(0)

        if drugs_idx is not None:
            assert drugs_idx.shape == dosages.shape and len(drugs_idx.shape) == 1
            latent_drugs = latent_drugs[drugs_idx]

        if self.doser_type == "mlp":
            if drugs_idx is None:
                doses = []
                for d in range(drugs.size(1)):
                    this_drug = drugs[:, d].view(-1, 1)
                    doses.append(self.dosers[d](this_drug).sigmoid() * this_drug.gt(0))
                scaled_dosages = torch.cat(doses, 1)
            else:
                scaled_dosages = []
                for idx, dosage in zip(drugs_idx, dosages):
                    scaled_dosages.append(
                        self.dosers[idx](dosage.unsqueeze(0)).sigmoid()
                    )
                scaled_dosages = torch.cat(scaled_dosages, 0)
        elif self.doser_type == "amortized":
            scaled_dosages = self.dosers(
                torch.concat([latent_drugs, torch.unsqueeze(dosages, dim=-1)], dim=1)
            ).squeeze()
        else:
            if drugs_idx is None:
                scaled_dosages = self.dosers(drugs)
            else:
                scaled_dosages = self.dosers(dosages, drugs_idx)

        if len(scaled_dosages.size()) == 0:
            scaled_dosages = scaled_dosages.unsqueeze(0)

        if not self.enable_cpa_mode:
            latent_drugs = self.drug_embedding_encoder(latent_drugs)

        if drugs_idx is None:
            return scaled_dosages @ latent_drugs
        else:
            return torch.einsum("b,be->be", [scaled_dosages, latent_drugs])

    def predict(
        self,
        genes,
        drugs=None,
        drugs_idx=None,
        dosages=None,
        covariates=None,
        return_latent_basal=False,
    ):
        """
        核心推理方法，完全复刻原始 chemCPA 的 predict 方法。

        返回值：
            normalized_reconstructions: [batch, num_genes*2]，前半为均值(经ReLU处理)，后半为方差(经softplus处理)
            cell_drug_embedding: [batch, dim+dim] (推理时可忽略)
            (可选) latent_basal: 隐空间表示
        """
        assert (drugs is not None) or (drugs_idx is not None and dosages is not None)
        genes, drugs, drugs_idx, dosages, covariates = _move_inputs(
            genes, drugs, drugs_idx, dosages, covariates, device=self.device
        )

        latent_basal = self.encoder(genes)
        latent_treated = latent_basal

        if self.num_drugs > 0:
            drug_embedding = self.compute_drug_embeddings_(
                drugs=drugs, drugs_idx=drugs_idx, dosages=dosages
            )
            latent_treated = latent_treated + drug_embedding

        if self.num_covariates[0] > 0 and covariates is not None:
            for cov_type, emb_cov in enumerate(self.covariates_embeddings):
                emb_cov = emb_cov.to(self.device)
                cov_idx = covariates[cov_type].argmax(1)
                latent_treated = latent_treated + emb_cov(cov_idx)

        # 构造 cell_drug_embedding (与原始代码保持一致)
        if self.num_covariates[0] > 0 and self.num_drugs > 0 and covariates is not None:
            emb_cov = self.covariates_embeddings[-1].to(self.device)
            cov_idx = covariates[-1].argmax(1)
            cell_drug_embedding = torch.cat([emb_cov(cov_idx), drug_embedding], dim=1)
        else:
            cell_drug_embedding = None

        gene_reconstructions = self.decoder(latent_treated)

        # 后处理：均值和方差分离
        dim = gene_reconstructions.size(1) // 2
        mean = gene_reconstructions[:, :dim]
        var = F.softplus(gene_reconstructions[:, dim:])
        normalized_reconstructions = torch.concat([mean, var], dim=1)

        if return_latent_basal:
            return normalized_reconstructions, cell_drug_embedding, latent_basal

        return normalized_reconstructions, cell_drug_embedding
