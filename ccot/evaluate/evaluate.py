
from condot.utils.loaders import load_model
import torch
from condot.models import condot
from ccot.losses.ps import compute_l2_distance
from ccot.losses.fid import compute_fid, compute_fid_optimized
from ccot.losses.r2 import compute_r2, compute_r2_sc
from ccot.losses.kl import compute_kl_divergence
from ccot.losses.energy_distance import choose_best_energy_distance_method, energy_distance_gpu
from condot import losses
import pandas as pd
from pathlib import Path
import numpy as np
import random
import os
import anndata as ad
from condot.data.data_chemcpa import create_data_loader
from tqdm import tqdm

random.seed(42)  # 固定 Python 随机种子
np.random.seed(42)  # 固定 NumPy 随机种子
torch.manual_seed(42)  # 固定 PyTorch 随机种子


def eval_fxn(outdir, config, adata=None, datasets=None, emb_pretrained=None, model_name=None):
    device = config.training.device if 'device' in config.training else 'cpu'
    beta = config.model.classifier_free_guidance.beta
    weight_matrix = torch.load(config.model.adaptive_mass_transport.path, map_location=device)['fc.weight']
    alpha = torch.tensor(config.model.adaptive_mass_transport.alpha, device=device)
    sigma = torch.tensor(config.model.adaptive_mass_transport.sigma, device=device)
    if (alpha - 0.0) > 1e-8:
        I = torch.eye(weight_matrix.shape[1], device=device)
        transform_matrix = torch.inverse(I + alpha * torch.matmul(weight_matrix.t(), weight_matrix))
    else:
        # transform_matrix = torch.eye(weight_matrix.shape[1], device=device)
        transform_matrix = None
    outdir = Path(outdir)
    evaldir = outdir / "eval"
    cachedir = outdir / "cache"
    datadir = evaldir / "data"
    if not os.path.exists(evaldir):
        os.mkdir(evaldir)
    optunadir = outdir / "optuna"

    emb_type = config.model.embedding.type
    if emb_type == 'value':
        condition_name = 'dose'
    elif 'smiles' in emb_type:
        condition_name = 'SMILES'
    else:
        condition_name = emb_type


    if datasets is None or emb_pretrained is None:
        if adata is None:
            adata = ad.read_h5ad(config.data.path)
            adata.X = adata.X.toarray()
        # 将 SMILES 映射到整数索引
        emb_pretrained = pd.read_parquet(config.model.embedding.path)
        unique_smiles = adata.obs['SMILES'].drop_duplicates()
        emb_pretrained = emb_pretrained.loc[unique_smiles]
        smiles_to_index = pd.Series(range(len(emb_pretrained)), index=emb_pretrained.index).to_dict()
        emb_pretrained = emb_pretrained.to_numpy(dtype=np.float32)

        # 将预训练嵌入转为tensor
        emb_pretrained = torch.tensor(emb_pretrained, dtype=torch.float32, device=device)
        # 进行标准化
        emb_pretrained = emb_pretrained / (torch.norm(emb_pretrained, dim=1, keepdim=True) + 1e-8)
        emb_pretrained.requires_grad = False

        # 一次性对 adata 按 split 进行预筛选
        split_key = config.datasplit.split_key
        split_data = {
            "train_control": adata[(adata.obs[split_key] == 'train') & (adata.obs['control'] == 1)],
            "train_treated": adata[(adata.obs[split_key] == 'train') & (adata.obs['control'] == 0)],
            "test_control": adata[(adata.obs[split_key] == 'test') & (adata.obs['control'] == 1)],
            "test_treated": adata[(adata.obs[split_key] == 'test') & (adata.obs['control'] == 0)],
            "ood": adata[(adata.obs[split_key] == 'ood')],
        }
        datasets = create_data_loader(split_data, config.dataloader.batch_size, smiles_to_index,
                                      align_by_cell_type=True, num_workers=8, pin_memory=True, return_loader=False)

    if emb_type == 'smiles_onehot':
        condition_labels = [[x] for x in range(emb_pretrained.shape[0])]
    elif emb_type == 'cell_type':
        condition_labels = [[x] for x in range(len(datasets['train_treated'].keys()))]
    elif emb_type == 'value':
        condition_labels = config.model.embedding.factor
    elif emb_type == 'smiles':
        condition_labels = emb_pretrained
    else:
        raise ValueError

    model_kwargs = {}
    model_kwargs["input_dim"] = split_data['test_control'].shape[1] \
        if 'dimension_reduction' not in config.data else config.data.dimension_reduction.dims
    (f, g, emb_encoder), opts = load_model(config, restore=cachedir / "best_model.pt" if 'optuna' not in config or model_name is None else optunadir/model_name,
                                           labels=condition_labels, loader=None, device=device, **model_kwargs)

    g.eval()
    f.eval()

    # initiate DataFrame per stage
    metrics_stage = pd.DataFrame(columns=['Loss_g', 'Loss_f', 'dist', 'mmd', 'wst', 'l2_loss', 'fid', 'r2', 'r2_sc', 'kl_divergence', 'energy_distance'])
    metrics_stage_de = pd.DataFrame(columns=['mmd', 'wst', 'l2_loss', 'fid', 'r2', 'r2_sc', 'kl_divergence', 'energy_distance'])

    # for stage in ['ood', 'test', 'train']:
    for stage in ['ood', 'test']:
        if stage != 'ood':
            ctl_key = stage+'_control'
            treat_key = stage+'_treated'
        else:
            ctl_key = 'test_control'
            treat_key = 'ood'

        groupby = split_data[treat_key].obs.groupby(['condition', 'cell_type'], observed=True).groups
        pert_list = split_data[treat_key].obs['condition'].unique().tolist()

        #  initialize DataFrame per pert
        metrics_pert = pd.DataFrame(
            columns=['Loss_g', 'Loss_f', 'dist', 'mmd', 'wst', 'l2_loss', 'fid', 'r2', 'r2_sc', 'kl_divergence', 'energy_distance'])

        metrics_pert_de = pd.DataFrame(
            columns=['mmd', 'wst', 'l2_loss', 'fid', 'r2', 'r2_sc', 'kl_divergence', 'energy_distance'])

        for pert in tqdm(pert_list, total=len(pert_list),
                    desc=f"evaluating {stage}", ncols=100, position=0, leave=True):
            # initialize metrics
            gl = torch.tensor(0.0, device=device)
            fl = torch.tensor(0.0, device=device)
            dist = torch.tensor(0.0, device=device)
            wst = torch.tensor(0.0, device=device)
            l2_loss = torch.tensor(0.0, device=device)
            r2 = torch.tensor(0.0, device=device)
            r2_sc = torch.tensor(0.0, device=device)
            mmd = 0.0
            fid = 0.0
            kl = torch.tensor(0.0, device=device)
            ed = 0.0
            # initialize metrics for DEGs
            wst_de = torch.tensor(0.0, device=device)
            l2_loss_de = torch.tensor(0.0, device=device)
            r2_de = torch.tensor(0.0, device=device)
            r2_sc_de = torch.tensor(0.0, device=device)
            mmd_de = 0.0
            fid_de = 0.0
            kl_de = torch.tensor(0.0, device=device)
            ed_de = 0.0

            for cell_type in datasets[ctl_key].keys():
                source = datasets[ctl_key][cell_type].features.to(device, non_blocking=True)
                source.requires_grad_(True)

                target = torch.tensor(split_data[treat_key][groupby[pert, cell_type]].X, device=device)
                condition_idx = split_data[treat_key][groupby[pert, cell_type]].obs[condition_name].values.tolist()

                if emb_type == 'smiles':
                    condition_idx = [smiles_to_index[key] for key in condition_idx]
                    condition_idx = torch.tensor(condition_idx, device=device)
                elif emb_type == 'smiles_onehot':
                    condition_idx = [smiles_to_index[key] for key in condition_idx]
                    condition_idx = [[item] for item in condition_idx]
                elif emb_type == 'cell_type':
                    cell_type_map = {'A549': 0, 'K562': 1, 'MCF7': 2}
                    condition_idx = [cell_type_map[key] for key in condition_idx]
                    condition_idx = [[item] for item in condition_idx]
                elif emb_type == 'value':
                    condition_idx = torch.tensor(condition_idx, device=device)
                else:
                    raise ValueError()

                # subsample
                if len(source) > 5000 and len(target) > 5000:
                    indices = torch.randperm(len(source))[:5000]
                    source = source[indices]
                    target = target[indices]
                    if emb_type in ['smiles_onehot', 'cell_type']:
                        condition_idx = [condition_idx[i] for i in indices]
                    else:
                        condition_idx = condition_idx[indices]
                if source.shape[0] > target.shape[0]:
                    indices = torch.randperm(target.shape[0])
                    source = source[indices]
                elif target.shape[0] > source.shape[0]:
                    indices = torch.randperm(source.shape[0])
                    target = target[indices]
                    if emb_type in ['smiles_onehot', 'cell_type']:
                        condition_idx = [condition_idx[i] for i in indices]
                    else:
                        condition_idx = condition_idx[indices]

                emb_condition = emb_encoder(condition_idx).to(device, non_blocking=True)

                if abs(beta - 1.0) < 0.00001:
                    transport = g.transport(source, emb_condition, is_conditional_generation=True)
                else:
                    # classifer-free guidance generation
                    transport_cond = g.transport(source, emb_condition, is_conditional_generation=True)
                    transport_uncond = g.transport(source, emb_condition, is_conditional_generation=False)
                    transport = beta * transport_cond - (beta - 1) * transport_uncond

                transport = transport.detach()
                # # -----------------
                # transport = torch.mean(target, dim=0, keepdim=True)
                # transport = transport.repeat(target.shape[0], 1)
                # # -----------------

                cov_drug_dose = cell_type + '_' + pert + '_1.0'
                bool_de = adata.var_names.isin(adata.uns['lincs_DEGs'][cov_drug_dose])
                idx_de = np.where(bool_de)[0]
                idx_de = torch.from_numpy(idx_de).to(device)

                # if not os.path.exists(datadir/ stage/ pert/ f"{cell_type}.npz"):
                #     os.makedirs(datadir.joinpath(stage, pert) , exist_ok=True)
                #     np.savez(datadir.joinpath(stage, pert, f'{cell_type}.npz'),
                #              pred=transport.cpu().numpy(),
                #              real=target.cpu().numpy(),
                #              source=source.detach().cpu().numpy()
                #              )
                #     loaded_data = np.load(datadir.joinpath(stage, pert, f'{cell_type}.npz'))
                #     assert np.allclose(transport.cpu().numpy(), loaded_data['pred']), "transport 数据不一致！"
                #     assert np.allclose(target.cpu().numpy(), loaded_data['real']), "target 数据不一致！"
                #     assert np.allclose(source.detach().cpu().numpy(), loaded_data['source']), "source 数据不一致！"

                # save intermediate data for DEGs
                if not os.path.exists(datadir/ (stage+'_DEGs')/ pert/ f"{cell_type}.npz"):
                    os.makedirs(datadir.joinpath((stage+'_DEGs'), pert) , exist_ok=True)
                    np.savez(datadir.joinpath((stage+'_DEGs'), pert, f'{cell_type}.npz'),
                             pred=transport[:, idx_de].cpu().numpy(),
                             real=target[:, idx_de].cpu().numpy(),
                             source=source[:, idx_de].detach().cpu().numpy()
                             )
                    loaded_data = np.load(datadir.joinpath((stage+'_DEGs'), pert, f'{cell_type}.npz'))
                    assert np.allclose(transport[:, idx_de].cpu().numpy(), loaded_data['pred']), "DEGs transport 数据不一致！"
                    assert np.allclose(target[:, idx_de].cpu().numpy(), loaded_data['real']), "DEGs target 数据不一致！"
                    assert np.allclose(source[:, idx_de].detach().cpu().numpy(), loaded_data['source']), "DEGs source 数据不一致！"

                with torch.no_grad():
                    # gl += condot.compute_loss_g(f, g, source, emb_condition, transport,
                    #                             transform_matrix=transform_matrix, sigma=sigma).mean()
                    # fl += condot.compute_loss_f(f, g, source, target, emb_condition, transport,
                    #                             transform_matrix=transform_matrix, sigma=sigma).mean()
                    # transport = transport @ transform_matrix if transform_matrix is not None else transport
                    # dist += condot.compute_w2_distance(
                    #     f, g, source, target,
                    #     emb_condition, transport
                    # )
                    # mmd += losses.compute_scalar_mmd(
                    #     target.detach().cpu().numpy(),
                    #     transport.detach().cpu().numpy()
                    # )
                    # wst += losses.wasserstein_loss(
                    #     target.detach(), transport.detach()
                    # )
                    # l2_loss += compute_l2_distance(target.detach(), transport)
                    # # fid += compute_fid(target.detach(), transport)
                    # fid += compute_fid_optimized(target.detach(), transport)
                    # r2 += compute_r2(target.detach().mean(dim=0), transport.mean(dim=0))
                    # r2_sc += compute_r2_sc(target.detach(), transport)
                    # kl += compute_kl_divergence(target.detach(), transport)
                    # # # 选择最佳方法
                    # # energy_distance_func = choose_best_energy_distance_method(target.detach().cpu().numpy().astype(np.float32), transport.cpu().numpy().astype(np.float32))
                    # # 计算能量距离
                    # # ed += energy_distance_func(target.detach().cpu().numpy().astype(np.float32), transport.cpu().numpy().astype(np.float32))
                    # ed += energy_distance_gpu(target.detach(), transport)

                    mmd_de += losses.compute_scalar_mmd(
                        target[:, idx_de].detach().cpu().numpy(),
                        transport[:, idx_de].detach().cpu().numpy()
                    )
                    wst_de += losses.wasserstein_loss(
                        target[:, idx_de].detach(), transport[:, idx_de].detach()
                    )
                    l2_loss_de += compute_l2_distance(target[:, idx_de].detach(), transport[:, idx_de])
                    # fid_de += compute_fid(target[:, idx_de].detach(), transport[:, idx_de])
                    fid_de += compute_fid_optimized(target[:, idx_de].detach(), transport[:, idx_de])
                    r2_de += compute_r2(target[:, idx_de].detach().mean(dim=0), transport[:, idx_de].mean(dim=0))
                    r2_sc_de += compute_r2_sc(target.detach(), transport)
                    kl_de += compute_kl_divergence(target[:, idx_de].detach(), transport[:, idx_de])
                    # # 选择最佳方法
                    # energy_distance_func = choose_best_energy_distance_method(target.detach().cpu().numpy().astype(np.float32), transport.cpu().numpy().astype(np.float32))
                    # 计算能量距离
                    # ed += energy_distance_func(target.detach().cpu().numpy().astype(np.float32), transport.cpu().numpy().astype(np.float32))
                    ed_de += energy_distance_gpu(target[:, idx_de].detach(), transport[:, idx_de])

            num_cell_type = len(list(datasets[treat_key].keys()))
            # metrics_pert.loc[pert] = {
            #     'Loss_g': gl.item() / num_cell_type,
            #     'Loss_f': fl.item() / num_cell_type,
            #     'dist': dist.item() / num_cell_type,
            #     'mmd': mmd.item() / num_cell_type,
            #     'wst': wst.item() / num_cell_type,
            #     'l2_loss': l2_loss.item() / num_cell_type,
            #     'fid': (fid / num_cell_type) if fid is not float('nan') and fid is not float('inf') else fid,
            #     'r2' : r2.item() / num_cell_type,
            #     'r2_sc' : r2_sc.item() / num_cell_type,
            #     'kl_divergence' : kl.item() / num_cell_type,
            #     'energy_distance' : ed.item() / num_cell_type
            # }

            metrics_pert_de.loc[pert] = {
                'mmd': mmd_de.item() / num_cell_type,
                'wst': wst_de.item() / num_cell_type,
                'l2_loss': l2_loss_de.item() / num_cell_type,
                'fid': (fid_de / num_cell_type) if fid_de is not float('nan') and fid_de is not float('inf') else fid_de,
                'r2': r2_de.item() / num_cell_type,
                'r2_sc' : r2_sc_de.item() / num_cell_type,
                'kl_divergence': kl_de.item() / num_cell_type,
                'energy_distance': ed_de.item() / num_cell_type
            }

            # if 'optuna' in config:
            #     return metrics_df.loc['test', 'mmd']
        # metrics_stage.loc[f'{stage}_mean'] = metrics_pert.mean()
        # metrics_stage.loc[f'{stage}_std'] = metrics_pert.std()
        metrics_stage_de.loc[f'{stage}_mean'] = metrics_pert_de.mean()
        metrics_stage_de.loc[f'{stage}_std'] = metrics_pert_de.std()

        # metrics_pert.to_csv(Path(evaldir) / f"{stage}_results.csv" if 'optuna' not in config else Path(evaldir) / f"{stage}_results_optuna.csv")
        metrics_pert_de.to_csv(Path(evaldir) / f"{stage}_DEGs_results.csv" if 'optuna' not in config else Path(evaldir) / f"{stage}_DEGs_results_optuna.csv")

    # 输出到文件
    # metrics_stage.to_csv(Path(evaldir) / f"summary_results.csv" if 'optuna' not in config else Path(evaldir) / f"summary_results_optuna.csv")
    metrics_stage_de.to_csv(Path(evaldir) / f"DEGs_summary_results.csv" if 'optuna' not in config else Path(evaldir) / f"DEGs_summary_results_optuna.csv")


    return
#    return metrics_stage.loc['test', 'mmd']


def test_fxn_unalign_by_cell_type(outdir, config, datasets=None, emb_pretrained=None, model_name=None):
    device = config.training.device if 'device' in config.training else 'cpu'
    beta = config.model.classifier_free_guidance.beta
    weight_matrix = torch.load(config.model.adaptive_mass_transport.path, map_location=device)['fc.weight']
    alpha = torch.tensor(config.model.adaptive_mass_transport.alpha, device=device)
    sigma = torch.tensor(config.model.adaptive_mass_transport.sigma, device=device)
    if (alpha - 0.0) > 1e-8:
        I = torch.eye(weight_matrix.shape[1], device=device)
        transform_matrix = torch.inverse(I + alpha * torch.matmul(weight_matrix.t(), weight_matrix))
    else:
        transform_matrix = None

    evaldir = outdir / "eval"
    cachedir = outdir / "cache"
    if not os.path.exists(evaldir):
        os.mkdir(evaldir)
    optunadir = outdir / "optuna"


    if datasets is None or emb_pretrained is None:
        adata = ad.read_h5ad(config.data.path)
        adata.X = adata.X.toarray()
        # 将 SMILES 映射到整数索引
        emb_pretrained = pd.read_parquet(config.model.embedding.path)
        unique_smiles = adata.obs['SMILES'].drop_duplicates()
        emb_pretrained = emb_pretrained.loc[unique_smiles]
        smiles_to_index = pd.Series(range(len(emb_pretrained)), index=emb_pretrained.index).to_dict()
        emb_pretrained = emb_pretrained.to_numpy(dtype=np.float32)

        # 将预训练嵌入转为tensor
        emb_pretrained = torch.tensor(emb_pretrained, dtype=torch.float32, device=device)
        # 进行标准化
        emb_pretrained = emb_pretrained / (torch.norm(emb_pretrained, dim=1, keepdim=True) + 1e-8)
        emb_pretrained.requires_grad = False

        # 一次性对 adata 按 split 进行预筛选
        split_key = config.datasplit.split_key
        split_data = {
            "train_control": adata[(adata.obs[split_key] == 'train') & (adata.obs['control'] == 1)],
            "train_treated": adata[(adata.obs[split_key] == 'train') & (adata.obs['control'] == 0)],
            "test_control": adata[(adata.obs[split_key] == 'test') & (adata.obs['control'] == 1)],
            "test_treated": adata[(adata.obs[split_key] == 'test') & (adata.obs['control'] == 0)],
            "ood": adata[(adata.obs[split_key] == 'ood')],
        }

        datasets = create_data_loader(split_data, config.dataloader.batch_size, smiles_to_index, align_by_cell_type=config.datasplit.align_by_cell_type, num_workers=8, pin_memory=True, return_loader=False)
    model_kwargs = {}
    model_kwargs["input_dim"] = datasets['test_control'].features.shape[1] \
        if 'dimension_reduction' not in config.data else config.data.dimension_reduction.dims
    (f, g, emb_encoder), opts, opt_emb = load_model(config, restore=cachedir / "best_model.pt" if 'optuna' not in config and model_name is None else optunadir/model_name,
                                                    loader=emb_pretrained, device=device, **model_kwargs)

    g.eval()
    f.eval()

    # 初始化一个空的 DataFrame
    metrics_df = pd.DataFrame(columns=['Loss_g', 'Loss_f', 'dist', 'mmd', 'wst', 'l2_loss', 'fid', 'r2', 'r2_sc'])

    for stage in ['test', 'ood', 'train']:
        if stage != 'ood':
            ctl_key = stage+'_control'
            treat_key = stage+'_treated'
        else:
            ctl_key = 'test_control'
            treat_key = 'ood'

        source = datasets[ctl_key].features.to(device, non_blocking=True)
        source.requires_grad_(True)

        target = datasets[treat_key].features.to(device, non_blocking=True)
        smiles_idx = datasets[treat_key].smiles_indices.to(device, non_blocking=True)
        # dosage = datasets[treat_key].doses.to(device, non_blocking=True)

        # subsample
        # if source.shape[0] > 5000 and target.shape[0] > 5000:
        #     indices = torch.randperm(data.size(0))[:num_samples]
        #     source = source[:5000]
        #     target = target[:5000]
        #     smiles_idx = smiles_idx[:5000]
        #     dosage = dosage[:5000]
        if source.shape[0] > target.shape[0]:
            indices = torch.randperm(target.shape[0])
            source = source[indices]
        elif target.shape[0] > source.shape[0]:
            indices = torch.randperm(source.shape[0])
            target = target[indices]
            smiles_idx = smiles_idx[indices]
            # dosage = dosage[indices]

        emb_condition = emb_encoder(smiles_idx).to(device, non_blocking=True)

        if abs(beta - 1.0) < 0.00001:
            transport = g.transport(source, emb_condition, is_conditional_generation=True)
        else:
            # classifer-free guidance generation
            transport_cond = g.transport(source, emb_condition, is_conditional_generation=True)
            transport_uncond = g.transport(source, emb_condition, is_conditional_generation=False)
            transport = beta * transport_cond - (beta - 1) * transport_uncond

        transport = transport.detach()

        with torch.no_grad():
            gl = condot.compute_loss_g(f, g, source, emb_condition, transport,
                                        transform_matrix=transform_matrix, sigma=sigma).mean()
            fl = condot.compute_loss_f(f, g, source, target, emb_condition, transport,
                                        transform_matrix=transform_matrix, sigma=sigma).mean()
            transport = transport @ transform_matrix if transform_matrix is not None else transport
            dist = condot.compute_w2_distance(
                f, g, source, target,
                emb_condition, transport
            )
            mmd = losses.compute_scalar_mmd(
                target.detach().cpu().numpy(),
                transport.detach().cpu().numpy()
            )
            indices = torch.randperm(len(target))[:5000]
            wst = losses.wasserstein_loss(
                target.detach(), transport.detach()
            ) if stage != 'train' else losses.wasserstein_loss(target[indices].detach(), transport[indices].detach())
            l2_loss = compute_l2_distance(target.detach(), transport)
            fid = compute_fid(target.detach(), transport)
            r2 = compute_r2(target.detach().mean(dim=0), transport.mean(dim=0))
            r2_sc = compute_r2_sc(target.detach(), transport)

        metrics_df.loc[stage] = {
            'Loss_g': gl.item(),
            'Loss_f': fl.item(),
            'dist': dist.item(),
            'mmd': mmd.item(),
            'wst': wst.item(),
            'l2_loss': l2_loss.item(),
            'fid': (fid.item()) if fid is not float('nan') and fid is not float('inf') else fid,
            'r2' : r2,
            'r2_sc' : r2_sc.item()
        }
        if 'optuna' in config:
            return metrics_df.loc['test', 'mmd']
    # 输出到文件
    metrics_df.to_csv(Path(evaldir) / f"metrics_results.csv" if 'optuna' not in config else Path(evaldir) / f"metrics_results_optuna.csv")

    return metrics_df.loc['test', 'mmd']