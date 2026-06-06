#!/usr/bin/python3

# imports
from pathlib import Path
import torch
import numpy as np
from absl import logging
from tqdm import trange
import anndata as ad
import pandas as pd
# import time
# import cProfile
# import pstats
# from io import StringIO
# import os
import threading

# internal imports
from condot import losses
from condot.utils.loaders import load
from condot.models import condot
from condot.data.utils import cast_loader_to_iterator
from condot.models.ae import compute_autoencoder_shift
from condot.data.data_chemcpa import create_data_loader
from condot.utils.loaders import load_model
from ccot.losses.ps import compute_l2_distance
from ccot.utils.logger import Logger
from ccot.evaluate.evaluate import eval_fxn

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)


def save_model_async(state, path):
    torch.save(state, path)

def load_lr_scheduler(optim, config):
    if "scheduler" not in config:
        return None

    return torch.optim.lr_scheduler.StepLR(optim, **config.scheduler)


def check_loss(*args):
    for arg in args:
        if torch.isnan(arg):
            raise ValueError


def load_item_from_save(path, key, default):
    path = Path(path)
    if not path.exists():
        return default

    ckpt = torch.load(path, map_location="cpu")
    if key not in ckpt:
        logging.warning(f"'{key}' not found in ckpt: {str(path)}")
        return default

    return ckpt[key]


def train_condot(outdir, config):

    def state_dict(f, g, opts, **kwargs):
        state = {
            "g_state": g.state_dict(),
            "f_state": f.state_dict(),
            # "emb_encoder_state": emb_encoder.state_dict(),
            "opt_g_state": opts.g.state_dict(),
            "opt_f_state": opts.f.state_dict(),
            # "opt_emb_state": opt_emb.state_dict(),
        }
        state.update(kwargs)

        return state

    def evaluate_aligned_by_cell_type():
        # initialize metrics
        # dist = torch.tensor(0.0, device=device)
        wst = torch.tensor(0.0, device=device)
        l2_loss = torch.tensor(0.0, device=device)
        # r2 = torch.tensor(0.0, device=device)
        mmd = 0.0
        # fid = 0.0

        # sample target cells
        target = next(test_treated_iterator)
        target_features = target['features'].to(device, non_blocking=True)  # 基因表达数据
        # target_smiles_idx = target['SMILES_idx'].to(device, non_blocking=True)  # SMILES 数据
        # target_doses = target['dose'].to(device, non_blocking=True)  # 药物剂量

        # sample source cells
        source = next(test_control_iterator)
        source_features = source['features'].to(device, non_blocking=True)  # 基因表达数据
        source_features.requires_grad_(True)

        if emb_type == 'smiles_onehot':
            target[condition_name] = [[x] for x in target[condition_name].tolist()]
        elif emb_type == 'smiles':
            target[condition_name] = target[condition_name].to(device, non_blocking=True)

        emb_condition = emb_encoder(target[condition_name]).to(device, non_blocking=True)

        if abs(beta - 1.0) <0.00001:
            transport = g.transport(source_features, emb_condition, is_conditional_generation=True)
        else:
            # classifer-free guidance generation
            transport_cond = g.transport(source_features, emb_condition, is_conditional_generation=True)
            transport_uncond = g.transport(source_features, emb_condition, is_conditional_generation=False)
            transport = beta * transport_cond - (beta - 1) * transport_uncond


        transport = transport.detach()

        with torch.no_grad():
            gl = condot.compute_loss_g(f, g, source_features, emb_condition, transport,
                                       transform_matrix=transform_matrix, sigma=sigma).mean()
            fl = condot.compute_loss_f(f, g, source_features, target_features, emb_condition, transport,
                                       transform_matrix=transform_matrix, sigma=sigma).mean()
            transport = transport @ transform_matrix if transform_matrix is not None else transport
            for i in range(3):
                start = i * config.dataloader.batch_size
                end = (i + 1) * config.dataloader.batch_size
                index_cell_type = list(range(start, end))
                # dist += condot.compute_w2_distance(
                #     f, g, source_features[index_cell_type], target_features[index_cell_type], emb_condition[index_cell_type], transport[index_cell_type]
                # )
                mmd += losses.compute_scalar_mmd(
                    target_features[index_cell_type].detach().cpu().numpy(), transport[index_cell_type].detach().cpu().numpy()
                )
                wst += losses.wasserstein_loss(
                    target_features[index_cell_type].detach(), transport[index_cell_type].detach()
                )
                l2_loss += compute_l2_distance(target_features[index_cell_type].detach(), transport[index_cell_type])
                # fid += compute_fid(target_features[index_cell_type].detach(), transport[index_cell_type])
                # r2 += compute_r2(target_features[index_cell_type].detach().mean(dim=0), transport[index_cell_type].mean(dim=0))
        #num_cell_type = len(target_cell_types.unique())
        num_cell_type=3
        loss_dict = {
            'Loss_g': gl.item(),
            'Loss_f': fl.item(),
            # 'dist': dist.item()/num_cell_type,
            'mmd': mmd.item()/num_cell_type,
            'wst': wst.item()/num_cell_type,
            'l2_loss': l2_loss.item()/num_cell_type,
            # 'fid': fid.item()/num_cell_type,
            # 'r2': r2.item()/num_cell_type
        }
        # log to logger object
        if 'optuna' not in config:
            logger.write(
                step,
                stage="Valid",
                **loss_dict
            )
            # logger.writer.flush()
        # check loss
        check_loss(gl, fl)

        # return mmd
        return loss_dict

    def evaluate_unaligned_by_cell_type():

        # sample target cells
        target = next(test_treated_iterator)
        target_features = target['features'].to(device, non_blocking=True)  # 基因表达数据
        # target_smiles_idx = target['SMILES_idx'].to(device, non_blocking=True)  # SMILES 数据

        # sample source cells
        source = next(test_control_iterator)
        source_features = source['features'].to(device, non_blocking=True)  # 基因表达数据
        source_features.requires_grad_(True)

        if emb_type == 'smiles_onehot':
            target[condition_name] = [[x] for x in target[condition_name].tolist()]
        elif emb_type == 'smiles':
            target[condition_name] = target[condition_name].to(device, non_blocking=True)

        emb_condition = emb_encoder(target[condition_name]).to(device, non_blocking=True)

        if abs(beta - 1.0) <0.00001:
            transport = g.transport(source_features, emb_condition, is_conditional_generation=True)
        else:
            # classifer-free guidance generation
            transport_cond = g.transport(source_features, emb_condition, is_conditional_generation=True)
            transport_uncond = g.transport(source_features, emb_condition, is_conditional_generation=False)
            transport = beta * transport_cond - (beta - 1) * transport_uncond


        transport = transport.detach()

        with torch.no_grad():
            gl = condot.compute_loss_g(f, g, source_features, emb_condition, transport,
                                       transform_matrix=transform_matrix, sigma=sigma).mean()
            fl = condot.compute_loss_f(f, g, source_features, target_features, emb_condition, transport,
                                       transform_matrix=transform_matrix, sigma=sigma).mean()
            transport = transport @ transform_matrix if transform_matrix is not None else transport

            # dist = condot.compute_w2_distance(
            #     f, g, source_features, target_features, emb_condition, transport
            # )
            mmd = losses.compute_scalar_mmd(
                target_features.detach().cpu().numpy(), transport.detach().cpu().numpy()
            )
            wst = losses.wasserstein_loss(
                target_features.detach(), transport.detach()
            )
            l2_loss = compute_l2_distance(target_features.detach(), transport)
            # fid = compute_fid(target_features.detach(), transport)
            # r2 = compute_r2(target_features.detach().mean(dim=0), transport.mean(dim=0))
        loss_dict = {
            'Loss_g': gl.item(),
            'Loss_f': fl.item(),
            # 'dist': dist.item(),
            'mmd': mmd.item(),
            'wst': wst.item(),
            'l2_loss': l2_loss.item(),
            # 'fid': fid.item(),
            # 'r2': r2.item()
        }
        # log to logger object
        if 'optuna' not in config:
            logger.write(
                step,
                stage="Valid",
                **loss_dict
            )
            # logger.writer.flush()
        # check loss
        check_loss(gl, fl)

        # return mmd
        return loss_dict


    logger = Logger(outdir / "log") if 'optuna' not in config else None
    cachedir = outdir / "cache"
    if 'optuna' in config:
        optunadir = outdir / 'optuna'
        optunadir.mkdir(parents=True, exist_ok=True)
    eval_freq_optuna = getattr(config, 'optuna', {}).get('eval_freq', float('inf'))
    model_name=None
    device = config.training.device
    split_key = config.datasplit.split_key
    emb_type = config.model.embedding.type
    if emb_type == 'value':
        condition_name = 'dose'
        return_fields = ['features', 'dose']
    elif 'smiles' in emb_type:
        condition_name = 'SMILES_idx'
        return_fields = ['features', 'SMILES_idx']
    elif emb_type == 'cell_type':
        condition_name = emb_type
        return_fields = ['features', 'cell_type']
    else:
        raise NotImplementedError(f"Embedding type {emb_type} not supported.")

    beta = config.model.classifier_free_guidance.beta
    weight_matrix = torch.load(config.model.adaptive_mass_transport.path, map_location=device)['fc.weight']
    weight_matrix.requires_grad = False
    alpha = torch.tensor(config.model.adaptive_mass_transport.alpha, device=device)
    sigma = torch.tensor(config.model.adaptive_mass_transport.sigma, device=device)
    if (alpha - 0.0) > 1e-8:
        I = torch.eye(weight_matrix.shape[1], device=device)
        transform_matrix = torch.inverse(I + alpha * torch.matmul(weight_matrix.t(), weight_matrix))
    else:
        # transform_matrix = torch.eye(weight_matrix.shape[1], device=device)
        transform_matrix = None
    # -----------------------------------------------------------------------
    logging.info(f"Starting to read in data: {config.data.path}\n...")
    adata = ad.read_h5ad(config.data.path)
    # adata = adata[adata.obs['condition'].isin(config.data.drug_list)]
    # if adata.is_view:
    #     adata = adata.copy()
    adata.X = adata.X.toarray()
    # 将 SMILES 映射到整数索引
    emb_pretrained = pd.read_parquet(config.model.embedding.path)
    logging.info(f"Finished data loading.")
    unique_smiles = adata.obs['SMILES'].drop_duplicates()
    emb_pretrained = emb_pretrained.loc[unique_smiles]
    smiles_to_index = pd.Series(range(len(emb_pretrained)), index=emb_pretrained.index).to_dict()

    emb_pretrained = emb_pretrained.to_numpy(dtype=np.float32)
    # pca = PCA(n_components=0.99)
    # emb_pretrained = pca.fit_transform(emb_pretrained)

    # 将预训练嵌入转为tensor
    emb_pretrained = torch.tensor(emb_pretrained, dtype=torch.float32, device=device)
    # 进行标准化
    emb_pretrained = emb_pretrained / (torch.norm(emb_pretrained, dim=1, keepdim=True) + 1e-8)
    emb_pretrained.requires_grad = False

    # 一次性对 adata 按 split 进行预筛选
    split_data = {
        "train_control": adata[(adata.obs[split_key] == 'train') & (adata.obs['control'] == 1)],
        "train_treated": adata[(adata.obs[split_key] == 'train') & (adata.obs['control'] == 0)],
        "test_control": adata[(adata.obs[split_key] == 'test') & (adata.obs['control'] == 1)],
        "test_treated": adata[(adata.obs[split_key] == 'test') & (adata.obs['control'] == 0)],
        "ood": adata[(adata.obs[split_key] == 'ood')],
    }

    datasets, dataloaders = create_data_loader(
        split_data, config.dataloader.batch_size, smiles_to_index,
        align_by_cell_type=config.datasplit.align_by_cell_type,
        num_workers=config.dataloader.get('num_workers', 10), pin_memory=True, return_fields=return_fields
    )

    # 获取迭代器
    iterators = {split: iter(dataloaders[split]) for split in dataloaders.keys()}

    train_control_iterator = iterators["train_control"]
    train_treated_iterator = iterators["train_treated"]
    test_control_iterator = iterators["test_control"]
    test_treated_iterator = iterators["test_treated"]
    # ood_iterator = iterators["ood"]

    # -----------------------------------------------------------------------
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
    model_kwargs["input_dim"] = adata.shape[1] if 'dimension_reduction' not in config.data else config.data.dimension_reduction.dims
    (f, g, emb_encoder), opts= load_model(
        config,
        restore=cachedir / "last.pt" if 'optuna' not in config else cachedir / "best_model.pt",
        labels=condition_labels,
        loader=None,
        device=device,
        **model_kwargs)
    # (f, g, emb_encoder), opts, opt_emb= load_model(config, restore=cachedir / "last.pt" if 'optuna' not in config else cachedir / "best_model.pt",
    #                                                loader=torch.tensor(np.unique(datasets['train_treated'].cell_types)), device=device, **model_kwargs)

    n_iters = config.training.n_iters if 'optuna' not in config else config.optuna.n_iters_per_search
    step = load_item_from_save(cachedir / "last.pt", "step", 0) if 'optuna' not in config else 0

    minmmd = load_item_from_save(cachedir / "best_model.pt", "minmmd", np.inf) if 'optuna' not in config else np.inf
    mmd = minmmd

    ticker = trange(step, n_iters, initial=step, total=n_iters,
                    desc="进度", ncols=100, position=0, leave=True)

    # # 创建 cProfile 性能分析器
    # profiler = cProfile.Profile()
    # profiler.enable()  # 开始性能分析

    # scaler = torch.cuda.amp.GradScaler()  # 初始化 GradScaler
    for step in ticker:
        g.train()
        f.train()

        # sample source cells
        source = next(train_control_iterator)
        source_features = source['features'].to(device, non_blocking=True)  # 基因表达数据
        source_features.requires_grad_(True)

        for _ in range(config.training.n_inner_iters):
            # sample target cells
            target = next(train_treated_iterator)
            # target_features = target['features'].to(device, non_blocking=True)  # 基因表达数据
            # target_smiles_idx = target['SMILES_idx'].to(device, non_blocking=True)
            # target_cell_type = target['cell_type']# SMILES 数据
            # target_doses = target['dose'].to(device, non_blocking=True)  # 药物剂量

            opts.g.zero_grad()
            # opt_emb.zero_grad()

            # with torch.cuda.amp.autocast():  # 自动混合精度
            if emb_type == 'smiles_onehot':
                target[condition_name] = [[x] for x in target[condition_name].tolist()]
            elif emb_type == 'smiles':
                target[condition_name] = target[condition_name].to(device, non_blocking=True)

            emb_condition = emb_encoder(target[condition_name]).to(device, non_blocking=True)
            # emb_condition = emb_encoder(target_cell_type)
            gl = condot.compute_loss_g(f, g, source_features, emb_condition,
                                       beta=beta, transform_matrix=transform_matrix, sigma=sigma).mean()
            if not g.softplus_wz_kernels and g.fnorm_penalty > 0:
                gl = gl + g.penalize_w()

            gl.backward()
            opts.g.step()
            # opt_emb.step()


        # profiler.disable()  # 停止性能分析

        # sample target cells
        target = next(train_treated_iterator)
        target_features = target['features'].to(device, non_blocking=True)  # 基因表达数据
        # target_smiles_idx = target['SMILES_idx'].to(device, non_blocking=True)  # SMILES 数据
        # target_doses = target['dose'].to(device, non_blocking=True)  # 药物剂量

        opts.f.zero_grad()
        # with torch.cuda.amp.autocast():
        if emb_type == 'smiles_onehot':
            target[condition_name] = [[x] for x in target[condition_name].tolist()]
        elif emb_type == 'smiles':
            target[condition_name] = target[condition_name].to(device, non_blocking=True)

        emb_condition = emb_encoder(target[condition_name]).to(device, non_blocking=True)
        fl = condot.compute_loss_f(f, g, source_features, target_features, emb_condition,
                                   beta=beta, transform_matrix=transform_matrix, sigma=sigma).mean()

        fl.backward()
        opts.f.step()
        check_loss(gl, fl)
        f.clamp_w()


        if step % config.training.logs_freq == 0 and 'optuna' not in config:
            # log to logger object
            loss_dict = {'Loss_g': gl.item(), 'Loss_f': fl.item()}
            logger.write(step, stage="Train", **loss_dict)
            # logger.writer.flush()

        if step % config.training.eval_freq == 0 or step % eval_freq_optuna == 0:
            g.eval()
            f.eval()

            loss_dict = evaluate_aligned_by_cell_type() if config.datasplit.align_by_cell_type else evaluate_unaligned_by_cell_type()

            # # 打印性能分析结果
            # s = StringIO()
            # ps = pstats.Stats(profiler, stream=s).strip_dirs().sort_stats('cumulative')
            # ps.print_stats(10)  # 打印累计时间最高的 10 个函数
            # print(s.getvalue())

            mmd = loss_dict['mmd']
            if mmd < minmmd:
                minmmd = mmd
                if 'optuna' not in config:
                    save_thread1 = threading.Thread(target=save_model_async, args=(state_dict(f, g, opts, step=step, minmmd=minmmd), cachedir / "best_model.pt"))
                    save_thread1.start()
                    loss_dict.update(step=step)
                    save_thread2 = threading.Thread(target=save_model_async, args=(loss_dict, cachedir / "best_loss.pt"))
                    save_thread2.start()
                    # 等待两个线程都完成保存
                    save_thread1.join()
                    save_thread2.join()

                    # torch.save(
                    #     state_dict(f, g, emb_encoder, opts, step=step, minmmd=minmmd),
                    #     cachedir / "best_model.pt",
                    # )
                    # loss_dict.update(step=step)
                    # torch.save(
                    #     loss_dict,
                    #     cachedir / "best_loss.pt",
                    # )
                else:
                    model_name = f"alpha_{config.model.adaptive_mass_transport.alpha}_sigma_{config.model.adaptive_mass_transport.sigma}_beta_{config.model.classifier_free_guidance.beta}.pt"
                    torch.save(state_dict(f, g, opts, step=step, minmmd=minmmd), optunadir / model_name)
                    print(f"\nNew best model saved: {model_name}, mmd: {minmmd}")

        if step % config.training.cache_freq == 0 and 'optuna' not in config:
            # torch.save(state_dict(f, g, emb_encoder, opts, step=step), cachedir / "last.pt")
            save_thread3 = threading.Thread(target=save_model_async, args=(state_dict(f, g, opts, step=step), cachedir / "last.pt"))
            save_thread3.start()

    if 'optuna' not in config:
        torch.save(state_dict(f, g, opts, step=step), cachedir / "last.pt")

    if 'optuna' not in config:
        logger.close()

    # evalmmd = eval_fxn(outdir, config, datasets=datasets, emb_pretrained=emb_pretrained, model_name=model_name) \
    #     if config.datasplit.align_by_cell_type \
    #     else eval_fxn(outdir, config, adata=adata, model_name=model_name)
    #
    # return evalmmd
    return



def train_autoencoder(outdir, config):
    def state_dict(model, optim, **kwargs):
        state = {
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
        }

        if hasattr(model, "code_means"):
            state["code_means"] = model.code_means

        state.update(kwargs)

        return state

    def evaluate(vinputs):
        with torch.no_grad():
            loss, comps, _ = model(vinputs)
            loss = loss.mean()
            comps = {k: v.mean().item() for k, v in comps._asdict().items()}
            check_loss(loss)
            logger.log("eval", loss=loss.item(), step=step, **comps)

        return loss

    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"
    model, optim, loader = load(config, restore=cachedir / "last.pt")
    iterator = cast_loader_to_iterator(loader, cycle_all=True)
    scheduler = load_lr_scheduler(optim, config)

    n_iters = config.training.n_iters
    step = load_item_from_save(cachedir / "last.pt", "step", 0)
    if scheduler is not None and step > 0:
        scheduler.last_epoch = step

    best_eval_loss = load_item_from_save(
        cachedir / "model.pt", "best_eval_loss", np.inf
    )

    eval_loss = best_eval_loss

    ticker = trange(step, n_iters, initial=step, total=n_iters)
    for step in ticker:

        model.train()
        inputs = next(iterator.train)
        optim.zero_grad()
        loss, comps, _ = model(inputs)
        loss = loss.mean()
        comps = {k: v.mean().item() for k, v in comps._asdict().items()}
        loss.backward()
        optim.step()
        check_loss(loss)

        if step % config.training.logs_freq == 0:
            # log to logger object
            logger.log("train", loss=loss.item(), step=step, **comps)

        if step % config.training.eval_freq == 0:
            model.eval()
            eval_loss = evaluate(next(iterator.test))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                sd = state_dict(model, optim, step=(step + 1), eval_loss=eval_loss)

                torch.save(sd, cachedir / "model.pt")

        if step % config.training.cache_freq == 0:
            torch.save(state_dict(model, optim, step=(step + 1)), cachedir / "last.pt")

            logger.flush()

        if scheduler is not None:
            scheduler.step()

    if config.model.name == "autoencoder" and config.get("compute_autoencoder_shift", True):
        labels = loader.train.dataset.adata.obs[config.data.condition]
        compute_autoencoder_shift(model, loader.train.dataset, labels=labels)

    torch.save(state_dict(model, optim, step=step), cachedir / "last.pt")

    logger.flush()
