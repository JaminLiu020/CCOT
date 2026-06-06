#!/usr/bin/python3

# imports
from pathlib import Path
from collections import namedtuple
import torch

# internal imports
from condot.networks.picnn import PICNN
from condot.networks.npicnn import NPICNN
from condot.networks import embedding
from condot.networks import combinator


FGPair = namedtuple("FGPair", "f g")


def load_networks(config, labels, loader=None, **kwargs):
    def unpack_kernel_init_fxn(name="uniform", **kwargs):
        if name == "normal":

            def init(*args):
                return torch.nn.init.normal_(*args, **kwargs)

        elif name == "uniform":

            def init(*args):
                return torch.nn.init.uniform_(*args, **kwargs)

        else:
            raise ValueError

        return init

    # load embedding and combinator
    emb = load_embedding(config, labels)

    kwargs.setdefault("hidden_units", [64] * 4)     # default: [64] * 4
    kwargs.update(dict(config.get("model", {})))
    if config.model.embedding.type == "onehot":
        input_dim_label = len(emb.lb.classes_)
        kwargs.update({"input_dim_label": input_dim_label})

    elif config.model.embedding.type == "smacof":
        kwargs.update({"input_dim_label": config.model.embedding.dim})

    elif config.model.embedding.type == "fingerprint":
        kwargs.update({"neural_embedding": (2048, 10)})
        kwargs.update({"input_dim_label": 10})

    elif config.model.embedding.type == "value":
        kwargs.update({"input_dim_label": 1})
    elif config.model.embedding.type == 'smiles':
        kwargs.update({"input_dim_label": config.model.embedding.dim})
    elif config.model.embedding.type in ['smiles_onehot', 'cell_type']:
        kwargs.update({"input_dim_label": len(labels)})
    else:
        raise ValueError

    if "input_dim_label" not in kwargs:
        kwargs.setdefault("input_dim_label", 2)

    # parameters specific to g are stored in config.model.g
    kwargs.pop("name")
    if "latent_dim" in kwargs:
        kwargs.pop("latent_dim")
    fupd = kwargs.pop("f", {})
    gupd = kwargs.pop("g", {})

    fkwargs = kwargs.copy()
    fkwargs.update(fupd)
    fkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
        **fkwargs.pop("kernel_init_fxn")
    )

    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
        **gkwargs.pop("kernel_init_fxn")
    )

    # if onehot embedding, do not use combinator module
    if config.model.embedding.type == "onehot":
        config.model.combinator = False

    # either load combinator with integrated embedding or embedding only
    if "combinator" in config.model and config.model.combinator:
        com = load_combinator(config, emb)
        gkwargs["combinator"] = com
        fkwargs["combinator"] = com
    elif config.model.embedding.type in ['cell_type', 'value', 'smiles', 'smiles_onehot']:
        gkwargs["embedding"] = False
        fkwargs["embedding"] = False
    # else:
    #     gkwargs["embedding"] = emb
    #     fkwargs["embedding"] = emb

    if "init" in config.model:
        gkwargs["init_type"] = config.model.init
        fkwargs["init_type"] = config.model.init

        if config.model.init == "identity":
            if "num_labels" not in config.model:
                config.model.num_labels = len(labels)

            gkwargs["num_labels"] = config.model.num_labels
            fkwargs["num_labels"] = config.model.num_labels
            gkwargs["init_inputs"] = labels
            fkwargs["init_inputs"] = labels

        elif "init" in config.model and config.model.init == "gaussian":
            if "num_labels" not in config.model:
                config.model.num_labels = len(loader.train.target.keys())
            gkwargs["num_labels"] = config.model.num_labels
            fkwargs["num_labels"] = config.model.num_labels
            gkwargs["init_inputs"] = loader.train
            fkwargs["init_inputs"] = loader.train
            gkwargs["name"] = "g"
            fkwargs["name"] = "f"

        else:
            raise ValueError()

        f = NPICNN(**fkwargs)
        g = NPICNN(**gkwargs)

    else:
        f = PICNN(**fkwargs)
        g = PICNN(**gkwargs)

    return f, g, emb


def load_opts(config, f, g):
    kwargs = dict(config.get("optim", {}))
    assert kwargs.pop("optimizer", "Adam") == "Adam"
    if 'emb' in kwargs.keys():
        kwargs.pop('emb')

    fupd = kwargs.pop("f", {})
    gupd = kwargs.pop("g", {})
    # embudp = kwargs.pop("emb", {})

    fkwargs = kwargs.copy()
    fkwargs.update(fupd)
    fkwargs["betas"] = (fkwargs.pop("beta1", 0.9), fkwargs.pop("beta2", 0.999))

    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gkwargs["betas"] = (gkwargs.pop("beta1", 0.9), gkwargs.pop("beta2", 0.999))

    # embkwargs = kwargs.copy()
    # embkwargs.update(embudp)


    opts = FGPair(
        f=torch.optim.Adam(f.parameters(), **fkwargs),
        g=torch.optim.Adam(g.parameters(), **gkwargs),
    )
    # opt_emb = torch.optim.Adam(emb_encoder.parameters(), **fkwargs)
    return opts


def load_embedding(config, labels, **kwargs):
    emb_type = config.get("model.embedding.type", "smacof")

    if emb_type == "smacof":
        model_embedding = embedding.EmbeddingSMACOF(
            config.model.embedding)
    elif emb_type == "fingerprint":
        model_embedding = embedding.EmbeddingFingerprint(
            config.model.embedding)
    elif emb_type == "onehot":
        model_embedding = embedding.EmbeddingOneHot(labels)
    elif emb_type == "value":
        model_embedding = embedding.EmbeddingValue(
            config.model.embedding.factor)
    elif emb_type == "smiles":
        emb_pretrained = labels
        model_embedding = embedding.EmbeddingSMILES(emb_pretrained)
    elif emb_type == "cell_type" or emb_type == "smiles_onehot":
        model_embedding = embedding.EmbeddingIdx2OneHot(labels)
    else:
        raise ValueError
    return model_embedding


def load_combinator(config, model_embedding, **kwargs):
    comb_type = config.get("model.combinator.type", "deepset")

    if comb_type == "deepset":
        model_combinator = combinator.DeepSet(
            input_dim=config.model.embedding.dim,
            hidden_units=config.model.combinator.hidden_units,
            pool=config.model.combinator.pool)
    else:
        raise ValueError
    return combinator.Combinator(model_combinator, model_embedding)


def load_condot_model(config, restore=None, labels=None, loader=None, device = 'cuda:0', **kwargs):
    f, g, emb_encoder = load_networks(config, labels, loader, **kwargs)
    f = f.to(device)
    g = g.to(device)
    emb_encoder = emb_encoder.to(device)

    opts = load_opts(config, f, g)

    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore, map_location=device)
        f.load_state_dict(ckpt["f_state"])
        opts.f.load_state_dict(ckpt["opt_f_state"])

        g.load_state_dict(ckpt["g_state"])
        # emb_encoder.load_state_dict(ckpt["emb_encoder_state"])
        opts.g.load_state_dict(ckpt["opt_g_state"])
        # opt_emb.load_state_dict(ckpt["opt_emb_state"])

    # return (f, g, emb_encoder), opts, opt_emb
    return (f, g, emb_encoder), opts


def compute_loss_g(f, g, source, condition, transport=None, beta = 1.0, transform_matrix=None, sigma=0.0):
    if transport is None:
        transport_cond = g.transport(source, condition, is_conditional_generation=True)
        if abs(beta - 1.0) < 0.00001:
            transport = transport_cond
        else:
            transport_uncond = g.transport(source, condition, is_conditional_generation=False)
            transport = beta * transport_cond - (beta -1 ) * transport_uncond

    if abs(sigma-0.0)<1e-8:
        return (f(transport @ transform_matrix, condition) - torch.einsum("ij,ij->i", source, transport).unsqueeze(-1)) \
            if transform_matrix is not None \
            else f(transport, condition) - torch.einsum("ij,ij->i", source, transport).unsqueeze(-1)
    else:
        norm_penalize = sigma * torch.clamp(
            0.5 * torch.norm(source, p=2, dim=1).unsqueeze(1) - g(source, condition), min=0.0)
        return (f(transport @ transform_matrix, condition) -
                torch.einsum("ij,ij->i", source, transport).unsqueeze(-1)) + norm_penalize \
            if transform_matrix is not None \
            else f(transport, condition) - torch.einsum("ij,ij->i", source, transport).unsqueeze(-1) + norm_penalize

    # return f(transport, condition) - torch.multiply(
    #     source, transport).sum(-1, keepdim=True)




def compute_g_constraint(g, form=None, beta=0):
    if form is None or form == "None":
        return 0

    if form == "clamp":
        g.clamp_w()
        return 0

    elif form == "fnorm":
        if beta == 0:
            return 0
        return beta * sum(map(lambda w: w.weight.norm(p="fro"), g.wz))

    raise ValueError


def compute_loss_f(f, g, source, target, condition, transport=None, beta = 1.0, transform_matrix=None, sigma=0.0):
    if transport is None:
        transport_cond = g.transport(source, condition, is_conditional_generation=True)
        if abs(beta - 1.0) < 0.00001:
            transport = transport_cond
        else:
            transport_uncond = g.transport(source, condition, is_conditional_generation=False)
            transport = beta * transport_cond - (beta - 1) * transport_uncond

    # 合并为单次矩阵乘法：
    combined_input = torch.cat([transport @ transform_matrix, target], dim=0) \
        if transform_matrix is not None \
        else torch.cat([transport, target], dim=0)
    combined_output = f(combined_input, condition.repeat(2,1)) if condition.dim()>1 else f(combined_input, condition.repeat(2))
    term1 = -combined_output[:len(transport)]
    term2 = combined_output[len(transport):]

    if abs(sigma-0.0)<1e-8:
        return term1 + term2
    else:
        norm_penalize = sigma * torch.clamp(
            0.5 * torch.norm(target, p=2, dim=1).unsqueeze(1) - f(target, condition), min=0.0)
    # return -f(transport @ transform_matrix, condition) + f(target, condition) + norm_penalize
        return term1 + term2 + norm_penalize


def compute_w2_distance(f, g, source, target, condition, transport=None):
    if transport is None:
        transport = g.transport(source, condition).squeeze()

    with torch.no_grad():
        Cpq = (source * source).sum(1, keepdim=True) + (target * target).sum(
            1, keepdim=True
        )
        Cpq = 0.5 * Cpq

        cost = (
            f(transport, condition)
            - torch.multiply(source, transport).sum(-1, keepdim=True)
            - f(target, condition)
            + Cpq
        )
        cost = cost.mean()
    return cost


def numerical_gradient(param, fxn, *args, eps=1e-4):
    with torch.no_grad():
        param += eps
    plus = float(fxn(*args))

    with torch.no_grad():
        param -= 2 * eps
    minus = float(fxn(*args))

    with torch.no_grad():
        param += eps

    return (plus - minus) / (2 * eps)
