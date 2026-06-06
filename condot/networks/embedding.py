#!/usr/bin/python3

# imports
import pandas as pd
import torch
from sklearn import preprocessing
# import torch.nn.init as init
import torch.nn as nn
# from chemCPA.model import MLP

class EmbeddingSMACOF():
    """Queries precomputed SMACOF embedding of conditions."""

    def __init__(self, path, **kwargs):
        self.embedding = self.setup(path)

    def setup(self, emb_config):
        # load pickled embedding
        return pd.read_hdf(emb_config.path, f'smacof_{emb_config.dim}d')

    def forward(self, y):
        return torch.Tensor(self.embedding[y].values)


class EmbeddingFingerprint():
    """Queries precomputed molecular fingerprint of conditions."""

    def __init__(self, path, **kwargs):
        self.embedding = self.setup(path)

    def setup(self, emb_config):
        # load precomputed fingerprints
        embedding = pd.read_csv(
          emb_config.path,
          converters={f"{emb_config.method} Fingerprint": lambda x: x.strip(
              "[]").split(", ")})
        return dict(zip(embedding.Name,
                        embedding[f"{emb_config.method} Fingerprint"]))

    def forward(self, y):
        return torch.Tensor(list(map(float, self.embedding[y])))


class EmbeddingOneHot(nn.Module):
    """Returns one-hot of conditions."""

    def __init__(self, labels, **kwargs):
        super(EmbeddingOneHot, self).__init__()
        self.setup(labels)

    def setup(self, labels):
        multi_labels = [i.split("+") for i in labels]
        self.lb = preprocessing.MultiLabelBinarizer()
        self.lb.fit(multi_labels)

    def forward(self, y):
        return torch.Tensor(self.lb.transform([y.split("+")])[0])

class EmbeddingIdx2OneHot(nn.Module):
    """Returns one-hot of conditions."""

    def __init__(self, labels, **kwargs):
        super(EmbeddingIdx2OneHot, self).__init__()
        self.setup(labels)

    def setup(self, labels):
        self.lb = preprocessing.MultiLabelBinarizer()
        self.lb.fit(labels)

    def forward(self, y):
        return torch.Tensor(self.lb.transform(y))


class EmbeddingValue(nn.Module):
    """Returns value to use as condition."""
    def __init__(self, factor):
        super(EmbeddingValue, self).__init__()
        self.factor = factor

    def forward(self, y):
        dose = y / self.factor
        return torch.Tensor(dose)

class EmbeddingSMILES(nn.Module):
    def __init__(self, emb_pretrained):
        super(EmbeddingSMILES, self).__init__()
        self.register_buffer("emb_pretrained", emb_pretrained)

    def to(self, device):
        self.emb_pretrained = self.emb_pretrained.to(device)
        return self

    def forward(self, indexes):
        return self.emb_pretrained[indexes]
