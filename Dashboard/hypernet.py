"""
PyTorch hypernetwork inference — architecture mirrors Modeling.ipynb.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import ENCDM_CONFIG, RGPH_CONFIG, ROOT


class MultiEmbedding(nn.Module):
    def __init__(self, embspecs):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(classes, dims) for classes, dims in embspecs]
        )
        self.totaldim = sum(dims for _, dims in embspecs)

    def forward(self, x):
        return torch.cat(
            [emb(x[:, i]) for i, emb in enumerate(self.embeddings)], dim=-1
        )


class FunctionalTargetNet:
    def __init__(self, inputdim, hiddendim, outputdim):
        self.shapes = {
            "w1": (hiddendim, inputdim),
            "b1": (hiddendim,),
            "w2": (outputdim, hiddendim),
            "b2": (outputdim,),
        }
        self.sizes = {
            key: torch.prod(torch.tensor(val)).item()
            for key, val in self.shapes.items()
        }
        self.totalparams = sum(self.sizes.values())

    def forward(self, features, weights):
        x = torch.bmm(features.unsqueeze(1), weights["w1"].transpose(1, 2)).squeeze(1)
        x = x + weights["b1"]
        x = F.relu(x)
        return torch.bmm(x.unsqueeze(1), weights["w2"].transpose(1, 2)).squeeze(1) + weights["b2"]


class Hypernetwork(nn.Module):
    def __init__(self, inputdim, targetnet):
        super().__init__()
        self.targetnet = targetnet
        self.mlp = nn.Sequential(
            nn.Linear(inputdim, 64),
            nn.ReLU(),
            nn.Linear(64, targetnet.totalparams),
        )

    def forward(self, features):
        batchsize = features.shape[0]
        chunks = torch.split(self.mlp(features), list(self.targetnet.sizes.values()), dim=-1)
        return {
            key: chunks[i].view(batchsize, shape[0], shape[1])
            if len(shape) == 2
            else chunks[i].view(batchsize, shape[0])
            for i, (key, shape) in enumerate(self.targetnet.shapes.items())
        }


class HypernetEngine:
    def __init__(self):
        self._ready = False
        self.strata_context = {}
        self.thresholds = {}
        self.targets = ENCDM_CONFIG["target"]

    def load(self, rgph_df):
        ckpt_path = ROOT / "Models/Classifier/Hypernet.pt"
        if not ckpt_path.exists():
            return False

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        dims_rgph = ckpt["dimsrgph"]
        dims_encdm = ckpt["dimsencdm"]
        hidden = ckpt["hiddendim"]
        output = ckpt["outputdim"]

        rgph_input = sum(d for _, d in dims_rgph) + len(RGPH_CONFIG["numerical"])
        encdm_input = sum(d for _, d in dims_encdm) + len(ENCDM_CONFIG["numerical"])

        self.rgph_embedding = MultiEmbedding(dims_rgph)
        self.encdm_embedding = MultiEmbedding(dims_encdm)
        self.target_net = FunctionalTargetNet(encdm_input, hidden, output)
        self.hypernet = Hypernetwork(rgph_input, self.target_net)

        self.rgph_embedding.load_state_dict(ckpt["rgphembedding"])
        self.encdm_embedding.load_state_dict(ckpt["encdmembedding"])
        self.hypernet.load_state_dict(ckpt["hypernet"])

        self.rgph_embedding.eval()
        self.encdm_embedding.eval()
        self.hypernet.eval()

        self.thresholds = {
            k: float(v) for k, v in ckpt.get("thresholds", {}).items()
        }
        self._build_strata_context(rgph_df)
        self._ready = True
        return True

    def _build_strata_context(self, rgph_df):
        df = rgph_df.copy()
        for col in RGPH_CONFIG["categorical"]:
            df[col] = df[col].fillna(0).astype(int)
        for col in RGPH_CONFIG["numerical"]:
            df[col] = df[col].fillna(df[col].median()).astype("float32")

        df["strata_key"] = (
            df["REG"].astype(int).astype(str) + "_" + df["MIL"].astype(int).astype(str)
        )

        self.strata_context = {}
        for key, subset in df.groupby("strata_key"):
            if len(subset) == 0:
                continue
            cat = torch.tensor(
                subset[RGPH_CONFIG["categorical"]].astype(int).mode().iloc[0].values,
                dtype=torch.long,
            ).unsqueeze(0)
            num = torch.tensor(
                subset[RGPH_CONFIG["numerical"]].astype("float32").mean().values,
                dtype=torch.float32,
            ).unsqueeze(0)
            self.strata_context[key] = {"cat": cat, "num": num}

        if not self.strata_context:
            self._ready = False

    def _resolve_context(self, region_code, milieu_code, rural_transfer=False):
        from data_loader import build_region_bridge

        milieu = 1 if rural_transfer else int(milieu_code)
        bridge = build_region_bridge()
        reg_rgph = bridge.get(int(region_code), int(region_code))
        key = f"{reg_rgph}_{milieu}"
        ctx = self.strata_context.get(key)
        if ctx is None:
            ctx = next(iter(self.strata_context.values()))
        return ctx

    @torch.no_grad()
    def predict(self, feature_row, region_code, milieu_code, rural_transfer=False):
        if not self._ready:
            return {"Pauvre": None, "Vulnérable": None}

        ctx = self._resolve_context(region_code, milieu_code, rural_transfer)

        rgph_features = torch.cat(
            [self.rgph_embedding(ctx["cat"]), ctx["num"]], dim=-1
        )
        weights = self.hypernet(rgph_features)

        cat_cols = ENCDM_CONFIG["categorical"]
        num_cols = ENCDM_CONFIG["numerical"]
        encdm_cat = torch.tensor(
            [[int(feature_row[c]) for c in cat_cols]], dtype=torch.long
        )
        encdm_num = torch.tensor(
            [[float(feature_row[c]) for c in num_cols]], dtype=torch.float32
        )
        encdm_features = torch.cat(
            [self.encdm_embedding(encdm_cat), encdm_num], dim=-1
        )

        logits = self.target_net.forward(encdm_features, weights)
        probs = torch.sigmoid(logits).squeeze(0).numpy()

        return {
            self.targets[i]: float(probs[i]) for i in range(len(self.targets))
        }


_engine = None


def get_hypernet_engine(rgph_df):
    global _engine
    if _engine is None:
        _engine = HypernetEngine()
        _engine.load(rgph_df)
    return _engine
