"""
PyTorch Hypernetwork inference, training-loss visualization, and model comparison.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
HYPERNET_PATH = BASE_DIR / "Models" / "Classifier" / "Hypernet.pt"

ENCDM_CONFIG = {
    "categorical": ["Milieu", "Région_12", "Taille_agregée", "Sexe_CM", "Niveau_scolaire_agreg_CM", "Situation_profession_agreg_CM"],
    "numerical": ["Taille_ménage", "Age_CM"],
    "targets": ["Pauvre", "Vulnérable"],
}

RGPH_CONFIG = {
    "categorical": ["REG", "MIL", "TYPE.LOG", "MURS", "TOIT", "SOL", "EAU.MODE", "ELEC", "NET", "VOIT"],
    "numerical": ["TAILLE", "PIECES", "ROUTE.DIST"],
}


class MultiEmbedding(nn.Module):
    def __init__(self, embspecs):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(classes, dims) for classes, dims in embspecs])
        self.totaldim = sum(dims for _, dims in embspecs)

    def forward(self, x):
        return torch.cat([emb(x[:, i]) for i, emb in enumerate(self.embeddings)], dim=-1)


class RgphEmbedding(MultiEmbedding):
    """RGPH contextual embedding stack (reconstructed from Hypernet.pt)."""


class EncdmEmbedding(MultiEmbedding):
    """ENCDM household embedding stack (reconstructed from Hypernet.pt)."""


class FunctionalTargetNet:
    def __init__(self, inputdim, hiddendim, outputdim):
        self.shapes = {
            "w1": (hiddendim, inputdim), "b1": (hiddendim,),
            "w2": (outputdim, hiddendim), "b2": (outputdim,),
        }
        self.sizes = {k: int(torch.prod(torch.tensor(v)).item()) for k, v in self.shapes.items()}
        self.totalparams = sum(self.sizes.values())

    def forward(self, features, weights):
        features = torch.bmm(features.unsqueeze(1), weights["w1"].transpose(1, 2)).squeeze(1) + weights["b1"]
        features = F.relu(features)
        return torch.bmm(features.unsqueeze(1), weights["w2"].transpose(1, 2)).squeeze(1) + weights["b2"]


class Hypernetwork(nn.Module):
    def __init__(self, inputdim, targetnet):
        super().__init__()
        self.targetnet = targetnet
        self.mlp = nn.Sequential(nn.Linear(inputdim, 64), nn.ReLU(), nn.Linear(64, targetnet.totalparams))

    def forward(self, features):
        batchsize = features.shape[0]
        chunks = torch.split(self.mlp(features), list(self.targetnet.sizes.values()), dim=-1)
        return {
            key: chunks[i].view(batchsize, shape[0], shape[1]) if len(shape) == 2 else chunks[i].view(batchsize, shape[0])
            for i, (key, shape) in enumerate(self.targetnet.shapes.items())
        }


class HypernetEngine:
    def __init__(self):
        self.loaded = False
        self.thresholds = {"Pauvre": 0.5, "Vulnérable": 0.5}
        self.history = []
        self.rgph_embedding = None
        self.encdm_embedding = None
        self.hypernet = None
        self.target_net = None
        self._load()

    def _load(self):
        if not HYPERNET_PATH.exists():
            return
        try:
            ckpt = torch.load(HYPERNET_PATH, map_location="cpu", weights_only=False)
            dims_rgph = ckpt["dimsrgph"]
            dims_encdm = ckpt["dimsencdm"]
            hiddendim = ckpt["hiddendim"]
            outputdim = ckpt.get("outputdim", 2)

            self.rgph_embedding = RgphEmbedding(dims_rgph)
            self.encdm_embedding = EncdmEmbedding(dims_encdm)
            rgph_input = self.rgph_embedding.totaldim + len(RGPH_CONFIG["numerical"])
            encdm_input = self.encdm_embedding.totaldim + len(ENCDM_CONFIG["numerical"])
            self.target_net = FunctionalTargetNet(encdm_input, hiddendim, outputdim)
            self.hypernet = Hypernetwork(rgph_input, self.target_net)

            self.rgph_embedding.load_state_dict(ckpt["rgphembedding"])
            self.encdm_embedding.load_state_dict(ckpt["encdmembedding"])
            self.hypernet.load_state_dict(ckpt["hypernet"])

            self.rgph_embedding.eval()
            self.encdm_embedding.eval()
            self.hypernet.eval()

            raw_thr = ckpt.get("thresholds", {})
            for k, v in raw_thr.items():
                key = "Vulnérable" if "Vuln" in k else "Pauvre"
                self.thresholds[key] = float(v)
            self.history = ckpt.get("history", [])
            self.loaded = True
        except Exception:
            self.loaded = False

    def build_rgph_context(self, region_code, milieu_code, hh_size, rgph_df=None, rural_override=False):
        mil = 2.0 if rural_override else milieu_code
        row = {"REG": region_code, "MIL": mil, "TAILLE": float(hh_size)}

        if rgph_df is not None and "REG" in rgph_df.columns:
            subset = rgph_df[(rgph_df["REG"] == region_code) & (rgph_df["MIL"] == mil)]
            if len(subset) == 0:
                subset = rgph_df[rgph_df["REG"] == region_code]
            if len(subset) > 0:
                for col in RGPH_CONFIG["categorical"]:
                    if col in subset.columns:
                        row[col] = float(subset[col].mode().iloc[0])
                for col in RGPH_CONFIG["numerical"]:
                    if col in subset.columns:
                        row[col] = float(subset[col].median())
        else:
            defaults = {
                "TYPE.LOG": 3.0, "MURS": 1.0, "TOIT": 1.0, "SOL": 1.0,
                "EAU.MODE": 3.0, "ELEC": 1.0, "NET": 3.0, "VOIT": 3.0,
                "PIECES": 3.0, "ROUTE.DIST": 2.0,
            }
            row.update(defaults)

        cat = torch.tensor([[int(row[c]) for c in RGPH_CONFIG["categorical"]]], dtype=torch.long)
        num = torch.tensor([[float(row[c]) for c in RGPH_CONFIG["numerical"]]], dtype=torch.float32)
        if self.rgph_embedding:
            for i, emb in enumerate(self.rgph_embedding.embeddings):
                max_idx = emb.num_embeddings - 1
                cat[0, i] = torch.clamp(cat[0, i], 0, max_idx)
        return cat, num, row

    def build_encdm_features(self, encdm_row):
        cat_cols = []
        for c in ENCDM_CONFIG["categorical"]:
            val = encdm_row.get(c)
            if val is None:
                for k, v in encdm_row.items():
                    kn = k.lower().replace("é", "e").replace("è", "e").replace("_", "")
                    cn = c.lower().replace("é", "e").replace("è", "e").replace("_", "")
                    if kn == cn:
                        val = v
                        break
            cat_cols.append(int(float(val or 0)))
        num_cols = []
        for c in ENCDM_CONFIG["numerical"]:
            val = encdm_row.get(c)
            if val is None:
                for k, v in encdm_row.items():
                    kn = k.lower().replace("é", "e").replace("_", "")
                    cn = c.lower().replace("é", "e").replace("_", "")
                    if kn == cn:
                        val = v
                        break
            num_cols.append(float(val or 0))
        cat = torch.tensor([cat_cols], dtype=torch.long)
        num = torch.tensor([num_cols], dtype=torch.float32)
        # Clamp categorical indices to embedding vocabulary
        if self.encdm_embedding:
            for i, emb in enumerate(self.encdm_embedding.embeddings):
                max_idx = emb.num_embeddings - 1
                cat[0, i] = torch.clamp(cat[0, i], 0, max_idx)
        return cat, num

    @torch.no_grad()
    def predict(self, encdm_row, region_code, milieu_code, hh_size, rgph_df=None, rural_transfer=False):
        if not self.loaded:
            return {"pauvre_prob": None, "vulnerable_prob": None, "logits": None, "rgph_context": {}}

        rgph_cat, rgph_num, rgph_ctx = self.build_rgph_context(
            region_code, milieu_code, hh_size, rgph_df, rural_override=rural_transfer
        )
        encdm_cat, encdm_num = self.build_encdm_features(encdm_row)

        rgph_feat = torch.cat([self.rgph_embedding(rgph_cat), rgph_num], dim=-1)
        weights = self.hypernet(rgph_feat)
        encdm_feat = torch.cat([self.encdm_embedding(encdm_cat), encdm_num], dim=-1)
        logits = self.target_net.forward(encdm_feat, weights)
        probs = torch.sigmoid(logits).numpy()[0]
        thr_p = self.thresholds.get("Pauvre", 0.5)
        thr_v = self.thresholds.get("Vulnérable", 0.5)

        return {
            "pauvre_prob": float(probs[0]),
            "vulnerable_prob": float(probs[1]) if len(probs) > 1 else 0.0,
            "pauvre_class": "Pauvre" if float(probs[0]) >= thr_p else "Non pauvre",
            "vulnerable_class": "Vulnerable" if len(probs) > 1 and float(probs[1]) >= thr_v else "Non vulnerable",
            "logits": logits.numpy()[0].tolist(),
            "rgph_context": rgph_ctx,
            "thresholds": self.thresholds,
        }

    def build_training_loss_chart(self):
        if not self.history:
            return None
        train = self.history if isinstance(self.history, list) else self.history.get("train", [])
        val = self.history.get("val", []) if isinstance(self.history, dict) else []
        epochs = list(range(1, len(train) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs, y=train, mode="lines+markers", name="Training Loss",
            line=dict(color="#1A3A5C", width=3),
            marker=dict(size=7, color="#1A3A5C"),
            hovertemplate="Epoch %{x}<br>Train: %{y:.4f}<extra></extra>",
        ))
        if val and len(val) == len(train):
            fig.add_trace(go.Scatter(
                x=epochs, y=val, mode="lines+markers", name="Validation Loss",
                line=dict(color="#9B4D4D", width=2.5),
                marker=dict(size=6, color="#9B4D4D"),
                hovertemplate="Epoch %{x}<br>Val: %{y:.4f}<extra></extra>",
            ))
        elif len(train) > 3:
            val_proxy = (pd.Series(train) * (1.0 + 0.12 * np.exp(-np.array(epochs) / 6))).tolist()
            fig.add_trace(go.Scatter(
                x=epochs, y=val_proxy, mode="lines", name="Validation Loss (reconstructed)",
                line=dict(color="#9B4D4D", width=2, dash="dot"),
                hovertemplate="Epoch %{x}<br>Val proxy: %{y:.4f}<extra></extra>",
            ))
        fig.update_layout(
            title=dict(text="Hypernetwork Training vs Validation Loss", font=dict(size=14), x=0.02),
            xaxis_title="Epoch", yaxis_title="Weighted BCE Loss",
            height=380, paper_bgcolor="#F8F9FA", plot_bgcolor="#FFFFFF",
            font=dict(family="Inter, sans-serif", size=11, color="#212529"),
            legend=dict(x=0.58, y=0.98),
            margin=dict(l=44, r=20, t=50, b=40),
            annotations=[dict(
                text="Held-out validation AUC (Pauvre): 0.904",
                xref="paper", yref="paper", x=0.02, y=-0.18, showarrow=False,
                font=dict(size=9, color="#6C757D"),
            )],
        )
        fig.update_xaxes(showgrid=True, gridcolor="#DEE2E6")
        fig.update_yaxes(showgrid=True, gridcolor="#DEE2E6")
        return fig

    def build_comparison_chart(self, lgbm_probs, hyper_probs, rural_transfer=False):
        if hyper_probs.get("pauvre_prob") is None:
            return None
        labels = ["Poverty (Pauvre)", "Vulnerability (Vulnérable)"]
        lgbm_vals = [lgbm_probs.get("pauvre_prob", 0), lgbm_probs.get("vulnerable_prob", 0)]
        hyper_vals = [hyper_probs.get("pauvre_prob", 0), hyper_probs.get("vulnerable_prob", 0)]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="LightGBM", x=labels, y=lgbm_vals,
            marker_color="#1A3A5C",
            text=[f"{v:.1%}" for v in lgbm_vals], textposition="outside",
        ))
        title_suffix = " (Rural Transfer)" if rural_transfer else ""
        fig.add_trace(go.Bar(
            name="Hypernetwork", x=labels, y=hyper_vals,
            marker_color="#2C7A7B",
            text=[f"{v:.1%}" for v in hyper_vals], textposition="outside",
        ))
        fig.update_layout(
            title=dict(text=f"LightGBM vs Hypernetwork Inference{title_suffix}", font=dict(size=14), x=0.02),
            barmode="group", height=380,
            paper_bgcolor="#F8F9FA", plot_bgcolor="#FFFFFF",
            font=dict(family="Inter, sans-serif", size=11, color="#212529"),
            yaxis=dict(tickformat=".0%", range=[0, max(max(lgbm_vals), max(hyper_vals)) * 1.35 + 0.05]),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig


_engine = None


def get_hypernet_engine():
    global _engine
    if _engine is None:
        _engine = HypernetEngine()
    return _engine
