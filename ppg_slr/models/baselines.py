"""Baselines. Each isolates one of PPG's two claims. All work with any backbone
returned by make_backbone (CNN, Swin or ViT).

  PlainBackbone      : no prototypes, no uncertainty  -> is any machinery needed?
  DeterministicProto : prototypes + hard max, no uncertainty -> does the
                       PROBABILISTIC part beat plain prototypes?
  MCDropoutBackbone  : uncertainty but no prototypes -> does the PROTOTYPE part
                       add anything over plain uncertainty?

All return the same dict shape as PPGNet for a uniform training/eval loop.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainBackbone(nn.Module):
    def __init__(self, cfg, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.out_channels, cfg.num_classes)

    def forward(self, x):
        feat = self.backbone(x)                       # [B, C, H, W]
        pooled = feat.mean(dim=(2, 3))                # global average pool
        return {"logits": self.head(pooled)}


class DeterministicProto(nn.Module):
    """ProtoPNet-style: cosine prototypes + hard max pooling, learnable head.
    Note: 'alpha' is the RAW cosine-similarity map (not a normalised attention),
    so its entropy is not comparable with PPG's attention maps."""
    def __init__(self, cfg, backbone):
        super().__init__()
        self.backbone = backbone
        P = cfg.total_protos()
        self.prototypes = nn.Parameter(torch.randn(P, backbone.out_channels) * 0.02)
        self.head = nn.Linear(P, cfg.num_classes, bias=False)

    def forward(self, x):
        feat = F.normalize(self.backbone(x), dim=1)
        p = F.normalize(self.prototypes, dim=1)
        S = torch.einsum("pc,bchw->bphw", p, feat)    # [B, P, H, W]
        act = S.flatten(2).max(dim=2).values          # hard max pooling
        return {"logits": self.head(F.relu(act)), "alpha": S}


class MCDropoutBackbone(nn.Module):
    """Backbone + always-on dropout + linear head; M passes at eval, logits averaged."""
    def __init__(self, cfg, backbone):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.head = nn.Linear(backbone.out_channels, cfg.num_classes)

    def _once(self, feat):
        # MC dropout stays ON at eval (nn.Dropout would switch off under model.eval(),
        # making all M passes identical)
        pooled = F.dropout(feat.mean(dim=(2, 3)), p=self.cfg.mc_dropout_p, training=True)
        return self.head(pooled)

    def forward(self, x):
        feat = self.backbone(x)
        M = self.cfg.mc_samples if not self.training else 1
        logits = torch.stack([self._once(feat) for _ in range(max(M, 1))], 0).mean(0)
        return {"logits": logits}


def build_model(cfg, backbone):
    """Factory used by the notebook / train.py. Anything not matching a baseline -> PPG."""
    name = cfg.exp_name.lower()
    if name.startswith("plain"):
        return PlainBackbone(cfg, backbone)
    if name.startswith("proto") or name.startswith("hiervit"):
        return DeterministicProto(cfg, backbone)
    if name.startswith("mcdropout"):
        return MCDropoutBackbone(cfg, backbone)
    from .ppg import PPGNet
    return PPGNet(cfg, backbone)